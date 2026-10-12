"""The deep ranker B1-DL: a small transformer over the patient's questions (decision D-12, EXP-021).

**Why a transformer.** DDXPlus findings interact: exertional pain means one thing beside rest pain
and another without it. Attention lets every finding see every other. The owner chose this model as
the project's one deep ranker (D-12, 2026-10-12); it is compared with, and may replace, the
logistic regression trained like it (B1-LR′+aug), only if it is at least as safe (D-13).

**How it reads a patient.** The encoder's row (:mod:`src.ml.features`) is regrouped into one
token per question, as the FT-Transformer's feature tokenizer does per feature (Gorishniy et al.
2021, "Revisiting deep learning models for tabular data"):

* every encoder column belongs to one token, named by the part of its name before ``=``:
  ``E_55``, ``E_55=V_16`` and ``E_55=unasked`` all belong to ``E_55``; ``age`` and ``sex=F`` are
  tokens of their own;
* a token is its bias plus the sum, over its columns, of the column's (max-abs scaled) value times
  that column's embedding. So a categorical answer adds its answer's embedding, a multi-choice
  question the sum of its answers', an ordinal its value times one embedding plus the "answered"
  flag's, and with the "asked" channel an unasked question adds its ``unasked`` embedding;
* a [CLS] token is appended, and its final state is classified.

So the 84 questions, age and sex make 86 tokens, plus [CLS]: **the same 87 for every patient**,
however long the history, which makes the cost of a forward pass constant.

**The blocks** follow the FT-Transformer's defaults (rtdl), at a smaller size fixed before any
training (T-6 allows no tuning): pre-normalisation, the first block without its attention
LayerNorm; multi-head attention; a ReGLU feed-forward of width 4/3 · d; dropout 0.2 on the
attention probabilities and 0.1 in the feed-forward, none on the residuals; the last block computes
the [CLS] row only; the head is LayerNorm → ReLU → linear. AdamW, lr 1e-4, weight decay 1e-5 (none
on the tokenizer, the norms or the biases). d = 96, 2 blocks, 4 heads (24 per head, as rtdl's
192 / 8).

**Two implementations, one model.** Training is PyTorch, on a cloud GPU (D-7), imported only by
:meth:`DeepRanker.fit`; nothing else here needs torch. Inference is NumPy only, so the laptop runs
the model with nothing to install. The weights are saved as plain JSON: each array base64-encoded
float32 with its shape and SHA-256, never a pickle, and the file names the encoder's fingerprint,
which a loader checks as for the B1 models. :func:`parity` measures how far the two
implementations disagree.
"""

from __future__ import annotations

import base64
import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.ml.baselines import LABELS, _check_classes, _check_fingerprint, _softmax

__all__ = [
    "DeepRanker",
    "TrainingConfig",
    "TransformerConfig",
    "parity",
    "token_map",
]

MODEL_NAME = "B1-DL transformer"
LAYER_NORM_EPS = 1e-5
"""PyTorch's default, so the two implementations normalise alike."""


@dataclass(frozen=True)
class TransformerConfig:
    """The architecture, fixed before training (D-12, T-6)."""

    d_token: int = 96
    n_blocks: int = 2
    n_heads: int = 4
    ffn_factor: float = 4 / 3
    attention_dropout: float = 0.2
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0

    @property
    def ffn_hidden(self) -> int:
        return int(self.d_token * self.ffn_factor)

    def __post_init__(self) -> None:
        if self.d_token % self.n_heads:
            raise ValueError(f"d_token {self.d_token} is not divisible by {self.n_heads} heads")
        if self.n_blocks < 1:
            raise ValueError("a transformer needs at least one block")


@dataclass(frozen=True)
class TrainingConfig:
    """How :meth:`DeepRanker.fit` trains: fixed values, never tuned (T-6)."""

    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 1024
    max_epochs: int = 50
    patience: int = 5
    eval_batch_size: int = 4096


def token_map(feature_names: Sequence[str]) -> tuple[tuple[str, ...], np.ndarray]:
    """The tokens, in order of first appearance, and each column's token number.

    A column belongs to the token named by the part of its name before ``=`` (``sex=F`` → ``sex``).
    """
    tokens: dict[str, int] = {}
    column_token = []
    for name in feature_names:
        token = name.partition("=")[0]
        column_token.append(tokens.setdefault(token, len(tokens)))
    return tuple(tokens), np.array(column_token, dtype=np.int64)


def _parameter_shapes(
    n_columns: int, n_tokens: int, n_classes: int, config: TransformerConfig
) -> dict[str, tuple[int, ...]]:
    """Every weight's name and shape, in one order shared by both implementations."""
    d, h = config.d_token, config.ffn_hidden
    shapes: dict[str, tuple[int, ...]] = {
        "tokenizer.weight": (n_columns, d),
        "tokenizer.bias": (n_tokens, d),
        "cls": (d,),
    }
    for i in range(config.n_blocks):
        block = f"blocks.{i}"
        if i > 0:
            shapes[f"{block}.attention_norm.weight"] = (d,)
            shapes[f"{block}.attention_norm.bias"] = (d,)
        for name in ("W_q", "W_k", "W_v", "W_out"):
            shapes[f"{block}.attention.{name}.weight"] = (d, d)
            shapes[f"{block}.attention.{name}.bias"] = (d,)
        shapes[f"{block}.ffn_norm.weight"] = (d,)
        shapes[f"{block}.ffn_norm.bias"] = (d,)
        shapes[f"{block}.ffn.linear_first.weight"] = (2 * h, d)
        shapes[f"{block}.ffn.linear_first.bias"] = (2 * h,)
        shapes[f"{block}.ffn.linear_second.weight"] = (d, h)
        shapes[f"{block}.ffn.linear_second.bias"] = (d,)
    shapes["head.norm.weight"] = (d,)
    shapes["head.norm.bias"] = (d,)
    shapes["head.linear.weight"] = (n_classes, d)
    shapes["head.linear.bias"] = (n_classes,)
    return shapes


def _encode_array(array: np.ndarray) -> dict[str, Any]:
    raw = np.ascontiguousarray(array, dtype="<f4").tobytes()
    return {
        "shape": list(array.shape),
        "dtype": "float32",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def _decode_array(name: str, entry: Mapping[str, Any]) -> np.ndarray:
    raw = base64.b64decode(entry["data"])
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"weight {name} does not match its SHA-256: the file is damaged")
    return np.frombuffer(raw, dtype="<f4").reshape(entry["shape"]).astype(np.float32)


# --------------------------------------------------------------------------- #
# The NumPy forward pass
# --------------------------------------------------------------------------- #


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + LAYER_NORM_EPS) * weight + bias


def _linear(x: np.ndarray, w: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    return x @ w[f"{name}.weight"].T + w[f"{name}.bias"]


def _attention(
    x_query: np.ndarray, x_kv: np.ndarray, w: Mapping[str, np.ndarray], block: str, heads: int
) -> np.ndarray:
    q = _linear(x_query, w, f"{block}.attention.W_q")
    k = _linear(x_kv, w, f"{block}.attention.W_k")
    v = _linear(x_kv, w, f"{block}.attention.W_v")
    batch, n_query, d = q.shape
    n_kv, d_head = k.shape[1], d // heads
    q = q.reshape(batch, n_query, heads, d_head).transpose(0, 2, 1, 3)
    k = k.reshape(batch, n_kv, heads, d_head).transpose(0, 2, 1, 3)
    v = v.reshape(batch, n_kv, heads, d_head).transpose(0, 2, 1, 3)
    logits = q @ k.transpose(0, 1, 3, 2) / math.sqrt(d_head)
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    out = (probabilities @ v).transpose(0, 2, 1, 3).reshape(batch, n_query, d)
    return _linear(out, w, f"{block}.attention.W_out")


def _ffn(x: np.ndarray, w: Mapping[str, np.ndarray], block: str) -> np.ndarray:
    hidden = _linear(x, w, f"{block}.ffn.linear_first")
    a, b = np.split(hidden, 2, axis=-1)
    return _linear(a * np.maximum(b, 0.0), w, f"{block}.ffn.linear_second")


@dataclass
class DeepRanker:
    """B1-DL: the trained transformer, scored with NumPy alone.

    Attributes:
        labels: The class order, :data:`~src.ml.baselines.LABELS`.
        feature_fingerprint: The encoder the model was trained on; a loader checks it.
        feature_names: The encoder's columns, from which the token map is rebuilt.
        scale: Each column's largest absolute value on the training rows (1 where it is 0): the
            row is divided by it first, as :class:`~src.ml.baselines.LogisticBaseline` does.
        config: The architecture.
        weights: Every array, named as :func:`_parameter_shapes` names them.
        params: How it was trained (the seed, the training configuration, the best epoch).
        history: One entry per epoch: the training and held-out losses and the time taken.
    """

    labels: tuple[str, ...]
    feature_fingerprint: str
    feature_names: tuple[str, ...] = field(repr=False)
    scale: np.ndarray = field(repr=False)
    config: TransformerConfig
    weights: dict[str, np.ndarray] = field(repr=False)
    params: Mapping[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.tokens, self.column_token = token_map(self.feature_names)
        expected = _parameter_shapes(
            len(self.feature_names), len(self.tokens), len(self.labels), self.config
        )
        if set(expected) != set(self.weights):
            missing = sorted(set(expected) - set(self.weights))
            extra = sorted(set(self.weights) - set(expected))
            raise ValueError(
                f"the weights do not fit the architecture: missing {missing}, extra {extra}"
            )
        for name, shape in expected.items():
            if tuple(self.weights[name].shape) != shape:
                raise ValueError(f"{name} has shape {self.weights[name].shape}, expected {shape}")
        if self.scale.shape != (len(self.feature_names),):
            raise ValueError("scale needs one value per column")
        self._columns = [
            np.flatnonzero(self.column_token == t).astype(np.int64) for t in range(len(self.tokens))
        ]

    # -- construction ------------------------------------------------------ #

    @classmethod
    def initialised(
        cls,
        feature_names: Sequence[str],
        *,
        feature_fingerprint: str,
        labels: Sequence[str] = LABELS,
        config: TransformerConfig | None = None,
        seed: int = 0,
        scale: np.ndarray | None = None,
    ) -> DeepRanker:
        """An untrained model with random weights, drawn with NumPy: for tests and for timing.

        The draws follow rtdl's initialisation in kind (uniform tokens, Xavier attention, PyTorch's
        default linear layers), not in its random stream.
        """
        config = config or TransformerConfig()
        names = tuple(feature_names)
        tokens, _ = token_map(names)
        rng = np.random.default_rng(seed)
        weights: dict[str, np.ndarray] = {}
        token_bound = 1 / math.sqrt(config.d_token)
        for name, shape in _parameter_shapes(len(names), len(tokens), len(labels), config).items():
            if name.startswith(("tokenizer", "cls")):
                array = rng.uniform(-token_bound, token_bound, shape)
            elif name.endswith("norm.weight"):
                array = np.ones(shape)
            elif name.endswith("norm.bias"):
                array = np.zeros(shape)
            else:
                fan_in = shape[-1] if name.endswith("weight") else config.d_token
                bound = 1 / math.sqrt(fan_in)
                array = rng.uniform(-bound, bound, shape)
            weights[name] = array.astype(np.float32)
        return cls(
            labels=tuple(labels),
            feature_fingerprint=feature_fingerprint,
            feature_names=names,
            scale=np.ones(len(names)) if scale is None else np.asarray(scale, dtype=float),
            config=config,
            weights=weights,
        )

    # -- inference --------------------------------------------------------- #

    def _tokenize(self, X: np.ndarray) -> np.ndarray:
        """Rows (already scaled) → tokens, ``(rows, tokens + 1, d)``, [CLS] last."""
        rows, d = X.shape[0], self.config.d_token
        embedding = self.weights["tokenizer.weight"]
        out = np.empty((rows, len(self.tokens) + 1, d), dtype=np.float32)
        out[:, :-1] = self.weights["tokenizer.bias"]
        for t, columns in enumerate(self._columns):
            out[:, t] += X[:, columns] @ embedding[columns]
        out[:, -1] = self.weights["cls"]
        return out

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Logits for a dense block of unscaled rows."""
        w = self.weights
        x = self._tokenize((X / self.scale).astype(np.float32))
        n_blocks = self.config.n_blocks
        for i in range(n_blocks):
            block, last = f"blocks.{i}", i == n_blocks - 1
            residual = (
                x
                if i == 0
                else _layer_norm(
                    x, w[f"{block}.attention_norm.weight"], w[f"{block}.attention_norm.bias"]
                )
            )
            query = residual[:, -1:] if last else residual
            residual = _attention(query, residual, w, block, self.config.n_heads)
            if last:
                x = x[:, -1:]
            x = x + residual
            residual = _layer_norm(x, w[f"{block}.ffn_norm.weight"], w[f"{block}.ffn_norm.bias"])
            x = x + _ffn(residual, w, block)
        x = _layer_norm(x[:, -1], w["head.norm.weight"], w["head.norm.bias"])
        return _linear(np.maximum(x, 0.0), w, "head.linear")

    def logits(self, X: Any, *, chunk: int = 1024) -> np.ndarray:
        """Logits, ``chunk`` rows at a time. Takes a dense array or any SciPy sparse matrix."""
        rows = X.shape[0]
        out = np.empty((rows, len(self.labels)), dtype=np.float64)
        for start in range(0, rows, chunk):
            block = X[start : start + chunk]
            dense = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
            out[start : start + chunk] = self._forward(dense.astype(np.float32))
        return out

    def predict_proba(self, X: Any) -> np.ndarray:
        """Softmax of the logits. NumPy only: no torch, no GPU."""
        return _softmax(self.logits(X))

    # -- size and cost ----------------------------------------------------- #

    def complexity(self) -> dict[str, Any]:
        """Parameters, float32 bytes and multiply-adds per patient (EXP-022).

        The tokenizer is counted dense (every column times its embedding), as a dense row costs;
        the NumPy path computes the same sum. The last block computes the [CLS] row only. Softmax,
        normalisation and activation costs are left out, as is customary.
        """
        d, h, n_cols = self.config.d_token, self.config.ffn_hidden, len(self.feature_names)
        n = len(self.tokens) + 1
        parameters = int(sum(w.size for w in self.weights.values()))
        tokenizer = n_cols * d
        full_block = 4 * n * d * d + 2 * n * n * d + n * 3 * d * h
        last_block = 2 * n * d * d + 2 * d * d + 2 * n * d + 3 * d * h
        blocks = (self.config.n_blocks - 1) * full_block + last_block
        head = d * len(self.labels)
        return {
            "parameters": parameters,
            "bytes_float32": 4 * parameters,
            "tokens": n,
            "macs_per_patient": int(tokenizer + blocks + head),
            "macs_by_part": {"tokenizer": tokenizer, "blocks": blocks, "head": head},
        }

    # -- files ------------------------------------------------------------- #

    def to_json(self) -> dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "labels": list(self.labels),
            "feature_fingerprint": self.feature_fingerprint,
            "feature_names": list(self.feature_names),
            "tokens": list(self.tokens),
            "scale": self.scale.tolist(),
            "config": asdict(self.config),
            "weights": {name: _encode_array(array) for name, array in self.weights.items()},
            "params": dict(self.params),
            "history": list(self.history),
        }

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], *, feature_fingerprint: str | None = None
    ) -> DeepRanker:
        """Raises ValueError if ``feature_fingerprint`` is given and differs, or a weight is
        damaged or does not fit the architecture."""
        if data.get("model") != MODEL_NAME:
            raise ValueError(f"not a {MODEL_NAME} file: {data.get('model')!r}")
        _check_fingerprint(data, feature_fingerprint)
        model = cls(
            labels=tuple(data["labels"]),
            feature_fingerprint=data["feature_fingerprint"],
            feature_names=tuple(data["feature_names"]),
            scale=np.array(data["scale"], dtype=float),
            config=TransformerConfig(**data["config"]),
            weights={name: _decode_array(name, e) for name, e in data["weights"].items()},
            params=dict(data.get("params", {})),
            history=list(data.get("history", [])),
        )
        if list(model.tokens) != list(data.get("tokens", model.tokens)):
            raise ValueError("the stored token map differs from the one its columns give")
        return model

    # -- training ---------------------------------------------------------- #

    @classmethod
    def fit(
        cls,
        X: Any,
        y: np.ndarray,
        X_holdout: Any,
        y_holdout: np.ndarray,
        *,
        feature_names: Sequence[str],
        feature_fingerprint: str,
        labels: Sequence[str] = LABELS,
        config: TransformerConfig | None = None,
        training: TrainingConfig | None = None,
        seed: int = 42,
        device: str = "cpu",
        log: Callable[[str], None] = print,
    ) -> DeepRanker:
        """Train with PyTorch, stopping early on the held-out cross-entropy; keep the best epoch.

        ``X`` may be dense or a SciPy sparse matrix; it is scaled, made dense and moved to
        ``device`` once. Deterministic algorithms are requested; a GPU run is still not
        bit-reproducible across machines, so the saved file, not a rerun, is the model.

        Raises:
            ValueError: if a class has no training rows.
            ImportError: without PyTorch.
        """
        import torch

        config = config or TransformerConfig()
        training = training or TrainingConfig()
        _check_classes(np.asarray(y), labels)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

        names = tuple(feature_names)
        scale = _max_abs(X)
        X_fit = _to_tensor(X, scale, device)
        X_hold = _to_tensor(X_holdout, scale, device)
        y_fit = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
        y_hold = torch.as_tensor(np.asarray(y_holdout), dtype=torch.long, device=device)
        tokens, column_token = token_map(names)
        net = _torch_module(len(names), column_token, len(tokens), len(labels), config).to(device)
        decay, no_decay = [], []
        for name, parameter in net.named_parameters():
            plain = name.startswith(("tokenizer", "cls")) or "norm" in name or name.endswith("bias")
            (no_decay if plain else decay).append(parameter)
        optimiser = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": training.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=training.lr,
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(seed)

        def holdout_loss() -> float:
            net.eval()
            total = 0.0
            with torch.no_grad():
                for start in range(0, len(y_hold), training.eval_batch_size):
                    end = start + training.eval_batch_size
                    total += float(
                        loss_fn(net(X_hold[start:end]), y_hold[start:end]) * len(y_hold[start:end])
                    )
            return total / len(y_hold)

        history: list[dict[str, Any]] = []
        best_loss, best_epoch, best_state = math.inf, 0, None
        rows = len(y_fit)
        for epoch in range(1, training.max_epochs + 1):
            started = time.perf_counter()
            net.train()
            order = torch.randperm(rows, generator=generator).to(device)
            running = 0.0
            for start in range(0, rows, training.batch_size):
                batch = order[start : start + training.batch_size]
                optimiser.zero_grad(set_to_none=True)
                loss = loss_fn(net(X_fit[batch]), y_fit[batch])
                loss.backward()
                optimiser.step()
                running += float(loss.detach()) * len(batch)
            held = holdout_loss()
            seconds = round(time.perf_counter() - started, 2)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(running / rows, 6),
                    "holdout_loss": round(held, 6),
                    "seconds": seconds,
                }
            )
            log(
                f"    epoch {epoch:>2}: train {running / rows:.5f}, holdout {held:.5f} ({seconds:.0f} s)"
            )
            if held < best_loss:
                best_loss, best_epoch = held, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            elif epoch - best_epoch >= training.patience:
                break
        assert best_state is not None
        return cls(
            labels=tuple(labels),
            feature_fingerprint=feature_fingerprint,
            feature_names=names,
            scale=scale,
            config=config,
            weights={k: v.numpy().astype(np.float32) for k, v in best_state.items()},
            params={
                "seed": seed,
                "device": device,
                "training": asdict(training),
                "best_epoch": best_epoch,
                "best_holdout_loss": round(best_loss, 6),
                "epochs_run": len(history),
                "torch": torch.__version__,
            },
            history=history,
        )


def _max_abs(X: Any) -> np.ndarray:
    """Each column's largest absolute value, 1 where it is 0 (as scikit-learn's MaxAbsScaler)."""
    if hasattr(X, "tocsc"):
        top = np.asarray(abs(X).max(axis=0).todense()).ravel()
    else:
        top = np.abs(np.asarray(X)).max(axis=0)
    top = np.asarray(top, dtype=float)
    top[top == 0] = 1.0
    return top


def _to_tensor(X: Any, scale: np.ndarray, device: str, chunk: int = 50_000) -> Any:
    """Scaled float32 rows on ``device``, densified ``chunk`` rows at a time."""
    import torch

    out = torch.empty((X.shape[0], X.shape[1]), dtype=torch.float32, device=device)
    for start in range(0, X.shape[0], chunk):
        block = X[start : start + chunk]
        dense = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        out[start : start + chunk] = torch.from_numpy((dense / scale).astype(np.float32))
    return out


def _torch_module(
    n_columns: int,
    column_token: np.ndarray,
    n_tokens: int,
    n_classes: int,
    config: TransformerConfig,
) -> Any:
    """The PyTorch twin of :meth:`DeepRanker._forward`, with the same parameter names."""
    import torch
    from torch import nn

    d, h, heads = config.d_token, config.ffn_hidden, config.n_heads

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.W_q, self.W_k, self.W_v, self.W_out = (nn.Linear(d, d) for _ in range(4))
            for m in (self.W_q, self.W_k, self.W_v):  # rtdl: Xavier, gain 1/sqrt(2), zero bias
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
                nn.init.zeros_(m.bias)
            nn.init.zeros_(self.W_out.bias)
            self.dropout = nn.Dropout(config.attention_dropout)

        def forward(self, x_q: Any, x_kv: Any) -> Any:
            q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
            batch, n_q, _ = q.shape
            d_head = d // heads

            def split(t: Any) -> Any:
                return t.reshape(batch, t.shape[1], heads, d_head).transpose(1, 2)

            logits = split(q) @ split(k).transpose(-1, -2) / math.sqrt(d_head)
            attention = self.dropout(torch.softmax(logits, dim=-1))
            out = (attention @ split(v)).transpose(1, 2).reshape(batch, n_q, d)
            return self.W_out(out)

    class FeedForward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_first = nn.Linear(d, 2 * h)
            self.linear_second = nn.Linear(h, d)
            self.dropout = nn.Dropout(config.ffn_dropout)

        def forward(self, x: Any) -> Any:
            a, b = self.linear_first(x).chunk(2, dim=-1)
            return self.linear_second(self.dropout(a * torch.relu(b)))

    class Block(nn.Module):
        def __init__(self, first: bool) -> None:
            super().__init__()
            if not first:
                self.attention_norm = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
            self.attention = Attention()
            self.ffn_norm = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
            self.ffn = FeedForward()

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d, eps=LAYER_NORM_EPS)
            self.linear = nn.Linear(d, n_classes)

    class Tokenizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            bound = 1 / math.sqrt(d)
            self.weight = nn.Parameter(torch.empty(n_columns, d).uniform_(-bound, bound))
            self.bias = nn.Parameter(torch.empty(n_tokens, d).uniform_(-bound, bound))
            one_hot = torch.zeros(n_columns, n_tokens)
            one_hot[torch.arange(n_columns), torch.as_tensor(column_token)] = 1.0
            self.register_buffer("one_hot", one_hot, persistent=False)

        def forward(self, x: Any) -> Any:
            # Every column's embedding placed in its token's slot, then one dense product:
            # deterministic on a GPU, unlike a scatter-add.
            spread = (self.one_hot[:, :, None] * self.weight[:, None, :]).reshape(n_columns, -1)
            return (x @ spread).reshape(x.shape[0], n_tokens, d) + self.bias

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = Tokenizer()
            bound = 1 / math.sqrt(d)
            self.cls = nn.Parameter(torch.empty(d).uniform_(-bound, bound))
            self.blocks = nn.ModuleList(Block(first=i == 0) for i in range(config.n_blocks))
            self.head = Head()
            self.residual_dropout = nn.Dropout(config.residual_dropout)

        def forward(self, x: Any) -> Any:
            x = self.tokenizer(x)
            x = torch.cat([x, self.cls.expand(x.shape[0], 1, d)], dim=1)
            for i, block in enumerate(self.blocks):
                last = i == len(self.blocks) - 1
                residual = x if i == 0 else block.attention_norm(x)
                query = residual[:, -1:] if last else residual
                residual = block.attention(query, residual)
                if last:
                    x = x[:, -1:]
                x = x + self.residual_dropout(residual)
                x = x + self.residual_dropout(block.ffn(block.ffn_norm(x)))
            x = self.head.norm(x[:, -1])
            return self.head.linear(torch.relu(x))

    return Transformer()


def torch_module_for(model: DeepRanker) -> Any:
    """A PyTorch module holding ``model``'s weights, in eval mode, on the CPU."""
    import torch

    net = _torch_module(
        len(model.feature_names),
        model.column_token,
        len(model.tokens),
        len(model.labels),
        model.config,
    )
    state = {name: torch.from_numpy(np.array(array)) for name, array in model.weights.items()}
    net.load_state_dict(state)
    return net.eval()


def parity(model: DeepRanker, X: Any) -> dict[str, Any]:
    """How far the NumPy forward pass is from PyTorch's on these rows (both on the CPU).

    Returns the largest absolute difference in probability, and in how many rows the two put a
    different condition first.
    """
    import torch

    dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    ours = model.predict_proba(dense)
    with torch.no_grad():
        scaled = torch.from_numpy((dense / model.scale).astype(np.float32))
        theirs = torch.softmax(torch_module_for(model)(scaled), dim=-1).double().numpy()
    return {
        "rows": int(dense.shape[0]),
        "max_abs_difference": float(np.abs(ours - theirs).max()) if len(dense) else 0.0,
        "top1_disagreements": int((ours.argmax(axis=1) != theirs.argmax(axis=1)).sum()),
    }
