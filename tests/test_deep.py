"""Tests for the deep ranker B1-DL, the small transformer (src/ml/deep.py, D-12).

Everything here is synthetic: random weights, toy rows and the made-up mini release of
tests/test_baselines.py. Fitting a toy model on synthetic rows is testing, not a training run on
project data (D-7). The tests marked ``torch`` need PyTorch, which CI and the main ``.venv`` do not
have; run them in ``.venv-gpu`` with the GPU hidden (``CUDA_VISIBLE_DEVICES=``), so they stay on
the CPU (D-9 is about the GPU). ``tests/fixtures/deep_parity.json`` was written by PyTorch, so the
NumPy forward pass is checked against it everywhere, CI included.

    CUDA_VISIBLE_DEVICES= .venv-gpu/Scripts/python.exe -m tests.test_deep   # rewrite the fixture
"""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from src.ml.baselines import LABELS
from src.ml.deep import (
    DeepRanker,
    TrainingConfig,
    TransformerConfig,
    _attention,
    _ffn,
    _layer_norm,
    _linear,
    token_map,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "deep_parity.json"
TRAINER = REPO_ROOT / "scripts" / "train_deep.py"
BUILDER = REPO_ROOT / "scripts" / "build_ddxplus_chestpain.py"
RELEASE = REPO_ROOT / "data" / "raw" / "ddxplus" / "release_evidences.json"
CONDITIONS = REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json"

has_torch = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not has_torch, reason="PyTorch is not installed (CI, .venv)")

NAMES = (
    "age",
    "sex=F",
    "E_1",
    "E_2",
    "E_2=V_1",
    "E_2=V_2",
    "E_3=value",
    "E_3=answered",
    "E_1=unasked",
    "E_2=unasked",
    "E_3=unasked",
)
"""A miniature of the encoder's columns: binary, categorical, ordinal and the asked channel."""
SMALL = TransformerConfig(d_token=8, n_blocks=2, n_heads=2)


def rows(count: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = (rng.random((count, len(NAMES))) < 0.4).astype(np.float32)
    X[:, 0] = rng.integers(18, 90, count)
    X[:, 6] = rng.integers(0, 11, count) * X[:, 7]
    return X


def small_model(seed: int = 0, config: TransformerConfig = SMALL) -> DeepRanker:
    model = DeepRanker.initialised(
        NAMES,
        feature_fingerprint="toy",
        config=config,
        seed=seed,
        scale=np.r_[90, np.ones(5), 10, np.ones(4)],
    )
    rng = np.random.default_rng(seed + 1)
    for name, array in model.weights.items():  # norms away from 1 and 0, so they are tested too
        if "norm" in name:
            model.weights[name] = (array + rng.normal(0, 0.3, array.shape)).astype(np.float32)
    return model


# --------------------------------------------------------------------------- #
# The token map and the architecture
# --------------------------------------------------------------------------- #


def test_columns_are_grouped_into_one_token_per_question():
    tokens, column_token = token_map(NAMES)
    assert tokens == ("age", "sex", "E_1", "E_2", "E_3")
    assert column_token.tolist() == [0, 1, 2, 3, 3, 3, 4, 4, 2, 3, 4]


def test_the_real_encoder_gives_87_tokens_and_the_recorded_size():
    """EXP-021 and docs/14 quote these numbers: 84 questions + age + sex + [CLS]."""
    if not (RELEASE.exists() and CONDITIONS.exists()):
        pytest.skip("data/ is not committed (CI)")
    from src.ml.features import EvidenceEncoder

    for asked, parameters in ((True, 225_645), (False, 217_581)):
        encoder = EvidenceEncoder.from_release(RELEASE, CONDITIONS, asked_channel=asked)
        model = DeepRanker.initialised(
            encoder.feature_names, feature_fingerprint=encoder.fingerprint
        )
        size = model.complexity()
        assert size["tokens"] == 87 and size["parameters"] == parameters
    assert size["macs_per_patient"] == 9_602_688  # the 607-column arm
    with_channel = DeepRanker.initialised(
        EvidenceEncoder.from_release(RELEASE, CONDITIONS, asked_channel=True).feature_names,
        feature_fingerprint="x",
    )
    assert with_channel.complexity()["macs_per_patient"] == 9_610_752


def test_complexity_counts_every_weight():
    model = small_model()
    size = model.complexity()
    assert size["parameters"] == sum(w.size for w in model.weights.values())
    assert size["bytes_float32"] == 4 * size["parameters"]
    d, h, n = 8, 10, 6  # ffn_hidden = int(8 * 4 / 3) = 10; 5 tokens + [CLS]
    full = 4 * n * d * d + 2 * n * n * d + 3 * n * d * h
    last = 2 * n * d * d + 2 * d * d + 2 * n * d + 3 * d * h
    assert size["macs_per_patient"] == len(NAMES) * d + full + last + d * len(LABELS)


def test_a_bad_configuration_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(d_token=10, n_heads=4)
    with pytest.raises(ValueError, match="at least one block"):
        TransformerConfig(n_blocks=0)


def test_weights_must_fit_the_architecture():
    model = small_model()
    weights = dict(model.weights)
    del weights["cls"]
    with pytest.raises(ValueError, match="missing"):
        DeepRanker(model.labels, "toy", NAMES, model.scale, SMALL, weights)
    weights = dict(model.weights, cls=np.zeros(9, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        DeepRanker(model.labels, "toy", NAMES, model.scale, SMALL, weights)


# --------------------------------------------------------------------------- #
# The NumPy forward pass
# --------------------------------------------------------------------------- #


def test_it_gives_one_probability_per_condition():
    probabilities = small_model().predict_proba(rows())
    assert probabilities.shape == (6, len(LABELS))
    assert np.all(np.isfinite(probabilities)) and np.allclose(probabilities.sum(axis=1), 1.0)


def test_dense_and_sparse_rows_answer_alike():
    model, X = small_model(), rows(20)
    assert np.array_equal(model.predict_proba(X), model.predict_proba(sparse.csr_matrix(X)))


def test_the_answer_does_not_depend_on_the_batch():
    model, X = small_model(), rows(20)
    together = model.predict_proba(X)
    one_by_one = np.vstack([model.predict_proba(X[i : i + 1]) for i in range(len(X))])
    chunked = _softmax_of(model.logits(X, chunk=3))
    assert np.allclose(together, one_by_one, atol=1e-6) and np.allclose(
        together, chunked, atol=1e-6
    )


def _softmax_of(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def test_rows_are_scaled_before_the_tokens():
    model, X = small_model(), rows()
    unscaled = copy.deepcopy(model)
    unscaled.scale = np.ones(len(NAMES))
    assert np.allclose(model.predict_proba(X), unscaled.predict_proba(X / model.scale), atol=1e-6)


def test_the_last_block_reading_only_cls_equals_the_full_computation():
    """The last block computes the [CLS] row only (as rtdl does). A full last block, then [CLS],
    must give the same logits: the shortcut changes the cost, never the answer."""
    model, X = small_model(), rows()
    w = model.weights
    x = model._tokenize((X / model.scale).astype(np.float32))
    for i in range(SMALL.n_blocks):
        block = f"blocks.{i}"
        residual = (
            x
            if i == 0
            else _layer_norm(
                x, w[f"{block}.attention_norm.weight"], w[f"{block}.attention_norm.bias"]
            )
        )
        x = x + _attention(residual, residual, w, block, SMALL.n_heads)
        x = x + _ffn(
            _layer_norm(x, w[f"{block}.ffn_norm.weight"], w[f"{block}.ffn_norm.bias"]), w, block
        )
    cls = _layer_norm(x[:, -1], w["head.norm.weight"], w["head.norm.bias"])
    full = _linear(np.maximum(cls, 0.0), w, "head.linear")
    assert np.allclose(model.logits(X), full, atol=1e-5)


def test_a_question_s_columns_reach_only_its_own_token():
    model = small_model()
    X = np.zeros((2, len(NAMES)), dtype=np.float32)
    X[1, 4] = 1.0  # E_2=V_1
    tokens = model._tokenize(X)
    changed = np.flatnonzero(np.abs(tokens[1] - tokens[0]).sum(axis=1) > 0)
    assert changed.tolist() == [model.tokens.index("E_2")]
    assert np.allclose(tokens[1, 3] - tokens[0, 3], model.weights["tokenizer.weight"][4])


def test_the_committed_pytorch_outputs_are_reproduced():
    """tests/fixtures/deep_parity.json holds a small model's weights and what PyTorch's module
    answered for six rows. The NumPy pass must agree: this is the parity check CI can run."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    model = DeepRanker.from_json(fixture["model"])
    expected = np.array(fixture["torch_probabilities"])
    got = model.predict_proba(np.array(fixture["rows"], dtype=np.float32))
    assert np.abs(got - expected).max() < 1e-6
    assert (got.argmax(axis=1) == expected.argmax(axis=1)).all()


# --------------------------------------------------------------------------- #
# The file
# --------------------------------------------------------------------------- #


def test_the_json_round_trip_is_exact():
    model, X = small_model(), rows()
    model.params = {"seed": 42}
    loaded = DeepRanker.from_json(json.loads(json.dumps(model.to_json())))
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))
    assert loaded.params == {"seed": 42} and loaded.config == SMALL
    assert loaded.tokens == model.tokens


def test_a_damaged_weight_is_refused():
    data = small_model().to_json()
    entry = data["weights"]["head.linear.bias"]
    raw = bytearray(base64.b64decode(entry["data"]))
    raw[0] ^= 1
    entry["data"] = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(ValueError, match="SHA-256"):
        DeepRanker.from_json(data)


def test_another_feature_set_or_file_kind_is_refused():
    data = small_model().to_json()
    with pytest.raises(ValueError, match="trained on features"):
        DeepRanker.from_json(data, feature_fingerprint="something-else")
    with pytest.raises(ValueError, match="not a"):
        DeepRanker.from_json({**data, "model": "B1 logistic regression"})


def test_open_ranker_opens_a_deep_bundle(tmp_path):
    from src.ml.features import EvidenceEncoder
    from src.ml.ranker import BACKENDS, ModelRanker, describe, open_ranker
    from tests.test_baselines import write_mini_release

    assert "deep" in BACKENDS
    raw, interim = write_mini_release(tmp_path)
    paths = {
        "evidences_path": raw / "release_evidences.json",
        "conditions_path": interim / "ddxplus_chestpain_conditions.json",
    }
    encoder = EvidenceEncoder.from_release(*paths.values(), asked_channel=True)
    model = DeepRanker.initialised(encoder.feature_names, feature_fingerprint=encoder.fingerprint)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "b1_dl.json").write_text(json.dumps(model.to_json()), encoding="utf-8")

    ranker = open_ranker("deep", model_dir=bundle, **paths)
    assert isinstance(ranker, ModelRanker) and ranker.encoder.asked_channel
    assert describe(ranker).startswith("ranker: the transformer B1-DL")
    assert describe(ranker).endswith("(deep) over 13 conditions")
    assert open_ranker("deep", model_dir=tmp_path / "nowhere", **paths).degraded


# --------------------------------------------------------------------------- #
# PyTorch: training, and the twin module
# --------------------------------------------------------------------------- #


def make_parity_fixture() -> dict:
    """A small model initialised by PyTorch, with its norms moved off 1 and 0, and what PyTorch
    answers for six rows."""
    import torch

    from src.ml.deep import _torch_module, torch_module_for

    torch.manual_seed(0)
    _, column_token = token_map(NAMES)
    net = _torch_module(len(NAMES), column_token, 5, len(LABELS), SMALL)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if "norm" in name:
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.3)
    weights = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    scale = np.r_[90.0, np.ones(5), 10.0, np.ones(4)]
    model = DeepRanker(tuple(LABELS), "toy", NAMES, scale, SMALL, weights)
    X = rows()
    with torch.no_grad():
        probabilities = torch.softmax(
            torch_module_for(model)(torch.from_numpy((X / scale).astype(np.float32))), dim=-1
        )
    return {
        "note": "written by tests/test_deep.py:make_parity_fixture with PyTorch; see its docstring",
        "torch": torch.__version__,
        "model": model.to_json(),
        "rows": X.tolist(),
        "torch_probabilities": probabilities.double().numpy().tolist(),
    }


@needs_torch
def test_the_fixture_is_what_pytorch_answers_today():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fresh = make_parity_fixture()
    assert np.allclose(fresh["torch_probabilities"], fixture["torch_probabilities"], atol=1e-6)


@needs_torch
def test_numpy_and_pytorch_agree_on_a_random_model():
    from src.ml.deep import parity

    checked = parity(small_model(seed=3), rows(50, seed=3))
    assert checked["max_abs_difference"] < 1e-5 and checked["top1_disagreements"] == 0


def toy(rows_per_class: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Each class lights up one of the columns, so a working model learns it quickly."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(len(LABELS)), rows_per_class)
    names = ["age", "sex=F", *(f"E_{k}" for k in range(len(LABELS)))]
    X = (rng.random((len(y), len(names))) < 0.03).astype(np.float32)
    X[:, 0] = rng.integers(18, 90, len(y))
    X[np.arange(len(y)), 2 + y] = 1.0
    return X, y, names


@needs_torch
def test_fit_learns_and_keeps_the_best_epoch():
    from src.ml.deep import parity

    X, y, names = toy()
    model = DeepRanker.fit(
        X,
        y,
        X[::5],
        y[::5],
        feature_names=names,
        feature_fingerprint="toy",
        config=SMALL,
        training=TrainingConfig(lr=1e-2, batch_size=32, max_epochs=40, patience=5),
        seed=0,
        log=lambda _: None,
    )
    # A tiny transformer learns this slowly (0.87 after 37 epochs); chance is 1 in 13.
    assert (model.predict_proba(X).argmax(axis=1) == y).mean() > 0.7
    losses = [epoch["holdout_loss"] for epoch in model.history]
    assert min(losses) < 0.6 * losses[0]
    assert model.params["best_holdout_loss"] == min(losses)
    assert losses.index(min(losses)) + 1 == model.params["best_epoch"]
    assert model.params["epochs_run"] == len(model.history) <= 40
    assert model.scale[0] == X[:, 0].max() and np.all(model.scale[1:] == 1.0)
    assert parity(model, X)["max_abs_difference"] < 1e-5


@needs_torch
def test_fit_is_reproducible_on_the_cpu():
    X, y, names = toy(10)
    kwargs = dict(
        feature_names=names,
        feature_fingerprint="toy",
        config=SMALL,
        training=TrainingConfig(lr=3e-3, batch_size=32, max_epochs=3, patience=3),
        seed=5,
        log=lambda _: None,
    )
    a = DeepRanker.fit(X, y, X[::4], y[::4], **kwargs)
    b = DeepRanker.fit(X, y, X[::4], y[::4], **kwargs)
    assert all(np.array_equal(a.weights[k], b.weights[k]) for k in a.weights)


@needs_torch
def test_every_class_needs_training_rows():
    X, y, names = toy(5)
    with pytest.raises(ValueError, match="no training rows"):
        DeepRanker.fit(X[y > 0], y[y > 0], X, y, feature_names=names, feature_fingerprint="t")


needs_the_job = pytest.mark.skipif(
    not (has_torch and importlib.util.find_spec("sklearn") and importlib.util.find_spec("scipy")),
    reason="the training job needs PyTorch, scikit-learn and SciPy together",
)


@needs_the_job
def test_the_training_job_runs_end_to_end_on_a_mini_release(tmp_path):
    from src.ml.features import EvidenceEncoder
    from src.ml.ranker import ModelRanker, open_ranker
    from tests.test_baselines import _run, write_mini_release

    raw, interim = write_mini_release(tmp_path)
    for split in ("train", "validate"):
        built = _run(
            str(BUILDER), "--split", split, "--raw-dir", str(raw), "--out-dir", str(interim)
        )
        assert built.returncode == 0, built.stderr
    (interim / "ddxplus_chestpain_test.parquet").write_bytes(b"not a parquet file")
    out = tmp_path / "out"
    command = (
        str(TRAINER),
        *("--interim", str(interim), "--raw-dir", str(raw), "--out", str(out)),
        *("--max-epochs", "3", "--resamples", "20", "--seed", "43"),
    )
    trained = _run(*command)
    assert trained.returncode == 0, trained.stdout + trained.stderr

    paths = {
        "evidences_path": raw / "release_evidences.json",
        "conditions_path": interim / "ddxplus_chestpain_conditions.json",
    }
    for arm, asked in (("b1_dl_asked_aug", True), ("b1_dl_aug", False)):
        bundle = out / arm / "seed43"
        assert {p.name for p in bundle.iterdir()} == {
            "b1_dl.json",
            "features.json",
            "metrics.json",
            "pip-freeze.txt",
            "run.json",
        }
        run = json.loads((bundle / "run.json").read_text(encoding="utf-8"))
        encoder = EvidenceEncoder.from_release(*paths.values(), asked_channel=asked)
        assert run["features"]["fingerprint"] == encoder.fingerprint
        assert run["rows"]["train"] == 2 * 13 * 40 and run["rows"]["validate"] == 13 * 15
        assert run["rows"]["train_fit"] + run["rows"]["train_holdout"] == 2 * 13 * 40
        assert run["seed"] == 43 and run["training"]["epochs_run"] <= 3
        assert run["parity"]["max_abs_difference"] < 1e-4
        assert len(run["augmentation_digest"]) == 16
        metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["models"]["b1_dl"]["model"] == "B1-DL" + ("′+aug" if asked else "+aug")
        ranker = open_ranker("deep", model_dir=bundle, **paths)
        assert isinstance(ranker, ModelRanker) and ranker.encoder.asked_channel is asked

    again = _run(*command)  # finished arms are skipped, so a dropped session resumes
    assert again.returncode == 0 and again.stdout.count("already done, skipped") == 2


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(make_parity_fixture(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE}")
    sys.exit(0)
