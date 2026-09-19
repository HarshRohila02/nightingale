"""The ML baselines B0 and B1 (task 1c; docs/05 §4; EXP-003 and EXP-004).

* **B0, the prevalence prior:** every patient gets the same ranking, the training split's class
  frequencies. It is the floor: anything below it is broken.
* **B1, ML-only:** multinomial logistic regression, a linear reference, and XGBoost, B1 proper,
  both on the features from :mod:`src.ml.features`.

Each model saves to plain JSON, not a pickle. The file names the feature set by its fingerprint,
the class order and the numbers, so it loads on any machine and cannot run code. XGBoost stores
its trees in its own binary JSON format (UBJSON, ``.ubj``: portable, about a third the size of
plain JSON) beside a small metadata file. Loading a model built for another
feature set is an error.

Training happens where the owner chooses (D-7): for B0 and B1, Google Colab, through
scripts/train_baselines.py. The unit tests fit toy models on a few synthetic rows, which is
testing, not training on project data.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.conditions import CONDITIONS
from src.eval.metrics import CaseOutcome, outcome_from_record

__all__ = [
    "LABELS",
    "LogisticBaseline",
    "PrevalencePrior",
    "XGBoostBaseline",
    "label_index",
    "outcomes",
    "rankings",
]

LABELS: tuple[str, ...] = tuple(c.id for c in CONDITIONS if c.in_training_data)
"""The 13 trainable conditions, in registry order: the class order of every model."""


def label_index(condition_ids: Iterable[str], labels: Sequence[str] = LABELS) -> np.ndarray:
    """Class numbers for condition ids.

    Raises:
        ValueError: for a condition that is not one of ``labels``.
    """
    index = {label: i for i, label in enumerate(labels)}
    try:
        return np.array([index[c] for c in condition_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"{exc.args[0]} is not a trainable condition") from None


def rankings(probabilities: np.ndarray, labels: Sequence[str] = LABELS) -> list[tuple[str, ...]]:
    """Each row's conditions, best first. Ties keep the registry order, as
    :func:`src.eval.metrics.ranking_from_scores` does."""
    order = np.argsort(-probabilities, axis=1, kind="stable")
    return [tuple(labels[j] for j in row) for row in order]


def outcomes(
    frame: pd.DataFrame, probabilities: np.ndarray, labels: Sequence[str] = LABELS
) -> list[CaseOutcome]:
    """One :class:`CaseOutcome` per row of a chest-pain parquet frame, for src/eval/metrics.py.

    The scores are raw model outputs, not calibrated probabilities (that is 2b). They are passed
    on so calibration can be measured.
    """
    records = frame[["case_id", "label_condition_id", "label_differential"]].to_dict("records")
    return [
        outcome_from_record(
            record,
            ranking,
            probabilities=dict(zip(labels, map(float, row), strict=True)),
        )
        for record, ranking, row in zip(
            records, rankings(probabilities, labels), probabilities, strict=True
        )
    ]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _check_classes(y: np.ndarray, labels: Sequence[str]) -> None:
    missing = sorted(set(range(len(labels))) - set(np.unique(y).tolist()))
    if missing:
        raise ValueError(f"no training rows for {[labels[i] for i in missing]}")


def _csr(X: Any) -> Any:
    """``X`` as a SciPy CSR matrix.

    XGBoost reads a sparse matrix's absent entries as *missing* but a dense array's zeros as
    *values*, and the two give different answers from the same trees. Training uses sparse
    matrices, so every input to XGBoost goes through here: a dense row at inference time then
    means exactly what it meant in training.
    """
    from scipy import sparse

    if sparse.issparse(X) and X.format == "csr":
        return X
    return sparse.csr_matrix(X)


def _check_fingerprint(data: Mapping[str, Any], expected: str | None) -> None:
    if expected is not None and data["feature_fingerprint"] != expected:
        raise ValueError(
            f"the model was trained on features {data['feature_fingerprint']}, "
            f"not {expected}; re-encode or retrain"
        )


# --------------------------------------------------------------------------- #
# B0
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PrevalencePrior:
    """B0: the training split's class frequencies, the same for every patient."""

    labels: tuple[str, ...]
    counts: tuple[int, ...]

    @classmethod
    def fit(cls, y: np.ndarray, labels: Sequence[str] = LABELS) -> PrevalencePrior:
        counts = np.bincount(np.asarray(y), minlength=len(labels))
        return cls(tuple(labels), tuple(int(c) for c in counts))

    @property
    def priors(self) -> np.ndarray:
        counts = np.array(self.counts, dtype=float)
        return counts / counts.sum()

    def predict_proba(self, rows: int) -> np.ndarray:
        return np.tile(self.priors, (rows, 1))

    def to_json(self) -> dict[str, Any]:
        return {
            "model": "B0 prevalence prior",
            "labels": list(self.labels),
            "counts": list(self.counts),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PrevalencePrior:
        return cls(tuple(data["labels"]), tuple(int(c) for c in data["counts"]))


# --------------------------------------------------------------------------- #
# B1, linear
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LogisticBaseline:
    """B1's linear reference: multinomial logistic regression on max-abs-scaled features.

    Scaling divides each column by its largest absolute value on the training rows, so age and
    the 0-10 scales land in [0, 1] like the binary columns, and sparsity is kept.
    """

    labels: tuple[str, ...]
    feature_fingerprint: str
    scale: np.ndarray = field(repr=False)
    coef: np.ndarray = field(repr=False)
    intercept: np.ndarray = field(repr=False)
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        X: Any,
        y: np.ndarray,
        *,
        feature_fingerprint: str,
        labels: Sequence[str] = LABELS,
        C: float = 1.0,
        max_iter: int = 1000,
        seed: int = 42,
    ) -> LogisticBaseline:
        """Fit on every row of ``X``.

        Raises:
            ValueError: if a class has no training rows.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import MaxAbsScaler

        _check_classes(y, labels)
        scaler = MaxAbsScaler().fit(X)
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        model.fit(scaler.transform(X), y)
        return cls(
            labels=tuple(labels),
            feature_fingerprint=feature_fingerprint,
            scale=np.asarray(scaler.scale_, dtype=float),
            coef=np.asarray(model.coef_, dtype=float),
            intercept=np.asarray(model.intercept_, dtype=float),
            params={
                "C": C,
                "max_iter": max_iter,
                "seed": seed,
                "iterations": int(model.n_iter_.max()),
            },
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        """Softmax of the linear scores. Needs only NumPy (and SciPy for sparse input)."""
        return _softmax(np.asarray(X @ (self.coef / self.scale).T) + self.intercept)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": "B1 logistic regression",
            "labels": list(self.labels),
            "feature_fingerprint": self.feature_fingerprint,
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "params": dict(self.params),
        }

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], *, feature_fingerprint: str | None = None
    ) -> LogisticBaseline:
        """Raises ValueError if ``feature_fingerprint`` is given and differs."""
        _check_fingerprint(data, feature_fingerprint)
        return cls(
            labels=tuple(data["labels"]),
            feature_fingerprint=data["feature_fingerprint"],
            scale=np.array(data["scale"], dtype=float),
            coef=np.array(data["coef"], dtype=float),
            intercept=np.array(data["intercept"], dtype=float),
            params=dict(data.get("params", {})),
        )


# --------------------------------------------------------------------------- #
# B1, XGBoost
# --------------------------------------------------------------------------- #

XGBOOST_DEFAULTS: Mapping[str, Any] = {
    "objective": "multi:softprob",
    "tree_method": "hist",
    "eta": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
}
"""Fixed, commonly used starting values. Tuning, if any, happens on validation only, later."""


@dataclass
class XGBoostBaseline:
    """B1: gradient-boosted trees, stopped early on a held-out part of the training split."""

    labels: tuple[str, ...]
    feature_fingerprint: str
    booster: Any = field(repr=False)
    best_iteration: int
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        X: Any,
        y: np.ndarray,
        X_holdout: Any,
        y_holdout: np.ndarray,
        *,
        feature_fingerprint: str,
        labels: Sequence[str] = LABELS,
        device: str = "cpu",
        seed: int = 42,
        max_rounds: int = 1000,
        early_stopping: int = 50,
    ) -> XGBoostBaseline:
        """Boost on ``X`` and stop when the held-out log loss stops improving.

        Raises:
            ValueError: if a class has no training rows.
        """
        import xgboost as xgb

        _check_classes(y, labels)
        params = {**XGBOOST_DEFAULTS, "num_class": len(labels), "device": device, "seed": seed}
        booster = xgb.train(
            params,
            xgb.DMatrix(_csr(X), label=y),
            num_boost_round=max_rounds,
            evals=[(xgb.DMatrix(_csr(X_holdout), label=y_holdout), "holdout")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        return cls(
            labels=tuple(labels),
            feature_fingerprint=feature_fingerprint,
            booster=booster,
            best_iteration=int(booster.best_iteration),
            params={**params, "max_rounds": max_rounds, "early_stopping": early_stopping},
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        import xgboost as xgb

        return np.asarray(
            self.booster.predict(xgb.DMatrix(_csr(X)), iteration_range=(0, self.best_iteration + 1))
        )

    def save(self, directory: Path, stem: str = "b1_xgboost") -> list[Path]:
        """Writes ``<stem>.ubj`` (the trees) and ``<stem>.meta.json``; returns both paths."""
        directory.mkdir(parents=True, exist_ok=True)
        trees, meta = directory / f"{stem}.ubj", directory / f"{stem}.meta.json"
        self.booster.save_model(trees)
        meta.write_text(
            json.dumps(
                {
                    "model": "B1 XGBoost",
                    "labels": list(self.labels),
                    "feature_fingerprint": self.feature_fingerprint,
                    "best_iteration": self.best_iteration,
                    "params": dict(self.params),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return [trees, meta]

    @classmethod
    def load(
        cls, directory: Path, stem: str = "b1_xgboost", *, feature_fingerprint: str | None = None
    ) -> XGBoostBaseline:
        """Raises ValueError if ``feature_fingerprint`` is given and differs."""
        import xgboost as xgb

        meta = json.loads((directory / f"{stem}.meta.json").read_text(encoding="utf-8"))
        _check_fingerprint(meta, feature_fingerprint)
        booster = xgb.Booster()
        booster.load_model(directory / f"{stem}.ubj")
        return cls(
            labels=tuple(meta["labels"]),
            feature_fingerprint=meta["feature_fingerprint"],
            booster=booster,
            best_iteration=int(meta["best_iteration"]),
            params=dict(meta.get("params", {})),
        )
