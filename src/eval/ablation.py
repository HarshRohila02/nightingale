"""The ablation matrix as rankings, and the paired comparisons between systems (docs/05 §5, §6).

``docs/05`` §5 defines the full system A0 and its ablations. On DDXPlus patients, which is what
EXP-012 scores, each one is a way of ranking the same 14 conditions from three arrays per
evidence level:

* ``ml``, the ML component's probabilities for the 13 conditions it is trained on, as
  ``predict_proba`` returns them (``MODEL_LABELS`` order, the registry's first 13; aortic
  dissection, which no model is trained on, has no column);
* ``graph``, the knowledge graph's log-likelihoods (``score_by_connectivity``);
* ``flagged``, which conditions the red-flag rules name.

The pipeline sorts its candidates by ``(red_flag, fused_score)`` (``src/pipeline.py``). These
functions are its vectorised twin, as ``fuse_arrays`` is for the fusion, and a test runs both on
the same inputs and requires the same order.

**What differs.**

========  =====  =====  =====  ===============================================================
Config    Model  Graph  Flags  Ranking
========  =====  =====  =====  ===============================================================
A0        yes    yes    yes    the logarithmic opinion pool at α, flagged conditions first
A1        yes    no     no     the model alone
A1b       yes    no     yes    the model alone, flagged conditions first (amendment 3a)
A2        no     yes    yes    the graph alone, flagged conditions first
A4        yes    yes    no     A0 without the flags
A7        yes    yes    yes    A0 with a B1 variant as the model (the caller passes it)
========  =====  =====  =====  ===============================================================

**What is the same by construction**, so it is not recomputed (the runner reports the identity
and a test holds it):

* **A3, no retrieval:** retrieval never ranks (``docs/02`` §6); it only attaches passages.
* **A5, no calibration:** a calibrator never reorders (``src/fusion/calibration.py``); A5 differs
  from A0 only in the probabilities shown, so only ECE and the Brier score can differ.
* **A6, no patient graph:** a DDXPlus patient has no history, so there is nothing to remove
  (decision of 2026-10-20: the patient graph is display-only).

The comparisons use the tests ``docs/05`` §6 names: McNemar's test on paired top-3 correctness
(over all patients, and over the must-not-miss ones alone), and a paired bootstrap of the
per-case reciprocal rank for MRR. Nothing here trains or tunes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.conditions import CONDITIONS
from src.eval.metrics import (
    DEFAULT_LEVEL,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Ratio,
    bootstrap_ratio,
    mcnemar,
    paired_bootstrap_difference,
)
from src.fusion import LOG_FLOOR, fuse_arrays
from src.fusion.pool import FLAT

__all__ = [
    "IDS",
    "MODEL_COLUMNS",
    "MODEL_LABELS",
    "SAME_AS_A0",
    "SIGNIFICANCE",
    "SPECS",
    "CaseArrays",
    "Spec",
    "ablation_scores",
    "case_arrays",
    "compare",
    "degradation_contrast",
    "orders",
    "point_metrics",
    "ratio_interval",
    "ranking",
    "seed_summary",
    "true_ranks",
]

IDS: tuple[str, ...] = tuple(c.id for c in CONDITIONS)
"""The 14 conditions in registry order: the columns of every array here."""
MODEL_LABELS: tuple[str, ...] = tuple(c.id for c in CONDITIONS if c.in_training_data)
"""The 13 conditions the ML models are trained on, in the order of ``predict_proba``'s columns
(``src.ml.baselines.LABELS``)."""
MODEL_COLUMNS: tuple[int, ...] = tuple(IDS.index(c) for c in MODEL_LABELS)
"""Where those 13 sit among the 14 registry columns."""
SIGNIFICANCE = 0.05
"""α for McNemar's test (docs/05 §6)."""


@dataclass(frozen=True)
class Spec:
    """What an ablation keeps (docs/05 §5)."""

    model: bool
    graph: bool
    flags: bool


SPECS: dict[str, Spec] = {
    "A0": Spec(model=True, graph=True, flags=True),
    "A1": Spec(model=True, graph=False, flags=False),
    "A1b": Spec(model=True, graph=False, flags=True),
    "A2": Spec(model=False, graph=True, flags=True),
    "A4": Spec(model=True, graph=True, flags=False),
    "A7": Spec(model=True, graph=True, flags=True),
}
"""The ablations that change the ranking. A7 ranks like A0 with another model."""

SAME_AS_A0: dict[str, str] = {
    "A3": "retrieval never ranks (docs/02 §6): it only attaches passages",
    "A5": "a calibrator never reorders; only the probabilities shown, so only ECE and Brier differ",
    "A6": "a DDXPlus patient has no history, so there is no patient graph to remove (2026-10-20)",
}
"""Ablations whose ranking is A0's by construction, and why."""


# --------------------------------------------------------------------------- #
# Rankings
# --------------------------------------------------------------------------- #


def orders(scores: np.ndarray, flagged: np.ndarray | None = None) -> np.ndarray:
    """Column indices by descending score, ties by registry order (``ranking_from_scores``).

    With ``flagged``, flagged conditions come first, as the pipeline's (red_flag, fused_score)
    sort puts them: two keys, so no precision is lost.
    """
    if flagged is None:
        return np.argsort(-scores, axis=1, kind="stable")
    return np.lexsort((-scores, ~flagged), axis=1)


def true_ranks(order: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """1-based rank of each patient's true condition, which every order must contain."""
    found = order == truth[:, None]
    if not found.any(axis=1).all():
        raise ValueError("a true condition is missing from its order, so it has no rank")
    return np.argmax(found, axis=1) + 1


def _model_only_scores(ml: np.ndarray) -> np.ndarray:
    """What the pipeline scores a result with a model and no graph: the model's log-probabilities,
    and the conditions it cannot score (dissection) at the floor. A row that abstains (all zero,
    or flat, as ``pool._model_probabilities`` defines it) scores everything at the floor."""
    ml = np.asarray(ml, dtype=float)
    if not np.all(np.isfinite(ml)) or np.any(ml < 0):
        raise ValueError("the ML ranker's scores must be finite and non-negative probabilities")
    columns = np.asarray(MODEL_COLUMNS)
    if ml.shape[1] != len(columns):
        raise ValueError(f"the model's probabilities have {ml.shape[1]} columns, not 13")
    values = ml
    total = values.sum(axis=1, keepdims=True)
    p = values / np.where(total > 0, total, 1.0)
    abstain = total[:, 0] <= 0
    if values.shape[1] > 1:
        abstain = abstain | ((p.max(axis=1) - p.min(axis=1)) < FLAT)
    with np.errstate(divide="ignore"):
        log_p = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), LOG_FLOOR)
    out = np.full((len(ml), len(IDS)), LOG_FLOOR)
    out[:, columns] = np.maximum(log_p, LOG_FLOOR)
    out[abstain] = LOG_FLOOR
    return out


def ablation_scores(spec: Spec, ml: np.ndarray, graph: np.ndarray, alpha: float) -> np.ndarray:
    """The score the pipeline sorts by for this ablation: ``(patients, 14)``, higher is better.

    Args:
        spec: What the ablation keeps.
        ml: The model's probabilities, ``(patients, 13)`` in ``MODEL_LABELS`` order.
        graph: The graph's log-likelihoods, ``(patients, 14)``.
        alpha: The graph's share of the pool (EXP-006). Used only when the model and the graph
            are both kept.
    """
    columns = MODEL_COLUMNS
    if spec.model and spec.graph:
        return fuse_arrays(np.asarray(ml), graph, columns, 1 - alpha, alpha)
    if spec.graph:
        # α = 1 is the graph's posterior exactly (``fuse_arrays``), the pipeline's graph-only score.
        none = np.zeros((len(graph), len(columns)))
        return fuse_arrays(none, graph, columns, 0.0, 1.0)
    if spec.model:
        return _model_only_scores(ml)
    raise ValueError("an ablation keeps the model, the graph or both")


def ranking(
    spec: Spec,
    ml: np.ndarray,
    graph: np.ndarray,
    flagged: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Each patient's conditions best first under this ablation: ``(patients, 14)`` indices."""
    scores = ablation_scores(spec, ml, graph, alpha)
    return orders(scores, flagged if spec.flags else None)


# --------------------------------------------------------------------------- #
# Per-patient results and the comparisons
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseArrays:
    """One system's results at one level, per patient.

    Attributes:
        ranks: The 1-based rank of each patient's true condition.
        critical: Whether the patient's true condition is must-not-miss. It is a property of the
            patients, so it is the same for every system at a level.
    """

    ranks: np.ndarray
    critical: np.ndarray

    @property
    def hit1(self) -> np.ndarray:
        return self.ranks <= 1

    @property
    def hit3(self) -> np.ndarray:
        return self.ranks <= 3

    @property
    def hit5(self) -> np.ndarray:
        return self.ranks <= 5

    @property
    def reciprocal(self) -> np.ndarray:
        return 1.0 / self.ranks


def case_arrays(order: np.ndarray, truth: np.ndarray, critical: np.ndarray) -> CaseArrays:
    """Per-patient results from a ranking, the true condition's column and the critical mask."""
    return CaseArrays(ranks=true_ranks(order, truth), critical=np.asarray(critical, dtype=bool))


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else math.nan


def point_metrics(cases: CaseArrays) -> dict[str, float]:
    """The headline ranking and safety metrics as point estimates (docs/05 §3.1-§3.2), for the
    seeds' mean and spread; the seed-42 figures also get intervals from ``metrics.evaluate``."""
    critical = cases.critical
    return {
        "top-1 accuracy": _mean(cases.hit1),
        "top-3 accuracy": _mean(cases.hit3),
        "top-5 accuracy": _mean(cases.hit5),
        "MRR": _mean(cases.reciprocal),
        "must-not-miss recall@3": _mean(cases.hit3[critical]),
        "dangerous false-negative rate": _mean(~cases.hit5[critical]),
    }


def _mcnemar_entry(hits_a: np.ndarray, hits_b: np.ndarray) -> dict[str, object]:
    test = mcnemar(hits_a, hits_b)
    only_a, only_b = int(test["only_a"]), int(test["only_b"])
    return {
        "a": _mean(hits_a),
        "b": _mean(hits_b),
        "difference": _mean(hits_a) - _mean(hits_b),
        "only_a": only_a,
        "only_b": only_b,
        "p_value": test["p_value"],
        "method": test["method"],
        "cases": int(len(hits_a)),
        "a_significantly_better": bool(test["p_value"] < SIGNIFICANCE and only_a > only_b),
    }


def compare(
    a: CaseArrays,
    b: CaseArrays,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, object]]:
    """System ``a`` against system ``b`` on the same patients, with the tests docs/05 §6 names.

    Returns:
        ``top3``: McNemar's test on paired top-3 correctness over all patients;
        ``must_not_miss_top3``: the same over the must-not-miss patients alone, which is how
        must-not-miss recall@3 is compared; ``mrr``: the paired bootstrap of the per-case
        reciprocal rank. Each carries both systems' value, the difference (a − b) and
        ``a_significantly_better``: significant at α = 0.05 **and** in a's favour. A pass says
        nothing about direction by itself, so a claim that a beats b needs that flag.
    """
    if len(a.ranks) != len(b.ranks) or not np.array_equal(a.critical, b.critical):
        raise ValueError("two systems are compared on the same patients")
    critical = a.critical
    mrr = paired_bootstrap_difference(a.reciprocal, b.reciprocal, resamples=resamples, seed=seed)
    return {
        "top3": _mcnemar_entry(a.hit3, b.hit3),
        "must_not_miss_top3": _mcnemar_entry(a.hit3[critical], b.hit3[critical]),
        "mrr": {
            "a": _mean(a.reciprocal),
            "b": _mean(b.reciprocal),
            "difference": mrr["difference"],
            "ci_low": mrr["ci_low"],
            "ci_high": mrr["ci_high"],
            "cases": mrr["cases"],
            "a_significantly_better": bool(mrr["better"] == "a"),
        },
    }


def degradation_contrast(
    full: CaseArrays,
    ablated: CaseArrays,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, float | bool]:
    """H4's two degradations and their difference (docs/05 §1, RQ4).

    H4 says removing the medical graph degrades must-not-miss recall more than it degrades
    top-1. With ``full`` the system that has the graph and ``ablated`` the one without it:
    ``degradation_must_not_miss`` = recall@3(full) − recall@3(ablated) over the must-not-miss
    patients, ``degradation_top1`` = top-1(full) − top-1(ablated) over all patients, and
    ``contrast`` their difference. The three intervals come from the same resamples of the
    patients (the repository's bootstrap: seed 42, 1,000 resamples, percentile).
    ``more_on_must_not_miss`` is True when the contrast's interval is above 0.

    The protocol does not say how "more than" is tested, so this is a choice that EXP-012's
    pre-registration must record before the test split is opened.
    """
    if len(full.ranks) != len(ablated.ranks) or not np.array_equal(full.critical, ablated.critical):
        raise ValueError("two systems are compared on the same patients")
    critical = full.critical
    n = len(full.ranks)
    drop_mnm = (full.hit3.astype(float) - ablated.hit3.astype(float)) * critical
    drop_top1 = full.hit1.astype(float) - ablated.hit1.astype(float)
    denominator = critical.astype(float)
    point_mnm = float(drop_mnm.sum() / denominator.sum()) if denominator.sum() else math.nan
    point_top1 = float(drop_top1.mean())
    rng = np.random.default_rng(seed)
    values = np.empty((resamples, 2))
    done = 0
    chunk = max(1, min(resamples, 20_000_000 // max(n, 1)))
    while done < resamples:
        size = min(chunk, resamples - done)
        index = rng.integers(0, n, size=(size, n), dtype=np.int32)
        with np.errstate(invalid="ignore", divide="ignore"):
            values[done : done + size, 0] = drop_mnm[index].sum(axis=1) / denominator[index].sum(
                axis=1
            )
        values[done : done + size, 1] = drop_top1[index].mean(axis=1)
        done += size
    tail = (1 - level) / 2
    contrast = values[:, 0] - values[:, 1]
    mnm_low, mnm_high = np.nanquantile(values[:, 0], [tail, 1 - tail])
    top1_low, top1_high = np.nanquantile(values[:, 1], [tail, 1 - tail])
    low, high = np.nanquantile(contrast, [tail, 1 - tail])
    return {
        "degradation_must_not_miss": point_mnm,
        "degradation_must_not_miss_ci": [float(mnm_low), float(mnm_high)],
        "degradation_top1": point_top1,
        "degradation_top1_ci": [float(top1_low), float(top1_high)],
        "contrast": point_mnm - point_top1,
        "contrast_ci": [float(low), float(high)],
        "more_on_must_not_miss": bool(low > 0),
    }


def ratio_interval(values: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """A mean with its percentile bootstrap interval (seed 42, 1,000 resamples)."""
    array = np.asarray(values, dtype=float)
    ratio = Ratio(array, np.ones_like(array))
    low, high = bootstrap_ratio(ratio)
    return ratio.value, low, high


def seed_summary(per_seed: Mapping[int, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation of each metric over the seeds (docs/05 §6)."""
    names = next(iter(per_seed.values())).keys() if per_seed else []
    out = {}
    for name in names:
        values = np.array([m[name] for m in per_seed.values()], dtype=float)
        out[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "seeds": int(len(values)),
        }
    return out
