"""Fusion: the ML ranker's probabilities and the graph's log-likelihoods as one ranking (2c).

``docs/02`` §6. A **logarithmic opinion pool** (Genest and Zidek, 1986), also called a product of
experts. Over the conditions the model was trained on, M:

    log q(c) = (1 − α) · log max(p_ml(c), ε) + α · log P_kg(c),    normalised over M,

where ``P_kg`` is the graph's posterior (a softmax of its log-likelihoods, with a uniform prior) and
α is the graph's weight, ``kg_weight / (ml_weight + kg_weight)``, the one quantity EXP-006 tunes on
validation. Two weights of 0.5 give the geometric mean of the two opinions. At α = 0 (all the weight
on the model) the floor ε is not applied, so the 13 keep the model's own order; at α = 1 the
result is the graph's posterior exactly, ties included.

Three choices were fixed on 2026-09-25, before any ranking under this design was computed for a
validation patient and before any fused ranking on validation was scored:

1. **A condition the model cannot score takes its probability from the graph alone.** DDXPlus has
   no aortic dissection, so the model is silent about it, not against it. The graph's posterior
   gives each such condition o its share, ``P(o) = P_kg(o)``, and the model's conditions share the
   rest in the pool's proportions, ``P(c) = (1 − Σ_o P_kg(o)) · pool(c)``. The first design,
   min-max normalisation, capped dissection at the graph's weight, so it could never rank first
   however strongly the graph supported it (EXP-005 point 7). **What this implies** (found in
   review, 2026-09-30): P(dissection) is the same at every α, and α moves its *rank* only through
   how sharply the 13 are pooled. Dissection enters the top 3 whenever P_kg(dissection) exceeds the
   fused probability of the pool's third condition. With a near-certain model that happens for
   findings that score 0 on the ADD-RS, such as pain going to the back or hypertension, so the
   fusion can rank dissection above where the graph alone does. Whether that is wanted is a
   clinical question for the team (A-2). The team kept the setting on 2026-10-02 without
   answering it separately, so it stays open.
2. **No model probability below ε = 0.01 is taken at face value.** A pool multiplies, so one
   expert's near-zero vetoes a condition whatever the other says. At α = 0.5 the graph must then
   prefer a condition about 100 times over a near-certain model's answer to put it first. GC-001
   motivated the floor (the retrained models give unstable angina 0.98–1.00 and MI almost 0, a
   history no model can settle: EXP-018, EXP-019 point 7). On the golden cases it changes no
   expectation and no required condition's rank (GC-001's MI is 2nd either way), though it
   reorders lower places (for B1-LR′+aug, GC-001's 3rd and 4th swap); its case rests on the
   argument, not on a measured case. The value was chosen with GC-001 in view, and no other was
   tried. It is not tuned: DDXPlus, where the models are right, cannot measure what it guards
   against.
3. **An absent model abstains.** A model with no scores, a zero sum, or flat probabilities (a
   degraded ranker) leaves the graph to rank alone, so the order is exactly the graph's; a graph
   with no scores leaves the model's order exactly, and a condition only the graph could score
   comes last. A *flat graph* (a case none of whose findings it knows) does not abstain: its
   posterior is then uniform, and dissection takes 1/14.

The scores are log-probabilities, at most 0, clamped at :data:`LOG_FLOOR`. They are not calibrated
(2b) and must never be shown as probabilities. The ranker's scores must be non-negative and finite
(probabilities, not logits): :func:`fuse_scores` raises otherwise, and the pipeline treats such a
ranker as failed (``degraded_components: ["ml"]``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_KG_WEIGHT",
    "DEFAULT_ML_WEIGHT",
    "LOG_FLOOR",
    "ML_FLOOR",
    "fuse_arrays",
    "fuse_scores",
    "graph_weight",
    "valid_model_scores",
]

DEFAULT_ML_WEIGHT = 0.5
DEFAULT_KG_WEIGHT = 0.5
ML_FLOOR = 0.01
"""ε: no model probability below this is taken at face value (choice 2 above)."""
LOG_FLOOR = -700.0
"""The lowest fused score: a log-probability under this is treated as zero."""
FLAT = 1e-12
"""Normalised model probabilities closer together than this count as flat, and abstain."""


def graph_weight(ml_weight: float, kg_weight: float) -> float:
    """α, the graph's share of the pool."""
    if not (math.isfinite(ml_weight) and math.isfinite(kg_weight)):
        raise ValueError(f"weights must be finite: {ml_weight}, {kg_weight}")
    if ml_weight < 0 or kg_weight < 0 or ml_weight + kg_weight <= 0:
        raise ValueError(f"weights must be non-negative, not both 0: {ml_weight}, {kg_weight}")
    return kg_weight / (ml_weight + kg_weight)


def valid_model_scores(ml_scores: Mapping[str, float]) -> bool:
    """Whether a ranker's scores are what the pool reads: finite and non-negative."""
    return all(math.isfinite(v) and v >= 0 for v in ml_scores.values())


def _log_softmax(scores: Mapping[str, float]) -> dict[str, float]:
    top = max(scores.values())
    total = top + math.log(sum(math.exp(v - top) for v in scores.values()))
    return {k: v - total for k, v in scores.items()}


def _log_sum_exp(values: list[float]) -> float:
    if not values:
        return -math.inf
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def _model_probabilities(ml_scores: Mapping[str, float]) -> dict[str, float] | None:
    """The model's scores as probabilities over its conditions, or None when it abstains."""
    if not ml_scores:
        return None
    if not valid_model_scores(ml_scores):
        raise ValueError("the ML ranker's scores must be finite and non-negative probabilities")
    total = sum(ml_scores.values())
    if total <= 0:
        return None
    p = {k: v / total for k, v in ml_scores.items()}
    if len(p) > 1 and max(p.values()) - min(p.values()) < FLAT:
        return None  # flat: a degraded ranker (src/ml/ranker.py)
    return p


def fuse_scores(
    ml_scores: Mapping[str, float],
    kg_scores: Mapping[str, float],
    ml_weight: float = DEFAULT_ML_WEIGHT,
    kg_weight: float = DEFAULT_KG_WEIGHT,
    *,
    ml_floor: float = ML_FLOOR,
) -> dict[str, float]:
    """Pool the two signals into one log-probability per condition.

    Args:
        ml_scores: {condition id: the model's probability}, over the conditions it was trained
            on. Scores are divided by their sum, so they need not sum to exactly 1, but they must
            be finite and non-negative.
        kg_scores: {condition id: the graph's log-likelihood} (``score_by_connectivity``).
        ml_weight: The model's weight; only the ratio of the two weights matters.
        kg_weight: The graph's weight. Tuned on validation only (``docs/05`` §2, §3.7) and
            reported: a result, not a hidden hyperparameter.
        ml_floor: ε, the lowest model probability taken at face value when α > 0.

    Returns:
        {condition id: fused log-probability} over the union of both inputs, empty when neither
        signal says anything. At α = 1 it is the graph's posterior exactly, over the graph's
        conditions.

    Raises:
        ValueError: for negative or non-finite model scores, or invalid weights.
    """
    alpha = graph_weight(ml_weight, kg_weight)
    p_ml = _model_probabilities(ml_scores)
    kg = _log_softmax(kg_scores) if kg_scores else None
    if p_ml is None and kg is None:
        return {}
    if p_ml is None or (alpha == 1 and kg is not None):
        # α = 1 is the graph alone, exactly: pooling would round the model's conditions
        # differently from the others, and break the graph's ties (scoring.ROUND makes them common).
        assert kg is not None
        return {c: max(v, LOG_FLOOR) for c, v in kg.items()}
    if kg is None:
        return {c: max(math.log(p), LOG_FLOOR) if p > 0 else LOG_FLOOR for c, p in p_ml.items()}

    lowest = min(kg.values())
    pooled = {}
    for condition, p in p_ml.items():
        if alpha == 0:
            model_term = math.log(p) if p > 0 else LOG_FLOOR
        else:
            model_term = math.log(max(p, ml_floor))
        pooled[condition] = (1 - alpha) * model_term + alpha * kg.get(condition, lowest)
    pool = _log_softmax(pooled)

    # The graph's posterior mass on the model's conditions; the rest belongs to the others.
    inside = _log_sum_exp([v for c, v in kg.items() if c in p_ml])
    fused = {c: max(inside + v, LOG_FLOOR) for c, v in pool.items()}
    for condition, v in kg.items():
        if condition not in p_ml:
            fused[condition] = max(v, LOG_FLOOR)
    return fused


def _log_sum_exp_rows(values: np.ndarray) -> np.ndarray:
    top = values.max(axis=1, keepdims=True)
    return top + np.log(np.exp(values - top).sum(axis=1, keepdims=True))


def fuse_arrays(
    ml: np.ndarray,
    kg: np.ndarray,
    model_columns: Sequence[int],
    ml_weight: float = DEFAULT_ML_WEIGHT,
    kg_weight: float = DEFAULT_KG_WEIGHT,
    *,
    ml_floor: float = ML_FLOOR,
) -> np.ndarray:
    """:func:`fuse_scores` for many patients at once, for the weight sweep (EXP-006).

    Args:
        ml: One row per patient, one column per condition the model was trained on.
        kg: One row per patient, one column per condition the graph scores (all of them).
        model_columns: For each column of ``ml``, the column of ``kg`` for the same condition.

    Returns:
        The fused log-probabilities, shaped like ``kg``; row for row what :func:`fuse_scores`
        returns for the same inputs (a test checks it). A row of ``ml`` that sums to 0 or is flat
        abstains.

    Raises:
        ValueError: for negative or non-finite model scores, or invalid weights.
    """
    alpha = graph_weight(ml_weight, kg_weight)
    ml = np.asarray(ml, dtype=float)
    kg = np.asarray(kg, dtype=float)
    if not np.all(np.isfinite(ml)) or np.any(ml < 0):
        raise ValueError("the ML ranker's scores must be finite and non-negative probabilities")
    columns = np.asarray(model_columns)
    kg_posterior = kg - _log_sum_exp_rows(kg)
    total = ml.sum(axis=1, keepdims=True)
    empty = total[:, 0] <= 0
    p = ml / np.where(total > 0, total, 1.0)
    with np.errstate(divide="ignore"):
        if alpha == 0:
            model_term = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), LOG_FLOOR)
        else:
            model_term = np.log(np.maximum(p, ml_floor))
    pooled = (1 - alpha) * model_term + alpha * kg_posterior[:, columns]
    pool = pooled - _log_sum_exp_rows(pooled)
    inside = _log_sum_exp_rows(kg_posterior[:, columns])
    fused = kg_posterior.copy()
    fused[:, columns] = inside + pool
    abstain = empty | ((p.max(axis=1) - p.min(axis=1)) < FLAT)
    if ml.shape[1] == 1:
        abstain = empty
    if alpha == 1:
        abstain[:] = True  # the graph alone, exactly (see fuse_scores)
    fused[abstain] = kg_posterior[abstain]
    return np.maximum(fused, LOG_FLOOR)
