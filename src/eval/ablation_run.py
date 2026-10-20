"""EXP-012's computation: every system and ablation at every level, compared and judged.

Pure arrays in, results out; no files, no models, no patients' labels beyond the arrays it is
given. ``scripts/evaluate_ablations.py`` loads the data and the models and calls this once per
seed. Keeping the computation here means it is tested on synthetic arrays, and the same code
scores validation (the dry run, EXP-012's pre-registration) and test (the single opening).

**The systems** (``docs/05`` §4, §5), for each evidence level:

* **B0**, the prevalence prior, and **B2**, the graph alone;
* every ML-only model it is given, under its own label (``B1-XGB``, ``B1-LR``, the ``+aug`` and
  ``′+aug`` variants, ``B1-DL+aug``, ``B1-DL′+aug``), ranked over its 13 conditions as
  ``scripts/evaluate_reduced_evidence.py`` ranks them;
* for A0 (the unprimed family) and A0′ (the primed one): the full system, A1, A1b, A4, and A7 with
  the logistic regression and with XGBoost trained like the component, each at its own α and at
  A0's α (errata item 18); **A2** (graph and flags, no model) once, for both.

A3, A5 and A6 have A0's ranking by construction (``ablation.SAME_AS_A0``) and are not recomputed;
A5's ECE is the one place it can differ, and it is reported when a calibrator is given.

The comparisons are the ones the verdicts read (``verdicts``) plus A0 against each ablation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.eval.ablation import (
    IDS,
    MODEL_COLUMNS,
    SPECS,
    CaseArrays,
    case_arrays,
    compare,
    degradation_contrast,
    orders,
    point_metrics,
    ranking,
)
from src.eval.metrics import (
    CALIBRATION_BINS,
    DEFAULT_RESAMPLES,
    Ratio,
    binned_gap_intervals,
    bootstrap_ratio,
)
from src.eval.verdicts import FAMILIES, PLAIN, POLICIES, Family, Table, evaluate_claims
from src.fusion import fuse_arrays

__all__ = [
    "LEVELS",
    "PRIME",
    "Level",
    "SeedRun",
    "breakdown",
    "comparison_pairs",
    "ece",
    "rank_level",
    "run_seed",
]

PRIME = "′"
LEVELS = (1.0, 0.5, 0.25)


@dataclass(frozen=True)
class Level:
    """The patients at one evidence level, whatever the model.

    Attributes:
        truth: ``(n,)`` the registry index of each patient's true condition.
        critical: ``(n,)`` whether that condition is must-not-miss.
        graph: ``(n, 14)`` the knowledge graph's log-likelihoods for the kept findings.
        flagged: ``(n, 14)`` which conditions the red-flag rules name for the kept findings.
    """

    truth: np.ndarray
    critical: np.ndarray
    graph: np.ndarray
    flagged: np.ndarray


def _suffix(family: str) -> str:
    return PRIME if family == "primed" else ""


def _kind(label: str) -> str:
    return "LR" if "LR" in label else "XGB"


def rank_level(
    level: Level,
    ml: Mapping[str, np.ndarray],
    alphas: Mapping[str, float],
    prior: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Every system's ranking at one level: ``{label: order}``.

    Args:
        level: The patients.
        ml: ``{model label: (n, 13) probabilities}`` for every model scored.
        alphas: ``{model label: α}`` for each fused component and each A7 comparator (EXP-006's
            choice for that model).
        prior: B0's 13 class probabilities, or None to leave B0 out.

    The orders of ML-only systems and B0 have 13 columns (the model's conditions, which are the
    registry's first 13, so a column is a registry index); the others have 14.
    """
    n = len(level.truth)
    out: dict[str, np.ndarray] = {}
    if prior is not None:
        out["B0"] = orders(np.tile(np.asarray(prior, dtype=float), (n, 1)))
    out["B2"] = orders(level.graph)
    out["A2"] = ranking(SPECS["A2"], None, level.graph, level.flagged, 1.0)
    for label, probabilities in ml.items():
        out[label] = orders(np.asarray(probabilities))
    for name, family in FAMILIES.items():
        if family.component not in ml:
            continue
        s = _suffix(name)
        alpha = alphas[family.component]
        component = ml[family.component]
        out[family.system] = ranking(SPECS["A0"], component, level.graph, level.flagged, alpha)
        out[f"A1{s}"] = ranking(SPECS["A1"], component, level.graph, level.flagged, alpha)
        out[f"A1b{s}"] = ranking(SPECS["A1b"], component, level.graph, level.flagged, alpha)
        out[f"A4{s}"] = ranking(SPECS["A4"], component, level.graph, level.flagged, alpha)
        for like in family.like:
            if like not in ml:
                continue
            name7 = f"A7-{_kind(like)}{s}"
            out[name7] = ranking(SPECS["A7"], ml[like], level.graph, level.flagged, alphas[like])
            out[f"{name7} (A0's α)"] = ranking(
                SPECS["A7"], ml[like], level.graph, level.flagged, alpha
            )
    return {label: order.astype(np.int8, copy=False) for label, order in out.items()}


def comparison_pairs(family_name: str, available: set[str]) -> list[tuple[str, str]]:
    """The pairs to compare for one family: what the verdicts read, then A0 against each ablation."""
    family = FAMILIES[family_name]
    s = _suffix(family_name)
    a = family.system
    against = [
        "B0",
        *PLAIN,
        "B2",
        *family.like,
        family.component,
        f"A1{s}",
        f"A1b{s}",
        "A2",
        f"A4{s}",
        *(f"A7-{_kind(x)}{s}{tag}" for x in family.like for tag in ("", " (A0's α)")),
    ]
    pairs = [(a, other) for other in against if other in available]
    pairs += [(family.component, like) for like in family.like if like in available]
    return pairs


# --------------------------------------------------------------------------- #
# Calibration of the fused score (H5, A5)
# --------------------------------------------------------------------------- #


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = CALIBRATION_BINS) -> float:
    """Top-label expected calibration error from arrays: Σ over equal-width bins of
    |Σ (confidence − correct)| / n, which is ``metrics.expected_calibration_error``."""
    confidence = np.asarray(confidence, dtype=float)
    cell = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    sums = np.bincount(cell, weights=confidence - np.asarray(correct, dtype=float), minlength=bins)
    return float(np.abs(sums).sum() / len(confidence))


def fused_calibration(
    level: Level,
    component: np.ndarray,
    alpha: float,
    calibrator: Any | None = None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """A0's top-label ECE with its interval: the probability of the top-ranked condition of the
    fused ranking (red flags reorder but change no probability), raw or calibrated.

    Args:
        calibrator: A fitted ``fusion.calibration.Calibrator`` whose columns are the registry's,
            or None for the raw fused probabilities (which is A5).
    """
    fused = fuse_arrays(component, level.graph, MODEL_COLUMNS, 1 - alpha, alpha)
    p = np.exp(fused)
    p /= p.sum(axis=1, keepdims=True)
    top = orders(fused)[:, 0]
    if calibrator is not None:
        p = calibrator.transform(p)
    rows = np.arange(len(top))
    confidence = p[rows, top]
    correct = (top == level.truth).astype(float)
    low, high = binned_gap_intervals(confidence[:, None], correct[:, None], resamples=resamples)[0]
    return {
        "value": ece(confidence, correct),
        "ci": [low, high],
        "calibrated": calibrator is not None,
        "cases": int(len(top)),
    }


# --------------------------------------------------------------------------- #
# One seed
# --------------------------------------------------------------------------- #


@dataclass
class SeedRun:
    """One seed's scored systems, their comparisons and the verdicts.

    Attributes:
        orders: ``{(label, level): order}``, as int8 indices.
        cases: ``{(label, level): CaseArrays}``.
        points: ``{(label, level): point metrics}``.
        comparisons: ``{(a, b, level): ablation.compare(...)}``.
        ece: ``{(A0 label, level): ECE entry}`` (calibrated when a calibrator was given).
        ece_raw: the same without the calibrator, which is A5's, when a calibrator was given.
        h4: ``{(A0 label, "A1" | "A1b"): degradation_contrast(...)}`` at 100%.
        claims: ``{family: {policy: {claim: verdict}}}``.
    """

    orders: dict[tuple[str, float], np.ndarray] = field(default_factory=dict)
    cases: dict[tuple[str, float], CaseArrays] = field(default_factory=dict)
    points: dict[tuple[str, float], dict[str, float]] = field(default_factory=dict)
    comparisons: dict[tuple[str, str, float], dict[str, Any]] = field(default_factory=dict)
    ece: dict[tuple[str, float], dict[str, Any]] = field(default_factory=dict)
    ece_raw: dict[tuple[str, float], dict[str, Any]] = field(default_factory=dict)
    h4: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    claims: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)


def run_seed(
    levels: Mapping[float, Level],
    ml: Mapping[str, Mapping[float, np.ndarray]],
    alphas: Mapping[str, float],
    *,
    prior: np.ndarray | None = None,
    calibrators: Mapping[str, Any] | None = None,
    unsupported_rate: Mapping[str, float] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    families: Sequence[str] | None = None,
) -> SeedRun:
    """Rank every system at every level, compare, and judge.

    Args:
        levels: ``{evidence level: Level}``; must include 1.0 and the reduced levels the policies
            use.
        ml: ``{model label: {level: (n, 13) probabilities}}``.
        alphas: ``{model label: α}`` (EXP-006), for each fused component and A7 comparator.
        prior: B0's class probabilities.
        calibrators: ``{family: Calibrator}`` for A0's / A0′'s fused score (EXP-007 stage 2), or
            none: A5 is then A0.
        unsupported_rate: ``{family: rate}`` from EXP-009/010 for the Target, if run.
        resamples: Bootstrap resamples (1,000 in the protocol; fewer only in tests).
        families: The families to judge; default all of :data:`~src.eval.verdicts.FAMILIES`
            whose component is in ``ml``.
    """
    run = SeedRun()
    for level, lv in levels.items():
        ranked = rank_level(lv, {label: by[level] for label, by in ml.items()}, alphas, prior)
        for label, order in ranked.items():
            run.orders[(label, level)] = order
            run.cases[(label, level)] = case_arrays(order, lv.truth, lv.critical)
            run.points[(label, level)] = point_metrics(run.cases[(label, level)])

    names = [
        name
        for name in (families or FAMILIES)
        if FAMILIES[name].component in ml and all(like in ml for like in FAMILIES[name].like)
    ]
    available = {label for label, _ in run.cases}
    for name in names:
        family = FAMILIES[name]
        for level in levels:
            for a, b in comparison_pairs(name, available):
                run.comparisons[(a, b, level)] = compare(
                    run.cases[(a, level)], run.cases[(b, level)], resamples=resamples
                )
        calibrator = (calibrators or {}).get(name)
        for level, lv in levels.items():
            component = np.asarray(ml[family.component][level])
            alpha = alphas[family.component]
            run.ece[(family.system, level)] = fused_calibration(
                lv, component, alpha, calibrator, resamples=resamples
            )
            if calibrator is not None:
                run.ece_raw[(family.system, level)] = fused_calibration(
                    lv, component, alpha, None, resamples=resamples
                )
        s = _suffix(name)
        for ablation_name in ("A1", "A1b"):
            run.h4[(family.system, ablation_name)] = degradation_contrast(
                run.cases[(family.system, 1.0)],
                run.cases[(f"{ablation_name}{s}", 1.0)],
                resamples=resamples,
            )
        table = Table(comparisons=run.comparisons, points=run.points, ece=run.ece, h4=run.h4)
        run.claims[name] = _judge(table, family, (unsupported_rate or {}).get(name))
    return run


def _judge(table: Table, family: Family, unsupported: float | None) -> dict[str, dict[str, Any]]:
    return {
        policy: evaluate_claims(table, family, levels, unsupported_rate=unsupported)
        for policy, levels in POLICIES.items()
    }


# --------------------------------------------------------------------------- #
# Breakdowns (4b)
# --------------------------------------------------------------------------- #


def breakdown(
    cases: CaseArrays, groups: np.ndarray, *, resamples: int = DEFAULT_RESAMPLES
) -> dict[str, dict[str, Any]]:
    """Per-group results with intervals: by condition or by sex (docs/05 §8 rule 3).

    Reports, for each group, the number of patients, top-1 and top-3 accuracy and MRR, and
    must-not-miss recall@3 where the group's patients are must-not-miss. DDXPlus's synthetic
    demographics make by-sex parity an artifact, not fairness (EXP-002).
    """
    groups = np.asarray(groups)
    out: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups.tolist())):
        sel = groups == group
        sub = CaseArrays(ranks=cases.ranks[sel], critical=cases.critical[sel])
        row: dict[str, Any] = {"cases": int(sel.sum())}
        metrics = (
            ("top-1 accuracy", sub.hit1),
            ("top-3 accuracy", sub.hit3),
            ("MRR", sub.reciprocal),
        )
        for name, values in metrics:
            ratio = Ratio(values.astype(float), np.ones(len(values)))
            low, high = bootstrap_ratio(ratio, resamples=resamples)
            row[name] = {"value": ratio.value, "ci": [low, high]}
        if sub.critical.any():
            hits = sub.hit3[sub.critical].astype(float)
            ratio = Ratio(hits, np.ones(len(hits)))
            low, high = bootstrap_ratio(ratio, resamples=resamples)
            row["must-not-miss recall@3"] = {"value": ratio.value, "ci": [low, high]}
        out[str(group)] = row
    return out


def condition_names(truth: np.ndarray) -> np.ndarray:
    """The true conditions' ids, for :func:`breakdown`."""
    ids = np.asarray(IDS)
    return ids[np.asarray(truth)]
