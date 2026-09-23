"""One model's scores as a report, shared by every training job.

Each training job writes a `metrics.json` that later jobs are compared against, so the two must
mean the same thing field for field. That only holds if they are produced by the same code:
`scripts/train_baselines.py` wrote this inline, and a second job — the deep ranker, the
mask-augmented arm — would otherwise have reimplemented it slightly differently and quietly made
the comparison invalid.

Everything here is `docs/05` as amended: every ratio metric carries its 95% bootstrap interval
(§8), the calibration figures describe *raw* model outputs, since calibration itself is 2b (§3.3),
and comparisons between two systems use McNemar on top-3 (§6).

This module scores; it never trains, never loads a model and never touches a file.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.eval.metrics import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    CaseOutcome,
    bootstrap_ratio,
    evaluate,
    expected_calibration_error,
    f1_by_condition,
    mcnemar,
    reliability_table,
    top_k,
)
from src.ml.baselines import LABELS, outcomes

__all__ = ["compare", "score", "top3_hits", "top_k_hits"]


def score(
    name: str,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    labels: Sequence[str] = LABELS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[CaseOutcome]]:
    """Every `docs/05` metric for one model on one split, ready for `metrics.json`.

    Args:
        name: How the model is named in the report, e.g. ``"B1 XGBoost"``.
        frame: The scored split, which must carry the label columns.
        probabilities: One row per patient, one column per class in ``labels`` order.
        labels: The model's class order; :data:`~src.ml.baselines.LABELS` by default.
        resamples: Bootstrap resamples per interval.
        seed: The bootstrap seed, so an interval reproduces exactly.
        extra: Anything the job wants recorded beside the metrics — the evidence level of a
            reduced-evidence run, the seed a stochastic model was trained with.

    Returns:
        ``(report, cases)``. The report is JSON-ready; the cases are kept so that the caller can
        run :func:`compare` without scoring twice.
    """
    cases = outcomes(frame, probabilities, labels)
    per_condition = {}
    for label in labels:
        subset = [o for o in cases if o.true_condition == label]
        ratio = top_k(subset, 3)
        low, high = bootstrap_ratio(ratio, resamples=resamples, seed=seed)
        per_condition[label] = {"cases": len(subset), "top3": ratio.value, "ci": [low, high]}
    report: dict[str, Any] = {
        "model": name,
        "metrics": [
            {"name": r.name, "value": r.value, "ci": [r.ci_low, r.ci_high], "cases": r.cases}
            for r in evaluate(cases, resamples=resamples, seed=seed)
        ],
        "f1": f1_by_condition(cases),
        "top3_by_condition": per_condition,
        "calibration_of_raw_scores": {
            "note": "uncalibrated model outputs; calibration is 2b (docs/05 §3.3)",
            "ece": expected_calibration_error(cases),
            "reliability": reliability_table(cases),
        },
    }
    if extra:
        report.update(extra)
    return report, cases


def top_k_hits(cases: Sequence[CaseOutcome], k: int = 3) -> list[bool]:
    """Per case: was the true condition in the top ``k``? The input McNemar needs."""
    return [o.true_condition in o.ranking[:k] for o in cases]


def top3_hits(cases: Sequence[CaseOutcome]) -> list[bool]:
    """``top_k_hits(cases, 3)``. The protocol compares systems on top-3 (docs/05 §6)."""
    return top_k_hits(cases, 3)


def compare(
    name_a: str,
    cases_a: Sequence[CaseOutcome],
    name_b: str,
    cases_b: Sequence[CaseOutcome],
    k: int = 3,
) -> dict[str, Any]:
    """McNemar between two systems scored on the same patients, in the same order (docs/05 §6).

    Raises:
        ValueError: if the two were not scored on the same cases. Comparing two different patient
            sets with a paired test is the quiet mistake this guards against.
    """
    if [o.case_id for o in cases_a] != [o.case_id for o in cases_b]:
        raise ValueError(
            f"{name_a} and {name_b} were scored on different cases; McNemar is a paired test"
        )
    return {
        "a": name_a,
        "b": name_b,
        "metric": f"top-{k}",
        **mcnemar(top_k_hits(cases_a, k), top_k_hits(cases_b, k)),
    }
