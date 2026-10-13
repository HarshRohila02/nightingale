"""The model's reasons for a ranking: leave-one-finding-out occlusion (2b).

The graph explains itself through its paths and the ✓ / ✗ / ? analysis; the model does not. This
asks the model directly, for one case: **remove each recorded finding in turn, rescore, and see
how each condition's odds move.** A finding whose removal lowers a condition's odds is a reason
for it; one whose removal raises them is a reason against it. A denial is a finding too: removing
"no pain on exertion" can raise or lower a condition.

* **By finding, not by question or column.** One finding can produce several tokens (an answer and
  the question it follows up), and two findings can share a question. Removing the finding from
  the case and encoding again (:func:`~src.ml.case_tokens.encode_case`) keeps every such link as
  the model would see it. Findings the pipeline derived for the graph (``source = "crosswalk"``,
  from :func:`~src.medical_kg.crosswalk.expand_case`) are not reasons themselves; they go with the
  finding they came from.
* **Measured in log-odds,** ``log p(c) − log Σ_{j≠c} p(j)``, with the sum over the other conditions
  taken directly, so a near-certain model's 0.9998 against 0.9990 still has a size. A change
  smaller than :data:`MIN_LOG_ODDS` is no reason either way.
* **Direction and order only.** The numbers are kept for tests and audits; the explanation names
  findings, strongest first, and never shows a probability (the models are near-certain on
  hand-written cases, EXP-018, EXP-019; calibration fixes none of that, EXP-007).
* **One batch:** the case and each case-without-one-finding are scored in a single call, which
  answers for every condition at once. Nothing is stored on the ranker, so a cached ranker (the
  dashboard's) is safe.

**Limits, stated wherever the reasons are shown:** each finding is judged alone, so two findings
that only matter together (most red-flag patterns) can each look weak; and a finding the crosswalk
cannot turn into a token is invisible to the model and is never a reason (``CaseTokens.dropped``
says why). A condition the model was not trained on (aortic dissection) has no model reasons: the
graph and the rules rank it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.contracts import Assertion, Finding, PatientCase
from src.ml.case_tokens import UnrepresentableCase, encode_case

__all__ = [
    "MIN_LOG_ODDS",
    "REASONS_SHOWN",
    "Reason",
    "Reasons",
    "log_odds",
    "occlusion",
    "reasons_sentence",
]

MIN_LOG_ODDS = 0.1
"""The smallest change in a condition's log-odds counted as a reason: about a 10% change in its
odds. A stated convention, not tuned."""
REASONS_SHOWN = 3
"""How many reasons each way an explanation names."""
TINY = 1e-300


@dataclass(frozen=True)
class Reason:
    """One recorded finding, and how much the condition's log-odds depend on it."""

    concept_id: str
    label: str
    denied: bool
    effect: float
    """Log-odds with the finding minus log-odds without it: positive supports the condition."""

    def text(self) -> str:
        return f"{self.label} (denied)" if self.denied else self.label


@dataclass(frozen=True)
class Reasons:
    """The model's reasons for and against one condition, strongest first."""

    condition_id: str
    supporting: tuple[Reason, ...] = ()
    against: tuple[Reason, ...] = ()

    def top(self, k: int) -> Reasons:
        return Reasons(self.condition_id, self.supporting[:k], self.against[:k])


def log_odds(probabilities: np.ndarray) -> np.ndarray:
    """``log p(c) − log Σ_{j≠c} p(j)`` for every row and column. The other columns are summed
    directly rather than as 1 − p(c), which would round a near-certain p(c) to an odds of ∞."""
    p = np.asarray(probabilities, dtype=float)
    others = p @ (np.ones((p.shape[1], p.shape[1])) - np.eye(p.shape[1]))
    return np.log(np.maximum(p, TINY)) - np.log(np.maximum(others, TINY))


def _own_findings(case: PatientCase) -> list[tuple[str, int, Finding]]:
    """(list name, index, finding) for every finding recorded for the case itself."""
    return [
        (name, i, finding)
        for name in ("findings", "risk_factors")
        for i, finding in enumerate(getattr(case, name))
        if finding.source != "crosswalk"
    ]


def _without(case: PatientCase, name: str, index: int) -> PatientCase:
    """The case without one recorded finding, nor the findings derived from it for the graph."""
    removed = getattr(case, name)[index]

    def keep(finding: Finding) -> bool:
        derived = finding.source == "crosswalk"
        return not (derived and finding.qualifiers.get("crosswalk_from") == removed.concept_id)

    changes = {
        list_name: [
            f
            for i, f in enumerate(getattr(case, list_name))
            if not (list_name == name and i == index) and keep(f)
        ]
        for list_name in ("findings", "risk_factors")
    }
    return case.model_copy(update=changes)


def occlusion(
    model: Any,
    encoder: Any,
    case: PatientCase,
    *,
    include_narrower: bool = True,
    encode: Callable[..., tuple[np.ndarray, Any]] | None = None,
    min_effect: float = MIN_LOG_ODDS,
) -> dict[str, Reasons] | None:
    """Every condition's reasons, from one batch: the case, then the case without each finding.

    Args:
        model: Anything with ``labels`` and ``predict_proba``.
        encoder: The model's :class:`~src.ml.features.EvidenceEncoder`.
        case: The case as the pipeline ranked it; findings derived for the graph are allowed.
        include_narrower: As :func:`~src.ml.case_tokens.encode_case` (decision A-8).
        encode: The encoder of one case (default :func:`~src.ml.case_tokens.encode_case`); tests
            pass a toy one.
        min_effect: The smallest change in log-odds that counts.

    Returns:
        {condition id: :class:`Reasons`} for the model's conditions, each list strongest first;
        None for a case the model cannot hold (``Sex.OTHER``).
    """
    encode = encode or encode_case
    recorded = _own_findings(case)
    labels: Sequence[str] = list(model.labels)
    if not recorded:
        return {label: Reasons(label) for label in labels}
    variants = [case, *(_without(case, name, i) for name, i, _ in recorded)]
    try:
        rows = np.vstack(
            [encode(v, encoder, include_narrower=include_narrower)[0] for v in variants]
        )
    except UnrepresentableCase:
        return None
    odds = log_odds(np.asarray(model.predict_proba(rows)))
    effect = odds[0] - odds[1:]  # one row per recorded finding, one column per condition
    out: dict[str, Reasons] = {}
    for j, label in enumerate(labels):
        reasons = [
            Reason(
                concept_id=finding.concept_id,
                label=finding.label,
                denied=finding.assertion is Assertion.ABSENT,
                effect=float(effect[r, j]),
            )
            for r, (_, _, finding) in enumerate(recorded)
        ]
        out[label] = Reasons(
            label,
            supporting=tuple(
                sorted((x for x in reasons if x.effect >= min_effect), key=lambda x: -x.effect)
            ),
            against=tuple(
                sorted((x for x in reasons if x.effect <= -min_effect), key=lambda x: x.effect)
            ),
        )
    return out


def reasons_sentence(reasons: Mapping[str, Reasons], condition_id: str, label: str, k: int) -> str:
    """One sentence naming the model's strongest reasons for and against ``label``, or ""."""
    entry = reasons.get(condition_id)
    if entry is None or not (entry.supporting or entry.against):
        return ""
    entry = entry.top(k)
    parts = []
    if entry.supporting:
        parts.append(
            f"The model's strongest reasons for {label}: "
            + ", ".join(r.text() for r in entry.supporting)
        )
    if entry.against:
        parts.append(
            ("against it: " if parts else f"The model's strongest reasons against {label}: ")
            + ", ".join(r.text() for r in entry.against)
        )
    return "; ".join(parts) + " (each finding removed alone; direction and order only)."
