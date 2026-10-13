"""What the dashboard shows, as plain data (task 2e). No Streamlit here, so CI tests it.

``app/streamlit_app.py`` draws these; this module decides what they say:

* :func:`build_case` turns the form into a :class:`~src.contracts.PatientCase`: every finding is
  **present**, **denied** or, by not being chosen, **not asked** (FR-1.3). A denial is evidence; a
  question not asked is none.
* :func:`candidate_rows` turns a result into one row per condition. A row gives **ranks, not raw
  scores**: the fused rank, the model's rank and the graph's rank. The project never shows a raw
  score as a probability, because the models are near-certain on hand-written cases (EXP-019), and
  calibration is 2b's. A condition the model was not trained on (aortic dissection) has no model
  rank, and when the graph failed there is no graph rank. The CLI used to show that case as "kg
  0.00", the best possible graph score (2e's note).
* :func:`example_cases`: the golden cases, as form values (FR-1.5).

A row also carries **the model's reasons** for its condition when the ranker gives them (2b,
``src/ml/reasons.py``): the recorded findings whose removal moves the condition's odds most, for
and against, names only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.conditions import BY_ID
from src.contracts import Assertion, DiagnosisResult, Finding, PatientCase
from src.medical_kg.crosswalk import BY_CONCEPT, recorded_answers

__all__ = [
    "build_case",
    "candidate_rows",
    "concept_options",
    "example_cases",
    "model_reasons",
    "ranks_of",
]

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden_cases.yaml"


def concept_options() -> dict[str, list[tuple[str, str]]]:
    """{"finding" | "risk_factor": [(concept id, label), ...]}, by label: what the form offers."""
    options: dict[str, list[tuple[str, str]]] = {"finding": [], "risk_factor": []}
    for concept_id, entry in BY_CONCEPT.items():
        kind = "risk_factor" if concept_id.startswith("RF:") else "finding"
        options[kind].append((concept_id, entry.label))
    for values in options.values():
        values.sort(key=lambda pair: pair[1].lower())
    return options


def build_case(
    age: int,
    sex: str,
    present: Sequence[str],
    denied: Sequence[str] = (),
    *,
    case_id: str = "dashboard",
) -> PatientCase:
    """The form as a case: chosen concepts present or denied, everything else not asked.

    Raises:
        ValueError: for a concept marked both present and denied, or one the crosswalk does not
            know.
    """
    both = sorted(set(present) & set(denied))
    if both:
        raise ValueError(f"marked both present and denied: {', '.join(both)}")
    unknown = sorted((set(present) | set(denied)) - set(BY_CONCEPT))
    if unknown:
        raise ValueError(f"not concepts the system knows: {', '.join(unknown)}")
    findings: list[Finding] = []
    risk_factors: list[Finding] = []
    for concept_id, assertion in [
        *((c, Assertion.PRESENT) for c in present),
        *((c, Assertion.ABSENT) for c in denied),
    ]:
        finding = Finding(
            concept_id=concept_id, label=BY_CONCEPT[concept_id].label, assertion=assertion
        )
        (risk_factors if concept_id.startswith("RF:") else findings).append(finding)
    return PatientCase(
        case_id=case_id, age=age, sex=sex, findings=findings, risk_factors=risk_factors
    )


def ranks_of(scores: Mapping[str, float]) -> dict[str, int]:
    """1-based ranks, best first; a tie shares the better rank."""
    ordered = sorted(scores.values(), reverse=True)
    return {cid: ordered.index(value) + 1 for cid, value in scores.items()}


def model_reasons(ranker: Any, case: PatientCase) -> Mapping[str, Any] | None:
    """The ranker's reasons for every condition (``ModelRanker.reasons``), or None: a ranker
    without reasons, a degraded one, or one that fails on this case. Never raises: the
    differential is shown either way."""
    reasons_of = getattr(ranker, "reasons", None)
    if reasons_of is None or getattr(ranker, "degraded", False):
        return None
    try:
        return reasons_of(case)
    except Exception:  # noqa: BLE001 - the reasons are an extra, not the result
        return None


def candidate_rows(
    result: DiagnosisResult,
    case: PatientCase | None = None,
    reasons: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One row per condition, in the pipeline's order (red flags first, then the fused score).

    Args:
        result: The pipeline's result.
        case: The case the pipeline ranked, after ``expand_case``. Its derived questions are then
            shown with the recorded finding that answers them.
        reasons: ``ModelRanker.reasons(case)``, {condition: Reasons}; left out when the model
            failed for this case, since its reasons would explain a ranking it did not make.
    """
    model_failed = "ml" in result.degraded_components
    graph_failed = "graph_backend" in result.degraded_components and all(
        c.kg_score == 0.0 for c in result.candidates
    )
    trained = {c.condition_id for c in result.candidates if BY_ID[c.condition_id].in_training_data}
    model_ranks = (
        {}
        if model_failed
        else ranks_of(
            {c.condition_id: c.ml_score for c in result.candidates if c.condition_id in trained}
        )
    )
    graph_ranks = (
        {} if graph_failed else ranks_of({c.condition_id: c.kg_score for c in result.candidates})
    )
    answers = recorded_answers(case) if case else {}
    reasons = {} if model_failed or reasons is None else reasons

    def label(assessment: Any) -> str:
        answer = answers.get(assessment.finding_id)
        return f"{assessment.label} ← {answer}" if answer else assessment.label

    rows = []
    for rank, candidate in enumerate(result.candidates, start=1):
        cid = candidate.condition_id
        rows.append(
            {
                "rank": rank,
                "condition_id": cid,
                "condition": candidate.label,
                "must_not_miss": candidate.is_must_not_miss,
                "red_flag": candidate.red_flag,
                "red_flag_reason": candidate.red_flag_reason,
                "model_rank": model_ranks.get(cid),
                "model_note": (
                    "not in the training data: the graph and the rules only"
                    if cid not in trained
                    else "the model is unavailable" if model_failed else None
                ),
                "graph_rank": graph_ranks.get(cid),
                "supporting": [label(a) for a in candidate.supporting()],
                "against": [label(a) for a in candidate.contradicting()],
                "missing": [a.label for a in candidate.missing()],
                "paths": [" → ".join(p.path) for p in candidate.paths],
                "model_for": [r.text() for r in getattr(reasons.get(cid), "supporting", ())],
                "model_against": [r.text() for r in getattr(reasons.get(cid), "against", ())],
            }
        )
    return rows


def example_cases() -> dict[str, dict[str, Any]]:
    """The golden cases as form values: {id: {description, age, sex, present, denied}}."""
    data = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    examples = {}
    for golden in data["cases"]:
        case = golden["case"]
        entries = [*case.get("findings", []), *case.get("risk_factors", [])]
        known = [e for e in entries if e["concept_id"] in BY_CONCEPT]
        examples[golden["id"]] = {
            "description": " ".join(golden["description"].split()),
            "age": int(case["age"]),
            "sex": case["sex"],
            "present": [
                e["concept_id"] for e in known if e.get("assertion", "present") == "present"
            ],
            "denied": [e["concept_id"] for e in known if e.get("assertion") == "absent"],
        }
    return examples
