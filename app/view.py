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

:func:`next_steps` is the "What to ask next" block (3d): the cited tests to consider, as the
pipeline put them in the result, and the findings worth asking about or checking next, ranked by
the knowledge graph (``src/reasoning/next_questions.py``), each with the conditions it bears on.

:func:`load_patient_cards`, :func:`patient_label` and :func:`patient_card_markdown` are the
"Patient history (Synthea, display only)" panel (1d): a synthetic patient's history, read from the
cards ``scripts/build_patient_display.py`` wrote, shown beside the case. **Display only**
(PROGRESS §6, 2026-10-20): nothing here builds a case, reads a finding or touches the ranking, and
this module imports nothing about the patient history's graph.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.conditions import BY_ID
from src.contracts import Assertion, DiagnosisResult, Finding, PatientCase
from src.medical_kg.crosswalk import BY_CONCEPT, recorded_answers
from src.reasoning.guardrails import citation_link

__all__ = [
    "MEDICATION_HEADING",
    "PATIENT_NOTE",
    "build_case",
    "candidate_rows",
    "concept_options",
    "example_cases",
    "load_patient_cards",
    "model_reasons",
    "next_steps",
    "patient_card_markdown",
    "patient_label",
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


def next_steps(result: DiagnosisResult, case: PatientCase | None, graph: Any) -> dict[str, Any]:
    """The "What to ask next" block (3d), as plain data.

    Args:
        result: The pipeline's result; its ``suggested_next_tests`` are shown as they are, after
            the safety layer (src/reasoning/safety.py), each ending with its source's tag.
        case: The case the pipeline ranked, as entered or after ``expand_case``.
        graph: The graph store that ranked it. No questions when there is no case or the graph
            cannot answer (``next_questions`` then returns none, rather than a guess).

    Returns:
        ``tests``: the lines to show. ``questions``: one dict per question, best first, with
        ``verb`` ("Ask about", or "Check" for an examination finding), ``label``, ``if_present``
        and ``if_denied`` (the conditions a yes or a no would favour, by label). ``entropy_bits``:
        the graph's uncertainty now, or None when it could not answer.
    """
    from src.reasoning.next_questions import next_questions

    ranking = next_questions(case, graph) if case is not None else None
    questions = [
        {
            "concept_id": q.concept_id,
            "label": q.label,
            "verb": "Check" if q.kind == "examination" else "Ask about",
            "if_present": [s.label for s in q.if_present],
            "if_denied": [s.label for s in q.if_denied],
        }
        for q in (ranking.questions if ranking else [])
    ]
    return {
        "tests": list(result.suggested_next_tests),
        "questions": questions,
        "entropy_bits": ranking.entropy_bits if ranking else None,
    }


# --------------------------------------------------------------------------- #
# Evidence (task 3a, FR-9.4): kept apart from the rest so it merges cleanly
# --------------------------------------------------------------------------- #

EVIDENCE_OFF = "No evidence bundle is loaded, so no literature is shown (docs/11 §4.6)."
"""The one quiet line an expander shows when retrieval is degraded (``rag``)."""

_PMC_ARTICLE = "https://pmc.ncbi.nlm.nih.gov/articles/"
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()<>#+\-|!$~])")


def _escape_markdown(text: str) -> str:
    """Literature text shown as itself: Streamlit's Markdown would otherwise turn a stray
    bracket into a link, an asterisk into emphasis and a dollar sign into LaTeX."""
    return _MARKDOWN.sub(r"\\\1", " ".join(text.split()))


def _pmc_link(citation: str) -> str:
    """The article's PMC page, from the one citation formatter (src/reasoning/guardrails.py, 3c),
    so a link never leads anywhere but PMC. A citation that is not a PMC identifier links to PMC's
    article index rather than to a page built from it. The evidence's own ``url`` is not used: the
    citation decides, as it does in the explanation's citations."""
    return citation_link(citation) or _PMC_ARTICLE


def evidence_note(result: DiagnosisResult) -> str | None:
    """:data:`EVIDENCE_OFF` when the result was produced without evidence retrieval, else None."""
    return EVIDENCE_OFF if "rag" in result.degraded_components else None


def evidence_rows(
    result: DiagnosisResult, retriever: Any = None
) -> dict[str, list[dict[str, Any]]]:
    """{condition id: one row per passage}, for the candidates the pipeline gave evidence.

    A row holds the ``snippet`` (the excerpt, verbatim), the ``title``, the ``citation``
    (``PMCID:…``), the ``url`` of its PMC page, and, from the retriever's bundle
    (``source_of``), the ``licence``, its ``licence_url`` and an ``attribution`` (first author,
    journal, year): Creative Commons licences ask for attribution, and ``Evidence`` has no field
    for it (src/contracts.py).
    """
    source_of = getattr(retriever, "source_of", None)
    out: dict[str, list[dict[str, Any]]] = {}
    for candidate in result.candidates:
        rows = []
        for evidence in candidate.evidence:
            source = (source_of(evidence.citation) if source_of else None) or {}
            year = source.get("year")
            rows.append(
                {
                    "snippet": evidence.passage,
                    "title": evidence.title,
                    "citation": evidence.citation,
                    "url": _pmc_link(evidence.citation),
                    "licence": source.get("licence"),
                    "licence_url": source.get("licence_url"),
                    "attribution": ", ".join(
                        str(part)
                        for part in (source.get("authors"), source.get("journal"), year)
                        if part
                    ),
                }
            )
        if rows:
            out[candidate.condition_id] = rows
    return out


def evidence_markdown(row: Mapping[str, Any]) -> str:
    """One passage as Markdown: the excerpt quoted, then its citation as a link, its title, its
    attribution and its licence."""
    licence = row.get("licence")
    if licence and str(row.get("licence_url") or "").startswith("https://creativecommons.org/"):
        licence = f"[{_escape_markdown(licence)}]({row['licence_url']})"
    elif licence:
        licence = _escape_markdown(licence)
    parts = [
        f"[{_escape_markdown(row['citation'])}]({row['url']})",
        f"*{_escape_markdown(row['title'])}*",
        _escape_markdown(row["attribution"]) if row.get("attribution") else None,
        licence,
    ]
    return f"> {_escape_markdown(row['snippet'])}\n\n" + " · ".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Credits and licences (docs/03 §6, §7: attribution on every public surface)
# --------------------------------------------------------------------------- #

CREDITS: tuple[str, ...] = (
    "**DDXPlus** (Tchango et al., NeurIPS 2022; CC-BY-4.0): the synthetic patients the rankers "
    "are trained and tested on.",
    "**BODHI-S** by **Eka Care** (CC-BY-NC-4.0): enriches four conditions of the knowledge graph. "
    "Used for non-commercial academic research only.",
    "**Synthea** (MITRE; Apache-2.0): synthetic patient histories, shown beside a case and never "
    "ranked on.",
    "**PubMed Central Open Access** (NCBI): the evidence passages, each shown with its own "
    "Creative Commons licence; some of those are non-commercial.",
    "**S-PubMedBert-MS-MARCO** (`pritamdeka` on Hugging Face; CC-BY-NC-2.0): the model that "
    "embedded the passages, run once in the cloud (not in this app).",
)
NON_COMMERCIAL = (
    "This is a university research prototype, for non-commercial academic use only. It is "
    "decision support, not a diagnosis."
)


def credits_markdown() -> str:
    """The credits block the dashboard shows: every source it draws on, its licence, and that the
    project is non-commercial (docs/03 §7)."""
    bullets = "\n".join(f"- {line}" for line in CREDITS)
    return f"{bullets}\n\n{NON_COMMERCIAL}"


# --------------------------------------------------------------------------- #
# Patient history (task 1d): a Synthea patient shown beside the case, display only
# --------------------------------------------------------------------------- #

PATIENT_NOTE = (
    "A synthetic Synthea history, shown for context. It does not enter the ranking: the "
    "findings you tick do. Medications are history, never advice."
)
"""Under the panel's card. The decision of 2026-10-20 (PROGRESS §6): a history is shown beside a
case and never feeds the ranking, the case's findings or its risk factors."""

MEDICATION_HEADING = "Medication history (not advice)"

_CARD_ROWS = 25
"""The most rows of any one list the panel draws, whatever the file holds."""
_CARD_TEXT = 160
_SEX_WORD = {"M": "male", "F": "female", "O": "other sex"}


def load_patient_cards(path: str | Path) -> list[dict[str, Any]] | None:
    """The cards ``scripts/build_patient_display.py`` wrote, or None when the file is absent or
    unreadable (not JSON, not the builder's layout). Never raises: the panel is an extra, and the
    page is shown either way. A valid file holding no patient gives an empty list."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        patients = raw.get("patients") if isinstance(raw, dict) else None
        if not isinstance(patients, list):
            return None
        return [card for card in patients if isinstance(card, dict) and card.get("id")]
    except Exception:  # noqa: BLE001 - absent, unreadable or malformed all mean "no cards"
        return None


def _text(value: Any) -> str:
    """Anything from the file as Markdown-safe text, cut to a line's length."""
    return _escape_markdown(str(value)[:_CARD_TEXT])


def _rows(card: Mapping[str, Any], key: str) -> list[Any]:
    rows = card.get(key)
    return list(rows)[:_CARD_ROWS] if isinstance(rows, list) else []


def _more(card: Mapping[str, Any], key: str, shown: int) -> list[str]:
    """The line "…and N more" when the card's full count for ``key`` exceeds what it lists."""
    total = card.get(f"{key}_total")
    if isinstance(total, int) and not isinstance(total, bool) and total > shown:
        return [f"- …and {total - shown} more"]
    return []


def _count(value: Any) -> str:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else "?"


def patient_label(card: Mapping[str, Any]) -> str:
    """One line to choose a card by: the short id, the age and the sex. Plain text, built from
    the id's own characters, the age as a number and a word from a fixed list."""
    short = "".join(ch for ch in str(card.get("id", "")) if ch.isalnum() or ch == "-")[:12]
    age = card.get("age")
    years = f"{age} y" if isinstance(age, int) and not isinstance(age, bool) else "age unknown"
    return f"{short or 'unknown'} · {years} · {_SEX_WORD.get(str(card.get('sex')), 'sex unknown')}"


def patient_card_markdown(card: Mapping[str, Any]) -> str:
    """One card as Markdown: who, the active conditions, the risk factors the history supports,
    the medication history, the counts on record and the conditions the table does not map. Every
    string from the file is escaped, so a description cannot become a link or a formula."""
    as_of = card.get("as_of")
    head = f"**{_text(patient_label(card))}**" + (
        f" · history as of {_text(as_of)}" if as_of else ""
    )
    parts = [head]

    conditions = [
        row
        for row in _rows(card, "conditions")
        if isinstance(row, Mapping) and "description" in row
    ]

    def line_of(row: Mapping[str, Any]) -> str:
        return f"- {_text(row['description'])}" + (
            f" · since {_text(row['since'])}" if row.get("since") else ""
        )

    disorders = [line_of(row) for row in conditions if row.get("finding") is not True]
    findings = [line_of(row) for row in conditions if row.get("finding") is True]
    more = _more(card, "conditions", len(conditions))
    # What a cut drops is the end of the list, the findings, so the "more" goes under them.
    (findings if findings else disorders).extend(more)
    parts.append("**Active conditions**\n" + ("\n".join(disorders) or "_none recorded_"))
    if findings:
        parts.append(
            "**Other findings on record** (social history and findings Synthea lists as active)\n"
            + "\n".join(findings)
        )

    risks = [
        row for row in _rows(card, "risk_factors") if isinstance(row, Mapping) and "label" in row
    ]
    lines = [
        f"- {_text(row['label'])}: {'present' if row.get('present') else 'absent'}"
        + (f" · since {_text(row['since'])}" if row.get("since") else "")
        for row in risks
    ]
    parts.append(
        "**Risk factors the history supports**\n" + ("\n".join(lines) or "_none recorded_")
    )

    medications = [m for m in _rows(card, "medications") if isinstance(m, str)]
    lines = [f"- {_text(m)}" for m in medications]
    lines += _more(card, "medications", len(medications))
    parts.append(f"**{MEDICATION_HEADING}**\n" + ("\n".join(lines) or "_none active_"))

    parts.append(
        f"**On record by that day**: {_count(card.get('procedures'))} procedures, "
        f"{_count(card.get('observations'))} observations."
    )

    unmapped, total = card.get("unmapped_total"), card.get("conditions_total")
    if isinstance(unmapped, int) and not isinstance(unmapped, bool) and unmapped > 0:
        # A count, not a list: the entries are the ones above, and only six concepts map.
        of = (
            f" of {_count(total)}" if isinstance(total, int) and not isinstance(total, bool) else ""
        )
        parts.append(f"No risk-factor mapping for {unmapped}{of} active entries.")
    return "\n\n".join(parts)
