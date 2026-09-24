"""The last check on every result before it leaves the pipeline (safety layer v1, task 2d).

``DiagnosisPipeline.run`` passes each result through :func:`safety_check`, which enforces three
of docs/01's safety requirements and repairs a result that breaks them rather than failing:

* **FR-6.4, the disclaimer.** It is always :data:`~src.contracts.DISCLAIMER`, word for word.
* **FR-6.2, red flags are prominent.** Every flagged candidate comes before every unflagged one,
  whatever it scored, and every flagged candidate is named in ``result.red_flags``.
* **FR-6.5, never a treatment or drug recommendation.** A sentence of the explanation, a red-flag
  reason or a suggested test that recommends a treatment, names a drug or gives a dose is removed.
  A removed sentence is listed in ``explanation.unsupported_claims`` and the explanation is marked
  not grounded, so the result says what was taken out. Investigations ("obtain ECG and
  troponin") are not treatments and are kept.

The treatment check reads the text the system *generated*: the labels of the case's own findings
and of the conditions are masked first, because DDXPlus's questions include "do you take
medications to treat high blood pressure?", which is a question, not advice.

v1 is deterministic and lexical. It catches the phrasing a template or an LLM would use, not every
possible paraphrase; the claim-support check against retrieved evidence is task 3c's. Repairs are
logged (the audit log itself, FR-6.6, is not built yet).

Reference: docs/01-srs.md FR-6, docs/04-safety-ethics.md §2-§3, docs/02-architecture.md §3.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from src.contracts import DISCLAIMER, DiagnosisResult, PatientCase

logger = logging.getLogger(__name__)

__all__ = ["TREATMENT", "recommends_treatment", "safety_check"]

_DRUGS = (
    r"aspirin|clopidogrel|ticagrelor|prasugrel|heparin|enoxaparin|fondaparinux|warfarin|"
    r"apixaban|rivaroxaban|edoxaban|dabigatran|anticoagula\w*|antiplatelet\w*|thromboly\w*|"
    r"fibrinoly\w*|alteplase|tenecteplase|nitroglycerin|glyceryl trinitrate|GTN|nitrates?|"
    r"morphine|opioids?|beta[- ]?blockers?|metoprolol|atenolol|bisoprolol|statins?|"
    r"atorvastatin|diuretics?|furosemide|frusemide|ACE inhibitors?|antibiotics?|NSAIDs?|"
    r"ibuprofen|colchicine|proton[- ]pump inhibitors?|PPIs?|omeprazole|pantoprazole|"
    r"antacids?|benzodiazepines?|lorazepam|diazepam|amiodarone|adenosine|digoxin|diltiazem|"
    r"verapamil|steroids?|corticosteroids?|prednisolone"
)
_PROCEDURES = (
    r"thrombolysis|PCI|angioplasty|stent(?:ing)?|CABG|bypass graft|chest (?:drain|tube)|"
    r"needle decompression|cardioversion|pericardiocentesis|surgical repair|endoscopic repair"
)
_VERBS = (
    r"administer\w*|prescrib\w*|commence\w*|initiate (?:treatment|therapy)|treat (?:it |this )?with|"
    r"start (?:on|treatment|therapy)|give (?:the patient|him|her|them)|should (?:receive|be given)"
)
_DOSE = r"\d+(?:\.\d+)?\s?(?:mg|mcg|µg|micrograms?|g|ml|mL|units?|IU)\b"

TREATMENT = re.compile(rf"\b(?:{_DRUGS}|{_PROCEDURES}|{_VERBS})\b|{_DOSE}", flags=re.IGNORECASE)
"""Treatment language: drugs and drug classes, treatments, prescribing verbs and doses."""

_SENTENCE = re.compile(r"(?<=[.!])\s+")


def recommends_treatment(text: str, known_labels: Iterable[str] = ()) -> bool:
    """Whether ``text`` contains treatment language outside the known labels."""
    return bool(TREATMENT.search(_mask(text, known_labels)))


def _mask(text: str, labels: Iterable[str]) -> str:
    for label in sorted({lab for lab in labels if lab}, key=len, reverse=True):
        text = text.replace(label, " ")
    return text


def _split(text: str, labels: list[str]) -> tuple[str, list[str]]:
    """The text without its treatment sentences, and those sentences."""
    kept, removed = [], []
    for sentence in _SENTENCE.split(text):
        (removed if recommends_treatment(sentence, labels) else kept).append(sentence)
    return " ".join(kept), removed


def _known_labels(result: DiagnosisResult, case: PatientCase | None) -> list[str]:
    labels = [c.label for c in result.candidates]
    labels += [a.label for c in result.candidates for a in c.assessments]
    if case is not None:
        labels += [f.label for f in [*case.findings, *case.risk_factors]]
    return labels


def safety_check(result: DiagnosisResult, case: PatientCase | None = None) -> DiagnosisResult:
    """``result`` with FR-6.2, FR-6.4 and FR-6.5 enforced; see the module docstring.

    Args:
        result: What the pipeline is about to return.
        case: The case it answers, whose finding labels are masked before the treatment check.

    Returns:
        A copy of ``result``, repaired where it broke a rule. Each repair is logged.
    """
    update: dict = {}
    case_id = result.case_id

    if result.disclaimer != DISCLAIMER:
        logger.warning("case %s: the disclaimer was changed; restored (FR-6.4)", case_id)
        update["disclaimer"] = DISCLAIMER

    candidates = sorted(result.candidates, key=lambda c: not c.red_flag)  # stable
    if [c.condition_id for c in candidates] != [c.condition_id for c in result.candidates]:
        logger.warning(
            "case %s: a flagged candidate ranked below an unflagged one (FR-6.2)", case_id
        )
        update["candidates"] = candidates

    labels = _known_labels(result, case)
    red_flags = []
    for flag in result.red_flags:
        kept, removed = _split(flag, labels)
        if removed:
            logger.warning("case %s: treatment advice removed from a red flag (FR-6.5)", case_id)
        red_flags.append(kept)
    named = " ".join(red_flags)
    for candidate in candidates:
        if candidate.red_flag and candidate.label not in named:
            logger.warning(
                "case %s: %s was flagged but not named (FR-6.2)", case_id, candidate.label
            )
            red_flags.append(f"{candidate.label}: {candidate.red_flag_reason or 'red flag'}")
    if red_flags != result.red_flags:
        update["red_flags"] = red_flags

    tests = [t for t in result.suggested_next_tests if not recommends_treatment(t, labels)]
    if tests != result.suggested_next_tests:
        logger.warning("case %s: a suggested test was a treatment; removed (FR-6.5)", case_id)
        update["suggested_next_tests"] = tests

    explanation = result.explanation
    if explanation is not None:
        text, removed = _split(explanation.text, labels)
        if removed:
            logger.warning(
                "case %s: treatment advice removed from the explanation (FR-6.5)", case_id
            )
            update["explanation"] = explanation.model_copy(
                update={
                    "text": text,
                    "grounded": False,
                    "unsupported_claims": [
                        *explanation.unsupported_claims,
                        *(f"treatment advice removed (FR-6.5): {s}" for s in removed),
                    ],
                }
            )

    return result.model_copy(update=update) if update else result
