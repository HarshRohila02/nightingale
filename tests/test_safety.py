"""Tests for the safety layer v1 (task 2d, src/reasoning/safety.py): FR-6.2, 6.4 and 6.5."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from src.contracts import (
    DISCLAIMER,
    Candidate,
    DiagnosisResult,
    Explanation,
    Finding,
    PatientCase,
)
from src.pipeline import DiagnosisPipeline
from src.reasoning.red_flags import RULES
from src.reasoning.safety import recommends_treatment, safety_check
from src.stubs import ConstantRanker, EmptyRetriever, InMemoryGraphStore, TemplateExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = yaml.safe_load(
    (REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml").read_text(encoding="utf-8")
)["cases"]


def candidate(cid: str, label: str, *, flagged: bool = False, score: float = 0.0) -> Candidate:
    return Candidate(
        condition_id=cid,
        label=label,
        fused_score=score,
        red_flag=flagged,
        red_flag_reason="a reason" if flagged else None,
    )


def result(**kwargs) -> DiagnosisResult:
    defaults = {
        "case_id": "T-S",
        "candidates": [candidate("COND:gerd", "GERD", score=0.9)],
        "explanation": Explanation(text="Top candidate: GERD.", grounded=True),
    }
    return DiagnosisResult(**(defaults | kwargs))


# --------------------------------------------------------------------------- #
# FR-6.5: the treatment lexicon
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Give aspirin 300 mg now.",
        "Start heparin.",
        "Consider thrombolysis.",
        "The patient should receive a beta-blocker.",
        "Insert a chest drain.",
        "Prescribe a proton pump inhibitor.",
        "Administer oxygen at 2 L via nasal cannula and morphine.",
        "Treat with colchicine.",
        "A dose of 40 mg.",
    ],
)
def test_treatment_language_is_recognised(text):
    assert recommends_treatment(text)


@pytest.mark.parametrize(
    "text",
    [
        "Obtain ECG and troponin.",
        "Aortic dissection must be excluded.",
        "Consider PE.",
        "Sudden-onset pleuritic chest pain with breathlessness — exclude pneumothorax.",
        "Supporting findings: Recent surgery, Previous deep vein thrombosis.",
        "Top candidate for case GC-001: Possible NSTEMI / STEMI (score 0.51).",
    ],
)
def test_investigations_and_findings_are_not_treatment(text):
    assert not recommends_treatment(text)


def test_a_question_about_medication_is_not_advice():
    """DDXPlus asks "do you take medications to treat high cholesterol?": a known label."""
    label = "Do you have high cholesterol or do you take medications to treat high cholesterol?"
    text = f"Not recorded, and would help confirm or exclude: {label}."
    assert recommends_treatment("Take statins.")
    assert not recommends_treatment(text, known_labels=[label])


def test_no_red_flag_reason_recommends_a_treatment():
    for rule in RULES:
        assert not recommends_treatment(rule.reason), rule.reason


# --------------------------------------------------------------------------- #
# The repairs
# --------------------------------------------------------------------------- #


def test_a_clean_result_is_returned_unchanged():
    clean = result()
    assert safety_check(clean) is clean


def test_the_disclaimer_is_restored(caplog):
    tampered = result(disclaimer="Diagnosis: GERD.")
    with caplog.at_level(logging.WARNING):
        assert safety_check(tampered).disclaimer == DISCLAIMER
    assert "FR-6.4" in caplog.text


def test_a_flagged_candidate_is_moved_above_every_unflagged_one():
    out_of_order = result(
        candidates=[
            candidate("COND:gerd", "GERD", score=0.9),
            candidate("COND:pericarditis", "Pericarditis", score=0.5),
            candidate("COND:pulmonary_embolism", "Pulmonary embolism", flagged=True, score=0.1),
        ],
        red_flags=["Pulmonary embolism: a reason"],
    )
    repaired = safety_check(out_of_order)
    assert [c.condition_id for c in repaired.candidates] == [
        "COND:pulmonary_embolism",
        "COND:gerd",
        "COND:pericarditis",
    ], "flags first, and the unflagged keep their order"


def test_a_flagged_candidate_is_always_named():
    unnamed = result(candidates=[candidate("COND:boerhaave", "Boerhaave syndrome", flagged=True)])
    assert safety_check(unnamed).red_flags == ["Boerhaave syndrome: a reason"]


def test_treatment_advice_is_removed_from_the_explanation_and_recorded():
    advised = result(
        explanation=Explanation(
            text="Top candidate: NSTEMI. Give aspirin 300 mg. Obtain ECG and troponin.",
            grounded=True,
        )
    )
    explanation = safety_check(advised).explanation
    assert explanation.text == "Top candidate: NSTEMI. Obtain ECG and troponin."
    assert not explanation.grounded
    assert explanation.unsupported_claims == [
        "treatment advice removed (FR-6.5): Give aspirin 300 mg."
    ]


def test_a_treatment_is_not_a_suggested_test():
    suggested = result(suggested_next_tests=["ECG", "Troponin", "Start heparin"])
    assert safety_check(suggested).suggested_next_tests == ["ECG", "Troponin"]


def test_a_finding_label_in_the_case_is_masked():
    case = PatientCase(
        case_id="T-S",
        age=60,
        sex="F",
        risk_factors=[Finding(concept_id="RF:x", label="Takes anticoagulants")],
    )
    mentioned = result(
        explanation=Explanation(text="Supporting findings: Takes anticoagulants.", grounded=True)
    )
    assert safety_check(mentioned, case) is mentioned


@pytest.mark.golden
def test_the_golden_cases_need_no_repair(caplog):
    """The pipeline calls safety_check last; the template never breaks a rule."""
    pipeline = DiagnosisPipeline(
        ConstantRanker(), InMemoryGraphStore(), EmptyRetriever(), TemplateExplainer()
    )
    with caplog.at_level(logging.WARNING, logger="src.reasoning.safety"):
        for golden in GOLDEN:
            out = pipeline.run(PatientCase(**golden["case"]))
            assert out.disclaimer == DISCLAIMER
            assert out.explanation is not None and out.explanation.grounded
    assert not caplog.records, caplog.text
