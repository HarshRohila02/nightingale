"""Contract tests — the schemas every module depends on.

If these break, every workstream breaks. Treat a failure here as a blocking bug.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.conditions import (
    BY_ID,
    CONDITIONS,
    DDXPLUS_LABELS,
    from_ddxplus_label,
    must_not_miss_ids,
    trainable_ids,
)
from src.contracts import (
    DISCLAIMER,
    Assertion,
    Candidate,
    DiagnosisResult,
    EvidenceRole,
    Finding,
    FindingAssessment,
    PatientCase,
    Sex,
)


class TestPatientCase:
    def test_minimal_case_is_valid(self):
        case = PatientCase(case_id="T-1", age=50, sex=Sex.MALE)
        assert case.findings == []
        assert case.free_text is None

    @pytest.mark.parametrize("age", [-1, 121])
    def test_age_out_of_range_is_rejected(self, age):
        with pytest.raises(ValidationError):
            PatientCase(case_id="T-1", age=age, sex=Sex.MALE)

    def test_present_and_absent_are_separated(self):
        """The ABSENT/UNKNOWN distinction is clinically load-bearing (FR-1.3)."""
        case = PatientCase(
            case_id="T-1",
            age=50,
            sex=Sex.MALE,
            findings=[
                Finding(concept_id="SYM:chest_pain", label="Chest pain"),
                Finding(
                    concept_id="SYM:leg_swelling_unilateral",
                    label="Leg swelling",
                    assertion=Assertion.ABSENT,
                ),
                Finding(
                    concept_id="SYM:diaphoresis",
                    label="Diaphoresis",
                    assertion=Assertion.UNKNOWN,
                ),
            ],
        )
        assert case.present_concept_ids() == {"SYM:chest_pain"}
        assert case.absent_concept_ids() == {"SYM:leg_swelling_unilateral"}
        # UNKNOWN appears in neither set — it is not evidence either way.
        assert "SYM:diaphoresis" not in case.present_concept_ids()
        assert "SYM:diaphoresis" not in case.absent_concept_ids()

    def test_risk_factors_count_as_findings_for_scoring(self):
        case = PatientCase(
            case_id="T-1",
            age=50,
            sex=Sex.MALE,
            risk_factors=[Finding(concept_id="RF:smoking", label="Smoking")],
        )
        assert "RF:smoking" in case.present_concept_ids()


class TestCandidate:
    def test_role_filters(self):
        candidate = Candidate(
            condition_id="COND:gerd",
            label="GERD",
            assessments=[
                FindingAssessment(finding_id="a", label="A", role=EvidenceRole.SUPPORTING),
                FindingAssessment(finding_id="b", label="B", role=EvidenceRole.CONTRADICTING),
                FindingAssessment(finding_id="c", label="C", role=EvidenceRole.MISSING),
                FindingAssessment(finding_id="d", label="D", role=EvidenceRole.SUPPORTING),
            ],
        )
        assert len(candidate.supporting()) == 2
        assert len(candidate.contradicting()) == 1
        assert len(candidate.missing()) == 1

    def test_calibrated_probability_defaults_to_none(self):
        """A raw score must never be presented as a probability."""
        candidate = Candidate(condition_id="COND:gerd", label="GERD", ml_score=0.87)
        assert candidate.calibrated_probability is None


class TestDiagnosisResult:
    def test_disclaimer_is_populated_by_default(self):
        """FR-6.4 — the disclaimer must be present on every result."""
        result = DiagnosisResult(case_id="T-1", candidates=[])
        assert result.disclaimer == DISCLAIMER
        assert "not a diagnosis" in result.disclaimer.lower()

    def test_degraded_components_defaults_empty(self):
        result = DiagnosisResult(case_id="T-1", candidates=[])
        assert result.degraded_components == []


class TestConditionRegistry:
    def test_thirteen_trainable_plus_one_kg_only(self):
        assert len(trainable_ids()) == 13
        assert len(CONDITIONS) == 14

    def test_aortic_dissection_is_kg_only(self):
        """The condition that justifies the knowledge graph."""
        dissection = BY_ID["COND:aortic_dissection"]
        assert dissection.in_training_data is False
        assert dissection.ddxplus_label is None
        assert dissection.is_must_not_miss is True
        assert dissection.red_flag_hint

    def test_ids_are_unique(self):
        assert len({c.id for c in CONDITIONS}) == len(CONDITIONS)

    def test_every_must_not_miss_has_a_red_flag_hint(self):
        for condition in CONDITIONS:
            if condition.is_must_not_miss:
                assert condition.red_flag_hint, f"{condition.id} lacks a red_flag_hint"

    def test_ddxplus_labels_map_back(self):
        assert from_ddxplus_label("Possible NSTEMI / STEMI") is BY_ID["COND:nstemi_stemi"]
        assert from_ddxplus_label("Boerhaave") is BY_ID["COND:boerhaave"]
        assert from_ddxplus_label("  GERD  ") is BY_ID["COND:gerd"]

    def test_out_of_scope_label_returns_none(self):
        """Pneumonia is in DDXPlus but out of scope for Nightingale."""
        assert from_ddxplus_label("Pneumonia") is None

    def test_no_trainable_condition_lacks_a_ddxplus_label(self):
        for condition in CONDITIONS:
            if condition.in_training_data:
                assert condition.ddxplus_label, f"{condition.id} claims trainable but has no label"
        assert len(DDXPLUS_LABELS) == 13

    def test_must_not_miss_set_is_expected(self):
        assert "COND:nstemi_stemi" in must_not_miss_ids()
        assert "COND:pulmonary_embolism" in must_not_miss_ids()
        assert "COND:gerd" not in must_not_miss_ids()
