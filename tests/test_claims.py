"""The claim-level measures of an explanation (src/eval/claims.py; EXP-009, EXP-010).

They count the outcome of guardrails v1's check, so the tests build small results by hand where
every sentence's fate can be worked out from the check's own rule: each word of a sentence must be a
word of one source, a stopword, a frame word, a number or part of the case id.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.contracts import (
    Candidate,
    DiagnosisResult,
    Evidence,
    EvidenceRole,
    FindingAssessment,
    PatientCase,
)
from src.eval.claims import (
    ClaimStats,
    citations_in,
    claim_stats,
    paired_rate_difference,
    pooled_rate,
    rate_interval,
)
from src.llm.baselines import given_result, reference_case

CASE = PatientCase(
    case_id="C-1",
    age=58,
    sex="M",
    findings=[
        {"concept_id": "SYM:chest_pain", "label": "Chest pain"},
        {"concept_id": "SYM:diaphoresis", "label": "Diaphoresis"},
    ],
)
PASSAGE = Evidence(
    source="PMC",
    title="Troponin kinetics",
    passage="Serial troponin separates infarction from angina.",
    citation="PMCID:PMC1234567",
)


def result_with(evidence: bool = True) -> DiagnosisResult:
    mi = Candidate(
        condition_id="COND:nstemi_stemi",
        label="Possible NSTEMI / STEMI",
        assessments=[
            FindingAssessment(
                finding_id="SYM:chest_pain", label="Chest pain", role=EvidenceRole.SUPPORTING
            )
        ],
        evidence=[PASSAGE] if evidence else [],
    )
    gerd = Candidate(condition_id="COND:gerd", label="GERD")
    return DiagnosisResult(case_id="C-1", candidates=[mi, gerd], suggested_next_tests=["ECG"])


# --------------------------------------------------------------------------------------------- #
# One explanation
# --------------------------------------------------------------------------------------------- #


def test_a_sentence_made_of_the_cases_own_words_is_supported():
    stats = claim_stats("Chest pain and diaphoresis.", result_with(), CASE)
    assert (stats.sentences, stats.claims, stats.unsupported) == (1, 1, 0)
    assert stats.supported == 1 and not stats.untraced


def test_a_sentence_with_a_word_the_system_was_not_given_is_unsupported():
    stats = claim_stats(
        "Chest pain and diaphoresis. Radiation to the jaw is typical.", result_with(), CASE
    )
    assert (stats.sentences, stats.claims, stats.unsupported) == (2, 2, 1)
    assert stats.untraced == ("Radiation to the jaw is typical.",)


def test_a_sentence_that_claims_nothing_is_not_counted():
    # Every word is a stopword, a frame word or a number: it joins facts and says none.
    stats = claim_stats("The patient is the first candidate. Chest pain.", result_with(), CASE)
    assert (stats.sentences, stats.claims, stats.unsupported) == (2, 1, 0)


def test_the_count_of_claims_agrees_with_the_guardrails_check():
    from src.reasoning.guardrails import untraced_sentences

    text = (
        "Chest pain. Troponin rise points to infarction. Take aspirin now. The patient is here. "
        "Pericarditis."
    )
    stats = claim_stats(text, result_with(), CASE)
    assert list(stats.untraced) == untraced_sentences(text, result_with(), CASE)
    assert stats.claims - stats.unsupported == stats.supported


def test_age_and_sex_are_a_source_once_the_reference_case_is_used():
    text = "A 58 years old male with chest pain."
    assert claim_stats(text, result_with(), CASE).unsupported == 1
    assert claim_stats(text, result_with(), reference_case(CASE)).unsupported == 0


def test_a_case_id_is_free():
    assert claim_stats("Case C-1.", result_with(), CASE).claims == 0


def test_treatment_language_is_found_outside_the_known_labels():
    assert claim_stats("Give aspirin 300 mg.", result_with(), CASE).treatment_language
    assert not claim_stats("Chest pain.", result_with(), CASE).treatment_language


# --------------------------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------------------------- #


def test_citations_are_found_in_either_form_and_listed_once():
    assert citations_in("See [PMCID:PMC1234567] and PMC7654321, also PMC1234567.") == [
        "PMCID:PMC1234567",
        "PMCID:PMC7654321",
    ]
    assert citations_in("No citation here.") == []


def test_a_citation_is_valid_only_if_it_is_a_passage_the_system_was_shown():
    text = "Serial troponin separates infarction [PMCID:PMC1234567]. Chest pain [PMC9999999]."
    stats = claim_stats(text, result_with(), CASE)
    assert stats.cited == ("PMCID:PMC1234567", "PMCID:PMC9999999")
    assert stats.invalid_citations == ("PMCID:PMC9999999",)
    assert stats.cited_sentences == 2
    assert stats.cited_unsupported == 1  # the second: its citation words are in no source


def test_a_system_given_no_passage_has_only_invalid_citations():
    result = given_result(CASE)
    stats = claim_stats("Chest pain [PMCID:PMC1234567].", result, CASE)
    assert stats.invalid_citations == ("PMCID:PMC1234567",)


def test_as_dict_is_json_ready():
    import json

    stats = claim_stats("Chest pain. Jaw.", result_with(), CASE)
    assert json.loads(json.dumps(stats.as_dict()))["unsupported"] == 1
    assert isinstance(stats, ClaimStats)


# --------------------------------------------------------------------------------------------- #
# The pooled rate and the paired comparison
# --------------------------------------------------------------------------------------------- #


def test_the_pooled_rate_is_unsupported_over_claims_across_cases():
    ratio = pooled_rate([1, 0, 3], [2, 0, 4])  # the middle case has no claim
    assert ratio.value == pytest.approx(4 / 6)
    assert ratio.cases == 2
    with pytest.raises(ValueError):
        pooled_rate([1], [1, 2])


def test_the_interval_of_a_rate_resamples_cases_and_is_repeatable():
    rate, low, high = rate_interval([1, 0, 1, 0, 2, 1], [3, 2, 4, 3, 5, 4])
    assert 0 <= low <= rate <= high <= 1
    assert (rate, low, high) == rate_interval([1, 0, 1, 0, 2, 1], [3, 2, 4, 3, 5, 4])


def test_a_higher_rate_in_every_case_is_a_difference_the_interval_excludes_zero_for():
    n = 40
    alone = pooled_rate([3] * n, [4] * n)
    grounded = pooled_rate([1] * n, [4] * n)
    result = paired_rate_difference(alone, grounded)
    assert result["difference"] == pytest.approx(0.5)
    assert result["excludes_zero"] and result["higher"] == "a" and result["cases"] == n
    reverse = paired_rate_difference(grounded, alone)
    assert reverse["higher"] == "b" and reverse["difference"] == pytest.approx(-0.5)


def test_equal_systems_do_not_differ():
    rng = np.random.default_rng(0)
    unsupported = rng.integers(0, 3, size=50)
    ratio = pooled_rate(unsupported, [4] * 50)
    result = paired_rate_difference(ratio, ratio)
    assert result["difference"] == 0 and not result["excludes_zero"] and result["higher"] is None


def test_a_noisy_small_difference_is_not_called_a_difference():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 3, size=30)
    b = rng.integers(0, 3, size=30)
    result = paired_rate_difference(pooled_rate(a, [3] * 30), pooled_rate(b, [3] * 30))
    assert result["ci_low"] < 0 < result["ci_high"] or result["excludes_zero"]
    assert math.isfinite(result["ci_low"]) and math.isfinite(result["ci_high"])


def test_a_paired_comparison_refuses_unequal_or_empty_cases():
    with pytest.raises(ValueError):
        paired_rate_difference(pooled_rate([1], [2]), pooled_rate([1, 1], [2, 2]))
    with pytest.raises(ValueError):
        paired_rate_difference(pooled_rate([0, 0], [0, 0]), pooled_rate([1, 1], [2, 2]))
