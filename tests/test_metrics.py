"""Tests for the evaluation metrics (task 1e, src/eval/metrics.py).

Four hand-built cases, with every expected value worked out by hand in the comments. MI and PE
are must-not-miss; GERD and PSVT are not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval.metrics import (
    CaseOutcome,
    MetricResult,
    Ratio,
    bootstrap_ratio,
    brier_score,
    dangerous_false_negative_rate,
    evaluate,
    expected_calibration_error,
    f1_by_condition,
    mcnemar,
    must_not_miss_recall_at_3,
    outcome_from_record,
    precision_at_3,
    ranking_from_scores,
    recall_at_5,
    reciprocal_rank,
    red_flag_precision,
    red_flag_sensitivity,
    reliability_table,
    top_k,
)

MI = "COND:nstemi_stemi"
UA = "COND:unstable_angina"
PE = "COND:pulmonary_embolism"
PNX = "COND:spontaneous_pneumothorax"
GERD = "COND:gerd"
PSVT = "COND:psvt"
AF = "COND:atrial_fibrillation"
PANIC = "COND:panic_attack"

CASES = [
    # MI first. D_in {MI, UA, PE} of |D| = 5. Flagged for MI.
    CaseOutcome("c1", MI, (MI, UA, GERD, PE, PSVT), frozenset({MI, UA, PE}), 5, frozenset({MI})),
    # GERD second. D_in {GERD, PE} of |D| = 4. Flagged for PE.
    CaseOutcome("c2", GERD, (PE, GERD, MI, UA, PSVT), frozenset({GERD, PE}), 4, frozenset({PE})),
    # PE sixth: a dangerous false negative. D_in {PE} of |D| = 3. No flag.
    CaseOutcome("c3", PE, (GERD, PSVT, MI, UA, PNX, PE), frozenset({PE}), 3),
    # PSVT first. D_in {PSVT, AF, PANIC} of |D| = 6. No flag.
    CaseOutcome("c4", PSVT, (PSVT, AF, GERD, PANIC, MI), frozenset({PSVT, AF, PANIC}), 6),
]
CRITICAL = {MI, PE}


# --------------------------------------------------------------------------- #
# §3.1 ranking
# --------------------------------------------------------------------------- #


def test_top_k_accuracy():
    assert top_k(CASES, 1).value == 2 / 4  # c1, c4
    assert top_k(CASES, 3).value == 3 / 4  # PE is sixth in c3
    assert top_k(CASES, 5).value == 3 / 4


def test_mean_reciprocal_rank():
    assert reciprocal_rank(CASES).value == pytest.approx((1 + 1 / 2 + 1 / 6 + 1) / 4)


def test_a_condition_the_system_did_not_rank_scores_zero():
    unranked = CaseOutcome("x", PE, (MI, GERD))
    assert reciprocal_rank([unranked]).value == 0.0
    assert unranked.rank() is None


def test_precision_at_3_uses_the_in_scope_differential():
    # c1 {MI, UA}, c2 {GERD, PE}, c3 none, c4 {PSVT, AF}
    assert precision_at_3(CASES).value == pytest.approx((2 / 3 + 2 / 3 + 0 + 2 / 3) / 4)


def test_recall_at_5_on_d_in_and_on_the_full_d():
    """Amendment 1: the headline uses D_in; the full-D figure is reported beside it."""
    assert recall_at_5(CASES).value == pytest.approx((3 / 3 + 2 / 2 + 0 + 3 / 3) / 4)
    assert recall_at_5(CASES, full_differential=True).value == pytest.approx(
        (3 / 5 + 2 / 4 + 0 / 3 + 3 / 6) / 4
    )


def test_a_perfect_system_reaches_the_d_in_ceiling():
    """EXP-013: with more than five in-scope entries, even a perfect ranking tops out below 1."""
    d_in = frozenset({MI, UA, PE, PNX, GERD, PSVT})
    perfect = CaseOutcome("p", MI, (MI, UA, PE, PNX, GERD, PSVT), d_in, 12)
    assert recall_at_5([perfect]).value == pytest.approx(5 / 6)
    assert recall_at_5([perfect], full_differential=True).value == pytest.approx(5 / 12)


def test_f1_by_condition():
    # Top-1 predictions MI, PE, GERD, PSVT against truths MI, GERD, PE, PSVT.
    f1 = f1_by_condition(CASES, [MI, GERD, PE, PSVT])
    assert (f1[MI], f1[GERD], f1[PE], f1[PSVT]) == (1.0, 0.0, 0.0, 1.0)
    assert f1["macro"] == 0.5
    assert f1["micro"] == top_k(CASES, 1).value


def test_ranking_from_scores_breaks_ties_by_registry_order():
    assert ranking_from_scores({GERD: 0.5, MI: 0.5, PE: 0.9}) == (PE, MI, GERD)


# --------------------------------------------------------------------------- #
# §3.2 safety
# --------------------------------------------------------------------------- #


def test_must_not_miss_recall_counts_only_must_not_miss_cases():
    recall = must_not_miss_recall_at_3(CASES, CRITICAL)
    assert (recall.value, recall.cases) == (1 / 2, 2)  # MI found, PE sixth
    hits = recall.numerator.tolist()
    assert hits == [1.0, 0.0, 0.0, 0.0], "GERD's and PSVT's top-3 hits must not count"


def test_dangerous_false_negative_rate():
    assert dangerous_false_negative_rate(CASES, CRITICAL).value == 1 / 2  # PE outside the top 5


def test_red_flag_sensitivity_is_measured_against_the_true_condition():
    """Amendment 2: c2's PE flag does not count, because c2 is GERD."""
    assert red_flag_sensitivity(CASES, CRITICAL).value == 1 / 2
    assert red_flag_sensitivity(CASES, CRITICAL, condition=MI).value == 1.0
    assert red_flag_sensitivity(CASES, CRITICAL, condition=PE).value == 0.0


def test_red_flag_precision_is_strict_unless_told_what_is_appropriate():
    assert red_flag_precision(CASES).value == 1 / 2  # the PE flag on GERD
    assert red_flag_precision(CASES, {PE: {PE, GERD}}).value == 1.0


def test_the_default_must_not_miss_set_is_the_registry():
    assert must_not_miss_recall_at_3(CASES).cases == 2


# --------------------------------------------------------------------------- #
# §3.3 calibration
# --------------------------------------------------------------------------- #

CALIBRATED = [
    CaseOutcome("k1", MI, (MI, GERD), probabilities={MI: 0.8, GERD: 0.2}),  # right, 0.8
    CaseOutcome("k2", GERD, (PE, GERD), probabilities={PE: 0.6, GERD: 0.4}),  # wrong, 0.6
    CaseOutcome("k3", PE, (GERD, PE), probabilities={GERD: 0.3, PE: 0.2}),  # wrong, 0.3
    CaseOutcome("k4", PSVT, (PSVT,), probabilities={PSVT: 0.95}),  # right, 0.95
]


def test_expected_calibration_error_over_ten_bins():
    # One case per bin: (|0.8 - 1| + |0.6 - 0| + |0.3 - 0| + |0.95 - 1|) / 4
    assert expected_calibration_error(CALIBRATED) == pytest.approx(1.15 / 4)
    assert [row["cases"] for row in reliability_table(CALIBRATED)] == [1, 1, 1, 1]


def test_brier_score():
    brier = brier_score(CALIBRATED[:2], [MI, GERD, PE])
    # k1: (0.8 - 1)^2 + 0.2^2 = 0.08 ; k2: 0.6^2 + (0.4 - 1)^2 = 0.72
    assert brier.value == pytest.approx((0.08 + 0.72) / 2)


def test_calibration_needs_probabilities():
    with pytest.raises(ValueError, match="probabilities"):
        expected_calibration_error(CASES)


# --------------------------------------------------------------------------- #
# §6 statistics
# --------------------------------------------------------------------------- #


def test_the_bootstrap_interval_is_reproducible_and_holds_the_value():
    ratio = top_k(CASES * 25, 3)
    interval = bootstrap_ratio(ratio, resamples=500, seed=42)
    assert interval == bootstrap_ratio(ratio, resamples=500, seed=42)
    assert interval[0] <= ratio.value <= interval[1]
    assert 0.0 <= interval[0] < interval[1] <= 1.0


def test_a_constant_metric_has_a_zero_width_interval():
    assert bootstrap_ratio(Ratio(np.ones(8), np.ones(8)), resamples=200) == (1.0, 1.0)


def test_a_metric_with_no_cases_is_nan():
    empty = must_not_miss_recall_at_3([CASES[1], CASES[3]], CRITICAL)
    assert math.isnan(empty.value)
    assert all(math.isnan(bound) for bound in bootstrap_ratio(empty, resamples=50))


def test_mcnemar_exact_for_few_discordant_pairs():
    result = mcnemar([True, True, True, False], [False, False, False, False])
    assert (result["only_a"], result["only_b"], result["method"]) == (3, 0, "exact binomial")
    assert result["p_value"] == pytest.approx(2 * 1 / 2**3)


def test_mcnemar_chi_square_for_many():
    a = [True] * 30 + [False] * 10 + [True] * 50
    b = [False] * 30 + [True] * 10 + [True] * 50
    result = mcnemar(a, b)
    assert result["statistic"] == pytest.approx((30 - 10 - 1) ** 2 / 40)
    assert result["p_value"] == pytest.approx(0.002663, rel=1e-3)  # chi-square, 1 df, 9.025


def test_mcnemar_edge_cases():
    assert mcnemar([True, False], [True, False])["p_value"] == 1.0
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


# --------------------------------------------------------------------------- #
# Records and the full report
# --------------------------------------------------------------------------- #


def test_outcome_from_a_parquet_record_keeps_out_of_scope_entries_in_the_size():
    record = {
        "case_id": "ddxplus-validate-0000007",
        "label_condition_id": GERD,
        "label_differential": [
            {"pathology": "GERD", "condition_id": GERD, "probability": 0.4},
            {"pathology": "Anemia", "condition_id": None, "probability": 0.35},
            {"pathology": "PSVT", "condition_id": PSVT, "probability": 0.25},
        ],
    }
    outcome = outcome_from_record(record, (GERD, PSVT, MI), flagged=[MI])
    assert outcome.differential == {GERD, PSVT} and outcome.differential_size == 3
    assert outcome.true_condition == GERD and outcome.flagged == {MI}


def test_evaluate_reports_every_ratio_metric_with_an_interval():
    results = evaluate(CASES * 10, must_not_miss=CRITICAL, resamples=200)
    by_name = {r.name: r for r in results}
    assert list(by_name) == [
        "top-1 accuracy",
        "top-3 accuracy",
        "top-5 accuracy",
        "MRR",
        "Precision@3",
        "Recall@5",
        "Recall@5, full D",
        "must-not-miss recall@3",
        "dangerous false-negative rate",
        "red-flag sensitivity",
        "red-flag precision",
    ]
    assert by_name["top-3 accuracy"].value == 3 / 4
    assert by_name["must-not-miss recall@3"].cases == 20
    assert all(isinstance(r, MetricResult) and r.ci_low <= r.value <= r.ci_high for r in results)
    assert "top-3 accuracy: 0.750 [" in str(by_name["top-3 accuracy"])


def test_evaluate_adds_brier_only_when_every_case_has_probabilities():
    names = [r.name for r in evaluate(CALIBRATED, resamples=50)]
    assert "Brier score" in names and "red-flag sensitivity" not in names
    assert "Brier score" not in [r.name for r in evaluate(CASES, resamples=50)]
