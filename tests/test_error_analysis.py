"""EXP-023's error analysis (3e): the helpers in src/eval/errors.py, the array forms added to
src/eval/metrics.py for it, and scripts/error_analysis.py's analysis and figures.

Everything here runs on tiny synthetic outcomes whose answers are worked out by hand. The real run
(scripts/error_analysis.py) scores the validation split; the one test that reads its output skips
where data/ is absent, as in CI.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from src.conditions import CONDITIONS
from src.eval import errors
from src.eval.metrics import (
    CaseOutcome,
    binned_gap_intervals,
    calibration_error_by_condition,
    calibration_error_by_condition_interval,
    calibration_error_interval,
    reliability_rows,
    reliability_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import error_analysis  # noqa: E402

MI = "COND:nstemi_stemi"
UA = "COND:unstable_angina"
SA = "COND:stable_angina"
GERD = "COND:gerd"
PE = "COND:pulmonary_embolism"
AF = "COND:atrial_fibrillation"
PSVT = "COND:psvt"
IDS = tuple(c.id for c in CONDITIONS)


def case(i: int, truth: str, *ranking: str, probabilities: dict | None = None) -> CaseOutcome:
    return CaseOutcome(f"c{i}", truth, tuple(ranking), probabilities=probabilities)


@pytest.fixture
def cases() -> list[CaseOutcome]:
    """Eight cases: MI right once, confused with UA twice (once missed from the top 3, at rank
    5); UA right once and missed once at rank 6; SA twice ranked below UA; GERD right."""
    return [
        case(1, MI, MI, UA, SA, GERD),
        case(2, MI, UA, MI, SA, GERD),
        case(3, MI, UA, SA, GERD, PE, MI),
        case(4, UA, UA, MI, SA, GERD),
        case(5, SA, UA, SA, MI, GERD),
        case(6, SA, UA, SA, MI, GERD),
        case(7, GERD, GERD, UA, MI, SA),
        case(8, UA, GERD, SA, PE, AF, PSVT, UA),
    ]


# --------------------------------------------------------------------------- #
# src/eval/errors.py
# --------------------------------------------------------------------------- #


def test_recall_by_condition_counts_each_condition_on_its_own_cases(cases):
    top1 = errors.recall_by_condition(cases, 1, [MI, UA, SA, GERD, PE])
    top3 = errors.recall_by_condition(cases, 3, [MI, UA, SA, GERD, PE])
    assert {c: e["value"] for c, e in top1.items()} == {MI: 1 / 3, UA: 0.5, SA: 0.0, GERD: 1.0}
    assert {c: e["value"] for c, e in top3.items()} == {MI: 2 / 3, UA: 0.5, SA: 1.0, GERD: 1.0}
    assert {c: e["cases"] for c, e in top1.items()} == {MI: 3, UA: 2, SA: 2, GERD: 1}
    assert PE not in top1  # no case has it: nothing to measure
    for entry in top1.values():
        low, high = entry["ci"]
        assert low <= entry["value"] <= high


def test_confusion_matrix_rows_are_truth_and_columns_first_ranked(cases):
    matrix = errors.confusion_matrix(cases, [MI, UA, SA, GERD], [MI, UA, SA, GERD])
    assert matrix.tolist() == [[1, 2, 0, 0], [0, 1, 0, 1], [0, 2, 0, 0], [0, 0, 0, 1]]
    assert matrix.sum() == len(cases)


def test_confusion_matrix_refuses_a_case_it_cannot_place(cases):
    with pytest.raises(ValueError, match="not a row"):
        errors.confusion_matrix(cases, [MI, UA, SA], [MI, UA, SA, GERD])
    with pytest.raises(ValueError, match="not a column"):
        errors.confusion_matrix(cases, [MI, UA, SA, GERD], [MI, UA, SA])
    with pytest.raises(ValueError, match="empty ranking"):
        errors.confusion_matrix([case(9, MI)], [MI], [MI])


def test_top_confusions_are_ordered_by_count_then_registry(cases):
    top = errors.top_confusions(cases, 10)
    assert [(c["true"], c["predicted"], c["count"]) for c in top] == [
        (MI, UA, 2),  # ties with SA → UA; MI comes first in the registry
        (SA, UA, 2),
        (UA, GERD, 1),
    ]
    assert [c["share"]["value"] for c in top] == [2 / 3, 1.0, 0.5]
    assert [c["share"]["cases"] for c in top] == [3, 2, 2]
    assert len(errors.top_confusions(cases, 1)) == 1


def test_must_not_miss_misses_say_where_the_missed_condition_went(cases):
    misses = errors.must_not_miss_misses(cases)
    mi, ua = misses["by_condition"][MI], misses["by_condition"][UA]
    assert (mi["cases"], mi["misses"]) == (3, 1)
    assert mi["rank"] == {"4": 0, "5": 1, ">5": 0}
    assert mi["first"] == {UA: 1}
    assert mi["above"] == {UA: 1, SA: 1, PE: 1, GERD: 1}  # registry order among equal counts
    assert mi["share_missed"]["value"] == pytest.approx(1 / 3)
    assert ua["rank"] == {"4": 0, "5": 0, ">5": 1}
    assert ua["first"] == {GERD: 1}
    assert set(ua["above"]) == {GERD, SA, PE, AF, PSVT}
    assert SA not in misses["by_condition"]  # stable angina is not must-not-miss
    assert (misses["all"]["cases"], misses["all"]["misses"]) == (5, 2)
    assert misses["all"]["first"] == {UA: 1, GERD: 1}


def test_must_not_miss_misses_counts_a_true_condition_never_ranked():
    missing = [case(1, MI, UA, SA, GERD)]
    entry = errors.must_not_miss_misses(missing)["by_condition"][MI]
    assert entry["rank"] == {"4": 0, "5": 0, ">5": 1}
    assert entry["above"] == {UA: 1, SA: 1, GERD: 1}


def test_error_overlap_splits_errors_into_persisting_new_and_resolved(cases):
    shorter = list(cases)
    shorter[0] = case(1, MI, UA, MI, SA)  # new top-1 error
    shorter[1] = case(2, MI, MI, UA, SA)  # resolved
    overlap = errors.error_overlap(cases, shorter, 1)
    assert overlap == {
        "errors_fuller": 5,
        "errors_shorter": 5,
        "persisting": 4,
        "new": 1,
        "resolved": 1,
    }
    assert errors.error_overlap(cases, cases, 3)["new"] == 0
    with pytest.raises(ValueError, match="same cases"):
        errors.error_overlap(cases, cases[::-1])


def test_headline_reproduces_the_protocol_definitions(cases):
    head = errors.headline(cases)
    assert head["top-1 accuracy"]["value"] == pytest.approx(3 / 8)
    assert head["top-3 accuracy"]["value"] == pytest.approx(6 / 8)
    # Must-not-miss: MI ×3 and UA ×2; MI at rank 5 and UA at rank 6 are missed from the top 3.
    assert head["must-not-miss recall@3"]["value"] == pytest.approx(3 / 5)
    assert head["must-not-miss recall@3"]["cases"] == 5
    # Outside the top 5: only UA at rank 6.
    assert head["dangerous false-negative rate"]["value"] == pytest.approx(1 / 5)
    mrr = (1 + 1 / 2 + 1 / 5 + 1 + 1 / 2 + 1 / 2 + 1 + 1 / 6) / 8
    assert head["MRR"]["value"] == pytest.approx(mrr)


# --------------------------------------------------------------------------- #
# Calibration, per condition and overall
# --------------------------------------------------------------------------- #


def probability_cases(n: int = 400, seed: int = 3) -> list[CaseOutcome]:
    """Cases over three conditions with random probabilities, the truth drawn from them."""
    rng = np.random.default_rng(seed)
    labels = (MI, UA, SA)
    out = []
    for i in range(n):
        p = rng.dirichlet(np.ones(3) * 0.7)
        truth = labels[rng.choice(3, p=p)]
        order = tuple(labels[j] for j in np.argsort(-p, kind="stable"))
        out.append(case(i, truth, *order, probabilities=dict(zip(labels, p.tolist(), strict=True))))
    return out


def test_reliability_rows_is_the_reliability_table_from_arrays():
    outcomes = probability_cases()
    confidence = np.array([o.probabilities[o.ranking[0]] for o in outcomes])
    correct = np.array([float(o.ranking[0] == o.true_condition) for o in outcomes])
    assert reliability_rows(confidence, correct) == reliability_table(outcomes)


def test_reliability_rows_by_hand():
    rows = reliability_rows(np.array([0.05, 0.15, 0.12, 0.95, 1.0]), np.array([0, 1, 0, 1, 1]))
    assert [(r["bin_low"], r["cases"]) for r in rows] == [(0.0, 1), (0.1, 2), (0.9, 2)]
    assert rows[1]["confidence"] == pytest.approx(0.135)
    assert rows[1]["accuracy"] == pytest.approx(0.5)
    assert rows[2]["confidence"] == pytest.approx(0.975)  # 1.0 falls in the last bin


def test_one_column_interval_is_the_top_label_interval():
    outcomes = probability_cases()
    confidence = np.array([o.probabilities[o.ranking[0]] for o in outcomes])
    correct = np.array([float(o.ranking[0] == o.true_condition) for o in outcomes])
    (interval,) = binned_gap_intervals(confidence[:, None], correct[:, None])
    assert interval == calibration_error_interval(outcomes)


def test_per_condition_intervals_share_the_resamples_of_a_single_column():
    outcomes = probability_cases()
    intervals = calibration_error_by_condition_interval(outcomes, [MI, UA, SA, GERD])
    assert set(intervals) == {MI, UA, SA}  # GERD has no case
    for label in (MI, UA, SA):
        given = np.array([o.probabilities[label] for o in outcomes])
        positive = np.array([float(o.true_condition == label) for o in outcomes])
        assert intervals[label] == binned_gap_intervals(given[:, None], positive[:, None])[0]
        low, high = intervals[label]
        assert 0 <= low <= high


def test_binned_gap_intervals_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="matching"):
        binned_gap_intervals(np.zeros((3, 2)), np.zeros((3, 1)))
    with pytest.raises(ValueError, match="matching"):
        binned_gap_intervals(np.zeros(3), np.zeros(3))


def test_calibration_gathers_every_figure_for_one_system():
    outcomes = probability_cases()
    figures = errors.calibration(outcomes, [MI, UA, SA])
    by_condition = calibration_error_by_condition(outcomes, [MI, UA, SA])
    assert set(figures["one_vs_rest"]) == {MI, UA, SA}
    for label, entry in figures["one_vs_rest"].items():
        assert entry["ece"] == by_condition[label]
        assert sum(r["cases"] for r in entry["reliability"]) == len(outcomes)
    assert figures["top_label"]["reliability"] == reliability_table(outcomes)
    assert len(figures["top_label"]["ci"]) == 2


def test_a_perfectly_calibrated_condition_has_no_gap():
    # Two bins: 0.25 given four times with one hit, 0.75 four times with three hits.
    outcomes = [
        case(i, MI if hit else UA, MI, UA, probabilities={MI: p, UA: 1 - p})
        for i, (p, hit) in enumerate(
            [(0.25, 1), (0.25, 0), (0.25, 0), (0.25, 0), (0.75, 1), (0.75, 1), (0.75, 1), (0.75, 0)]
        )
    ]
    figures = errors.calibration(outcomes, [MI])
    assert figures["one_vs_rest"][MI]["ece"] == pytest.approx(0.0)
    rows = figures["one_vs_rest"][MI]["reliability"]
    assert [(r["confidence"], r["accuracy"]) for r in rows] == [(0.25, 0.25), (0.75, 0.75)]


# --------------------------------------------------------------------------- #
# scripts/error_analysis.py
# --------------------------------------------------------------------------- #


def test_slugs_name_the_figure_files():
    assert error_analysis.slug("B1-DL′+aug") == "b1_dl_prime_aug"
    assert error_analysis.slug("B1-XGB′+aug") == "b1_xgb_prime_aug"
    assert error_analysis.slug("B2") == "b2"


def test_the_script_scores_no_fused_ranking_and_never_the_test_split():
    """E-2 is unanswered, so no fused validation ranking may be read (EXP-006), and the test
    split stays closed until Phase 4."""
    source = (REPO_ROOT / "scripts" / "error_analysis.py").read_text(encoding="utf-8")
    for name in ("fuse_arrays", "fuse_scores", "from src.fusion", "DiagnosisPipeline"):
        assert name not in source
    assert "test.csv" not in source and "_test.parquet" not in source
    assert error_analysis.SYSTEMS == ("B1-DL′+aug", "B1-LR′+aug", "B1-XGB′+aug", "B2")


def synthetic_results(seed: int = 5, per_condition: int = 12) -> dict:
    """Results in the run's shape, from random outcomes over every condition: what the figures
    are drawn from."""
    rng = np.random.default_rng(seed)
    labels = error_analysis.ROWS
    results: dict = {"systems": {}}
    for system in error_analysis.SYSTEMS:
        results["systems"][system] = {"levels": {}}
        for level in error_analysis.LEVELS:
            outcomes = []
            for t, truth in enumerate(labels):
                for i in range(per_condition):
                    p = rng.dirichlet(np.ones(len(IDS)) * 0.3)
                    p[IDS.index(truth)] += 2.0
                    p /= p.sum()
                    order = np.argsort(-p, kind="stable")
                    outcomes.append(
                        case(
                            t * per_condition + i,
                            truth,
                            *(IDS[j] for j in order),
                            probabilities=dict(zip(IDS, p.tolist(), strict=True)),
                        )
                    )
            results["systems"][system]["levels"][level] = error_analysis.analyse(outcomes)
    return results


def test_every_figure_is_drawn_from_the_results_alone(tmp_path):
    written = error_analysis.figures(synthetic_results(), tmp_path)
    systems, levels = len(error_analysis.SYSTEMS), len(error_analysis.LEVELS)
    assert len(written) == systems * (2 + levels) + 2
    for path in written:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        assert root.tag.endswith("svg")
        text = "".join(root.itertext())
        assert "DDXPlus is synthetic" in text
        assert "EXP-023" in text
    names = {p.name for p in written}
    assert "exp023_confusion_b1_dl_prime_aug_25.svg" in names
    assert "exp023_reliability_by_condition_b2.svg" in names
    assert "R-12" in (tmp_path / "exp023_reliability_b2.svg").read_text(encoding="utf-8")


def test_analyse_is_json_ready_and_names_no_patient():
    results = synthetic_results(per_condition=3)
    text = json.dumps(results)
    assert '"c1"' not in text and "case_id" not in text


REAL_RESULTS = REPO_ROOT / "data" / "interim" / "exp023_error_analysis.json"


@pytest.mark.skipif(not REAL_RESULTS.exists(), reason="EXP-023's results are not on this machine")
def test_the_recorded_run_reproduces_its_references_and_redraws(tmp_path):
    results = json.loads(REAL_RESULTS.read_text(encoding="utf-8"))
    assert results["digests"] == {"tokens": "e99a7a8fbf792675", "asked": "af3a357f6454838a"}
    assert results["split"] == "validate"
    for system, check in results["reproduction"].items():
        assert check["reproduced"], system
    for path in error_analysis.figures(results, tmp_path):
        ET.fromstring(path.read_text(encoding="utf-8"))
