"""EXP-012's computation (src/eval/ablation_run.py) on synthetic but informative arrays."""

from __future__ import annotations

import numpy as np
import pytest

from src.conditions import must_not_miss_ids
from src.eval import ablation_run as run
from src.eval.ablation import IDS, MODEL_COLUMNS, SPECS, CaseArrays, ranking
from src.eval.ablation_run import PRIME, Level
from src.eval.metrics import CaseOutcome, expected_calibration_error
from src.eval.verdicts import FAMILIES, POLICIES
from src.fusion.calibration import Calibrator

N = 500
LEVELS = (1.0, 0.5, 0.25)
QUALITY = {1.0: 3.0, 0.5: 2.0, 0.25: 1.2}
MODELS = (
    "B1-XGB",
    "B1-LR",
    "B1-XGB+aug",
    "B1-LR+aug",
    "B1-DL+aug",
    "B1-XGB′+aug",
    "B1-LR′+aug",
    "B1-DL′+aug",
)
CRITICAL_IDS = must_not_miss_ids()


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def synthetic(seed: int = 0, patients: int = N):
    """Patients whose true condition is one of the model's 13; a graph and models that know it
    more or less well (less at lower evidence levels)."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, len(MODEL_COLUMNS), size=patients)
    critical = np.array([IDS[t] in CRITICAL_IDS for t in truth])
    rows = np.arange(patients)
    levels, ml = {}, {m: {} for m in MODELS}
    for level in LEVELS:
        graph = -np.abs(rng.normal(size=(patients, len(IDS))) * 3)
        graph[rows, truth] += 2.0 * QUALITY[level] / 3
        flagged = rng.random((patients, len(IDS))) < 0.1
        flagged[rows[critical], truth[critical]] |= rng.random(critical.sum()) < 0.5
        levels[level] = Level(truth=truth, critical=critical, graph=graph, flagged=flagged)
        for i, label in enumerate(MODELS):
            logits = rng.normal(size=(patients, len(MODEL_COLUMNS)))
            logits[rows, truth] += QUALITY[level] + 0.1 * i
            ml[label][level] = softmax(logits)
    alphas = {label: 0.3 for label in MODELS}
    prior = np.full(len(MODEL_COLUMNS), 1 / len(MODEL_COLUMNS))
    return levels, ml, alphas, prior


@pytest.fixture(scope="module")
def seed_run():
    levels, ml, alphas, prior = synthetic()
    return run.run_seed(levels, ml, alphas, prior=prior, resamples=100)


def test_every_system_is_ranked_at_every_level(seed_run):
    labels = {
        "B0",
        "B2",
        "A2",
        *MODELS,
        "A0",
        "A1",
        "A1b",
        "A4",
        "A7-LR",
        "A7-XGB",
        "A7-LR (A0's α)",
        "A7-XGB (A0's α)",
        f"A0{PRIME}",
        f"A1{PRIME}",
        f"A1b{PRIME}",
        f"A4{PRIME}",
        f"A7-LR{PRIME}",
        f"A7-XGB{PRIME}",
        f"A7-LR{PRIME} (A0's α)",
        f"A7-XGB{PRIME} (A0's α)",
    }
    for level in LEVELS:
        assert labels <= {label for label, lv in seed_run.cases if lv == level}


def test_a_component_alone_ranks_the_same_as_ablation_a1(seed_run):
    """A1 ranks 14 columns with dissection last; the model alone ranks its 13: same true ranks."""
    for level in LEVELS:
        assert np.array_equal(
            seed_run.cases[("A1", level)].ranks, seed_run.cases[("B1-DL+aug", level)].ranks
        )
        assert np.array_equal(
            seed_run.cases[(f"A1{PRIME}", level)].ranks,
            seed_run.cases[("B1-DL′+aug", level)].ranks,
        )


def test_a2_without_flags_is_b2():
    levels, ml, alphas, prior = synthetic(3)
    quiet = {
        level: Level(lv.truth, lv.critical, lv.graph, np.zeros_like(lv.flagged))
        for level, lv in levels.items()
    }
    out = run.run_seed(quiet, ml, alphas, prior=prior, resamples=50)
    for level in LEVELS:
        assert np.array_equal(out.cases[("A2", level)].ranks, out.cases[("B2", level)].ranks)


def test_a4_is_a0_without_the_flags_and_differs_when_flags_move_things(seed_run):
    for level in LEVELS:
        assert not np.array_equal(
            seed_run.orders[("A0", level)], seed_run.orders[("A4", level)]
        ), "the synthetic flags should reorder some patients"


def test_a0_ranks_what_the_ablation_ranking_ranks(seed_run):
    levels, ml, alphas, _ = synthetic()
    lv = levels[0.5]
    expected = ranking(SPECS["A0"], ml["B1-DL+aug"][0.5], lv.graph, lv.flagged, 0.3)
    assert np.array_equal(seed_run.orders[("A0", 0.5)], expected.astype(np.int8))


def test_the_comparisons_the_verdicts_read_exist_and_the_claims_are_judged(seed_run):
    for name, family in FAMILIES.items():
        for level in LEVELS:
            for other in ("B0", "B1-XGB", "B1-LR", "B2", *family.like, family.component):
                assert (family.system, other, level) in seed_run.comparisons
            for like in family.like:
                assert (family.component, like, level) in seed_run.comparisons
        assert set(seed_run.claims[name]) == set(POLICIES)
        for policy in POLICIES:
            claims = seed_run.claims[name][policy]
            assert {"H1", "H2", "H1-R", "H2-R", "H4", "H5", "H6", "Target", "Stretch"} <= set(
                claims
            )
            assert all(v["status"] for v in claims.values())


def test_the_two_readings_of_e2_differ_only_in_the_levels_they_check(seed_run):
    both = seed_run.claims["unprimed"]["both"]["H1-R"]["checks"]
    only = seed_run.claims["unprimed"]["25-only"]["H1-R"]["checks"]
    assert {c["level"] for c in both} == {0.5, 0.25}
    assert {c["level"] for c in only} == {0.25}
    assert only == [c for c in both if c["level"] == 0.25]


def test_h4_is_computed_for_each_family_against_a1_and_a1b(seed_run):
    for family in FAMILIES.values():
        assert (family.system, "A1") in seed_run.h4 and (family.system, "A1b") in seed_run.h4


def test_a_family_without_its_component_is_left_out():
    levels, ml, alphas, prior = synthetic(5)
    ml = {k: v for k, v in ml.items() if "′" not in k}
    out = run.run_seed(levels, ml, alphas, prior=prior, resamples=50)
    assert set(out.claims) == {"unprimed"}
    assert not any(label.endswith(PRIME) for label, _ in out.cases if label.startswith("A"))


def test_the_ece_from_arrays_is_the_metrics_functions():
    rng = np.random.default_rng(11)
    confidence = rng.random(300)
    correct = (rng.random(300) < confidence).astype(float)
    outcomes = [
        CaseOutcome(
            case_id=str(i),
            true_condition="COND:a",
            ranking=("COND:a" if c else "COND:b", "COND:c"),
            probabilities={"COND:a": p, "COND:b": p, "COND:c": 0.0},
        )
        for i, (p, c) in enumerate(zip(confidence, correct, strict=True))
    ]
    assert run.ece(confidence, correct) == pytest.approx(expected_calibration_error(outcomes))


def test_a_calibrator_that_changes_nothing_changes_no_ece_and_is_marked():
    levels, ml, alphas, prior = synthetic(7)
    identity = Calibrator(method="none", labels=IDS, fitted=(True,) * len(IDS))
    out = run.run_seed(
        levels, ml, alphas, prior=prior, calibrators={"unprimed": identity}, resamples=50
    )
    key = ("A0", 1.0)
    assert out.ece[key]["calibrated"] is True and out.ece_raw[key]["calibrated"] is False
    assert out.ece[key]["value"] == pytest.approx(out.ece_raw[key]["value"])
    plain = run.run_seed(levels, ml, alphas, prior=prior, resamples=50)
    assert plain.ece_raw == {} and plain.ece[key]["calibrated"] is False


def test_breakdown_by_group_has_intervals_and_the_right_counts():
    ranks = np.array([1, 2, 5, 1, 1, 4, 1, 3] * 20)
    cases = CaseArrays(ranks=ranks, critical=np.array([True, False] * 80))
    groups = np.array(["F", "M"] * 80)
    out = run.breakdown(cases, groups, resamples=100)
    assert set(out) == {"F", "M"} and out["F"]["cases"] == 80
    assert out["F"]["top-3 accuracy"]["value"] == pytest.approx(cases.hit3[groups == "F"].mean())
    low, high = out["F"]["top-3 accuracy"]["ci"]
    assert low <= out["F"]["top-3 accuracy"]["value"] <= high
    assert "must-not-miss recall@3" in out["F"]


def test_condition_names_map_registry_indices_to_ids():
    assert run.condition_names(np.array([0, 3])).tolist() == [IDS[0], IDS[3]]
