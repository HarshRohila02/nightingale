"""The protocol's verdicts (src/eval/verdicts.py), on comparisons built from crafted rankings.

Each scenario is real ``ablation.compare`` output over 400 patients (200 must-not-miss), so the
rules are exercised on the same shapes the runner produces. A system "misses" a patient when it
ranks the true condition 4th; the misses are nested, so a system with fewer misses beats one with
more on exactly the patients between them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval import ablation, verdicts
from src.eval.ablation import CaseArrays
from src.eval.verdicts import FAMILIES, POLICIES, Table

REPO_ROOT = Path(__file__).resolve().parents[1]
LEVELS = (1.0, 0.5, 0.25)
N = 400
CRITICAL = np.array([True] * 200 + [False] * 200)

# (must-not-miss patients missed, others missed): nested, so fewer misses is strictly better.
BASE = {
    "A0": (0, 0),
    "B0": (150, 150),
    "B1-XGB": (60, 60),
    "B1-LR": (50, 50),
    "B2": (100, 100),
    "like-xgb": (40, 40),
    "like-lr": (30, 30),
    "component": (20, 20),
}


def system(critical_missed: int, other_missed: int) -> CaseArrays:
    ranks = np.ones(N, dtype=int)
    ranks[:critical_missed] = 4
    ranks[200 : 200 + other_missed] = 4
    return CaseArrays(ranks=ranks, critical=CRITICAL)


def labels(family: verdicts.Family) -> dict[str, str]:
    return {
        "A0": family.system,
        "like-xgb": family.like[0],
        "like-lr": family.like[1],
        "component": family.component,
    }


def build(
    family_name: str = "unprimed",
    overrides: dict[tuple[str, float], tuple[int, int]] | None = None,
    *,
    ece: float = 0.02,
    h4_more: bool = True,
) -> Table:
    family = FAMILIES[family_name]
    names = labels(family)
    overrides = overrides or {}

    def arrays(key: str, level: float) -> CaseArrays:
        return system(*overrides.get((key, level), BASE[key]))

    def label(key: str) -> str:
        return names.get(key, key)

    comparisons, points, eces = {}, {}, {}
    for level in LEVELS:
        a = arrays("A0", level)
        for other in ("B0", "B1-XGB", "B1-LR", "B2", "like-xgb", "like-lr", "component"):
            comparisons[(family.system, label(other), level)] = ablation.compare(
                a, arrays(other, level), resamples=100
            )
        for other in ("like-xgb", "like-lr"):
            comparisons[(family.component, label(other), level)] = ablation.compare(
                arrays("component", level), arrays(other, level), resamples=100
            )
        points[(family.system, level)] = ablation.point_metrics(a)
        eces[(family.system, level)] = {"value": ece, "ci": [ece - 0.01, ece + 0.01]}
    contrast = {"more_on_must_not_miss": h4_more, "contrast": 0.1, "contrast_ci": [0.05, 0.15]}
    h4 = {(family.system, "A1"): contrast, (family.system, "A1b"): contrast}
    return Table(comparisons=comparisons, points=points, ece=eces, h4=h4)


def judge(table: Table, family_name: str = "unprimed", policy: str = "both", **external):
    return verdicts.evaluate_claims(table, FAMILIES[family_name], POLICIES[policy], **external)


# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family_name", ["unprimed", "primed"])
def test_when_the_full_system_beats_every_comparator_every_hypothesis_is_supported(family_name):
    out = judge(build(family_name), family_name)
    for claim in ("H1", "H2", "H1-R", "H2-R", "H4", "H5", "H6"):
        assert out[claim]["status"] == "supported", claim
    assert out["Target, reduced evidence"]["status"] == "met"
    assert out["Stretch, reduced evidence"]["status"] == "met"


def test_the_checks_name_every_comparison_passed_or_not():
    out = judge(build())
    h1r = out["H1-R"]
    # 2 levels × (B1-XGB, B1-LR, the two trained like A0, the component, B2) × (top-3, MRR)
    assert len(h1r["checks"]) == 2 * 6 * 2
    assert {c["level"] for c in h1r["checks"]} == {0.5, 0.25}
    assert {c["against"] for c in h1r["checks"]} == {
        "B1-XGB",
        "B1-LR",
        "B1-XGB+aug",
        "B1-LR+aug",
        "B1-DL+aug",
        "B2",
    }
    assert out["H6"]["checks"] and out["H6"]["reported_beside"]


def test_a_perfect_comparator_at_50_percent_defeats_the_claim_as_approved_but_not_at_25_only():
    """The E-2 situation (docs/05 §1): a `+aug` comparator that is perfect at 50% cannot be beaten,
    so H1-R, H2-R and the reduced Target need the team's answer to be satisfiable at all."""
    table = build(overrides={("like-lr", 0.5): (0, 0), ("like-xgb", 0.5): (0, 0)})
    both = judge(table, policy="both")
    only_25 = judge(table, policy="25-only")
    for claim in ("H1-R", "H2-R"):
        assert both[claim]["status"] == "not supported"
        assert both[claim]["failed_at_a_ceiling"] == both[claim]["failed"] > 0
        assert only_25[claim]["status"] == "supported"
    assert both["Target, reduced evidence"]["status"] == "not met"
    assert only_25["Target, reduced evidence"]["status"] == "met"
    ceiling = [c for c in both["H1-R"]["checks"] if "ceiling" in c]
    assert all(c["level"] == 0.5 and "perfect" in c["ceiling"] for c in ceiling)


def test_a_comparator_that_misses_five_cases_cannot_be_beaten_significantly_but_can_by_point():
    table = build(overrides={("like-lr", 0.5): (3, 2)})  # 5 misses, 3 of them must-not-miss
    out = judge(table)
    failed = [c for c in out["H1-R"]["checks"] if not c["passed"]]
    assert failed and all("McNemar cannot reach 0.05" in c["ceiling"] for c in failed)
    assert out["H1-R"]["status"] == "not supported"
    assert out["Target, reduced evidence"]["status"] == "met"  # point estimates: still higher
    assert out["Stretch, reduced evidence"]["status"] == "not met"  # needs every test to pass


def test_a_target_compares_point_estimates_and_a_hypothesis_needs_significance():
    # B1-XGB misses 2 patients, both must-not-miss: A0 is higher on both metrics, not significantly.
    out = judge(build(overrides={("B1-XGB", 1.0): (2, 0)}))
    assert out["Target"]["checks"][0]["passed"] and out["Target"]["checks"][1]["passed"]
    assert out["H1"]["status"] == "not supported"
    assert out["H2"]["status"] == "not supported"


def test_one_failed_comparison_defeats_a_hypothesis():
    # B2 (the graph alone) beats A0 on the 50% history: H1-R fails, H2-R (ML-only) does not.
    out = judge(build(overrides={("B2", 0.5): (0, 0), ("A0", 0.5): (30, 30)}))
    assert out["H1-R"]["status"] == "not supported"
    failing = [c["against"] for c in out["H1-R"]["checks"] if not c["passed"] and c["level"] == 0.5]
    assert "B2" in failing


def test_a_system_worse_than_the_comparator_fails_even_when_the_difference_is_significant():
    out = judge(build(overrides={("B1-XGB", 1.0): (0, 0), ("A0", 1.0): (60, 60)}))
    assert out["H1"]["status"] == "not supported"
    assert out["Target"]["status"] == "not met"
    check = out["Target"]["checks"][0]
    assert not check["passed"] and check["difference"] < 0


def test_the_target_is_incomplete_without_the_explanation_results():
    out = judge(build())
    assert out["Target"]["status"] == "incomplete"
    assert out["Target"]["not_assessed"] == ["unsupported-claim rate (EXP-009/010 not run)"]
    assert judge(build(), unsupported_rate=0.12)["Target"]["status"] == "met"
    assert judge(build(), unsupported_rate=0.25)["Target"]["status"] == "not met"


def test_the_stretch_is_incomplete_without_a_clinically_trained_reviewer():
    out = judge(build())
    assert out["Stretch"]["status"] == "incomplete"
    assert out["Stretch"]["not_assessed"] == ["clinician-reviewed explainability"]
    assert judge(build(), clinician_review=True)["Stretch"]["status"] == "met"


def test_the_stretch_needs_must_not_miss_recall_of_at_least_95_percent():
    # A0 misses 12 of the 200 must-not-miss patients (recall 0.94) but beats the weaker B1-XGB.
    out = judge(build(overrides={("A0", 1.0): (12, 0)}), clinician_review=True)
    assert out["Stretch"]["status"] == "not met"
    recall = out["Stretch"]["checks"][-1]
    assert recall["metric"] == "must-not-miss recall@3" and recall["a"] == pytest.approx(0.94)


def test_h5_reads_the_point_estimate_of_the_ece():
    assert judge(build(ece=0.099))["H5"]["status"] == "supported"
    bad = judge(build(ece=0.12))
    assert bad["H5"]["status"] == "not supported"
    assert bad["Target"]["status"] == "not met"


def test_h4_follows_the_degradation_contrast():
    assert judge(build(h4_more=True))["H4"]["status"] == "supported"
    out = judge(build(h4_more=False))
    assert out["H4"]["status"] == "not supported"
    assert out["H4"]["reported_beside"][0]["against"] == "A1b"


def test_the_minimum_needs_a0_to_beat_b0():
    assert judge(build())["Minimum"]["status"] == "met"
    worse = judge(build(overrides={("B0", 1.0): (0, 0), ("A0", 1.0): (50, 50)}))
    assert worse["Minimum"]["status"] == "not met"


def test_a_missing_comparison_is_an_error_not_a_silent_pass():
    table = build()
    comparisons = dict(table.comparisons)
    del comparisons[("A0", "B2", 0.25)]
    broken = Table(comparisons=comparisons, points=table.points, ece=table.ece, h4=table.h4)
    with pytest.raises(KeyError, match="was not computed"):
        judge(broken)


def test_the_policies_are_the_two_readings_of_e2():
    assert POLICIES["both"] == (0.5, 0.25)
    assert POLICIES["25-only"] == (0.25,)


def test_the_families_use_the_labels_the_scoring_scripts_use():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import evaluate_reduced_evidence as exp019  # noqa: PLC0415

    scored = {label for label, _, _ in exp019.MODELS}
    for family in FAMILIES.values():
        for name in family.like:
            assert name in scored, name
    assert FAMILIES["primed"].component == "B1-DL′+aug"
    assert FAMILIES["unprimed"].component == "B1-DL+aug"
