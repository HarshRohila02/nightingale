"""Tests for the red-flag rules (task 2d, src/reasoning/red_flags.py; EXP-008).

The synthetic cases pin each rule to its clinical pattern (docs/04 §3): what must fire it and,
as important after R-15, what must not. The rates on DDXPlus validate patients are checked
against EXP-008 and skip when data/ is absent, as in CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.conditions import CONDITIONS
from src.contracts import Finding, PatientCase
from src.reasoning.red_flags import APPROPRIATE, RULES, RedFlagRule, evaluate_red_flags

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = yaml.safe_load(
    (REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml").read_text(encoding="utf-8")
)["cases"]
PARQUET = REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_validate.parquet"

AD = "COND:aortic_dissection"
PE = "COND:pulmonary_embolism"
UA = "COND:unstable_angina"
MYO = "COND:myocarditis"
APE = "COND:acute_pulmonary_edema"


def flags(*present: str) -> set[str]:
    case = PatientCase(
        case_id="T-RF",
        age=55,
        sex="M",
        findings=[Finding(concept_id=c, label=c) for c in present],
    )
    return set(evaluate_red_flags(case))


# --------------------------------------------------------------------------- #
# The rule model
# --------------------------------------------------------------------------- #


class TestRuleModel:
    RULE = RedFlagRule(
        condition_id="COND:x",
        reason="x",
        all_of=frozenset({"A"}),
        groups=(frozenset({"B1", "B2"}), frozenset({"C"}), frozenset({"D"})),
        min_groups=2,
    )

    @pytest.mark.parametrize(
        ("present", "fires"),
        [
            ({"A", "B1", "C"}, True),
            ({"A", "B2", "D"}, True),
            ({"A", "B1", "B2"}, False),  # two findings, one category
            ({"B1", "C", "D"}, False),  # all_of still holds
            ({"A"}, False),
        ],
    )
    def test_groups_count_categories_not_findings(self, present, fires):
        case = PatientCase(
            case_id="T",
            age=40,
            sex="F",
            findings=[Finding(concept_id=c, label=c) for c in present],
        )
        assert self.RULE.matches(case) is fires

    def test_without_min_groups_every_group_is_needed(self):
        rule = RedFlagRule("COND:x", "x", groups=(frozenset({"A"}), frozenset({"B"})))
        assert rule.min_groups is None and rule.concepts == {"A", "B"}

    def test_an_empty_rule_never_fires(self):
        assert not RedFlagRule("COND:x", "x").matches(PatientCase(case_id="T", age=1, sex="M"))


def test_every_must_not_miss_condition_has_a_rule():
    """docs/05 amendment 2 counts a condition without a rule as missed."""
    ruled = {rule.condition_id for rule in RULES}
    assert {c.id for c in CONDITIONS if c.is_must_not_miss} <= ruled


# --------------------------------------------------------------------------- #
# Each rule, what fires it and what must not
# --------------------------------------------------------------------------- #


class TestAorticDissection:
    def test_back_radiation_alone_no_longer_fires(self):
        """R-15: it did, for 50% of DDXPlus validate patients (EXP-014)."""
        assert AD not in flags("SYM:chest_pain", "SYM:radiation_back")

    @pytest.mark.parametrize(
        "sign",
        ["SYM:pain_character_tearing", "SYM:interarm_bp_difference", "SYM:pulse_deficit"],
    )
    def test_one_specific_sign_is_enough(self, sign):
        assert AD in flags("SYM:chest_pain", sign)

    @pytest.mark.parametrize(
        "pair",
        [
            ("SYM:sudden_onset", "SYM:hypotension"),  # pain features + examination
            ("SYM:severe_pain", "RF:connective_tissue_disease"),  # pain features + condition
            ("SYM:focal_neuro_deficit", "RF:aortic_valve_disease"),  # examination + condition
        ],
    )
    def test_two_add_rs_categories_fire(self, pair):
        assert AD in flags("SYM:back_pain", *pair)

    def test_one_add_rs_category_does_not(self):
        """Sudden severe pain is one category: PE, pneumothorax and MI patients have it too."""
        assert AD not in flags("SYM:chest_pain", "SYM:sudden_onset", "SYM:severe_pain")

    def test_it_needs_pain(self):
        assert AD not in flags("SYM:interarm_bp_difference", "SYM:hypotension")


class TestPulmonaryEmbolism:
    def test_a_risk_factor_alone_no_longer_fires(self):
        """The reason always said "pleuritic chest pain or breathlessness with a thrombosis
        risk factor"; before 2d the rule checked only the risk factor."""
        assert PE not in flags("SYM:recent_immobilisation")

    @pytest.mark.parametrize(
        "risk",
        [
            "SYM:leg_swelling_unilateral",
            "SYM:calf_pain",
            "SYM:recent_immobilisation",
            "RF:recent_surgery",
            "RF:previous_dvt",
        ],
    )
    @pytest.mark.parametrize("symptom", ["SYM:pleuritic", "SYM:breathlessness"])
    def test_a_symptom_with_a_wells_risk_fires(self, symptom, risk):
        assert PE in flags(symptom, risk)


class TestUnstableAngina:
    def test_a_crescendo_pattern_fires(self):
        assert UA in flags("SYM:chest_pain", "SYM:crescendo_pattern")

    def test_rest_pain_needs_an_ischaemic_character(self):
        """DDXPlus asks "chest pain even at rest?" of pneumothorax patients too (56% say yes)."""
        assert UA not in flags("SYM:chest_pain", "SYM:rest_pain")
        assert UA in flags("SYM:chest_pain", "SYM:rest_pain", "SYM:pain_character_pressure")


class TestMyocarditis:
    def test_post_viral_chest_pain_with_breathlessness_or_palpitations(self):
        assert MYO in flags("SYM:chest_pain", "SYM:recent_viral_illness", "SYM:breathlessness")
        assert MYO in flags("SYM:chest_pain", "SYM:recent_viral_illness", "SYM:palpitations")

    def test_not_without_the_viral_illness(self):
        assert MYO not in flags("SYM:chest_pain", "SYM:breathlessness", "SYM:palpitations")


class TestAcutePulmonaryEdema:
    def test_paroxysmal_nocturnal_dyspnoea_is_enough(self):
        assert APE in flags("SYM:paroxysmal_nocturnal_dyspnoea")

    @pytest.mark.parametrize("feature", ["SYM:orthopnoea", "RF:heart_failure"])
    def test_breathlessness_with_orthopnoea_or_heart_failure(self, feature):
        assert APE in flags("SYM:breathlessness", feature)

    def test_heart_failure_without_breathlessness_is_not(self):
        assert APE not in flags("RF:heart_failure")


@pytest.mark.golden
@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("GC-001", {"COND:nstemi_stemi"}),
        ("GC-002", {PE, "COND:spontaneous_pneumothorax"}),  # sudden pleuritic pain, breathless
        ("GC-003", {AD}),
        ("GC-004", set()),
    ],
)
def test_the_golden_cases_raise_exactly_these_flags(case_id, expected):
    golden = next(c for c in GOLDEN if c["id"] == case_id)
    assert set(evaluate_red_flags(PatientCase(**golden["case"]))) == expected


# --------------------------------------------------------------------------- #
# Decision A-7: which flags are appropriate for which true conditions
# --------------------------------------------------------------------------- #


def test_every_flag_has_an_appropriateness_entry_that_includes_itself():
    assert set(APPROPRIATE) == {rule.condition_id for rule in RULES}
    assert all(flag in allowed for flag, allowed in APPROPRIATE.items())
    known = {c.id for c in CONDITIONS}
    assert all(allowed <= known for allowed in APPROPRIATE.values())


@pytest.mark.parametrize(
    ("flag", "true_condition", "appropriate"),
    [
        # The pairs EXP-008 saw fire, with the panel's verdicts (docs/04 §3).
        ("COND:nstemi_stemi", "COND:unstable_angina", True),
        ("COND:nstemi_stemi", "COND:stable_angina", True),
        ("COND:nstemi_stemi", "COND:acute_pulmonary_edema", True),
        ("COND:myocarditis", "COND:pericarditis", True),
        ("COND:aortic_dissection", "COND:boerhaave", False),
        ("COND:aortic_dissection", "COND:spontaneous_pneumothorax", False),
        ("COND:pulmonary_embolism", "COND:acute_pulmonary_edema", False),
        ("COND:spontaneous_pneumothorax", "COND:pulmonary_embolism", False),
        ("COND:spontaneous_pneumothorax", "COND:pericarditis", False),
    ],
)
def test_a7_verdicts_on_the_pairs_that_fire(flag, true_condition, appropriate):
    assert (true_condition in APPROPRIATE[flag]) is appropriate


def test_a7_counts_24_cross_condition_pairs():
    """The panel's majority: 13 unanimous and 11 two-to-one (docs/04 §3)."""
    assert sum(len(allowed) - 1 for allowed in APPROPRIATE.values()) == 24


# --------------------------------------------------------------------------- #
# On DDXPlus validate patients (EXP-008)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not PARQUET.exists(), reason="data/ is not committed (CI)")
def test_exp008_on_validate(tmp_path):
    """Guards EXP-008's findings: the dissection rule no longer fires on half the patients, every
    must-not-miss condition's flag reaches its own patients, and the false-alarm burden fell."""
    (tmp_path / PARQUET.name).write_bytes(PARQUET.read_bytes())
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_red_flags.py")]
        + ["--interim", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "red_flags_check.json").read_text(encoding="utf-8"))
    rules = summary["rates"]["rules"]
    assert rules[AD]["other_patients_flagged"] < 0.10, "R-15: it was 0.50"
    assert all(
        rules[c]["own_patients_flagged"] > 0.3 for c in rules if c != AD
    ), "every rule reaches its own patients"
    assert summary["false_alarm_burden"]["other_patients_flagged"] < 0.31, "0.31 before 2d"
    assert summary["red_flag_sensitivity"]["overall"]["value"] > 0.8, "0.449 before 2d"
