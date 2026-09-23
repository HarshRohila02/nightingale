"""Tests for the concept -> DDXPlus-token inversion (task 1c, src/ml/case_tokens.py).

This inversion has no ground truth (risk R-17): no dataset of (hand-authored case -> tokens)
pairs exists, so a wrong representative answer looks exactly like a model error downstream. These
tests are the substitute, and they check three separate things:

1. **The tables are complete and legal.** Every concept that needs a representative answer has
   one, and every chosen answer is one the release file actually allows.
2. **The inversion round-trips.** Feeding the tokens back through
   ``concepts_from_evidences`` recovers the concepts we started from. That closes the loop
   against the crosswalk's own, independently written, forward direction.
3. **Nothing is lost silently.** Every finding that yields no token is in ``dropped`` with a
   reason.

The tests that need the release files skip when data/ is absent, as in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.contracts import Assertion, Finding, PatientCase, Sex
from src.ddxplus import TOKEN_SEP, load_evidence_specs
from src.medical_kg.crosswalk import (
    BY_CONCEPT,
    CROSSWALK,
    Match,
    concepts_from_evidences,
    expand_case,
)
from src.ml.case_tokens import (
    ORDINAL_REPRESENTATIVE,
    REPRESENTATIVE,
    DropReason,
    UnrepresentableCase,
    encode_case,
    tokens_for_case,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"
REAL_EVIDENCES = REPO_ROOT / "data" / "raw" / "ddxplus" / "release_evidences.json"
REAL_CONDITIONS = REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json"
needs_real_data = pytest.mark.skipif(
    not all(p.exists() for p in (REAL_EVIDENCES, REAL_CONDITIONS)),
    reason="data/ is not committed (CI)",
)

GOLDEN_CASES = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


def case_with(*concepts: str, sex: str = "M", age: int = 55) -> PatientCase:
    """A case holding exactly these concepts as present findings."""
    return PatientCase(
        case_id="T-001",
        age=age,
        sex=sex,
        findings=[
            Finding(concept_id=c, label=BY_CONCEPT[c].label if c in BY_CONCEPT else c)
            for c in concepts
        ],
    )


def codes_of(tokens: tuple[str, ...]) -> list[str]:
    return [t.partition(TOKEN_SEP)[0] for t in tokens]


def dropped_reasons(case: PatientCase, **kw) -> dict[str, DropReason]:
    return {d.concept_id: d.reason for d in tokens_for_case(case, **kw).dropped}


# --------------------------------------------------------------------------- #
# The judgement tables
# --------------------------------------------------------------------------- #


class TestTables:
    def test_every_concept_needing_an_answer_has_one(self):
        """A new crosswalk entry with answers must not silently become unrepresentable."""
        needed = {
            e.concept_id
            for e in CROSSWALK
            if e.pattern is not None and e.pattern.values and e.match is not Match.RELATED
        }
        assert needed - set(REPRESENTATIVE) == set(), "missing from REPRESENTATIVE"

    def test_every_scale_concept_has_a_value(self):
        needed = {
            e.concept_id
            for e in CROSSWALK
            if e.pattern is not None
            and e.pattern.min_ordinal is not None
            and e.match is not Match.RELATED
        }
        assert needed - set(ORDINAL_REPRESENTATIVE) == set(), "missing from ORDINAL_REPRESENTATIVE"

    def test_no_table_entry_is_unused(self):
        """A concept removed from the crosswalk must not leave a stale answer behind."""
        for concept in (*REPRESENTATIVE, *ORDINAL_REPRESENTATIVE):
            assert concept in BY_CONCEPT, f"{concept} is no longer in the crosswalk"

    def test_chosen_answers_satisfy_their_own_pattern(self):
        """The answer must actually match the pattern it stands for, sides included."""
        for concept in REPRESENTATIVE:
            tokens = tokens_for_case(case_with(concept)).tokens
            pattern = BY_CONCEPT[concept].pattern
            assert pattern is not None
            chosen = {
                t.partition(TOKEN_SEP)[2] for t in tokens if t.startswith(pattern.code + TOKEN_SEP)
            }
            assert chosen <= pattern.values, f"{concept}: {chosen} is not in its pattern"
            assert chosen, f"{concept}: produced no answer"

    def test_scale_values_sit_inside_the_qualifying_range(self):
        for concept, value in ORDINAL_REPRESENTATIVE.items():
            pattern = BY_CONCEPT[concept].pattern
            assert pattern is not None and pattern.min_ordinal is not None
            assert value >= pattern.min_ordinal, f"{concept}: {value} does not qualify"
            assert value > pattern.min_ordinal, (
                f"{concept}: {value} sits exactly on the threshold, so a hand-written case would "
                "land on the model's decision boundary"
            )

    def test_bilateral_swelling_needs_both_sides(self):
        """Side.BOTH is the one pattern a single answer cannot satisfy."""
        assert len(REPRESENTATIVE["SYM:leg_swelling_bilateral"]) == 2

    @needs_real_data
    def test_chosen_answers_are_allowed_by_the_release(self):
        """A value DDXPlus does not allow would make the encoder refuse the whole row."""
        specs = load_evidence_specs(REAL_EVIDENCES)
        for concept, values in REPRESENTATIVE.items():
            spec = specs[BY_CONCEPT[concept].pattern.code]
            for value in values:
                assert value in spec.possible_values, f"{concept}: {value} is not an allowed answer"
                assert value != spec.default_value, f"{concept}: {value} is the 'no' answer"
        for concept, value in ORDINAL_REPRESENTATIVE.items():
            spec = specs[BY_CONCEPT[concept].pattern.code]
            assert str(value) in spec.possible_values, f"{concept}: {value} is off the scale"


# --------------------------------------------------------------------------- #
# The round trip
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    @pytest.mark.parametrize(
        "concept",
        sorted(
            e.concept_id
            for e in CROSSWALK
            if e.pattern is not None
            and e.match in (Match.EXACT, Match.CLOSE, Match.NARROWER)
            and e.concept_id not in {"SYM:leg_swelling"}  # implied by both swelling concepts
        ),
    )
    def test_each_concept_comes_back(self, concept: str):
        """concept -> tokens -> concept, through the crosswalk's independently written forward
        direction. A wrong representative answer fails here."""
        tokens = tokens_for_case(case_with(concept)).tokens
        assert concept in concepts_from_evidences(tokens), f"{tokens} does not imply {concept}"

    @pytest.mark.parametrize("golden", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
    def test_a_golden_case_loses_only_what_ddxplus_cannot_express(self, golden: dict):
        case = PatientCase(**golden["case"])
        tokens = tokens_for_case(case)
        recovered = set(concepts_from_evidences(tokens.tokens))
        present = {
            f.concept_id
            for f in [*case.findings, *case.risk_factors]
            if f.assertion is Assertion.PRESENT
        }
        # A BROADER concept is not recoverable by construction: the answer covers more than the
        # concept, so it cannot imply it back. Everything else must return.
        expected_back = {
            c
            for c in present
            if c in BY_CONCEPT and BY_CONCEPT[c].match in (Match.EXACT, Match.CLOSE, Match.NARROWER)
        }
        assert expected_back <= recovered, f"lost: {sorted(expected_back - recovered)}"
        # Whatever did not come back is accounted for, either as dropped or as broader.
        unaccounted = present - recovered - {d.concept_id for d in tokens.dropped}
        assert all(BY_CONCEPT[c].match is Match.BROADER for c in unaccounted), unaccounted

    def test_expanding_a_case_first_changes_nothing(self):
        """expand_case() adds DDX: question ids, which name no answer, so they add no tokens."""
        case = PatientCase(**GOLDEN_CASES[0]["case"])
        assert tokens_for_case(expand_case(case)).tokens == tokens_for_case(case).tokens

    def test_the_result_is_deterministic(self):
        case = PatientCase(**GOLDEN_CASES[0]["case"])
        assert tokens_for_case(case).tokens == tokens_for_case(case).tokens


# --------------------------------------------------------------------------- #
# What the tokens say
# --------------------------------------------------------------------------- #


class TestTokens:
    def test_a_yes_no_concept_is_its_own_token(self):
        assert tokens_for_case(case_with("SYM:diaphoresis")).tokens == ("E_50",)

    def test_an_answer_implies_its_parent_question(self):
        """Pain somewhere (E_53) follows from answering where the pain is (E_55)."""
        assert tokens_for_case(case_with("SYM:chest_pain")).tokens == ("E_53", "E_55_@_V_101")

    def test_a_scale_answers_once(self):
        tokens = tokens_for_case(case_with("SYM:severe_pain", "SYM:sudden_onset")).tokens
        assert codes_of(tokens).count("E_56") == 1
        assert codes_of(tokens).count("E_59") == 1

    def test_two_concepts_on_one_question_both_answer_it(self):
        """E_55 is multi-choice, so chest pain and back pain are not in conflict."""
        tokens = tokens_for_case(case_with("SYM:chest_pain", "SYM:back_pain")).tokens
        assert tokens == ("E_53", "E_55_@_V_39", "E_55_@_V_101")

    def test_tokens_are_in_evidence_code_order(self):
        tokens = tokens_for_case(PatientCase(**GOLDEN_CASES[0]["case"])).tokens
        numbers = [int(c.split("_")[1]) for c in codes_of(tokens)]
        assert numbers == sorted(numbers)

    def test_every_token_names_the_concept_it_came_from(self):
        tokens = tokens_for_case(PatientCase(**GOLDEN_CASES[0]["case"]))
        assert set(tokens.tokens) == {
            t for produced in tokens.by_concept.values() for t in produced
        }

    def test_the_codes_filter_drops_rather_than_raises(self):
        """The encoder refuses a whole row over one unknown evidence, so filter first."""
        tokens = tokens_for_case(case_with("SYM:diaphoresis", "RF:diabetes"), codes=["E_50"])
        assert tokens.tokens == ("E_50",)
        assert dropped_reasons(case_with("RF:diabetes"), codes=["E_50"]) == {
            "RF:diabetes": DropReason.NOT_ENCODED
        }


class TestNarrower:
    """Decision A-8: admit narrower matches, and say which tokens they produced."""

    def test_narrower_concepts_are_admitted_by_default(self):
        assert tokens_for_case(case_with("SYM:sudden_onset")).tokens == ("E_53", "E_59_@_9")

    def test_excluding_them_drops_the_discriminators(self):
        """What the project loses if the team rejects A-8."""
        case = case_with("SYM:sudden_onset", "SYM:exertional", "SYM:relieved_by_rest")
        assert tokens_for_case(case, include_narrower=False).tokens == ()
        assert set(dropped_reasons(case, include_narrower=False).values()) == {
            DropReason.NARROWER_EXCLUDED
        }

    def test_only_tokens_no_sound_match_produced_count_as_inferred(self):
        """E_53 follows soundly from chest pain, so sudden onset does not make it an inference."""
        tokens = tokens_for_case(case_with("SYM:chest_pain", "SYM:sudden_onset"))
        assert tokens.narrower == ("E_59_@_9",)
        assert "E_53" in tokens.tokens


class TestNothingIsLostSilently:
    def test_a_denial_is_recorded_but_not_encoded(self):
        """The encoder has no 'asked' channel yet: 0 means denied and never-asked alike (R2)."""
        case = PatientCase(
            case_id="T-002",
            age=40,
            sex="F",
            findings=[
                Finding(concept_id="SYM:chest_pain", label="Chest pain"),
                Finding(
                    concept_id="SYM:diaphoresis", label="Diaphoresis", assertion=Assertion.ABSENT
                ),
            ],
        )
        tokens = tokens_for_case(case)
        assert tokens.denied == ("SYM:diaphoresis",)
        assert "E_50" not in tokens.tokens

    def test_gc004_denies_three_findings_the_model_cannot_see(self):
        """The case written to guard against over-flagging is the one that loses the most."""
        gc004 = next(c for c in GOLDEN_CASES if c["id"] == "GC-004")
        assert len(tokens_for_case(PatientCase(**gc004["case"])).denied) == 3

    def test_a_concept_ddxplus_cannot_express_is_dropped_with_its_reason(self):
        assert dropped_reasons(case_with("SYM:interarm_bp_difference")) == {
            "SYM:interarm_bp_difference": DropReason.NO_DDXPLUS_EQUIVALENT
        }

    def test_a_related_only_concept_draws_no_inference(self):
        assert dropped_reasons(case_with("SYM:irregular_pulse")) == {
            "SYM:irregular_pulse": DropReason.RELATED_ONLY
        }

    def test_an_unknown_concept_is_dropped_not_raised(self):
        assert dropped_reasons(case_with("SYM:not_a_real_concept")) == {
            "SYM:not_a_real_concept": DropReason.NOT_IN_CROSSWALK
        }

    def test_an_unknown_assertion_is_dropped(self):
        case = PatientCase(
            case_id="T-003",
            age=40,
            sex="F",
            findings=[
                Finding(
                    concept_id="SYM:diaphoresis", label="Diaphoresis", assertion=Assertion.UNKNOWN
                )
            ],
        )
        assert dropped_reasons(case) == {"SYM:diaphoresis": DropReason.UNKNOWN_ASSERTION}

    def test_a_ddx_question_id_names_no_answer(self):
        case = case_with("DDX:E_55")
        assert tokens_for_case(case).tokens == ()
        assert dropped_reasons(case) == {"DDX:E_55": DropReason.ALREADY_A_QUESTION}

    def test_the_summary_line_counts_everything(self):
        summary = tokens_for_case(PatientCase(**GOLDEN_CASES[3]["case"])).summary()
        assert "denied" in summary and "dropped" in summary


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


class TestEncodeCase:
    def test_sex_other_has_no_row(self):
        """DDXPlus records M or F only. The pipeline degrades rather than guessing."""

        class FakeEncoder:
            codes = ("E_50",)

        case = case_with("SYM:diaphoresis")
        with pytest.raises(UnrepresentableCase, match="Sex.OTHER"):
            encode_case(case.model_copy(update={"sex": Sex.OTHER}), FakeEncoder())

    @needs_real_data
    def test_a_golden_case_encodes_to_a_row_the_model_accepts(self):
        from src.ml.features import EvidenceEncoder

        encoder = EvidenceEncoder.from_release(REAL_EVIDENCES, REAL_CONDITIONS)
        row, tokens = encode_case(PatientCase(**GOLDEN_CASES[0]["case"]), encoder)
        assert row.shape == (len(encoder.feature_names),)
        assert row.dtype.name == "float32"
        assert row[0] == 58.0 and row[1] == 0.0  # age; sex=F is 0 for a man
        assert (row != 0).sum() > len(tokens.tokens) - 2  # a question and its answer each score

    @needs_real_data
    def test_every_golden_case_encodes(self):
        from src.ml.features import EvidenceEncoder

        encoder = EvidenceEncoder.from_release(REAL_EVIDENCES, REAL_CONDITIONS)
        for golden in GOLDEN_CASES:
            row, tokens = encode_case(PatientCase(**golden["case"]), encoder)
            assert tokens.tokens, f"{golden['id']} produced no tokens at all"
            assert row.any()
