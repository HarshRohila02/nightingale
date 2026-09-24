"""Tests for the crosswalk between hand-authored concepts and DDXPlus evidence (task 1b).

The synthetic tests pin down the inference rules: which way each match type lets a finding
travel, and why a denial rarely carries over. The tests that read the real DDXPlus release and
the real knowledge graph skip when data/ is absent, as it is in CI.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.contracts import Assertion, EvidenceRole, PatientCase
from src.ddxplus import EvidenceSpec
from src.medical_kg.bodhi_s import RISK_FACTORS as BODHI_RISK_FACTORS
from src.medical_kg.bodhi_s import SYMPTOMS as BODHI_SYMPTOMS
from src.medical_kg.crosswalk import (
    BY_CONCEPT,
    CROSSWALK,
    DERIVED_SOURCE,
    LEFT_LEG,
    PARENT_QUESTION,
    RIGHT_LEG,
    SIDE_OF,
    SUDDEN_ONSET_MIN,
    CrosswalkEntry,
    Match,
    Pattern,
    Side,
    case_from_ddxplus,
    concepts_from_evidences,
    expand_case,
    validate_crosswalk,
)
from src.medical_kg.hand_authored import FACTS as HAND_AUTHORED_FACTS
from src.medical_kg.loader import EdgeSource, KGEdge, KGNode, KnowledgeGraph, NodeType, Relation
from src.medical_kg.networkx_store import NetworkXGraphStore
from src.pipeline import DiagnosisPipeline
from src.reasoning.red_flags import RULES, evaluate_red_flags
from src.stubs import STUB_SYMPTOM_MAP, ConstantRanker, EmptyRetriever, TemplateExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"
RELEASE_PATH = REPO_ROOT / "data" / "raw" / "ddxplus" / "release_evidences.json"
KG_FILES = (
    REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_evidences.json",
)
needs_release = pytest.mark.skipif(not RELEASE_PATH.exists(), reason="data/ is not committed (CI)")
needs_kg = pytest.mark.skipif(
    not all(p.exists() for p in KG_FILES), reason="data/ is not committed (CI)"
)

GOLDEN = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


def _case(present=(), absent=(), risk_factors=()) -> PatientCase:
    findings = [{"concept_id": c, "label": f"label {c}"} for c in present]
    findings += [{"concept_id": c, "label": f"label {c}", "assertion": "absent"} for c in absent]
    return PatientCase(
        case_id="T-XW",
        age=50,
        sex="M",
        findings=findings,
        risk_factors=[{"concept_id": c, "label": f"label {c}"} for c in risk_factors],
    )


def _derived(case: PatientCase) -> dict[str, str]:
    """{DDX id: assertion} for the findings the crosswalk added."""
    return {
        f.concept_id: f.assertion.value
        for f in [*case.findings, *case.risk_factors]
        if f.source == DERIVED_SOURCE
    }


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def _concepts_in_use() -> set[str]:
    used = {c for rule in RULES for c in rule.concepts}
    used |= {c for concepts in STUB_SYMPTOM_MAP.values() for c in concepts}
    used |= {fact.concept for fact in HAND_AUTHORED_FACTS}
    for table in (BODHI_SYMPTOMS, BODHI_RISK_FACTORS):
        used |= {c for mapping in table.values() for c in mapping.concepts}
    used |= {
        f["concept_id"]
        for golden in GOLDEN
        for key in ("findings", "risk_factors")
        for f in golden["case"].get(key, [])
    }
    return used


class TestTable:
    def test_every_concept_in_use_is_mapped(self):
        """A red-flag rule on an unmapped concept could never fire on a DDXPlus patient."""
        missing = sorted(_concepts_in_use() - set(BY_CONCEPT))
        assert not missing, f"add these to CROSSWALK (Match.NONE if DDXPlus has nothing): {missing}"

    def test_concepts_are_unique_and_namespaced(self):
        ids = [e.concept_id for e in CROSSWALK]
        assert len(ids) == len(set(ids))
        assert all(i.startswith(("SYM:", "RF:")) for i in ids)

    def test_a_pattern_exists_exactly_when_ddxplus_has_something(self):
        for entry in CROSSWALK:
            assert (entry.pattern is None) == (entry.match is Match.NONE), entry.concept_id

    def test_no_pattern_counts_a_no_or_na_answer(self):
        for entry in CROSSWALK:
            if entry.pattern is not None:
                assert not entry.pattern.values & {"V_11", "V_123", "V_10"}, entry.concept_id

    def test_legs_are_paired_and_every_side_pattern_value_has_a_side(self):
        assert len(RIGHT_LEG) == len(LEFT_LEG) and not RIGHT_LEG & LEFT_LEG
        for entry in CROSSWALK:
            if entry.pattern is not None and entry.pattern.side is not None:
                assert entry.pattern.values <= SIDE_OF.keys(), entry.concept_id

    def test_the_tearing_descriptor_is_mapped_from_the_french(self):
        """'déchirante' is the aortic-dissection descriptor that the English calls
        'heartbreaking' (CLAUDE.md gotcha)."""
        tearing = BY_CONCEPT["SYM:pain_character_tearing"]
        assert tearing.pattern == Pattern("E_54", values=frozenset({"V_71"}))
        assert tearing.match is Match.EXACT

    def test_a_concept_without_ddxplus_evidence_is_recorded_not_dropped(self):
        assert BY_CONCEPT["SYM:interarm_bp_difference"].match is Match.NONE


# --------------------------------------------------------------------------- #
# Validation against the DDXPlus release
# --------------------------------------------------------------------------- #


def _release() -> dict[str, dict[str, Any]]:
    """A tiny release_evidences.json with every question PARENT_QUESTION names."""

    def question(data_type="B", values=(), default=0, parent=None, meanings=None) -> dict:
        return {
            "data_type": data_type,
            "possible-values": list(values),
            "default_value": default,
            "value_meaning": meanings or {},
            "code_question": parent,
        }

    release = {code: question(parent=code) for code in ("E_53", "E_151", "E_220")}
    release |= {child: question(parent=parent) for child, parent in PARENT_QUESTION.items()}
    release["E_54"] = question("M", ["V_11", "V_71", "V_183"], "V_11", "E_53")
    release["E_59"] = question("C", range(11), 0, "E_53")
    release["E_152"] = question(
        "M",
        ["V_123", "V_119", "V_120"],
        "V_123",
        "E_151",
        {"V_119": {"fr": "mollet(D)"}, "V_120": {"fr": "mollet(G)"}},
    )
    return release


def _problems(entry: CrosswalkEntry, release: dict | None = None) -> list[str]:
    return validate_crosswalk(release or _release(), entries=[entry])


def _x(
    pattern: Pattern | None, match: Match = Match.EXACT, concept: str = "SYM:x"
) -> CrosswalkEntry:
    return CrosswalkEntry(concept, "X", match, pattern)


class TestValidation:
    def test_a_sound_table_has_no_problems(self):
        entries = [
            CrosswalkEntry("SYM:a", "A", Match.EXACT, Pattern("E_220")),
            CrosswalkEntry("SYM:b", "B", Match.EXACT, Pattern("E_54", frozenset({"V_71"}))),
            CrosswalkEntry("SYM:c", "C", Match.NARROWER, Pattern("E_59", min_ordinal=8)),
            CrosswalkEntry(
                "SYM:d", "D", Match.EXACT, Pattern("E_152", frozenset({"V_119"}), side=Side.ONE)
            ),
            CrosswalkEntry("SYM:e", "E", Match.NONE),
        ]
        assert validate_crosswalk(_release(), entries=entries) == []

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            (_x(Pattern("E_999")), "not a DDXPlus question"),
            (_x(Pattern("E_54", frozenset({"V_7"}))), "not answers"),
            (_x(Pattern("E_54", frozenset({"V_11"}))), "'no' or NA"),
            (_x(Pattern("E_54")), "needs the answers"),
            (_x(Pattern("E_220", frozenset({"V_1"}))), "yes/no"),
            (_x(Pattern("E_59", frozenset({"V_1"}))), "0-10 scale"),
            (_x(Pattern("E_59", min_ordinal=0)), "not a positive"),
            (_x(None), "pattern is required"),
            (_x(Pattern("E_220"), Match.NONE), "pattern is required"),
            (_x(None, Match.NONE, "COND:x"), "SYM:* or RF:*"),
        ],
    )
    def test_each_kind_of_mistake_is_reported(self, entry, expected):
        problems = _problems(entry)
        assert any(expected in p for p in problems), problems

    def test_a_value_on_the_wrong_side_is_reported(self):
        release = _release()
        release["E_152"]["value_meaning"]["V_119"] = {"fr": "mollet(G)"}  # mislabelled
        entry = CrosswalkEntry(
            "SYM:x", "X", Match.EXACT, Pattern("E_152", frozenset({"V_119"}), side=Side.ONE)
        )
        assert any("not on side R" in p for p in _problems(entry, release))

    def test_duplicates_and_an_unlisted_parent_are_reported(self):
        release = _release()
        release["E_220"]["code_question"] = "E_53"  # now a follow-up PARENT_QUESTION lacks
        entry = CrosswalkEntry("SYM:x", "X", Match.EXACT, Pattern("E_220"))
        problems = validate_crosswalk(release, entries=[entry, entry])
        assert any("duplicate" in p for p in problems)
        assert any("add it to PARENT_QUESTION" in p for p in problems)

    def test_a_parent_link_the_release_does_not_have_is_reported(self):
        release = _release()
        release["E_152"]["code_question"] = "E_152"
        problems = validate_crosswalk(release, entries=[])
        assert problems == ["PARENT_QUESTION: E_152 does not follow up E_151 in the release"]


# --------------------------------------------------------------------------- #
# Hand-authored case -> knowledge graph
# --------------------------------------------------------------------------- #


class TestExpandCase:
    def test_a_present_concept_adds_its_question_and_the_question_it_follows(self):
        case = expand_case(_case(present=["SYM:pain_character_pressure"]))
        assert _derived(case) == {"DDX:E_53": "present", "DDX:E_54": "present"}

    def test_derived_findings_are_labelled_by_their_question_and_traceable(self):
        labels = {"DDX:E_54": "Characterize your pain:"}
        case = expand_case(_case(present=["SYM:pain_character_pressure"]), labels)
        derived = {f.concept_id: f for f in case.findings if f.source == DERIVED_SOURCE}
        assert derived["DDX:E_54"].label == "Characterize your pain:", "never 'Pressure-type pain'"
        assert derived["DDX:E_53"].label == "DDX:E_53", "unlabelled questions keep their id"
        assert derived["DDX:E_54"].qualifiers == {"crosswalk_from": "SYM:pain_character_pressure"}

    def test_the_original_findings_stay_first_and_unchanged(self):
        original = _case(present=["SYM:chest_pain"], absent=["SYM:diaphoresis"])
        expanded = expand_case(original)
        assert expanded.findings[: len(original.findings)] == original.findings
        assert (
            original.findings
            == _case(present=["SYM:chest_pain"], absent=["SYM:diaphoresis"]).findings
        )

    def test_a_narrower_answer_is_not_implied_by_the_concept(self):
        """Exertional pain need not be relieved by rest, which E_218 also asks."""
        assert _derived(expand_case(_case(present=["SYM:exertional"]))) == {}

    def test_a_denial_carries_over_to_a_yes_no_question_no_broader_than_the_concept(self):
        case = expand_case(_case(absent=["SYM:exertional", "SYM:diaphoresis"]))
        assert _derived(case) == {"DDX:E_50": "absent", "DDX:E_218": "absent"}

    def test_denying_one_answer_does_not_deny_the_question(self):
        """No radiation to the jaw or arm says nothing about radiation to the back."""
        assert _derived(expand_case(_case(absent=["SYM:radiation_jaw_arm"]))) == {}

    def test_a_broader_answer_is_implied_but_its_denial_is_not(self):
        assert _derived(expand_case(_case(present=["SYM:orthopnoea"]))) == {"DDX:E_217": "present"}
        assert _derived(expand_case(_case(absent=["SYM:orthopnoea"]))) == {}

    def test_related_and_unmapped_concepts_add_nothing(self):
        case = _case(present=["SYM:irregular_pulse", "SYM:interarm_bp_difference", "SYM:new"])
        assert _derived(expand_case(case)) == {}

    def test_risk_factors_stay_risk_factors(self):
        case = expand_case(_case(risk_factors=["RF:diabetes"]))
        assert [f.concept_id for f in case.risk_factors] == ["RF:diabetes", "DDX:E_69"]
        assert case.findings == []

    def test_a_contradiction_is_left_out_and_logged(self, caplog):
        """Orthopnoea implies E_217; denying worse-lying-flat denies it."""
        case = _case(present=["SYM:orthopnoea"], absent=["SYM:worse_lying_flat"])
        with caplog.at_level(logging.WARNING, logger="src.medical_kg.crosswalk"):
            expanded = expand_case(case)
        assert "DDX:E_217" not in _derived(expanded)
        assert "DDX:E_217" in caplog.text

    def test_expanding_twice_changes_nothing(self):
        once = expand_case(_case(present=["SYM:chest_pain"], absent=["SYM:exertional"]))
        assert expand_case(once) == once

    def test_a_question_the_case_already_has_is_not_duplicated(self):
        case = expand_case(_case(present=["SYM:breathlessness", "DDX:E_66"]))
        assert [f.concept_id for f in case.findings].count("DDX:E_66") == 1


# --------------------------------------------------------------------------- #
# DDXPlus -> concept
# --------------------------------------------------------------------------- #


class TestConceptsFromEvidences:
    def test_binary_and_categorical_answers(self):
        concepts = concepts_from_evidences(["E_53", "E_55_@_V_101", "E_54_@_V_71", "E_220"])
        assert concepts == ["SYM:chest_pain", "SYM:pain_character_tearing", "SYM:pleuritic"]

    def test_no_answers_and_na_answers_express_nothing(self):
        """E_57_@_V_123 is 'radiates nowhere', listed for many patients (EXP-013)."""
        assert concepts_from_evidences(["E_57_@_V_123", "E_54_@_V_11", "E_204_@_V_10"]) == []

    def test_the_sudden_onset_cut_off(self):
        below = f"E_59_@_{SUDDEN_ONSET_MIN - 1}"
        at = f"E_59_@_{SUDDEN_ONSET_MIN}"
        assert "SYM:sudden_onset" not in concepts_from_evidences([below])
        assert "SYM:sudden_onset" in concepts_from_evidences([at])

    @pytest.mark.parametrize(
        ("tokens", "expected"),
        [
            (["E_152_@_V_119"], ["SYM:leg_swelling", "SYM:leg_swelling_unilateral"]),
            (
                ["E_152_@_V_119", "E_152_@_V_34"],
                ["SYM:leg_swelling", "SYM:leg_swelling_unilateral"],
            ),
            (
                ["E_152_@_V_119", "E_152_@_V_120"],
                ["SYM:leg_swelling", "SYM:leg_swelling_bilateral"],
            ),
            (["E_152_@_V_123"], []),
        ],
    )
    def test_leg_swelling_on_one_side_or_both(self, tokens, expected):
        assert concepts_from_evidences(tokens) == expected

    def test_a_broader_or_related_answer_never_implies_the_concept(self):
        """E_217 is also GERD's and pulmonary edema's; E_164 is felt, not examined."""
        concepts = concepts_from_evidences(["E_217", "E_164", "E_175"])
        assert concepts == ["SYM:worse_lying_flat"]

    def test_a_malformed_token_fails_loudly(self):
        with pytest.raises(ValueError):
            concepts_from_evidences(["E_53", "fever"])

    def test_the_red_flag_rules_can_now_run_on_a_ddxplus_patient(self):
        pe_like = ["E_53", "E_55_@_V_55", "E_220", "E_66", "E_110", "E_151", "E_152_@_V_119"]
        case = _case(present=concepts_from_evidences(pe_like))
        assert "COND:pulmonary_embolism" in evaluate_red_flags(case)


class _Record(Mapping[str, Any]):
    """A parquet row that records which columns were read."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.read: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def test_case_from_ddxplus_uses_inputs_only_and_files_risk_factors():
    record = _Record(
        {
            "case_id": "ddxplus-validate-0000042",
            "age": 61,
            "sex": "M",
            "evidences": ["E_53", "E_55_@_V_101", "E_79", "E_57_@_V_123"],
            "positive_codes": ["E_53", "E_55", "E_79"],
            "label_condition_id": "COND:nstemi_stemi",
            "label_differential": [],
        }
    )
    specs = {
        "E_53": EvidenceSpec("E_53", "B", None, False),
        "E_55": EvidenceSpec("E_55", "M", "V_123", False),
        "E_79": EvidenceSpec("E_79", "B", None, True),
    }
    case = case_from_ddxplus(record, specs, labels={"DDX:E_79": "Do you smoke cigarettes?"})
    assert not any(key.startswith("label_") for key in record.read), "labels are never inputs"
    assert [f.concept_id for f in case.findings] == ["DDX:E_53", "DDX:E_55", "SYM:chest_pain"]
    assert [f.concept_id for f in case.risk_factors] == ["DDX:E_79", "RF:smoking"]
    assert case.risk_factors[0].label == "Do you smoke cigarettes?"
    assert {f.source for f in case.findings} == {"ddxplus", DERIVED_SOURCE}
    assert all(f.assertion is Assertion.PRESENT for f in [*case.findings, *case.risk_factors])


# --------------------------------------------------------------------------- #
# The pipeline on a crosswalk-expanded case
# --------------------------------------------------------------------------- #

PE = "COND:pulmonary_embolism"
GERD = "COND:gerd"


@pytest.fixture
def small_graph() -> NetworkXGraphStore:
    labels = {
        "DDX:E_53": "Pain related to the consultation?",
        "DDX:E_54": "Characterize your pain:",
        "DDX:E_66": "Short of breath?",
        "DDX:E_220": "Pain worse on deep breathing?",
        "DDX:E_110": "Immobilised for 3 days or more?",
        "DDX:E_215": "Worse after eating?",
    }
    nodes = {
        PE: KGNode(PE, NodeType.CONDITION, "Pulmonary embolism"),
        GERD: KGNode(GERD, NodeType.CONDITION, "GERD"),
    }
    nodes |= {cid: KGNode(cid, NodeType.SYMPTOM, label) for cid, label in labels.items()}
    edges = [
        KGEdge(PE, cid, Relation.HAS_SYMPTOM, EdgeSource.DDXPLUS)
        for cid in ("DDX:E_53", "DDX:E_54", "DDX:E_66", "DDX:E_220", "DDX:E_110")
    ]
    edges += [
        KGEdge(GERD, cid, Relation.HAS_SYMPTOM, EdgeSource.DDXPLUS)
        for cid in ("DDX:E_53", "DDX:E_54", "DDX:E_215")
    ]
    return NetworkXGraphStore(KnowledgeGraph(nodes=nodes, edges=edges))


def test_the_pipeline_scores_a_hand_authored_case_on_the_ddxplus_graph(small_graph):
    pipeline = DiagnosisPipeline(
        ConstantRanker(), small_graph, EmptyRetriever(), TemplateExplainer()
    )
    labels = dict(small_graph.graph.nodes(data="label"))
    raw = _case(present=["SYM:pleuritic", "SYM:breathlessness", "SYM:recent_immobilisation"])

    assert pipeline.run(raw).candidates[0].kg_score == 0.0, "SYM ids alone never meet DDX ids"

    expanded = expand_case(raw, labels)
    result = pipeline.run(expanded)
    pe = next(c for c in result.candidates if c.condition_id == PE)
    assert result.candidates[0].condition_id == PE
    graph_scores = small_graph.score_by_connectivity(expanded)
    assert max(graph_scores, key=graph_scores.get) == PE, "the graph itself ranks PE first"
    assert {a.finding_id for a in pe.assessments if a.role is EvidenceRole.SUPPORTING} == {
        "DDX:E_66",
        "DDX:E_110",
        "DDX:E_220",
    }
    assert "COND:pulmonary_embolism" in {c.condition_id for c in result.candidates if c.red_flag}


def test_a_path_through_a_question_names_the_question_not_the_answer(small_graph):
    """Burning pain matches GERD's E_54 'characterize your pain'. The path must not claim that
    GERD presents with *pressure-type* pain just because the same question was asked."""
    labels = dict(small_graph.graph.nodes(data="label"))
    case = expand_case(_case(present=["SYM:pain_character_pressure"]), labels)
    paths = small_graph.paths_for(case, GERD)
    assert [p.path[0] for p in paths] == [
        "Pain related to the consultation?",
        "Characterize your pain:",
    ]


def test_the_check_script_exits_1_when_the_release_does_not_match(tmp_path):
    (tmp_path / "release_evidences.json").write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_crosswalk.py")]
        + ["--raw-dir", str(tmp_path), "--interim", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 1
    summary = json.loads((tmp_path / "crosswalk_check.json").read_text("utf-8"))
    assert summary["entries"] == len(CROSSWALK)
    assert any("E_55 is not a DDXPlus question" in p for p in summary["problems"])


# --------------------------------------------------------------------------- #
# The real release and the real graph (local only)
# --------------------------------------------------------------------------- #


@needs_release
def test_the_crosswalk_matches_the_real_ddxplus_release():
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    assert validate_crosswalk(release) == []


@needs_kg
@pytest.mark.golden
@pytest.mark.parametrize("golden", GOLDEN, ids=[c["id"] for c in GOLDEN])
def test_golden_case_on_the_real_graph(golden):
    """The golden expectations, with the real graph in place of the stub (docs/02 §5.2).

    They hold because red flags rank first. On graph score alone, GC-001's MI ranks fifth, so
    2a must not weaken the red flags without fixing the scoring.
    """
    store = NetworkXGraphStore.from_files(*KG_FILES)
    pipeline = DiagnosisPipeline(ConstantRanker(), store, EmptyRetriever(), TemplateExplainer())
    labels = dict(store.graph.nodes(data="label"))
    result = pipeline.run(expand_case(PatientCase(**golden["case"]), labels))

    expect = golden["expect"]
    top_ids = [c.condition_id for c in result.candidates[: expect.get("top_k", 3)]]
    flagged = {c.condition_id for c in result.candidates if c.red_flag}
    for required in expect.get("must_include", []):
        assert required in top_ids, f"{golden['id']}: {required} not in {top_ids}"
    for required in expect.get("red_flag_for", []):
        assert required in flagged
    if expect.get("no_red_flag"):
        assert not flagged
    assert "graph_backend" not in result.degraded_components
