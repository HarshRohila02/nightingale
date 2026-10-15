"""Tests for "what to ask next" (task 3d; FR-5.5): the cited tests and the graph-ranked questions.

Two halves, as docs/02 §5.3 describes them:

* **The tests to consider** (src/reasoning/investigations.py): a hand-authored table. It must cover
  every condition, cite a guideline for every line, say "consider", and never recommend a
  treatment or name a drug (FR-6.5); the pipeline puts its lines in ``suggested_next_tests``.
* **The questions to ask** (src/reasoning/next_questions.py): expected information gain over the
  knowledge graph's posterior. Checked on hand-computed examples, against the scorer it must agree
  with, and for its properties: never negative, 0 for a question that splits nothing, never asking
  what is recorded, deterministic, nothing from a degraded graph.

The synthetic tests run everywhere; those on the real graph skip when data/ is absent, as in CI.
"""

from __future__ import annotations

import importlib.util
import math
import random
import re
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.view import next_steps
from src.conditions import BY_ID, CONDITIONS
from src.contracts import (
    Assertion,
    Candidate,
    DiagnosisResult,
    Finding,
    PatientCase,
)
from src.medical_kg.crosswalk import BY_CONCEPT, Match, expand_case
from src.medical_kg.loader import (
    EdgeSource,
    KGEdge,
    KGNode,
    KnowledgeGraph,
    NodeType,
    Relation,
)
from src.medical_kg.networkx_store import NetworkXGraphStore
from src.medical_kg.scoring import LikelihoodScorer
from src.pipeline import DiagnosisPipeline
from src.reasoning.investigations import (
    INVESTIGATIONS,
    SOURCES,
    TESTS_FOR_TOP,
    investigations_to_consider,
    suggested_tests,
)
from src.reasoning.next_questions import (
    EXAMINATION,
    QUESTIONS_SHOWN,
    SEPARATES_SHOWN,
    QuestionRanking,
    _expand,
    next_questions,
    settled,
)
from src.reasoning.safety import TREATMENT, recommends_treatment, safety_check
from src.stubs import ConstantRanker, EmptyRetriever, InMemoryGraphStore, TemplateExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERIM = REPO_ROOT / "data" / "interim"
KG_FILES = (INTERIM / "ddxplus_chestpain_conditions.json", INTERIM / "ddxplus_evidences.json")
BODHI_DIR = REPO_ROOT / "data" / "raw" / "bodhi_s"
needs_kg = pytest.mark.skipif(
    not all(p.exists() for p in KG_FILES), reason="data/ is not committed (CI)"
)
GOLDEN = yaml.safe_load(
    (REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml").read_text(encoding="utf-8")
)["cases"]
BY_GOLDEN = {g["id"]: g for g in GOLDEN}

MI, UA, PE, GERD, AD = (
    "COND:nstemi_stemi",
    "COND:unstable_angina",
    "COND:pulmonary_embolism",
    "COND:gerd",
    "COND:aortic_dissection",
)

# Words that would make a line advice rather than an investigation, beyond the safety layer's own
# list (src/reasoning/safety.py). Stricter on purpose: the table is short, and a test is cheap.
ADVICE = re.compile(
    r"\b(treat\w*|therap\w*|drugs?|medicat\w*|doses?|dosing|prescri\w*|administer\w*|"
    r"manage\w*|trial of|start\w*|give|given|initiat\w*|commenc\w*)\b",
    flags=re.IGNORECASE,
)


def case(present=(), denied=(), unknown=(), case_id="T") -> PatientCase:
    findings = [Finding(concept_id=c, label=c) for c in present]
    findings += [Finding(concept_id=c, label=c, assertion=Assertion.ABSENT) for c in denied]
    findings += [Finding(concept_id=c, label=c, assertion=Assertion.UNKNOWN) for c in unknown]
    return PatientCase(case_id=case_id, age=50, sex="M", findings=findings)


class Graph:
    """The least a graph store needs to be asked questions of: a likelihood scorer."""

    def __init__(self, expected: dict[str, dict[str, float]]) -> None:
        concepts = {k for edges in expected.values() for k in edges}
        self.scorer = LikelihoodScorer(expected, concepts)


def networkx_store(expected: dict[str, dict[str, float]]) -> NetworkXGraphStore:
    """A NetworkX store over a hand-made graph, closed under the crosswalk as the real one is."""
    nodes: dict[str, KGNode] = {}
    edges: list[KGEdge] = []
    for condition, concepts in expected.items():
        nodes[condition] = KGNode(condition, NodeType.CONDITION, BY_ID[condition].label)
        for concept, weight in concepts.items():
            nodes.setdefault(concept, KGNode(concept, NodeType.SYMPTOM, concept))
            edges.append(
                KGEdge(condition, concept, Relation.HAS_SYMPTOM, EdgeSource.HAND_AUTHORED, weight)
            )
    return NetworkXGraphStore(KnowledgeGraph(nodes=nodes, edges=edges))


# A miniature of the real graph's two vocabularies: DDXPlus questions for MI and GERD, answers for
# dissection, so the crosswalk closure (scoring.py) and the expansion both matter.
MIXED = {
    MI: {
        "DDX:E_53": 1.0,
        "DDX:E_54": 1.0,
        "DDX:E_55": 1.0,
        "DDX:E_57": 1.0,
        "DDX:E_218": 1.0,
        "DDX:E_50": 1.0,
        "DDX:E_66": 1.0,
    },
    GERD: {
        "DDX:E_53": 1.0,
        "DDX:E_54": 1.0,
        "DDX:E_55": 1.0,
        "DDX:E_215": 1.0,
        "DDX:E_217": 1.0,
        "SYM:pain_character_burning": 0.65,
    },
    PE: {
        "DDX:E_53": 1.0,
        "DDX:E_55": 1.0,
        "DDX:E_66": 1.0,
        "DDX:E_220": 1.0,
        "DDX:E_151": 1.0,
        "DDX:E_152": 1.0,
        "SYM:leg_swelling_unilateral": 0.35,
        "DDX:E_110": 1.0,
    },
    AD: {
        "SYM:chest_pain": 0.65,
        "SYM:pain_character_tearing": 0.35,
        "SYM:sudden_onset": 0.9,
        "SYM:radiation_back": 0.35,
        "SYM:interarm_bp_difference": 0.12,
    },
}


# --------------------------------------------------------------------------- #
# 1. The tests to consider: the table
# --------------------------------------------------------------------------- #


class TestTheTable:
    def test_every_condition_has_an_entry(self):
        assert set(INVESTIGATIONS) == {c.id for c in CONDITIONS}
        assert len(INVESTIGATIONS) == 14
        assert all(entry.tests for entry in INVESTIGATIONS.values())

    def test_every_line_cites_a_source_the_table_lists(self):
        for entry in INVESTIGATIONS.values():
            assert entry.sources, entry.condition_id
            assert set(entry.sources) <= set(SOURCES), entry.condition_id
            for key in entry.sources:
                assert f"{SOURCES[key].short}" in entry.text()
        cited = {key for entry in INVESTIGATIONS.values() for key in entry.sources}
        assert cited == set(SOURCES), "a listed source no entry cites"
        assert all(source.reference.strip() and source.short for source in SOURCES.values())

    def test_no_reference_awaits_checking(self):
        """The two citations written from memory (BTS 2023 pleural, WSES 2019 oesophageal) were
        checked against their publishers' pages on 2026-10-15 and found exact. A reference added
        unchecked must be flagged, and flagging one should be a deliberate change here."""
        assert {k for k, s in SOURCES.items() if s.verify} == set()

    def test_every_line_says_consider(self):
        for entry in INVESTIGATIONS.values():
            assert entry.text().startswith(f"{entry.label}: consider "), entry.text()

    @pytest.mark.parametrize("cid", sorted(INVESTIGATIONS))
    def test_no_treatment_or_drug_language(self, cid):
        """Investigations only (FR-6.5, docs/04 §3): neither the safety layer's patterns nor the
        stricter list above may find anything in a line."""
        entry = INVESTIGATIONS[cid]
        for text in (entry.text(), *entry.tests):
            assert not recommends_treatment(text), text
            assert not TREATMENT.search(text), text
            assert not ADVICE.search(text), f"{text!r}: {ADVICE.search(text).group(0)!r}"

    def test_the_safety_layer_would_keep_every_line(self):
        lines = [entry.text() for entry in INVESTIGATIONS.values()]
        result = DiagnosisResult(case_id="T", candidates=[], suggested_next_tests=lines)
        assert safety_check(result).suggested_next_tests == lines

    def test_the_safety_layer_would_remove_a_treatment_line(self):
        """The guard the table relies on: a line that advises a treatment never leaves."""
        line = "Possible NSTEMI / STEMI: consider aspirin 300 mg"
        result = DiagnosisResult(case_id="T", candidates=[], suggested_next_tests=[line])
        assert safety_check(result).suggested_next_tests == []


def ranked(*specs: tuple[str, bool]) -> list[Candidate]:
    return [
        Candidate(condition_id=cid, label=BY_ID[cid].label, red_flag=flag) for cid, flag in specs
    ]


class TestWhichConditionsGetTests:
    def test_the_top_three_and_every_flagged_candidate(self):
        """A flagged candidate below the top three still gets its tests."""
        candidates = ranked(
            (GERD, False),
            ("COND:pericarditis", False),
            ("COND:panic_attack", False),
            ("COND:stable_angina", False),
            (AD, True),
        )
        chosen = [e.condition_id for e in investigations_to_consider(candidates)]
        assert chosen == [GERD, "COND:pericarditis", "COND:panic_attack", AD]
        assert TESTS_FOR_TOP == 3

    def test_each_condition_once_and_none_when_nothing_ranks(self):
        candidates = ranked((PE, True), (PE, True), (MI, False))
        assert [e.condition_id for e in investigations_to_consider(candidates)] == [PE, MI]
        assert [e.condition_id for e in investigations_to_consider(candidates, top=0)] == [PE]
        assert suggested_tests([]) == []


# --------------------------------------------------------------------------- #
# 2. The tests in the pipeline
# --------------------------------------------------------------------------- #


def stub_pipeline(**kw) -> DiagnosisPipeline:
    return DiagnosisPipeline(
        ranker=ConstantRanker(),
        graph=InMemoryGraphStore(),
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
        **kw,
    )


def golden_case(gid: str) -> PatientCase:
    return PatientCase(**BY_GOLDEN[gid]["case"])


@pytest.mark.golden
class TestTheGoldenCasesGetSensibleTests:
    """On the stub graph, red flags on and off. Fix the component, never the expectation."""

    @pytest.mark.parametrize("flags", [True, False])
    def test_pulmonary_embolism(self, flags):
        tests = stub_pipeline(enable_red_flags=flags).run(golden_case("GC-002"))
        line = next(t for t in tests.suggested_next_tests if t.startswith("Pulmonary embolism"))
        for expected in ("Wells or revised Geneva", "D-dimer", "CT pulmonary angiography"):
            assert expected in line
        assert "ESC 2019 PE" in line

    @pytest.mark.parametrize("flags", [True, False])
    def test_aortic_dissection_gets_imaging(self, flags):
        tests = stub_pipeline(enable_red_flags=flags).run(golden_case("GC-003"))
        line = next(t for t in tests.suggested_next_tests if t.startswith("Aortic dissection"))
        assert "CT angiography of the aorta" in line and "echocardiography" in line

    @pytest.mark.parametrize("flags", [True, False])
    def test_acute_coronary_syndrome_gets_ecg_and_troponin(self, flags):
        tests = stub_pipeline(enable_red_flags=flags).run(golden_case("GC-001"))
        line = next(t for t in tests.suggested_next_tests if t.startswith("Possible NSTEMI"))
        assert "12-lead ECG" in line and "troponin" in line

    def test_reflux_gets_the_cardiac_cause_excluded_first(self):
        result = stub_pipeline().run(golden_case("GC-004"))
        assert not result.red_flags
        assert len(result.suggested_next_tests) == TESTS_FOR_TOP
        line = next(t for t in result.suggested_next_tests if t.startswith("GERD"))
        assert line.index("ECG") < line.index("endoscopy")

    @pytest.mark.parametrize("golden", GOLDEN, ids=[g["id"] for g in GOLDEN])
    def test_every_flagged_candidate_gets_its_tests(self, golden):
        result = stub_pipeline().run(PatientCase(**golden["case"]))
        flagged = [c.label for c in result.candidates if c.red_flag]
        for label in flagged:
            assert any(t.startswith(f"{label}: consider") for t in result.suggested_next_tests)
        top = [c.label for c in result.candidates[:TESTS_FOR_TOP]]
        assert [t.split(": consider")[0] for t in result.suggested_next_tests] == list(
            dict.fromkeys([*[c.label for c in result.candidates if c.red_flag], *top])
        )

    @pytest.mark.parametrize("golden", GOLDEN, ids=[g["id"] for g in GOLDEN])
    def test_the_golden_expectations_still_hold(self, golden):
        """3d adds to the result and changes no ranking."""
        result = stub_pipeline().run(PatientCase(**golden["case"]))
        expect = golden["expect"]
        top = [c.condition_id for c in result.candidates[: expect.get("top_k", 3)]]
        for required in expect.get("must_include", []):
            assert required in top


class Broken:
    def score(self, case):  # noqa: ARG002
        raise RuntimeError("no model")

    def score_by_connectivity(self, case):  # noqa: ARG002
        raise RuntimeError("no graph")


def test_with_nothing_ranking_only_the_flagged_get_tests():
    """Neither the model nor the graph: the order is the registry's and says nothing, so the
    leading three get no tests; the red-flag rules still run, and their conditions do."""
    pipeline = DiagnosisPipeline(ranker=Broken(), graph=Broken())
    result = pipeline.run(golden_case("GC-003"))
    assert {"ml", "graph_backend"} <= set(result.degraded_components)
    assert [t.split(":")[0] for t in result.suggested_next_tests] == ["Aortic dissection"]


# --------------------------------------------------------------------------- #
# 3. The questions: hand-computed
# --------------------------------------------------------------------------- #


def binary_entropy(x: float) -> float:
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


class TestHandComputed:
    def test_one_question_two_conditions(self):
        """Fever (``SYM:fever`` is DDXPlus's yes/no ``E_91``): P = 0.8 for A, 0.2 for B, a uniform
        posterior, so q = 0.5, H(now) = 1 bit, and either answer leaves (0.8, 0.2):
        gain = 1 - H(0.8) = 0.2780719051 bits."""
        graph = Graph({MI: {"DDX:E_91": 0.8}, GERD: {"DDX:E_91": 0.2}})
        ranking = next_questions(case(), graph)
        assert ranking.entropy_bits == pytest.approx(1.0)
        [question] = ranking.questions
        assert question.concept_id == "SYM:fever" and question.label == "Fever"
        assert question.gain_bits == pytest.approx(1 - binary_entropy(0.8), abs=1e-9)
        assert question.gain_bits == pytest.approx(0.2780719051, abs=1e-9)
        assert [s.condition_id for s in question.if_present] == [MI]
        assert [s.condition_id for s in question.if_denied] == [GERD]
        assert question.kind == "history"

    def test_an_unequal_posterior(self):
        """A finding already recorded tilts the posterior; the gain is then the mutual information
        under it. Diaphoresis present: P(E_50) = 0.6 for A, 0.3 for B, so p = (2/3, 1/3); fever
        as above: q = 2/3 * 0.8 + 1/3 * 0.2 = 0.6."""
        graph = Graph(
            {MI: {"DDX:E_91": 0.8, "DDX:E_50": 0.6}, GERD: {"DDX:E_91": 0.2, "DDX:E_50": 0.3}}
        )
        ranking = next_questions(case(present=["SYM:diaphoresis"]), graph)
        p, theta, q = np.array([2 / 3, 1 / 3]), np.array([0.8, 0.2]), 0.6
        expected = binary_entropy(q) - float(p @ [binary_entropy(t) for t in theta])
        assert ranking.entropy_bits == pytest.approx(binary_entropy(2 / 3))
        assert [x.concept_id for x in ranking.questions] == ["SYM:fever"]
        assert ranking.questions[0].gain_bits == pytest.approx(expected, abs=1e-9)

    def test_a_question_that_splits_no_conditions_gains_nothing(self):
        """Every condition expects fever equally, and none states breathlessness at all."""
        graph = Graph({MI: {"DDX:E_91": 0.6, "DDX:E_66": 0.0}, GERD: {"DDX:E_91": 0.6}})
        ranking = next_questions(case(), graph)
        assert ranking.questions == []
        assert ranking.entropy_bits == pytest.approx(1.0)

    def test_a_certain_graph_has_no_uncertainty_to_reduce_but_still_orders(self):
        """The entropy can be tiny; the questions are still ordered by the gain, however small."""
        graph = Graph({MI: {"DDX:E_91": 0.9, "DDX:E_50": 0.9}, GERD: {"DDX:E_50": 0.3}})
        ranking = next_questions(case(present=["SYM:fever"]), graph)
        assert 0 < ranking.entropy_bits < 0.2
        assert [q.concept_id for q in ranking.questions] == ["SYM:diaphoresis"]
        assert 0 < ranking.questions[0].gain_bits <= ranking.entropy_bits

    def test_the_gain_matches_the_scorer_s_own_probabilities(self):
        """On a store closed under the crosswalk, a finding that lands on one node is judged by
        P(node | c), the probability the scorer itself uses, for every condition."""
        store = networkx_store(MIXED)
        scorer = store.scorer
        recorded = case(present=["SYM:chest_pain", "SYM:sudden_onset"])
        ranking = next_questions(recorded, store, limit=100)
        base = store.score_by_connectivity(expand_case(recorded))
        conditions = list(scorer.conditions)
        scores = np.array([base[c] for c in conditions])
        p = np.exp(scores - scores.max())
        p /= p.sum()
        for question, node in (
            ("SYM:pain_character_tearing", "SYM:pain_character_tearing"),
            ("SYM:diaphoresis", "DDX:E_50"),
        ):
            theta = np.array([scorer.probability(c, node) for c in conditions])
            # The no-edge case: LEAK against nothing, normalised (next_questions' docstring).
            theta = np.where(theta == 0.01, 0.01 / 1.01, theta)
            q = float(p @ theta)
            kl = theta * np.log2(theta / q) + (1 - theta) * np.log2((1 - theta) / (1 - q))
            found = next(x for x in ranking.questions if x.concept_id == question)
            assert found.gain_bits == pytest.approx(float(p @ kl), rel=1e-8)

    def test_the_posterior_after_an_answer_is_the_graph_s(self):
        """In the usual case the model's posterior after "present" is the graph's posterior for
        the case with that finding recorded: p · θ, normalised, equals the softmax of the scores."""
        store = networkx_store(MIXED)
        before = case(present=["SYM:chest_pain"])
        after = case(present=["SYM:chest_pain", "SYM:pain_character_tearing"])
        conditions = list(store.scorer.conditions)

        def posterior(c: PatientCase) -> np.ndarray:
            s = store.score_by_connectivity(expand_case(c))
            v = np.array([s[k] for k in conditions])
            e = np.exp(v - v.max())
            return e / e.sum()

        theta = np.array(
            [store.scorer.probability(c, "SYM:pain_character_tearing") for c in conditions]
        )
        updated = posterior(before) * theta
        np.testing.assert_allclose(updated / updated.sum(), posterior(after), rtol=1e-9)


# --------------------------------------------------------------------------- #
# 4. The questions: what is never asked
# --------------------------------------------------------------------------- #


class TestNeverAskedAgain:
    @pytest.mark.parametrize("assertion", ["present", "denied"])
    def test_a_recorded_finding(self, assertion):
        store = networkx_store(MIXED)
        everything = {q.concept_id for q in next_questions(case(), store, limit=100).questions}
        assert {"SYM:pain_character_tearing", "SYM:diaphoresis"} <= everything
        kw = {assertion: ["SYM:pain_character_tearing", "SYM:diaphoresis"]}
        asked = {q.concept_id for q in next_questions(case(**kw), store, limit=100).questions}
        assert not asked & {"SYM:pain_character_tearing", "SYM:diaphoresis"}

    def test_an_expanded_case_asks_the_same_as_the_case_as_entered(self):
        """The API and the dashboard pass the case after ``expand_case``; the derived questions
        are set aside, so the answer is the same."""
        store = networkx_store(MIXED)
        entered = case(present=["SYM:chest_pain", "SYM:diaphoresis"], denied=["SYM:exertional"])
        labels = dict(store.graph.nodes(data="label"))
        assert next_questions(entered, store) == next_questions(expand_case(entered, labels), store)

    def test_a_finding_the_record_settles(self):
        """Unilateral leg swelling is leg swelling; a denied leg swelling denies both kinds;
        epigastric pain is abdominal pain. Inferences the crosswalk's match types allow only."""
        assert settled("SYM:leg_swelling", ["SYM:leg_swelling_unilateral"], [])
        assert settled("SYM:leg_swelling_unilateral", [], ["SYM:leg_swelling"])
        assert settled("SYM:leg_swelling_bilateral", [], ["SYM:leg_swelling"])
        assert settled("SYM:abdominal_pain", ["SYM:epigastric_pain"], [])
        # Not the other way round, and not across different findings on one question.
        assert not settled("SYM:leg_swelling_unilateral", ["SYM:leg_swelling"], [])
        assert not settled("SYM:epigastric_pain", ["SYM:abdominal_pain"], [])
        assert not settled("SYM:back_pain", ["SYM:chest_pain"], [])
        assert not settled("SYM:relieved_by_rest", ["SYM:exertional"], [])  # both narrower
        assert not settled("SYM:exertional", [], ["SYM:relieved_by_rest"])
        assert not settled("SYM:heartburn", ["SYM:regurgitation"], [])  # related only

        store = networkx_store(MIXED)
        asked = {
            q.concept_id
            for q in next_questions(
                case(present=["SYM:leg_swelling_unilateral"]), store, limit=100
            ).questions
        }
        assert "SYM:leg_swelling" not in asked

    @pytest.mark.parametrize("sex, asked", [("M", False), ("F", True), ("O", True)])
    def test_pregnancy_is_never_asked_of_a_man(self, sex, asked):
        """The graph does not know the patient's sex, so the ranking must (NEVER_ASKED_OF)."""
        graph = Graph({MI: {"DDX:E_167": 0.8}, GERD: {"DDX:E_167": 0.2}})
        patient = PatientCase(case_id="T", age=30, sex=sex, findings=[])
        ranking = next_questions(patient, graph)
        assert ("RF:pregnancy" in {q.concept_id for q in ranking.questions}) is asked

    def test_an_unknown_assertion_is_still_asked(self):
        """Not known is neither recorded nor denied (FR-1.3)."""
        graph = Graph({MI: {"DDX:E_91": 0.8}, GERD: {"DDX:E_91": 0.2}})
        ranking = next_questions(case(unknown=["SYM:fever"]), graph)
        assert [q.concept_id for q in ranking.questions] == ["SYM:fever"]


# --------------------------------------------------------------------------- #
# 5. The questions: properties
# --------------------------------------------------------------------------- #


ANSWERABLE = sorted(k for k, e in BY_CONCEPT.items() if e.pattern is not None)


def random_case(rng: random.Random, pool: list[str]) -> PatientCase:
    chosen = rng.sample(pool, rng.randint(0, 8))
    split = rng.randint(0, len(chosen))
    unknown = [k for k in rng.sample(pool, 2) if k not in chosen]
    return case(present=chosen[:split], denied=chosen[split:], unknown=unknown)


def random_graph(rng: random.Random) -> NetworkXGraphStore:
    nodes = sorted(
        {n for edges in MIXED.values() for n in edges}
        | {"SYM:back_pain", "SYM:exertional", "SYM:leg_swelling", "DDX:E_91", "DDX:E_14"}
    )
    conditions = rng.sample([c.id for c in CONDITIONS], rng.randint(2, 6))
    weights = (0.03, 0.12, 0.35, 0.65, 0.9, 1.0)
    return networkx_store(
        {
            c: {n: rng.choice(weights) for n in rng.sample(nodes, rng.randint(1, len(nodes)))}
            for c in conditions
        }
    )


class TestProperties:
    def test_the_expansion_agrees_with_expand_case_and_the_scorer(self):
        """``_expand`` mirrors ``expand_case`` on id sets: scored, the two give the same
        log-likelihoods, on random graphs and random cases (conflicts and repeats included)."""
        rng = random.Random(42)
        for _ in range(30):
            store = random_graph(rng)
            for _ in range(10):
                c = random_case(rng, ANSWERABLE)
                present, denied = c.present_concept_ids(), c.absent_concept_ids()
                every = {f.concept_id for f in c.findings}
                [row] = store.scorer.score_many([_expand(present, denied, every)])
                expected = store.score_by_connectivity(expand_case(c))
                got = dict(zip(store.scorer.conditions, row, strict=True))
                for condition, value in expected.items():
                    assert got[condition] == pytest.approx(value, abs=1e-9)

    def test_increments_are_score_differences(self):
        rng = random.Random(7)
        store = random_graph(rng)
        base = ({"SYM:chest_pain"}, {"SYM:exertional"})
        others = [({"SYM:chest_pain", "SYM:back_pain"}, {"SYM:exertional"}), (set(), set())]
        absolute = store.scorer.score_many([base, *others])
        relative = store.scorer.score_many(others, relative_to=base)
        np.testing.assert_allclose(relative, absolute[1:] - absolute[0], atol=1e-12)

    def test_the_gain_is_never_negative_nor_more_than_the_answer_can_hold(self):
        """A mutual information: at least 0, at most the entropy now and at most 1 bit (a yes/no
        answer)."""
        rng = random.Random(3)
        for _ in range(40):
            store = random_graph(rng)
            ranking = next_questions(random_case(rng, ANSWERABLE), store, limit=100)
            assert ranking.entropy_bits is not None and ranking.entropy_bits >= 0
            gains = [q.gain_bits for q in ranking.questions]
            assert all(0 < g <= min(1.0, ranking.entropy_bits) + 1e-9 for g in gains)
            assert gains == sorted(gains, reverse=True)

    def test_the_ranking_is_deterministic_with_a_stable_tie_rule(self):
        """Fever and diaphoresis split the two conditions identically, so they tie exactly and
        fall to the concept id."""
        graph = Graph(
            {MI: {"DDX:E_91": 0.8, "DDX:E_50": 0.8}, GERD: {"DDX:E_91": 0.2, "DDX:E_50": 0.2}}
        )
        first = next_questions(case(), graph)
        assert [q.concept_id for q in first.questions] == ["SYM:diaphoresis", "SYM:fever"]
        assert first.questions[0].gain_bits == first.questions[1].gain_bits
        for _ in range(3):
            assert next_questions(case(), graph) == first
        store = networkx_store(MIXED)
        gc = golden_case("GC-001")
        assert next_questions(gc, store) == next_questions(gc, store)

    def test_limit_and_the_conditions_named(self):
        store = networkx_store(MIXED)
        assert len(next_questions(case(), store).questions) == QUESTIONS_SHOWN
        assert len(next_questions(case(), store, limit=2).questions) == 2
        assert next_questions(case(), store, limit=0).questions == []
        for q in next_questions(case(), store, limit=100).questions:
            assert len(q.if_present) <= SEPARATES_SHOWN and len(q.if_denied) <= SEPARATES_SHOWN
            assert q.if_present or q.if_denied
            named = [s.condition_id for s in (*q.if_present, *q.if_denied)]
            assert len(named) == len(set(named))

    def test_examination_findings_say_so(self):
        store = networkx_store(MIXED)
        found = {q.concept_id: q for q in next_questions(case(), store, limit=100).questions}
        assert found["SYM:interarm_bp_difference"].kind == "examination"
        assert EXAMINATION.issubset(BY_CONCEPT)
        assert all(
            BY_CONCEPT[k].match is Match.NONE or k == "SYM:irregular_pulse" for k in EXAMINATION
        )

    def test_only_what_the_form_offers_is_asked_and_never_advice(self):
        """A question is a finding the form offers (``GET /concepts``), named by its label, and no
        label reads as a treatment (FR-6.5)."""
        store = networkx_store(MIXED)
        for q in next_questions(case(), store, limit=100).questions:
            assert q.concept_id in BY_CONCEPT and q.label == BY_CONCEPT[q.concept_id].label
            expected = (
                "risk_factor"
                if q.concept_id.startswith("RF:")
                else "examination" if q.concept_id in EXAMINATION else "history"
            )
            assert q.kind == expected
        assert not any(recommends_treatment(e.label) for e in BY_CONCEPT.values())


class TestDegraded:
    def test_a_store_without_likelihoods(self):
        class NoScorer:
            def score_by_connectivity(self, case):  # noqa: ARG002
                return {}

        for graph in (NoScorer(), None, Broken()):
            assert next_questions(case(), graph) == QuestionRanking()
            assert QuestionRanking().entropy_bits is None

    def test_a_scorer_that_fails(self):
        class FailingScorer:
            conditions = (MI, GERD)
            concepts = ("DDX:E_91",)

            def score_many(self, cases, relative_to=None):  # noqa: ARG002
                raise RuntimeError("the graph is gone")

        class Store:
            scorer = FailingScorer()

        ranking = next_questions(case(), Store())
        assert ranking.questions == [] and ranking.entropy_bits is None


# --------------------------------------------------------------------------- #
# 6. The stub graph, the golden cases and speed
# --------------------------------------------------------------------------- #


@pytest.mark.golden
@pytest.mark.parametrize("golden", GOLDEN, ids=[g["id"] for g in GOLDEN])
def test_the_golden_cases_on_the_stub_graph(golden):
    """Questions on the stub's naive-Bayes view: never a recorded finding, every gain positive.
    The stub is a toy, so what is asked is not checked clinically; that is the lead's check on
    the real graph."""
    graph = InMemoryGraphStore()
    c = PatientCase(**golden["case"])
    ranking = next_questions(c, graph)
    recorded = c.present_concept_ids() | c.absent_concept_ids()
    assert ranking.questions, golden["id"]
    assert not {q.concept_id for q in ranking.questions} & recorded
    assert all(q.gain_bits > 0 for q in ranking.questions)
    assert ranking == next_questions(c, graph)


def test_the_stub_scorer_reads_the_map_as_listed_frequency_unknown():
    graph = InMemoryGraphStore({MI: ("SYM:fever",), GERD: ()})
    assert graph.scorer.probability(MI, "SYM:fever") == pytest.approx(0.9)
    assert graph.scorer.probability(GERD, "SYM:fever") == pytest.approx(0.01)
    assert graph.scorer is graph.scorer  # built once


def test_speed_on_the_stub_graph():
    """The dashboard re-ranks on every change (D-14, EXP-022: the pipeline takes ~10-13 ms with
    the transformer), so the questions must cost a few milliseconds. A generous bound, to catch a
    regression by an order of magnitude, not to benchmark."""
    graph = InMemoryGraphStore()
    c = golden_case("GC-001")
    next_questions(c, graph)  # the per-graph precomputation
    times = []
    for _ in range(20):
        started = time.perf_counter()
        next_questions(c, graph)
        times.append(time.perf_counter() - started)
    assert statistics.median(times) < 0.05


# --------------------------------------------------------------------------- #
# 7. The dashboard's block
# --------------------------------------------------------------------------- #


def test_the_dashboard_block_shows_the_tests_and_the_questions():
    graph = InMemoryGraphStore()
    c = golden_case("GC-002")
    result = stub_pipeline().run(c)
    steps = next_steps(result, c, graph)
    assert steps["tests"] == result.suggested_next_tests and steps["tests"]
    assert steps["questions"] and steps["entropy_bits"] is not None
    for q in steps["questions"]:
        assert q["verb"] in ("Ask about", "Check") and q["label"]
        assert q["concept_id"] not in c.present_concept_ids() | c.absent_concept_ids()
        assert q["if_present"] or q["if_denied"]


def test_the_dashboard_block_without_a_graph():
    c = golden_case("GC-002")
    result = stub_pipeline().run(c)
    steps = next_steps(result, c, Broken())
    assert steps["questions"] == [] and steps["entropy_bits"] is None
    assert steps["tests"] == result.suggested_next_tests
    assert next_steps(result, None, InMemoryGraphStore())["questions"] == []


def test_an_examination_finding_says_check():
    store = networkx_store(MIXED)
    result = stub_pipeline().run(case())
    steps = next_steps(result, case(), store)
    verbs = {q["concept_id"]: q["verb"] for q in steps["questions"]}
    assert all(v == ("Check" if k in EXAMINATION else "Ask about") for k, v in verbs.items())


@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None, reason="needs Streamlit")
def test_the_app_shows_the_block_on_the_stub_graph(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NIGHTINGALE_GRAPH", "stub")
    app = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    app.selectbox(key="example").select("GC-002").run()
    assert not app.exception, app.exception
    assert any(h.value == "What to ask next" for h in app.subheader)
    text = " ".join(m.value for m in app.markdown)
    assert "Tests to consider" in text and "D-dimer" in text
    assert "a yes favours" in text or "a no favours" in text


# --------------------------------------------------------------------------- #
# 8. The real graph (skips without data/, as in CI)
# --------------------------------------------------------------------------- #


@needs_kg
@pytest.mark.golden
@pytest.mark.parametrize("golden", GOLDEN, ids=[g["id"] for g in GOLDEN])
def test_the_golden_cases_on_the_real_graph(golden):
    store = NetworkXGraphStore.from_files(*KG_FILES, BODHI_DIR if BODHI_DIR.exists() else None)
    labels = dict(store.graph.nodes(data="label"))
    entered = PatientCase(**golden["case"])
    expanded = expand_case(entered, labels)

    # The expansion on id sets scores exactly as the pipeline's graph does.
    present, denied = entered.present_concept_ids(), entered.absent_concept_ids()
    every = {f.concept_id for f in [*entered.findings, *entered.risk_factors]}
    [row] = store.scorer.score_many([_expand(present, denied, every)])
    expected = store.score_by_connectivity(expanded)
    for condition, value in zip(store.scorer.conditions, row, strict=True):
        assert value == pytest.approx(expected[condition], abs=1e-9)

    ranking = next_questions(expanded, store)
    assert ranking == next_questions(entered, store)
    assert not {q.concept_id for q in ranking.questions} & (present | denied)
    assert all(q.gain_bits > 0 for q in ranking.questions)

    times = []
    for _ in range(10):
        started = time.perf_counter()
        next_questions(expanded, store)
        times.append(time.perf_counter() - started)
    assert statistics.median(times) < 0.05
