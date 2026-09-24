"""Tests for the knowledge-graph score (task 2a, EXP-005, src/medical_kg/scoring.py).

Three regressions, one per defect of the overlap score it replaced, then the properties the design
rests on. The synthetic tests run everywhere; the ones on the real graph skip when data/ is absent,
as in CI.

1. **Unstable above stable angina when rest pain is present** (docs/02 §5.1, limitation 2). The
   overlap scored stable 1.00 and unstable 0.875.
2. **GC-001's infarction in the top three on graph score alone** (EXP-014). The overlap put it
   fifth, then third once BODHI-S arrived.
3. **Knowing more does not punish a condition** (EXP-016). The overlap divided by everything a
   condition might show, so BODHI-S dropped MI's graph-only top-1 on validate from 0.857 to 0.602.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from src.contracts import Assertion, Finding, PatientCase
from src.medical_kg.crosswalk import expand_case
from src.medical_kg.loader import LIKELIHOOD_WEIGHT, EdgeSource, Likelihood, only_sources
from src.medical_kg.networkx_store import NetworkXGraphStore
from src.medical_kg.scoring import CAP, LEAK, EdgeKind, LikelihoodScorer

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERIM = REPO_ROOT / "data" / "interim"
KG_FILES = (INTERIM / "ddxplus_chestpain_conditions.json", INTERIM / "ddxplus_evidences.json")
BODHI_DIR = REPO_ROOT / "data" / "raw" / "bodhi_s"
PARQUET = INTERIM / "ddxplus_chestpain_validate.parquet"
needs_kg = pytest.mark.skipif(
    not all(p.exists() for p in KG_FILES), reason="data/ is not committed (CI)"
)
GOLDEN = yaml.safe_load(
    (REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml").read_text(encoding="utf-8")
)["cases"]

UA, SA, MI = "COND:unstable_angina", "COND:stable_angina", "COND:nstemi_stemi"
REST_PAIN = "DDX:E_14"


def case(present=(), absent=(), unknown=()) -> PatientCase:
    findings = [Finding(concept_id=c, label=c) for c in present]
    findings += [Finding(concept_id=c, label=c, assertion=Assertion.ABSENT) for c in absent]
    findings += [Finding(concept_id=c, label=c, assertion=Assertion.UNKNOWN) for c in unknown]
    return PatientCase(case_id="T", age=60, sex="M", findings=findings)


# A miniature of the angina problem: stable's evidence sits inside unstable's.
SHARED = ["DDX:E_53", "DDX:E_54", "DDX:E_55"]
ANGINA = LikelihoodScorer(
    {SA: {c: 1.0 for c in SHARED}, UA: {**{c: 1.0 for c in SHARED}, REST_PAIN: 1.0}},
    [*SHARED, REST_PAIN],
)


# --------------------------------------------------------------------------- #
# 1. The anginas
# --------------------------------------------------------------------------- #


class TestTheAnginas:
    def test_rest_pain_puts_unstable_above_stable(self):
        """Rest pain defines unstable angina, and stable angina cannot explain it."""
        scores = ANGINA.score(case(present=[*SHARED, REST_PAIN]))
        assert scores[UA] > scores[SA]
        assert scores[UA] - scores[SA] == pytest.approx(math.log(CAP) - math.log(LEAK))

    def test_denied_rest_pain_puts_stable_above_unstable(self):
        scores = ANGINA.score(case(present=SHARED, absent=[REST_PAIN]))
        assert scores[SA] > scores[UA]

    def test_a_stable_picture_alone_cannot_separate_them(self):
        """Rest pain not mentioned is not rest pain denied. The graph must not pretend to know,
        so the two tie on graph score. (Where the fused scores tie too, the pipeline keeps the
        registry's order, which lists unstable angina, the must-not-miss one, first.)"""
        scores = ANGINA.score(case(present=SHARED))
        assert scores[SA] == scores[UA]

    @needs_kg
    def test_on_the_real_graph(self):
        """docs/02 §5.1, limitation 2: stable angina's full picture plus rest pain."""
        store = NetworkXGraphStore.from_files(*KG_FILES)
        stable = sorted(c for c in store._expected[SA] if c.startswith("DDX:"))
        assert REST_PAIN not in stable
        with_rest = store.score_by_connectivity(case(present=[*stable, REST_PAIN]))
        assert with_rest[UA] > with_rest[SA], "the overlap score had stable 1.00, unstable 0.875"
        rest_denied = store.score_by_connectivity(case(present=stable, absent=[REST_PAIN]))
        assert rest_denied[SA] > rest_denied[UA]


# --------------------------------------------------------------------------- #
# 2. The golden cases on graph score alone
# --------------------------------------------------------------------------- #


@needs_kg
@pytest.mark.golden
@pytest.mark.parametrize("golden", GOLDEN, ids=[g["id"] for g in GOLDEN])
def test_the_graph_alone_ranks_each_golden_case(golden):
    """Without the ML ranker and without red flags. GC-001's infarction was fifth under the
    overlap score (EXP-014), third with BODHI-S (EXP-016); 2a's own regression case."""
    store = NetworkXGraphStore.from_files(*KG_FILES, BODHI_DIR if BODHI_DIR.exists() else None)
    labels = dict(store.graph.nodes(data="label"))
    scores = store.score_by_connectivity(expand_case(PatientCase(**golden["case"]), labels))
    ranking = sorted(scores, key=scores.get, reverse=True)
    top_k = golden["expect"].get("top_k", 3)
    for required in golden["expect"].get("must_include", []):
        assert required in ranking[:top_k], (
            f"{golden['id']}: {required} is ranked {ranking.index(required) + 1} by the graph "
            "alone. Fix the component, not this expectation."
        )


# --------------------------------------------------------------------------- #
# 3. Knowing more does not punish a condition
# --------------------------------------------------------------------------- #


class TestKnowledge:
    def test_an_edge_to_a_finding_the_case_does_not_mention_changes_nothing(self):
        """EXP-016's defect: the overlap divided by everything a condition might show."""
        lean = LikelihoodScorer({MI: {"DDX:E_53": 1.0}, SA: {"DDX:E_53": 1.0}}, ["DDX:E_53", "X"])
        rich = LikelihoodScorer(
            {MI: {"DDX:E_53": 1.0, "X": 0.65}, SA: {"DDX:E_53": 1.0}}, ["DDX:E_53", "X"]
        )
        patient = case(present=["DDX:E_53"])
        assert rich.score(patient) == lean.score(patient)

    def test_an_edge_to_a_present_finding_raises_the_condition(self):
        without = LikelihoodScorer({MI: {}, SA: {"X": 0.9}}, ["X"])
        with_edge = LikelihoodScorer({MI: {"X": 0.12}, SA: {"X": 0.9}}, ["X"])
        patient = case(present=["X"])
        assert with_edge.score(patient)[MI] > without.score(patient)[MI], "any band beats LEAK"

    @needs_kg
    @pytest.mark.skipif(not PARQUET.exists(), reason="data/ is not committed (CI)")
    def test_bodhi_s_does_not_lower_mis_graph_only_top1_on_validate(self):
        """EXP-016 on the validate patients with MI: 0.857 -> 0.602 under the overlap score.
        Circular (R-12), so it is a check that enrichment does no harm, not a skill figure."""
        import pandas as pd

        from src.medical_kg.cardiac_kg import build_cardiac_kg
        from src.medical_kg.crosswalk import case_from_ddxplus

        kg = build_cardiac_kg(*KG_FILES, BODHI_DIR)
        stores = {
            "without": NetworkXGraphStore(
                only_sources(kg, [EdgeSource.DDXPLUS, EdgeSource.HAND_AUTHORED])
            ),
            "with": NetworkXGraphStore(kg),
        }
        columns = ["case_id", "age", "sex", "evidences", "positive_codes", "label_condition_id"]
        frame = pd.read_parquet(PARQUET, columns=columns)
        records = frame[frame["label_condition_id"] == MI].to_dict("records")
        top1 = {}
        for name, store in stores.items():
            labels = dict(store.graph.nodes(data="label"))
            hits = 0
            for record in records:
                scores = store.score_by_connectivity(case_from_ddxplus(record, {}, labels))
                hits += max(scores, key=scores.get) == MI
            top1[name] = hits / len(records)
        assert top1["with"] >= top1["without"]


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #


class TestProperties:
    def test_the_constants_sit_inside_the_likelihood_scale(self):
        """LEAK below the lowest band: an unassociated finding is rarer than a rare associated
        one. EXP-005 found every verdict stable from 0.001 to 0.01, and moving at 0.03."""
        assert LIKELIHOOD_WEIGHT[Likelihood.RARE] > LEAK
        assert LIKELIHOOD_WEIGHT[Likelihood.VERY_HIGH] == CAP

    def test_unknown_findings_count_for_nothing(self):
        assert ANGINA.score(case(present=SHARED, unknown=[REST_PAIN])) == ANGINA.score(
            case(present=SHARED)
        )

    def test_denying_what_a_condition_never_shows_says_nothing_against_it(self):
        base = ANGINA.score(case(present=SHARED))
        denied = ANGINA.score(case(present=SHARED, absent=[REST_PAIN]))
        assert denied[SA] == base[SA]
        assert denied[UA] == pytest.approx(base[UA] + math.log(1 - CAP))

    def test_contradictory_findings_are_ignored(self):
        both = ANGINA.score(case(present=[*SHARED, REST_PAIN], absent=[REST_PAIN]))
        assert both == ANGINA.score(case(present=SHARED))

    def test_a_concept_no_condition_knows_is_skipped(self):
        scorer = LikelihoodScorer({SA: {"DDX:E_53": 1.0}, UA: {}}, ["DDX:E_53", "ORPHAN"])
        assert scorer.score(case(present=["DDX:E_53", "ORPHAN"])) == scorer.score(
            case(present=["DDX:E_53"])
        )

    def test_contributions_sum_to_the_score_and_name_their_edge(self):
        patient = case(present=[*SHARED, REST_PAIN], absent=["DDX:E_54"])
        for condition in (SA, UA):
            parts = ANGINA.contributions(patient, condition)
            assert sum(p.log_likelihood for p in parts) == pytest.approx(
                ANGINA.score(patient)[condition]
            )
        stable = {p.concept_id: p for p in ANGINA.contributions(patient, SA)}
        assert stable[REST_PAIN].edge is EdgeKind.NONE
        assert stable[REST_PAIN].probability == LEAK
        assert stable["DDX:E_53"].edge is EdgeKind.EXPLICIT
        assert ANGINA.contributions(patient, "COND:not_a_condition") == []

    def test_every_condition_gets_a_score_even_with_nothing_asserted(self):
        assert ANGINA.score(case()) == {SA: 0.0, UA: 0.0}


class TestCrosswalkClosure:
    """The graph mixes question-level (DDXPlus) and answer-level (hand-authored, BODHI-S) edges."""

    # A knows two answers to "where is your pain?" (E_55); B knows only the questions.
    SCORER = LikelihoodScorer(
        {
            "A": {"SYM:chest_pain": 0.65, "SYM:back_pain": 0.35},
            "B": {"DDX:E_55": 1.0, "DDX:E_53": 1.0},
        },
        ["SYM:chest_pain", "SYM:back_pain", "DDX:E_55", "DDX:E_53"],
    )

    def test_answers_imply_their_question_and_its_parent_by_noisy_or(self):
        """Not the maximum, which is only a lower bound on giving *some* answer."""
        noisy_or = 1 - (1 - 0.65) * (1 - 0.35)
        for question in ("DDX:E_55", "DDX:E_53"):
            assert self.SCORER.edge_kind("A", question) is EdgeKind.IMPLIED
            assert self.SCORER.probability("A", question) == pytest.approx(noisy_or)

    def test_a_condition_asking_the_question_gets_the_stating_conditions_mean(self):
        """Ignorance is average: without this, every chest-pain patient's SYM:chest_pain would
        count LEAK against every DDXPlus condition, and hand the case to aortic dissection."""
        assert self.SCORER.edge_kind("B", "SYM:chest_pain") is EdgeKind.IMPUTED
        assert self.SCORER.probability("B", "SYM:chest_pain") == pytest.approx(0.65)
        patient = case(present=["SYM:chest_pain", "DDX:E_55", "DDX:E_53"])
        scores = self.SCORER.score(patient)
        assert scores["B"] > scores["A"] + math.log(
            LEAK
        ), "B is not treated as unable to explain it"

    def test_an_answer_only_one_condition_states_cannot_separate_it_from_the_others(self):
        """The price of imputing the mean, and limitation 1 of the KG card: the graph does not
        know how rarely B shows the answer, so the score must not invent it."""
        only_a = case(present=["SYM:chest_pain"])
        parts = {c: self.SCORER.contributions(only_a, c)[0].log_likelihood for c in ("A", "B")}
        assert parts["A"] == pytest.approx(parts["B"])

    def test_explicit_edges_are_never_overwritten(self):
        assert self.SCORER.edge_kind("B", "DDX:E_55") is EdgeKind.EXPLICIT
        assert self.SCORER.probability("B", "DDX:E_55") == CAP

    def test_paths_cite_only_edges_the_graph_holds(self):
        """An explanation must not present an implied or imputed edge as a stated fact."""
        from src.medical_kg.loader import KGEdge, KGNode, KnowledgeGraph, NodeType, Relation

        nodes = {
            "COND:gerd": KGNode("COND:gerd", NodeType.CONDITION, "GERD"),
            "DDX:E_55": KGNode("DDX:E_55", NodeType.SYMPTOM, "Where?"),
            "SYM:chest_pain": KGNode("SYM:chest_pain", NodeType.SYMPTOM, "Chest pain"),
        }
        edges = [KGEdge("COND:gerd", "DDX:E_55", Relation.HAS_SYMPTOM, EdgeSource.DDXPLUS)]
        store = NetworkXGraphStore(KnowledgeGraph(nodes=nodes, edges=edges))
        patient = case(present=["SYM:chest_pain", "DDX:E_55"])
        assert [p.finding_id for p in store.paths_for(patient, "COND:gerd")] == ["DDX:E_55"]


class TestAFactIsCountedOnce:
    """Found in review: an expanded case holds an answer *and* the questions it implies, and
    naive-Bayes counted them as independent findings. Chest pain alone cost a condition that
    cannot explain it 3 * log(LEAK) instead of log(LEAK)."""

    def test_an_answer_and_its_implied_questions_count_once(self):
        scorer = LikelihoodScorer(
            {"A": {"SYM:chest_pain": 0.65}, "B": {}},
            ["SYM:chest_pain", "DDX:E_55", "DDX:E_53"],
        )
        expanded = case(present=["SYM:chest_pain", "DDX:E_55", "DDX:E_53"])  # as expand_case gives
        assert scorer.score(expanded)["B"] == pytest.approx(math.log(LEAK)), "once, not three times"
        assert [p.concept_id for p in scorer.contributions(expanded, "A")] == ["SYM:chest_pain"]

    def test_a_question_stays_when_its_answer_is_not_a_usable_concept(self):
        """SYM:diaphoresis lives on the DDX:E_50 node, so DDX:E_50 is its only carrier."""
        scorer = LikelihoodScorer({"A": {"DDX:E_50": 1.0}, "B": {}}, ["DDX:E_50"])
        scores = scorer.score(case(present=["SYM:diaphoresis", "DDX:E_50"]))
        assert scores["A"] == pytest.approx(math.log(CAP))
        assert scores["B"] == pytest.approx(math.log(LEAK))

    def test_a_denial_and_the_denial_it_entails_count_once(self):
        """Denying exertional pain denies E_218 (worse on exertion, better at rest), which is
        narrower. GC-004 denies both."""
        scorer = LikelihoodScorer(
            {"A": {"SYM:exertional": 0.65, "DDX:E_218": 1.0}}, ["SYM:exertional", "DDX:E_218"]
        )
        denied = scorer.score(case(absent=["SYM:exertional", "DDX:E_218"]))
        assert denied["A"] == pytest.approx(math.log(1 - 0.65))

    def test_denying_a_follow_up_does_not_deny_its_parent(self):
        """Denying radiation to the jaw says nothing about whether there is pain at all."""
        scorer = LikelihoodScorer(
            {"A": {"SYM:radiation_jaw_arm": 0.65, "DDX:E_53": 1.0}},
            ["SYM:radiation_jaw_arm", "DDX:E_53"],
        )
        scores = scorer.score(case(present=["DDX:E_53"], absent=["SYM:radiation_jaw_arm"]))
        assert scores["A"] == pytest.approx(math.log(CAP) + math.log(1 - 0.65))
