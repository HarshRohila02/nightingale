"""Tests for composing the knowledge graph from its sources (task 1b).

Covers the shared likelihood scale, canonical concept nodes, merging and filtering by source,
and the hand-authored aortic-dissection facts. The test that reads the real interim files skips
when data/ is absent, as it is in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.conditions import BY_ID
from src.contracts import PatientCase
from src.medical_kg.crosswalk import BY_CONCEPT, canonical_concept_id, expand_case
from src.medical_kg.hand_authored import (
    AORTIC_DISSECTION,
    FACTS,
    REFERENCES,
    load_hand_authored_kg,
)
from src.medical_kg.loader import (
    LIKELIHOOD_WEIGHT,
    EdgeSource,
    KGEdge,
    KGNode,
    KnowledgeGraph,
    Likelihood,
    NodeType,
    Relation,
    concept_node,
    condition_nodes,
    merge_graphs,
    only_sources,
)
from src.medical_kg.networkx_store import NetworkXGraphStore

REPO_ROOT = Path(__file__).resolve().parents[1]
KG_FILES = (
    REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_evidences.json",
)
needs_kg = pytest.mark.skipif(
    not all(p.exists() for p in KG_FILES), reason="data/ is not committed (CI)"
)
GOLDEN = {
    g["id"]: g
    for g in yaml.safe_load(
        (REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml").read_text(encoding="utf-8")
    )["cases"]
}
PE = "COND:pulmonary_embolism"


def _graph(nodes: list[KGNode], edges: list[KGEdge]) -> KnowledgeGraph:
    return KnowledgeGraph(nodes={**condition_nodes(), **{n.id: n for n in nodes}}, edges=edges)


class TestLikelihood:
    def test_weights_rise_with_the_band_and_stay_below_ddxplus(self):
        weights = [LIKELIHOOD_WEIGHT[band] for band in Likelihood]
        assert weights == sorted(weights) and weights[0] == 0.0
        assert max(weights) < 1.0, "a DDXPlus edge (1.0) always outweighs a banded one"


class TestConceptNode:
    def test_a_yes_no_equivalent_concept_sits_on_the_ddxplus_question(self):
        node = concept_node("SYM:diaphoresis", {"DDX:E_50": "Sweating?"}, EdgeSource.BODHI_S)
        assert (node.id, node.label, node.type) == ("DDX:E_50", "Sweating?", NodeType.SYMPTOM)

    def test_an_answer_level_concept_keeps_its_own_node_and_label(self):
        node = concept_node("SYM:chest_pain", {}, EdgeSource.HAND_AUTHORED)
        assert (node.id, node.label) == ("SYM:chest_pain", "Chest pain")
        assert node.properties["crosswalk_match"] == "close"

    def test_risk_factors_are_typed_as_risk_factors(self):
        assert concept_node("RF:hypertension", {}, EdgeSource.HAND_AUTHORED).type is (
            NodeType.RISK_FACTOR
        )
        assert concept_node("RF:cocaine_use", {}, EdgeSource.HAND_AUTHORED).id == "RF:cocaine_use"

    def test_an_unknown_concept_is_refused(self):
        with pytest.raises(ValueError, match="CROSSWALK"):
            canonical_concept_id("SYM:not_a_concept")


class TestMergeGraphs:
    def test_one_fact_from_two_sources_is_one_node_with_two_edges(self):
        ddx = _graph(
            [KGNode("DDX:E_50", NodeType.SYMPTOM, "Sweating?")],
            [KGEdge(PE, "DDX:E_50", Relation.HAS_SYMPTOM, EdgeSource.DDXPLUS)],
        )
        other = _graph(
            [KGNode("DDX:E_50", NodeType.SYMPTOM, "Diaphoresis")],
            [KGEdge(PE, "DDX:E_50", Relation.HAS_SYMPTOM, EdgeSource.BODHI_S, 0.65)],
        )
        merged = merge_graphs(ddx, other)
        assert merged.nodes["DDX:E_50"].label == "Sweating?", "the first source's node is kept"
        assert [e.source for e in merged.edges] == [EdgeSource.DDXPLUS, EdgeSource.BODHI_S]
        store = NetworkXGraphStore(merged)
        assert store.graph.number_of_edges(PE, "DDX:E_50") == 2
        case = PatientCase(
            case_id="T", age=50, sex="M", findings=[{"concept_id": "DDX:E_50", "label": "x"}]
        )
        assert store.score_by_connectivity(case)[PE] == pytest.approx(1.0), "counted once"

    def test_an_edge_takes_the_relation_of_its_target_node(self):
        """DDXPlus calls E_110 (immobilisation) a risk factor; a source that lists it as a
        symptom does not change that."""
        ddx = _graph(
            [KGNode("DDX:E_110", NodeType.RISK_FACTOR, "Immobilised?")],
            [KGEdge(PE, "DDX:E_110", Relation.HAS_RISK_FACTOR, EdgeSource.DDXPLUS)],
        )
        other = _graph(
            [KGNode("DDX:E_110", NodeType.SYMPTOM, "Immobilised", {"source": "bodhi_s"})],
            [KGEdge(PE, "DDX:E_110", Relation.HAS_SYMPTOM, EdgeSource.BODHI_S, 0.65)],
        )
        merged = merge_graphs(ddx, other)
        assert {e.relation for e in merged.edges} == {Relation.HAS_RISK_FACTOR}
        assert any("DDX:E_110" in a and "kept RiskFactor" in a for a in merged.anomalies)

    def test_a_source_stating_an_edge_twice_is_an_error(self):
        edge = KGEdge(PE, "SYM:chest_pain", Relation.HAS_SYMPTOM, EdgeSource.BODHI_S, 0.35)
        graph = _graph([KGNode("SYM:chest_pain", NodeType.SYMPTOM, "Chest pain")], [edge, edge])
        with pytest.raises(ValueError, match="twice"):
            merge_graphs(graph)

    def test_an_edge_to_a_missing_node_is_an_error(self):
        edge = KGEdge(PE, "SYM:ghost", Relation.HAS_SYMPTOM, EdgeSource.BODHI_S)
        with pytest.raises(ValueError, match="missing node"):
            merge_graphs(_graph([], [edge]))

    def test_only_sources_keeps_the_nodes_and_drops_the_other_edges(self):
        merged = merge_graphs(load_hand_authored_kg())
        assert only_sources(merged, [EdgeSource.DDXPLUS]).edges == []
        assert only_sources(merged, [EdgeSource.DDXPLUS]).nodes == merged.nodes
        assert len(only_sources(merged, [EdgeSource.HAND_AUTHORED]).edges) == len(FACTS)


# --------------------------------------------------------------------------- #
# Hand-authored aortic dissection
# --------------------------------------------------------------------------- #

ADD_RS_MARKERS = {
    # high-risk conditions
    "RF:connective_tissue_disease",
    "RF:family_history_aortic_disease",
    "RF:aortic_valve_disease",
    "RF:aortic_manipulation",
    "RF:thoracic_aortic_aneurysm",
    # high-risk pain features
    "SYM:sudden_onset",
    "SYM:severe_pain",
    "SYM:pain_character_tearing",
    # high-risk examination features
    "SYM:pulse_deficit",
    "SYM:interarm_bp_difference",
    "SYM:focal_neuro_deficit",
    "SYM:aortic_regurgitation_murmur",
    "SYM:hypotension",
}


class TestHandAuthored:
    def test_every_add_rs_marker_is_in(self):
        assert {f.concept for f in FACTS} >= ADD_RS_MARKERS

    def test_facts_are_well_formed(self):
        for fact in FACTS:
            assert fact.condition_id in BY_ID and fact.concept in BY_CONCEPT, fact
            assert fact.reference in REFERENCES, fact
            assert fact.likelihood is not Likelihood.ZERO and fact.evidence, fact
        pairs = [(f.condition_id, canonical_concept_id(f.concept)) for f in FACTS]
        assert len(pairs) == len(set(pairs)), "a condition-concept pair appears twice"

    def test_the_graph_holds_only_hand_authored_edges_for_aortic_dissection(self):
        kg = load_hand_authored_kg({"DDX:E_104": "High blood pressure?"})
        assert {(e.condition_id, e.source) for e in kg.edges} == {
            (AORTIC_DISSECTION, EdgeSource.HAND_AUTHORED)
        }
        hypertension = next(e for e in kg.edges if e.properties["concept"] == "RF:hypertension")
        assert hypertension.concept_id == "DDX:E_104"
        assert kg.nodes["DDX:E_104"].label == "High blood pressure?"
        assert hypertension.weight == LIKELIHOOD_WEIGHT[Likelihood.HIGH]
        assert hypertension.properties["reference"] == "IRAD-2000"

    def test_gc003_ranks_aortic_dissection_on_the_graph_alone(self):
        """Before these edges, aortic dissection scored 0 and only its red flag surfaced it."""
        store = NetworkXGraphStore(merge_graphs(load_hand_authored_kg()))
        case = expand_case(PatientCase(**GOLDEN["GC-003"]["case"]))
        assert store.score_by_connectivity(case)[AORTIC_DISSECTION] > 0.4


@needs_kg
def test_gc003_puts_aortic_dissection_first_by_graph_score_on_the_real_graph():
    store = NetworkXGraphStore.from_files(*KG_FILES)
    labels = dict(store.graph.nodes(data="label"))
    scores = store.score_by_connectivity(
        expand_case(PatientCase(**GOLDEN["GC-003"]["case"]), labels)
    )
    assert max(scores, key=scores.get) == AORTIC_DISSECTION
