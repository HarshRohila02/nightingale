"""Tests for the knowledge-graph loader and the NetworkX GraphStore (task 1b).

The fixtures are tiny, hand-written and synthetic, in the shape of the interim files that
scripts/decode_ddxplus.py writes. The tests that read the real interim files skip when
data/ is absent (as it is in CI, because data/ is never committed).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

from src.conditions import CONDITIONS, trainable_ids
from src.contracts import DISCLAIMER, EvidenceRole, GraphStore, PatientCase
from src.medical_kg.loader import (
    EdgeSource,
    KGEdge,
    KGNode,
    KnowledgeGraph,
    NodeType,
    Relation,
    load_ddxplus_kg,
)
from src.medical_kg.networkx_store import NetworkXGraphStore
from src.pipeline import DiagnosisPipeline
from src.stubs import ConstantRanker, EmptyRetriever, TemplateExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_INTERIM = REPO_ROOT / "data" / "interim"
REAL_FILES = (
    REAL_INTERIM / "ddxplus_chestpain_conditions.json",
    REAL_INTERIM / "ddxplus_evidences.json",
)
needs_real_data = pytest.mark.skipif(
    not all(p.exists() for p in REAL_FILES), reason="data/ is not committed (CI)"
)

PE = "COND:pulmonary_embolism"
GERD = "COND:gerd"


def _vocab(label: str, is_antecedent: bool) -> dict:
    return {
        "label_en": label,
        "label_fr": f"{label} (fr)",
        "data_type": "B",
        "is_antecedent": is_antecedent,
        "kind": "question",
    }


VOCABULARY = {
    "E_2": _vocab("Recently immobilised?", True),
    "E_10": _vocab("Burning pain?", False),
    "E_16": _vocab("Feeling anxious?", True),  # flagged antecedent, but used as a symptom
    "E_53": _vocab("Pain related to the consultation?", False),
    "E_66": _vocab("Short of breath?", False),
    "E_79": _vocab("Smoker?", True),
}

CONDITIONS_JSON = {
    PE: {
        "condition_id": PE,
        "icd10": "I26",
        "severity": 2,
        "symptoms": [{"code": "E_66"}, {"code": "E_53"}],
        "antecedents": [{"code": "E_79"}, {"code": "E_2"}],
    },
    GERD: {
        "condition_id": GERD,
        "icd10": "K21",
        "severity": 3,
        "symptoms": [{"code": "E_53"}, {"code": "E_10"}, {"code": "E_16"}],
        "antecedents": [{"code": "E_79"}],
    },
}


def _write(tmp_path: Path, conditions: dict, vocabulary: dict) -> tuple[Path, Path]:
    conditions_path = tmp_path / "ddxplus_chestpain_conditions.json"
    vocabulary_path = tmp_path / "ddxplus_evidences.json"
    conditions_path.write_text(json.dumps(conditions), encoding="utf-8")
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    return conditions_path, vocabulary_path


@pytest.fixture
def kg(tmp_path: Path) -> KnowledgeGraph:
    return load_ddxplus_kg(*_write(tmp_path, CONDITIONS_JSON, VOCABULARY))


@pytest.fixture
def store(kg: KnowledgeGraph) -> NetworkXGraphStore:
    return NetworkXGraphStore(kg)


def _case(present=(), absent=(), risk_factors=()) -> PatientCase:
    findings = [{"concept_id": c, "label": f"label {c}"} for c in present]
    findings += [{"concept_id": c, "label": f"label {c}", "assertion": "absent"} for c in absent]
    return PatientCase(
        case_id="T-KG",
        age=50,
        sex="F",
        findings=findings,
        risk_factors=[{"concept_id": c, "label": f"label {c}"} for c in risk_factors],
    )


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


class TestLoader:
    def test_condition_nodes_come_from_the_registry(self, kg):
        assert {c.id for c in CONDITIONS} <= set(kg.nodes)
        assert kg.nodes["COND:aortic_dissection"].type is NodeType.CONDITION
        assert kg.nodes[PE].properties["icd10"] == "I26"
        assert kg.nodes["COND:aortic_dissection"].properties["icd10"] is None

    def test_evidence_nodes_are_namespaced_and_typed_by_use(self, kg):
        assert kg.nodes["DDX:E_53"].type is NodeType.SYMPTOM
        assert kg.nodes["DDX:E_2"].type is NodeType.RISK_FACTOR
        assert kg.nodes["DDX:E_53"].label == "Pain related to the consultation?"
        assert kg.nodes["DDX:E_53"].properties["ddxplus_code"] == "E_53"

    def test_every_edge_records_its_source(self, kg):
        """R-12: the KG's contribution must be reportable by source."""
        assert len(kg.edges) == 8
        assert {e.source for e in kg.edges} == {EdgeSource.DDXPLUS}
        assert all(e.weight == 1.0 for e in kg.edges)

    def test_a_disagreeing_antecedent_flag_is_an_anomaly_not_an_error(self, kg):
        assert kg.nodes["DDX:E_16"].type is NodeType.SYMPTOM
        assert any(a.startswith("E_16 ") for a in kg.anomalies)

    def test_trainable_conditions_without_evidence_are_reported(self, kg):
        reported = {a.split()[0] for a in kg.anomalies if "no DDXPlus evidence" in a}
        assert reported == trainable_ids() - {PE, GERD}

    def test_unknown_condition_is_a_data_error(self, tmp_path):
        bad = {**CONDITIONS_JSON, "COND:pneumonia": {"symptoms": [], "antecedents": []}}
        with pytest.raises(ValueError, match="COND:pneumonia"):
            load_ddxplus_kg(*_write(tmp_path, bad, VOCABULARY))

    def test_code_missing_from_the_vocabulary_is_a_data_error(self, tmp_path):
        vocabulary = {k: v for k, v in VOCABULARY.items() if k != "E_66"}
        with pytest.raises(ValueError, match="E_66"):
            load_ddxplus_kg(*_write(tmp_path, CONDITIONS_JSON, vocabulary))

    def test_code_used_as_symptom_and_risk_factor_is_a_data_error(self, tmp_path):
        bad = json.loads(json.dumps(CONDITIONS_JSON))
        bad[GERD]["antecedents"].append({"code": "E_66"})  # a symptom for PE
        with pytest.raises(ValueError, match="E_66"):
            load_ddxplus_kg(*_write(tmp_path, bad, VOCABULARY))


# --------------------------------------------------------------------------- #
# NetworkX store — the GraphStore contract
# --------------------------------------------------------------------------- #


class TestNetworkXGraphStore:
    def test_implements_the_graph_store_protocol(self, store):
        assert isinstance(store, GraphStore)

    def test_scores_every_condition_by_weighted_overlap(self, store):
        scores = store.score_by_connectivity(_case(present=["DDX:E_53", "DDX:E_66"]))
        assert set(scores) == {c.id for c in CONDITIONS}
        assert scores[PE] == pytest.approx(2 / 4)
        assert scores[GERD] == pytest.approx(1 / 4)
        assert scores["COND:aortic_dissection"] == 0.0, "no edges until hand-authored"

    def test_risk_factors_count_towards_the_score(self, store):
        scores = store.score_by_connectivity(_case(present=["DDX:E_53"], risk_factors=["DDX:E_2"]))
        assert scores[PE] == pytest.approx(2 / 4)

    def test_denied_findings_count_against(self, store):
        scores = store.score_by_connectivity(_case(present=["DDX:E_53"], absent=["DDX:E_66"]))
        assert scores[PE] == pytest.approx((1 - 0.5) / 4)
        only_denied = store.score_by_connectivity(_case(absent=["DDX:E_53"]))
        assert only_denied[PE] == 0.0, "clamped at zero"

    def test_paths_link_present_findings_and_risk_factors(self, store):
        case = _case(
            present=["DDX:E_53", "DDX:E_53", "DDX:E_10"],
            absent=["DDX:E_66"],
            risk_factors=["DDX:E_2"],
        )
        paths = store.paths_for(case, PE)
        assert [p.finding_id for p in paths] == ["DDX:E_53", "DDX:E_2"], "deduplicated"
        assert paths[0].path == ["label DDX:E_53", "HAS_SYMPTOM", "Pulmonary embolism"]
        assert paths[1].path == ["label DDX:E_2", "HAS_RISK_FACTOR", "Pulmonary embolism"]
        assert store.paths_for(case, "COND:aortic_dissection") == []

    def test_expected_findings_list_symptoms_then_risk_factors_in_code_order(self, store):
        expected = store.expected_findings(PE)
        assert [f.concept_id for f in expected] == ["DDX:E_53", "DDX:E_66", "DDX:E_2", "DDX:E_79"]
        assert all(f.assertion.value == "unknown" for f in expected)
        assert expected[1].label == "Short of breath?"
        assert store.expected_findings("COND:not_a_condition") == []

    def test_the_graph_is_read_only(self, store):
        with pytest.raises(nx.NetworkXError):
            store.graph.add_node("DDX:E_999")

    def test_parallel_edges_keep_their_sources_and_count_once(self):
        """The same fact from two sources must not be double-counted (R-12)."""
        nodes = {
            PE: KGNode(PE, NodeType.CONDITION, "Pulmonary embolism"),
            "DDX:E_53": KGNode("DDX:E_53", NodeType.SYMPTOM, "Pain"),
            "DDX:E_66": KGNode("DDX:E_66", NodeType.SYMPTOM, "Breathless"),
        }
        edges = [
            KGEdge(PE, "DDX:E_53", Relation.HAS_SYMPTOM, EdgeSource.DDXPLUS, 1.0),
            KGEdge(PE, "DDX:E_53", Relation.HAS_SYMPTOM, EdgeSource.BODHI_S, 0.5),
            KGEdge(PE, "DDX:E_66", Relation.HAS_SYMPTOM, EdgeSource.DDXPLUS, 1.0),
        ]
        store = NetworkXGraphStore(KnowledgeGraph(nodes=nodes, edges=edges))
        assert store.graph.number_of_edges(PE, "DDX:E_53") == 2
        scores = store.score_by_connectivity(_case(present=["DDX:E_53"]))
        assert scores[PE] == pytest.approx(1.0 / 2.0)


def test_drop_in_replacement_for_the_stub_in_the_pipeline(store):
    """The pipeline runs unchanged on the real store — the stub-first rule's promise."""
    pipeline = DiagnosisPipeline(
        ranker=ConstantRanker(),
        graph=store,
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
    )
    result = pipeline.run(_case(present=["DDX:E_53", "DDX:E_66"], risk_factors=["DDX:E_2"]))
    top = result.candidates[0]
    assert top.condition_id == PE
    assert {a.finding_id for a in top.assessments if a.role is EvidenceRole.SUPPORTING} == {
        "DDX:E_53",
        "DDX:E_66",
        "DDX:E_2",
    }
    assert top.paths
    assert "graph_backend" not in result.degraded_components
    assert result.disclaimer == DISCLAIMER


def test_build_script_fails_when_trainable_conditions_have_no_evidence(tmp_path):
    _write(tmp_path, CONDITIONS_JSON, VOCABULARY)  # only 2 of the 13 trainable conditions
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_cardiac_kg.py")]
        + ["--interim", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 1
    summary = json.loads((tmp_path / "cardiac_kg_summary.json").read_text("utf-8"))
    assert summary["edges_by_source"] == {"ddxplus": 8}
    assert len(summary["trainable_without_edges"]) == 11


# --------------------------------------------------------------------------- #
# The real graph (local only)
# --------------------------------------------------------------------------- #


@needs_real_data
class TestRealGraph:
    @pytest.fixture(scope="class")
    def real(self) -> NetworkXGraphStore:
        return NetworkXGraphStore.from_files(*REAL_FILES)

    def test_shape_matches_the_kg_card(self, real):
        """docs/02 §5.1. If this changes, update the card."""
        types = [data["type"] for _, data in real.graph.nodes(data=True)]
        assert (types.count("Condition"), types.count("Symptom"), types.count("RiskFactor")) == (
            14,
            41,
            43,
        )
        relations = [d["relation"] for _, _, d in real.graph.edges(data=True)]
        assert (relations.count("HAS_SYMPTOM"), relations.count("HAS_RISK_FACTOR")) == (160, 85)
        assert {d["source"] for _, _, d in real.graph.edges(data=True)} == {"ddxplus"}

    def test_every_trainable_condition_has_evidence_and_only_e16_is_odd(self, real):
        assert all(real.expected_findings(cid) for cid in trainable_ids())
        assert real.expected_findings("COND:aortic_dissection") == []
        assert len(real.anomalies) == 1 and real.anomalies[0].startswith("E_16 ")
