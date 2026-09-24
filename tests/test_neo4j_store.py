"""Tests for the Neo4j store (task 1b).

Most tests need no database. They cover the settings, the graph's round trip through plain rows,
its fingerprint, and the fallback to the in-memory store. The tests marked ``neo4j`` need a live
Neo4j: CI starts a throwaway one and sets NEO4J_TEST_URI (.github/workflows/ci.yml), and locally
they run against the instance in .env only when NEO4J_TEST_DOTENV=1. They work in a namespace of
their own and delete it afterwards, so they never touch the real graph.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from src.contracts import GraphStore, PatientCase
from src.medical_kg import neo4j_store
from src.medical_kg.bodhi_s import TRIPLES_FILE
from src.medical_kg.cardiac_kg import build_cardiac_kg
from src.medical_kg.loader import (
    EdgeSource,
    KGEdge,
    KGNode,
    KnowledgeGraph,
    Likelihood,
    NodeType,
    Relation,
)
from src.medical_kg.neo4j_store import (
    GraphIntegrityError,
    GraphOutOfDate,
    GraphUnavailable,
    Neo4jGraphStore,
    Neo4jSettings,
    delete_kg,
    describe_failure,
    kg_fingerprint,
    kg_from_rows,
    kg_to_rows,
    open_graph_store,
    read_kg,
    settings_from_env,
    write_kg,
)
from src.medical_kg.networkx_store import NetworkXGraphStore
from src.pipeline import DiagnosisPipeline
from src.stubs import ConstantRanker, EmptyRetriever, TemplateExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_FILES = (
    REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_evidences.json",
)
BODHI_DIR = REPO_ROOT / "data" / "raw" / "bodhi_s"
LOAD_SCRIPT = REPO_ROOT / "scripts" / "load_neo4j.py"

TEST_NAMESPACE = "NightingaleTest"
UNREACHABLE = Neo4jSettings(uri="bolt://127.0.0.1:9", user="neo4j", password="not-the-password")

PE = "COND:pulmonary_embolism"
GERD = "COND:gerd"


def small_kg() -> KnowledgeGraph:
    """Every kind of value the real graph holds: None, tuples, enums, lists, parallel edges."""
    nodes = {
        PE: KGNode(
            PE,
            NodeType.CONDITION,
            "Pulmonary embolism",
            {"category": "mimic", "is_must_not_miss": True, "icd10": "I26", "ddxplus_severity": 2},
        ),
        GERD: KGNode(GERD, NodeType.CONDITION, "GERD", {"is_must_not_miss": False, "icd10": None}),
        "DDX:E_53": KGNode("DDX:E_53", NodeType.SYMPTOM, "Pain?", {"label_fr": "Douleur ?"}),
        "DDX:E_66": KGNode("DDX:E_66", NodeType.SYMPTOM, "Short of breath?", {"data_type": "B"}),
        "SYM:calf_pain": KGNode(
            "SYM:calf_pain",
            NodeType.SYMPTOM,
            "Calf pain",
            {"ddxplus_code": None, "source": "bodhi_s"},
        ),
        "DDX:E_79": KGNode("DDX:E_79", NodeType.RISK_FACTOR, "Smoker?", {"ddxplus_code": "E_79"}),
    }
    edges = [
        KGEdge(PE, "DDX:E_66", Relation.HAS_SYMPTOM, EdgeSource.DDXPLUS),
        KGEdge(PE, "DDX:E_53", Relation.HAS_SYMPTOM, EdgeSource.DDXPLUS),
        KGEdge(PE, "DDX:E_79", Relation.HAS_RISK_FACTOR, EdgeSource.DDXPLUS),
        KGEdge(GERD, "DDX:E_53", Relation.HAS_SYMPTOM, EdgeSource.DDXPLUS),
        KGEdge(
            PE,
            "DDX:E_66",
            Relation.HAS_SYMPTOM,
            EdgeSource.BODHI_S,
            0.65,
            {
                "likelihood": Likelihood.HIGH,
                "p_condition_given_finding": None,
                "concepts": ("SYM:breathlessness",),
                "bodhi_ids": ["b-2", "b-1"],
            },
        ),
        KGEdge(
            PE,
            "SYM:calf_pain",
            Relation.HAS_SYMPTOM,
            EdgeSource.BODHI_S,
            0.12,
            {"likelihood": "low", "concepts": ["SYM:calf_pain"], "bodhi_ids": ["b-3"]},
        ),
        KGEdge(
            GERD,
            "DDX:E_79",
            Relation.HAS_RISK_FACTOR,
            EdgeSource.HAND_AUTHORED,
            0.35,
            {"likelihood": "medium", "evidence": "a paraphrase", "reference": "REF-1"},
        ),
    ]
    return KnowledgeGraph(nodes=nodes, edges=edges, anomalies=["E_16 is typed by its use"])


def _case(present=(), absent=()) -> PatientCase:
    findings = [{"concept_id": c, "label": f"label {c}"} for c in present]
    findings += [{"concept_id": c, "label": f"label {c}", "assertion": "absent"} for c in absent]
    return PatientCase(case_id="T-NEO", age=60, sex="M", findings=findings)


CASES = [
    _case(),
    _case(present=["DDX:E_66", "DDX:E_53"]),
    _case(present=["DDX:E_66", "SYM:calf_pain", "DDX:E_79"], absent=["DDX:E_53"]),
    _case(present=["DDX:E_53"], absent=["DDX:E_66", "DDX:E_79"]),
]


def assert_same_answers(store: GraphStore, reference: GraphStore, condition_ids) -> None:
    """The two stores give identical scores, paths and expected findings."""
    for case in CASES:
        assert store.score_by_connectivity(case) == reference.score_by_connectivity(case)
        for cid in condition_ids:
            assert store.paths_for(case, cid) == reference.paths_for(case, cid)
    for cid in condition_ids:
        assert store.expected_findings(cid) == reference.expected_findings(cid)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


class TestSettings:
    def test_reads_the_neo4j_variables(self):
        settings = settings_from_env(
            {
                "NEO4J_URI": " neo4j+s://abc.databases.neo4j.io ",
                "NEO4J_USER": "reader",
                "NEO4J_PASSWORD": "s3cret",
                "NEO4J_DATABASE": "cardiac",
            }
        )
        assert settings == Neo4jSettings(
            "neo4j+s://abc.databases.neo4j.io", "reader", "s3cret", "cardiac"
        )

    def test_accepts_the_spelling_of_auras_credentials_file(self):
        settings = settings_from_env(
            {"NEO4J_URI": "neo4j+s://x", "NEO4J_USERNAME": "abc", "NEO4J_PASSWORD": "pw"}
        )
        assert settings.user == "abc"

    def test_no_database_means_the_home_database(self):
        """On AuraDB the home database is named after the instance id, not 'neo4j'."""
        base = {"NEO4J_URI": "neo4j+s://x", "NEO4J_USER": "u", "NEO4J_PASSWORD": "pw"}
        assert settings_from_env(base).database is None
        assert settings_from_env({**base, "NEO4J_DATABASE": "  "}).database is None

    def test_missing_values_are_named(self):
        with pytest.raises(GraphUnavailable, match="NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"):
            settings_from_env({})

    def test_the_password_never_appears_in_repr(self):
        settings = Neo4jSettings("neo4j+s://x", "u", "p4ssw0rd-value")
        assert "p4ssw0rd-value" not in repr(settings)

    def test_the_environment_wins_over_dotenv_which_is_left_out_of_os_environ(
        self, tmp_path, monkeypatch
    ):
        for name in (
            "NEO4J_URI",
            "NEO4J_USER",
            "NEO4J_USERNAME",
            "NEO4J_PASSWORD",
            "NEO4J_DATABASE",
        ):
            monkeypatch.delenv(name, raising=False)
        dotenv = tmp_path / ".env"
        dotenv.write_text(
            "NEO4J_URI=neo4j+s://from-file\nNEO4J_USER=file-user\nNEO4J_PASSWORD=file-pw\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("NEO4J_URI", "neo4j+s://from-env")
        settings = settings_from_env(dotenv_path=dotenv)
        assert (settings.uri, settings.user) == ("neo4j+s://from-env", "file-user")
        assert "NEO4J_USER" not in os.environ


# --------------------------------------------------------------------------- #
# Rows and fingerprint
# --------------------------------------------------------------------------- #


class TestRows:
    def test_the_round_trip_rebuilds_the_graph_in_its_order(self):
        kg = small_kg()
        nodes, edges = kg_to_rows(kg)
        random.Random(0).shuffle(nodes)  # Neo4j returns rows in no particular order
        random.Random(1).shuffle(edges)
        back = kg_from_rows(nodes, edges, kg.anomalies)
        assert list(back.nodes) == list(kg.nodes)
        assert [(e.condition_id, e.concept_id, e.source) for e in back.edges] == [
            (e.condition_id, e.concept_id, e.source) for e in kg.edges
        ]
        assert back.anomalies == kg.anomalies
        assert kg_fingerprint(back) == kg_fingerprint(kg)

    def test_the_rebuilt_graph_answers_exactly_as_the_original(self):
        kg = small_kg()
        back = kg_from_rows(*kg_to_rows(kg))
        assert_same_answers(NetworkXGraphStore(back), NetworkXGraphStore(kg), [PE, GERD])

    def test_values_become_what_neo4j_can_store(self):
        nodes, edges = kg_to_rows(small_kg())
        gerd = next(row for row in nodes if row["id"] == GERD)
        assert "icd10" not in gerd, "Neo4j cannot store a null"
        bodhi = next(row for row in edges if row["props"]["source"] == "bodhi_s")
        assert bodhi["props"]["likelihood"] == "high"
        assert bodhi["props"]["concepts"] == ["SYM:breathlessness"]
        assert "p_condition_given_finding" not in bodhi["props"]
        assert bodhi["relation"] == "HAS_SYMPTOM" and bodhi["props"]["weight"] == 0.65

    def test_the_fingerprint_ignores_order_but_not_content(self):
        kg = small_kg()
        fingerprint = kg_fingerprint(kg)
        assert kg_fingerprint(replace(kg, edges=list(reversed(kg.edges)))) == fingerprint
        reweighted = [replace(kg.edges[0], weight=0.5), *kg.edges[1:]]
        assert kg_fingerprint(replace(kg, edges=reweighted)) != fingerprint
        relabelled = dict(kg.nodes)
        relabelled[GERD] = replace(kg.nodes[GERD], label="Reflux")
        assert kg_fingerprint(replace(kg, nodes=relabelled)) != fingerprint

    @pytest.mark.parametrize("properties", [{"type": "x"}, {"ordinal": 3}, {"nested": {"a": 1}}])
    def test_properties_the_store_cannot_hold_are_refused(self, properties):
        kg = small_kg()
        nodes = dict(kg.nodes)
        nodes[GERD] = replace(kg.nodes[GERD], properties=properties)
        with pytest.raises(ValueError):
            kg_to_rows(replace(kg, nodes=nodes))

    def test_edges_may_not_use_the_stores_own_names(self):
        kg = small_kg()
        edges = [replace(kg.edges[0], properties={"weight": 2.0}), *kg.edges[1:]]
        with pytest.raises(ValueError, match="reserved"):
            kg_to_rows(replace(kg, edges=edges))


# --------------------------------------------------------------------------- #
# Choosing a backend, and falling back
# --------------------------------------------------------------------------- #


class TestOpenGraphStore:
    def test_the_networkx_backend_serves_the_local_graph(self):
        store = open_graph_store("networkx", local_kg=small_kg())
        assert type(store) is NetworkXGraphStore
        assert (store.backend, store.degraded) == ("networkx", False)

    def test_the_networkx_backend_needs_a_local_graph(self):
        with pytest.raises(GraphUnavailable):
            open_graph_store("networkx")

    def test_an_unknown_backend_is_an_error(self):
        with pytest.raises(ValueError, match="unknown graph backend"):
            open_graph_store("sqlite", local_kg=small_kg())

    def test_falls_back_when_neo4j_is_not_configured(self, monkeypatch):
        def not_configured():
            raise GraphUnavailable("NEO4J_URI not set")

        monkeypatch.setattr(neo4j_store, "settings_from_env", not_configured)
        store = open_graph_store("neo4j", local_kg=small_kg())
        assert store.degraded and "NEO4J_URI not set" in store.degraded_reason

    def test_falls_back_when_neo4j_cannot_be_reached(self):
        kg = small_kg()
        store = open_graph_store("neo4j", local_kg=kg, settings=UNREACHABLE, timeout=3)
        assert type(store) is NetworkXGraphStore
        assert store.degraded and store.degraded_reason.startswith("ServiceUnavailable")
        assert "not-the-password" not in store.degraded_reason
        assert_same_answers(store, NetworkXGraphStore(kg), [PE, GERD])

    def test_without_neo4j_or_a_local_graph_there_is_no_graph(self):
        with pytest.raises(GraphUnavailable, match="no local graph"):
            open_graph_store("neo4j", settings=UNREACHABLE, timeout=3)

    def test_failure_descriptions_give_a_hint_and_no_credentials(self):
        assert "NEO4J_PASSWORD in .env" in describe_failure(AuthError("unauthorized"))
        assert "paused" in describe_failure(ServiceUnavailable("could not connect"))
        assert describe_failure(RuntimeError()) == "RuntimeError"


def test_the_pipeline_reports_a_fallback_but_still_uses_it():
    """docs/02 §7: a stand-in graph still ranks and explains; the result says it is a stand-in."""
    store = NetworkXGraphStore(small_kg(), degraded_reason="ServiceUnavailable: offline")
    pipeline = DiagnosisPipeline(
        ranker=ConstantRanker(),
        graph=store,
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
    )
    case = _case(present=["DDX:E_66", "SYM:calf_pain"])  # PE's findings only
    result = pipeline.run(case)
    assert "graph_backend" in result.degraded_components
    top = result.candidates[0]
    graph_scores = store.score_by_connectivity(case)
    assert top.condition_id == PE == max(graph_scores, key=graph_scores.get), "the graph ranked it"
    assert top.paths and top.assessments


def test_the_load_script_reports_an_unreachable_neo4j_without_leaking_the_password(tmp_path):
    env = {
        **os.environ,
        "NEO4J_URI": UNREACHABLE.uri,
        "NEO4J_USER": UNREACHABLE.user,
        "NEO4J_PASSWORD": UNREACHABLE.password,
    }
    result = subprocess.run(
        [sys.executable, str(LOAD_SCRIPT), "--interim", str(tmp_path), "--timeout", "3"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    assert result.returncode == 2
    assert "unavailable" in result.stdout
    assert UNREACHABLE.password not in result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# Against a live Neo4j
# --------------------------------------------------------------------------- #


def _live_settings() -> Neo4jSettings | None:
    if os.environ.get("NEO4J_TEST_URI"):
        return Neo4jSettings(
            uri=os.environ["NEO4J_TEST_URI"],
            user=os.environ.get("NEO4J_TEST_USER", "neo4j"),
            password=os.environ.get("NEO4J_TEST_PASSWORD", ""),
            database=os.environ.get("NEO4J_TEST_DATABASE") or None,
        )
    if os.environ.get("NEO4J_TEST_DOTENV") == "1":
        return settings_from_env()
    return None


@pytest.fixture(scope="module")
def live():
    """(settings, driver) for a live Neo4j, with the test namespace empty before and after."""
    settings = _live_settings()
    if settings is None:
        pytest.skip(
            "no test Neo4j: CI sets NEO4J_TEST_URI; set NEO4J_TEST_DOTENV=1 to use the .env instance"
        )
    driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))
    deadline = time.monotonic() + 90  # CI's container may still be starting
    while True:
        try:
            driver.verify_connectivity()
            break
        except ServiceUnavailable:
            if time.monotonic() > deadline:
                driver.close()
                raise
            time.sleep(2)
    delete_kg(driver, database=settings.database, namespace=TEST_NAMESPACE)
    yield settings, driver
    delete_kg(driver, database=settings.database, namespace=TEST_NAMESPACE)
    driver.close()


@pytest.fixture
def empty(live):
    settings, driver = live
    delete_kg(driver, database=settings.database, namespace=TEST_NAMESPACE)
    return settings, driver


def _connect(settings, **kwargs) -> Neo4jGraphStore:
    return Neo4jGraphStore.connect(settings, namespace=TEST_NAMESPACE, **kwargs)


@pytest.mark.neo4j
class TestLiveNeo4j:
    def test_write_then_read_round_trips_the_graph(self, empty):
        settings, driver = empty
        kg = small_kg()
        fingerprint = write_kg(driver, kg, database=settings.database, namespace=TEST_NAMESPACE)
        stored = read_kg(driver, database=settings.database, namespace=TEST_NAMESPACE)
        assert stored.fingerprint == stored.recorded == fingerprint == kg_fingerprint(kg)
        assert list(stored.kg.nodes) == list(kg.nodes)
        assert stored.kg.anomalies == kg.anomalies

    def test_an_empty_namespace_reads_as_no_graph(self, empty):
        settings, driver = empty
        assert read_kg(driver, database=settings.database, namespace=TEST_NAMESPACE) is None
        with pytest.raises(GraphUnavailable, match="holds no"):
            _connect(settings)

    def test_connect_writes_once_then_serves_the_same_answers(self, empty):
        settings, _ = empty
        kg = small_kg()
        first = _connect(settings, local_kg=kg)
        second = _connect(settings, local_kg=kg)
        assert (first.wrote, second.wrote) == (True, False)
        assert second.fingerprint == kg_fingerprint(kg)
        assert second.backend == "neo4j" and not second.degraded
        assert second.server.startswith("Neo4j/")
        assert_same_answers(second, NetworkXGraphStore(kg), [PE, GERD])
        assert _connect(settings).fingerprint == second.fingerprint, "no local graph: stored copy"

    def test_an_edit_made_in_neo4j_is_caught_and_undone(self, empty):
        settings, driver = empty
        kg = small_kg()
        write_kg(driver, kg, database=settings.database, namespace=TEST_NAMESPACE)
        driver.execute_query(
            f"MATCH (:{TEST_NAMESPACE} {{id: $c}})-[r {{source: 'ddxplus'}}]->"
            f"(:{TEST_NAMESPACE} {{id: $x}}) SET r.weight = 0.5",
            c=PE,
            x="DDX:E_53",
            database_=settings.database,
        )
        with pytest.raises(GraphIntegrityError):
            _connect(settings)
        repaired = _connect(settings, local_kg=kg)
        assert repaired.wrote and repaired.fingerprint == kg_fingerprint(kg)

    def test_a_changed_local_graph_is_written_unless_sync_is_off(self, empty):
        settings, driver = empty
        kg = small_kg()
        write_kg(driver, kg, database=settings.database, namespace=TEST_NAMESPACE)
        changed = replace(kg, edges=[replace(kg.edges[0], weight=0.9), *kg.edges[1:]])
        with pytest.raises(GraphOutOfDate):
            _connect(settings, local_kg=changed, sync=False)
        store = _connect(settings, local_kg=changed)
        assert store.wrote and store.fingerprint == kg_fingerprint(changed)

    def test_open_graph_store_uses_neo4j_when_it_answers(self, empty):
        settings, _ = empty
        store = open_graph_store(
            "neo4j", local_kg=small_kg(), settings=settings, namespace=TEST_NAMESPACE
        )
        assert type(store) is Neo4jGraphStore and not store.degraded

    def test_a_refused_login_falls_back(self, live):
        settings, _ = live
        wrong = replace(settings, password=settings.password + "-wrong")
        store = open_graph_store(
            "neo4j", local_kg=small_kg(), settings=wrong, namespace=TEST_NAMESPACE, timeout=5
        )
        assert store.degraded and "login was refused" in store.degraded_reason

    def test_the_load_script_reports_the_stored_graph(self, empty, tmp_path):
        settings, driver = empty
        write_kg(driver, small_kg(), database=settings.database, namespace=TEST_NAMESPACE)
        result = _run_load_script(settings, "--interim", str(tmp_path))
        assert result.returncode == 0, result.stdout
        assert "Local graph: none" in result.stdout and "7 edges" in result.stdout

    @pytest.mark.skipif(
        not all(p.exists() for p in REAL_FILES), reason="data/ is not committed (CI)"
    )
    def test_the_real_graph_loads_and_answers_the_golden_cases_identically(self, empty):
        settings, _ = empty
        result = _run_load_script(settings)
        assert result.returncode == 0, result.stdout
        assert "identical" in result.stdout and "by this run" in result.stdout
        again = _run_load_script(settings, "--check")
        assert again.returncode == 0, again.stdout
        assert "already current" in again.stdout
        bodhi = BODHI_DIR if (BODHI_DIR / TRIPLES_FILE).exists() else None
        stored = _connect(settings)
        assert stored.fingerprint == kg_fingerprint(build_cardiac_kg(*REAL_FILES, bodhi))


def _run_load_script(settings: Neo4jSettings, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "NEO4J_URI": settings.uri,
        "NEO4J_USER": settings.user,
        "NEO4J_PASSWORD": settings.password,
        "NEO4J_DATABASE": settings.database or "",
    }
    return subprocess.run(
        [sys.executable, str(LOAD_SCRIPT), "--namespace", TEST_NAMESPACE, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
