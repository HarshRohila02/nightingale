"""Tests for the HTTP service (src/api/app.py, task 1e).

The golden clinical cases go through ``POST /diagnose`` with the stub pipeline everywhere, and with
the configured pipeline (the real graph and model) where data/ and models/ exist.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import yaml

from src.api import create_app
from src.conditions import CONDITIONS
from src.contracts import DISCLAIMER
from src.pipeline import DiagnosisPipeline
from src.stubs import ConstantRanker, EmptyRetriever, InMemoryGraphStore, TemplateExplainer

with warnings.catch_warnings():  # starlette prefers httpx2 for its test client; httpx still works
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
GOLDEN_CASES = yaml.safe_load(
    (REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml").read_text(encoding="utf-8")
)["cases"]
REAL = [
    REPO_ROOT / "data" / "raw" / "ddxplus" / "release_evidences.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_evidences.json",
    REPO_ROOT / "models" / "nightingale_b1_dl" / "b1_dl_asked_aug" / "seed42" / "b1_dl.json",
]


def stub_client() -> TestClient:
    pipeline = DiagnosisPipeline(
        ranker=ConstantRanker(),
        graph=InMemoryGraphStore(),
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
    )
    return TestClient(create_app(pipeline, about={"ranker": "stub", "graph": "stub"}))


def check_golden(response, golden: dict) -> None:
    assert response.status_code == 200, response.text
    body = response.json()
    expect = golden["expect"]
    top = [c["condition_id"] for c in body["candidates"][: expect.get("top_k", 3)]]
    for required in expect.get("must_include", []):
        assert required in top, f"{golden['id']}: {required} not in the top {len(top)}: {top}"
    flagged = {c["condition_id"] for c in body["candidates"] if c["red_flag"]}
    for required in expect.get("red_flag_for", []):
        assert required in flagged, f"{golden['id']}: no red flag for {required}"
    if expect.get("no_red_flag"):
        assert not flagged
    assert body["disclaimer"] == DISCLAIMER
    assert float(response.headers["X-Elapsed-Ms"]) >= 0


def test_health_names_what_answers_and_repeats_the_disclaimer():
    body = stub_client().get("/health").json()
    assert body["status"] == "ok" and body["degraded_components"] == []
    assert body["ranker"] == "stub" and body["disclaimer"] == DISCLAIMER


def test_health_reports_a_degraded_ranker():
    from src.ml.ranker import open_ranker

    pipeline = DiagnosisPipeline(ranker=open_ranker("none"), graph=InMemoryGraphStore())
    body = TestClient(create_app(pipeline)).get("/health").json()
    assert body["status"] == "degraded" and body["degraded_components"] == ["ml"]


def test_conditions_are_the_fourteen_the_system_knows():
    body = stub_client().get("/conditions").json()
    assert [c["id"] for c in body] == [c.id for c in CONDITIONS] and len(body) == 14
    dissection = next(c for c in body if c["id"] == "COND:aortic_dissection")
    assert dissection["must_not_miss"] and not dissection["in_training_data"]


def test_concepts_offer_the_hand_authored_findings():
    body = stub_client().get("/concepts").json()
    by_id = {c["concept_id"]: c for c in body}
    assert by_id["SYM:chest_pain"]["label"] == "Chest pain"
    assert by_id["SYM:chest_pain"]["kind"] == "finding"
    assert all(c["kind"] == "risk_factor" for c in body if c["concept_id"].startswith("RF:"))


@pytest.mark.golden
@pytest.mark.parametrize("golden", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
def test_golden_cases_through_the_api_with_the_stubs(golden):
    check_golden(stub_client().post("/diagnose", json=golden["case"]), golden)


def test_a_malformed_case_is_refused():
    client = stub_client()
    assert (
        client.post("/diagnose", json={"case_id": "x", "age": 200, "sex": "M"}).status_code == 422
    )
    assert client.post("/diagnose", json={"case_id": "x", "age": 50, "sex": "X"}).status_code == 422


@pytest.mark.golden
@pytest.mark.skipif(
    not all(p.exists() for p in REAL), reason="data/ and models/ are not committed (CI)"
)
@pytest.mark.parametrize("golden", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
def test_golden_cases_through_the_api_with_the_configured_pipeline(golden):
    from _config import build_pipeline, load_config

    built = build_pipeline(load_config(), "networkx")
    client = TestClient(create_app(built["pipeline"], labels=built["labels"], about=built["about"]))
    response = client.post("/diagnose", json=golden["case"])
    check_golden(response, golden)
    assert "ml" not in response.json()["degraded_components"]


# --------------------------------------------------------------------------- #
# 3d: what to ask next
# --------------------------------------------------------------------------- #


def test_diagnose_suggests_the_cited_tests():
    """``suggested_next_tests``: the investigations for the leading and red-flagged candidates,
    each saying "consider" and ending with its guideline's tag (src/reasoning/investigations.py)."""
    pe = next(g for g in GOLDEN_CASES if g["id"] == "GC-002")
    body = stub_client().post("/diagnose", json=pe["case"]).json()
    tests = body["suggested_next_tests"]
    assert tests and all(": consider " in t and t.endswith(")") for t in tests)
    assert any(t.startswith("Pulmonary embolism") and "D-dimer" in t for t in tests)


@pytest.mark.parametrize("golden", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
def test_next_questions_never_asks_what_is_recorded(golden):
    response = stub_client().post("/next-questions", json=golden["case"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["case_id"] == golden["case"]["case_id"]
    assert body["disclaimer"] == DISCLAIMER
    assert body["entropy_bits"] is not None and body["questions"]
    recorded = {
        f["concept_id"]
        for f in [*golden["case"].get("findings", []), *golden["case"].get("risk_factors", [])]
    }
    for question in body["questions"]:
        assert question["concept_id"] not in recorded
        assert question["gain_bits"] > 0
        assert question["if_present"] or question["if_denied"]
    assert float(response.headers["X-Elapsed-Ms"]) >= 0


def test_next_questions_from_a_failed_graph_are_none_rather_than_a_guess():
    class FailedGraph:
        def score_by_connectivity(self, case):  # noqa: ARG002
            raise RuntimeError("the graph is gone")

    pipeline = DiagnosisPipeline(ranker=ConstantRanker(), graph=FailedGraph())
    body = TestClient(create_app(pipeline)).post("/next-questions", json=GOLDEN_CASES[0]["case"])
    assert body.json()["questions"] == [] and body.json()["entropy_bits"] is None
    assert body.json()["disclaimer"] == DISCLAIMER


def test_next_questions_refuses_a_malformed_case():
    response = stub_client().post("/next-questions", json={"case_id": "x", "age": 200, "sex": "M"})
    assert response.status_code == 422
