"""Tests for the dashboard (task 2e): its logic (app/view.py) everywhere, the app itself headless
where Streamlit and the real data are installed."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from app.view import build_case, candidate_rows, concept_options, example_cases, ranks_of
from src.contracts import Assertion, DiagnosisResult
from src.pipeline import DiagnosisPipeline
from src.stubs import ConstantRanker, EmptyRetriever, InMemoryGraphStore, TemplateExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "app" / "streamlit_app.py"
REAL = [
    REPO_ROOT / "data" / "raw" / "ddxplus" / "release_evidences.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_evidences.json",
    REPO_ROOT / "models" / "nightingale_b1_dl" / "b1_dl_asked_aug" / "seed42" / "b1_dl.json",
]


def stub_pipeline() -> DiagnosisPipeline:
    return DiagnosisPipeline(
        ranker=ConstantRanker(),
        graph=InMemoryGraphStore(),
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
    )


def test_the_form_offers_findings_and_risk_factors_by_label():
    options = concept_options()
    assert ("SYM:chest_pain", "Chest pain") in options["finding"]
    assert all(cid.startswith("RF:") for cid, _ in options["risk_factor"])
    labels = [label.lower() for _, label in options["finding"]]
    assert labels == sorted(labels)


def test_a_case_keeps_present_and_denied_apart_and_routes_risk_factors():
    case = build_case(60, "F", ["SYM:chest_pain", "RF:smoking"], ["SYM:diaphoresis"])
    assert [(f.concept_id, f.assertion) for f in case.findings] == [
        ("SYM:chest_pain", Assertion.PRESENT),
        ("SYM:diaphoresis", Assertion.ABSENT),
    ]
    assert [f.concept_id for f in case.risk_factors] == ["RF:smoking"]
    assert case.findings[0].label == "Chest pain"


def test_a_concept_cannot_be_both_present_and_denied_or_unknown():
    with pytest.raises(ValueError, match="both present and denied"):
        build_case(60, "F", ["SYM:chest_pain"], ["SYM:chest_pain"])
    with pytest.raises(ValueError, match="not concepts"):
        build_case(60, "F", ["SYM:not_a_concept"])


def test_ranks_share_a_tie():
    assert ranks_of({"a": 0.5, "b": 0.9, "c": 0.5}) == {"b": 1, "a": 2, "c": 2}


def test_rows_give_ranks_and_say_why_dissection_has_no_model_rank():
    golden = example_cases()["GC-003"]
    case = build_case(golden["age"], golden["sex"], golden["present"], golden["denied"])
    rows = candidate_rows(stub_pipeline().run(case), case)
    assert [r["rank"] for r in rows] == list(range(1, 15))
    dissection = next(r for r in rows if r["condition_id"] == "COND:aortic_dissection")
    assert (
        dissection["model_rank"] is None and "not in the training data" in dissection["model_note"]
    )
    assert dissection["must_not_miss"] and dissection["red_flag"]
    trained = [r for r in rows if r["condition_id"] != "COND:aortic_dissection"]
    assert all(r["model_rank"] is not None and r["graph_rank"] is not None for r in trained)


def test_a_failed_graph_shows_no_graph_rank_rather_than_the_best_score():
    """2e's note: a failed graph store gave kg_score 0.0 everywhere, the best graph score."""

    class Broken(InMemoryGraphStore):
        def score_by_connectivity(self, case):
            raise RuntimeError("down")

    pipeline = DiagnosisPipeline(ranker=ConstantRanker(), graph=Broken())
    result = pipeline.run(build_case(50, "M", ["SYM:chest_pain"]))
    assert "graph_backend" in result.degraded_components
    assert all(r["graph_rank"] is None for r in candidate_rows(result))


def test_a_failed_model_shows_no_model_rank():
    result = DiagnosisResult(case_id="x", candidates=[], degraded_components=["ml"])
    assert candidate_rows(result) == []
    from src.ml.ranker import open_ranker

    pipeline = DiagnosisPipeline(ranker=open_ranker("none"), graph=InMemoryGraphStore())
    rows = candidate_rows(pipeline.run(build_case(50, "M", ["SYM:chest_pain"])))
    assert all(r["model_rank"] is None for r in rows)


def test_the_examples_are_the_golden_cases_with_their_denials():
    examples = example_cases()
    assert list(examples) == ["GC-001", "GC-002", "GC-003", "GC-004"]
    assert "SYM:exertional" in examples["GC-004"]["denied"]
    assert "RF:diabetes" in examples["GC-001"]["present"]


@pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None or not all(p.exists() for p in REAL),
    reason="needs Streamlit and data/ and models/ (not in CI)",
)
def test_the_app_runs_and_follows_an_example(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NIGHTINGALE_GRAPH", "networkx")  # never reach for Aura in a test
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception, app.exception
    assert any("decision support" in w.value.lower() for w in app.warning)  # the disclaimer
    app.selectbox(key="example").select("GC-003").run()
    assert not app.exception, app.exception
    table = app.dataframe[0].value
    assert table.iloc[0]["Condition"] == "Aortic dissection"
    assert any("Red flag" in e.value for e in app.error)
    assert os.environ["NIGHTINGALE_GRAPH"] == "networkx"
