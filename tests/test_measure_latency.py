"""scripts/measure_latency.py (EXP-022) with the configured retriever (docs/17 T2).

The script times the trained models on validate rows, so a whole run needs data/ and models/; CI
has neither. These tests drive its pipeline timing on the stubs instead: the stub ranker and graph,
and three retrievers (EXP-022's stub, the tests' invented corpus bundle, and an absent bundle). They
check that retrieval is its own stage when it retrieves, that an absent bundle degrades to ``rag``
and costs nothing, that ``--empty-retriever`` keeps EXP-022's setting, that the questions to ask
next are timed outside the pipeline's total, and that the ranking never depends on the retriever.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import src.pipeline as pipeline_module
from src.rag.retriever import HybridRetriever, open_retriever
from src.stubs import ConstantRanker, EmptyRetriever, InMemoryGraphStore
from tests.rag_fixture import write_fake_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "measure_latency.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import measure_latency  # noqa: E402

GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"
CASES = {c["id"]: c for c in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]}
FEW = SimpleNamespace(warmup=1, pipeline_calls=3)


def timed(retriever=None) -> dict:
    return measure_latency.time_pipeline(
        ConstantRanker(), InMemoryGraphStore(), {}, CASES, FEW, retriever
    )


@pytest.fixture(scope="module")
def bundle_retriever(tmp_path_factory) -> HybridRetriever:
    retriever = open_retriever(write_fake_bundle(tmp_path_factory.mktemp("bundle")))
    assert isinstance(retriever, HybridRetriever)
    return retriever


def test_exp022_s_stub_keeps_retrieval_out_of_the_stages():
    """``--empty-retriever``: as EXP-022 was first measured, nothing degraded, no stage added."""
    for case_id, r in timed().items():
        assert "retrieval" not in r["mean_ms_by_stage"], case_id
        assert "rag" not in r["degraded"], case_id
        assert r["evidence_passages"] == 0
        assert {
            "ranker",
            "graph",
            "fusion",
            "red_flags",
            "explanation",
            "safety_check",
            "rest",
        } <= set(r["mean_ms_by_stage"])


def test_with_a_bundle_retrieval_is_its_own_stage(bundle_retriever):
    results = timed(bundle_retriever)
    for case_id, r in results.items():
        assert r["mean_ms_by_stage"]["retrieval"] > 0, case_id
        assert "rag" not in r["degraded"], case_id
    assert results["GC-002"]["evidence_passages"] > 0, "the invented corpus has PE passages"


def test_without_a_bundle_retrieval_degrades_and_costs_nothing(tmp_path):
    for case_id, r in timed(open_retriever(tmp_path / "absent")).items():
        assert "rag" in r["degraded"], case_id
        assert "retrieval" not in r["mean_ms_by_stage"], case_id
        assert r["evidence_passages"] == 0


def test_the_retriever_never_changes_the_ranking(bundle_retriever, tmp_path):
    tops = [
        {case_id: r["top"] for case_id, r in timed(retriever).items()}
        for retriever in (None, bundle_retriever, open_retriever(tmp_path / "absent"))
    ]
    assert tops[0] == tops[1] == tops[2]


def test_the_questions_are_timed_apart_from_the_pipeline():
    for r in timed().values():
        assert r["questions_ms"]["calls"] == FEW.pipeline_calls
        assert r["questions_ms"]["p95"] >= 0
        assert "questions" not in r["mean_ms_by_stage"]


def test_the_pipeline_s_functions_are_restored_after_timing():
    originals = (
        pipeline_module.fuse_scores,
        pipeline_module.evaluate_red_flags,
        pipeline_module.safety_check,
    )
    timed()
    assert originals == (
        pipeline_module.fuse_scores,
        pipeline_module.evaluate_red_flags,
        pipeline_module.safety_check,
    )


def test_the_retriever_is_the_configured_one_unless_asked_otherwise(tmp_path):
    retriever, about = measure_latency.retriever_for({}, empty=True)
    assert type(retriever) is EmptyRetriever and "EXP-022's stub" in about[0]

    retriever, about = measure_latency.retriever_for({"retrieval": {"bundle": None}})
    assert retriever.degraded
    assert about[0].startswith("retrieval: none") and "without evidence" in about[1]

    bundle = write_fake_bundle(tmp_path / "bundle")
    retriever, about = measure_latency.retriever_for({"retrieval": {"bundle": str(bundle)}})
    assert isinstance(retriever, HybridRetriever) and len(about) == 1
    assert about[0].startswith("retrieval: BM25 + dense")


def test_a_run_without_data_stops_and_says_why(tmp_path):
    """Without data/ and models/ nothing can be timed: the script names what is missing and
    exits, rather than failing with a traceback."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--interim",
            str(tmp_path),
            "--raw-dir",
            str(tmp_path),
            "--calls",
            "1",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "B1-LR′+aug:" in result.stderr and "Traceback" not in result.stderr
