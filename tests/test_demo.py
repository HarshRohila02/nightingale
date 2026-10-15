"""scripts/demo.py, the walking skeleton, end to end (the W3 milestone).

Run as a subprocess, the way the owner runs it. Every run names ``--graph``, so no test ever
tries AuraDB. The stub run needs nothing and runs in CI; the real-graph run skips when data/ is
absent, as it is in CI.

Since ``v0.2-prototype``'s work (docs/17 T1) the demo builds its pipeline through
``build_pipeline`` and prints the evidence (3a), the questions to ask next and the tests to
consider (3d) below the differential. The tests here check those blocks against the code that
computes them, and that building the pipeline that way changed no ranking.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from app.view import EVIDENCE_OFF
from src.contracts import PatientCase
from src.medical_kg.crosswalk import expand_case
from src.pipeline import DiagnosisPipeline
from src.reasoning.next_questions import next_questions
from src.stubs import EmptyRetriever, InMemoryGraphStore, TemplateExplainer
from tests.rag_fixture import write_fake_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _config import (  # noqa: E402
    build_pipeline,
    fusion_weights_from_config,
    load_config,
    open_graph_from_config,
    open_ranker_from_config,
)

DEMO = REPO_ROOT / "scripts" / "demo.py"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"
KG_FILES = (
    REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_evidences.json",
)
needs_kg = pytest.mark.skipif(
    not all(p.exists() for p in KG_FILES), reason="data/ is not committed (CI)"
)
GOLDEN_IDS = [c["id"] for c in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]]


def run_demo(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DEMO), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )


def golden_case(case_id: str) -> dict:
    cases = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]
    return next(c for c in cases if c["id"] == case_id)["case"]


def recorded_labels(case_id: str) -> set[str]:
    """The labels of the findings and risk factors a golden case records."""
    case = golden_case(case_id)
    return {f["label"] for key in ("findings", "risk_factors") for f in case.get(key, [])}


def flat(text: str) -> str:
    """``text`` with its whitespace collapsed, so a line the demo wrapped still matches."""
    return " ".join(text.split())


def section(stdout: str, title: str) -> str:
    """The lines of the section headed ``title``, up to the next section's rule, collapsed."""
    lines = stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(title)) + 2
    end = next(
        (i for i in range(start, len(lines)) if lines[i].strip().startswith("---")), len(lines)
    )
    return flat("\n".join(lines[start:end]))


def why_section(stdout: str) -> list[str]:
    """The lines under "WHY <top candidate>?"."""
    lines = stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("WHY ")) + 2
    end = next(i for i in range(start, len(lines)) if lines[i].strip().startswith("---"))
    return [line.strip() for line in lines[start:end] if line.strip()]


def test_the_stub_graph_runs_anywhere():
    result = run_demo("--graph", "stub", "--ranker", "none")
    assert result.returncode == 0, result.stderr
    assert "Graph: the stub" in result.stdout
    assert "not a diagnosis" in result.stdout, "the disclaimer is never dropped"


def test_without_a_graph_it_says_how_to_get_one(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"paths": {"interim": str(tmp_path), "raw": str(tmp_path)}}),
        encoding="utf-8",
    )
    result = run_demo("--config", str(config), "--graph", "networkx", "--ranker", "none")
    assert result.returncode == 1
    assert "No knowledge graph" in result.stderr
    assert "--graph stub" in result.stderr


@needs_kg
@pytest.mark.golden
def test_the_real_graph_explains_with_what_the_case_recorded():
    """GC-001 on the real graph. The case is expanded through the crosswalk for the graph, which
    links conditions to DDXPlus *questions*. A supporting line is a finding the case recorded, or a
    question the graph links to the condition with the recorded finding that answered it, never
    the answer alone (docs/02 §5.2). Only a handful of the two dozen unrecorded ones are listed."""
    result = run_demo("--case", "GC-001", "--graph", "networkx")
    assert result.returncode == 0, result.stderr
    assert "Graph: networkx" in result.stdout and "hand_authored" in result.stdout
    assert "graph_backend" not in result.stdout, "networkx was asked for, so nothing stands in"

    why = why_section(result.stdout)
    recorded = recorded_labels("GC-001")
    supporting = [line[2:] for line in why if line.startswith(("✓", "+"))]
    direct = [line for line in supporting if "<-" not in line]
    through = [[s.strip() for s in line.split("<-")] for line in supporting if "<-" in line]
    assert direct and set(direct) <= recorded, direct
    assert through and all(q not in recorded and a in recorded for q, a in through), through
    assert len([line for line in why if line.startswith("?")]) <= 6


# --------------------------------------------------------------------------- #
# The evidence, the questions to ask next and the tests to consider (docs/17 T1)
# --------------------------------------------------------------------------- #


def without_a_bundle(tmp_path: Path) -> tuple[Path, dict]:
    """The committed configuration with retrieval off, written to ``tmp_path``: on the laptop
    ``retrieval.bundle`` is set once the corpus is unzipped (docs/11 §4.6), and these tests are
    about the run without one."""
    config = load_config()
    config["retrieval"] = {**(config.get("retrieval") or {}), "bundle": None}
    config.setdefault("ml", {})["backend"] = "none"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config


def test_the_stub_run_prints_evidence_questions_and_tests(tmp_path):
    """On the stub, without a corpus bundle: the evidence block says why it is empty, and the
    questions and tests are exactly those the code computes for the case."""
    path, config = without_a_bundle(tmp_path)
    result = run_demo("--config", str(path), "--graph", "stub", "--case", "GC-001")
    assert result.returncode == 0, result.stderr
    assert "retrieval: none" in result.stdout
    assert "rag" in result.stdout.split("Degraded components:")[1].splitlines()[0]

    assert section(result.stdout, "EVIDENCE") == flat(EVIDENCE_OFF)

    steps = section(result.stdout, "WHAT TO ASK NEXT")
    ranking = next_questions(PatientCase(**golden_case("GC-001")), InMemoryGraphStore())
    assert ranking.questions, "the stub graph has questions for GC-001"
    asked = [
        f"? {'Check' if q.kind == 'examination' else 'Ask about'} {q.label}"
        for q in ranking.questions
    ]
    positions = [steps.find(line) for line in asked]
    assert -1 not in positions and positions == sorted(positions), (asked, steps)

    built = build_pipeline(config, "stub")
    tests = built["pipeline"].run(PatientCase(**golden_case("GC-001"))).suggested_next_tests
    assert tests and all(flat(f"- {line}") in steps for line in tests)
    assert "The clinician decides." in steps


def test_every_golden_case_prints_its_tests_on_the_stub(tmp_path):
    path, _ = without_a_bundle(tmp_path)
    for case_id in GOLDEN_IDS:
        result = run_demo("--config", str(path), "--graph", "stub", "--case", case_id)
        assert result.returncode == 0, (case_id, result.stderr)
        steps = section(result.stdout, "WHAT TO ASK NEXT")
        assert "Tests to consider" in steps and "consider " in steps.split("Tests to")[1]
        assert "not a diagnosis" in result.stdout


def test_evidence_is_printed_with_its_citation_and_pmc_link(tmp_path):
    """With a bundle (the tests' invented one, tests/rag_fixture.py), each excerpt is shown
    with its citation, its PMC page, its title, attribution and licence, and nothing degrades."""
    config = tmp_path / "config.yaml"
    bundle = write_fake_bundle(tmp_path / "bundle")
    config.write_text(yaml.safe_dump({"retrieval": {"bundle": str(bundle)}}), encoding="utf-8")
    result = run_demo(
        "--config", str(config), "--graph", "stub", "--ranker", "none", "--case", "GC-002"
    )
    assert result.returncode == 0, result.stderr
    assert "retrieval: BM25 + dense" in result.stdout
    assert "rag" not in result.stdout.split("Degraded components:")[-1].splitlines()[0]

    evidence = section(result.stdout, "EVIDENCE")
    assert evidence.startswith('Pulmonary embolism - "Pulmonary embolism presents')
    assert "PMCID:PMC99990002 https://pmc.ncbi.nlm.nih.gov/articles/PMC99990002/" in evidence
    assert "Pulmonary embolism: an invented review | Doe J et al." in evidence
    assert "CC BY-NC 4.0" in evidence
    assert "http" not in evidence.replace("https://pmc.ncbi.nlm.nih.gov/articles/", "")


def _old_demo_result(config: dict, backend: str, case_id: str):
    """The result as scripts/demo.py built it before T1: its own pipeline, no retriever."""
    ranker = open_ranker_from_config(config)
    graph = open_graph_from_config(config, backend)
    case = PatientCase(**golden_case(case_id))
    if hasattr(graph, "graph"):
        case = expand_case(case, dict(graph.graph.nodes(data="label")))
    pipeline = DiagnosisPipeline(
        ranker=ranker,
        graph=graph,
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(ranker=ranker),
        **fusion_weights_from_config(config),
    )
    return pipeline.run(case)


def _new_demo_result(config: dict, backend: str, case_id: str):
    built = build_pipeline(config, backend)
    case = PatientCase(**golden_case(case_id))
    if built["labels"] is not None:
        case = expand_case(case, built["labels"])
    return built["pipeline"].run(case)


def _ranking(result) -> list[tuple]:
    return [
        (c.condition_id, c.fused_score, c.ml_score, c.kg_score, c.red_flag)
        for c in result.candidates
    ]


@pytest.mark.parametrize("backend", ["stub", pytest.param("networkx", marks=needs_kg)])
@pytest.mark.parametrize("ranker", [None, "none"])
@pytest.mark.parametrize("with_bundle", [False, True])
def test_building_the_pipeline_through_the_config_changed_no_ranking(
    backend, ranker, with_bundle, tmp_path
):
    """The demo now builds the pipeline the dashboard and the API run. Evidence is attached after
    ranking and the configured calibrator is none, so every golden case ranks exactly as it did:
    the same order, scores and red flags, with and without a corpus bundle."""
    config = load_config()
    if ranker:
        config.setdefault("ml", {})["backend"] = ranker
    bundle = str(write_fake_bundle(tmp_path / "bundle")) if with_bundle else None
    config["retrieval"] = {**(config.get("retrieval") or {}), "bundle": bundle}
    for case_id in GOLDEN_IDS:
        old = _old_demo_result(config, backend, case_id)
        new = _new_demo_result(config, backend, case_id)
        assert _ranking(new) == _ranking(old), case_id
        assert new.red_flags == old.red_flags, case_id
        assert ("rag" in new.degraded_components) is not with_bundle, case_id
