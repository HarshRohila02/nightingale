"""scripts/demo.py, the walking skeleton, end to end (the W3 milestone).

Run as a subprocess, the way the owner runs it. Every run names ``--graph``, so no test ever
tries AuraDB. The stub run needs nothing and runs in CI; the real-graph run skips when data/ is
absent, as it is in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "scripts" / "demo.py"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"
KG_FILES = (
    REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_evidences.json",
)
needs_kg = pytest.mark.skipif(
    not all(p.exists() for p in KG_FILES), reason="data/ is not committed (CI)"
)


def run_demo(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DEMO), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )


def recorded_labels(case_id: str) -> set[str]:
    """The labels of the findings and risk factors a golden case records."""
    cases = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]
    case = next(c for c in cases if c["id"] == case_id)["case"]
    return {f["label"] for key in ("findings", "risk_factors") for f in case.get(key, [])}


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
