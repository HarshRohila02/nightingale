"""Pipeline tests, including the golden clinical cases.

The golden cases are the clinically meaningful ones: they assert what a doctor
would expect the system to conclude. They run against the stub components, so
they will need their expectations revisited — not deleted — as the real ranker
and knowledge graph replace the stubs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.contracts import DISCLAIMER, PatientCase
from src.pipeline import DiagnosisPipeline, fuse_scores
from src.reasoning.red_flags import evaluate_red_flags
from src.stubs import ConstantRanker, EmptyRetriever, InMemoryGraphStore, TemplateExplainer

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_cases.yaml"


def _load_golden() -> list[dict]:
    data = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["cases"]


GOLDEN_CASES = _load_golden()


@pytest.fixture
def pipeline() -> DiagnosisPipeline:
    return DiagnosisPipeline(
        ranker=ConstantRanker(),
        graph=InMemoryGraphStore(),
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
    )


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


class TestFusion:
    def test_normalises_before_weighting(self):
        """A large-scale signal must not dominate purely because of its scale."""
        fused = fuse_scores({"a": 1000.0, "b": 0.0}, {"a": 0.0, "b": 1.0})
        assert fused["a"] == pytest.approx(0.5)
        assert fused["b"] == pytest.approx(0.5)

    def test_flat_input_yields_zero(self):
        fused = fuse_scores({"a": 0.3, "b": 0.3}, {})
        assert all(v == pytest.approx(0.0) for v in fused.values())

    def test_weights_shift_the_result(self):
        ml = {"a": 1.0, "b": 0.0}
        kg = {"a": 0.0, "b": 1.0}
        ml_heavy = fuse_scores(ml, kg, ml_weight=0.9, kg_weight=0.1)
        assert ml_heavy["a"] > ml_heavy["b"]

    def test_union_of_both_signals(self):
        fused = fuse_scores({"a": 1.0}, {"b": 1.0})
        assert set(fused) == {"a", "b"}


# --------------------------------------------------------------------------- #
# Degradation contract (docs/02-architecture.md §7)
# --------------------------------------------------------------------------- #


class TestDegradation:
    def test_no_retriever_or_explainer_still_returns_a_result(self):
        pipe = DiagnosisPipeline(ranker=ConstantRanker(), graph=InMemoryGraphStore())
        result = pipe.run(PatientCase(case_id="T-1", age=50, sex="M"))
        assert result.candidates
        assert "rag" in result.degraded_components
        assert "llm" in result.degraded_components

    def test_ranker_failure_degrades_rather_than_crashes(self):
        class BrokenRanker:
            def score(self, case):
                raise RuntimeError("model not loaded")

        pipe = DiagnosisPipeline(ranker=BrokenRanker(), graph=InMemoryGraphStore())
        result = pipe.run(PatientCase(case_id="T-1", age=50, sex="M"))
        assert "ml" in result.degraded_components
        assert result.candidates, "must still rank using the knowledge graph alone"

    def test_graph_failure_degrades_rather_than_crashes(self):
        class BrokenGraph:
            def score_by_connectivity(self, case):
                raise RuntimeError("neo4j unreachable")

            def paths_for(self, case, condition_id):
                return []

            def expected_findings(self, condition_id):
                return []

        pipe = DiagnosisPipeline(ranker=ConstantRanker(), graph=BrokenGraph())
        result = pipe.run(PatientCase(case_id="T-1", age=50, sex="M"))
        assert "graph_backend" in result.degraded_components
        assert result.candidates

    def test_disclaimer_survives_degradation(self):
        """FR-6.4 — never dropped, whatever else fails."""
        pipe = DiagnosisPipeline(ranker=ConstantRanker(), graph=InMemoryGraphStore())
        result = pipe.run(PatientCase(case_id="T-1", age=50, sex="M"))
        assert result.disclaimer == DISCLAIMER


# --------------------------------------------------------------------------- #
# Red flags
# --------------------------------------------------------------------------- #


class TestRedFlags:
    def test_run_independently_of_ml_score(self):
        """Aortic dissection has no ML score at all, yet must still be flagged."""
        case = PatientCase(
            case_id="T-1",
            age=64,
            sex="M",
            findings=[
                {"concept_id": "SYM:chest_pain", "label": "Chest pain"},
                {"concept_id": "SYM:pain_character_tearing", "label": "Tearing pain"},
                {"concept_id": "SYM:radiation_back", "label": "Radiation to back"},
            ],
        )
        fired = evaluate_red_flags(case)
        assert "COND:aortic_dissection" in fired

    def test_min_any_threshold_is_enforced(self):
        """One ischaemic feature is not enough; the MI rule needs two."""
        one_feature = PatientCase(
            case_id="T-1",
            age=58,
            sex="M",
            findings=[
                {"concept_id": "SYM:chest_pain", "label": "Chest pain"},
                {"concept_id": "SYM:exertional", "label": "Exertional"},
            ],
        )
        assert "COND:nstemi_stemi" not in evaluate_red_flags(one_feature)

    def test_benign_presentation_raises_nothing(self):
        case = PatientCase(
            case_id="T-1",
            age=30,
            sex="F",
            findings=[{"concept_id": "SYM:post_prandial", "label": "Post-prandial"}],
        )
        assert evaluate_red_flags(case) == {}


# --------------------------------------------------------------------------- #
# Golden clinical cases
# --------------------------------------------------------------------------- #


@pytest.mark.golden
@pytest.mark.parametrize("golden", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
def test_golden_clinical_case(pipeline: DiagnosisPipeline, golden: dict):
    """Assert the clinical expectation encoded in tests/fixtures/golden_cases.yaml."""
    case = PatientCase(**golden["case"])
    expect = golden["expect"]
    result = pipeline.run(case)

    top_k = expect.get("top_k", 3)
    top_ids = [c.condition_id for c in result.candidates[:top_k]]

    for required in expect.get("must_include", []):
        assert required in top_ids, (
            f"{golden['id']}: expected {required} in top {top_k}, got {top_ids}. "
            f"Description: {golden['description'].strip()}"
        )

    flagged = {c.condition_id for c in result.candidates if c.red_flag}
    for required in expect.get("red_flag_for", []):
        assert required in flagged, f"{golden['id']}: expected a red flag for {required}"

    if expect.get("no_red_flag"):
        assert (
            not flagged
        ), f"{golden['id']}: benign presentation raised unexpected red flags: {flagged}"


@pytest.mark.golden
def test_every_golden_case_produces_an_explanation(pipeline: DiagnosisPipeline):
    for golden in GOLDEN_CASES:
        result = pipeline.run(PatientCase(**golden["case"]))
        assert result.explanation is not None
        assert result.explanation.grounded, "template explanations are grounded by construction"
        assert result.explanation.text
