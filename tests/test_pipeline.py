"""Pipeline tests, including the golden clinical cases.

The golden cases are the clinically meaningful ones: they assert what a doctor
would expect the system to conclude. They run against the stub components, so
they will need their expectations revisited — not deleted — as the real ranker
and knowledge graph replace the stubs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.contracts import (
    DISCLAIMER,
    Candidate,
    EvidenceRole,
    FindingAssessment,
    PatientCase,
)
from src.fusion import LOG_FLOOR, fuse_arrays
from src.pipeline import DiagnosisPipeline, fuse_scores
from src.reasoning.red_flags import evaluate_red_flags
from src.stubs import (
    MISSING_SHOWN,
    ConstantRanker,
    EmptyRetriever,
    InMemoryGraphStore,
    TemplateExplainer,
)

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
    """The logarithmic opinion pool of src/fusion/pool.py (2c, docs/02 §6)."""

    def test_the_model_s_scale_does_not_matter(self):
        """Scores are divided by their sum: 1000 / 0 is the same opinion as 1 / 0."""
        kg = {"a": -3.0, "b": -1.0}
        assert fuse_scores({"a": 1000.0, "b": 0.0}, kg) == pytest.approx(
            fuse_scores({"a": 1.0, "b": 0.0}, kg)
        )

    def test_the_result_is_a_distribution_over_every_condition(self):
        fused = fuse_scores({"a": 0.7, "b": 0.3}, {"a": -2.0, "b": -1.0, "d": -4.0})
        assert set(fused) == {"a", "b", "d"}
        assert sum(math.exp(v) for v in fused.values()) == pytest.approx(1.0)
        assert all(v <= 0.0 for v in fused.values())

    def test_no_signal_gives_no_scores(self):
        assert fuse_scores({"a": 0.3, "b": 0.3}, {}) == {}

    def test_a_flat_model_leaves_the_graph_s_order(self):
        """A degraded ranker's flat scores abstain: the ranking is the graph's, exactly."""
        kg = {"a": -5.0, "b": -1.0, "c": -3.0}
        fused = fuse_scores({"a": 1.0, "b": 1.0, "c": 1.0}, kg)
        assert sorted(fused, key=fused.get, reverse=True) == ["b", "c", "a"]

    def test_no_graph_leaves_the_model_s_order(self):
        fused = fuse_scores({"a": 0.2, "b": 0.5, "c": 0.3}, {})
        assert sorted(fused, key=fused.get, reverse=True) == ["b", "c", "a"]
        assert fused["b"] == pytest.approx(math.log(0.5))

    def test_the_graph_s_weight_shifts_the_result(self):
        ml = {"a": 0.9, "b": 0.1}
        kg = {"a": -6.0, "b": 0.0}
        ml_heavy = fuse_scores(ml, kg, ml_weight=0.9, kg_weight=0.1)
        kg_heavy = fuse_scores(ml, kg, ml_weight=0.1, kg_weight=0.9)
        assert ml_heavy["a"] > ml_heavy["b"]
        assert kg_heavy["b"] > kg_heavy["a"]

    def test_only_the_ratio_of_the_weights_matters(self):
        ml, kg = {"a": 0.9, "b": 0.1}, {"a": -6.0, "b": 0.0}
        assert fuse_scores(ml, kg, 1.0, 3.0) == pytest.approx(fuse_scores(ml, kg, 0.25, 0.75))

    def test_a_condition_the_model_cannot_score_takes_the_graph_s_probability(self):
        """EXP-005 point 7: min-max capped dissection at the graph's weight. Now the graph's
        posterior decides it, whatever the model thinks of the others."""
        kg = {"a": -9.0, "b": -9.0, "COND:aortic_dissection": 0.0}
        fused = fuse_scores({"a": 0.99, "b": 0.01}, kg)
        posterior = 1 / (1 + 2 * math.exp(-9.0))
        assert fused["COND:aortic_dissection"] == pytest.approx(math.log(posterior))
        assert max(fused, key=fused.get) == "COND:aortic_dissection"

    def test_a_near_certain_model_cannot_veto_what_the_graph_supports(self):
        """The floor: without it, the model's 0 would silence the graph (EXP-018, GC-001)."""
        ml = {"a": 1.0, "b": 0.0}
        kg = {"a": -20.0, "b": 0.0}
        fused = fuse_scores(ml, kg)
        assert fused["b"] > fused["a"]
        # Pooled by hand: (1 - α) log max(p, ε) + α kg, then normalised.
        a = 0.5 * math.log(1.0) + 0.5 * -20.0
        b = 0.5 * math.log(0.01) + 0.5 * 0.0
        assert fused["b"] - fused["a"] == pytest.approx(b - a)

    def test_all_the_weight_on_the_model_ignores_the_floor(self):
        fused = fuse_scores({"a": 0.999, "b": 0.001}, {"a": -9.0, "b": 0.0}, 1.0, 0.0)
        assert fused["a"] > fused["b"]
        assert fused["a"] - fused["b"] == pytest.approx(math.log(0.999 / 0.001))

    def test_all_the_weight_on_the_graph_is_the_graph_exactly(self):
        """Ties included: the graph rounds its scores (scoring.ROUND), so a condition outside the
        model often ties with ones inside it, and pooling must not break the tie (EXP-006's α = 1
        is B2)."""
        tied = -9.210340371976
        kg = {"a": tied, "COND:aortic_dissection": tied, "b": tied, "c": -1.0}
        ml = {"a": 0.9, "b": 0.05, "c": 0.05}
        assert fuse_scores(ml, kg, 0.0, 1.0) == fuse_scores({}, kg)
        columns = ["a", "COND:aortic_dissection", "b", "c"]
        graph = np.array([[kg[c] for c in columns]])
        fused = fuse_arrays(np.array([[0.9, 0.05, 0.05]]), graph, [0, 2, 3], 0.0, 1.0)
        assert fused[0, 0] == fused[0, 1] == fused[0, 2], "the tie survives"
        assert fused[0] == pytest.approx([fuse_scores({}, kg)[c] for c in columns], abs=1e-12)

    def test_union_of_both_signals(self):
        fused = fuse_scores({"a": 1.0}, {"b": 1.0})
        assert set(fused) == {"a", "b"}

    @pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
    def test_scores_that_are_not_probabilities_are_refused(self, bad):
        with pytest.raises(ValueError, match="finite and non-negative"):
            fuse_scores({"a": 0.5, "b": bad}, {"a": 0.0, "b": -1.0})
        with pytest.raises(ValueError, match="finite and non-negative"):
            fuse_arrays(np.array([[0.5, bad]]), np.array([[0.0, -1.0]]), [0, 1])

    def test_a_model_whose_scores_sum_to_zero_abstains(self):
        kg = {"a": -5.0, "b": -1.0}
        assert fuse_scores({"a": 0.0}, kg) == fuse_scores({}, kg)
        assert fuse_scores({"a": 0.0, "b": 0.0}, kg) == fuse_scores({}, kg)

    def test_without_a_graph_the_scores_are_clamped_too(self):
        fused = fuse_scores({"a": 1.0, "b": 1e-320}, {})
        assert fused["b"] == LOG_FLOOR

    @pytest.mark.parametrize("weights", [(0.0, 0.0), (-1.0, 1.0), (1.0, -0.5)])
    def test_weights_must_be_non_negative_and_not_both_zero(self, weights):
        with pytest.raises(ValueError):
            fuse_scores({"a": 0.5, "b": 0.5}, {"a": 0.0}, *weights)


# --------------------------------------------------------------------------- #
# Degradation contract (docs/02-architecture.md §7)
# --------------------------------------------------------------------------- #


class ScoresRanker:
    """A ranker that returns whatever scores it is given, valid or not."""

    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, case: PatientCase) -> dict[str, float]:  # noqa: ARG002
        from src.conditions import CONDITIONS

        scores = {c.id: 0.1 for c in CONDITIONS if c.in_training_data}
        scores["COND:gerd"] = self.value
        return scores


class TestDegradation:
    @pytest.mark.parametrize("value", [-2.5, math.nan, math.inf])
    def test_a_ranker_whose_scores_are_not_probabilities_degrades_to_the_graph(self, value):
        """2c: the pool reads probabilities; a logit, NaN or infinity must not crash the fusion
        or corrupt every fused score silently (docs/02 §7)."""
        case = PatientCase(**GOLDEN_CASES[0]["case"])
        bad = DiagnosisPipeline(ranker=ScoresRanker(value), graph=InMemoryGraphStore()).run(case)
        graph_only = DiagnosisPipeline(ranker=ConstantRanker(), graph=InMemoryGraphStore()).run(
            case
        )
        assert "ml" in bad.degraded_components
        assert all(math.isfinite(c.fused_score) for c in bad.candidates)
        assert [c.condition_id for c in bad.candidates] == [
            c.condition_id for c in graph_only.candidates
        ]

    @pytest.mark.parametrize(
        "scores",
        [{"COND:gerd": 0.0, "COND:psvt": 0.0}, {}, {"COND:gerd": 1e308, "COND:psvt": 1e308}],
        ids=["zeros", "none", "overflowing sum"],
    )
    def test_a_ranker_with_nothing_to_say_is_reported(self, scores):
        """The pool would abstain on these anyway; the pipeline must say so (docs/02 §7)."""
        case = PatientCase(**GOLDEN_CASES[0]["case"])
        pipe = DiagnosisPipeline(ranker=ScoresRanker(0.0), graph=InMemoryGraphStore())
        pipe.ranker.score = lambda case: scores
        assert "ml" in pipe.run(case).degraded_components

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


def test_the_template_names_a_few_unrecorded_findings_and_counts_the_rest():
    """The real graph expects two dozen findings of some conditions; the template names the first
    MISSING_SHOWN and counts the rest, so an explanation stays readable (ranking them is 3d's)."""
    candidate = Candidate(
        condition_id="COND:gerd",
        label="GERD",
        assessments=[
            FindingAssessment(finding_id=f"SYM:x{i}", label=f"x{i}", role=EvidenceRole.MISSING)
            for i in range(MISSING_SHOWN + 2)
        ],
    )
    case = PatientCase(case_id="T", age=40, sex="F")
    text = TemplateExplainer().explain(case, [candidate]).text
    assert f"x{MISSING_SHOWN - 1} (and 2 more)." in text
    assert f"x{MISSING_SHOWN}," not in text


@pytest.mark.golden
@pytest.mark.parametrize("golden", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
def test_golden_case_ranks_without_its_red_flag(golden: dict):
    """`run` sorts by (red_flag, fused_score), so a flagged candidate wins whatever it scores.

    Three of the four golden cases raise a flag, so with the flags on they cannot detect a
    ranking regression: a ranker returning noise would still pass them. Turning the flags off
    puts the fused ranking itself under test. This runs on the stubs, like the rest of this file;
    tests/test_ranker.py does the same against the real graph and the real model.
    """
    pipeline = DiagnosisPipeline(
        ranker=ConstantRanker(),
        graph=InMemoryGraphStore(),
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
        enable_red_flags=False,
    )
    result = pipeline.run(PatientCase(**golden["case"]))
    assert not any(c.red_flag for c in result.candidates), "red flags are off"
    top_k = golden["expect"].get("top_k", 3)
    top_ids = [c.condition_id for c in result.candidates[:top_k]]
    for required in golden["expect"].get("must_include", []):
        assert required in top_ids, (
            f"{golden['id']}: {required} is not in the top {top_k} on score alone, only through "
            f"its red flag. Got {top_ids}. Fix the component, not this expectation."
        )


@pytest.mark.parametrize("weights", [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.0, 1.0)])
def test_the_vectorised_pool_equals_the_scalar_one(weights):
    """The weight sweep (EXP-006) scores with fuse_arrays; the pipeline with fuse_scores."""
    rng = np.random.default_rng(7)
    conditions = [f"c{i}" for i in range(6)]
    model = [0, 1, 2, 4, 5]  # c3 is outside the model, as aortic dissection is
    ml = rng.dirichlet(np.full(len(model), 0.3), size=40)
    ml[0] = 1.0  # a flat row abstains
    ml[1, 2] = 0.0  # an exact zero
    kg = rng.normal(-8.0, 4.0, size=(40, len(conditions)))
    fused = fuse_arrays(ml, kg, model, *weights)
    for row in range(40):
        expected = fuse_scores(
            {conditions[j]: ml[row, i] for i, j in enumerate(model)},
            dict(zip(conditions, kg[row], strict=True)),
            *weights,
        )
        assert fused[row] == pytest.approx([expected[c] for c in conditions], abs=1e-9)
