"""Tests for the ranker seam (task 1c, src/ml/ranker.py).

Two things matter here and are tested separately:

1. **It degrades rather than failing.** No release files, no model, or a model trained on other
   features must all give a working pipeline whose ranking is the knowledge graph's alone. Both
   release files are gitignored, so this is CI's ordinary state and needs no data to test.
2. **When a real model is behind it, the pipeline's ranking actually changes.** That is asserted
   against the stub ranker, so it cannot pass by accident.

The "knowledge-graph only" claim is checked by *equality* with a pipeline that has no ranker at
all, rather than by inspecting scores: that is what the claim means.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.contracts import PatientCase
from src.ml.ranker import (
    BACKENDS,
    DEFAULT_BACKEND,
    DegradedRanker,
    ModelRanker,
    describe,
    open_ranker,
)
from src.pipeline import DiagnosisPipeline
from src.stubs import ConstantRanker, EmptyRetriever, InMemoryGraphStore, TemplateExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"
REAL_EVIDENCES = REPO_ROOT / "data" / "raw" / "ddxplus" / "release_evidences.json"
REAL_CONDITIONS = REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json"
REAL_MODELS = REPO_ROOT / "models" / "nightingale_b0_b1"
needs_real_data = pytest.mark.skipif(
    not all(p.exists() for p in (REAL_EVIDENCES, REAL_CONDITIONS, REAL_MODELS)),
    reason="data/ and models/ are not committed (CI)",
)

GOLDEN_CASES = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]
GC001 = PatientCase(**GOLDEN_CASES[0]["case"])


def pipeline_with(ranker) -> DiagnosisPipeline:
    return DiagnosisPipeline(
        ranker=ranker,
        graph=InMemoryGraphStore(),
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
    )


# --------------------------------------------------------------------------- #
# Degrading
# --------------------------------------------------------------------------- #


class TestDegrading:
    def test_missing_release_files_give_a_working_ranker(self, tmp_path):
        ranker = open_ranker(evidences_path=tmp_path / "nope.json", conditions_path=tmp_path / "x")
        assert isinstance(ranker, DegradedRanker)
        assert ranker.degraded
        assert "not on this machine" in ranker.reason
        assert ranker.score(GC001), "a degraded ranker still has to answer"

    @needs_real_data
    def test_a_missing_model_degrades_with_the_path_in_the_reason(self, tmp_path):
        ranker = open_ranker(
            model_dir=tmp_path,
            evidences_path=REAL_EVIDENCES,
            conditions_path=REAL_CONDITIONS,
        )
        assert isinstance(ranker, DegradedRanker)
        assert "no trained model at" in ranker.reason

    @needs_real_data
    def test_a_model_trained_on_other_features_is_refused(self, tmp_path):
        """A fingerprint mismatch must degrade, never silently score the wrong columns."""
        model = json.loads((REAL_MODELS / "b1_logreg.json").read_text(encoding="utf-8"))
        model["feature_fingerprint"] = "0000000000000000"
        (tmp_path / "b1_logreg.json").write_text(json.dumps(model), encoding="utf-8")
        ranker = open_ranker(
            model_dir=tmp_path, evidences_path=REAL_EVIDENCES, conditions_path=REAL_CONDITIONS
        )
        assert isinstance(ranker, DegradedRanker)
        assert "retrain" in ranker.reason

    def test_backend_none_is_a_deliberate_ablation(self):
        ranker = open_ranker("none")
        assert isinstance(ranker, DegradedRanker)
        assert "switched off" in ranker.reason

    def test_an_unknown_backend_is_a_typo_and_raises(self):
        """A missing file is a fact about the machine; a bad backend is a broken config."""
        with pytest.raises(ValueError, match="unknown ml backend"):
            open_ranker("mlp")

    def test_the_default_backend_is_not_xgboost(self):
        """EXP-017 / R-18: XGBoost answers atrial fibrillation to everything short."""
        assert DEFAULT_BACKEND == "logreg"
        assert set(BACKENDS) == {"logreg", "xgboost", "none"}

    def test_a_degraded_ranker_makes_the_pipeline_say_so(self):
        result = pipeline_with(open_ranker("none")).run(GC001)
        assert "ml" in result.degraded_components

    def test_a_degraded_ranking_equals_a_graph_only_ranking(self):
        """The real meaning of 'knowledge-graph only': the same order, not merely low scores."""
        degraded = pipeline_with(open_ranker("none")).run(GC001)
        flat = pipeline_with(ConstantRanker()).run(GC001)
        assert [c.condition_id for c in degraded.candidates] == [
            c.condition_id for c in flat.candidates
        ]

    def test_the_stub_ranker_does_not_claim_to_be_degraded(self):
        """The pipeline's check is duck-typed, so a stub without the attribute must be fine."""
        assert "ml" not in pipeline_with(ConstantRanker()).run(GC001).degraded_components

    def test_describe_names_what_is_ranking(self):
        assert "none" in describe(open_ranker("none"))


# --------------------------------------------------------------------------- #
# A real model behind the pipeline
# --------------------------------------------------------------------------- #


class FakeModel:
    """Scores one condition highly, so a change in the ranking is unmistakable."""

    labels = ("COND:gerd", "COND:nstemi_stemi", "COND:pulmonary_embolism")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0.7, 0.2, 0.1]), (len(X), 1))


class TestModelRanker:
    @needs_real_data
    def test_it_scores_a_case_without_the_pipeline_knowing_how(self):
        from src.ml.features import EvidenceEncoder

        encoder = EvidenceEncoder.from_release(REAL_EVIDENCES, REAL_CONDITIONS)
        ranker = ModelRanker(FakeModel(), encoder)
        scores = ranker.score(GC001)
        assert scores == pytest.approx(
            {"COND:gerd": 0.7, "COND:nstemi_stemi": 0.2, "COND:pulmonary_embolism": 0.1}
        )
        assert not ranker.degraded

    @needs_real_data
    def test_it_keeps_the_audit_trail_of_what_the_model_saw(self):
        """R-17: a ranking must be traceable to the tokens the inversion chose."""
        from src.ml.features import EvidenceEncoder

        ranker = ModelRanker(
            FakeModel(), EvidenceEncoder.from_release(REAL_EVIDENCES, REAL_CONDITIONS)
        )
        assert ranker.last_tokens is None
        ranker.score(GC001)
        assert ranker.last_tokens is not None
        assert "E_55_@_V_101" in ranker.last_tokens.tokens
        assert ranker.last_tokens.by_concept["SYM:chest_pain"]

    @needs_real_data
    def test_the_real_model_opens_and_ranks(self):
        ranker = open_ranker(
            model_dir=REAL_MODELS,
            evidences_path=REAL_EVIDENCES,
            conditions_path=REAL_CONDITIONS,
        )
        assert isinstance(ranker, ModelRanker)
        scores = ranker.score(GC001)
        assert len(scores) == 13, "the 13 trainable conditions"
        assert "COND:aortic_dissection" not in scores, "DDXPlus has no dissection cases"
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)

    @needs_real_data
    def test_the_real_model_changes_the_pipeline_ranking(self):
        """If the ranking is identical to the stub's, the model is not really wired in."""
        ranker = open_ranker(
            model_dir=REAL_MODELS,
            evidences_path=REAL_EVIDENCES,
            conditions_path=REAL_CONDITIONS,
        )
        real = pipeline_with(ranker).run(GC001)
        flat = pipeline_with(ConstantRanker()).run(GC001)
        assert "ml" not in real.degraded_components
        assert [c.condition_id for c in real.candidates] != [
            c.condition_id for c in flat.candidates
        ]

    @needs_real_data
    def test_narrower_matches_reach_the_model_by_default(self):
        """Decision A-8, made visible: without them GC-002 loses its sudden onset."""
        kw = dict(
            model_dir=REAL_MODELS, evidences_path=REAL_EVIDENCES, conditions_path=REAL_CONDITIONS
        )
        gc002 = PatientCase(**GOLDEN_CASES[1]["case"])
        with_narrower = open_ranker(**kw)
        without = open_ranker(**kw, include_narrower=False)
        with_narrower.score(gc002)
        without.score(gc002)
        assert "E_59_@_9" in with_narrower.last_tokens.tokens
        assert "E_59_@_9" not in without.last_tokens.tokens


# --------------------------------------------------------------------------- #
# The ranking, with the red-flag crutch removed
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_graph():
    from src.medical_kg.networkx_store import NetworkXGraphStore

    return NetworkXGraphStore.from_files(
        REAL_CONDITIONS, REAL_EVIDENCES, REPO_ROOT / "data" / "raw" / "bodhi_s"
    )


def rank_without_flags(case: PatientCase, graph, ranker) -> list[str]:
    from src.medical_kg.crosswalk import expand_case

    pipeline = DiagnosisPipeline(
        ranker=ranker,
        graph=graph,
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
        enable_red_flags=False,
    )
    expanded = expand_case(case, dict(graph.graph.nodes(data="label")))
    return [c.condition_id for c in pipeline.run(expanded).candidates]


@needs_real_data
@pytest.mark.golden
@pytest.mark.parametrize("golden", GOLDEN_CASES[1:], ids=[c["id"] for c in GOLDEN_CASES[1:]])
def test_the_real_components_rank_a_golden_case_without_its_red_flag(golden, real_graph):
    """`pipeline.run` sorts by (red_flag, fused_score), so a flagged candidate wins whatever it
    scores. Three of the four golden cases are flagged, so with the flags on they cannot detect a
    ranking regression at all. This turns them off and puts the ranking itself under test, on the
    real graph and the real model.
    """
    ranker = open_ranker(
        model_dir=REAL_MODELS, evidences_path=REAL_EVIDENCES, conditions_path=REAL_CONDITIONS
    )
    ranking = rank_without_flags(PatientCase(**golden["case"]), real_graph, ranker)
    top_k = golden["expect"].get("top_k", 3)
    for required in golden["expect"].get("must_include", []):
        assert required in ranking[:top_k], (
            f"{golden['id']}: {required} is ranked {ranking.index(required) + 1} of "
            f"{len(ranking)} without its red flag. Fix the component, not this expectation."
        )


@needs_real_data
@pytest.mark.golden
@pytest.mark.xfail(
    strict=True,
    reason="GC-001's infarction ranks 4th without its red flag: the graph's overlap score is "
    "divided by the size of each condition's evidence set, so Boerhaave and pericarditis outrank "
    "it (EXP-014), and the model prefers unstable angina (EXP-004). 2a replaces the graph score "
    "and adds this exact case as a regression test; this xfail is strict, so it fails loudly the "
    "day 2a fixes it and must then be promoted to a plain assertion.",
)
def test_the_real_components_rank_gc001_without_its_red_flag(real_graph):
    ranker = open_ranker(
        model_dir=REAL_MODELS, evidences_path=REAL_EVIDENCES, conditions_path=REAL_CONDITIONS
    )
    ranking = rank_without_flags(PatientCase(**GOLDEN_CASES[0]["case"]), real_graph, ranker)
    assert "COND:nstemi_stemi" in ranking[:3]
