"""End-to-end orchestrator.

Wires the modules together in the order defined in docs/02-architecture.md §3,
and enforces the degradation contract from §7: if an optional component fails,
the pipeline still returns a ranked, explained differential and records what was
lost in `DiagnosisResult.degraded_components`.

With the stubs from src/stubs.py this runs today — that is what makes the
Week-3 walking skeleton reachable before any real component exists.
"""

from __future__ import annotations

import logging

from src.conditions import BY_ID, CONDITIONS
from src.contracts import (
    Candidate,
    ConditionRanker,
    DiagnosisResult,
    EvidenceRetriever,
    EvidenceRole,
    Explainer,
    Explanation,
    FindingAssessment,
    GraphStore,
    PatientCase,
)
from src.reasoning.red_flags import evaluate_red_flags

logger = logging.getLogger(__name__)

__all__ = ["DiagnosisPipeline", "fuse_scores"]

DEFAULT_ML_WEIGHT = 0.5
DEFAULT_KG_WEIGHT = 0.5


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalise to [0, 1]. A flat input maps to all zeros."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse_scores(
    ml_scores: dict[str, float],
    kg_scores: dict[str, float],
    ml_weight: float = DEFAULT_ML_WEIGHT,
    kg_weight: float = DEFAULT_KG_WEIGHT,
) -> dict[str, float]:
    """Combine ML and KG signals into a single ranking score.

    Both signals are normalised before weighting so that one cannot dominate
    purely because of its scale.

    Args:
        ml_scores: {condition_id: raw model score}.
        kg_scores: {condition_id: graph connectivity score}.
        ml_weight: Weight on the (normalised) ML signal.
        kg_weight: Weight on the (normalised) KG signal.

    Returns:
        {condition_id: fused score} over the union of both inputs.

    Note:
        Weights are tuned on validation only (docs/05-evaluation-protocol.md §2)
        and must be reported in the final write-up — they are a result, not a
        hidden hyperparameter.
    """
    ml_norm = _normalise(ml_scores)
    kg_norm = _normalise(kg_scores)
    return {
        cid: ml_weight * ml_norm.get(cid, 0.0) + kg_weight * kg_norm.get(cid, 0.0)
        for cid in set(ml_norm) | set(kg_norm)
    }


class DiagnosisPipeline:
    """Runs a patient case through the full reasoning pipeline.

    Every collaborator is injected as a Protocol, so any of them can be a stub,
    the real implementation, or disabled entirely for an ablation run
    (docs/05-evaluation-protocol.md §5).
    """

    def __init__(
        self,
        ranker: ConditionRanker,
        graph: GraphStore,
        retriever: EvidenceRetriever | None = None,
        explainer: Explainer | None = None,
        *,
        ml_weight: float = DEFAULT_ML_WEIGHT,
        kg_weight: float = DEFAULT_KG_WEIGHT,
        enable_red_flags: bool = True,
        top_k_evidence: int = 3,
    ) -> None:
        self.ranker = ranker
        self.graph = graph
        self.retriever = retriever
        self.explainer = explainer
        self.ml_weight = ml_weight
        self.kg_weight = kg_weight
        self.enable_red_flags = enable_red_flags
        self.top_k_evidence = top_k_evidence

    # -- public API -------------------------------------------------------- #

    def run(self, case: PatientCase) -> DiagnosisResult:
        """Produce a ranked, explained differential for one case."""
        degraded: list[str] = []

        ml_scores = self._safe_ml_scores(case, degraded)
        kg_scores = self._safe_kg_scores(case, degraded)
        graph_ok = kg_scores is not None
        kg_scores = kg_scores or {}
        fused = fuse_scores(ml_scores, kg_scores, self.ml_weight, self.kg_weight)

        candidates = self._build_candidates(case, ml_scores, kg_scores, fused, graph_ok)
        if graph_ok and getattr(self.graph, "degraded", False):
            # A fallback store serves the graph (docs/02-architecture.md §7). The results are
            # complete, but the configured backend is not the one answering.
            degraded.append("graph_backend")

        red_flags = evaluate_red_flags(case) if self.enable_red_flags else {}
        self._apply_red_flags(candidates, red_flags)

        candidates.sort(key=lambda c: (c.red_flag, c.fused_score), reverse=True)

        self._attach_evidence(case, candidates, degraded)
        explanation = self._explain(case, candidates, degraded)

        return DiagnosisResult(
            case_id=case.case_id,
            candidates=candidates,
            red_flags=[
                f"{BY_ID[cid].label}: {reason}" for cid, reason in red_flags.items() if cid in BY_ID
            ],
            explanation=explanation,
            degraded_components=degraded,
        )

    # -- internals --------------------------------------------------------- #

    def _safe_ml_scores(self, case: PatientCase, degraded: list[str]) -> dict[str, float]:
        try:
            return self.ranker.score(case)
        except Exception:  # noqa: BLE001 - degradation is the contract
            logger.exception("ML ranker failed; continuing with KG-only ranking")
            degraded.append("ml")
            return {}

    def _safe_kg_scores(self, case: PatientCase, degraded: list[str]) -> dict[str, float] | None:
        """The graph's scores, or None if the graph store failed."""
        try:
            return self.graph.score_by_connectivity(case)
        except Exception:  # noqa: BLE001
            logger.exception("Graph store failed; continuing with ML-only ranking")
            degraded.append("graph_backend")
            return None

    def _build_candidates(
        self,
        case: PatientCase,
        ml_scores: dict[str, float],
        kg_scores: dict[str, float],
        fused: dict[str, float],
        graph_ok: bool,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for condition in CONDITIONS:
            cid = condition.id
            candidate = Candidate(
                condition_id=cid,
                label=condition.label,
                ml_score=ml_scores.get(cid, 0.0),
                kg_score=kg_scores.get(cid, 0.0),
                fused_score=fused.get(cid, 0.0),
                is_must_not_miss=condition.is_must_not_miss,
            )
            candidate.assessments = self._assess(case, cid, graph_ok)
            if graph_ok:
                try:
                    candidate.paths = self.graph.paths_for(case, cid)
                except Exception:  # noqa: BLE001
                    logger.exception("Path extraction failed for %s", cid)
            candidates.append(candidate)
        return candidates

    def _assess(
        self, case: PatientCase, condition_id: str, graph_ok: bool
    ) -> list[FindingAssessment]:
        """Classify each finding as supporting, contradicting, or missing.

        This is derived from graph structure, never generated — the LLM only
        phrases what this method produces.
        """
        if not graph_ok:
            return []
        try:
            expected = self.graph.expected_findings(condition_id)
        except Exception:  # noqa: BLE001
            logger.exception("expected_findings failed for %s", condition_id)
            return []

        expected_ids = {f.concept_id: f for f in expected}
        present = case.present_concept_ids()
        absent = case.absent_concept_ids()
        assessments: list[FindingAssessment] = []

        for concept_id, expected_finding in expected_ids.items():
            if concept_id in present:
                role = EvidenceRole.SUPPORTING
            elif concept_id in absent:
                role = EvidenceRole.CONTRADICTING
            else:
                role = EvidenceRole.MISSING
            assessments.append(
                FindingAssessment(
                    finding_id=concept_id,
                    label=expected_finding.label,
                    role=role,
                )
            )
        return assessments

    def _apply_red_flags(self, candidates: list[Candidate], fired: dict[str, str]) -> None:
        for candidate in candidates:
            reason = fired.get(candidate.condition_id)
            if reason:
                candidate.red_flag = True
                candidate.red_flag_reason = reason

    def _attach_evidence(
        self, case: PatientCase, candidates: list[Candidate], degraded: list[str]
    ) -> None:
        if self.retriever is None:
            degraded.append("rag")
            return
        for candidate in candidates[: self.top_k_evidence]:
            try:
                candidate.evidence = self.retriever.retrieve(candidate.label, case)
            except Exception:  # noqa: BLE001
                logger.exception("Retrieval failed for %s", candidate.condition_id)
                if "rag" not in degraded:
                    degraded.append("rag")
                return

    def _explain(
        self, case: PatientCase, candidates: list[Candidate], degraded: list[str]
    ) -> Explanation | None:
        if self.explainer is None:
            degraded.append("llm")
            return None
        try:
            return self.explainer.explain(case, candidates)
        except Exception:  # noqa: BLE001
            logger.exception("Explainer failed; returning result without explanation")
            degraded.append("llm")
            return None
