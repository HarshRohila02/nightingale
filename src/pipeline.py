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
import math

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
from src.fusion import (
    DEFAULT_KG_WEIGHT,
    DEFAULT_ML_WEIGHT,
    LOG_FLOOR,
    fuse_scores,
    valid_model_scores,
)
from src.reasoning.red_flags import evaluate_red_flags
from src.reasoning.safety import safety_check

logger = logging.getLogger(__name__)

__all__ = ["DiagnosisPipeline", "fuse_scores"]


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
        if "ml" not in degraded and getattr(self.ranker, "degraded", False):
            # No usable model: its flat scores abstain in the pool (src/fusion/pool.py), so the
            # ranking is exactly the graph's. Same duck-typed check as the graph below.
            degraded.append("ml")
        if graph_ok and getattr(self.graph, "degraded", False):
            # A fallback store serves the graph (docs/02-architecture.md §7). The results are
            # complete, but the configured backend is not the one answering.
            degraded.append("graph_backend")

        red_flags = evaluate_red_flags(case) if self.enable_red_flags else {}
        self._apply_red_flags(candidates, red_flags)

        candidates.sort(key=lambda c: (c.red_flag, c.fused_score), reverse=True)

        self._attach_evidence(case, candidates, degraded)
        explanation = self._explain(case, candidates, degraded)

        result = DiagnosisResult(
            case_id=case.case_id,
            candidates=candidates,
            red_flags=[
                f"{BY_ID[cid].label}: {reason}" for cid, reason in red_flags.items() if cid in BY_ID
            ],
            explanation=explanation,
            degraded_components=degraded,
        )
        # The last step, whatever produced the result: the disclaimer, flags first, and no
        # treatment advice (FR-6.2, FR-6.4, FR-6.5; src/reasoning/safety.py).
        return safety_check(result, case)

    # -- internals --------------------------------------------------------- #

    def _safe_ml_scores(self, case: PatientCase, degraded: list[str]) -> dict[str, float]:
        try:
            scores = self.ranker.score(case)
        except Exception:  # noqa: BLE001 - degradation is the contract
            logger.exception("ML ranker failed; continuing with KG-only ranking")
            degraded.append("ml")
            return {}
        total = sum(scores.values()) if valid_model_scores(scores) else math.nan
        if not (math.isfinite(total) and total > 0):
            # The pool reads the scores as probabilities (src/fusion/pool.py). A logit, a NaN or
            # an infinity would otherwise crash the fusion or silently corrupt every fused score;
            # no scores, a zero sum or one that overflows would abstain without saying so.
            logger.error(
                "case %s: the ML ranker returned scores that are not finite, non-negative "
                "probabilities with a positive sum; continuing with KG-only ranking",
                case.case_id,
            )
            degraded.append("ml")
            return {}
        return scores

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
                fused_score=fused.get(cid, LOG_FLOOR),
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
