"""The FastAPI service over the pipeline (task 1e; docs/02 §3, FR-7).

    python scripts/serve_api.py            # http://127.0.0.1:8000/docs

**Decision support only, never a diagnosis.** Every result carries the disclaimer, word for word
(:func:`~src.reasoning.safety.safety_check` runs last inside the pipeline), and ``/health`` repeats
it.

The app is built around a ready :class:`~src.pipeline.DiagnosisPipeline`, so this module reads no
configuration file (nothing in ``src/`` does). ``scripts/serve_api.py`` turns
``configs/config.yaml`` into the pipeline, exactly as ``scripts/demo.py`` does, and the tests pass
in one made of stubs.

Endpoints:

* ``POST /diagnose`` takes a :class:`~src.contracts.PatientCase` and returns a
  :class:`~src.contracts.DiagnosisResult`, with ``X-Elapsed-Ms`` saying how long the ranking took.
  A case written in the hand-authored ``SYM:*`` / ``RF:*`` concepts is passed through the crosswalk
  first (``expand_case``), as the real graph needs; a malformed case is refused with 422.
  Its ``suggested_next_tests`` are the cited investigations to consider for the leading and the
  red-flagged candidates (3d, ``src/reasoning/investigations.py``).
* ``POST /next-questions`` takes the same case and returns the findings worth asking about next,
  ranked by how much their answer would reduce the knowledge graph's uncertainty (3d,
  ``src/reasoning/next_questions.py``), with the conditions each one bears on. A separate endpoint
  because ``DiagnosisResult`` has no field for them, and ``src/contracts.py`` changes only with all
  four team members (docs/02 §5.3). None when the graph cannot answer, rather than a guess.
* ``GET /conditions``: the 14 conditions the system knows, which are all it can rank (R-13).
* ``GET /concepts``: the hand-authored findings it understands, for a form to offer.
* ``GET /health``: which ranker, graph and evidence retriever answer, and what is degraded
  (docs/02 §7). Each candidate's ``evidence`` (task 3a) cites its passage by PMCID with a link;
  without a retrieval bundle it is empty and every result reports ``rag``.

Requests are handled one at a time by the pipeline's components, which are not written for
concurrent use (the ranker keeps its last tokens for the audit trail); a demo serves one user.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from src.conditions import CONDITIONS
from src.contracts import DISCLAIMER, DiagnosisResult, PatientCase
from src.medical_kg.crosswalk import BY_CONCEPT, Match, expand_case
from src.pipeline import DiagnosisPipeline
from src.reasoning.next_questions import NextQuestion, next_questions

__all__ = ["NextQuestions", "create_app"]


class NextQuestions(BaseModel):
    """``POST /next-questions``: what to ask next for one case (3d). Not a contract type."""

    case_id: str
    questions: list[NextQuestion] = Field(
        default_factory=list,
        description="Best first; empty when the graph cannot answer or no answer would move it",
    )
    entropy_bits: float | None = Field(
        default=None,
        description="The knowledge graph's uncertainty over the conditions now, in bits; None "
        "when the graph cannot answer. Not calibrated, and not a probability.",
    )
    disclaimer: str = DISCLAIMER


def create_app(
    pipeline: DiagnosisPipeline,
    *,
    labels: Mapping[str, str] | None = None,
    about: Mapping[str, str] | None = None,
) -> FastAPI:
    """The service, answering with ``pipeline``.

    Args:
        pipeline: Built by the caller (``scripts/serve_api.py``, or a test).
        labels: ``DDX:E_nn`` → label, the real graph's node labels. When given, each case is
            expanded through the crosswalk before it is ranked, as the real graph needs; leave it
            out for the stub graph, which speaks the hand-authored concepts directly.
        about: One-line descriptions (``ranker``, ``graph``, ``retriever``) for ``/health``.
    """
    app = FastAPI(
        title="Nightingale",
        summary="Decision support for acute chest pain. Never a diagnosis.",
        version="0.1",
    )
    lock = threading.Lock()
    startup_degraded = [
        name
        for name, part in (
            ("ml", pipeline.ranker),
            ("graph_backend", pipeline.graph),
            ("rag", pipeline.retriever),
        )
        if getattr(part, "degraded", False)
    ]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "degraded" if startup_degraded else "ok",
            "degraded_components": startup_degraded,
            **dict(about or {}),
            "disclaimer": DISCLAIMER,
        }

    @app.get("/conditions")
    def conditions() -> list[dict[str, Any]]:
        return [
            {
                "id": c.id,
                "label": c.label,
                "category": c.category.value,
                "must_not_miss": c.is_must_not_miss,
                "in_training_data": c.in_training_data,
                "red_flag_hint": c.red_flag_hint,
            }
            for c in CONDITIONS
        ]

    @app.get("/concepts")
    def concepts() -> list[dict[str, Any]]:
        out = []
        for concept_id in sorted(BY_CONCEPT):
            entry = BY_CONCEPT[concept_id]
            out.append(
                {
                    "concept_id": concept_id,
                    "label": entry.label,
                    "kind": "risk_factor" if concept_id.startswith("RF:") else "finding",
                    "reaches_ddxplus": entry.match is not Match.NONE,
                }
            )
        return out

    @app.post("/diagnose", response_model=DiagnosisResult)
    def diagnose(case: PatientCase, response: Response) -> DiagnosisResult:
        started = time.perf_counter()
        expanded = expand_case(case, labels) if labels is not None else case
        with lock:
            result = pipeline.run(expanded)
        response.headers["X-Elapsed-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return result

    @app.post("/next-questions", response_model=NextQuestions)
    def questions(case: PatientCase, response: Response) -> NextQuestions:
        started = time.perf_counter()
        with lock:
            # The ranking expands the case through the crosswalk itself, once per hypothetical
            # answer, so it takes the case as entered.
            ranking = next_questions(case, pipeline.graph)
        response.headers["X-Elapsed-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return NextQuestions(
            case_id=case.case_id,
            questions=ranking.questions,
            entropy_bits=ranking.entropy_bits,
        )

    return app
