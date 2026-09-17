"""Shared data contracts for Nightingale.

This module is the single source of truth for every schema that crosses a module
boundary. Each workstream imports from here and never defines its own variant —
that is what allows four people to build against stubs in parallel.

Changing anything in this file requires agreement from all four team members
(see docs/06-engineering-conventions.md §4).

Reference: docs/02-architecture.md §4.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = [
    "DISCLAIMER",
    "Assertion",
    "Sex",
    "EvidenceRole",
    "Finding",
    "PatientCase",
    "ReasoningPath",
    "FindingAssessment",
    "Evidence",
    "Candidate",
    "Explanation",
    "DiagnosisResult",
    "ConditionRanker",
    "GraphStore",
    "EvidenceRetriever",
    "Explainer",
]


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #

DISCLAIMER: str = (
    "⚠️ Clinical decision support — not a diagnosis. Nightingale is a research "
    "prototype. Its output is generated from synthetic training data and must not "
    "be used for real patient care. Rankings are not certainties and confidence "
    "values are estimates. A qualified clinician is responsible for all diagnostic "
    "and treatment decisions."
)
"""Required on every result path. See docs/04-safety-ethics.md §2 (FR-6.4).

Must never be removed, shortened, or rendered less prominently than the ranking.
"""


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Assertion(str, Enum):
    """Whether a finding is present, explicitly denied, or simply not known.

    The distinction between ABSENT and UNKNOWN is clinically load-bearing:
    "patient denies leg swelling" is evidence against pulmonary embolism, while
    "leg swelling not asked" is no evidence at all. Collapsing the two will
    mis-rank PE (FR-1.3).
    """

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class Sex(str, Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"


class EvidenceRole(str, Enum):
    """How a finding relates to a candidate condition."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    MISSING = "missing"


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #


class Finding(BaseModel):
    """One clinical fact about a patient.

    Qualifiers carry the detail that actually discriminates between conditions —
    for chest pain, radiation and character matter more than its presence.
    """

    concept_id: str = Field(description="Canonical KG concept id, e.g. 'SYM:chest_pain'")
    label: str = Field(description="Human-readable label, e.g. 'Chest pain'")
    assertion: Assertion = Assertion.PRESENT
    qualifiers: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'radiate': 'to jaw', 'onset': 'sudden', 'character': 'pressure'}",
    )
    source: str = Field(default="structured", description="structured | free_text | synthea")
    evidence_span: str | None = Field(
        default=None, description="Original text span, when extracted from free text"
    )


class PatientCase(BaseModel):
    """A single chest-pain presentation submitted for ranking."""

    case_id: str
    age: int = Field(ge=0, le=120)
    sex: Sex
    findings: list[Finding] = Field(default_factory=list)
    risk_factors: list[Finding] = Field(default_factory=list)
    vitals: dict[str, float] = Field(
        default_factory=dict, description="e.g. {'hr': 98, 'sbp': 150, 'temp_c': 37.1}"
    )
    free_text: str | None = Field(
        default=None,
        description="Untrusted input — never interpolate directly into an LLM prompt (FR-8.5)",
    )

    def present_concept_ids(self) -> set[str]:
        """Concept ids the patient is asserted to have. Used by KG scoring."""
        return {
            f.concept_id
            for f in [*self.findings, *self.risk_factors]
            if f.assertion is Assertion.PRESENT
        }

    def absent_concept_ids(self) -> set[str]:
        """Concept ids explicitly denied. Drives contradiction detection."""
        return {
            f.concept_id
            for f in [*self.findings, *self.risk_factors]
            if f.assertion is Assertion.ABSENT
        }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


class ReasoningPath(BaseModel):
    """A knowledge-graph path explaining why a finding supports a condition."""

    finding_id: str
    condition_id: str
    path: list[str] = Field(description="e.g. ['Chest pain', 'HAS_SYMPTOM', 'Acute MI']")
    weight: float = 1.0


class FindingAssessment(BaseModel):
    """How one finding bears on one candidate condition."""

    finding_id: str
    label: str
    role: EvidenceRole
    rationale: str | None = None


class Evidence(BaseModel):
    """A retrieved passage supporting a statement about a candidate."""

    source: str = Field(description="PubMed | PMC | KG")
    title: str
    passage: str
    citation: str = Field(description="e.g. 'PMID:12345678'")
    url: str | None = None
    score: float = 0.0


class Candidate(BaseModel):
    """One condition in the differential, with its scores and explanation."""

    condition_id: str
    label: str
    ml_score: float = 0.0
    kg_score: float = 0.0
    fused_score: float = 0.0
    calibrated_probability: float | None = Field(
        default=None,
        description="None until calibration is applied. Never display a raw score as a probability.",
    )
    is_must_not_miss: bool = False
    red_flag: bool = False
    red_flag_reason: str | None = None
    assessments: list[FindingAssessment] = Field(default_factory=list)
    paths: list[ReasoningPath] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    def supporting(self) -> list[FindingAssessment]:
        return [a for a in self.assessments if a.role is EvidenceRole.SUPPORTING]

    def contradicting(self) -> list[FindingAssessment]:
        return [a for a in self.assessments if a.role is EvidenceRole.CONTRADICTING]

    def missing(self) -> list[FindingAssessment]:
        return [a for a in self.assessments if a.role is EvidenceRole.MISSING]


class Explanation(BaseModel):
    """Natural-language explanation, constrained to retrieved evidence."""

    text: str
    grounded: bool = Field(description="False if any claim could not be traced to a source")
    unsupported_claims: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class DiagnosisResult(BaseModel):
    """The complete response to a diagnosis request."""

    case_id: str
    candidates: list[Candidate] = Field(description="Ordered by fused_score, descending")
    red_flags: list[str] = Field(default_factory=list)
    suggested_next_tests: list[str] = Field(default_factory=list)
    explanation: Explanation | None = None
    disclaimer: str = DISCLAIMER
    degraded_components: list[str] = Field(
        default_factory=list,
        description="e.g. ['rag', 'llm'] — MUST be rendered by the UI. A silently "
        "degraded medical tool is a safety problem (docs/02-architecture.md §7).",
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
# Module interfaces
# --------------------------------------------------------------------------- #
# Every Protocol below gets a trivial stub in src/stubs.py so that the pipeline
# runs end-to-end from week one (the "stub-first rule").


@runtime_checkable
class ConditionRanker(Protocol):
    """src/ml — scores conditions from patient features."""

    def score(self, case: PatientCase) -> dict[str, float]:
        """Return {condition_id: raw_score}. Scores need not be normalised."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """src/medical_kg — the cardiac knowledge graph (Neo4j or NetworkX)."""

    def score_by_connectivity(self, case: PatientCase) -> dict[str, float]:
        """Return {condition_id: graph_score} from patient↔condition connectivity."""
        ...

    def paths_for(self, case: PatientCase, condition_id: str) -> list[ReasoningPath]:
        """Return the KG paths linking this patient's findings to this condition."""
        ...

    def expected_findings(self, condition_id: str) -> list[Finding]:
        """Findings typically seen with this condition. Drives 'missing' analysis."""
        ...


@runtime_checkable
class EvidenceRetriever(Protocol):
    """src/rag — retrieves literature passages for a candidate."""

    def retrieve(self, condition: str, case: PatientCase, k: int = 5) -> list[Evidence]: ...


@runtime_checkable
class Explainer(Protocol):
    """src/llm — phrases the explanation. Must not introduce new medical facts."""

    def explain(self, case: PatientCase, candidates: list[Candidate]) -> Explanation: ...
