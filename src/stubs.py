"""Trivial implementations of every module Protocol.

The "stub-first rule" (docs/02-architecture.md §4): each Protocol has a working
stub from week one, so the pipeline runs end-to-end before any real component
exists. Replace them one at a time; nothing downstream needs to change.

`TemplateExplainer` is not only a stub — it is also the production fallback when
the LLM is unavailable (docs/02-architecture.md §7). It cannot hallucinate,
because it only renders findings the reasoning layer already derived.
"""

from __future__ import annotations

from src.conditions import BY_ID, CONDITIONS
from src.contracts import (
    Assertion,
    Candidate,
    Evidence,
    Explanation,
    Finding,
    PatientCase,
    ReasoningPath,
)

__all__ = [
    "MISSING_SHOWN",
    "STUB_SYMPTOM_MAP",
    "ConstantRanker",
    "InMemoryGraphStore",
    "EmptyRetriever",
    "TemplateExplainer",
]


# A deliberately small condition -> expected-symptom map, hand-written from the
# domain primer in docs/09-prerequisites.md §1.2. It exists so the skeleton runs
# and so P1 has a concrete example of the shape the real BODHI-S-derived graph
# must produce. It is NOT clinical reference data and must be replaced.
STUB_SYMPTOM_MAP: dict[str, tuple[str, ...]] = {
    "COND:nstemi_stemi": (
        "SYM:chest_pain",
        "SYM:pain_character_pressure",
        "SYM:radiation_jaw_arm",
        "SYM:diaphoresis",
        "SYM:exertional",
        "SYM:breathlessness",
    ),
    "COND:unstable_angina": (
        "SYM:chest_pain",
        "SYM:pain_character_pressure",
        "SYM:exertional",
        "SYM:radiation_jaw_arm",
    ),
    "COND:stable_angina": ("SYM:chest_pain", "SYM:exertional", "SYM:relieved_by_rest"),
    "COND:pericarditis": (
        "SYM:chest_pain",
        "SYM:pleuritic",
        "SYM:positional_relief_sitting_forward",
        "SYM:recent_viral_illness",
    ),
    "COND:myocarditis": (
        "SYM:chest_pain",
        "SYM:breathlessness",
        "SYM:recent_viral_illness",
        "SYM:fatigue",
    ),
    "COND:acute_pulmonary_edema": (
        "SYM:breathlessness",
        "SYM:orthopnoea",
        "SYM:frothy_sputum",
        "SYM:leg_swelling_bilateral",
    ),
    "COND:atrial_fibrillation": ("SYM:palpitations", "SYM:irregular_pulse", "SYM:breathlessness"),
    "COND:psvt": ("SYM:palpitations", "SYM:sudden_onset", "SYM:lightheadedness"),
    "COND:pulmonary_embolism": (
        "SYM:chest_pain",
        "SYM:pleuritic",
        "SYM:breathlessness",
        "SYM:leg_swelling_unilateral",
        "SYM:recent_immobilisation",
        "SYM:sudden_onset",
    ),
    "COND:spontaneous_pneumothorax": (
        "SYM:chest_pain",
        "SYM:pleuritic",
        "SYM:sudden_onset",
        "SYM:breathlessness",
    ),
    "COND:boerhaave": ("SYM:chest_pain", "SYM:recent_forceful_vomiting", "SYM:sudden_onset"),
    "COND:gerd": (
        "SYM:chest_pain",
        "SYM:pain_character_burning",
        "SYM:worse_lying_flat",
        "SYM:post_prandial",
    ),
    "COND:panic_attack": (
        "SYM:palpitations",
        "SYM:hyperventilation",
        "SYM:paraesthesia",
        "SYM:anxiety",
    ),
    "COND:aortic_dissection": (
        "SYM:chest_pain",
        "SYM:pain_character_tearing",
        "SYM:radiation_back",
        "SYM:sudden_onset",
        "SYM:interarm_bp_difference",
    ),
}


class ConstantRanker:
    """Returns a flat score for every trainable condition.

    Implements `ConditionRanker`. Replace with the real model in src/ml.
    A flat ranker is a useful sanity floor: anything that cannot beat it is broken.
    """

    def score(self, case: PatientCase) -> dict[str, float]:  # noqa: ARG002
        trainable = [c for c in CONDITIONS if c.in_training_data]
        uniform = 1.0 / len(trainable)
        return {c.id: uniform for c in trainable}


class InMemoryGraphStore:
    """A NetworkX-free, dictionary-backed graph over STUB_SYMPTOM_MAP.

    Implements `GraphStore`. Scoring is plain symptom overlap, normalised by the
    number of symptoms the condition expects, so conditions with long symptom
    lists are not unfairly favoured. Replace with the Neo4j-backed store in
    src/medical_kg.
    """

    def __init__(self, symptom_map: dict[str, tuple[str, ...]] | None = None) -> None:
        self._map = symptom_map if symptom_map is not None else STUB_SYMPTOM_MAP

    def score_by_connectivity(self, case: PatientCase) -> dict[str, float]:
        present = case.present_concept_ids()
        absent = case.absent_concept_ids()
        scores: dict[str, float] = {}
        for condition_id, expected in self._map.items():
            if not expected:
                scores[condition_id] = 0.0
                continue
            expected_set = set(expected)
            matched = len(present & expected_set)
            denied = len(absent & expected_set)
            # Overlap minus a penalty for explicitly denied expected findings.
            scores[condition_id] = max(0.0, (matched - 0.5 * denied) / len(expected_set))
        return scores

    def paths_for(self, case: PatientCase, condition_id: str) -> list[ReasoningPath]:
        expected = set(self._map.get(condition_id, ()))
        condition = BY_ID.get(condition_id)
        condition_label = condition.label if condition else condition_id
        paths: list[ReasoningPath] = []
        for finding in case.findings:
            if finding.assertion is Assertion.PRESENT and finding.concept_id in expected:
                paths.append(
                    ReasoningPath(
                        finding_id=finding.concept_id,
                        condition_id=condition_id,
                        path=[finding.label, "HAS_SYMPTOM", condition_label],
                        weight=1.0,
                    )
                )
        return paths

    def expected_findings(self, condition_id: str) -> list[Finding]:
        return [
            Finding(concept_id=cid, label=_humanise(cid), assertion=Assertion.UNKNOWN)
            for cid in self._map.get(condition_id, ())
        ]


class EmptyRetriever:
    """Returns no evidence. Implements `EvidenceRetriever`.

    The pipeline must remain fully functional with this in place (FR-7.4) — that
    is the point of the degradation contract.
    """

    def retrieve(
        self, condition: str, case: PatientCase, k: int = 5
    ) -> list[Evidence]:  # noqa: ARG002
        return []


MISSING_SHOWN = 5
"""How many unrecorded findings the template names before it just counts the rest."""


class TemplateExplainer:
    """Builds an explanation from derived findings only — never generates facts.

    Implements `Explainer`. Also the production fallback when the LLM is
    unavailable. `grounded` is always True because every sentence is rendered
    from a `FindingAssessment` the reasoning layer produced.
    """

    def explain(self, case: PatientCase, candidates: list[Candidate]) -> Explanation:
        if not candidates:
            return Explanation(text="No candidate conditions were generated.", grounded=True)

        top = candidates[0]
        lines = [
            f"Top candidate for case {case.case_id}: {top.label} " f"(score {top.fused_score:.2f})."
        ]

        supporting = top.supporting()
        if supporting:
            lines.append("Supporting findings: " + ", ".join(a.label for a in supporting) + ".")

        contradicting = top.contradicting()
        if contradicting:
            lines.append("Findings against: " + ", ".join(a.label for a in contradicting) + ".")

        missing = top.missing()
        if missing:
            # The real graph expects two dozen findings of some conditions. Choosing which to ask
            # first is task 3d's; until then, the first few in the graph's order, and a count.
            shown = ", ".join(a.label for a in missing[:MISSING_SHOWN])
            more = len(missing) - MISSING_SHOWN
            lines.append(
                "Not recorded, and would help confirm or exclude: "
                + shown
                + (f" (and {more} more)" if more > 0 else "")
                + "."
            )

        flagged = [c for c in candidates if c.red_flag]
        if flagged:
            lines.append(
                "⚠️ Red flags raised: "
                + "; ".join(f"{c.label} — {(c.red_flag_reason or '').rstrip('.')}" for c in flagged)
                + "."
            )

        return Explanation(text=" ".join(lines), grounded=True, unsupported_claims=[])


def _humanise(concept_id: str) -> str:
    """'SYM:radiation_jaw_arm' -> 'Radiation jaw arm'. Placeholder for real labels."""
    return concept_id.split(":", 1)[-1].replace("_", " ").capitalize()
