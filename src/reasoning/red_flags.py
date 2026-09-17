"""Red-flag rules for must-not-miss conditions.

These rules run **independently of the ML ranker** (FR-6.1). A rule may fire on a
condition the model scored at zero — which is exactly how aortic dissection, absent
from the training data entirely, is covered.

The asymmetry is deliberate: a false alarm costs a test, a missed myocardial
infarction costs a life. These rules favour recall over precision
(docs/04-safety-ethics.md §3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.contracts import PatientCase

__all__ = ["RedFlagRule", "RULES", "evaluate_red_flags"]


@dataclass(frozen=True)
class RedFlagRule:
    """A pattern that, if matched, escalates a condition regardless of its rank.

    Attributes:
        condition_id: The condition to escalate.
        any_of: Fires if ANY of these concept ids are present (optional).
        all_of: Fires only if ALL of these concept ids are present (optional).
        min_any: How many of `any_of` must be present. Default 1.
        reason: Clinician-facing explanation of why the flag fired.
    """

    condition_id: str
    reason: str
    any_of: frozenset[str] = field(default_factory=frozenset)
    all_of: frozenset[str] = field(default_factory=frozenset)
    min_any: int = 1

    def matches(self, case: PatientCase) -> bool:
        present = case.present_concept_ids()
        if self.all_of and not self.all_of.issubset(present):
            return False
        if self.any_of:
            return len(self.any_of & present) >= self.min_any
        # all_of satisfied and no any_of constraint
        return bool(self.all_of)


RULES: tuple[RedFlagRule, ...] = (
    RedFlagRule(
        condition_id="COND:aortic_dissection",
        reason="Tearing chest pain radiating to the back and/or an inter-arm blood-pressure "
        "difference — aortic dissection must be excluded. This condition is absent from the "
        "training data and is detected by rule only.",
        all_of=frozenset({"SYM:chest_pain"}),
        any_of=frozenset(
            {"SYM:pain_character_tearing", "SYM:radiation_back", "SYM:interarm_bp_difference"}
        ),
    ),
    RedFlagRule(
        condition_id="COND:nstemi_stemi",
        reason="Ischaemic pattern: chest pain with two or more of exertional onset, "
        "radiation to jaw/arm, pressure character, or diaphoresis. Obtain ECG and troponin.",
        all_of=frozenset({"SYM:chest_pain"}),
        any_of=frozenset(
            {
                "SYM:exertional",
                "SYM:radiation_jaw_arm",
                "SYM:pain_character_pressure",
                "SYM:diaphoresis",
            }
        ),
        min_any=2,
    ),
    RedFlagRule(
        condition_id="COND:pulmonary_embolism",
        reason="Pleuritic chest pain or breathlessness with a thrombosis risk factor "
        "(unilateral leg swelling, recent immobilisation or surgery). Consider PE.",
        any_of=frozenset({"SYM:leg_swelling_unilateral", "SYM:recent_immobilisation"}),
    ),
    RedFlagRule(
        condition_id="COND:spontaneous_pneumothorax",
        reason="Sudden-onset pleuritic chest pain with breathlessness — exclude pneumothorax.",
        all_of=frozenset({"SYM:sudden_onset", "SYM:pleuritic", "SYM:breathlessness"}),
    ),
    RedFlagRule(
        condition_id="COND:boerhaave",
        reason="Severe chest pain following forceful vomiting — consider oesophageal rupture.",
        all_of=frozenset({"SYM:recent_forceful_vomiting", "SYM:chest_pain"}),
    ),
)


def evaluate_red_flags(case: PatientCase) -> dict[str, str]:
    """Evaluate every rule against a case.

    Args:
        case: The patient presentation.

    Returns:
        Mapping of condition_id -> reason, for every rule that fired. Empty if none.
    """
    fired: dict[str, str] = {}
    for rule in RULES:
        if rule.matches(case):
            # First matching rule per condition wins; rules are ordered by severity.
            fired.setdefault(rule.condition_id, rule.reason)
    return fired
