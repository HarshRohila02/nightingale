"""Red-flag rules for must-not-miss conditions.

These rules run **independently of the ML ranker** (FR-6.1). A rule may fire on a
condition the model scored at zero — which is exactly how aortic dissection, absent
from the training data entirely, is covered.

The asymmetry is deliberate: a false alarm costs a test, a missed myocardial
infarction costs a life. These rules favour recall over precision
(docs/04-safety-ethics.md §3). But a flag on half of all patients teaches users to
ignore flags (R-15), so each rule follows a published clinical pattern rather than any
single suggestive finding:

* **Aortic dissection** follows the ADD-RS (the Aortic Dissection Detection Risk Score of
  the 2010 AHA/ACC guideline): chest, back or abdominal pain with findings from at least
  two of its three categories, or with one of its most specific features alone — tearing
  pain, a pulse deficit or an inter-arm blood-pressure difference. Back radiation is in
  none of them, so it no longer fires the rule alone (it did for 50% of DDXPlus patients,
  EXP-014).
* **Pulmonary embolism** needs a symptom PE causes (pleuritic pain or breathlessness) *and*
  a Wells-criteria thrombosis risk (a DVT sign, immobilisation, recent surgery or previous
  DVT), as its reason always said. Before 2d it fired on the risk factor alone.
* **Unstable angina** (Braunwald): a crescendo pattern, or chest pain at rest with an
  ischaemic character.
* **Myocarditis**: chest pain after a viral illness, with breathlessness or palpitations.
* **Acute pulmonary edema**: breathlessness with orthopnoea, paroxysmal nocturnal dyspnoea or
  known heart failure.

A rule's reason says what fired it and what must be excluded. It never recommends a
treatment or a drug (FR-6.5); src/reasoning/safety.py checks that.

Reference: docs/04-safety-ethics.md §3 (the must-not-miss table), docs/08 EXP-014 and
EXP-008 (the rates on DDXPlus validate), docs/07 R-15.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.contracts import PatientCase

__all__ = ["RedFlagRule", "RULES", "evaluate_red_flags"]


@dataclass(frozen=True)
class RedFlagRule:
    """A pattern that, if matched, escalates a condition regardless of its rank.

    Every constraint given must hold.

    Attributes:
        condition_id: The condition to escalate.
        reason: Clinician-facing explanation of why the flag fired.
        all_of: Every one of these concept ids must be present.
        any_of: At least ``min_any`` of these must be present.
        min_any: How many of ``any_of`` must be present. Default 1.
        groups: Sets of concept ids; a group is met when any one of its concepts is present.
        min_groups: How many ``groups`` must be met. None (the default) means all of them, so
            ``groups`` reads "one of each"; a number reads "findings from at least that many
            categories", as a clinical score such as the ADD-RS counts them.
    """

    condition_id: str
    reason: str
    any_of: frozenset[str] = field(default_factory=frozenset)
    all_of: frozenset[str] = field(default_factory=frozenset)
    min_any: int = 1
    groups: tuple[frozenset[str], ...] = ()
    min_groups: int | None = None

    @property
    def concepts(self) -> frozenset[str]:
        """Every concept id the rule reads."""
        return frozenset(self.any_of | self.all_of).union(*self.groups)

    def matches(self, case: PatientCase) -> bool:
        present = case.present_concept_ids()
        if self.all_of and not self.all_of.issubset(present):
            return False
        if self.any_of and len(self.any_of & present) < self.min_any:
            return False
        if self.groups:
            met = sum(bool(group & present) for group in self.groups)
            if met < (len(self.groups) if self.min_groups is None else self.min_groups):
                return False
        return bool(self.all_of or self.any_of or self.groups)


# --- Aortic dissection: the ADD-RS ------------------------------------------------ #
AORTIC_PAIN = frozenset({"SYM:chest_pain", "SYM:back_pain", "SYM:abdominal_pain"})
"""The ADD-RS applies to chest, back or abdominal pain."""
ADD_RS_CONDITIONS = frozenset(
    {
        "RF:connective_tissue_disease",
        "RF:family_history_aortic_disease",
        "RF:aortic_valve_disease",
        "RF:thoracic_aortic_aneurysm",
        "RF:aortic_manipulation",
    }
)
ADD_RS_PAIN = frozenset({"SYM:sudden_onset", "SYM:severe_pain", "SYM:pain_character_tearing"})
ADD_RS_EXAM = frozenset(
    {
        "SYM:pulse_deficit",
        "SYM:interarm_bp_difference",
        "SYM:focal_neuro_deficit",
        "SYM:aortic_regurgitation_murmur",
        "SYM:hypotension",
    }
)
AORTIC_SPECIFIC = frozenset(
    {"SYM:pain_character_tearing", "SYM:pulse_deficit", "SYM:interarm_bp_difference"}
)
"""Features specific enough to flag alone: docs/04 §3 names tearing pain and the inter-arm
difference, and a pulse deficit is the same malperfusion sign."""

# --- Shared groups ------------------------------------------------------------------ #
ISCHAEMIC_CHARACTER = frozenset(
    {"SYM:pain_character_pressure", "SYM:radiation_jaw_arm", "SYM:diaphoresis"}
)
THROMBOSIS_RISK = frozenset(
    {
        "SYM:leg_swelling_unilateral",
        "SYM:calf_pain",
        "SYM:recent_immobilisation",
        "RF:recent_surgery",
        "RF:previous_dvt",
    }
)
"""Wells criteria that a history can record: signs of DVT, immobilisation or surgery, and a
previous DVT."""


RULES: tuple[RedFlagRule, ...] = (
    RedFlagRule(
        condition_id="COND:aortic_dissection",
        reason="Pain with a highly specific sign of aortic dissection (tearing character, pulse "
        "deficit or inter-arm blood-pressure difference) — aortic dissection must be excluded. "
        "This condition is absent from the training data and is detected by rule only.",
        any_of=AORTIC_PAIN,
        groups=(AORTIC_SPECIFIC,),
    ),
    RedFlagRule(
        condition_id="COND:aortic_dissection",
        reason="Pain with high-risk features from two or more ADD-RS categories (predisposing "
        "condition, pain features, examination findings) — aortic dissection must be excluded. "
        "This condition is absent from the training data and is detected by rule only.",
        any_of=AORTIC_PAIN,
        groups=(ADD_RS_CONDITIONS, ADD_RS_PAIN, ADD_RS_EXAM),
        min_groups=2,
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
        condition_id="COND:unstable_angina",
        reason="Crescendo pattern: symptoms worsening over recent weeks and brought on by less "
        "and less effort — unstable angina must be excluded. Obtain ECG and troponin.",
        all_of=frozenset({"SYM:chest_pain", "SYM:crescendo_pattern"}),
    ),
    RedFlagRule(
        condition_id="COND:unstable_angina",
        reason="Chest pain at rest with an ischaemic character (pressure, radiation to jaw/arm "
        "or diaphoresis) — unstable angina must be excluded. Obtain ECG and troponin.",
        all_of=frozenset({"SYM:chest_pain", "SYM:rest_pain"}),
        groups=(ISCHAEMIC_CHARACTER,),
    ),
    RedFlagRule(
        condition_id="COND:pulmonary_embolism",
        reason="Pleuritic chest pain or breathlessness with a thrombosis risk factor "
        "(unilateral leg swelling or calf pain, recent immobilisation or surgery, previous "
        "DVT). Consider PE.",
        groups=(frozenset({"SYM:pleuritic", "SYM:breathlessness"}), THROMBOSIS_RISK),
    ),
    RedFlagRule(
        condition_id="COND:spontaneous_pneumothorax",
        reason="Sudden-onset pleuritic chest pain with breathlessness — exclude pneumothorax.",
        all_of=frozenset({"SYM:sudden_onset", "SYM:pleuritic", "SYM:breathlessness"}),
    ),
    RedFlagRule(
        condition_id="COND:myocarditis",
        reason="Chest pain after a recent viral illness, with breathlessness or palpitations — "
        "myocarditis must be excluded. Obtain ECG and troponin.",
        all_of=frozenset({"SYM:chest_pain", "SYM:recent_viral_illness"}),
        groups=(frozenset({"SYM:breathlessness", "SYM:palpitations"}),),
    ),
    RedFlagRule(
        condition_id="COND:acute_pulmonary_edema",
        reason="Breathlessness with orthopnoea, paroxysmal nocturnal dyspnoea or known heart "
        "failure — acute pulmonary oedema must be excluded.",
        groups=(
            frozenset({"SYM:breathlessness", "SYM:paroxysmal_nocturnal_dyspnoea"}),
            frozenset({"SYM:orthopnoea", "SYM:paroxysmal_nocturnal_dyspnoea", "RF:heart_failure"}),
        ),
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
            # First matching rule per condition wins; a condition's rules are ordered from the
            # most specific reason to the most general.
            fired.setdefault(rule.condition_id, rule.reason)
    return fired
