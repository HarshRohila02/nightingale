"""Canonical registry of the conditions Nightingale ranks.

Scope is deliberately narrow: the cardiac and must-not-miss causes of acute
chest pain. Anything not in this registry is out of scope and must never appear
in a result (FR-8.4).

`ddxplus_label` is the exact `PATHOLOGY` string used by DDXPlus and is what
scripts/decode_ddxplus.py filters on — do not "tidy" these strings.

Reference: docs/01-srs.md §2.4, docs/04-safety-ethics.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Category",
    "Condition",
    "CONDITIONS",
    "BY_ID",
    "DDXPLUS_LABELS",
    "must_not_miss_ids",
    "trainable_ids",
    "from_ddxplus_label",
]


class Category(str, Enum):
    CARDIAC_ISCHAEMIC = "cardiac_ischaemic"
    CARDIAC_INFLAMMATORY = "cardiac_inflammatory"
    CARDIAC_FAILURE = "cardiac_failure"
    CARDIAC_ARRHYTHMIA = "cardiac_arrhythmia"
    MIMIC = "non_cardiac_mimic"
    VASCULAR = "vascular"


@dataclass(frozen=True)
class Condition:
    """One condition in scope.

    Attributes:
        id: Stable internal identifier. Used as the key everywhere.
        label: Human-readable name shown in the UI.
        category: Clinical grouping.
        is_must_not_miss: A missed or low ranking risks death. Drives the
            headline safety metric (docs/05-evaluation-protocol.md §3.2).
        in_training_data: False means the ML model cannot learn it and it is
            covered by a knowledge-graph red-flag rule instead.
        ddxplus_label: Exact DDXPlus PATHOLOGY string, or None if absent.
        red_flag_hint: Short description of the pattern that should fire a flag.
    """

    id: str
    label: str
    category: Category
    is_must_not_miss: bool
    in_training_data: bool
    ddxplus_label: str | None
    red_flag_hint: str | None = None
    aliases: tuple[str, ...] = field(default=())


CONDITIONS: tuple[Condition, ...] = (
    # --- Cardiac: ischaemic ------------------------------------------------ #
    Condition(
        id="COND:nstemi_stemi",
        label="Possible NSTEMI / STEMI",
        category=Category.CARDIAC_ISCHAEMIC,
        is_must_not_miss=True,
        in_training_data=True,
        ddxplus_label="Possible NSTEMI / STEMI",
        red_flag_hint="Crushing/pressure chest pain, exertional, radiating to jaw or arm, "
        "with diaphoresis and cardiac risk factors",
        aliases=("acute myocardial infarction", "heart attack", "acute coronary syndrome"),
    ),
    Condition(
        id="COND:unstable_angina",
        label="Unstable angina",
        category=Category.CARDIAC_ISCHAEMIC,
        is_must_not_miss=True,
        in_training_data=True,
        ddxplus_label="Unstable angina",
        red_flag_hint="Ischaemic-pattern chest pain at rest, or a worsening exertional pattern",
    ),
    Condition(
        id="COND:stable_angina",
        label="Stable angina",
        category=Category.CARDIAC_ISCHAEMIC,
        is_must_not_miss=False,
        in_training_data=True,
        ddxplus_label="Stable angina",
    ),
    # --- Cardiac: inflammatory --------------------------------------------- #
    Condition(
        id="COND:pericarditis",
        label="Pericarditis",
        category=Category.CARDIAC_INFLAMMATORY,
        is_must_not_miss=False,
        in_training_data=True,
        ddxplus_label="Pericarditis",
    ),
    Condition(
        id="COND:myocarditis",
        label="Myocarditis",
        category=Category.CARDIAC_INFLAMMATORY,
        is_must_not_miss=True,
        in_training_data=True,
        ddxplus_label="Myocarditis",
        red_flag_hint="Post-viral chest pain with breathlessness; can mimic infarction",
    ),
    # --- Cardiac: failure --------------------------------------------------- #
    Condition(
        id="COND:acute_pulmonary_edema",
        label="Acute pulmonary edema",
        category=Category.CARDIAC_FAILURE,
        is_must_not_miss=True,
        in_training_data=True,
        ddxplus_label="Acute pulmonary edema",
        red_flag_hint="Severe breathlessness with orthopnoea and frothy sputum",
    ),
    # --- Cardiac: arrhythmia ------------------------------------------------ #
    Condition(
        id="COND:atrial_fibrillation",
        label="Atrial fibrillation",
        category=Category.CARDIAC_ARRHYTHMIA,
        is_must_not_miss=False,
        in_training_data=True,
        ddxplus_label="Atrial fibrillation",
    ),
    Condition(
        id="COND:psvt",
        label="PSVT",
        category=Category.CARDIAC_ARRHYTHMIA,
        is_must_not_miss=False,
        in_training_data=True,
        ddxplus_label="PSVT",
        aliases=("paroxysmal supraventricular tachycardia",),
    ),
    # --- Non-cardiac mimics -------------------------------------------------- #
    Condition(
        id="COND:pulmonary_embolism",
        label="Pulmonary embolism",
        category=Category.MIMIC,
        is_must_not_miss=True,
        in_training_data=True,
        ddxplus_label="Pulmonary embolism",
        red_flag_hint="Pleuritic chest pain with breathlessness, unilateral leg swelling, "
        "or recent immobilisation/surgery",
    ),
    Condition(
        id="COND:spontaneous_pneumothorax",
        label="Spontaneous pneumothorax",
        category=Category.MIMIC,
        is_must_not_miss=True,
        in_training_data=True,
        ddxplus_label="Spontaneous pneumothorax",
        red_flag_hint="Sudden pleuritic chest pain with breathlessness",
    ),
    Condition(
        id="COND:boerhaave",
        label="Boerhaave syndrome",
        category=Category.MIMIC,
        is_must_not_miss=True,
        in_training_data=True,
        ddxplus_label="Boerhaave",
        red_flag_hint="Severe chest pain following forceful vomiting",
        aliases=("oesophageal rupture", "esophageal rupture"),
    ),
    Condition(
        id="COND:gerd",
        label="GERD",
        category=Category.MIMIC,
        is_must_not_miss=False,
        in_training_data=True,
        ddxplus_label="GERD",
        aliases=("gastro-oesophageal reflux disease", "acid reflux"),
    ),
    Condition(
        id="COND:panic_attack",
        label="Panic attack",
        category=Category.MIMIC,
        is_must_not_miss=False,
        in_training_data=True,
        ddxplus_label="Panic attack",
    ),
    # --- KG-only: not learnable from DDXPlus --------------------------------- #
    Condition(
        id="COND:aortic_dissection",
        label="Aortic dissection",
        category=Category.VASCULAR,
        is_must_not_miss=True,
        in_training_data=False,  # deliberately absent from DDXPlus
        ddxplus_label=None,
        red_flag_hint="Tearing chest pain radiating to the back, or a blood-pressure "
        "difference between arms",
    ),
)
"""The 13 trainable conditions plus aortic dissection.

Aortic dissection is the clearest justification for the knowledge graph in this
architecture: DDXPlus contains no cases of it, so the ML ranker can never score
it, yet it is among the most dangerous causes of chest pain. It is reachable
only through a KG red-flag rule (docs/04-safety-ethics.md §3).
"""


BY_ID: dict[str, Condition] = {c.id: c for c in CONDITIONS}

DDXPLUS_LABELS: dict[str, Condition] = {
    c.ddxplus_label: c for c in CONDITIONS if c.ddxplus_label is not None
}
"""Exact DDXPlus PATHOLOGY string -> Condition. Used to filter the dataset."""


def must_not_miss_ids() -> set[str]:
    """Conditions whose omission risks death. Drives must-not-miss recall."""
    return {c.id for c in CONDITIONS if c.is_must_not_miss}


def trainable_ids() -> set[str]:
    """Conditions the ML ranker can actually learn (present in DDXPlus)."""
    return {c.id for c in CONDITIONS if c.in_training_data}


def from_ddxplus_label(label: str) -> Condition | None:
    """Map a DDXPlus PATHOLOGY string to a Condition, or None if out of scope."""
    return DDXPLUS_LABELS.get(label.strip())
