"""From Synthea's codes to Nightingale's risk-factor concepts (task 1d).

Synthea writes a patient's history as SNOMED CT codes (conditions, and the values of some
observations) and LOINC codes (observations). The red-flag rules, the graph and the model read
hand-authored ``RF:*`` concepts (src/medical_kg/crosswalk.py). This table is the bridge, and it is
deliberately small: a code is mapped only where it means what the concept means.

**Every mapped code was read in Synthea's own module files, at the pinned release**
(:data:`SYNTHEA_VERSION`; ``src/main/resources/modules/`` on GitHub, read 2026-10-14), and each
entry names the file. A code that could not be checked there would carry ``verify=True``; none
does. The findings it leads to have ``source="synthea"``, a source ``src/contracts.py`` already
allows, and it adds no ``RF:*`` concept: each concept is one the crosswalk already holds, and a
test enforces it.

**What is not mapped, and why**, is kept too: codes Synthea writes that resemble a concept but are
not it (:data:`NOT_MAPPED`: prediabetes is not diabetes, raised triglycerides are not raised
cholesterol), and the concepts no checked module writes at all (:data:`CONCEPTS_WITHOUT_CODES`).
Any other code is simply outside the table, and the builder counts it with that reason.

**Medications are history, never advice.** No medication becomes a finding, and nothing here
suggests one. A patient's medications stay in their graph as what they took (src/patient_kg/graph.py).

Reference: docs/16 §1 (1d's plan), docs/17 T7, docs/02 §5.2 (the crosswalk).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.contracts import Assertion

__all__ = [
    "CONCEPTS_WITHOUT_CODES",
    "CONDITION_CODES",
    "NOT_MAPPED",
    "SMOKING_STATUS",
    "SMOKING_STATUS_LOINC",
    "SYNTHEA_VERSION",
    "CodeMapping",
]

SYNTHEA_VERSION = "v4.0.0"
"""The Synthea release the notebook runs (notebooks/kaggle_synthea.ipynb) and whose module files
the table was checked against."""


@dataclass(frozen=True)
class CodeMapping:
    """One Synthea code and the concept it stands for.

    Attributes:
        code: The SNOMED CT code, as Synthea writes it.
        display: Synthea's display text for it, as its module writes it.
        concept_id: The ``RF:*`` concept (a crosswalk entry).
        assertion: What the code says about the concept: present, or (a never-smoker) absent.
        module: The Synthea module file the code was read in.
        verify: True if the code could not be checked in Synthea's files. None is, today.
    """

    code: str
    display: str
    concept_id: str
    assertion: Assertion
    module: str
    verify: bool = False


def _present(code: str, display: str, concept: str, module: str) -> CodeMapping:
    return CodeMapping(code, display, concept, Assertion.PRESENT, module)


CONDITION_CODES: dict[str, CodeMapping] = {
    m.code: m
    for m in (
        _present(
            "59621000", "Essential hypertension (disorder)", "RF:hypertension", "hypertension.json"
        ),
        _present(
            "44054006",
            "Diabetes mellitus type 2 (disorder)",
            "RF:diabetes",
            "metabolic_syndrome_care.json",
        ),
        _present(
            "88805009",
            "Chronic congestive heart failure (disorder)",
            "RF:heart_failure",
            "congestive_heart_failure.json",
        ),
        _present(
            "162864005",
            "Body mass index 30+ - obesity (finding)",
            "RF:obesity",
            "wellness_encounters.json",
        ),
        _present(
            "408512008",
            "Body mass index 40+ - severely obese (finding)",
            "RF:obesity",
            "wellness_encounters.json",
        ),
        _present("72892002", "Normal pregnancy (finding)", "RF:pregnancy", "pregnancy.json"),
        _present("47200007", "High risk pregnancy (finding)", "RF:pregnancy", "pregnancy.json"),
        _present(
            "449868002",
            "Smokes tobacco daily (finding)",
            "RF:smoking",
            "veteran_substance_abuse_treatment.json",
        ),
    )
}
"""Condition codes (conditions.csv's ``CODE``), each counted while the condition is active."""

SMOKING_STATUS_LOINC = "72166-2"
"""LOINC "Tobacco smoking status", the observation Synthea records at wellness visits
(wellness_encounters.json), whose value is a SNOMED finding."""

SMOKING_STATUS: dict[str, CodeMapping] = {
    m.display: m
    for m in (
        _present(
            "449868002", "Smokes tobacco daily (finding)", "RF:smoking", "wellness_encounters.json"
        ),
        CodeMapping(
            "8517006",
            "Ex-smoker (finding)",
            "RF:smoking",
            Assertion.ABSENT,
            "wellness_encounters.json",
        ),
        CodeMapping(
            "266919005",
            "Never smoked tobacco (finding)",
            "RF:smoking",
            Assertion.ABSENT,
            "wellness_encounters.json",
        ),
    )
}
"""The smoking-status values, keyed by display: the CSV holds a coded value's display text, not
its code (Synthea's ExportHelper.getObservationValue). ``RF:smoking`` is daily smoking now (the
crosswalk: 'fumez-vous la cigarette quotidiennement'), so a former smoker answers no, as a
never-smoker does."""

NOT_MAPPED: dict[str, str] = {
    "714628002": "Prediabetes (finding) is not diabetes (RF:diabetes)",
    "302870006": "Hypertriglyceridemia (disorder): raised triglycerides, not raised cholesterol "
    "(RF:hypercholesterolaemia)",
    "49436004": "Atrial fibrillation (disorder) is a condition Nightingale ranks, not a risk "
    "factor it reads",
    "368581000119106": "Neuropathy due to type 2 diabetes mellitus (disorder): a complication; "
    "its diabetes is coded on its own (44054006)",
}
"""Codes the checked modules write that resemble a concept but are not it."""

CONCEPTS_WITHOUT_CODES: dict[str, str] = {
    "RF:hypercholesterolaemia": "no hyperlipidaemia or hypercholesterolaemia diagnosis code in "
    "the modules read; only hypertriglyceridaemia, which is not it",
    "RF:previous_dvt": "no deep vein thrombosis code in the modules read",
    "RF:recent_surgery": "'recent' needs a presentation date, and a procedure list is not a "
    "surgery list: not inferred from procedures.csv",
    "RF:aortic_valve_disease": "no aortic valve disease code in the modules read",
    "RF:cocaine_use": "no cocaine code in the modules read",
    "RF:connective_tissue_disease": "Synthea models no Marfan, Loeys-Dietz or Ehlers-Danlos "
    "syndrome in the modules read",
    "RF:thoracic_aortic_aneurysm": "no thoracic aortic aneurysm code in the modules read",
    "RF:family_history_aortic_disease": "Synthea records no family history",
    "RF:aortic_manipulation": "not inferred from procedures.csv",
}
"""The ``RF:*`` concepts no checked module writes, so no Synthea patient will have them."""
