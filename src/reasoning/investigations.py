"""Tests to consider next: a small, hand-authored, cited table (task 3d; FR-5.5).

Each of the 14 conditions maps to the **diagnostic investigations** that confirm or exclude it, as
the guideline cited for it recommends. :func:`investigations_to_consider` picks the entries for the
leading candidates and every red-flagged one, and the pipeline puts one line per condition into
``DiagnosisResult.suggested_next_tests``:

    "Pulmonary embolism: consider a clinical probability assessment (Wells or revised Geneva
    score); D-dimer if that probability is not high; CT pulmonary angiography if it is high or the
    D-dimer is raised (ESC 2019 PE)"

**Investigations only, never a treatment or a drug** (FR-6.5, docs/04 §3). Every line says
"consider": the system supports a decision, it does not make one. The safety layer
(src/reasoning/safety.py) would remove any line with treatment language anyway, and a test scans
this table for it. Where a guideline's next investigation depends on a result ("D-dimer if the
probability is not high"), the condition is kept, because it is what makes the order of the tests
right; nothing here goes beyond the next investigation.

**What this table is not.** It is not derived from the knowledge graph, which holds no test nodes:
docs/02 §5 drew a ``Test`` node and a ``DISCRIMINATED_BY`` edge, and neither was ever built. It is
the smallest honest substitute, written from the guidelines below, one entry per condition, and
it does not rank tests against each other. The questions worth asking first, which *are* computed
from the graph, are :mod:`src.reasoning.next_questions`.

**The references** (:data:`SOURCES`) are guidelines or position statements of the societies named.
Each entry's investigations paraphrase that source's diagnostic recommendations; none quotes it.
Two entries rest on a reference marked ``verify=True``: the citation details were written from
memory and must be checked against the publisher before the report cites them.

Reference: docs/01-srs.md FR-5.5 and FR-6.5, docs/04-safety-ethics.md §3 (the must-not-miss table).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.conditions import BY_ID

__all__ = [
    "INVESTIGATIONS",
    "SOURCES",
    "TESTS_FOR_TOP",
    "Investigation",
    "Source",
    "investigations_to_consider",
    "suggested_tests",
]

TESTS_FOR_TOP = 3
"""How many leading candidates get their tests, beside every red-flagged one. Three, because the
headline metrics judge the top 3 (docs/05 §3)."""


@dataclass(frozen=True)
class Source:
    """A published guideline or position statement.

    Attributes:
        short: The tag shown beside a suggestion, e.g. "ESC 2019 PE".
        reference: The full citation.
        verify: True when the citation's details (volume, pages, authors) were written from memory
            and have not been checked against the publisher. Flagged for the report's author.
    """

    short: str
    reference: str
    verify: bool = False


SOURCES: dict[str, Source] = {
    "aha_acc_chest_pain_2021": Source(
        "AHA/ACC 2021 chest pain",
        "Gulati M, Levy PD, Mukherjee D, et al. 2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR Guideline "
        "for the Evaluation and Diagnosis of Chest Pain. Circulation. 2021;144(22):e368-e454.",
    ),
    "esc_acs_2023": Source(
        "ESC 2023 ACS",
        "Byrne RA, Rossello X, Coughlan JJ, et al. 2023 ESC Guidelines for the management of acute "
        "coronary syndromes. Eur Heart J. 2023;44(38):3720-3826.",
    ),
    "esc_ccs_2024": Source(
        "ESC 2024 CCS",
        "Vrints C, Andreotti F, Koskinas KC, et al. 2024 ESC Guidelines for the management of "
        "chronic coronary syndromes. Eur Heart J. 2024;45(36):3415-3537.",
    ),
    "nice_cg95": Source(
        "NICE CG95",
        "National Institute for Health and Care Excellence. Recent-onset chest pain of suspected "
        "cardiac origin: assessment and diagnosis. Clinical guideline CG95, 2010 (updated 2016).",
    ),
    "esc_pericardial_2015": Source(
        "ESC 2015 pericardial",
        "Adler Y, Charron P, Imazio M, et al. 2015 ESC Guidelines for the diagnosis and management "
        "of pericardial diseases. Eur Heart J. 2015;36(42):2921-2964.",
    ),
    "esc_myocarditis_2013": Source(
        "ESC 2013 myocarditis",
        "Caforio ALP, Pankuweit S, Arbustini E, et al. Current state of knowledge on aetiology, "
        "diagnosis, management, and therapy of myocarditis: a position statement of the European "
        "Society of Cardiology Working Group on Myocardial and Pericardial Diseases. Eur Heart J. "
        "2013;34(33):2636-2648.",
    ),
    "cmr_myocarditis_2018": Source(
        "JACC 2018 CMR",
        "Ferreira VM, Schulz-Menger J, Holmvang G, et al. Cardiovascular magnetic resonance in "
        "nonischemic myocardial inflammation: expert recommendations. J Am Coll Cardiol. "
        "2018;72(24):3158-3176.",
    ),
    "esc_hf_2021": Source(
        "ESC 2021 heart failure",
        "McDonagh TA, Metra M, Adamo M, et al. 2021 ESC Guidelines for the diagnosis and treatment "
        "of acute and chronic heart failure. Eur Heart J. 2021;42(36):3599-3726.",
    ),
    "esc_af_2024": Source(
        "ESC 2024 AF",
        "Van Gelder IC, Rienstra M, Bunting KV, et al. 2024 ESC Guidelines for the management of "
        "atrial fibrillation developed in collaboration with EACTS. Eur Heart J. "
        "2024;45(36):3314-3414.",
    ),
    "esc_svt_2019": Source(
        "ESC 2019 SVT",
        "Brugada J, Katritsis DG, Arbelo E, et al. 2019 ESC Guidelines for the management of "
        "patients with supraventricular tachycardia. Eur Heart J. 2020;41(5):655-720.",
    ),
    "esc_pe_2019": Source(
        "ESC 2019 PE",
        "Konstantinides SV, Meyer G, Becattini C, et al. 2019 ESC Guidelines for the diagnosis and "
        "management of acute pulmonary embolism developed in collaboration with the European "
        "Respiratory Society (ERS). Eur Heart J. 2020;41(4):543-603.",
    ),
    "bts_pleural_2023": Source(
        "BTS 2023 pleural",
        "Roberts ME, Rahman NM, Maskell NA, et al. British Thoracic Society Guideline for pleural "
        "disease. Thorax. 2023;78(Suppl 3):s1-s42.",
        verify=True,
    ),
    "bts_pneumothorax_2010": Source(
        "BTS 2010 pneumothorax",
        "MacDuff A, Arnold A, Harvey J; BTS Pleural Disease Guideline Group. Management of "
        "spontaneous pneumothorax: British Thoracic Society pleural disease guideline 2010. "
        "Thorax. 2010;65(Suppl 2):ii18-ii31.",
    ),
    "wses_oesophageal_2019": Source(
        "WSES 2019 oesophageal",
        "Chirica M, Kelly MD, Siboni S, et al. Esophageal emergencies: WSES guidelines. World J "
        "Emerg Surg. 2019;14:26.",
        verify=True,
    ),
    "acg_gerd_2022": Source(
        "ACG 2022 GERD",
        "Katz PO, Dunbar KB, Schnoll-Sussman FH, Greer KB, Yadlapati R, Spechler SJ. ACG Clinical "
        "Guideline for the Diagnosis and Management of Gastroesophageal Reflux Disease. Am J "
        "Gastroenterol. 2022;117(1):27-56.",
    ),
    "acc_aha_aorta_2022": Source(
        "ACC/AHA 2022 aortic",
        "Isselbacher EM, Preventza O, Hamilton Black J 3rd, et al. 2022 ACC/AHA Guideline for the "
        "Diagnosis and Management of Aortic Disease. Circulation. 2022;146(24):e334-e482.",
    ),
    "esc_aorta_2014": Source(
        "ESC 2014 aortic",
        "Erbel R, Aboyans V, Boileau C, et al. 2014 ESC Guidelines on the diagnosis and treatment "
        "of aortic diseases. Eur Heart J. 2014;35(41):2873-2926.",
    ),
}
"""Every guideline the table cites, by key. ``verify`` marks the two whose details need checking.

Newer ESC guidance may supersede two of them: a 2025 guideline on myocarditis and pericarditis, and
a 2024 guideline on peripheral arterial and aortic diseases, are understood to exist. They are not
cited because their details could not be checked here; whether they change any investigation
listed is for the report's author to confirm.
"""


@dataclass(frozen=True)
class Investigation:
    """The diagnostic investigations that confirm or exclude one condition.

    Attributes:
        condition_id: The condition.
        tests: Noun phrases, in the order the guideline uses them, each one an investigation, never
            a treatment. A condition on a test ("if the probability is not high") is part of it.
        sources: Keys of :data:`SOURCES`.
    """

    condition_id: str
    tests: tuple[str, ...]
    sources: tuple[str, ...]

    @property
    def label(self) -> str:
        return BY_ID[self.condition_id].label

    def text(self) -> str:
        """One line for ``suggested_next_tests``: the condition, "consider", the tests, the tags."""
        tags = "; ".join(SOURCES[s].short for s in self.sources)
        return f"{self.label}: consider {'; '.join(self.tests)} ({tags})"


def _entry(condition_id: str, tests: Sequence[str], *sources: str) -> Investigation:
    return Investigation(condition_id, tuple(tests), tuple(sources))


INVESTIGATIONS: dict[str, Investigation] = {
    e.condition_id: e
    for e in (
        # --- Ischaemic: the ECG within 10 minutes and high-sensitivity troponin (ESC 2023; AHA/ACC
        # 2021 recommends an ECG within 10 minutes of arrival for every acute chest pain) ------- #
        _entry(
            "COND:nstemi_stemi",
            (
                "a 12-lead ECG within 10 minutes of arrival",
                "high-sensitivity cardiac troponin, repeated by the 0/1-hour or 0/2-hour algorithm",
            ),
            "esc_acs_2023",
            "aha_acc_chest_pain_2021",
        ),
        _entry(
            "COND:unstable_angina",
            (
                "a 12-lead ECG, repeated if the symptoms recur",
                "serial high-sensitivity cardiac troponin: ischaemic symptoms without a troponin "
                "rise point to unstable angina rather than NSTEMI",
            ),
            "esc_acs_2023",
        ),
        _entry(
            "COND:stable_angina",
            (
                "a resting 12-lead ECG",
                "CT coronary angiography or functional (stress) imaging, once an acute coronary "
                "syndrome is excluded",
            ),
            "esc_ccs_2024",
            "nice_cg95",
        ),
        # --- Inflammatory ----------------------------------------------------------------------- #
        _entry(
            "COND:pericarditis",
            (
                "a 12-lead ECG (widespread ST elevation, PR depression)",
                "transthoracic echocardiography for an effusion",
                "CRP and troponin",
                "a chest X-ray",
            ),
            "esc_pericardial_2015",
        ),
        _entry(
            "COND:myocarditis",
            (
                "a 12-lead ECG",
                "high-sensitivity cardiac troponin",
                "transthoracic echocardiography",
                "cardiac MRI",
            ),
            "esc_myocarditis_2013",
            "cmr_myocarditis_2018",
        ),
        # --- Failure and arrhythmia --------------------------------------------------------------- #
        _entry(
            "COND:acute_pulmonary_edema",
            (
                "a chest X-ray",
                "a natriuretic peptide (BNP or NT-proBNP)",
                "transthoracic echocardiography",
                "a 12-lead ECG",
            ),
            "esc_hf_2021",
        ),
        _entry(
            "COND:atrial_fibrillation",
            ("a 12-lead ECG to document the rhythm", "transthoracic echocardiography"),
            "esc_af_2024",
        ),
        _entry(
            "COND:psvt",
            (
                "a 12-lead ECG, during symptoms if possible",
                "ambulatory ECG monitoring (Holter or event recorder) if the episodes have stopped",
            ),
            "esc_svt_2019",
        ),
        # --- Non-cardiac mimics ------------------------------------------------------------------- #
        _entry(
            "COND:pulmonary_embolism",
            (
                "a clinical probability assessment (Wells or revised Geneva score)",
                "D-dimer if that probability is not high",
                "CT pulmonary angiography if it is high or the D-dimer is raised",
            ),
            "esc_pe_2019",
        ),
        _entry(
            "COND:spontaneous_pneumothorax",
            ("an erect chest X-ray", "CT of the chest if the X-ray is uncertain"),
            "bts_pleural_2023",
            "bts_pneumothorax_2010",
        ),
        _entry(
            "COND:boerhaave",
            ("CT of the chest with oral contrast", "a water-soluble contrast oesophagram"),
            "wses_oesophageal_2019",
        ),
        _entry(
            "COND:gerd",
            (
                "a 12-lead ECG and troponin first, to exclude a cardiac cause",
                "then upper gastrointestinal endoscopy or ambulatory reflux monitoring",
            ),
            "acg_gerd_2022",
            "aha_acc_chest_pain_2021",
        ),
        _entry(
            "COND:panic_attack",
            (
                "a 12-lead ECG, and troponin if an acute coronary syndrome is possible, to exclude "
                "a cardiac cause before attributing the pain to panic",
            ),
            "aha_acc_chest_pain_2021",
        ),
        # --- Vascular ----------------------------------------------------------------------------- #
        _entry(
            "COND:aortic_dissection",
            (
                "CT angiography of the aorta",
                "echocardiography (transthoracic, or transoesophageal) when CT is not immediately "
                "possible",
            ),
            "acc_aha_aorta_2022",
            "esc_aorta_2014",
        ),
    )
}
"""Condition -> its investigations. Every condition in ``src/conditions.py`` has an entry, and a
test enforces it. The tests are listed in each guideline's order, not ranked against each other."""


def investigations_to_consider(
    candidates: Iterable, top: int = TESTS_FOR_TOP
) -> list[Investigation]:
    """The investigations for the first ``top`` candidates and every red-flagged one, in order.

    Args:
        candidates: The pipeline's candidates in its order (red flags first, then the fused score),
            anything with ``condition_id`` and ``red_flag``.
        top: How many leading candidates to cover beside the flagged ones.

    Returns:
        One :class:`Investigation` per condition, each condition once: the red-flagged first, as
        the pipeline orders them. A condition without an entry is skipped.
    """
    chosen: list[Investigation] = []
    seen: set[str] = set()
    for rank, candidate in enumerate(candidates):
        cid = candidate.condition_id
        if (rank < top or candidate.red_flag) and cid not in seen and cid in INVESTIGATIONS:
            seen.add(cid)
            chosen.append(INVESTIGATIONS[cid])
    return chosen


def suggested_tests(candidates: Iterable, top: int = TESTS_FOR_TOP) -> list[str]:
    """``DiagnosisResult.suggested_next_tests``: one line per condition of
    :func:`investigations_to_consider`."""
    return [entry.text() for entry in investigations_to_consider(candidates, top)]
