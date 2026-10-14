"""Keeping treatment out of the evidence: which literature text may ever be shown (task 3a).

Nightingale never recommends a treatment or a drug (FR-6.5, docs/04 §3), and literature about a
condition talks about its treatment all the time: "colchicine reduces recurrence", "PCI within 90
minutes". A passage shown beside a candidate, under the system's name, would read as advice. So
**evidence is about diagnosis only, and the rule is conservative: a sentence with any treatment
language is never stored, indexed, embedded or shown.** Concretely:

1. **A sentence is screened out** when it matches the safety layer's own lexicon
   (:data:`src.reasoning.safety.TREATMENT`: drugs and drug classes, procedures, prescribing verbs,
   doses) **or** this module's broader :data:`THERAPY` pattern: words that put a sentence in a
   treatment context without naming a drug ("treated", "therapy", "management", "surgical",
   "ablation", "first-line"…) and the common drug-name stems ("-olol", "-sartan", "-parin"…).
   It errs towards dropping: "recent surgery" is a risk factor and is kept, but "patients with
   pericarditis managed as outpatients" goes, though it is about triage. Losing a diagnostic
   sentence costs a little recall; showing a treatment would break FR-6.5.
2. **What is kept is a run of consecutive clean sentences** (:func:`diagnostic_spans`), never
   clean sentences glued across a removed one: "X was suspected. [removed] Symptoms resolved."
   must not become "X was suspected. Symptoms resolved." A run shorter than
   :data:`MIN_SEGMENT_CHARS` is dropped as too short to stand alone. The Kaggle notebook chunks
   these runs into passages, so every passage is a verbatim, contiguous piece of its article.
3. **An article whose title names a treatment is dropped** (:func:`title_names_treatment`), since
   the title is shown with every excerpt: "Colchicine for recurrent pericarditis" would itself be
   a suggestion. Only the safety layer's lexicon applies to titles, so a review titled "Diagnosis
   and management of acute pericarditis" stays: its title names no treatment, and its excerpts
   are screened by rule 1.
4. **The laptop screens again** (src/rag/retriever.py): every title on load; every passage on
   load when the bundle was screened under other rules (an older notebook, or rules tightened
   since: the manifest records :data:`SCREEN_DIGEST`); and every excerpt just before it is shown.
   A passage that fails is never shown. Last, ``safety_check`` drops any evidence that
   recommends a treatment (src/reasoning/safety.py), whatever retriever produced it.

It is lexical, like the safety layer v1: it catches how papers usually phrase treatment, not every
paraphrase, and the sentence splitter is crude (src/rag/text.py). Its digest
(:data:`SCREEN_DIGEST`) goes into each bundle's manifest, so a bundle built under other rules can
be recognised. Standard library only, apart from the safety layer's pattern.
"""

from __future__ import annotations

import hashlib
import re

from src.rag.text import sentences
from src.reasoning.safety import TREATMENT

__all__ = [
    "MIN_SEGMENT_CHARS",
    "SCREEN_DIGEST",
    "THERAPY",
    "diagnostic_spans",
    "title_names_treatment",
    "treatment_language",
]

_CONTEXT = (
    r"treat\w*|\w*therap\w*|manag\w*|regimens?|drug(?![\s-]+(?:use|abuse|misuse|induced))s?|"
    r"medications?|medicines?|doses?|dosing|dosage\w*|prophyla\w*|pharmaco\w*|"
    r"first[- ]line|second[- ]line|rehabilitat\w*|"
    r"surgical\w*|(?<!recent )(?<!after )(?<!following )(?<!prior )(?<!previous )surger(?:y|ies)|"
    r"operation|operative(?:ly)?|intervention\w*|ablation|pacemakers?|pacing|defibrillat\w*|"
    r"implantable|revasculari[sz]\w*|thrombectom\w*|embolectom\w*|pericardiectom\w*|"
    r"o?esophagectom\w*|pleurodesis|thoracostom\w*|thoracotom\w*|needle aspiration|"
    r"transplant\w*|dialysis|analges\w*|painkillers?|anti-?inflammator\w*|antiarrhythmic\w*|"
    r"vasodilat\w*|inotrop\w*|vasopressors?|sedati\w*|anxiolytic\w*|antidepressants?|SSRIs?|"
    r"SNRIs?|cognitive[- ]behaviou?ral"
)
_DRUG_STEMS = (
    r"\w+(?:olol|sartan|dipine|parin|xaban|gatran|azepam|oxetine|mycin|cillin|floxacin|"
    r"cycline|statin|prazole)s?|\w{2,}pril"
)

THERAPY = re.compile(rf"\b(?:{_CONTEXT}|{_DRUG_STEMS})\b", flags=re.IGNORECASE)
"""Treatment context the safety layer's lexicon does not name: see the module docstring."""

MIN_SEGMENT_CHARS = 150
"""The shortest run of clean sentences kept: about one full sentence. Shorter runs are mostly
fragments between two removed sentences, which say little alone."""

SCREEN_DIGEST = hashlib.sha256(
    "\n".join([TREATMENT.pattern, THERAPY.pattern, str(MIN_SEGMENT_CHARS)]).encode("utf-8")
).hexdigest()[:16]
"""Identifies these rules; recorded in each bundle's manifest (``screening``)."""


def treatment_language(text: str) -> bool:
    """Whether ``text`` holds treatment language by either pattern (rule 1)."""
    return bool(TREATMENT.search(text) or THERAPY.search(text))


def title_names_treatment(title: str) -> bool:
    """Whether an article's title names a drug, a procedure, a dose or a prescribing verb: the
    safety layer's lexicon only (rule 3)."""
    return bool(TREATMENT.search(title))


def diagnostic_spans(text: str, min_chars: int = MIN_SEGMENT_CHARS) -> list[tuple[int, int]]:
    """``(start, end)`` character spans of the runs of consecutive sentences in ``text`` with no
    treatment language, each at least ``min_chars`` long (rule 2). ``text[start:end]`` is a
    verbatim piece of ``text`` that holds no treatment sentence."""
    spans: list[tuple[int, int]] = []
    run: tuple[int, int] | None = None
    for start, end in sentences(text):
        if treatment_language(text[start:end]):
            if run is not None:
                spans.append(run)
            run = None
        else:
            run = (start, end) if run is None else (run[0], end)
    if run is not None:
        spans.append(run)
    return [(s, e) for s, e in spans if len(text[s:e].strip()) >= min_chars]
