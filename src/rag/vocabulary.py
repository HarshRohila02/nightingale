"""What retrieval searches for: per condition, the literature searches and the words that make a
passage about it; per finding, the text its query vector is embedded from (task 3a).

Hand-curated, like the crosswalk, and read by both sides:

* **The Kaggle notebook** (``notebooks/kaggle_retrieval_corpus.ipynb``) builds its PMC searches
  from :data:`CONDITION_TERMS` (:func:`esearch_terms`) and embeds :func:`query_vocabulary`, a
  fixed set of query texts, once, on the GPU.
* **The laptop** (src/rag/retriever.py) never runs the embedding model. It looks the precomputed
  vectors up by condition and concept, and uses :data:`CONDITION_TERMS`'s ``mentions`` to decide
  which passages may be shown under a condition at all.

A change to ``title_terms`` changes the corpus and needs the notebook re-run; a change to
``mentions`` takes effect on the laptop at once; a change to the query texts needs only the
notebook's embedding cells (see the runbook).

**These are search terms, not medical knowledge**: none of it reaches a ranking. Evidence is shown
beside a candidate the ranker already chose (docs/02 §3), and about diagnosis only: no search here
asks for treatment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from src.conditions import BY_ID, CONDITIONS

__all__ = [
    "CONDITION_TERMS",
    "FOCUS",
    "QUERY_TEMPLATES",
    "ConditionTerms",
    "compose_query_text",
    "concept_query_texts",
    "condition_query_texts",
    "esearch_terms",
    "query_vocabulary",
    "resolve_condition",
]


@dataclass(frozen=True)
class ConditionTerms:
    """How retrieval names one condition.

    Attributes:
        condition_id: The registry id (src/conditions.py).
        name: A readable name for the query templates, e.g. "acute pericarditis".
        title_terms: Phrases one of which an article's title must contain to be fetched (PMC
            ESearch, ``[ti]``). Spelled as authors spell them, British and American.
        mentions: Phrases one of which a passage, or its article's title, must contain to be shown
            under this condition. Compared as tokens (src/rag/text.py), so plurals and
            British/American spellings are already folded.
    """

    condition_id: str
    name: str
    title_terms: tuple[str, ...]
    mentions: tuple[str, ...]


def _terms(cid: str, name: str, titles: Sequence[str], mentions: Sequence[str]) -> ConditionTerms:
    return ConditionTerms(cid, name, tuple(titles), tuple(mentions))


# fmt: off
CONDITION_TERMS: Mapping[str, ConditionTerms] = {
    t.condition_id: t
    for t in (
        _terms("COND:nstemi_stemi", "acute myocardial infarction (NSTEMI or STEMI)",
               ["myocardial infarction", "NSTEMI", "STEMI", "acute coronary syndrome"],
               ["myocardial infarction", "nstemi", "stemi", "acute coronary syndrome",
                "heart attack"]),
        _terms("COND:unstable_angina", "unstable angina",
               ["unstable angina"],
               ["unstable angina"]),
        _terms("COND:stable_angina", "stable angina",
               ["stable angina", "angina pectoris", "chronic coronary syndrome"],
               ["stable angina", "angina pectoris", "chronic coronary syndrome",
                "stable coronary artery disease"]),
        _terms("COND:pericarditis", "acute pericarditis",
               ["pericarditis"],
               ["pericarditis", "myopericarditis", "perimyocarditis"]),
        _terms("COND:myocarditis", "acute myocarditis",
               ["myocarditis"],
               ["myocarditis", "myopericarditis", "perimyocarditis"]),
        _terms("COND:acute_pulmonary_edema", "acute pulmonary edema",
               ["pulmonary edema", "pulmonary oedema", "acute heart failure"],
               ["pulmonary edema", "acute heart failure", "acute decompensated heart failure"]),
        _terms("COND:atrial_fibrillation", "atrial fibrillation",
               ["atrial fibrillation"],
               ["atrial fibrillation"]),
        _terms("COND:psvt", "paroxysmal supraventricular tachycardia",
               ["supraventricular tachycardia", "PSVT", "AVNRT"],
               ["supraventricular tachycardia", "psvt", "svt", "avnrt"]),
        _terms("COND:pulmonary_embolism", "acute pulmonary embolism",
               ["pulmonary embolism", "pulmonary embolus"],
               ["pulmonary embolism", "pulmonary embolus", "pulmonary emboli"]),
        _terms("COND:spontaneous_pneumothorax", "spontaneous pneumothorax",
               ["pneumothorax"],
               ["pneumothorax", "pneumothoraces"]),
        _terms("COND:boerhaave", "Boerhaave syndrome (spontaneous esophageal rupture)",
               ["Boerhaave", "esophageal rupture", "oesophageal rupture",
                "esophageal perforation", "oesophageal perforation"],
               ["boerhaave", "esophageal rupture", "esophageal perforation"]),
        _terms("COND:gerd", "gastroesophageal reflux disease",
               ["gastroesophageal reflux", "gastro-oesophageal reflux", "GERD", "GORD"],
               ["gastroesophageal reflux", "gastro esophageal reflux", "gerd", "gord",
                "reflux esophagitis", "acid reflux"]),
        _terms("COND:panic_attack", "panic attack",
               ["panic attack", "panic attacks", "panic disorder"],
               ["panic attack", "panic disorder"]),
        _terms("COND:aortic_dissection", "acute aortic dissection",
               ["aortic dissection", "acute aortic syndrome"],
               ["aortic dissection", "acute aortic syndrome", "dissection of the aorta"]),
    )
}
# fmt: on
"""Every condition in the registry, and nothing else (tests/test_retrieval.py checks)."""

FOCUS: Mapping[str, str] = {
    "diagnosis": (
        "(diagnosis[ti] OR diagnostic[ti] OR presentation[ti] OR review[ti] OR "
        '"case report"[ti] OR "chest pain"[ti])'
    ),
    "chest_pain": '("chest pain"[ab] OR "differential diagnosis"[ab])',
    "any": "",
}
"""What each PMC search adds to a condition's title terms, in the order the notebook fills a
condition's quota: articles about its diagnosis first, then ones whose abstract places it among the
causes of chest pain, then any article about it."""

QUERY_TEMPLATES: tuple[str, ...] = (
    "{name}",
    "symptoms and signs of {name}",
    "how is {name} diagnosed in a patient with acute chest pain",
)
"""A condition's query vector is the mean of these texts' embeddings: its name, its presentation
and its diagnosis, short questions of the kind the MS MARCO-trained model was tuned on."""


def resolve_condition(condition: str) -> str:
    """A condition id from an id, a registry label or an alias, case-insensitively.

    The pipeline passes the candidate's label (``retrieve(candidate.label, case)``), not its id.

    Raises:
        ValueError: for a condition the registry does not know (R-13: there are 14).
    """
    if condition in BY_ID:
        return condition
    wanted = condition.strip().casefold()
    for c in CONDITIONS:
        if wanted == c.label.casefold() or wanted in (a.casefold() for a in c.aliases):
            return c.id
    raise ValueError(f"{condition!r} is not one of the conditions Nightingale knows")


def condition_query_texts(condition_id: str) -> tuple[str, ...]:
    """The texts whose embeddings make up ``condition_id``'s query vector."""
    name = CONDITION_TERMS[condition_id].name
    return tuple(t.format(name=name) for t in QUERY_TEMPLATES)


def concept_query_texts() -> dict[str, str]:
    """{concept id: text}: one query text per hand-authored concept, its crosswalk label.

    Every concept a form can offer (``BY_CONCEPT``) gets a vector, so any finding entered in the
    dashboard can steer the dense query. A concept added later has none until the notebook's
    vocabulary cell is re-run; the retriever then names it in ``missing_concepts`` and searches
    without it.
    """
    from src.medical_kg.crosswalk import BY_CONCEPT

    return {cid: entry.label for cid, entry in BY_CONCEPT.items()}


def compose_query_text(condition_id: str, concept_ids: Iterable[str] = ()) -> str:
    """The keyword query for ``condition_id`` and a case's present findings: the condition's name,
    then each finding's label. BM25 reads it; the notebook also embeds it for a few cases, so that
    EXP-011 can measure how far the laptop's mixed query vector is from the real one."""
    labels = concept_query_texts()
    words = [CONDITION_TERMS[condition_id].name]
    words += [labels[c] for c in dict.fromkeys(concept_ids) if c in labels]
    return " ".join(words)


def esearch_terms(condition_id: str) -> list[tuple[str, str]]:
    """``[(focus, ESearch term)]`` for one condition, in :data:`FOCUS`'s order.

    Every term ends with ``"open access"[filter]``, PMC's filter for its Open Access subset, the
    only part of PMC this project may keep (docs/03 §1.4).
    """
    titles = " OR ".join(f'"{t}"[ti]' for t in CONDITION_TERMS[condition_id].title_terms)
    out = []
    for focus, clause in FOCUS.items():
        parts = [f"({titles})"] + ([clause] if clause else []) + ['"open access"[filter]']
        out.append((focus, " AND ".join(parts)))
    return out


def query_vocabulary(checks: Mapping[str, Sequence[str]] | None = None) -> list[dict[str, str]]:
    """The fixed list of query texts the notebook embeds, one row per vector, in this order:

    * ``condition`` rows: :func:`condition_query_texts` for each condition;
    * ``concept`` rows: :func:`concept_query_texts`;
    * ``check`` rows: for each named case in ``checks`` ({case id: present concept ids}) and each
      condition, :func:`compose_query_text`. They are never searched with; EXP-011 compares them
      with the mixed vectors the laptop builds for the same case.
    """
    rows = [
        {"kind": "condition", "id": cid, "text": text}
        for cid in CONDITION_TERMS
        for text in condition_query_texts(cid)
    ]
    rows += [{"kind": "concept", "id": cid, "text": t} for cid, t in concept_query_texts().items()]
    for case_id, concepts in (checks or {}).items():
        rows += [
            {"kind": "check", "id": f"{case_id}|{cid}", "text": compose_query_text(cid, concepts)}
            for cid in CONDITION_TERMS
        ]
    return rows
