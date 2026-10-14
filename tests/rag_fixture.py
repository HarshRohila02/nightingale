"""A tiny, invented retrieval bundle for the tests (task 3a).

**Everything here is made up**: the titles, the passages, the journals and the authors are written
for these tests, and the PMCIDs (PMC99990001…) lie far beyond PMC's range, so none names a real
article. No real article text is used anywhere in the tests: the corpus itself is never committed
(docs/03 §1.4).

The vectors are hand-made in 8 dimensions, one axis per idea, so every cosine can be worked out
by hand: e0 pericarditis, e1 pulmonary embolism, e2 pleuritic pain, e3 fever, e4 leg swelling,
e5 an unrelated topic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.rag.bundle import write_jsonl, write_manifest
from src.rag.corpus import article_url
from src.rag.screening import SCREEN_DIGEST

__all__ = ["ARTICLES", "DIMENSION", "PASSAGES", "axis", "mixed", "write_fake_bundle"]

DIMENSION = 8


def axis(i: int, scale: float = 1.0) -> np.ndarray:
    vector = np.zeros(DIMENSION)
    vector[i] = scale
    return vector


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def mixed(condition: np.ndarray, *findings: np.ndarray, weight: float = 0.6) -> np.ndarray:
    """The retriever's dense query, worked out independently (src/rag/retriever.py, step 3)."""
    if not findings:
        return _unit(condition)
    return _unit(weight * _unit(condition) + (1 - weight) * _unit(np.mean(findings, axis=0)))


def _article(pmcid: str, title: str, licence: str, licence_url: str) -> dict:
    return {
        "pmcid": pmcid,
        "title": title,
        "licence": licence,
        "licence_url": licence_url,
        "pmid": None,
        "doi": None,
        "journal": "Invented Journal of Test Medicine",
        "year": 2020,
        "authors": "Doe J et al.",
        "article_type": "research-article",
        "url": article_url(pmcid),
    }


CC_BY = "https://creativecommons.org/licenses/by/4.0/"
ARTICLES = [
    _article("PMC99990001", "Pericarditis with chest pain: an invented case", "CC BY 4.0", CC_BY),
    _article(
        "PMC99990002",
        "Pulmonary embolism: an invented review",
        "CC BY-NC 4.0",
        "https://creativecommons.org/licenses/by-nc/4.0/",
    ),
    _article("PMC99990003", "Chest pain in an invented emergency room", "CC BY 4.0", CC_BY),
    # Its title names a drug, so none of its passages may ever be shown (src/rag/screening.py).
    _article("PMC99990004", "Colchicine for invented pericarditis", "CC BY 4.0", CC_BY),
    _article("PMC99990005", "Pericarditis: an invented series", "CC BY 4.0", CC_BY),
]

PASSAGES = [
    (
        "PMC99990001",
        "Abstract",
        "Acute pericarditis typically causes sharp pleuritic chest pain that eases when sitting "
        "forward. A friction rub may be heard.",
        axis(0) + axis(2, 0.5),
    ),
    (
        "PMC99990001",
        "Case",
        "Pericarditis often follows a viral illness, and fever is common at presentation.",
        axis(0) + axis(3, 0.5),
    ),
    (
        "PMC99990002",
        "Abstract",
        "Pulmonary embolism presents with sudden breathlessness and pleuritic pain. Unilateral leg "
        "swelling suggests a deep vein thrombosis.",
        axis(1) + axis(2, 0.5) + axis(4, 0.3),
    ),
    (
        "PMC99990003",
        "Discussion",
        "Pleuritic chest pain has many causes, including pericarditis and pulmonary embolism.",
        axis(0, 0.5) + axis(1, 0.5) + axis(2),
    ),
    (
        "PMC99990003",
        "Discussion",
        "Chest pain with sweating calls for an electrocardiogram.",
        axis(5),
    ),
    (
        "PMC99990004",
        "Abstract",
        "Pericarditis was recognised from its electrocardiogram changes.",
        axis(0),
    ),
    # Treatment language in the text: a bundle built by today's notebook would not hold it, and the
    # laptop must hide it all the same (an older bundle, or rules tightened since).
    (
        "PMC99990005",
        "Results",
        "Patients with pericarditis were treated with ibuprofen.",
        axis(0) + axis(3, 0.1),
    ),
]

FEVER_CHECK = mixed(axis(0), axis(3))
"""The check row's vector: equal to the mixed query for pericarditis with fever, so the mix's
fidelity is exactly 1 on it."""


def query_rows() -> list[tuple[dict, np.ndarray]]:
    rows = [
        ({"kind": "condition", "id": "COND:pericarditis", "text": "acute pericarditis"}, axis(0)),
        (
            {"kind": "condition", "id": "COND:pericarditis", "text": "signs of pericarditis"},
            axis(0) + axis(6, 0.2),
        ),
        (
            {"kind": "condition", "id": "COND:pericarditis", "text": "diagnosing pericarditis"},
            axis(0) - axis(6, 0.2),
        ),
        ({"kind": "condition", "id": "COND:pulmonary_embolism", "text": "embolism"}, axis(1)),
        ({"kind": "concept", "id": "SYM:pleuritic", "text": "Pleuritic pain"}, axis(2)),
        ({"kind": "concept", "id": "SYM:fever", "text": "Fever"}, axis(3)),
        (
            {"kind": "concept", "id": "SYM:leg_swelling_unilateral", "text": "Unilateral leg"},
            axis(4),
        ),
        (
            {
                "kind": "check",
                "id": "GC-T|COND:pericarditis",
                "text": "acute pericarditis Fever",
                "condition": "COND:pericarditis",
                "concepts": ["SYM:fever"],
            },
            FEVER_CHECK,
        ),
    ]
    return rows


def write_fake_bundle(path: Path, *, passage_dtype: str = "float32") -> Path:
    """Write the invented bundle to ``path`` with a manifest, as the notebook would; ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(path / "articles.jsonl", ARTICLES)
    write_jsonl(
        path / "passages.jsonl",
        [{"pmcid": p, "section": s, "text": t} for p, s, t, _ in PASSAGES],
    )
    vectors = np.array([v / np.linalg.norm(v) for *_, v in PASSAGES], dtype=passage_dtype)
    np.save(path / "passage_embeddings.npy", vectors, allow_pickle=False)
    rows = query_rows()
    (path / "query_vocabulary.json").write_text(
        json.dumps({"rows": [r for r, _ in rows]}), encoding="utf-8"
    )
    np.save(
        path / "query_embeddings.npy",
        np.array([v for _, v in rows], dtype=np.float32),
        allow_pickle=False,
    )
    write_manifest(
        path,
        {
            "created": "2026-10-14T00:00:00+00:00",
            "model": "invented",
            "screening": {"digest": SCREEN_DIGEST},
        },
    )
    return path
