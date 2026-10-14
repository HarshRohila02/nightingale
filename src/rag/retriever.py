"""Evidence retrieval on the laptop: BM25 and dense search, fused by reciprocal rank (task 3a).

Implements ``EvidenceRetriever`` (src/contracts.py): for a candidate condition and a case, the
passages of the PMC Open Access corpus that best bear on that condition in that patient, each
with its citation (``PMCID:…``) and a link. **Evidence is shown beside a ranking, never used to
make one**: the pipeline attaches it after ranking (docs/02 §3), and it is about diagnosis only.

**Why this design.** The corpus is fetched and embedded on Kaggle (D-7, 2026-10-13), and nothing
is installed on the laptop, so no embedding model runs here: no sentence-transformers, no torch,
no FAISS (D-4). Everything at query time is NumPy and the standard library:

1. **Which passages may be shown** under a condition: those whose text, or whose article's
   title, mentions it by one of its names (``mentions`` in src/rag/vocabulary.py). A passage that
   never names a condition cannot support a statement about it, and the filter keeps a passage
   about pulmonary embolism from appearing under pericarditis because both mention pleuritic
   pain. The price: a passage that discusses a condition only by a name not listed is missed.
2. **BM25** (src/rag/bm25.py) over those passages, for the condition's name and the labels of
   the case's present findings (``compose_query_text``).
3. **Dense search**: the cosine of each passage's precomputed embedding with a query vector built
   from **precomputed** vectors (the notebook embeds a fixed vocabulary): the condition's vector
   (the mean of its query texts', ``QUERY_TEMPLATES``) mixed with the mean of the case's findings'
   vectors, ``q = normalise(w · condition + (1 − w) · findings)``, w = 0.6, fixed a priori.
   A bi-encoder is not linear in its input, so this mix only **approximates** the embedding of the
   sentence "pericarditis with pleuritic pain and fever"; the notebook embeds such sentences for
   the golden cases, so EXP-011 can measure how close the approximation is. Only the hand-authored
   concepts the vocabulary embedded can steer it; a concept added since is ignored, and named in
   ``missing_concepts``.
4. **Reciprocal rank fusion** (Cormack, Clarke & Büttcher 2009), k = 60: each method ranks the
   eligible passages, keeping one passage per article (its best), down to ``depth``
   (``retrieval.top_k``, 10); a passage scores Σ 1 / (k + rank) over the lists it is in; one per
   article again, the best ``k`` returned. RRF needs no score calibration between BM25 and cosine,
   which have no common scale.

**What it does not do.** No cross-encoder rerank: a reranker reads the query and the passage
together, so it needs a model at query time, the one thing the laptop does not run, and a rerank
precomputed on Kaggle for per-condition queries would ignore the patient; ``retrieval.rerank_top_k``
is unused until a model can run where the pipeline runs. No FAISS: an exact dot product over a
few tens of thousands of passages takes milliseconds in NumPy, and an approximate index would only
lose recall. Denied findings and free text never enter a query: searching "no diaphoresis" finds
diaphoresis, and free text is untrusted (FR-8.5). Retrieval quality is **unmeasured** until
EXP-011; the system is closed-world (R-13), so evidence exists only for its 14 conditions.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import Assertion, Evidence, PatientCase
from src.rag.bm25 import BM25Index
from src.rag.bundle import Bundle, BundleError, load_bundle
from src.rag.text import excerpt, mentions_any, phrase_tokens, tokenize
from src.rag.vocabulary import CONDITION_TERMS, compose_query_text, resolve_condition
from src.stubs import EmptyRetriever

logger = logging.getLogger(__name__)

__all__ = [
    "CONDITION_WEIGHT",
    "DEFAULT_BUNDLE",
    "DEPTH",
    "RRF_K",
    "HybridRetriever",
    "UnavailableRetriever",
    "describe",
    "open_retriever",
    "present_concepts",
    "reciprocal_rank_fusion",
]

DEFAULT_BUNDLE = Path("data/raw/retrieval_corpus")
RRF_K = 60
"""Cormack et al.'s constant: it damps the difference between first and second place, so one
method's top hit cannot outvote agreement lower down."""
DEPTH = 10
"""How far down each method's list enters the fusion (``retrieval.top_k``)."""
CONDITION_WEIGHT = 0.6
"""The condition's share of the dense query vector; the findings' mean has the rest. More than a
half, so the candidate decides what the evidence is about and the findings which of it comes
first. Fixed before any bundle existed; EXP-011 may revisit it, on validation queries only."""


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """``[(item, score)]``, best first: score = Σ 1 / (k + rank), ranks counted from 1, over the
    lists the item appears in. A tie goes to the item ranked higher in an earlier list, then the
    smaller item, so the order is fully determined."""
    scores: dict[int, float] = {}
    first_seen: dict[int, tuple[int, int]] = {}
    for list_index, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(item, (list_index, rank))
    return sorted(scores.items(), key=lambda pair: (-pair[1], first_seen[pair[0]], pair[0]))


class UnavailableRetriever(EmptyRetriever):
    """No evidence, and says so: the pipeline reports ``rag`` in ``degraded_components`` for it
    (docs/02 §7), where a plain ``EmptyRetriever`` (the tests' stub) reports nothing."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    @property
    def degraded(self) -> bool:
        return True


class HybridRetriever:
    """BM25 and dense search over a loaded :class:`~src.rag.bundle.Bundle`, fused by RRF.

    Implements ``EvidenceRetriever``. Building it tokenizes every passage once (seconds for tens
    of thousands); every query after that is a few vector operations.
    """

    degraded = False

    def __init__(
        self,
        bundle: Bundle,
        *,
        depth: int = DEPTH,
        rrf_k: int = RRF_K,
        condition_weight: float = CONDITION_WEIGHT,
        excerpt_chars: int = 320,
    ) -> None:
        if depth < 1 or rrf_k < 0 or not 0.0 <= condition_weight <= 1.0:
            raise ValueError("need depth >= 1, rrf_k >= 0 and 0 <= condition_weight <= 1")
        self.bundle = bundle
        self.depth = depth
        self.rrf_k = rrf_k
        self.condition_weight = condition_weight
        self.excerpt_chars = excerpt_chars
        passages = bundle.passages
        self._article_of = [p["pmcid"] for p in passages]
        article_index = {pmcid: i for i, pmcid in enumerate(dict.fromkeys(self._article_of))}
        self._article_ids = np.array([article_index[a] for a in self._article_of], dtype=np.int64)
        titles = {pmcid: tokenize(a.get("title", "")) for pmcid, a in bundle.articles.items()}
        documents = [titles[p["pmcid"]] + tokenize(p["text"]) for p in passages]
        self.bm25 = BM25Index(documents)
        token_sets = [set(doc) for doc in documents]
        self._eligible = {}
        for cid, terms in CONDITION_TERMS.items():
            phrases = [phrase_tokens(m) for m in terms.mentions]
            self._eligible[cid] = np.array(
                [mentions_any(doc, phrases, seen) for doc, seen in zip(documents, token_sets)],
                dtype=bool,
            )
        self._condition_vectors: dict[str, np.ndarray] = {}
        self._concept_vectors: dict[str, np.ndarray] = {}
        by_condition: dict[str, list[int]] = {}
        for i, row in enumerate(bundle.query_rows):
            if row.get("kind") == "condition":
                by_condition.setdefault(row["id"], []).append(i)
            elif row.get("kind") == "concept":
                self._concept_vectors[row["id"]] = bundle.query_vectors[i]
        for cid, rows in by_condition.items():
            self._condition_vectors[cid] = _unit(bundle.query_vectors[rows].mean(axis=0))
        from src.medical_kg.crosswalk import BY_CONCEPT

        self.missing_concepts = sorted(set(BY_CONCEPT) - set(self._concept_vectors))
        if self.missing_concepts:
            logger.warning(
                "the retrieval bundle has no query vector for %d concept(s), which will not "
                "steer the dense search: %s",
                len(self.missing_concepts),
                ", ".join(self.missing_concepts),
            )

    # -- the EvidenceRetriever protocol ----------------------------------------------------- #

    def retrieve(self, condition: str, case: PatientCase, k: int = 5) -> list[Evidence]:
        """At most ``k`` passages for ``condition`` (an id, a label or an alias) in this case, one
        per article, best first; [] when no passage mentions the condition.

        Raises:
            ValueError: for a condition the registry does not know.
        """
        cid = resolve_condition(condition)
        concepts = present_concepts(case)
        ranked = self.rank(cid, concepts)[: max(k, 0)]
        terms = self.bm25.query_terms(compose_query_text(cid, concepts))
        return [self._evidence(i, score, terms) for i, score in ranked]

    # -- the parts, public for EXP-011 and the tests --------------------------------------- #

    def query_vector(self, condition_id: str, concepts: Iterable[str] = ()) -> np.ndarray | None:
        """The dense query: the condition's vector mixed with its findings' (module docstring),
        or None when the bundle has no vector for the condition."""
        base = self._condition_vectors.get(condition_id)
        if base is None:
            return None
        vectors = [self._concept_vectors[c] for c in concepts if c in self._concept_vectors]
        if not vectors:
            return base
        findings = _unit(np.mean(vectors, axis=0))
        return _unit(self.condition_weight * base + (1.0 - self.condition_weight) * findings)

    def rank(self, condition_id: str, concepts: Sequence[str] = ()) -> list[tuple[int, float]]:
        """``[(passage index, RRF score)]`` for the condition and findings, one per article."""
        eligible = self._eligible[condition_id]
        if not eligible.any():
            return []
        lists = []
        bm25 = self.bm25.scores(compose_query_text(condition_id, concepts))
        lists.append(self._top_per_article(bm25, eligible & (bm25 > 0)))
        query = self.query_vector(condition_id, concepts)
        if query is not None:
            lists.append(self._top_per_article(self.bundle.passage_vectors @ query, eligible))
        fused = reciprocal_rank_fusion(lists, self.rrf_k)
        seen: set[int] = set()
        out = []
        for index, score in fused:
            article = int(self._article_ids[index])
            if article not in seen:
                seen.add(article)
                out.append((index, score))
        return out

    def source_of(self, citation: str) -> dict[str, Any] | None:
        """The cited article's record (journal, year, first author, licence…), for showing the
        attribution its licence asks for beside an excerpt; None for a citation not in the
        bundle."""
        return self.bundle.articles.get(citation.removeprefix("PMCID:"))

    # -- internals ------------------------------------------------------------------------- #

    def _top_per_article(self, scores: np.ndarray, mask: np.ndarray) -> list[int]:
        """The ``depth`` best passages under ``mask``, the best one of each article only."""
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        order = candidates[np.lexsort((candidates, -scores[candidates]))]
        out, seen = [], set()
        for index in order:
            article = int(self._article_ids[index])
            if article in seen:
                continue
            seen.add(article)
            out.append(int(index))
            if len(out) == self.depth:
                break
        return out

    def _evidence(self, index: int, score: float, terms: Sequence[str]) -> Evidence:
        passage = self.bundle.passages[index]
        article = self.bundle.articles[passage["pmcid"]]
        return Evidence(
            source="PMC",
            title=article.get("title") or passage["pmcid"],
            passage=excerpt(passage["text"], terms, self.excerpt_chars),
            citation=f"PMCID:{passage['pmcid']}",
            url=article.get("url"),
            score=float(score),
        )


def present_concepts(case: PatientCase) -> list[str]:
    """The hand-authored concepts the case records as present, findings then risk factors, in
    order. The ``DDX:E_nn`` questions ``expand_case`` adds are left out: they name questions, not
    findings (docs/02 §5.2)."""
    return list(
        dict.fromkeys(
            f.concept_id
            for f in [*case.findings, *case.risk_factors]
            if f.assertion is Assertion.PRESENT and not f.concept_id.startswith("DDX:")
        )
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def open_retriever(
    bundle_dir: Path | str | None = DEFAULT_BUNDLE, **options: Any
) -> HybridRetriever | UnavailableRetriever:
    """The retriever over the bundle at ``bundle_dir``, degrading instead of failing (docs/02 §7).

    Args:
        bundle_dir: The unzipped bundle (``retrieval.bundle``); None switches retrieval off.
        options: For :class:`HybridRetriever` (``depth``, ``rrf_k``, ``condition_weight``).

    Returns:
        A :class:`HybridRetriever`, or an :class:`UnavailableRetriever` naming why not: no bundle
        (the ordinary case, since ``data/`` is gitignored: CI and a fresh clone), one that fails
        its manifest's checks, or one that cannot be read. **Never raises for a missing or bad
        bundle**: the prototype still ranks without evidence, and says ``rag``.
    """
    if bundle_dir is None:
        return UnavailableRetriever("retrieval is switched off in the configuration")
    try:
        bundle = load_bundle(bundle_dir)
        retriever = HybridRetriever(bundle, **options)
    except BundleError as error:
        logger.warning("retrieving no evidence: %s", error)
        return UnavailableRetriever(str(error))
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.exception("the retrieval bundle at %s cannot be loaded", bundle_dir)
        return UnavailableRetriever(f"the retrieval bundle at {bundle_dir} cannot be loaded: {error}")
    logger.info(
        "retrieval: %d passages from %d articles", len(bundle.passages), len(bundle.articles)
    )
    return retriever


def describe(retriever: Any) -> str:
    """One line naming what retrieves evidence, for a demo header or ``/health``."""
    if retriever is None:
        return "retrieval: none"
    if getattr(retriever, "degraded", False):
        return f"retrieval: none ({getattr(retriever, 'reason', 'degraded')})"
    bundle = getattr(retriever, "bundle", None)
    if bundle is None:
        return f"retrieval: {type(retriever).__name__}"
    fetched = bundle.manifest.get("fetched") or bundle.manifest.get("created", "?")
    return (
        f"retrieval: BM25 + dense (RRF) over {len(bundle.passages)} passages from "
        f"{len(bundle.articles)} PMC Open Access articles, fetched {str(fetched)[:10]}"
    )

