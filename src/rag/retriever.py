"""Evidence retrieval on the laptop: BM25 and dense search, fused by reciprocal rank (task 3a).

Implements ``EvidenceRetriever`` (src/contracts.py): for a candidate condition and a case, the
passages of the PMC Open Access corpus that best bear on that condition in that patient, each
with its citation (``PMCID:…``) and a link. **Evidence is shown beside a ranking, never used to
make one**: the pipeline attaches it after ranking (docs/02 §3), to the top candidates only, and it
is about **diagnosis only**: no passage or title with treatment language is ever shown (FR-6.5;
src/rag/screening.py, which the Kaggle notebook applies before embedding and this module applies
again on load).

**Why this design.** The corpus is fetched and embedded on Kaggle (D-7, 2026-10-13), and nothing
is installed on the laptop, so no embedding model runs here: no sentence-transformers, no torch,
no FAISS (D-4). Everything at query time is NumPy and the standard library:

1. **Which passages may be shown** under a condition: those whose text, or whose article's
   title, mentions it by one of its names (``mentions`` in src/rag/vocabulary.py), and that pass
   the treatment screen. A passage that never names a condition cannot support a statement about
   it, and the filter keeps a passage about pulmonary embolism from appearing under pericarditis
   because both mention pleuritic pain. The price: a passage that discusses a condition only by a
   name not listed is missed.
2. **BM25** (src/rag/bm25.py) over those passages (each indexed with its article's title), for
   the condition's name and the labels of the case's present findings (``compose_query_text``).
3. **Dense search**: the cosine of each passage's precomputed embedding with a query vector built
   from **precomputed** vectors (the notebook embeds a fixed vocabulary): the condition's vector
   (the mean of its query texts', ``QUERY_TEMPLATES``) mixed with the mean of the case's findings'
   vectors, ``q = normalise(w · condition + (1 − w) · findings)``, w = 0.6, fixed a priori.
   A bi-encoder is not linear in its input, so this mix only **approximates** the embedding of the
   sentence "pericarditis with pleuritic pain and fever"; the notebook embeds such sentences for
   the golden cases, so EXP-011 can measure how close the approximation is (:func:`mix_fidelity`).
   Only the hand-authored concepts the vocabulary embedded can steer it; a concept added since is
   ignored, and named in ``missing_concepts``. **Because q is linear in the precomputed vectors,
   so is every passage's cosine with it**: the passages' cosines with each condition and concept
   vector are computed once, on load, and a query only combines a few of those rows over the
   eligible passages. That is exact, not an approximation, and makes a query cost microseconds
   per passage instead of a pass over every embedding.
4. **Reciprocal rank fusion** (Cormack, Clarke & Büttcher 2009), k = 60: each method ranks the
   eligible passages, keeping one passage per article (its best), down to ``depth``
   (``retrieval.top_k``, 10); a passage scores Σ 1 / (k + rank) over the lists it is in; one per
   article again, the best ``k`` returned. RRF needs no score calibration between BM25 and cosine,
   which have no common scale.

**What it does not do.** No cross-encoder rerank: a reranker reads the query and the passage
together, so it needs a model at query time, the one thing the laptop does not run, and a rerank
precomputed on Kaggle for per-condition queries would ignore the patient; ``retrieval.rerank_top_k``
is unused until a model can run where the pipeline runs. No FAISS: the dense search is exact.
Denied findings and free text never enter a query: searching "no diaphoresis" finds diaphoresis,
and free text is untrusted (FR-8.5). Retrieval quality is **unmeasured** until EXP-011; the system
is closed-world (R-13), so evidence exists only for its 14 conditions.
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
from src.rag.screening import SCREEN_DIGEST, title_names_treatment, treatment_language
from src.rag.text import excerpt, mentions_any, phrase_tokens, tokenize
from src.rag.vocabulary import CONDITION_TERMS, compose_query_text, resolve_condition
from src.stubs import EmptyRetriever

logger = logging.getLogger(__name__)

__all__ = [
    "CONDITION_WEIGHT",
    "DEFAULT_BUNDLE",
    "DEPTH",
    "METHODS",
    "RRF_K",
    "HybridRetriever",
    "UnavailableRetriever",
    "describe",
    "mix_fidelity",
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
first. Fixed before any bundle existed; EXP-011 may revisit it, on its own queries only."""
METHODS = ("bm25", "dense")
"""The two searches the hybrid fuses; EXP-011 also runs each alone (``rank(..., methods=)``)."""


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

    Implements ``EvidenceRetriever``. Building it tokenizes every passage once and scores every
    passage against the condition and concept vectors once (seconds for tens of thousands of
    passages, EXP-011); every query after that is a few vector operations over the passages that
    name the condition.

    Args:
        rescreen: Whether to screen every passage's text again on load. None (the default) does
            so only when the manifest's ``screening.digest`` is not these rules'
            (:data:`~src.rag.screening.SCREEN_DIGEST`).

    Attributes:
        screened_out: How many passages the treatment screen hides (src/rag/screening.py): their
            article's title names a treatment, or, when re-screened, their own text has
            treatment language under the rules as they are now. 0 for a bundle built under the
            same rules.
        rescreened: Whether the passages' text was screened again on load.
        missing_concepts: Hand-authored concepts with no query vector in the bundle.
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
        rescreen: bool | None = None,
    ) -> None:
        if depth < 1 or rrf_k < 0 or not 0.0 <= condition_weight <= 1.0:
            raise ValueError("need depth >= 1, rrf_k >= 0 and 0 <= condition_weight <= 1")
        self.bundle = bundle
        self.depth = depth
        self.rrf_k = rrf_k
        self.condition_weight = condition_weight
        self.excerpt_chars = excerpt_chars
        passages = bundle.passages
        article_index = {pmcid: i for i, pmcid in enumerate(bundle.articles)}
        self._article_ids = np.array([article_index[p["pmcid"]] for p in passages], np.int64)

        title_tokens = {pmcid: tokenize(a.get("title", "")) for pmcid, a in bundle.articles.items()}
        text_tokens = [tokenize(p["text"]) for p in passages]
        self.bm25 = BM25Index(
            [title_tokens[p["pmcid"]] + t for p, t in zip(passages, text_tokens, strict=True)]
        )

        # The treatment screen (src/rag/screening.py, rule 4). Titles are always checked again:
        # there are few and they are short. Passages are checked again only when the bundle was
        # screened under other rules: the manifest's SHA-256s guarantee the text is exactly what
        # the notebook screened, and the screen costs about 0.3 µs a character (25 s for 20,000
        # passages, EXP-011). Every excerpt is checked once more before it is shown (_evidence).
        screened_under = (bundle.manifest.get("screening") or {}).get("digest")
        self.rescreened = rescreen if rescreen is not None else screened_under != SCREEN_DIGEST
        unsafe_titles = {
            pmcid for pmcid, a in bundle.articles.items() if title_names_treatment(a["title"])
        }
        shown = np.array(
            [
                p["pmcid"] not in unsafe_titles
                and not (self.rescreened and treatment_language(p["text"]))
                for p in passages
            ],
            dtype=bool,
        )
        self.screened_out = int((~shown).sum())
        if self.screened_out:
            logger.warning(
                "the treatment screen hides %d of %d passages (bundle screened under %s, these "
                "rules are %s)",
                self.screened_out,
                len(passages),
                screened_under or "unknown rules",
                SCREEN_DIGEST,
            )
        # Which passages may be shown under each condition: mentioned in the passage's text or in
        # its article's title (checked apart, so a phrase cannot straddle the two), and screened.
        self._eligible: dict[str, np.ndarray] = {}
        text_sets = [set(t) for t in text_tokens]
        for cid, terms in CONDITION_TERMS.items():
            phrases = [phrase_tokens(m) for m in terms.mentions]
            titled = {
                pmcid for pmcid, tokens in title_tokens.items() if mentions_any(tokens, phrases)
            }
            mask = np.array(
                [
                    p["pmcid"] in titled or mentions_any(tokens, phrases, seen)
                    for p, tokens, seen in zip(passages, text_tokens, text_sets, strict=True)
                ],
                dtype=bool,
            )
            self._eligible[cid] = np.flatnonzero(mask & shown)

        # The query vectors, and every passage's cosine with each (the module docstring, step 3).
        by_condition: dict[str, list[int]] = {}
        concept_rows: dict[str, int] = {}
        for i, row in enumerate(bundle.query_rows):
            if row.get("kind") == "condition":
                by_condition.setdefault(row["id"], []).append(i)
            elif row.get("kind") == "concept":
                concept_rows[row["id"]] = i
        self._condition_vectors = {
            cid: _unit(bundle.query_vectors[rows].mean(axis=0))
            for cid, rows in by_condition.items()
        }
        self._concept_vectors = {c: bundle.query_vectors[i] for c, i in concept_rows.items()}
        names = [*self._condition_vectors, *self._concept_vectors]
        vectors = [*self._condition_vectors.values(), *self._concept_vectors.values()]
        scores = bundle.passage_scores(np.array(vectors)) if vectors else np.zeros((0, 0))
        self._score_row = {name: scores[i] for i, name in enumerate(names)}

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
        per article, best first; [] when no screened passage mentions the condition.

        Raises:
            ValueError: for a condition the registry does not know.
        """
        cid = resolve_condition(condition)
        concepts = present_concepts(case)
        terms = self.bm25.query_terms(compose_query_text(cid, concepts))
        out = []
        for index, score in self.rank(cid, concepts):
            if len(out) >= max(k, 0):
                break
            evidence = self._evidence(index, score, terms)
            if evidence is not None:
                out.append(evidence)
        return out

    # -- the parts, public for EXP-011 and the tests --------------------------------------- #

    def query_vector(self, condition_id: str, concepts: Iterable[str] = ()) -> np.ndarray | None:
        """The dense query: the condition's vector mixed with its findings' (module docstring),
        or None when the bundle has no vector for the condition."""
        base = self._condition_vectors.get(condition_id)
        if base is None:
            return None
        used = dict.fromkeys(c for c in concepts if c in self._concept_vectors)
        vectors = [self._concept_vectors[c] for c in used]
        if not vectors:
            return base
        findings = _unit(np.mean(vectors, axis=0))
        return _unit(self.condition_weight * base + (1.0 - self.condition_weight) * findings)

    def eligible(self, condition_id: str) -> np.ndarray:
        """The indices of the passages that may be shown under the condition (step 1)."""
        return self._eligible[condition_id]

    def dense_scores(
        self, condition_id: str, concepts: Sequence[str], indices: np.ndarray
    ) -> np.ndarray | None:
        """The cosine of passages ``indices`` with :meth:`query_vector`, from the precomputed
        rows; None without a condition vector. Equal to the passages' embeddings times the query
        vector, up to float32 rounding (tests/test_retrieval.py checks)."""
        base = self._condition_vectors.get(condition_id)
        if base is None:
            return None
        cosine = self._score_row[condition_id][indices].astype(np.float64)
        used = list(dict.fromkeys(c for c in concepts if c in self._concept_vectors))
        if not used:
            return cosine
        w = self.condition_weight
        mean = np.mean([self._concept_vectors[c] for c in used], axis=0)
        mean_norm = float(np.linalg.norm(mean))
        if mean_norm == 0:
            return cosine
        mixed_norm = float(np.linalg.norm(w * base + (1.0 - w) * mean / mean_norm))
        findings = np.mean([self._score_row[c][indices] for c in used], axis=0) / mean_norm
        return (w * cosine + (1.0 - w) * findings) / mixed_norm

    def rank(
        self, condition_id: str, concepts: Sequence[str] = (), methods: Sequence[str] = METHODS
    ) -> list[tuple[int, float]]:
        """``[(passage index, RRF score)]`` for the condition and findings, one per article, best
        first. ``methods`` picks the searches fused: both (the hybrid), or one alone (EXP-011)."""
        unknown = set(methods) - set(METHODS)
        if unknown or not methods:
            raise ValueError(f"methods must be some of {METHODS}, not {sorted(unknown)}")
        concepts = list(dict.fromkeys(concepts))
        eligible = self._eligible[condition_id]
        if eligible.size == 0:
            return []
        lists = []
        if "bm25" in methods:
            bm25 = self.bm25.scores(compose_query_text(condition_id, concepts))[eligible]
            lists.append(self._top_per_article(eligible[bm25 > 0], bm25[bm25 > 0]))
        if "dense" in methods:
            dense = self.dense_scores(condition_id, concepts, eligible)
            if dense is not None:
                lists.append(self._top_per_article(eligible, dense))
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
        bundle. ``Evidence`` has no licence field (src/contracts.py), so the dashboard asks here."""
        return self.bundle.articles.get(citation.removeprefix("PMCID:"))

    # -- internals ------------------------------------------------------------------------- #

    def _top_per_article(self, indices: np.ndarray, scores: np.ndarray) -> list[int]:
        """The ``depth`` best of ``indices`` by ``scores`` (aligned), one per article; a tie goes
        to the earlier passage."""
        if indices.size == 0:
            return []
        order = np.lexsort((indices, -scores))
        out, seen = [], set()
        for position in order:
            index = int(indices[position])
            article = int(self._article_ids[index])
            if article in seen:
                continue
            seen.add(article)
            out.append(index)
            if len(out) == self.depth:
                break
        return out

    def _evidence(self, index: int, score: float, terms: Sequence[str]) -> Evidence | None:
        """The passage as ``Evidence``, or None if its excerpt or title fails the treatment
        screen: a last check on exactly what would be shown."""
        passage = self.bundle.passages[index]
        article = self.bundle.articles[passage["pmcid"]]
        title = article.get("title") or passage["pmcid"]
        text = excerpt(passage["text"], terms, self.excerpt_chars)
        if not text or treatment_language(text) or title_names_treatment(title):
            return None
        return Evidence(
            source="PMC",
            title=title,
            passage=text,
            citation=f"PMCID:{passage['pmcid']}",
            url=article.get("url"),
            score=float(score),
        )


def present_concepts(case: PatientCase) -> list[str]:
    """The hand-authored concepts the case records as present, findings then risk factors, in
    order. The ``DDX:E_nn`` questions ``expand_case`` adds are left out: they name questions, not
    findings (docs/02 §5.2). Denied and unknown findings never enter a query."""
    return list(
        dict.fromkeys(
            f.concept_id
            for f in [*case.findings, *case.risk_factors]
            if f.assertion is Assertion.PRESENT and not f.concept_id.startswith("DDX:")
        )
    )


def mix_fidelity(retriever: HybridRetriever, depth: int | None = None) -> list[dict[str, Any]]:
    """For each ``check`` row of the bundle (a golden case and a condition): how close the
    laptop's mixed query vector comes to the embedding of the full query sentence (EXP-011).

    Returns one record per check row: ``case``, ``condition``, ``cosine`` between the two query
    vectors, and ``overlap``, the share of the dense search's top ``depth`` passages (one per
    article, eligible passages only) that the two queries have in common, None when no passage
    may be shown under the condition. Empty for a bundle without check rows.
    """
    depth = depth or retriever.depth
    bundle = retriever.bundle
    out = []
    for i, row in enumerate(bundle.query_rows):
        if row.get("kind") != "check":
            continue
        cid = row.get("condition") or row["id"].split("|", 1)[-1]
        concepts = list(row.get("concepts") or [])
        mixed = retriever.query_vector(cid, concepts)
        if mixed is None:
            continue
        real = bundle.query_vectors[i]
        eligible = retriever.eligible(cid)
        mixed_top = retriever._top_per_article(
            eligible, retriever.dense_scores(cid, concepts, eligible)
        )[:depth]
        real_top = retriever._top_per_article(
            eligible, bundle.passage_scores(real[None, :])[0][eligible]
        )[:depth]
        shared = len(set(mixed_top) & set(real_top))
        out.append(
            {
                "case": row["id"].split("|", 1)[0],
                "condition": cid,
                "cosine": float(np.dot(mixed, real)),
                # None when no passage may be shown under the condition: nothing to compare.
                "overlap": shared / len(real_top) if real_top else None,
            }
        )
    return out


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
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        logger.exception("the retrieval bundle at %s cannot be loaded", bundle_dir)
        return UnavailableRetriever(
            f"the retrieval bundle at {bundle_dir} cannot be loaded: {error}"
        )
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
    created = bundle.manifest.get("created") or bundle.manifest.get("fetched") or "?"
    return (
        f"retrieval: BM25 + dense (RRF) over {len(bundle.passages)} passages from "
        f"{len(bundle.articles)} PMC Open Access articles, built {str(created)[:10]}"
    )
