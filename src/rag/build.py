"""Building the retrieval bundle on Kaggle (task 3a): search, fetch, screen, chunk, embed, write.

``scripts/build_retrieval_corpus.py`` runs :func:`build_bundle` inside
``notebooks/kaggle_retrieval_corpus.ipynb``, with the real E-utilities client and the embedding
model. Everything here takes them as arguments, so tests/test_retrieval.py runs the whole build
offline, with a fake NCBI and a fake model, before the owner spends a Kaggle run on it.

The steps, in order (docs/11 §4.6):

1. **Search** (:func:`plan_searches`): for each condition, PMC ESearch over
   :func:`~src.rag.vocabulary.esearch_terms`, diagnosis-focused first, always restricted to the
   Open Access subset (``"open access"[filter]``), until ``per_condition × fetch_factor`` ids are
   planned. Every search is recorded (``search.json``).
2. **Fetch** (:func:`fetch_articles`): EFetch in batches of :data:`BATCH`, parsed by
   :func:`~src.rag.corpus.parse_articles`, which keeps an article only with a Creative Commons
   licence in its XML and records why each other one was dropped. Each fetched id is written to a
   cache as it arrives, so a stopped run resumes where it stopped.
3. **Select** (:func:`select_articles`): per condition, the first ``per_condition`` kept articles
   in search order, **without one whose title names a treatment** (src/rag/screening.py).
4. **Chunk** (:func:`~src.rag.corpus.article_passages`): only the runs of sentences free of
   treatment language, cut into the model's token windows. An article left with no passage is
   dropped.
5. **Embed** the passages and the fixed query vocabulary
   (:func:`~src.rag.vocabulary.query_vocabulary`, with check rows for the golden cases), and
   **write** the bundle with its manifest (src/rag/bundle.py).

No DDXPlus file is read anywhere: retrieval needs only the condition registry and the crosswalk's
concept labels, which are code.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.rag.bundle import read_jsonl, write_jsonl, write_manifest
from src.rag.corpus import Article, article_passages, normalise_pmcid, parse_articles
from src.rag.eutils import EUtils, EUtilsError
from src.rag.screening import MIN_SEGMENT_CHARS, SCREEN_DIGEST, title_names_treatment
from src.rag.vocabulary import CONDITION_TERMS, esearch_terms, query_vocabulary

__all__ = [
    "BATCH",
    "FETCH_FACTOR",
    "build_bundle",
    "fetch_articles",
    "golden_checks",
    "plan_searches",
    "select_articles",
]

BATCH = 20
"""Articles per EFetch request: full-text XML is large, and a failed batch is refetched one by one."""
FETCH_FACTOR = 1.5
"""Ids planned per condition, as a multiple of the quota: some fetched articles are dropped (no
Creative Commons licence, a treatment in the title, no screened passage), and the quota is filled
from the rest in search order."""

Log = Callable[[str], None]
Embed = Callable[[list[str]], np.ndarray]
"""Texts → their vectors, (len(texts), dimension), each row L2-normalised."""
OffsetsOf = Callable[[str], Sequence[tuple[int, int]]]


def _today() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def plan_searches(
    eutils: EUtils,
    per_condition: int,
    *,
    conditions: Iterable[str] | None = None,
    fetch_factor: float = FETCH_FACTOR,
    log: Log = print,
) -> tuple[dict[str, list[tuple[str, str]]], list[dict[str, Any]]]:
    """Step 1: ``({condition: [(pmcid, focus), ...]}, [search record, ...])``.

    Each condition's list holds distinct PMCIDs in search order, at most
    ``ceil(per_condition × fetch_factor)``. ESearch is asked to sort by relevance; if NCBI refuses
    that for PMC, the search is repeated in its default order, and the record says which was used.
    """
    wanted = math.ceil(per_condition * fetch_factor)
    plan: dict[str, list[tuple[str, str]]] = {}
    searches: list[dict[str, Any]] = []
    for cid in conditions or CONDITION_TERMS:
        planned: list[tuple[str, str]] = []
        seen: set[str] = set()
        for focus, term in esearch_terms(cid):
            if len(planned) >= wanted:
                break
            sort: str | None = "relevance"
            try:
                result = eutils.esearch(term, retmax=wanted, sort=sort)
            except EUtilsError:
                sort = None
                result = eutils.esearch(term, retmax=wanted, sort=None)
            ids = [p for p in (normalise_pmcid(i) for i in result.get("idlist", [])) if p]
            searches.append(
                {
                    "condition": cid,
                    "focus": focus,
                    "term": term,
                    "sort": sort,
                    "count": int(result.get("count", 0) or 0),
                    "translation": result.get("querytranslation"),
                    "ids": ids,
                    "date": _today(),
                }
            )
            for pmcid in ids:
                if pmcid not in seen and len(planned) < wanted:
                    seen.add(pmcid)
                    planned.append((pmcid, focus))
        plan[cid] = planned
        log(f"{cid}: {len(planned)} articles planned")
    return plan, searches


def _article_from(record: Mapping[str, Any]) -> Article:
    data = dict(record)
    data["sections"] = [tuple(s) for s in data.get("sections", [])]
    return Article(**data)


def fetch_articles(
    eutils: EUtils,
    pmcids: Sequence[str],
    *,
    cache: Path | None = None,
    batch: int = BATCH,
    max_chars: int = 20_000,
    log: Log = print,
) -> tuple[dict[str, Article], list[dict[str, str]]]:
    """Step 2: ``({pmcid: Article}, [{"pmcid", "reason"}, ...])`` for ``pmcids``.

    ``cache`` (a JSONL file) receives one line per id once it is fetched, kept or dropped, and
    an id already in it is not fetched again: a stopped run resumes. A batch whose XML does not
    parse is fetched again one article at a time. An id EFetch does not return, or that fails
    after every retry, is recorded as dropped and not cached, so a later run tries it again.
    """
    done: dict[str, dict[str, Any]] = {}
    if cache is not None and cache.is_file():
        done = {row["pmcid"]: row for row in read_jsonl(cache)}
        log(f"resuming: {len(done)} of {len(pmcids)} articles already fetched")

    def remember(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            done[row["pmcid"]] = row
        if cache is not None and rows:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "a", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    failed: list[dict[str, str]] = []

    def fetch(ids: list[str]) -> None:
        try:
            body = eutils.efetch([i.removeprefix("PMC") for i in ids])
            kept, dropped = parse_articles(body, max_chars=max_chars)
        except ET.ParseError:
            if len(ids) == 1:
                remember([{"pmcid": ids[0], "kept": None, "reason": "unparseable XML"}])
                return
            for one in ids:
                fetch([one])
            return
        except EUtilsError as error:
            failed.extend({"pmcid": i, "reason": f"fetch failed: {error}"} for i in ids)
            return
        rows = [{"pmcid": a.pmcid, "kept": asdict(a), "reason": None} for a in kept]
        rows += [
            {"pmcid": d["pmcid"], "kept": None, "reason": d["reason"]}
            for d in dropped
            if d["pmcid"]
        ]
        returned = {row["pmcid"] for row in rows}
        rows += [
            {"pmcid": i, "kept": None, "reason": "not returned by EFetch"}
            for i in ids
            if i not in returned
        ]
        remember([row for row in rows if row["pmcid"] in ids])

    todo = [p for p in dict.fromkeys(pmcids) if p not in done]
    for start in range(0, len(todo), batch):
        fetch(todo[start : start + batch])
        if (start // batch) % 10 == 0:
            log(f"fetched {min(start + batch, len(todo))} of {len(todo)} new articles")

    articles = {p: _article_from(done[p]["kept"]) for p in pmcids if done.get(p, {}).get("kept")}
    skipped = [
        {"pmcid": p, "reason": done[p]["reason"]}
        for p in dict.fromkeys(pmcids)
        if p in done and not done[p].get("kept")
    ]
    return articles, skipped + failed


def select_articles(
    plan: Mapping[str, Sequence[tuple[str, str]]],
    articles: Mapping[str, Article],
    per_condition: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Step 3: ``({pmcid: {"conditions": [...], "focus": {condition: focus}}}, dropped)``.

    Per condition, the first ``per_condition`` fetched articles in search order whose title names
    no treatment; an article chosen for several conditions is kept once and lists them all.
    """
    chosen: dict[str, dict[str, Any]] = {}
    dropped: dict[str, dict[str, str]] = {}
    for cid, planned in plan.items():
        taken = 0
        for pmcid, focus in planned:
            if taken >= per_condition:
                break
            article = articles.get(pmcid)
            if article is None:
                continue
            if title_names_treatment(article.title):
                dropped[pmcid] = {"pmcid": pmcid, "reason": "the title names a treatment"}
                continue
            entry = chosen.setdefault(pmcid, {"conditions": [], "focus": {}})
            entry["conditions"].append(cid)
            entry["focus"][cid] = focus
            taken += 1
    return chosen, list(dropped.values())


def golden_checks(path: Path) -> dict[str, list[str]]:
    """{golden case id: its present hand-authored concepts}, for the vocabulary's check rows."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    checks = {}
    for golden in data["cases"]:
        case = golden["case"]
        entries = [*case.get("findings", []), *case.get("risk_factors", [])]
        checks[golden["id"]] = [
            e["concept_id"] for e in entries if e.get("assertion", "present") == "present"
        ]
    return checks


def build_bundle(
    out: Path,
    *,
    eutils: EUtils,
    embed: Embed,
    offsets_of: OffsetsOf,
    window: int,
    overlap: int,
    per_condition: int,
    checks: Mapping[str, Sequence[str]] | None = None,
    conditions: Iterable[str] | None = None,
    cache: Path | None = None,
    info: Mapping[str, Any] | None = None,
    passage_dtype: str = "float16",
    log: Log = print,
) -> dict[str, Any]:
    """Steps 1–5: the bundle written to ``out`` (src/rag/bundle.py); its manifest.

    Args:
        out: The bundle's folder; created, and every file in it overwritten.
        eutils, embed, offsets_of: The NCBI client, the embedding model and its tokenizer's
            offsets (the tests pass fakes).
        window, overlap: Tokens per passage and shared between passages (:func:`chunk_spans`).
        per_condition: Articles kept per condition.
        checks: {case id: present concepts}: check rows for EXP-011 (:func:`golden_checks`).
        conditions: A subset of the conditions, for the notebook's smoke test.
        cache: The fetch cache (:func:`fetch_articles`).
        info: Recorded in the manifest beside what this function records (the model, its
            revision, versions, the commit).
        passage_dtype: How the passage vectors are stored: float16 halves the bundle, and the
            laptop widens them to float32 before any product.
    """
    out.mkdir(parents=True, exist_ok=True)
    plan, searches = plan_searches(eutils, per_condition, conditions=conditions, log=log)
    wanted = [pmcid for planned in plan.values() for pmcid, _ in planned]
    fetched, skipped = fetch_articles(eutils, wanted, cache=cache, log=log)
    chosen, dropped = select_articles(plan, fetched, per_condition)
    skipped += dropped

    passages: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []
    for pmcid, entry in chosen.items():
        article = fetched[pmcid]
        pieces = article_passages(article, offsets_of, window, overlap)
        if not pieces:
            skipped.append({"pmcid": pmcid, "reason": "no passage free of treatment language"})
            continue
        passages += pieces
        metadata.append({**article.metadata(), **entry})
    if not passages:
        raise RuntimeError("no passage survived the fetch and the screen: nothing to embed")
    log(f"{len(metadata)} articles, {len(passages)} passages; embedding")

    vectors = np.asarray(embed([p["text"] for p in passages]), dtype=np.float32)
    rows = query_vocabulary(checks)
    query_vectors = np.asarray(embed([r["text"] for r in rows]), dtype=np.float32)
    if vectors.shape[0] != len(passages) or query_vectors.shape[0] != len(rows):
        raise RuntimeError("the model returned a different number of vectors than texts")

    write_jsonl(out / "articles.jsonl", metadata)
    write_jsonl(out / "passages.jsonl", passages)
    np.save(out / "passage_embeddings.npy", vectors.astype(passage_dtype), allow_pickle=False)
    (out / "query_vocabulary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    np.save(out / "query_embeddings.npy", query_vectors, allow_pickle=False)
    (out / "search.json").write_text(
        json.dumps(searches, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    write_jsonl(out / "skipped.jsonl", skipped)
    licences: dict[str, int] = {}
    for article in metadata:
        licences[article["licence"]] = licences.get(article["licence"], 0) + 1
    record = {
        "created": _today(),
        "source": "PMC Open Access subset, via NCBI E-utilities (no API key)",
        "per_condition": per_condition,
        "chunking": {"window_tokens": window, "overlap_tokens": overlap},
        "screening": {"digest": SCREEN_DIGEST, "min_segment_chars": MIN_SEGMENT_CHARS},
        "passage_dtype": passage_dtype,
        "counts": {
            "articles": len(metadata),
            "passages": len(passages),
            "skipped": len(skipped),
            "searches": len(searches),
            "query_rows": len(rows),
            "dimension": int(vectors.shape[1]),
        },
        "licences": dict(sorted(licences.items())),
        "eutils_requests": eutils.requests,
        **dict(info or {}),
    }
    manifest = write_manifest(out, record)
    log(f"bundle written to {out}: {json.dumps(record['counts'])}")
    return manifest
