"""The retrieval bundle: what the Kaggle notebook writes and the laptop reads (task 3a).

The notebook (``notebooks/kaggle_retrieval_corpus.ipynb``) fetches the corpus, embeds it and
writes one folder, zipped; the owner unzips it into ``data/raw/retrieval_corpus/`` (gitignored:
the corpus and its embeddings are never committed, docs/03 §5). Its files:

==========================  ======================================================================
``manifest.json``           What made the bundle (commit, model id and revision, versions, chunking,
                            the date), its counts, and every other file's SHA-256 and size.
``articles.jsonl``          One article per line: PMCID, PMID, DOI, title, journal, year, first
                            author, licence and its URL, article type, PMC link, and which
                            conditions' searches found it (``conditions``, ``focus``).
``passages.jsonl``          One passage per line, ``{"pmcid", "section", "text"}``; line i is row i
                            of the embeddings.
``passage_embeddings.npy``  float16 or float32, (passages, dimension), each row L2-normalised.
``query_vocabulary.json``   ``{"rows": [{"kind", "id", "text"}, …]}`` (src/rag/vocabulary.py); row
                            i is row i of the query embeddings.
``query_embeddings.npy``    The query texts' vectors, normalised like the passages'.
``search.json``             Every PMC search: the term, NCBI's translation of it, its hit count and
                            the ids it returned, with the date (a record; the laptop only checks
                            its hash).
``skipped.jsonl``           Articles fetched and dropped, with the reason (a record).
==========================  ======================================================================

**The laptop refuses a bundle whose files do not match the manifest** (:func:`load_bundle`
raises :class:`BundleError`, and ``open_retriever`` then degrades to no evidence, reporting
``rag``): a file changed or truncated after the notebook wrote it would otherwise show passages
under citations they did not come from. Arrays are read with ``allow_pickle=False``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "REQUIRED_FILES",
    "Bundle",
    "BundleError",
    "load_bundle",
    "read_jsonl",
    "sha256_of",
    "write_jsonl",
    "write_manifest",
]

FORMAT = "nightingale-retrieval-bundle"
FORMAT_VERSION = 1
REQUIRED_FILES = (
    "articles.jsonl",
    "passages.jsonl",
    "passage_embeddings.npy",
    "query_vocabulary.json",
    "query_embeddings.npy",
)


class BundleError(ValueError):
    """The bundle is missing, incomplete, or not what its manifest says."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write one JSON object per line (UTF-8, keys in the order given); the number written."""
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_manifest(bundle_dir: Path | str, info: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``manifest.json``: ``info``, plus the SHA-256 and size of every other file in the
    folder. The notebook calls it last, and so do the tests' fixtures: one writer, one reader.

    Raises:
        BundleError: when a required file is missing.
    """
    bundle_dir = Path(bundle_dir)
    missing = [name for name in REQUIRED_FILES if not (bundle_dir / name).is_file()]
    if missing:
        raise BundleError(f"cannot write a manifest without {', '.join(missing)}")
    files = {
        path.name: {"sha256": sha256_of(path), "bytes": path.stat().st_size}
        for path in sorted(bundle_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, **dict(info), "files": files}
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass
class Bundle:
    """A loaded bundle; the embeddings are float32 and normalised."""

    path: Path
    manifest: dict[str, Any]
    articles: dict[str, dict[str, Any]]
    passages: list[dict[str, Any]]
    passage_vectors: np.ndarray
    query_rows: list[dict[str, str]]
    query_vectors: np.ndarray


def _normalised(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2 or array.shape[1] == 0:
        raise BundleError(f"{name} should be a 2-D array, not of shape {array.shape}")
    vectors = array.astype(np.float32)
    if not np.all(np.isfinite(vectors)):
        raise BundleError(f"{name} holds values that are not finite")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise BundleError(f"{name} holds a zero vector")
    return vectors / norms


def load_bundle(bundle_dir: Path | str) -> Bundle:
    """The bundle at ``bundle_dir``, every file checked against the manifest first.

    Raises:
        BundleError: for a missing folder or manifest, an unknown format, a file the manifest
            lists that is missing or whose SHA-256 differs, a required file the manifest does not
            list, or contents that do not fit together (row counts, dimensions, unknown articles).
    """
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"no retrieval bundle at {bundle_dir} (no manifest.json)")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BundleError(f"{manifest_path} is unreadable: {error}") from error
    if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
        raise BundleError(
            f"{manifest_path} is not a {FORMAT} v{FORMAT_VERSION} manifest "
            f"({manifest.get('format')!r} v{manifest.get('format_version')!r})"
        )
    files = manifest.get("files") or {}
    unlisted = [name for name in REQUIRED_FILES if name not in files]
    if unlisted:
        raise BundleError(f"the manifest does not list {', '.join(unlisted)}")
    for name, expected in files.items():
        path = bundle_dir / name
        if Path(name).name != name:
            raise BundleError(f"the manifest names a file outside the bundle: {name!r}")
        if not path.is_file():
            raise BundleError(f"{name} is missing from {bundle_dir}")
        actual = sha256_of(path)
        if actual != expected.get("sha256"):
            raise BundleError(
                f"{name} does not match the manifest (sha256 {actual[:12]}…, expected "
                f"{str(expected.get('sha256'))[:12]}…): the bundle was changed after it was written"
            )

    articles = {a["pmcid"]: a for a in read_jsonl(bundle_dir / "articles.jsonl")}
    passages = read_jsonl(bundle_dir / "passages.jsonl")
    vocabulary = json.loads((bundle_dir / "query_vocabulary.json").read_text(encoding="utf-8"))
    query_rows = vocabulary.get("rows") or []
    passage_vectors = _normalised(
        np.load(bundle_dir / "passage_embeddings.npy", allow_pickle=False), "passage_embeddings"
    )
    query_vectors = _normalised(
        np.load(bundle_dir / "query_embeddings.npy", allow_pickle=False), "query_embeddings"
    )
    if len(passages) != passage_vectors.shape[0]:
        raise BundleError(
            f"{len(passages)} passages but {passage_vectors.shape[0]} passage embeddings"
        )
    if len(query_rows) != query_vectors.shape[0]:
        raise BundleError(f"{len(query_rows)} query texts but {query_vectors.shape[0]} vectors")
    if passage_vectors.shape[1] != query_vectors.shape[1]:
        raise BundleError(
            f"passages are {passage_vectors.shape[1]}-dimensional, queries "
            f"{query_vectors.shape[1]}-dimensional: two models?"
        )
    orphans = {p["pmcid"] for p in passages} - set(articles)
    if orphans:
        raise BundleError(f"{len(orphans)} passages cite an article not in articles.jsonl")
    return Bundle(
        bundle_dir, manifest, articles, passages, passage_vectors, query_rows, query_vectors
    )
