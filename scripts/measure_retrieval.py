"""Measure evidence retrieval's cost on this machine (task 3a, EXP-011).

    python scripts/measure_retrieval.py --synthetic 20000          # a synthetic bundle, deleted after
    python scripts/measure_retrieval.py --bundle data/raw/retrieval_corpus   # the real one

Times what the laptop pays for retrieval: loading a bundle (every file's SHA-256, the JSON, the
vectors' norms), the retriever's start-up (tokenizing, BM25, the condition filters, the score
rows), one ``retrieve`` per candidate, and the stub pipeline with and without evidence for its top
three candidates. **No model and no network**: a synthetic bundle is random words (each passage
naming one condition) and random 768-dimensional vectors, of the size a real bundle would have, in
a temporary folder removed afterwards. The queries are random sets of 3–8 hand-authored concepts.
Its figures say what retrieval costs, never how good it is. A synthetic bundle of 50,000 passages
takes about 20 s.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import Assertion, Finding, PatientCase  # noqa: E402
from src.pipeline import DiagnosisPipeline  # noqa: E402
from src.rag.build import golden_checks  # noqa: E402
from src.rag.bundle import load_bundle, write_jsonl, write_manifest  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402
from src.rag.screening import SCREEN_DIGEST  # noqa: E402
from src.rag.vocabulary import CONDITION_TERMS, query_vocabulary  # noqa: E402
from src.stubs import ConstantRanker, InMemoryGraphStore, TemplateExplainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DIMENSION = 768
WORDS_PER_PASSAGE = 380  # about what 510 WordPiece tokens of an article hold
PASSAGES_PER_ARTICLE = 8
COMMON_WORDS = (
    "chest pain pleuritic fever breathlessness sudden onset radiation back jaw arm sweating "
    "nausea palpitations syncope pressure tearing burning exertion rest sitting forward"
)
"""Words a query shares with many passages, so BM25's postings are as long as a real corpus's."""


def write_synthetic(out: Path, passages: int, dtype: str = "float16", seed: int = 42) -> Path:
    """A bundle of ``passages`` random passages and random unit vectors, screened under today's
    rules as the notebook's would be; ``out``."""
    rng = random.Random(seed)
    words = [f"term{i}" for i in range(30_000)] + COMMON_WORDS.split() * 40
    names = [terms.mentions[0] for terms in CONDITION_TERMS.values()]
    n_articles = max(passages // PASSAGES_PER_ARTICLE, 1)
    articles = [
        {
            "pmcid": f"PMC9{i:08d}",
            "title": f"An invented article about {names[i % len(names)]}",
            "licence": "CC BY 4.0",
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC9{i:08d}/",
        }
        for i in range(n_articles)
    ]
    rows = []
    for i in range(passages):
        article = min(i // PASSAGES_PER_ARTICLE, n_articles - 1)
        text = rng.choices(words, k=WORDS_PER_PASSAGE)
        text[rng.randrange(WORDS_PER_PASSAGE)] = names[article % len(names)]
        sentences = [" ".join(text[j : j + 20]).capitalize() + "." for j in range(0, len(text), 20)]
        rows.append(
            {"pmcid": articles[article]["pmcid"], "section": "Body", "text": " ".join(sentences)}
        )
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "articles.jsonl", articles)
    write_jsonl(out / "passages.jsonl", rows)
    gen = np.random.default_rng(seed)
    vectors = gen.standard_normal((passages, DIMENSION)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(out / "passage_embeddings.npy", vectors.astype(dtype), allow_pickle=False)
    vocabulary = query_vocabulary(
        golden_checks(REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml")
    )
    queries = gen.standard_normal((len(vocabulary), DIMENSION)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    np.save(out / "query_embeddings.npy", queries, allow_pickle=False)
    (out / "query_vocabulary.json").write_text(json.dumps({"rows": vocabulary}), encoding="utf-8")
    write_manifest(out, {"created": "synthetic", "screening": {"digest": SCREEN_DIGEST}})
    return out


def _percentiles(times: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "max_ms": round(float(np.max(times)), 3),
    }


def measure(bundle_dir: Path, cases: int = 20, seed: int = 42) -> dict[str, Any]:
    """The timings of one bundle, as a dict (the module docstring)."""
    started = time.perf_counter()
    bundle = load_bundle(bundle_dir)
    loaded = time.perf_counter() - started
    started = time.perf_counter()
    retriever = HybridRetriever(bundle)
    built = time.perf_counter() - started

    rng = random.Random(seed)
    concepts = [r["id"] for r in bundle.query_rows if r["kind"] == "concept"]
    patients = [
        PatientCase(
            case_id=f"timing-{i}",
            age=50,
            sex="M",
            findings=[
                Finding(concept_id=c, label=c, assertion=Assertion.PRESENT)
                for c in rng.sample(concepts, k=min(rng.randint(3, 8), len(concepts)))
            ],
        )
        for i in range(cases)
    ]
    one = []
    for patient in patients:
        for cid in CONDITION_TERMS:
            started = time.perf_counter()
            retriever.retrieve(cid, patient, k=3)
            one.append((time.perf_counter() - started) * 1000)
    pipelines = {
        "with_evidence": DiagnosisPipeline(
            ConstantRanker(), InMemoryGraphStore(), retriever, TemplateExplainer(), top_k_evidence=3
        ),
        "without": DiagnosisPipeline(
            ConstantRanker(), InMemoryGraphStore(), None, TemplateExplainer()
        ),
    }
    runs: dict[str, list[float]] = {name: [] for name in pipelines}
    for patient in patients * 3:
        for name, pipeline in pipelines.items():
            started = time.perf_counter()
            pipeline.run(patient)
            runs[name].append((time.perf_counter() - started) * 1000)
    return {
        "passages": len(bundle.passages),
        "articles": len(bundle.articles),
        "vector_dtype": str(bundle.passage_vectors.dtype),
        "load_s": round(loaded, 3),
        "startup_s": round(built, 3),
        "rescreened": retriever.rescreened,
        "score_rows_mb": round(sum(r.nbytes for r in retriever._score_row.values()) / 1e6, 1),
        "vectors_mb": round(bundle.passage_vectors.nbytes / 1e6, 1),
        "retrieve_one_candidate": _percentiles(one),
        "pipeline": {name: _percentiles(times) for name, times in runs.items()},
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--bundle", type=Path, help="an unzipped bundle")
    which.add_argument("--synthetic", type=int, help="passages in a synthetic bundle")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--out", type=Path, help="also write the timings here, as JSON")
    args = parser.parse_args(argv)

    if args.bundle:
        report = measure(args.bundle)
    else:
        with tempfile.TemporaryDirectory() as folder:
            report = measure(write_synthetic(Path(folder) / "bundle", args.synthetic, args.dtype))
    report["source"] = str(args.bundle) if args.bundle else f"synthetic ({args.dtype})"
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
