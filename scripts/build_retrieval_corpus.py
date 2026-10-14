"""Build the retrieval bundle: PMC Open Access passages and their embeddings (task 3a, EXP-011).

    python scripts/build_retrieval_corpus.py --device cuda --out /kaggle/working/retrieval_corpus
    python scripts/build_retrieval_corpus.py --device cuda --smoke --out /tmp/smoke_bundle

**A cloud job** (D-7: Kaggle, 2026-10-13), run by ``notebooks/kaggle_retrieval_corpus.ipynb``
(docs/11 §4.6), **never on the laptop**: it downloads the embedding model and several thousand
articles. It needs ``sentence-transformers`` (preinstalled on Kaggle; the notebook installs it if
not), which nothing on the laptop imports.

What it does is :func:`src.rag.build.build_bundle`:
- PMC searches per condition, restricted to the Open Access subset;
- EFetch with no API key, at most 3 requests a second (src/rag/eutils.py);
- articles kept only with a Creative Commons licence in their XML, recorded with it;
- no sentence or title with treatment language (src/rag/screening.py);
- passages of at most ``retrieval.chunk_size`` model tokens;
- the passages and the fixed query vocabulary embedded with ``retrieval.embedding_model``,
  normalised.

The bundle's manifest records the commit, the model and the exact revision downloaded, the
versions, the chunking, the screen's digest, the counts and every file's SHA-256. ``--email`` is
the only personal datum it can send, and only if given: NCBI asks tools for a contact address.
DDXPlus is never read.
"""

from __future__ import annotations

import argparse
import contextlib
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import load_config  # noqa: E402

from src.rag.build import build_bundle, golden_checks  # noqa: E402
from src.rag.eutils import EUtils  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONDITIONS = ("COND:pericarditis", "COND:pulmonary_embolism")
SMOKE_PER_CONDITION = 3


def _commit() -> str:
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    return "unknown"


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "torch", "transformers", "sentence_transformers", "huggingface_hub"):
        with contextlib.suppress(ImportError):
            versions[name] = __import__(name).__version__
    return versions


def load_model(name: str, revision: str | None, device: str) -> tuple[Any, str]:
    """The sentence-transformers model at an exact snapshot, and that snapshot's commit id."""
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    path = snapshot_download(name, revision=revision)
    model = SentenceTransformer(path, device=device)
    if not model.tokenizer.is_fast:
        raise RuntimeError("the chunker needs a fast tokenizer's offsets")
    if device == "cuda":
        model.half()
    return model, Path(path).name


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the bundle's folder")
    parser.add_argument("--cache", type=Path, help="the fetch cache (default: next to --out)")
    parser.add_argument("--config", type=Path, help="default: configs/config.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="default: compute.device")
    parser.add_argument("--per-condition", type=int, help="default: articles_per_condition")
    parser.add_argument("--revision", help="the model's Hugging Face revision (default: latest)")
    parser.add_argument("--email", default="", help="optional contact address NCBI asks tools for")
    parser.add_argument("--smoke", action="store_true", help="2 conditions x 3 articles, fast")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    retrieval = config.get("retrieval") or {}
    device = args.device or (config.get("compute") or {}).get("device", "cpu")
    per_condition = args.per_condition or int(retrieval.get("articles_per_condition", 150))
    conditions = None
    if args.smoke:
        per_condition, conditions = SMOKE_PER_CONDITION, SMOKE_CONDITIONS
    cache = args.cache or args.out.parent / (args.out.name + "_cache") / "fetched.jsonl"

    started = time.time()
    name = retrieval.get("embedding_model", "pritamdeka/S-PubMedBert-MS-MARCO")
    model, revision = load_model(name, args.revision, device)
    max_tokens = min(int(retrieval.get("chunk_size", 512)), int(model.max_seq_length))
    model.max_seq_length = max_tokens
    window = max_tokens - 2  # the model adds [CLS] and [SEP]
    overlap = min(int(retrieval.get("chunk_overlap", 64)), window - 1)
    print(f"model {name} @ {revision} on {device}; passages of {window} tokens, {overlap} shared")

    def offsets_of(text: str) -> list[tuple[int, int]]:
        encoded = model.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return [tuple(pair) for pair in encoded["offset_mapping"]]

    def embed(texts: list[str]) -> Any:
        return model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    manifest = build_bundle(
        args.out,
        eutils=EUtils(email=args.email or None),
        embed=embed,
        offsets_of=offsets_of,
        window=window,
        overlap=overlap,
        per_condition=per_condition,
        checks=golden_checks(REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"),
        conditions=conditions,
        cache=cache,
        info={
            "commit": _commit(),
            "model": name,
            "model_revision": revision,
            "model_dtype": "float16" if device == "cuda" else "float32",
            "max_seq_length": max_tokens,
            "device": device,
            "smoke": args.smoke,
            "versions": _versions(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        },
    )
    print(f"done in {(time.time() - started) / 60:.1f} min: {manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
