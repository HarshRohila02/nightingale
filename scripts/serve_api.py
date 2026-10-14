"""Serve the Nightingale API on this machine (task 1e).

    python scripts/serve_api.py                   # http://127.0.0.1:8000/docs
    python scripts/serve_api.py --graph networkx  # the local graph, not Aura
    python scripts/serve_api.py --port 8001

Builds the pipeline from configs/config.yaml, exactly as scripts/demo.py does (the configured
ranker and graph, the template explainer, the configured fusion weights), and serves it with
uvicorn on 127.0.0.1 only: a prototype, never exposed beyond this machine. **Decision support
only, never a diagnosis.**
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import GRAPH_BACKENDS, build_pipeline, load_config  # noqa: E402

from src.api import create_app  # noqa: E402
from src.medical_kg.neo4j_store import GraphUnavailable  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="default: configs/config.yaml")
    parser.add_argument("--graph", choices=GRAPH_BACKENDS, help="overrides graph.backend")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    try:
        built = build_pipeline(load_config(args.config), args.graph)
    except GraphUnavailable as exc:
        print(f"No knowledge graph: {exc}. Run with --graph stub to try the API.", file=sys.stderr)
        return 1
    print(built["about"]["ranker"])
    print(built["about"]["graph"])
    print(built["about"]["retriever"])
    app = create_app(built["pipeline"], labels=built["labels"], about=built["about"])
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
