"""Reading configs/config.yaml, for the scripts.

Nothing in ``src/`` reads YAML: every component takes its settings as arguments, so it can be
built three ways — by a script, by a test, by the app — without a file existing. The scripts are
where a configuration file turns into arguments, and this is the shared part of that.

Paths in the file are relative to the repository, so they are resolved here rather than against
whatever directory the script was run from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"

__all__ = [
    "DEFAULT_CONFIG",
    "GRAPH_BACKENDS",
    "REPO_ROOT",
    "describe_graph",
    "load_config",
    "open_graph_from_config",
    "open_ranker_from_config",
]

GRAPH_BACKENDS = ("neo4j", "networkx", "stub")
"""``graph.backend`` values the scripts accept: ``stub`` is the toy graph in src/stubs.py."""


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """The configuration, or an empty one if the file is missing.

    A missing file is not an error: every caller has defaults, and a prototype has to run on a
    machine that has not been set up yet.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def open_ranker_from_config(config: dict[str, Any]) -> Any:
    """The ranker the ``ml:`` block asks for (src/ml/ranker.py).

    Returns a degraded ranker, never an exception, when the model or the release files are
    absent — which is the ordinary case on a fresh clone, since both are gitignored.
    """
    from src.ml.ranker import DEFAULT_BACKEND, DEFAULT_MODEL_DIR, open_ranker

    ml = config.get("ml") or {}
    paths = config.get("paths") or {}
    raw = REPO_ROOT / paths.get("raw", "data/raw")
    interim = REPO_ROOT / paths.get("interim", "data/interim")
    return open_ranker(
        ml.get("backend", DEFAULT_BACKEND),
        model_dir=REPO_ROOT / ml.get("model_dir", DEFAULT_MODEL_DIR),
        evidences_path=raw / "ddxplus" / "release_evidences.json",
        conditions_path=interim / "ddxplus_chestpain_conditions.json",
        stem=ml.get("stem"),
        include_narrower=bool(ml.get("include_narrower", True)),
    )


def open_graph_from_config(config: dict[str, Any], backend: str | None = None) -> Any:
    """The graph store the ``graph:`` block asks for (src/medical_kg/neo4j_store.py).

    The graph is built from the local files when they are here: DDXPlus, the hand-authored facts
    and, when ``data/raw/bodhi_s`` holds it, BODHI-S. ``neo4j`` serves the copy in Aura and falls
    back to the local build, reporting ``graph_backend``, when Aura is paused, unconfigured or
    out of date. Read-only: a script never writes Aura (``scripts/load_neo4j.py`` does).

    Args:
        config: From :func:`load_config`.
        backend: Overrides ``graph.backend``: ``neo4j``, ``networkx`` or ``stub``. ``stub`` is the
            hand-written toy graph the tests use, not medical knowledge.

    Raises:
        GraphUnavailable: there is no graph: Neo4j failed, or was not asked for, and the local
            files are missing (they come from ``scripts/decode_ddxplus.py``).
        ValueError: for an unknown backend.
    """
    graph = config.get("graph") or {}
    backend = backend or graph.get("backend", "neo4j")
    if backend == "stub":
        from src.stubs import InMemoryGraphStore

        return InMemoryGraphStore()
    if backend not in GRAPH_BACKENDS:
        raise ValueError(f"unknown graph backend {backend!r}; expected one of {GRAPH_BACKENDS}")

    from src.medical_kg.cardiac_kg import build_cardiac_kg
    from src.medical_kg.neo4j_store import (
        DEFAULT_NAMESPACE,
        DEFAULT_TIMEOUT_SECONDS,
        open_graph_store,
    )

    paths = config.get("paths") or {}
    interim = REPO_ROOT / paths.get("interim", "data/interim")
    bodhi = REPO_ROOT / paths.get("raw", "data/raw") / "bodhi_s"
    files = (interim / "ddxplus_chestpain_conditions.json", interim / "ddxplus_evidences.json")
    local_kg = (
        build_cardiac_kg(*files, bodhi if (bodhi / "triples.jsonl").exists() else None)
        if all(f.exists() for f in files)
        else None
    )
    return open_graph_store(
        backend,
        local_kg=local_kg,
        namespace=graph.get("namespace", DEFAULT_NAMESPACE),
        timeout=float(graph.get("connection_timeout_seconds", DEFAULT_TIMEOUT_SECONDS)),
        sync=False,
    )


def describe_graph(store: Any) -> str:
    """One line saying which graph is answering, for a script to print beside the ranker's."""
    backend = getattr(store, "backend", None)
    if backend is None:
        return "Graph: the stub (a toy graph for tests, not medical knowledge)"
    sources: dict[str, int] = {}
    for _, _, source in store.graph.edges(data="source"):
        sources[source] = sources.get(source, 0) + 1
    by_source = ", ".join(f"{name} {n}" for name, n in sorted(sources.items()))
    line = (
        f"Graph: {backend}, {store.graph.number_of_nodes()} nodes, "
        f"{store.graph.number_of_edges()} edges ({by_source})"
    )
    if getattr(store, "degraded", False):
        line += f"; standing in for Neo4j: {store.degraded_reason}"
    return line
