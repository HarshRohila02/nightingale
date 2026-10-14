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
    "build_pipeline",
    "calibrator_from_config",
    "describe_graph",
    "explainer_from_config",
    "fusion_weights_from_config",
    "load_config",
    "open_graph_from_config",
    "open_ranker_from_config",
    "open_retriever_from_config",
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


def fusion_weights_from_config(config: dict[str, Any]) -> dict[str, float]:
    """``ml_weight`` and ``kg_weight`` from the ``ranking:`` block, for ``DiagnosisPipeline``.

    The graph's share of the fusion, α = kg_weight / (ml_weight + kg_weight), is the one quantity
    EXP-006 tunes (``src/fusion/pool.py``). Reading it here is what lets the demo,
    ``scripts/check_crosswalk.py`` and the golden-case ranking tests run at the configured α. The
    other ``ranking:`` keys are not read by the scripts yet.

    Raises:
        ValueError: for a pair the fusion cannot use (negative, both 0, not finite).
    """
    from src.fusion import DEFAULT_KG_WEIGHT, DEFAULT_ML_WEIGHT, graph_weight

    ranking = config.get("ranking") or {}
    ml_weight = float(ranking.get("ml_weight", DEFAULT_ML_WEIGHT))
    kg_weight = float(ranking.get("kg_weight", DEFAULT_KG_WEIGHT))
    graph_weight(ml_weight, kg_weight)
    return {"ml_weight": ml_weight, "kg_weight": kg_weight}


def calibrator_from_config(config: dict[str, Any]) -> Any:
    """The calibrator ``calibration.file`` names for the configured system, or None.

    None when no file is configured: the system is not calibrated, and shows ranks only. A file
    that is missing or unreadable, or a calibrator fitted for another fused system (another model,
    another α), gives a :class:`~src.fusion.calibration.RefusedCalibrator`, so every result says
    ``calibration`` rather than showing probabilities that belong to something else.
    """
    import json

    from src.fusion import graph_weight
    from src.fusion.calibration import Calibrator, RefusedCalibrator

    name = (config.get("calibration") or {}).get("file")
    if not name:
        return None
    try:
        calibrator = Calibrator.from_json(
            json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        return RefusedCalibrator(f"{name}: {error}")
    ml = config.get("ml") or {}
    reason = calibrator.mismatch(
        {
            "kind": "fused",
            "backend": ml.get("backend"),
            "model_dir": ml.get("model_dir"),
            "alpha": graph_weight(**fusion_weights_from_config(config)),
        }
    )
    return RefusedCalibrator(f"{name}: {reason}") if reason else calibrator


def explainer_from_config(config: dict[str, Any], ranker: Any = None) -> Any:
    """The explainer the ``llm:`` block asks for (task 3b).

    With ``llm.enabled`` false, the default, the template explainer (with the model's reasons,
    2b), exactly as before. With it true, :class:`~src.llm.explainer.LLMExplainer` over an
    :class:`~src.llm.client.OllamaClient` for ``llm.model``, with ``temperature``, ``max_tokens``
    and ``timeout_seconds`` from the block, and the template explainer standing in for it.
    Building it contacts nothing: the model is asked only when a case is explained. Which model
    (D-5), and where it runs (D-7, D-9), are the owner's.

    Raises:
        ValueError: for a provider other than ollama, or no model named.
    """
    from src.stubs import TemplateExplainer

    llm = config.get("llm") or {}
    if not llm.get("enabled", False):
        return TemplateExplainer(ranker=ranker)
    if llm.get("provider", "ollama") != "ollama":
        raise ValueError(f"llm.provider {llm.get('provider')!r}: only ollama is implemented")
    from src.llm.client import OllamaClient
    from src.llm.explainer import LLMExplainer

    client = OllamaClient(
        str(llm.get("model") or ""),
        temperature=float(llm.get("temperature", 0.1)),
        max_tokens=int(llm.get("max_tokens", 512)),
    )
    return LLMExplainer(
        client, ranker=ranker, timeout_seconds=float(llm.get("timeout_seconds", 30))
    )


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


def open_retriever_from_config(config: dict[str, Any]) -> Any:
    """The evidence retriever the ``retrieval:`` block asks for (src/rag/retriever.py, task 3a).

    ``retrieval.bundle`` names the unzipped bundle the Kaggle notebook writes
    (``notebooks/kaggle_retrieval_corpus.ipynb``, docs/11 §4.6); null switches retrieval off.
    ``top_k`` is how far down each of BM25's and the dense search's lists enters the fusion, and
    ``rrf_k`` the fusion's constant. Returns a retriever that gives no evidence and reports
    ``rag``, never an exception, when the bundle is null, absent — the ordinary case, since
    ``data/`` is gitignored — or fails its manifest's checks.
    """
    from src.rag.retriever import DEPTH, RRF_K, open_retriever

    retrieval = config.get("retrieval") or {}
    bundle = retrieval.get("bundle")
    return open_retriever(
        REPO_ROOT / bundle if bundle else None,
        depth=int(retrieval.get("top_k", DEPTH)),
        rrf_k=int(retrieval.get("rrf_k", RRF_K)),
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


def build_pipeline(config: dict[str, Any], graph_backend: str | None = None) -> dict[str, Any]:
    """The pipeline as ``scripts/demo.py`` builds it, for the app and the dashboard.

    The configured ranker and graph, the configured evidence retriever (3a; without its bundle it
    gives no evidence and every result reports ``rag``), the configured explainer (the template,
    with the model's reasons, 2b, unless ``llm.enabled``: :func:`explainer_from_config`), the
    configured fusion weights, how many candidates get evidence (``ranking.top_k_evidence``) and
    how many passages each (``retrieval.passages_per_candidate``), and the configured calibrator,
    if any (2b).

    Returns:
        ``pipeline``, ``ranker``, ``graph``, ``retriever``, ``labels`` (the real graph's node
        labels, for ``expand_case``; None for the stub) and ``about`` (one line each for the
        ranker, the graph and the retriever).

    Raises:
        GraphUnavailable: as :func:`open_graph_from_config`.
    """
    from src.ml.ranker import describe
    from src.pipeline import DiagnosisPipeline
    from src.rag.retriever import describe as describe_retriever

    ranker = open_ranker_from_config(config)
    graph = open_graph_from_config(config, graph_backend)
    retriever = open_retriever_from_config(config)
    labels = dict(graph.graph.nodes(data="label")) if hasattr(graph, "graph") else None
    pipeline = DiagnosisPipeline(
        ranker=ranker,
        graph=graph,
        retriever=retriever,
        explainer=explainer_from_config(config, ranker),
        **fusion_weights_from_config(config),
        top_k_evidence=int((config.get("ranking") or {}).get("top_k_evidence", 3)),
        evidence_per_candidate=int(
            (config.get("retrieval") or {}).get("passages_per_candidate", 3)
        ),
        calibrator=calibrator_from_config(config),
    )
    return {
        "pipeline": pipeline,
        "ranker": ranker,
        "graph": graph,
        "retriever": retriever,
        "labels": labels,
        "about": {
            "ranker": describe(ranker),
            "graph": describe_graph(graph),
            "retriever": describe_retriever(retriever),
        },
    }
