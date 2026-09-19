"""Neo4j-backed ``GraphStore``: the cardiac KG on AuraDB (task 1b, decision D-6).

Neo4j holds the team's shared, persistent copy of the knowledge graph, which anyone with the
credentials can browse in the Aura console and query in Cypher. The store reads that copy once,
at start-up, into the same in-memory graph the NetworkX store uses, and scores it there:

* scores, paths and expected findings are identical to the NetworkX store's by construction,
  and the tests check the round trip;
* a paused Aura instance or a dropped connection cannot stall a consultation half-way; and
* the graph is small (about 130 nodes and 320 edges), so reading all of it is cheap.

**Where the graph comes from.** :func:`src.medical_kg.cardiac_kg.build_cardiac_kg` builds it from
local files, and :meth:`Neo4jGraphStore.connect` writes it to Neo4j whenever the stored copy
differs. A fingerprint over every node and edge decides that. Without local files, the store
serves the stored copy as it is, once it has checked that nobody edited it after it was written.

**When Neo4j is unavailable.** :func:`open_graph_store` is the entry point. When Neo4j is not
configured, cannot be reached, refuses the login or holds no usable graph, it falls back to the
NetworkX store over the locally built graph and marks it ``degraded``. The pipeline then reports
``graph_backend`` in ``degraded_components`` (docs/02-architecture.md §7).

**Credentials** come from ``.env`` (docs/11-compute-runbook.md §5) and are never logged. On
AuraDB the home database is named after the instance id, not ``neo4j``, so the store uses the
home database unless ``NEO4J_DATABASE`` names one.

**Layout in Neo4j.** Every node carries the namespace label (``CardiacKG``) and its type label
(``Condition``, ``Symptom`` or ``RiskFactor``). Edges are ``HAS_SYMPTOM`` or ``HAS_RISK_FACTOR``
and carry their ``source`` and ``weight`` (R-12). One ``KGMeta`` node per namespace records the
fingerprint. Tests use a namespace of their own, so they never touch the real graph.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from neo4j import READ_ACCESS, Driver, GraphDatabase, ManagedTransaction
from neo4j.exceptions import AuthError, DriverError, Neo4jError, ServiceUnavailable

from src.medical_kg.loader import EdgeSource, KGEdge, KGNode, KnowledgeGraph, NodeType, Relation
from src.medical_kg.networkx_store import NetworkXGraphStore

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_NAMESPACE",
    "DEFAULT_TIMEOUT_SECONDS",
    "GraphIntegrityError",
    "GraphOutOfDate",
    "GraphUnavailable",
    "Neo4jGraphStore",
    "Neo4jSettings",
    "StoredGraph",
    "delete_kg",
    "describe_failure",
    "kg_fingerprint",
    "kg_from_rows",
    "kg_to_rows",
    "open_graph_store",
    "read_kg",
    "settings_from_env",
    "write_kg",
]

DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_NAMESPACE = "CardiacKG"
DEFAULT_TIMEOUT_SECONDS = 10.0
"""How long to wait for Neo4j before falling back. A paused Aura instance fails within it."""

NODE_KEYS = frozenset({"id", "type", "label", "ordinal"})
"""Node properties this module owns. A graph's own properties may not use these names."""
EDGE_KEYS = frozenset({"source", "weight", "ordinal"})
"""Edge properties this module owns."""

_NAMESPACE = re.compile(r"[A-Z][A-Za-z0-9]*")


class GraphUnavailable(RuntimeError):
    """No usable graph in Neo4j: not configured, unreachable, empty or out of date."""


class GraphIntegrityError(GraphUnavailable):
    """The graph in Neo4j is not the graph that was written to it."""


class GraphOutOfDate(GraphUnavailable):
    """The graph in Neo4j differs from the local build, and writing it was not allowed."""


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Neo4jSettings:
    """Where the graph lives. Build it with :func:`settings_from_env`.

    Attributes:
        uri: ``neo4j+s://<instance-id>.databases.neo4j.io`` on AuraDB.
        user: The database user.
        password: Never printed: it is left out of ``repr``.
        database: A database name, or None for the user's home database (the right choice on
            AuraDB, where it is named after the instance id).
    """

    uri: str
    user: str
    password: str = field(repr=False)
    database: str | None = None


def settings_from_env(
    env: Mapping[str, str] | None = None, *, dotenv_path: Path | None = DOTENV_PATH
) -> Neo4jSettings:
    """Read the connection settings from the environment and ``.env``.

    Reads ``NEO4J_URI``, ``NEO4J_USER`` (or ``NEO4J_USERNAME``, as Aura's credentials file
    spells it), ``NEO4J_PASSWORD`` and, optionally, ``NEO4J_DATABASE``. A variable already set
    in the environment wins over ``.env``, which is read without changing ``os.environ``.

    Args:
        env: Variables to use instead of the environment and ``.env`` (for tests).
        dotenv_path: The ``.env`` file; None to ignore it.

    Raises:
        GraphUnavailable: if the URI, the user or the password is missing.
    """
    if env is None:
        from_file = {}
        if dotenv_path is not None and dotenv_path.exists():
            from_file = {k: v for k, v in dotenv_values(dotenv_path).items() if v is not None}
        env = {**from_file, **os.environ}
    uri = (env.get("NEO4J_URI") or "").strip()
    user = (env.get("NEO4J_USER") or env.get("NEO4J_USERNAME") or "").strip()
    password = env.get("NEO4J_PASSWORD") or ""
    database = (env.get("NEO4J_DATABASE") or "").strip() or None
    missing = [
        name
        for name, value in (("NEO4J_URI", uri), ("NEO4J_USER", user), ("NEO4J_PASSWORD", password))
        if not value
    ]
    if missing:
        raise GraphUnavailable(f"{', '.join(missing)} not set; fill them in .env (docs/11 §5)")
    return Neo4jSettings(uri=uri, user=user, password=password, database=database)


# --------------------------------------------------------------------------- #
# The graph as rows, and its fingerprint
# --------------------------------------------------------------------------- #


def _plain(value: Any) -> Any:
    """A value Neo4j can store and JSON can hash: enums by value, sequences as lists."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _plain_properties(
    properties: Mapping[str, Any], reserved: frozenset[str], owner: str
) -> dict[str, Any]:
    """The properties as Neo4j will store them. None and empty lists are left out: Neo4j
    cannot store a null, and an empty list has no element type."""
    clash = reserved & set(properties)
    if clash:
        raise ValueError(f"{owner} uses property names reserved by the store: {sorted(clash)}")
    plain: dict[str, Any] = {}
    for key, value in properties.items():
        value = _plain(value)
        if value is None or value == []:
            continue
        if isinstance(value, Mapping):
            raise ValueError(f"{owner}.{key} is a map, which a Neo4j property cannot hold")
        plain[key] = value
    return plain


def kg_to_rows(kg: KnowledgeGraph) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The graph as plain rows, in its own order, ready for Cypher.

    Returns:
        ``(nodes, edges)``. A node row holds every property of the node, including ``id``,
        ``type``, ``label`` and its position (``ordinal``). An edge row holds ``condition_id``,
        ``concept_id``, ``relation`` and ``props``: ``source``, ``weight``, ``ordinal`` and the
        edge's own properties.

    Raises:
        ValueError: if a property uses a reserved name or holds a map.
    """
    nodes = [
        {
            **_plain_properties(node.properties, NODE_KEYS, node.id),
            "id": node.id,
            "type": node.type.value,
            "label": node.label,
            "ordinal": ordinal,
        }
        for ordinal, node in enumerate(kg.nodes.values())
    ]
    edges = [
        {
            "condition_id": edge.condition_id,
            "concept_id": edge.concept_id,
            "relation": edge.relation.value,
            "props": {
                **_plain_properties(
                    edge.properties, EDGE_KEYS, f"{edge.condition_id} -> {edge.concept_id}"
                ),
                "source": edge.source.value,
                "weight": float(edge.weight),
                "ordinal": ordinal,
            },
        }
        for ordinal, edge in enumerate(kg.edges)
    ]
    return nodes, edges


def kg_from_rows(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    anomalies: Sequence[str] = (),
) -> KnowledgeGraph:
    """The inverse of :func:`kg_to_rows`: rebuild the graph, in its original order."""
    kg_nodes: dict[str, KGNode] = {}
    for row in sorted(nodes, key=lambda r: r["ordinal"]):
        kg_nodes[row["id"]] = KGNode(
            id=row["id"],
            type=NodeType(row["type"]),
            label=row["label"],
            properties={k: v for k, v in row.items() if k not in NODE_KEYS},
        )
    kg_edges = [
        KGEdge(
            condition_id=row["condition_id"],
            concept_id=row["concept_id"],
            relation=Relation(row["relation"]),
            source=EdgeSource(row["props"]["source"]),
            weight=float(row["props"]["weight"]),
            properties={k: v for k, v in row["props"].items() if k not in EDGE_KEYS},
        )
        for row in sorted(edges, key=lambda r: r["props"]["ordinal"])
    ]
    return KnowledgeGraph(nodes=kg_nodes, edges=kg_edges, anomalies=list(anomalies))


def kg_fingerprint(kg: KnowledgeGraph) -> str:
    """SHA-256 over every node and edge, as Neo4j stores them. Order does not matter."""
    nodes, edges = kg_to_rows(kg)
    lines = sorted(
        json.dumps({k: v for k, v in row.items() if k != "ordinal"}, sort_keys=True)
        for row in nodes
    )
    lines += sorted(
        json.dumps(
            {**row, "props": {k: v for k, v in row["props"].items() if k != "ordinal"}},
            sort_keys=True,
        )
        for row in edges
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Writing and reading
# --------------------------------------------------------------------------- #


def _checked(namespace: str) -> str:
    """Labels cannot be Cypher parameters, so a namespace must be a plain CamelCase label."""
    if not _NAMESPACE.fullmatch(namespace):
        raise ValueError(f"a namespace must be a CamelCase label, got {namespace!r}")
    return namespace


def _constraint_name(namespace: str) -> str:
    return f"{namespace.lower()}_id"


def write_kg(
    driver: Driver,
    kg: KnowledgeGraph,
    *,
    database: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> str:
    """Replace the namespace's graph in Neo4j with ``kg``, in one transaction.

    Only nodes with the namespace label and that namespace's ``KGMeta`` node are touched.

    Returns:
        The fingerprint recorded with the graph.
    """
    ns = _checked(namespace)
    nodes, edges = kg_to_rows(kg)
    fingerprint = kg_fingerprint(kg)
    meta: dict[str, Any] = {
        "namespace": ns,
        "fingerprint": fingerprint,
        "nodes": len(nodes),
        "edges": len(edges),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if kg.anomalies:
        meta["anomalies"] = list(kg.anomalies)
    nodes_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in nodes:
        nodes_by_type[NodeType(row["type"]).value].append(row)
    edges_by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edges:
        edges_by_relation[Relation(row["relation"]).value].append(row)

    # A schema change cannot share a transaction with data writes.
    driver.execute_query(
        f"CREATE CONSTRAINT {_constraint_name(ns)} IF NOT EXISTS "
        f"FOR (n:{ns}) REQUIRE n.id IS UNIQUE",
        database_=database,
    )
    with driver.session(database=database) as session:
        session.execute_write(_replace_graph, ns, nodes_by_type, edges_by_relation, meta)
    return fingerprint


def _replace_graph(
    tx: ManagedTransaction,
    ns: str,
    nodes_by_type: Mapping[str, list[dict[str, Any]]],
    edges_by_relation: Mapping[str, list[dict[str, Any]]],
    meta: dict[str, Any],
) -> None:
    tx.run(f"MATCH (n:{ns}) DETACH DELETE n").consume()
    tx.run("MATCH (m:KGMeta {namespace: $ns}) DELETE m", ns=ns).consume()
    # The type labels and relation types come from the NodeType and Relation enums.
    for node_type, rows in nodes_by_type.items():
        tx.run(f"UNWIND $rows AS row CREATE (n:{ns}:{node_type}) SET n = row", rows=rows).consume()
    for relation, rows in edges_by_relation.items():
        tx.run(
            f"UNWIND $rows AS row "
            f"MATCH (c:{ns} {{id: row.condition_id}}) MATCH (x:{ns} {{id: row.concept_id}}) "
            f"CREATE (c)-[r:{relation}]->(x) SET r = row.props",
            rows=rows,
        ).consume()
    tx.run("CREATE (m:KGMeta) SET m = $meta", meta=meta).consume()


@dataclass(frozen=True)
class StoredGraph:
    """A graph read from Neo4j.

    Attributes:
        kg: The graph, rebuilt in the order it was written.
        fingerprint: Computed from what was read.
        recorded: The fingerprint recorded when it was written. A difference means someone
            edited the graph in Neo4j afterwards.
        written_at: When it was written (UTC, ISO 8601).
    """

    kg: KnowledgeGraph
    fingerprint: str
    recorded: str
    written_at: str | None


def read_kg(
    driver: Driver, *, database: str | None = None, namespace: str = DEFAULT_NAMESPACE
) -> StoredGraph | None:
    """Read the namespace's graph from Neo4j in one read transaction, or None if it has none."""
    ns = _checked(namespace)
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        return session.execute_read(_read_graph, ns)


def _read_graph(tx: ManagedTransaction, ns: str) -> StoredGraph | None:
    meta_rows = tx.run(
        "MATCH (m:KGMeta {namespace: $ns}) RETURN properties(m) AS meta", ns=ns
    ).data()
    if not meta_rows:
        return None
    meta = meta_rows[0]["meta"]
    nodes = [r["props"] for r in tx.run(f"MATCH (n:{ns}) RETURN properties(n) AS props").data()]
    edges = tx.run(
        f"MATCH (c:{ns})-[r]->(x:{ns}) RETURN c.id AS condition_id, x.id AS concept_id, "
        "type(r) AS relation, properties(r) AS props"
    ).data()
    kg = kg_from_rows(nodes, edges, meta.get("anomalies", []))
    return StoredGraph(
        kg=kg,
        fingerprint=kg_fingerprint(kg),
        recorded=meta["fingerprint"],
        written_at=meta.get("written_at"),
    )


def delete_kg(
    driver: Driver, *, database: str | None = None, namespace: str = DEFAULT_NAMESPACE
) -> None:
    """Remove everything this module created for the namespace: nodes, meta and constraint."""
    ns = _checked(namespace)
    driver.execute_query(f"MATCH (n:{ns}) DETACH DELETE n", database_=database)
    driver.execute_query("MATCH (m:KGMeta {namespace: $ns}) DELETE m", ns=ns, database_=database)
    driver.execute_query(f"DROP CONSTRAINT {_constraint_name(ns)} IF EXISTS", database_=database)


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


def _driver(settings: Neo4jSettings, timeout: float) -> Driver:
    return GraphDatabase.driver(
        settings.uri,
        auth=(settings.user, settings.password),
        connection_timeout=timeout,
        connection_acquisition_timeout=timeout,
        max_transaction_retry_time=timeout,
    )


class Neo4jGraphStore(NetworkXGraphStore):
    """The cardiac KG as stored in Neo4j, served from an in-memory copy. Implements ``GraphStore``.

    Build it with :meth:`connect`, or with :func:`open_graph_store`, which falls back to the
    NetworkX store when Neo4j cannot be reached.

    Attributes:
        fingerprint: The fingerprint of the graph being served.
        written_at: When that graph was written to Neo4j.
        server: Neo4j's version string, e.g. ``Neo4j/5.27-aura``.
        wrote: True if :meth:`connect` had to write the graph first.
        namespace: The label on the graph's nodes.
    """

    backend = "neo4j"

    def __init__(self, stored: StoredGraph, *, server: str, wrote: bool, namespace: str) -> None:
        super().__init__(stored.kg)
        self.fingerprint = stored.fingerprint
        self.written_at = stored.written_at
        self.server = server
        self.wrote = wrote
        self.namespace = namespace

    @classmethod
    def connect(
        cls,
        settings: Neo4jSettings,
        *,
        local_kg: KnowledgeGraph | None = None,
        sync: bool = True,
        namespace: str = DEFAULT_NAMESPACE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Neo4jGraphStore:
        """Read the graph from Neo4j, first writing ``local_kg`` there if the stored copy differs.

        The connection is closed before this returns: the store works from its copy.

        Args:
            settings: From :func:`settings_from_env`.
            local_kg: The graph built from local files, or None to serve the stored copy as is.
            sync: Write ``local_kg`` when the stored copy differs. If False, raise instead.
            namespace: The label that marks this graph's nodes.
            timeout: Seconds to wait for Neo4j.

        Raises:
            GraphUnavailable: Neo4j holds no graph and none was given.
            GraphOutOfDate: the stored graph differs from ``local_kg`` and ``sync`` is False.
            GraphIntegrityError: the stored graph was edited after it was written, or a write
                did not read back identically.
            neo4j.exceptions.Neo4jError, neo4j.exceptions.DriverError: Neo4j refused the login
                or could not be reached.
        """
        with _driver(settings, timeout) as driver:
            driver.verify_connectivity()
            server = driver.get_server_info().agent
            stored = read_kg(driver, database=settings.database, namespace=namespace)
            wrote = False
            if local_kg is not None:
                wanted = kg_fingerprint(local_kg)
                current = (
                    stored is not None
                    and stored.fingerprint == wanted
                    and stored.recorded == wanted
                )
                if not current:
                    if not sync:
                        raise GraphOutOfDate(
                            f"the {namespace} graph in Neo4j is out of date; "
                            "run scripts/load_neo4j.py"
                        )
                    write_kg(driver, local_kg, database=settings.database, namespace=namespace)
                    wrote = True
                    stored = read_kg(driver, database=settings.database, namespace=namespace)
                    if stored is None or stored.fingerprint != wanted:
                        raise GraphIntegrityError(
                            f"the {namespace} graph read back from Neo4j differs from the one "
                            "just written"
                        )
            elif stored is None:
                raise GraphUnavailable(
                    f"Neo4j holds no {namespace} graph; load it with scripts/load_neo4j.py"
                )
            elif stored.fingerprint != stored.recorded:
                raise GraphIntegrityError(
                    f"the {namespace} graph in Neo4j was changed after it was written; "
                    "reload it with scripts/load_neo4j.py"
                )
        return cls(stored, server=server, wrote=wrote, namespace=namespace)


def describe_failure(exc: BaseException) -> str:
    """A short reason for logs and the degradation note. It never includes a credential."""
    if isinstance(exc, AuthError):
        detail = "the login was refused; check NEO4J_USER and NEO4J_PASSWORD in .env"
    elif isinstance(exc, ServiceUnavailable):
        detail = (
            "no connection; the Aura instance may be paused (resume it in the console) "
            "or the machine is offline"
        )
    else:
        message = str(exc).strip()
        detail = message.splitlines()[0][:200] if message else ""
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def open_graph_store(
    backend: str = "neo4j",
    *,
    local_kg: KnowledgeGraph | None = None,
    settings: Neo4jSettings | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> NetworkXGraphStore:
    """The graph store the configuration asks for, degrading instead of failing (docs/02 §7).

    Args:
        backend: ``graph.backend`` in configs/config.yaml. ``networkx`` serves ``local_kg`` from
            memory. ``neo4j`` uses :meth:`Neo4jGraphStore.connect`. If Neo4j is not configured,
            cannot be reached, refuses the login or holds no usable graph, the NetworkX store
            over ``local_kg`` stands in, marked ``degraded`` with the reason.
        local_kg: The graph built from local files, if there is one.
        settings: Connection settings; by default :func:`settings_from_env`.
        namespace: The label that marks this graph's nodes in Neo4j.
        timeout: Seconds to wait for Neo4j.

    Raises:
        GraphUnavailable: there is no graph at all. Neo4j failed and ``local_kg`` is None, or
            the networkx backend was asked for without a local graph.
        ValueError: for an unknown backend.
    """
    if backend == "networkx":
        if local_kg is None:
            raise GraphUnavailable("the networkx backend needs the graph built from local files")
        return NetworkXGraphStore(local_kg)
    if backend != "neo4j":
        raise ValueError(f"unknown graph backend {backend!r}; expected 'neo4j' or 'networkx'")
    try:
        return Neo4jGraphStore.connect(
            settings or settings_from_env(),
            local_kg=local_kg,
            namespace=namespace,
            timeout=timeout,
        )
    except (GraphUnavailable, Neo4jError, DriverError, OSError) as exc:
        reason = describe_failure(exc)
        if local_kg is None:
            raise GraphUnavailable(
                f"Neo4j is unavailable ({reason}) and there is no local graph to fall back on"
            ) from exc
        logger.warning(
            "Neo4j is unavailable (%s); serving the graph built from local files", reason
        )
        return NetworkXGraphStore(local_kg, degraded_reason=reason)
