"""Load the cardiac medical knowledge graph from DDXPlus (task 1b).

The KG backbone is DDXPlus ``release_conditions.json``, the R-01 fallback
(docs/10-spike-r01-crosswalk.md). scripts/decode_ddxplus.py has already filtered it to
our 13 conditions (``data/interim/ddxplus_chestpain_conditions.json``) and decoded the
evidence vocabulary (``data/interim/ddxplus_evidences.json``). This module turns those two
files into a backend-neutral :class:`KnowledgeGraph` of plain nodes and edges. Any
``GraphStore`` can be built from it: NetworkX (``networkx_store``) or Neo4j (``neo4j_store``).

Schema (docs/02-architecture.md §5)::

    (Condition)-[:HAS_SYMPTOM]->(Symptom)
    (Condition)-[:HAS_RISK_FACTOR]->(RiskFactor)

Two properties matter for honest reporting:

* **Every edge records its source**: ``ddxplus`` here, later ``bodhi_s`` and
  ``hand_authored``. That lets the KG's contribution be reported by source, which the
  circularity disclosure for risk R-12 requires (docs/05-evaluation-protocol.md §8.7).
* **DDXPlus links conditions to evidence questions, not answers.** It says pericarditis
  involves "characterize your pain", not that its pain is sharp. The graph therefore
  cannot tell tearing pain from burning pain. That must come from BODHI-S enrichment,
  the hand-authored crosswalk and red-flag rules, or the ML ranker.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from src.conditions import BY_ID, CONDITIONS
from src.ddxplus import CONCEPT_PREFIX, concept_id
from src.medical_kg.crosswalk import BY_CONCEPT, canonical_concept_id

__all__ = [
    "LIKELIHOOD_WEIGHT",
    "EdgeSource",
    "KGEdge",
    "KGNode",
    "KnowledgeGraph",
    "Likelihood",
    "NodeType",
    "Relation",
    "concept_node",
    "condition_nodes",
    "load_ddxplus_kg",
    "merge_graphs",
    "only_sources",
    "relation_for",
]


class NodeType(str, Enum):
    CONDITION = "Condition"
    SYMPTOM = "Symptom"
    RISK_FACTOR = "RiskFactor"


class Relation(str, Enum):
    HAS_SYMPTOM = "HAS_SYMPTOM"
    HAS_RISK_FACTOR = "HAS_RISK_FACTOR"


class EdgeSource(str, Enum):
    """Where an edge's knowledge came from — reported separately for R-12."""

    DDXPLUS = "ddxplus"
    BODHI_S = "bodhi_s"
    HAND_AUTHORED = "hand_authored"


class Likelihood(str, Enum):
    """How often a finding is present in a condition, P(finding | condition), in bands.

    BODHI-S states its likelihoods in these categories, and the hand-authored facts are banded the
    same way, so both sources share one scale.
    """

    ZERO = "zero"
    RARE = "rare"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


LIKELIHOOD_WEIGHT: Mapping[Likelihood, float] = {
    Likelihood.ZERO: 0.0,
    Likelihood.RARE: 0.03,  # under 5% of patients
    Likelihood.LOW: 0.12,  # 5-19%
    Likelihood.MEDIUM: 0.35,  # 20-49%
    Likelihood.HIGH: 0.65,  # 50-79%
    Likelihood.VERY_HIGH: 0.9,  # 80% or more
}
"""Edge weight per category: the middle of its band. BODHI-S publishes no numeric bands, so these
are an assumption (open decision A-6, docs/02 §9). DDXPlus edges keep weight 1.0, because
DDXPlus lists a finding without saying how often it occurs."""

_NODE_TYPE_FOR: dict[Relation, NodeType] = {
    Relation.HAS_SYMPTOM: NodeType.SYMPTOM,
    Relation.HAS_RISK_FACTOR: NodeType.RISK_FACTOR,
}
_RELATION_FOR: dict[NodeType, Relation] = {v: k for k, v in _NODE_TYPE_FOR.items()}


@dataclass(frozen=True)
class KGNode:
    id: str
    type: NodeType
    label: str
    properties: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class KGEdge:
    """A directed edge from a condition to one of its evidence concepts."""

    condition_id: str
    concept_id: str
    relation: Relation
    source: EdgeSource
    weight: float = 1.0
    properties: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass
class KnowledgeGraph:
    """Backend-neutral KG: what a GraphStore is built from.

    Attributes:
        nodes: {node id: node}.
        edges: Every edge. The same condition-concept pair may appear once per source.
        anomalies: Oddities in the source data that the loader tolerated. Surfaced, never
            hidden. scripts/build_cardiac_kg.py prints them.
    """

    nodes: dict[str, KGNode]
    edges: list[KGEdge]
    anomalies: list[str] = field(default_factory=list)


def load_ddxplus_kg(conditions_path: Path, vocabulary_path: Path) -> KnowledgeGraph:
    """Build the KG from the two interim files written by scripts/decode_ddxplus.py.

    Condition nodes come from the canonical registry (src/conditions.py), so all 14 exist,
    including aortic dissection. DDXPlus has no aortic dissection, so here it gets no
    edges; hand-authoring them is a separate 1b task.

    Evidence nodes are typed by the relation that uses them. That is ``symptoms`` versus
    ``antecedents`` in ``release_conditions.json``, the curated statement. Where DDXPlus's
    own ``is_antecedent`` flag disagrees, the node follows the relation and the
    disagreement is recorded as an anomaly.

    Raises:
        ValueError: if the file names a condition the registry does not know, cites a
            code missing from the vocabulary, or uses one code both as a symptom and as a
            risk factor. Each is a data error that needs a human decision.
    """
    conditions = json.loads(Path(conditions_path).read_text(encoding="utf-8"))
    vocabulary = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))

    unknown = sorted(set(conditions) - set(BY_ID))
    if unknown:
        raise ValueError(f"conditions not in src/conditions.py: {unknown}")

    nodes: dict[str, KGNode] = {}
    for condition in CONDITIONS:
        spec = conditions.get(condition.id, {})
        nodes[condition.id] = KGNode(
            id=condition.id,
            type=NodeType.CONDITION,
            label=condition.label,
            properties={
                "category": condition.category.value,
                "is_must_not_miss": condition.is_must_not_miss,
                "in_training_data": condition.in_training_data,
                "icd10": spec.get("icd10"),
                "ddxplus_severity": spec.get("severity"),
            },
        )

    edges: list[KGEdge] = []
    relations_by_code: dict[str, set[Relation]] = defaultdict(set)
    for condition_id, spec in conditions.items():
        for key, relation in (
            ("symptoms", Relation.HAS_SYMPTOM),
            ("antecedents", Relation.HAS_RISK_FACTOR),
        ):
            for entry in spec.get(key, []):
                code = entry["code"]
                relations_by_code[code].add(relation)
                edges.append(
                    KGEdge(
                        condition_id=condition_id,
                        concept_id=concept_id(code),
                        relation=relation,
                        source=EdgeSource.DDXPLUS,
                    )
                )

    anomalies: list[str] = []
    for code, relations in sorted(relations_by_code.items()):
        if len(relations) > 1:
            raise ValueError(f"{code} is used both as a symptom and as a risk factor")
        if code not in vocabulary:
            raise ValueError(f"{code} is cited by a condition but missing from the vocabulary")
        entry = vocabulary[code]
        node_type = _NODE_TYPE_FOR[next(iter(relations))]
        if bool(entry.get("is_antecedent")) != (node_type is NodeType.RISK_FACTOR):
            anomalies.append(
                f"{code} ({entry.get('label_en', '')!r}) is used as a {node_type.value} "
                f"but DDXPlus flags is_antecedent={entry.get('is_antecedent')}; "
                "typed by its use"
            )
        nodes[concept_id(code)] = KGNode(
            id=concept_id(code),
            type=node_type,
            label=entry.get("label_en", code),
            properties={
                "ddxplus_code": code,
                "label_fr": entry.get("label_fr"),
                "data_type": entry.get("data_type"),
                "ddxplus_is_antecedent": bool(entry.get("is_antecedent")),
                "source": EdgeSource.DDXPLUS.value,
            },
        )

    for condition in CONDITIONS:
        if condition.in_training_data and condition.id not in conditions:
            anomalies.append(f"{condition.id} is trainable but has no DDXPlus evidence")

    return KnowledgeGraph(nodes=nodes, edges=edges, anomalies=anomalies)


# --------------------------------------------------------------------------- #
# Composing sources
# --------------------------------------------------------------------------- #


def relation_for(node_type: NodeType) -> Relation:
    """The relation from a condition to an evidence node of this type."""
    try:
        return _RELATION_FOR[node_type]
    except KeyError:
        raise ValueError(f"a {node_type.value} node is not evidence") from None


def condition_nodes() -> dict[str, KGNode]:
    """One node per registry condition, with the registry's properties only.

    A source other than DDXPlus starts from these. When graphs are merged, DDXPlus's richer
    condition nodes (with ICD-10 codes and severity) come first and are the ones kept.
    """
    return {
        c.id: KGNode(
            id=c.id,
            type=NodeType.CONDITION,
            label=c.label,
            properties={
                "category": c.category.value,
                "is_must_not_miss": c.is_must_not_miss,
                "in_training_data": c.in_training_data,
            },
        )
        for c in CONDITIONS
    }


def concept_node(concept: str, labels: Mapping[str, str], source: EdgeSource) -> KGNode:
    """The node a non-DDXPlus source uses for a hand-authored concept.

    It sits at the concept's canonical id (:func:`canonical_concept_id`). A concept on a DDXPlus
    question's node takes that question's label from ``labels``; any other concept keeps its
    crosswalk label. ``RF:*`` concepts are risk factors and ``SYM:*`` concepts symptoms; where
    DDXPlus types the question differently, :func:`merge_graphs` keeps DDXPlus's type.
    """
    node_id = canonical_concept_id(concept)
    entry = BY_CONCEPT[concept]
    node_type = NodeType.RISK_FACTOR if concept.startswith("RF:") else NodeType.SYMPTOM
    if node_id.startswith(CONCEPT_PREFIX):
        return KGNode(
            id=node_id,
            type=node_type,
            label=labels.get(node_id, entry.label),
            properties={"ddxplus_code": node_id[len(CONCEPT_PREFIX) :], "source": source.value},
        )
    return KGNode(
        id=node_id,
        type=node_type,
        label=entry.label,
        properties={
            "crosswalk_match": entry.match.value,
            "ddxplus_code": entry.pattern.code if entry.pattern else None,
            "source": source.value,
        },
    )


def merge_graphs(*graphs: KnowledgeGraph) -> KnowledgeGraph:
    """Combine the graphs of several sources into one.

    Nodes are united by id. Where two sources type a node differently, the earlier graph's type
    is kept and the difference is recorded as an anomaly, so pass DDXPlus first. Every edge then
    takes the relation that its target's type implies: a risk factor that another source calls a
    symptom stays a risk factor. Edges stay per source. A fact two sources state becomes two
    parallel edges, which the store counts once.

    Raises:
        ValueError: if a source states the same condition-concept edge twice (it must combine
            them first), or an edge points at a node no graph defines.
    """
    nodes: dict[str, KGNode] = {}
    anomalies: list[str] = []
    for graph in graphs:
        anomalies.extend(graph.anomalies)
        for node_id, node in graph.nodes.items():
            kept = nodes.setdefault(node_id, node)
            if kept.type is not node.type:
                anomalies.append(
                    f"{node_id} ({kept.label!r}) is a {node.type.value} for the "
                    f"{node.properties.get('source', 'later')} source but was already a "
                    f"{kept.type.value}; kept {kept.type.value}"
                )

    edges: list[KGEdge] = []
    seen: set[tuple[str, str, EdgeSource]] = set()
    for graph in graphs:
        for edge in graph.edges:
            key = (edge.condition_id, edge.concept_id, edge.source)
            if key in seen:
                raise ValueError(f"{edge.source.value} states {key[0]} -> {key[1]} twice")
            seen.add(key)
            if edge.condition_id not in nodes or edge.concept_id not in nodes:
                raise ValueError(f"edge {key[0]} -> {key[1]} points at a missing node")
            relation = relation_for(nodes[edge.concept_id].type)
            edges.append(edge if edge.relation is relation else replace(edge, relation=relation))
    return KnowledgeGraph(nodes=nodes, edges=edges, anomalies=anomalies)


def only_sources(kg: KnowledgeGraph, sources: Iterable[EdgeSource]) -> KnowledgeGraph:
    """The same graph, keeping only the edges from these sources.

    For ablations, and for reporting the KG by source (R-12). All nodes are kept, so the labels
    and the condition list do not depend on which sources are switched on.
    """
    keep = frozenset(sources)
    return KnowledgeGraph(
        nodes=dict(kg.nodes),
        edges=[e for e in kg.edges if e.source in keep],
        anomalies=list(kg.anomalies),
    )
