"""Load the cardiac medical knowledge graph from DDXPlus (task 1b).

The KG backbone is DDXPlus ``release_conditions.json``, the R-01 fallback
(docs/10-spike-r01-crosswalk.md). scripts/decode_ddxplus.py has already filtered it to
our 13 conditions (``data/interim/ddxplus_chestpain_conditions.json``) and decoded the
evidence vocabulary (``data/interim/ddxplus_evidences.json``). This module turns those two
files into a backend-neutral :class:`KnowledgeGraph` of plain nodes and edges. Any
``GraphStore`` can be built from it: NetworkX now, Neo4j once Docker is running (D-6).

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
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.conditions import BY_ID, CONDITIONS
from src.ddxplus import concept_id

__all__ = [
    "EdgeSource",
    "KGEdge",
    "KGNode",
    "KnowledgeGraph",
    "NodeType",
    "Relation",
    "load_ddxplus_kg",
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


_NODE_TYPE_FOR: dict[Relation, NodeType] = {
    Relation.HAS_SYMPTOM: NodeType.SYMPTOM,
    Relation.HAS_RISK_FACTOR: NodeType.RISK_FACTOR,
}


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
