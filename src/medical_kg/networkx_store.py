"""NetworkX-backed ``GraphStore``: the in-memory backend (risk R-06).

Implements the ``GraphStore`` Protocol (src/contracts.py) over a :class:`KnowledgeGraph`
from :mod:`src.medical_kg.loader`. The architecture names it as the fallback for when
Neo4j is unavailable (docs/02-architecture.md §7), and it is the working backend until
Docker is up (decision D-6).

Scoring is the same weighted overlap as the stub ``InMemoryGraphStore``, which makes this
a drop-in replacement. Personalised PageRank arrives in 2a (EXP-005).

The graph is a ``MultiDiGraph`` keyed by edge source, so the same fact from DDXPlus and
from BODHI-S can sit side by side without one overwriting the other (R-12). The store is
immutable after construction: to change the graph, build a new store.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.contracts import Assertion, Finding, PatientCase, ReasoningPath
from src.ddxplus import CONCEPT_PREFIX, code_sort_key
from src.medical_kg.loader import KnowledgeGraph, NodeType, Relation, load_ddxplus_kg

__all__ = ["NetworkXGraphStore"]

DENIED_PENALTY = 0.5
"""Same as InMemoryGraphStore: an explicitly denied expected finding costs half a match."""


class NetworkXGraphStore:
    """The cardiac KG in memory. Implements ``GraphStore``."""

    def __init__(self, kg: KnowledgeGraph) -> None:
        graph = nx.MultiDiGraph()
        for node in kg.nodes.values():
            graph.add_node(node.id, type=node.type.value, label=node.label, **node.properties)
        for edge in kg.edges:
            graph.add_edge(
                edge.condition_id,
                edge.concept_id,
                key=edge.source.value,
                relation=edge.relation.value,
                source=edge.source.value,
                weight=edge.weight,
                **edge.properties,
            )
        self.graph = nx.freeze(graph)
        self.anomalies = list(kg.anomalies)
        self._conditions = [
            n for n, data in graph.nodes(data=True) if data["type"] == NodeType.CONDITION.value
        ]
        self._expected = {cid: self._collect_expected(cid) for cid in self._conditions}

    @classmethod
    def from_files(cls, conditions_path: Path, vocabulary_path: Path) -> NetworkXGraphStore:
        """Load the DDXPlus-derived KG from the interim files and build the store."""
        return cls(load_ddxplus_kg(conditions_path, vocabulary_path))

    # -- GraphStore -------------------------------------------------------- #

    def score_by_connectivity(self, case: PatientCase) -> dict[str, float]:
        """Weighted overlap between the case and each condition's expected findings.

        ``(matched - 0.5 * denied) / total``, clamped at 0, where each term is a sum of
        edge weights. A condition with no edges scores 0; that is aortic dissection until
        its edges are hand-authored.
        """
        present = case.present_concept_ids()
        absent = case.absent_concept_ids()
        scores: dict[str, float] = {}
        for condition_id in self._conditions:
            expected = self._expected[condition_id]
            total = sum(weight for _, weight in expected.values())
            if total <= 0:
                scores[condition_id] = 0.0
                continue
            matched = sum(w for cid, (_, w) in expected.items() if cid in present)
            denied = sum(w for cid, (_, w) in expected.items() if cid in absent)
            scores[condition_id] = max(0.0, (matched - DENIED_PENALTY * denied) / total)
        return scores

    def paths_for(self, case: PatientCase, condition_id: str) -> list[ReasoningPath]:
        """One path per present finding or risk factor that the condition expects."""
        expected = self._expected.get(condition_id, {})
        if not expected:
            return []
        condition_label = self.graph.nodes[condition_id]["label"]
        paths: list[ReasoningPath] = []
        seen: set[str] = set()
        for finding in [*case.findings, *case.risk_factors]:
            cid = finding.concept_id
            if finding.assertion is not Assertion.PRESENT or cid not in expected or cid in seen:
                continue
            seen.add(cid)
            relation, weight = expected[cid]
            paths.append(
                ReasoningPath(
                    finding_id=cid,
                    condition_id=condition_id,
                    path=[finding.label, relation, condition_label],
                    weight=weight,
                )
            )
        return paths

    def expected_findings(self, condition_id: str) -> list[Finding]:
        """Symptoms first, then risk factors, each in code order. Drives 'missing' analysis."""
        expected = self._expected.get(condition_id, {})
        ordered = sorted(
            expected,
            key=lambda cid: (expected[cid][0] != Relation.HAS_SYMPTOM.value, _node_key(cid)),
        )
        return [
            Finding(
                concept_id=cid,
                label=self.graph.nodes[cid]["label"],
                assertion=Assertion.UNKNOWN,
            )
            for cid in ordered
        ]

    # -- internals --------------------------------------------------------- #

    def _collect_expected(self, condition_id: str) -> dict[str, tuple[str, float]]:
        """{concept id: (relation, weight)}. Parallel edges from different sources
        collapse to the strongest one, so a fact is never double-counted."""
        expected: dict[str, tuple[str, float]] = {}
        for _, concept, data in self.graph.out_edges(condition_id, data=True):
            previous = expected.get(concept)
            if previous is None or data["weight"] > previous[1]:
                expected[concept] = (data["relation"], data["weight"])
        return expected


def _node_key(concept_id: str) -> tuple[int, int, str]:
    """DDX:E_2 < DDX:E_10, then any other namespace alphabetically."""
    if concept_id.startswith(CONCEPT_PREFIX):
        return (0, code_sort_key(concept_id[len(CONCEPT_PREFIX) :]), "")
    return (1, 0, concept_id)
