"""Build the cardiac knowledge graph and print its card (task 1b).

    python scripts/build_cardiac_kg.py

Loads data/interim/ddxplus_chestpain_conditions.json and ddxplus_evidences.json (run
scripts/decode_ddxplus.py first), builds the NetworkX store, and reports what the graph
contains and what it cannot do:

- edges by source, for the R-12 circularity disclosure;
- evidence questions shared by most conditions, which barely discriminate;
- self-retrieval: give the store a case holding exactly one condition's expected
  findings. That condition always scores 1.0, so a *tie* marks a condition whose whole
  evidence set sits inside another's: the question-level graph cannot separate them.

Writes data/interim/cardiac_kg_summary.json. Exits 1 if a trainable condition has no
DDXPlus evidence at all.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditions import BY_ID, CONDITIONS  # noqa: E402
from src.contracts import Finding, PatientCase  # noqa: E402
from src.medical_kg.networkx_store import NetworkXGraphStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERIM = REPO_ROOT / "data" / "interim"

LOW_DISCRIMINATION_SHARE = 0.75
"""A question linked to at least this share of the trainable conditions barely helps rank."""


def card(store: NetworkXGraphStore) -> dict:
    """Everything the KG card in docs/02 §5.1 reports."""
    graph = store.graph
    node_types = Counter(data["type"] for _, data in graph.nodes(data=True))
    edges = list(graph.edges(data=True))

    expected = {c.id: {f.concept_id for f in store.expected_findings(c.id)} for c in CONDITIONS}
    with_edges = [cid for cid, concepts in expected.items() if concepts]
    degree = Counter(concept for cid in with_edges for concept in expected[cid])
    threshold = LOW_DISCRIMINATION_SHARE * len(with_edges)

    conditions = []
    for cid in with_edges:
        others = [o for o in with_edges if o != cid]
        nearest = max(others, key=lambda o: _jaccard(expected[cid], expected[o]), default=None)
        relations = Counter(data["relation"] for _, _, data in graph.out_edges(cid, data=True))
        conditions.append(
            {
                "condition_id": cid,
                "symptoms": relations["HAS_SYMPTOM"],
                "risk_factors": relations["HAS_RISK_FACTOR"],
                "unique_to_it": sum(1 for c in expected[cid] if degree[c] == 1),
                "nearest": nearest,
                "nearest_jaccard": (
                    round(_jaccard(expected[cid], expected[nearest]), 2) if nearest else None
                ),
            }
        )

    ties = {}
    for cid in with_edges:
        case = PatientCase(
            case_id=f"self-{cid}",
            age=50,
            sex="M",
            findings=[Finding(concept_id=c, label=c) for c in sorted(expected[cid])],
        )
        scores = store.score_by_connectivity(case)
        tied = sorted(o for o in with_edges if o != cid and scores[o] >= scores[cid])
        if tied:
            ties[cid] = tied

    subsets = [
        [a, b]
        for a, b in combinations(with_edges, 2)
        if expected[a] <= expected[b] or expected[b] <= expected[a]
    ]

    return {
        "nodes": dict(node_types),
        "edges": len(edges),
        "edges_by_relation": dict(Counter(d["relation"] for _, _, d in edges)),
        "edges_by_source": dict(Counter(d["source"] for _, _, d in edges)),
        "conditions_without_edges": [c.id for c in CONDITIONS if not expected[c.id]],
        "trainable_without_edges": [
            c.id for c in CONDITIONS if c.in_training_data and not expected[c.id]
        ],
        "degree_histogram": dict(sorted(Counter(degree.values()).items())),
        "low_discrimination": sorted(
            (
                [concept, n, graph.nodes[concept]["label"]]
                for concept, n in degree.items()
                if n >= threshold
            ),
            key=lambda row: (-row[1], row[0]),
        ),
        "conditions": conditions,
        "self_retrieval_ties": ties,
        "nested_evidence_sets": subsets,
        "anomalies": store.anomalies,
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interim", type=Path, default=INTERIM)
    args = parser.parse_args(argv)

    paths = [
        args.interim / "ddxplus_chestpain_conditions.json",
        args.interim / "ddxplus_evidences.json",
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {missing}. Run: python scripts/decode_ddxplus.py")

    store = NetworkXGraphStore.from_files(*paths)
    summary = card(store)

    print(f"Nodes      : {summary['nodes']}")
    print(f"Edges      : {summary['edges']}  {summary['edges_by_relation']}")
    print(f"By source  : {summary['edges_by_source']}   (R-12: report the KG by source)")
    print(f"No edges   : {', '.join(summary['conditions_without_edges']) or '—'}")
    print(f"Degree     : {summary['degree_histogram']}  (conditions sharing a question -> count)")
    print(f"Questions shared by ≥{LOW_DISCRIMINATION_SHARE:.0%} of conditions:")
    for concept, n, label in summary["low_discrimination"]:
        print(f"  {concept:<10} {n:>2}  {label[:60]}")
    print("\nPer condition (sx = symptoms, rf = risk factors):")
    for row in summary["conditions"]:
        nearest = BY_ID[row["nearest"]].label if row["nearest"] else "—"
        print(
            f"  {BY_ID[row['condition_id']].label:<26} sx {row['symptoms']:>2}  "
            f"rf {row['risk_factors']:>2}  unique {row['unique_to_it']:>2}  nearest "
            f"{nearest} (J={row['nearest_jaccard']})"
        )
    print(f"\nSelf-retrieval ties : {summary['self_retrieval_ties'] or 'none'}")
    print(f"Nested evidence sets: {summary['nested_evidence_sets'] or 'none'}")
    for anomaly in summary["anomalies"]:
        print(f"ANOMALY: {anomaly}")

    out = args.interim / "cardiac_kg_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")

    if summary["trainable_without_edges"]:
        print(f"ERROR: trainable conditions without evidence: {summary['trainable_without_edges']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
