"""Build the cardiac knowledge graph and print its card (task 1b).

    python scripts/build_cardiac_kg.py

Loads data/interim/ddxplus_chestpain_conditions.json and ddxplus_evidences.json (run
scripts/decode_ddxplus.py first), builds the NetworkX store over the whole cardiac KG (DDXPlus
plus the hand-authored facts, src/medical_kg/cardiac_kg.py), and reports what the graph contains
and what it cannot do:

- edges by source, overall and per condition, for the R-12 circularity disclosure;
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
from src.medical_kg.cardiac_kg import build_cardiac_kg  # noqa: E402
from src.medical_kg.crosswalk import case_from_ddxplus  # noqa: E402
from src.medical_kg.loader import EdgeSource, only_sources  # noqa: E402
from src.medical_kg.networkx_store import NetworkXGraphStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERIM = REPO_ROOT / "data" / "interim"
AORTIC_DISSECTION = "COND:aortic_dissection"

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
        out_edges = list(graph.out_edges(cid, data=True))
        relations = Counter(data["relation"] for _, _, data in out_edges)
        conditions.append(
            {
                "condition_id": cid,
                "symptoms": relations["HAS_SYMPTOM"],
                "risk_factors": relations["HAS_RISK_FACTOR"],
                "edges_by_source": dict(Counter(data["source"] for _, _, data in out_edges)),
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


def graph_only_ranking(stores: dict[str, NetworkXGraphStore], parquet: Path) -> dict:
    """Rank every validate patient by graph score alone, once per store.

    Each patient becomes a case with :func:`case_from_ddxplus`, which reads the input columns
    only; the label is read here just to score the ranking. Ties are broken at random, in
    expectation: a condition tied with t others for the places after a higher-scoring ones
    counts as top-k with probability (k - a) / t, clamped to [0, 1].

    Returns, per store: expected top-1 and top-3 accuracy overall and per condition, and how
    often aortic dissection, which no DDXPlus patient has, would take first place or a top-3 place.
    """
    import pandas as pd  # local import: only this optional section needs pandas

    columns = ["case_id", "age", "sex", "evidences", "positive_codes", "label_condition_id"]
    records = pd.read_parquet(parquet, columns=columns).to_dict("records")
    result = {}
    for name, store in stores.items():
        labels = dict(store.graph.nodes(data="label"))
        totals: Counter = Counter()
        per_condition: dict[str, Counter] = {}
        for record in records:
            scores = store.score_by_connectivity(case_from_ddxplus(record, {}, labels))
            truth = record["label_condition_id"]
            hits = {k: _expected_in_top(scores, truth, k) for k in (1, 3)}
            totals.update(
                top1=hits[1],
                top3=hits[3],
                dissection_first=_expected_in_top(scores, AORTIC_DISSECTION, 1),
                dissection_top3=_expected_in_top(scores, AORTIC_DISSECTION, 3),
            )
            per_condition.setdefault(truth, Counter()).update(n=1, top1=hits[1])
        n = len(records)
        result[name] = {
            "patients": n,
            **{key: round(value / n, 3) for key, value in totals.items()},
            "top1_by_condition": {
                cid: round(c["top1"] / c["n"], 3) for cid, c in sorted(per_condition.items())
            },
        }
    return result


def _expected_in_top(scores: dict[str, float], target: str, k: int) -> float:
    """Probability that ``target`` ranks in the top k when ties are broken at random."""
    score = scores[target]
    above = sum(value > score for value in scores.values())
    tied = sum(value == score for value in scores.values())
    return min(max((k - above) / tied, 0.0), 1.0)


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

    kg = build_cardiac_kg(*paths)
    store = NetworkXGraphStore(kg)
    summary = card(store)

    print(f"Nodes      : {summary['nodes']}")
    print(f"Edges      : {summary['edges']}  {summary['edges_by_relation']}")
    print(f"By source  : {summary['edges_by_source']}   (R-12: report the KG by source)")
    print(f"No edges   : {', '.join(summary['conditions_without_edges']) or '—'}")
    print(f"Degree     : {summary['degree_histogram']}  (conditions sharing a question -> count)")
    print(f"Questions shared by ≥{LOW_DISCRIMINATION_SHARE:.0%} of conditions:")
    for concept, n, label in summary["low_discrimination"]:
        print(f"  {concept:<10} {n:>2}  {label[:60]}")
    print("\nPer condition (sx = symptom edges, rf = risk-factor edges, then edges by source):")
    for row in summary["conditions"]:
        nearest = BY_ID[row["nearest"]].label if row["nearest"] else "—"
        sources = ", ".join(f"{s} {n}" for s, n in sorted(row["edges_by_source"].items()))
        print(
            f"  {BY_ID[row['condition_id']].label:<26} sx {row['symptoms']:>2}  "
            f"rf {row['risk_factors']:>2}  unique {row['unique_to_it']:>2}  nearest "
            f"{nearest} (J={row['nearest_jaccard']})  [{sources}]"
        )
    print(f"\nSelf-retrieval ties : {summary['self_retrieval_ties'] or 'none'}")
    print(f"Nested evidence sets: {summary['nested_evidence_sets'] or 'none'}")
    for anomaly in summary["anomalies"]:
        print(f"ANOMALY: {anomaly}")

    parquet = args.interim / "ddxplus_chestpain_validate.parquet"
    if parquet.exists():
        stores = {
            "ddxplus only": NetworkXGraphStore(only_sources(kg, [EdgeSource.DDXPLUS])),
            "all sources": store,
        }
        summary["graph_only_ranking_validate"] = ranking = graph_only_ranking(stores, parquet)
        print("\nRanking validate patients by graph score alone (R-12: DDXPlus generated them")
        print("from the same condition definitions the graph is built from):")
        for name, row in ranking.items():
            worst = sorted(row["top1_by_condition"].items(), key=lambda kv: kv[1])[:2]
            print(
                f"  {name:<13} top-1 {row['top1']:.3f}  top-3 {row['top3']:.3f}  aortic dissection "
                f"first {row['dissection_first']:.1%}, top 3 {row['dissection_top3']:.1%}  "
                f"weakest: {', '.join(f'{BY_ID[c].label} {v:.2f}' for c, v in worst)}"
            )

    out = args.interim / "cardiac_kg_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")

    if summary["trainable_without_edges"]:
        print(f"ERROR: trainable conditions without evidence: {summary['trainable_without_edges']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
