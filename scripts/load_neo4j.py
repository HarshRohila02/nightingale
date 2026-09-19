"""Load the cardiac knowledge graph into Neo4j and check it reads back identically (task 1b).

    python scripts/load_neo4j.py            # write the graph if Neo4j's copy differs, then check
    python scripts/load_neo4j.py --check    # check only; exit 1 if Neo4j's copy is out of date

Builds the graph from local files, as scripts/build_cardiac_kg.py does (DDXPlus, the
hand-authored aortic dissection and, when data/raw/bodhi_s is there, BODHI-S). It then connects
with the settings in .env (docs/11-compute-runbook.md §5), writes the graph when the stored copy's
fingerprint differs, and reads it back. Last, it runs the golden cases on both the Neo4j store and
the in-memory store and requires identical scores, paths and expected findings.

Without the local files it only reports what Neo4j holds. It never prints a credential.

Exit codes: 0 all good · 1 out of date (--check) or the two stores disagree · 2 Neo4j is not
configured or cannot be reached, or there is no graph at all.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402
from neo4j.exceptions import DriverError, Neo4jError  # noqa: E402

from src.conditions import CONDITIONS  # noqa: E402
from src.contracts import PatientCase  # noqa: E402
from src.medical_kg.bodhi_s import TRIPLES_FILE  # noqa: E402
from src.medical_kg.cardiac_kg import build_cardiac_kg  # noqa: E402
from src.medical_kg.crosswalk import expand_case  # noqa: E402
from src.medical_kg.neo4j_store import (  # noqa: E402
    DEFAULT_NAMESPACE,
    DEFAULT_TIMEOUT_SECONDS,
    GraphOutOfDate,
    GraphUnavailable,
    Neo4jGraphStore,
    describe_failure,
    settings_from_env,
)
from src.medical_kg.networkx_store import NetworkXGraphStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERIM = REPO_ROOT / "data" / "interim"
BODHI_DIR = REPO_ROOT / "data" / "raw" / "bodhi_s"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"


def disagreements(store: NetworkXGraphStore, reference: NetworkXGraphStore) -> list[str]:
    """Where the two stores answer the golden cases differently. Empty means identical."""
    labels = dict(reference.graph.nodes(data="label"))
    problems: list[str] = []
    for golden in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]:
        case = expand_case(PatientCase(**golden["case"]), labels)
        if store.score_by_connectivity(case) != reference.score_by_connectivity(case):
            problems.append(f"{golden['id']}: scores differ")
        for condition in CONDITIONS:
            if store.paths_for(case, condition.id) != reference.paths_for(case, condition.id):
                problems.append(f"{golden['id']}: paths for {condition.id} differ")
    for condition in CONDITIONS:
        if store.expected_findings(condition.id) != reference.expected_findings(condition.id):
            problems.append(f"expected findings for {condition.id} differ")
    return problems


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--interim", type=Path, default=INTERIM)
    parser.add_argument("--bodhi-dir", type=Path, default=BODHI_DIR)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--check", action="store_true", help="never write; exit 1 if Neo4j's copy is out of date"
    )
    args = parser.parse_args(argv)

    paths = (
        args.interim / "ddxplus_chestpain_conditions.json",
        args.interim / "ddxplus_evidences.json",
    )
    local_kg = None
    if all(p.exists() for p in paths):
        bodhi_dir = args.bodhi_dir if (args.bodhi_dir / TRIPLES_FILE).exists() else None
        local_kg = build_cardiac_kg(*paths, bodhi_dir)
        print(f"Local graph: built from {args.interim}" + (" and BODHI-S" if bodhi_dir else ""))
    else:
        print(f"Local graph: none ({paths[0].name} is missing); reporting what Neo4j holds")

    try:
        settings = settings_from_env()
        store = Neo4jGraphStore.connect(
            settings,
            local_kg=local_kg,
            sync=not args.check,
            namespace=args.namespace,
            timeout=args.timeout,
        )
    except GraphOutOfDate as exc:
        print(f"Neo4j      : {exc}")
        return 1
    except GraphUnavailable as exc:
        print(f"Neo4j      : {exc}")
        return 2
    except (Neo4jError, DriverError, OSError) as exc:
        print(f"Neo4j      : unavailable. {describe_failure(exc)}")
        return 2

    database = f"database {settings.database}" if settings.database else "home database"
    node_types = Counter(t for _, t in store.graph.nodes(data="type"))
    edge_sources = Counter(s for _, _, s in store.graph.edges(data="source"))
    print(f"Neo4j      : {store.server}, {database}, namespace {store.namespace}")
    print(
        f"Graph      : {store.graph.number_of_nodes()} nodes ("
        + ", ".join(f"{n} {t}" for t, n in sorted(node_types.items()))
        + f"), {store.graph.number_of_edges()} edges ("
        + ", ".join(f"{n} {s}" for s, n in sorted(edge_sources.items()))
        + ")"
    )
    print(
        f"Fingerprint: {store.fingerprint[:12]}, written {store.written_at}"
        + (" by this run" if store.wrote else "; already current, nothing written")
    )
    if store.anomalies:
        print(f"Anomalies  : {len(store.anomalies)} (scripts/build_cardiac_kg.py lists them)")

    if local_kg is None:
        return 0
    problems = disagreements(store, NetworkXGraphStore(local_kg))
    if problems:
        print("Parity     : the Neo4j store and the in-memory store DISAGREE")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        "Parity     : identical to the in-memory graph on the golden cases "
        "(scores, paths, expected findings)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
