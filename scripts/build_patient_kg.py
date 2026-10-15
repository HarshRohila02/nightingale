"""1d: Synthea's CSV output -> one patient knowledge graph per patient, and the findings they yield.

    python scripts/build_patient_kg.py --synthea data/raw/synthea/csv --as-of 2026-01-01

Reads Synthea's ``csv/`` folder (src/patient_kg/synthea.py), builds each living patient's graph as
of ``--as-of`` (src/patient_kg/graph.py) and writes, under ``--out``:

* ``findings.jsonl``: one line per patient alive on the day: id, age, sex, and the ``RF:*``
  findings their history supports (``source="synthea"``);
* ``summary.json``: how many patients, how many have each concept (present or absent), and **every
  active condition code the table does not map**, with its description, the patients who have it
  and the reason, so that the table can be reviewed against what Synthea actually produced.

The notebook (notebooks/kaggle_synthea.ipynb) runs it on Kaggle right after Synthea; on the laptop
it reads the unzipped output. Synthetic patients only; ``--out`` defaults to data/interim/, which
is never committed. Nothing here feeds the ranking (docs/17 T7).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.patient_kg.graph import (  # noqa: E402
    PATIENT,
    build_patient_graph,
    patient_findings,
    unmapped_codes,
)
from src.patient_kg.mapping import CONCEPTS_WITHOUT_CODES, SYNTHEA_VERSION  # noqa: E402
from src.patient_kg.synthea import read_synthea  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def build(csv_dir: Path, as_of: date | None, out: Path) -> dict:
    """Build every patient's graph and write ``findings.jsonl`` and ``summary.json``; the summary."""
    records = read_synthea(csv_dir)
    if as_of is None:
        dates = [e.start for r in records.values() for e in r.entries if e.start]
        as_of = max(dates) if dates else date.today()
    out.mkdir(parents=True, exist_ok=True)
    present: Counter[str] = Counter()
    absent: Counter[str] = Counter()
    unmapped: dict[str, dict] = {}
    alive = 0
    with (out / "findings.jsonl").open("w", encoding="utf-8") as handle:
        for patient_id in sorted(records):
            record = records[patient_id]
            if not record.alive_on(as_of):
                continue
            alive += 1
            graph = build_patient_graph(record, as_of)
            findings = patient_findings(graph)
            for finding in findings:
                (present if finding.assertion.value == "present" else absent)[
                    finding.concept_id
                ] += 1
            for row in unmapped_codes(graph):
                entry = unmapped.setdefault(
                    row["code"], {**{k: row[k] for k in ("description", "reason")}, "patients": 0}
                )
                entry["patients"] += 1
            patient = graph.nodes[PATIENT]
            handle.write(
                json.dumps(
                    {
                        "patient_id": patient_id,
                        "age": patient["age"],
                        "sex": patient["sex"],
                        "findings": [f.model_dump(mode="json") for f in findings],
                    }
                )
                + "\n"
            )
    summary = {
        "synthea_version": SYNTHEA_VERSION,
        "as_of": as_of.isoformat(),
        "patients": len(records),
        "alive_on_as_of": alive,
        "concepts": {
            concept: {"present": present[concept], "absent": absent[concept]}
            for concept in sorted(set(present) | set(absent))
        },
        "concepts_without_codes": CONCEPTS_WITHOUT_CODES,
        "unmapped_active_conditions": dict(
            sorted(unmapped.items(), key=lambda item: (-item[1]["patients"], item[0]))
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthea", type=Path, required=True, help="Synthea's csv/ folder")
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="the day the graphs describe, YYYY-MM-DD (default: the latest date in the data)",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "interim" / "patient_kg")
    args = parser.parse_args(argv)
    try:
        summary = build(args.synthea, args.as_of, args.out)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    print(
        f"{summary['patients']} patients, {summary['alive_on_as_of']} alive on "
        f"{summary['as_of']} (Synthea {summary['synthea_version']})"
    )
    for concept, counts in summary["concepts"].items():
        print(f"  {concept:<26} present {counts['present']:>6}  absent {counts['absent']:>6}")
    unmapped = summary["unmapped_active_conditions"]
    print(f"\n{len(unmapped)} active condition code(s) the table does not map:")
    for code, row in unmapped.items():
        print(f"  {code:<17} {row['patients']:>6}  {row['description']}  [{row['reason']}]")
    print(f"\nWrote {args.out / 'findings.jsonl'} and {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
