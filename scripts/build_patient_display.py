"""1d: Synthea's CSV output -> one small display card per patient, for the dashboard's history panel.

    python scripts/build_patient_display.py
    python scripts/build_patient_display.py --synthea <csv folder> --as-of 2026-01-01

Reads Synthea's ``csv/`` folder (src/patient_kg/synthea.py), builds each patient's graph as of
``--as-of`` (src/patient_kg/graph.py) and writes ``patients.json`` under ``--out``: one card per
patient alive on the day, holding only what the dashboard's "Patient history (Synthea, display
only)" panel shows (app/view.py):

* a short id (the first 8 characters of Synthea's UUID), age and sex on the day;
* the active conditions (description, since), disorders first and then the findings and social
  history Synthea lists among them (flagged ``finding``), each oldest first;
* the risk-factor concepts the history supports (label, present or absent, since);
* the active medications, **descriptions only** (history, never advice);
* the counts of procedures and observations on record by the day;
* the active conditions the SNOMED table does not map (descriptions only).

Every list is capped (:data:`CAPS`), with its full length kept beside it so the panel can say "and
N more"; the file is a few MB. **Display only:** the card feeds nothing, neither the ranking, nor a
case's findings, nor its risk factors (decision of 2026-10-20, PROGRESS §6; the panel is tested for
it). The script prints counts, never patient rows. Synthetic patients only; ``--out`` defaults to
data/interim/, which is never committed. ``scripts/build_patient_kg.py`` is the builder of the
findings and is left as it is.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.patient_kg.graph import (  # noqa: E402
    PATIENT,
    build_patient_graph,
    patient_findings,
    unmapped_codes,
)
from src.patient_kg.mapping import SYNTHEA_VERSION  # noqa: E402
from src.patient_kg.synthea import PatientRecord, read_synthea  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHEA = REPO_ROOT / "data" / "raw" / "synthea_patient_kg" / "synthea_patient_kg" / "csv"
"""Where the Kaggle run's output sits once unzipped (note the doubled folder)."""
DEFAULT_OUT = REPO_ROOT / "data" / "interim" / "patient_display"
DEFAULT_AS_OF = date(2026, 1, 1)

SHORT_ID = 8
"""Characters of Synthea's patient UUID kept: enough to tell cards apart, not an identifier."""
CAPS = {"conditions": 12, "medications": 10, "unmapped": 12, "risk_factors": 12}
"""The longest list a card holds, by name. The risk factors never reach it (six concepts have
codes), but a card must not grow with a table that does."""
TEXT_LIMIT = 120
"""The longest description kept, in characters."""
NOT_A_DISORDER = {
    "finding",
    "situation",
    "procedure",
    "event",
    "regime/therapy",
    "observable entity",
    "person",
    "social concept",
    "environment",
    "morphologic abnormality",
}
"""SNOMED semantic tags that mark an entry as social history or a finding, not a disease. Synthea
lists them among the active conditions ("Full-time employment (finding)"); the panel keeps them
apart from the disorders and draws the disorders first."""
_TAG = re.compile(r"\(([^()]*)\)\s*$")


def is_finding(description: str | None) -> bool:
    """True when the description ends in a semantic tag that is not a disorder. Read on the full
    description, before it is clipped, so a long one does not lose its tag."""
    match = _TAG.search((description or "").strip())
    return bool(match) and match.group(1).strip().lower() in NOT_A_DISORDER


def _clip(text: str | None) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= TEXT_LIMIT else text[: TEXT_LIMIT - 1] + "…"


def _once(rows: list[dict], key: str) -> list[dict]:
    """``rows`` without a second row of the same ``key``, the first (the oldest) kept."""
    seen: set[str] = set()
    out = []
    for row in rows:
        if row[key] in seen:
            continue
        seen.add(row[key])
        out.append(row)
    return out


def card_of(record: PatientRecord, as_of: date) -> dict:
    """One patient's display card as of ``as_of`` (see the module docstring)."""
    graph = build_patient_graph(record, as_of)
    nodes = [data for _, data in graph.nodes(data=True)]
    patient = graph.nodes[PATIENT]

    conditions = sorted(
        (
            {
                "description": _clip(d["description"]),
                "since": d["start"],
                "code": d["code"],
                "finding": is_finding(d["description"]),
            }
            for d in nodes
            if d.get("kind") == "condition" and d.get("active")
        ),
        # Disorders first, then findings, each oldest first: a card cut at its cap keeps the
        # diseases and loses the social history.
        key=lambda row: (row["finding"], row["since"] or "", row["description"]),
    )
    conditions = _once(conditions, "code")
    medications = [
        _clip(d["description"])
        for d in sorted(
            (d for d in nodes if d.get("kind") == "medication" and d.get("active")),
            key=lambda d: (d["start"] or "", d["description"]),
        )
    ]
    medications = list(dict.fromkeys(medications))
    unmapped = list(dict.fromkeys(_clip(row["description"]) for row in unmapped_codes(graph)))
    risk_factors = [
        {
            "label": _clip(f.label),
            "present": f.assertion.value == "present",
            "since": f.qualifiers.get("since"),
        }
        for f in patient_findings(graph)
    ]

    def on_record(kind: str) -> int:
        return sum(1 for e in record.entries if e.kind == kind and (e.start or as_of) <= as_of)

    return {
        "id": record.patient_id[:SHORT_ID],
        "age": patient["age"],
        "sex": patient["sex"],
        "as_of": as_of.isoformat(),
        "conditions": [
            {
                "description": r["description"],
                "since": r["since"],
                **({"finding": True} if r["finding"] else {}),
            }
            for r in conditions[: CAPS["conditions"]]
        ],
        "conditions_total": len(conditions),
        "risk_factors": risk_factors[: CAPS["risk_factors"]],
        "medications": medications[: CAPS["medications"]],
        "medications_total": len(medications),
        "procedures": on_record("procedure"),
        "observations": on_record("observation"),
        "unmapped": unmapped[: CAPS["unmapped"]],
        "unmapped_total": len(unmapped),
    }


def build(csv_dir: Path, as_of: date, out: Path) -> dict:
    """Write ``patients.json`` for every patient alive on ``as_of``; the counts (never rows)."""
    records = read_synthea(csv_dir)
    cards = []
    for patient_id in sorted(records):
        if records[patient_id].alive_on(as_of):
            cards.append(card_of(records[patient_id], as_of))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "patients.json"
    path.write_text(
        json.dumps(
            {
                "synthea_version": SYNTHEA_VERSION,
                "as_of": as_of.isoformat(),
                "caps": CAPS,
                "patients": cards,
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    short_ids = [c["id"] for c in cards]
    return {
        "path": path,
        "bytes": path.stat().st_size,
        "as_of": as_of.isoformat(),
        "read": len(records),
        "written": len(cards),
        "repeated_short_ids": len(short_ids) - len(set(short_ids)),
        "with_a_list_cut": sum(
            1
            for c in cards
            if c["conditions_total"] > len(c["conditions"])
            or c["medications_total"] > len(c["medications"])
            or c["unmapped_total"] > len(c["unmapped"])
        ),
        "present": sum(1 for c in cards for r in c["risk_factors"] if r["present"]),
        "conditions": sum(len(c["conditions"]) for c in cards),
        "medications": sum(len(c["medications"]) for c in cards),
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--synthea",
        type=Path,
        default=DEFAULT_SYNTHEA,
        help="Synthea's csv/ folder (default: the Kaggle run's output under data/raw/)",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=DEFAULT_AS_OF,
        help="the day the cards describe, YYYY-MM-DD (default: 2026-01-01)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        counts = build(args.synthea, args.as_of, args.out)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    print(f"{counts['read']} patients read, {counts['written']} alive on {counts['as_of']}")
    print(
        f"  listed: {counts['conditions']} conditions, {counts['medications']} medications "
        f"(descriptions only), {counts['present']} risk factors present"
    )
    print(
        f"  {counts['with_a_list_cut']} card(s) have a list cut at its cap {CAPS}; "
        f"{counts['repeated_short_ids']} short id(s) repeated"
    )
    print(f"Wrote {counts['path']} ({counts['bytes'] / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
