"""Decode one DDXPlus split and keep the 13 in-scope chest-pain conditions (task 1a).

    python scripts/build_ddxplus_chestpain.py                  # the validate split
    python scripts/build_ddxplus_chestpain.py --split train    # once train.csv is downloaded

For split S this writes:

    data/interim/ddxplus_chestpain_S.parquet        one row per in-scope patient
    data/interim/ddxplus_chestpain_S.summary.json   counts, token kinds, label audit

Column roles (inputs / labels / metadata) are fixed in src/ddxplus.py and described in
docs/03-data-management.md §2.1. The test split is refused unless configs/config.yaml
sets ``evaluation.allow_test_split: true``, which happens once, in Phase 4.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ddxplus import (  # noqa: E402
    INPUT_COLUMNS,
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    SPLITS,
    TokenKind,
    check_split_allowed,
    decode_row,
    is_positive,
    load_evidence_specs,
    parse_token,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = pa.schema(
    [
        ("case_id", pa.string()),
        ("split", pa.string()),
        ("source_row", pa.int64()),
        ("initial_evidence", pa.string()),
        ("age", pa.int64()),
        ("sex", pa.string()),
        ("evidences", pa.list_(pa.string())),
        ("positive_codes", pa.list_(pa.string())),
        ("label_condition_id", pa.string()),
        ("label_pathology", pa.string()),
        (
            "label_differential",
            pa.list_(
                pa.struct(
                    [
                        ("pathology", pa.string()),
                        ("condition_id", pa.string()),
                        ("probability", pa.float64()),
                    ]
                )
            ),
        ),
    ]
)
assert SCHEMA.names == [*METADATA_COLUMNS, *INPUT_COLUMNS, *LABEL_COLUMNS]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarise(records: list[dict[str, Any]], source_rows: int, specs: dict) -> dict[str, Any]:
    """Counts and the label audit (EXP-013) for one decoded split."""
    kinds: Counter = Counter()
    defaults: Counter = Counter()
    rows_with_default = 0
    n_positive = []
    diff_size, diff_in_scope, oos_mass = [], [], []
    rows_with_oos = true_in_diff = true_first = 0
    recall5_full, recall5_in_scope, precision3 = [], [], []

    for rec in records:
        row_has_default = False
        for raw in rec["evidences"]:
            token = parse_token(raw)
            kinds[token.kind.value] += 1
            if token.kind is not TokenKind.BINARY and not is_positive(token, specs):
                defaults[raw] += 1
                row_has_default = True
        rows_with_default += row_has_default
        n_positive.append(len(rec["positive_codes"]))

        diff = rec["label_differential"]
        in_scope = [e for e in diff if e["condition_id"] is not None]
        total = sum(e["probability"] for e in diff) or 1.0
        diff_size.append(len(diff))
        diff_in_scope.append(len(in_scope))
        oos_mass.append(sum(e["probability"] for e in diff if e["condition_id"] is None) / total)
        rows_with_oos += len(in_scope) < len(diff)
        names = [e["pathology"] for e in diff]
        true_in_diff += rec["label_pathology"] in names
        true_first += bool(names) and names[0] == rec["label_pathology"]
        # Best scores a perfect closed-world system could reach: it can only ever list
        # in-scope conditions. docs/05 §3.1: Recall@5 = |top-5 ∩ D| / |D| and
        # Precision@3 = |top-3 ∩ D| / 3.
        if diff:
            recall5_full.append(min(5, len(in_scope)) / len(diff))
            precision3.append(min(3, len(in_scope)) / 3)
        if in_scope:
            recall5_in_scope.append(min(5, len(in_scope)) / len(in_scope))

    n = len(records)
    return {
        "split": records[0]["split"] if records else None,
        "source_rows": source_rows,
        "in_scope_rows": n,
        "per_condition": dict(Counter(r["label_condition_id"] for r in records).most_common()),
        "token_kinds": dict(kinds),
        "default_valued_tokens": {
            "total": sum(defaults.values()),
            "rows_with_any": rows_with_default,
            "most_common": defaults.most_common(10),
        },
        "positive_codes_per_row_mean": statistics.fmean(n_positive) if n else None,
        "differential": (
            {
                "size_mean": statistics.fmean(diff_size),
                "size_median": statistics.median(diff_size),
                "in_scope_entries_mean": statistics.fmean(diff_in_scope),
                "rows_with_out_of_scope_entries": rows_with_oos,
                "rows_with_out_of_scope_entries_share": rows_with_oos / n,
                "out_of_scope_mass_mean": statistics.fmean(oos_mass),
                "out_of_scope_mass_median": statistics.median(oos_mass),
                "out_of_scope_mass_p90": _percentile(oos_mass, 0.9),
                "true_pathology_in_differential_share": true_in_diff / n,
                "true_pathology_ranked_first_share": true_first / n,
                "recall_at_5_ceiling_full_differential": statistics.fmean(recall5_full),
                "recall_at_5_ceiling_in_scope_differential": statistics.fmean(recall5_in_scope),
                "precision_at_3_ceiling": statistics.fmean(precision3),
            }
            if n
            else None
        ),
    }


def build(split: str, raw_dir: Path, out_dir: Path, *, allow_test_split: bool) -> dict[str, Any]:
    """Decode ``raw_dir/<split>.csv`` and write the parquet and its summary to ``out_dir``."""
    check_split_allowed(split, allow_test_split=allow_test_split)  # before touching the file

    csv_path = raw_dir / f"{split}.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Download it first (decision D-1).")
    specs = load_evidence_specs(raw_dir / "release_evidences.json")
    raw = pd.read_csv(csv_path)

    records = []
    for row in raw.itertuples(index=True):
        record = decode_row(
            split=split,
            source_row=row.Index,
            age=row.AGE,
            sex=row.SEX,
            pathology=row.PATHOLOGY,
            evidences_cell=row.EVIDENCES,
            initial_evidence=row.INITIAL_EVIDENCE,
            differential_cell=row.DIFFERENTIAL_DIAGNOSIS,
            specs=specs,
        )
        if record is not None:
            records.append(record)

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"ddxplus_chestpain_{split}.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), parquet_path)

    summary = summarise(records, source_rows=len(raw), specs=specs)
    summary_path = out_dir / f"ddxplus_chestpain_{split}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["_paths"] = [str(parquet_path), str(summary_path)]
    return summary


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which cannot print every character used here
    # (CLAUDE.md gotcha). stderr too: the test-split refusal message contains "§".
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=SPLITS, default="validate")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    allow_test = bool((config.get("evaluation") or {}).get("allow_test_split", False))
    try:
        summary = build(args.split, args.raw_dir, args.out_dir, allow_test_split=allow_test)
    except PermissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    d = summary["differential"] or {}
    print(f"Split                  : {args.split}")
    print(f"Source rows            : {summary['source_rows']:,}")
    print(f"In scope (13 conds)    : {summary['in_scope_rows']:,}")
    print(f"Token kinds            : {summary['token_kinds']}")
    dv = summary["default_valued_tokens"]
    print(f"Tokens that mean 'no'  : {dv['total']:,} in {dv['rows_with_any']:,} rows")
    print(f"  most common          : {', '.join(f'{t} ({c:,})' for t, c in dv['most_common'][:4])}")
    if d:
        print(
            f"Differential (label)   : {d['size_mean']:.1f} entries on average, "
            f"{d['in_scope_entries_mean']:.1f} in scope"
        )
        print(
            f"  out-of-scope entries : in {d['rows_with_out_of_scope_entries_share']:.1%} of rows, "
            f"{d['out_of_scope_mass_mean']:.1%} of the probability mass on average"
        )
        print(
            f"  Recall@5 ceiling     : {d['recall_at_5_ceiling_full_differential']:.3f} "
            f"(full D) · {d['recall_at_5_ceiling_in_scope_differential']:.3f} (in-scope D)"
        )
        print(f"  Precision@3 ceiling  : {d['precision_at_3_ceiling']:.3f} (either D)")
    for path in summary["_paths"]:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
