"""EXP-002 — class balance of the 13 chest-pain conditions in DDXPlus.

Answers risk R-03 (docs/07-risk-register.md): are any in-scope conditions too
rare to learn? The R-03 trigger is **< 500 training cases** for a condition.

Runs on the VALIDATE split only (decision D-1). Training counts are projected by
the exact train/validate size ratio; the real counts arrive when train.csv is
downloaded. The test split is never read — see docs/05-evaluation-protocol.md §2.

Also checks that every evidence token in the patient rows can be decoded by
scripts/decode_ddxplus.py, and classifies the ones that cannot.

    python scripts/class_balance.py
"""

from __future__ import annotations

import ast
import contextlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditions import from_ddxplus_label  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "ddxplus"
INTERIM = REPO_ROOT / "data" / "interim"

# Exact split sizes from the Hugging Face datasets-server API, 2026-09-18.
SPLIT_ROWS = {"train": 1_025_602, "validate": 132_448, "test": 134_529}
TRAIN_PER_VALIDATE = SPLIT_ROWS["train"] / SPLIT_ROWS["validate"]

R03_MIN_TRAIN_CASES = 500
NUMERIC_VALUE = re.compile(r"^(E_\d+)_@_(\d+)$")


def load_validate() -> pd.DataFrame:
    path = RAW / "validate.csv"
    if "test" in path.name:  # belt-and-braces: this script must never read the test split
        raise SystemExit("Refusing to read the test split before Phase 4.")
    if not path.exists():
        raise SystemExit(f"Missing {path} — download it first (decision D-1).")
    return pd.read_csv(path)


def load_json(name: str) -> dict:
    path = INTERIM / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python scripts/decode_ddxplus.py")
    return json.loads(path.read_text(encoding="utf-8"))


def classify_tokens(rows: pd.Series, vocabulary: dict) -> tuple[Counter, Counter]:
    """Count evidence tokens as decodable, numeric-ordinal, or unknown."""
    kinds: Counter = Counter()
    unknown: Counter = Counter()
    for raw in rows:
        for token in ast.literal_eval(raw):
            if token in vocabulary:
                kinds["decoded"] += 1
                continue
            numeric = NUMERIC_VALUE.match(token)
            if numeric and numeric.group(1) in vocabulary:
                kinds["numeric_ordinal"] += 1  # e.g. pain intensity E_56_@_4
                continue
            kinds["unknown"] += 1
            unknown[token] += 1
    return kinds, unknown


def main() -> int:
    # Windows consoles default to cp1252, which cannot print ≈ or ⚠ (CLAUDE.md gotcha).
    with contextlib.suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    df = load_validate()
    vocabulary = load_json("ddxplus_evidences.json")
    conditions_meta = load_json("ddxplus_chestpain_conditions.json")

    df["condition"] = df["PATHOLOGY"].map(lambda p: from_ddxplus_label(p))
    in_scope = df[df["condition"].notna()].copy()
    in_scope["condition_id"] = in_scope["condition"].map(lambda c: c.id)
    in_scope["label"] = in_scope["condition"].map(lambda c: c.label)

    total, n_scope = len(df), len(in_scope)
    print("=" * 96)
    print("  EXP-002 — class balance, DDXPlus VALIDATE split (chest-pain scope)")
    print("=" * 96)
    print(f"Validate rows          : {total:,}")
    print(f"In scope (13 conds)    : {n_scope:,}  ({n_scope / total:.1%} of validate)")
    print(f"Train/validate ratio   : {TRAIN_PER_VALIDATE:.3f}  (exact split sizes)")
    print(f"Projected in-scope train: ≈{n_scope * TRAIN_PER_VALIDATE:,.0f}")

    rows = []
    for cid, group in in_scope.groupby("condition_id"):
        meta = conditions_meta.get(cid, {})
        count = len(group)
        rows.append(
            {
                "condition_id": cid,
                "label": group["label"].iloc[0],
                "validate": count,
                "share": count / n_scope,
                "train_projected": round(count * TRAIN_PER_VALIDATE),
                "pct_female": (group["SEX"] == "F").mean(),
                "median_age": float(group["AGE"].median()),
                "must_not_miss": bool(meta.get("is_must_not_miss")),
                "severity": meta.get("severity"),
            }
        )
    rows.sort(key=lambda r: r["validate"], reverse=True)

    print("\n" + "-" * 96)
    print(
        f"{'condition':<28} {'validate':>8} {'share':>7} {'train≈':>9} "
        f"{'%F':>6} {'age':>5} {'sev':>4}  {'MNM':<4} R-03"
    )
    print("-" * 96)
    flagged = []
    for r in rows:
        low = r["train_projected"] < R03_MIN_TRAIN_CASES
        if low:
            flagged.append(r["label"])
        print(
            f"{r['label']:<28} {r['validate']:>8,} {r['share']:>6.1%} {r['train_projected']:>9,} "
            f"{r['pct_female']:>5.0%} {r['median_age']:>5.0f} {str(r['severity'] or '-'):>4}  "
            f"{'YES' if r['must_not_miss'] else '':<4} {'⚠ LOW' if low else 'ok'}"
        )

    counts = [r["validate"] for r in rows]
    imbalance = max(counts) / min(counts)
    print("-" * 96)
    print(f"Imbalance (largest / smallest class): {imbalance:.1f}×")
    print(
        f"R-03 (< {R03_MIN_TRAIN_CASES} projected training cases): "
        f"{', '.join(flagged) if flagged else 'no condition triggers it'}"
    )

    kinds, unknown = classify_tokens(in_scope["EVIDENCES"], vocabulary)
    n_tokens = sum(kinds.values())
    print("\n" + "-" * 96)
    print("Evidence-token decodability (in-scope rows)")
    print("-" * 96)
    for kind in ("decoded", "numeric_ordinal", "unknown"):
        print(f"  {kind:<16} {kinds[kind]:>10,}  {kinds[kind] / n_tokens:>6.1%}")
    if unknown:
        print("  most frequent unknown tokens:", ", ".join(t for t, _ in unknown.most_common(8)))

    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / "exp002_class_balance.json"
    out.write_text(
        json.dumps(
            {
                "split": "validate",
                "validate_rows": total,
                "in_scope_rows": n_scope,
                "train_per_validate": TRAIN_PER_VALIDATE,
                "imbalance_ratio": imbalance,
                "r03_flagged": flagged,
                "conditions": rows,
                "token_kinds": dict(kinds),
                "top_unknown_tokens": unknown.most_common(20),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
