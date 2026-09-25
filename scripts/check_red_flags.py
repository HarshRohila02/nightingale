"""EXP-008: the red-flag rules on DDXPlus validate patients (task 2d).

    python scripts/check_red_flags.py

Each validate patient's evidence becomes hand-authored concepts through the crosswalk
(``concepts_from_evidences``), and every rule in src/reasoning/red_flags.py runs on them. Reports:

1. Per rule: the share of its own condition's patients it flags, and of everyone else (R-15).
2. **The false-alarm burden**: the share of patients whose true condition is *not* must-not-miss
   who get any flag. Half of validate patients have a must-not-miss condition, so the share of
   all patients flagged cannot show alarm fatigue: a perfect rule set would flag half of them.
3. docs/05 red-flag sensitivity (amendment 2), overall and per condition, and red-flag precision
   under decision A-7 (``APPROPRIATE`` in src/reasoning/red_flags.py), with the strict version (a
   flag is appropriate only when it names the true condition) beside it. Each with its 95%
   bootstrap interval.
4. Open decision A-5: the pneumothorax rule's reach with the sudden-onset cut-off (``E_59``) at 6,
   7 and 8, the only rule the cut-off changes on DDXPlus.

Validate only; the test split stays closed until Phase 4. **Circularity (R-12):** DDXPlus built
these patients from the same condition definitions the crosswalk reads, so a rule's reach here is
an upper bound, and aortic dissection has no DDXPlus patients at all. Runs in seconds and writes
data/interim/red_flags_check.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_crosswalk import red_flag_rates  # noqa: E402

from src.conditions import BY_ID, CONDITIONS, must_not_miss_ids  # noqa: E402
from src.contracts import Finding, PatientCase  # noqa: E402
from src.ddxplus import parse_token  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    CaseOutcome,
    bootstrap_ratio,
    red_flag_precision,
    red_flag_sensitivity,
)
from src.medical_kg.crosswalk import concepts_from_evidences  # noqa: E402
from src.reasoning.red_flags import APPROPRIATE, RULES, evaluate_red_flags  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERIM = REPO_ROOT / "data" / "interim"
ONSET_CUTOFFS = (6, 7, 8)


def _case(concepts: list[str]) -> PatientCase:
    return PatientCase(
        case_id="validate",
        age=50,
        sex="F",
        findings=[Finding(concept_id=c, label=c) for c in concepts],
    )


def _metric(ratio) -> dict:
    low, high = bootstrap_ratio(ratio)
    return {
        "value": round(ratio.value, 4),
        "ci": [round(low, 4), round(high, 4)],
        "cases": ratio.cases,
    }


def onset_cutoffs(frame: pd.DataFrame) -> dict:
    """The pneumothorax rule (sudden onset, pleuritic, breathless) at each onset cut-off."""
    rows = []
    for evidences, positive, label in zip(
        frame["evidences"], frame["positive_codes"], frame["label_condition_id"], strict=True
    ):
        onset = next((parse_token(t).ordinal for t in evidences if t.startswith("E_59_@_")), None)
        rows.append((label, onset, {"E_220", "E_66"} <= set(positive)))
    own = Counter(label for label, *_ in rows)["COND:spontaneous_pneumothorax"]
    others = len(rows) - own
    result = {}
    for cutoff in ONSET_CUTOFFS:
        fired = Counter(
            label == "COND:spontaneous_pneumothorax"
            for label, onset, rest in rows
            if rest and onset is not None and onset >= cutoff
        )
        result[str(cutoff)] = {
            "own_patients_flagged": round(fired[True] / own, 3),
            "other_patients_flagged": round(fired[False] / others, 3),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interim", type=Path, default=INTERIM)
    args = parser.parse_args(argv)

    parquet = args.interim / "ddxplus_chestpain_validate.parquet"
    if not parquet.exists():
        raise SystemExit(f"Missing {parquet}. Run scripts/build_ddxplus_chestpain.py first.")
    frame = pd.read_parquet(
        parquet, columns=["case_id", "evidences", "positive_codes", "label_condition_id"]
    )
    records = [
        (label, concepts_from_evidences(evidences))
        for evidences, label in zip(frame["evidences"], frame["label_condition_id"], strict=True)
    ]
    outcomes = [
        CaseOutcome(
            case_id=str(case_id),
            true_condition=label,
            ranking=(),
            differential=(),
            differential_size=0,
            flagged=tuple(evaluate_red_flags(_case(concepts))),
        )
        for case_id, (label, concepts) in zip(frame["case_id"], records, strict=True)
    ]
    critical = must_not_miss_ids()
    benign = [o for o in outcomes if o.true_condition not in critical]
    serious = [o for o in outcomes if o.true_condition in critical]
    burden = {
        "patients": len(outcomes),
        "any_flag": round(sum(bool(o.flagged) for o in outcomes) / len(outcomes), 3),
        "must_not_miss_patients": len(serious),
        "other_patients": len(benign),
        "other_patients_flagged": round(sum(bool(o.flagged) for o in benign) / len(benign), 3),
        "other_patients_flagged_by_condition": {
            c.id: round(
                sum(bool(o.flagged) for o in benign if o.true_condition == c.id)
                / max(1, sum(o.true_condition == c.id for o in benign)),
                3,
            )
            for c in CONDITIONS
            if c.id not in critical and c.ddxplus_label
        },
    }
    sensitivity = {"overall": _metric(red_flag_sensitivity(outcomes))}
    for condition in sorted(critical):
        ratio = red_flag_sensitivity(outcomes, condition=condition)
        if ratio.cases:
            sensitivity[condition] = _metric(ratio)
    summary = {
        "rules": len(RULES),
        "rates": red_flag_rates(records),
        "false_alarm_burden": burden,
        "red_flag_sensitivity": sensitivity,
        "red_flag_precision": _metric(red_flag_precision(outcomes, APPROPRIATE)),
        "red_flag_precision_strict": _metric(red_flag_precision(outcomes)),
        "a5_pneumothorax_by_onset_cutoff": onset_cutoffs(frame),
    }

    print(f"Red flags on validate: {len(RULES)} rules, {len(outcomes):,} patients")
    for cid, row in summary["rates"]["rules"].items():
        own = row["own_patients_flagged"]
        own_text = "no DDXPlus patients" if own is None else f"{own:.0%}"
        print(
            f"  {BY_ID[cid].label:<26} own patients {own_text:<20} "
            f"everyone else {row['other_patients_flagged']:.0%}"
        )
    print(
        f"\nAny flag: {burden['any_flag']:.0%} of all patients; "
        f"{burden['other_patients_flagged']:.0%} of the {burden['other_patients']:,} whose "
        "condition is not must-not-miss (the false-alarm burden)"
    )
    for cid, share in burden["other_patients_flagged_by_condition"].items():
        print(f"  {BY_ID[cid].label:<26} {share:.0%} flagged")
    print("\nRed-flag sensitivity (docs/05 amendment 2), 95% intervals:")
    for key, m in sensitivity.items():
        name = "overall" if key == "overall" else BY_ID[key].label
        print(f"  {name:<26} {m['value']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")
    for key, name in (("red_flag_precision", "A-7"), ("red_flag_precision_strict", "strict")):
        p = summary[key]
        print(
            f"Red-flag precision, {name}: {p['value']:.3f} "
            f"[{p['ci'][0]:.3f}, {p['ci'][1]:.3f}] over {p['cases']:,} patients with a flag"
        )
    print("\nA-5: the pneumothorax rule by sudden-onset cut-off (E_59 >= k):")
    for cutoff, row in summary["a5_pneumothorax_by_onset_cutoff"].items():
        print(
            f"  >= {cutoff}: own patients {row['own_patients_flagged']:.0%}, "
            f"everyone else {row['other_patients_flagged']:.0%}"
        )
    out = args.interim / "red_flags_check.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
