"""Does EXP-012's rehearsal reproduce what the earlier experiments recorded? (docs/08, EXP-012)

    python scripts/check_rehearsal.py data/interim/exp012_validate_rehearsal.json

The rehearsal (``scripts/evaluate_ablations.py`` on validation, with EXP-006's α) re-scores systems
that earlier experiments already scored, from the same masks and the same models. Before the freeze it
must give **the same figures**, so any difference is a bug, not a finding. This compares, for the
seed-42 models, every metric the two records share (its value, its interval and its case count):

* **EXP-019:** B0, the original and ``+aug`` B1 models, B2 (``data/interim/exp019_reduced_evidence.json``);
* **EXP-021:** both transformer arms (``models/nightingale_b1_dl/reduced_evidence_seed42.json``);
* **EXP-006:** each fused component at its chosen α (``data/interim/exp006_fusion.json``): the fused
  ranking without the flags is the rehearsal's A4, the one with them is A0 (or A7, for the `+aug` B1
  models), and the α itself.

It also checks that the masks' digests are EXP-019's and that the rehearsal says it is valid as a result.
Exit status 1 if anything differs, 2 if a record to compare with is missing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP019 = REPO_ROOT / "data" / "interim" / "exp019_reduced_evidence.json"
EXP021 = REPO_ROOT / "models" / "nightingale_b1_dl" / "reduced_evidence_seed42.json"
EXP006 = REPO_ROOT / "data" / "interim" / "exp006_fusion.json"
LEVELS = ("1.0", "0.5", "0.25")
TOLERANCE = 1e-9
REQUIRED = {
    "top-1 accuracy",
    "top-3 accuracy",
    "top-5 accuracy",
    "MRR",
    "Precision@3",
    "Recall@5",
    "must-not-miss recall@3",
    "dangerous false-negative rate",
}
"""Metrics every comparison must have shared, so an empty overlap cannot pass."""
FUSED_AS = {
    # EXP-006 component -> (rehearsal system without flags, with flags)
    "B1-DL+aug": ("A4", "A0"),
    "B1-DL′+aug": ("A4′", "A0′"),
    "B1-LR+aug": (None, "A7-LR"),
    "B1-XGB+aug": (None, "A7-XGB"),
    "B1-LR′+aug": (None, "A7-LR′"),
    "B1-XGB′+aug": (None, "A7-XGB′"),
}


def metric_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {m["name"]: m for m in report["metrics"]}


def compare_metrics(
    what: str,
    ours: Mapping[str, Any],
    theirs: Mapping[str, Any],
    tolerance: float = TOLERANCE,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Every metric the two reports share: value, interval and case count.

    Returns:
        The rows (``status``: ``exact``, ``close`` within ``tolerance``, or ``DIFFERS``) and the names
        of the metrics compared.
    """
    a, b = metric_map(ours), metric_map(theirs)
    shared = sorted(set(a) & set(b))
    rows = []
    for name in shared:
        for part, x, y in (
            ("value", a[name]["value"], b[name]["value"]),
            ("ci low", a[name]["ci"][0], b[name]["ci"][0]),
            ("ci high", a[name]["ci"][1], b[name]["ci"][1]),
            ("cases", a[name]["cases"], b[name]["cases"]),
        ):
            gap = abs(float(x) - float(y))
            if part == "cases":
                status = "exact" if gap == 0 else "DIFFERS"
            else:
                status = "exact" if gap == 0 else "close" if gap <= tolerance else "DIFFERS"
            rows.append(
                {"what": f"{what}, {name}, {part}", "ours": x, "theirs": y, "status": status}
            )
    return rows, shared


def _need(record: Mapping[str, Any], *path: str) -> Any:
    node: Any = record
    for step in path:
        node = node[step]
    return node


def check(
    rehearsal: Mapping[str, Any],
    exp019: Mapping[str, Any] | None,
    exp021: Mapping[str, Any] | None,
    exp006: Mapping[str, Any] | None,
    tolerance: float = TOLERANCE,
) -> tuple[list[dict[str, Any]], list[str]]:
    """All comparisons, and the problems that are not a differing figure (a missing system, a
    metric the two records do not share, an invalid or mis-masked run)."""
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    systems = rehearsal["systems"]

    if not rehearsal.get("valid_as_a_result"):
        problems.append(
            f"the rehearsal is not valid as a result: {rehearsal.get('not_a_result_because')}"
        )
    if exp019 is not None and rehearsal.get("digests") != exp019.get("digests"):
        problems.append(
            f"mask digests {rehearsal.get('digests')} are not EXP-019's {exp019.get('digests')}"
        )

    def against(source: str, record: Mapping[str, Any] | None, labels: Sequence[str]) -> None:
        if record is None:
            return
        for label in labels:
            if label not in record["systems"]:
                problems.append(f"{source} has no {label}")
                continue
            if label not in systems:
                problems.append(f"the rehearsal has no {label} (needed against {source})")
                continue
            for level in LEVELS:
                got, shared = compare_metrics(
                    f"{source} {label} @ {level}",
                    systems[label][level],
                    record["systems"][label][level],
                    tolerance,
                )
                rows.extend(got)
                if not set(shared) >= REQUIRED:
                    problems.append(
                        f"{source} {label} @ {level}: not compared: {sorted(REQUIRED - set(shared))}"
                    )

    if exp019 is not None:
        against("EXP-019", exp019, [s for s in exp019["systems"] if s in systems])
        for label in exp019["systems"]:
            if label not in systems:
                problems.append(f"EXP-019's {label} is missing from the rehearsal")
    if exp021 is not None:
        against("EXP-021", exp021, list(exp021["systems"]))

    if exp006 is not None:
        for label, (plain, flagged) in FUSED_AS.items():
            entry = exp006["components"].get(label)
            chosen = (entry or {}).get("chosen", {}).get("alpha")
            if chosen is None:
                problems.append(f"EXP-006 chose no α for {label}")
                continue
            used = rehearsal["alphas"][label]["alpha"]
            if abs(used - chosen) > tolerance:
                problems.append(
                    f"{label}: the rehearsal fused at α = {used}, EXP-006 chose {chosen}"
                )
            for level in LEVELS:
                at = entry["at_chosen"][level]
                if plain and plain in systems:
                    got, shared = compare_metrics(
                        f"EXP-006 {label} fused @ {level} vs {plain}",
                        systems[plain][level],
                        at["fused"],
                        tolerance,
                    )
                    rows.extend(got)
                    if not set(shared) >= REQUIRED:
                        problems.append(
                            f"EXP-006 {label} fused @ {level}: not compared: {sorted(REQUIRED - set(shared))}"
                        )
                if flagged in systems:
                    got, shared = compare_metrics(
                        f"EXP-006 {label} fused + flags @ {level} vs {flagged}",
                        systems[flagged][level],
                        at["fused + red flags"],
                        tolerance,
                    )
                    rows.extend(got)
                    if not set(shared) >= REQUIRED:
                        problems.append(f"EXP-006 {label} fused + flags @ {level}: not compared")
                else:
                    problems.append(f"the rehearsal has no {flagged}")
    return rows, problems


def summarise(rows: Sequence[Mapping[str, Any]], problems: Sequence[str]) -> str:
    counts = {s: sum(r["status"] == s for r in rows) for s in ("exact", "close", "DIFFERS")}
    lines = [
        f"compared {len(rows)} figures: {counts['exact']} exact, {counts['close']} within tolerance, "
        f"{counts['DIFFERS']} DIFFER; {len(problems)} other problem(s)"
    ]
    lines += [
        f"  DIFFERS  {r['what']}: rehearsal {r['ours']!r}, record {r['theirs']!r}"
        for r in rows
        if r["status"] == "DIFFERS"
    ][:40]
    lines += [f"  PROBLEM  {p}" for p in problems]
    return "\n".join(lines)


def _load(path: Path | None, name: str) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        print(f"missing: {name} ({path})", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "rehearsal", type=Path, help="the rehearsal's JSON (exp012_validate_rehearsal.json)"
    )
    parser.add_argument("--exp019", type=Path, default=EXP019)
    parser.add_argument("--exp021", type=Path, default=EXP021)
    parser.add_argument(
        "--exp006", type=Path, default=EXP006, help="EXP-006's result, from the same run"
    )
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args(argv)
    rows, problems = check(
        _load(args.rehearsal, "the rehearsal"),
        _load(args.exp019, "EXP-019's record"),
        _load(args.exp021, "EXP-021's record"),
        _load(args.exp006, "EXP-006's result"),
        args.tolerance,
    )
    print(summarise(rows, problems))
    return 1 if problems or any(r["status"] == "DIFFERS" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
