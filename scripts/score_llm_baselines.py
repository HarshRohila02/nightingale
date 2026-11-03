"""EXP-009 / EXP-010: the pre-registered figures, from a run's ``results.jsonl``.

    python scripts/score_llm_baselines.py RESULTS_DIR [--sheet DIR]

Reads what ``scripts/run_llm_baselines.py`` wrote and applies, unchanged, the analysis recorded in
``docs/08`` (EXP-009) before the run:

* **Ranking** (B3, B3r, B4; A0+LLM for context): the ``docs/05`` §3.1–§3.2 metrics with their 95%
  bootstrap intervals over the sample's validation patients, at each level the arm ran. A call
  that failed and an answer that did not parse are misses, and are counted beside the figures.
* **Explanations** (100% only, ``docs/05`` §3.4–§3.5): the pooled unsupported-claim rate with its
  interval over patients, the patients with any unsupported claim, citations (invented ones and
  the cited sentences that trace to nothing), treatment language, and for A0+LLM how often the
  template stood in and what the guardrails let through.
* **H3**: the paired difference in the unsupported-claim rate of each LLM-alone arm (B3, B3r)
  and each grounded arm (B4, A0+LLM's raw answer), four comparisons, all of which must pass.
* **The checkpoint of 2026-11-03** (decision D-5).
* **The rating sheet** for the two team raters (``docs/05`` §3.5), blind to the arm.

Validation patients only. The golden cases are listed apart, as examples. No model is called.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditions import BY_ID  # noqa: E402
from src.eval.claims import paired_rate_difference, pooled_rate, rate_interval  # noqa: E402
from src.eval.metrics import CaseOutcome, evaluate  # noqa: E402
from src.llm.baselines import A0_LLM, B3, B3R, B4  # noqa: E402

UNSUPPORTED_TARGET = 0.20
"""docs/05 §7: the unsupported-claim rate must be under 20%; D-5's checkpoint uses it."""
MAX_FAILED_SHARE = 0.10
"""D-5's checkpoint: more than this share of A0+LLM calls failing counts as a failed run."""
SLOW_SECONDS = 30.0
"""The production timeout (``llm.timeout_seconds``): calls slower than this are counted."""
H3_PAIRS = ((B3, B4), (B3, A0_LLM), (B3R, B4), (B3R, A0_LLM))
RANKING_ARMS = (B3, B3R, B4)

__all__ = [
    "checkpoint",
    "h3",
    "load_results",
    "main",
    "rating_sheet",
    "status_lines",
    "summarise",
]


def load_results(folder: Path) -> list[dict[str, Any]]:
    """The records of ``results.jsonl``, in the order they were made."""
    path = folder / "results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _select(records: Iterable[dict], arm: str, level: float, source: str) -> list[dict]:
    return [r for r in records if r["arm"] == arm and r["level"] == level and r["source"] == source]


def _ranking(record: Mapping[str, Any]) -> tuple[str, ...]:
    if record["arm"] == A0_LLM:
        return tuple(record.get("ranking") or ())
    return tuple((record.get("parsed") or {}).get("ranking") or ())


def _outcome(record: Mapping[str, Any]) -> CaseOutcome:
    truth = record["truth"]
    return CaseOutcome(
        case_id=record["case_id"],
        true_condition=truth["true_condition"],
        ranking=_ranking(record),
        differential=frozenset(truth["differential"]),
        differential_size=int(truth["differential_size"]),
    )


def _failed(record: Mapping[str, Any]) -> bool:
    return bool((record.get("reply") or {}).get("error"))


def ranking_block(records: Sequence[dict]) -> dict[str, Any]:
    """The §3.1–§3.2 metrics for these records, and what went wrong in reading them."""
    metrics = [
        {"name": m.name, "value": m.value, "ci": [m.ci_low, m.ci_high], "cases": m.cases}
        for m in evaluate([_outcome(r) for r in records])
    ]
    flags: dict[str, int] = {}
    for r in records:
        for flag in (r.get("parsed") or {}).get("flags", ()):
            flags[flag] = flags.get(flag, 0) + 1
    return {
        "cases": len(records),
        "metrics": metrics,
        "call_failed": sum(_failed(r) for r in records),
        "no_ranking": sum(not _ranking(r) for r in records),
        "flags": flags,
        "unmatched_items": sum(len((r.get("parsed") or {}).get("unmatched", ())) for r in records),
        "truncated": sum((r.get("reply") or {}).get("done_reason") == "length" for r in records),
        "context_full": sum(bool((r.get("reply") or {}).get("context_full")) for r in records),
    }


def explanation_block(records: Sequence[dict]) -> dict[str, Any]:
    """The §3.4–§3.5 measures of the explanations of one arm at 100%."""
    stats = [r.get("claims") for r in records]
    unsupported = [s["unsupported"] if s else 0 for s in stats]
    claims = [s["claims"] if s else 0 for s in stats]
    block: dict[str, Any] = {
        "cases": len(records),
        "with_explanation": sum(s is not None for s in stats),
        "call_failed": sum(_failed(r) for r in records),
        "claims": sum(claims),
        "unsupported": sum(unsupported),
        "cases_with_unsupported": sum(u > 0 for u in unsupported),
        "sentences_per_explanation": (
            sum(s["sentences"] for s in stats if s) / max(sum(s is not None for s in stats), 1)
        ),
        "treatment_language": sum(bool(s and s["treatment_language"]) for s in stats),
        "citations": sum(len(s["cited"]) for s in stats if s),
        "invalid_citations": sum(len(s["invalid_citations"]) for s in stats if s),
        "cited_sentences": sum(s["cited_sentences"] for s in stats if s),
        "cited_unsupported": sum(s["cited_unsupported"] for s in stats if s),
        "slower_than_30s": sum(
            ((r.get("reply") or {}).get("elapsed_s") or 0) > SLOW_SECONDS for r in records
        ),
        "mean_call_s": (
            sum((r.get("reply") or {}).get("elapsed_s") or 0 for r in records)
            / max(len(records), 1)
        ),
        "truncated": sum((r.get("reply") or {}).get("done_reason") == "length" for r in records),
        "context_full": sum(bool((r.get("reply") or {}).get("context_full")) for r in records),
        "conditions_with_a_passage": (
            sum(sum(bool(v) for v in r.get("passages", {}).values()) for r in records)
            / max(len(records), 1)
        ),
    }
    if sum(claims):
        rate, low, high = rate_interval(unsupported, claims)
        block["unsupported_rate"] = {"value": rate, "ci": [low, high]}
    else:
        block["unsupported_rate"] = None
    return block


def grounded_block(records: Sequence[dict]) -> dict[str, Any]:
    """A0+LLM beside its raw answer: what the guardrails let through, and how often the template
    stood in."""
    delivered = [r.get("delivered_claims") for r in records]
    return {
        "fell_back_to_template": sum(r.get("fallback_reason") is not None for r in records),
        "fallback_reasons": sorted(
            {r["fallback_reason"] for r in records if r.get("fallback_reason")}
        ),
        "delivered_unsupported": sum(d["unsupported"] for d in delivered if d),
        "delivered_treatment_language": sum(bool(d and d["treatment_language"]) for d in delivered),
        "delivered_not_grounded": sum(not r["delivered"]["grounded"] for r in records),
        "degraded": sorted({c for r in records for c in r.get("degraded", ())}),
    }


def h3(per_arm: Mapping[str, Sequence[dict]]) -> dict[str, Any]:
    """H3 (docs/05 §1): each LLM-alone arm against each grounded arm, on the unsupported-claim
    rate, paired over the patients both have. It is supported only if all four comparisons pass:
    the interval of (LLM alone − grounded) excludes 0 and is above it."""
    comparisons = []
    for alone, grounded in H3_PAIRS:
        if not per_arm.get(alone) or not per_arm.get(grounded):
            comparisons.append({"alone": alone, "grounded": grounded, "status": "not run"})
            continue
        a = {r["case_id"]: r for r in per_arm[alone]}
        b = {r["case_id"]: r for r in per_arm[grounded]}
        ids = sorted(set(a) & set(b))

        def counts(rows: Mapping[str, dict], ids: list[str] = ids) -> tuple[list[int], list[int]]:
            stats = [rows[i].get("claims") for i in ids]
            return [s["unsupported"] if s else 0 for s in stats], [
                s["claims"] if s else 0 for s in stats
            ]

        try:
            result = paired_rate_difference(pooled_rate(*counts(a)), pooled_rate(*counts(b)))
        except ValueError as error:
            comparisons.append(
                {"alone": alone, "grounded": grounded, "status": f"not testable: {error}"}
            )
            continue
        comparisons.append(
            {
                "alone": alone,
                "grounded": grounded,
                "cases": len(ids),
                **result,
                "passed": bool(result["excludes_zero"] and result["higher"] == "a"),
            }
        )
    supported = len(comparisons) == 4 and all(c.get("passed") for c in comparisons)
    return {"comparisons": comparisons, "supported": supported}


def checkpoint(summary: Mapping[str, Any]) -> dict[str, Any]:
    """D-5's checkpoint of 2026-11-03, applied to A0+LLM's raw answers at 100% on the sample.

    It fails (the template explainer stays, B3 and B4 are reported as not run) if the unsupported-
    claim rate is not under 20%, or more than 10% of the calls failed, or the arm did not run, or
    anything the guardrails delivered holds an unsupported claim or treatment language (a defect).
    """
    arm = summary["explanations"].get(A0_LLM)
    if not arm or not arm["cases"]:
        return {"passed": False, "reasons": ["A0+LLM did not run (it needs the retrieval bundle)"]}
    reasons = []
    rate = arm["unsupported_rate"]
    if rate is None:
        reasons.append("A0+LLM produced no claim to judge")
    elif not rate["value"] < UNSUPPORTED_TARGET:
        reasons.append(
            f"the unsupported-claim rate is {rate['value']:.3f}, not under {UNSUPPORTED_TARGET}"
        )
    if arm["call_failed"] / arm["cases"] > MAX_FAILED_SHARE:
        reasons.append(f"{arm['call_failed']} of {arm['cases']} calls failed")
    grounded = summary.get("grounded", {})
    if grounded.get("delivered_unsupported") or grounded.get("delivered_treatment_language"):
        reasons.append("a delivered explanation holds an unsupported claim or treatment language")
    return {"passed": not reasons, "reasons": reasons}


def summarise(records: Sequence[dict]) -> dict[str, Any]:
    """Every pre-registered figure, for the validation patients."""
    validation = [r for r in records if r["source"] == "validation"]
    arms = sorted({r["arm"] for r in validation})
    summary: dict[str, Any] = {
        "cases": len({r["case_id"] for r in validation}),
        "arms": arms,
        "ranking": {},
        "explanations": {},
    }
    for arm in arms:
        for level in sorted({r["level"] for r in validation if r["arm"] == arm}, reverse=True):
            rows = _select(validation, arm, level, "validation")
            block = ranking_block(rows)
            if arm == A0_LLM:
                block["note"] = "via the crosswalk inversion (R-17): not EXP-012's A0 ranking"
            summary["ranking"].setdefault(arm, {})[str(level)] = block
    per_arm = {arm: _select(validation, arm, 1.0, "validation") for arm in arms}
    for arm, rows in per_arm.items():
        summary["explanations"][arm] = explanation_block(rows)
    if per_arm.get(A0_LLM):
        summary["grounded"] = grounded_block(per_arm[A0_LLM])
    summary["h3"] = h3(per_arm)
    summary["checkpoint"] = checkpoint(summary)
    summary["golden"] = [
        {
            "arm": r["arm"],
            "case_id": r["case_id"],
            "top3": [BY_ID[c].label for c in _ranking(r)[:3] if c in BY_ID],
            "claims": (r.get("claims") or {}).get("claims"),
            "unsupported": (r.get("claims") or {}).get("unsupported"),
            "fallback": r.get("fallback_reason"),
        }
        for r in records
        if r["source"] == "golden"
    ]
    return summary


def status_lines(summary: Mapping[str, Any]) -> list[str]:
    """The headline, for STATUS.txt."""
    lines = [f"{summary['cases']} validation patients; arms: {', '.join(summary['arms'])}"]
    for arm, block in summary["explanations"].items():
        rate = block["unsupported_rate"]
        shown = (
            "no claims"
            if rate is None
            else (f"{rate['value']:.3f} [{rate['ci'][0]:.3f}, {rate['ci'][1]:.3f}]")
        )
        lines.append(
            f"{arm} at 100%: unsupported-claim rate {shown} "
            f"({block['unsupported']} of {block['claims']} claims); "
            f"{block['call_failed']} call(s) failed"
        )
    for c in summary["h3"]["comparisons"]:
        if "passed" in c:
            lines.append(
                f"H3 {c['alone']} - {c['grounded']}: {c['difference']:+.3f} "
                f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] "
                f"{'passes' if c['passed'] else 'does not pass'}"
            )
        else:
            lines.append(f"H3 {c['alone']} - {c['grounded']}: {c['status']}")
    lines.append(
        f"H3 {'SUPPORTED' if summary['h3']['supported'] else 'NOT supported'} (all four must pass)"
    )
    verdict = summary["checkpoint"]
    lines.append(
        "CHECKPOINT (2026-11-03): "
        + ("PASS" if verdict["passed"] else "FAIL: " + "; ".join(verdict["reasons"]))
    )
    return lines


# --------------------------------------------------------------------------------------------- #
# The rating sheet (docs/05 §3.5)
# --------------------------------------------------------------------------------------------- #

RATED_ARMS = (B3, B4, A0_LLM)
RATING_SEED = 42
RATING_COLUMNS = (
    "item",
    "patient",
    "top3",
    "explanation",
    "reasoning_validity_1_5",
    "evidence_relevance_1_5",
    "clarity_1_5",
    "citations_support_the_claims",
)


def _rated_text(record: Mapping[str, Any]) -> tuple[list[str], str] | None:
    """(top-3 labels, the explanation shown) for a record, or None when it has no explanation."""
    if record["arm"] == A0_LLM:
        text = (record.get("delivered") or {}).get("text")
    else:
        text = (record.get("parsed") or {}).get("explanation")
    if not text:
        return None
    top3 = [BY_ID[c].label for c in _ranking(record)[:3] if c in BY_ID]
    return top3, text


def rating_sheet(records: Sequence[dict], out: Path, seed: int = RATING_SEED) -> dict[str, int]:
    """Write ``rating_items.csv`` (what the raters see, in a fixed shuffled order, the arm hidden)
    and ``rating_key.json`` (item → arm, case, whether A0's template stood in). Items without an
    explanation are not rated, and are counted here."""
    rows = []
    unrated = {arm: 0 for arm in RATED_ARMS}
    for arm in RATED_ARMS:
        for r in _select(records, arm, 1.0, "validation"):
            shown = _rated_text(r)
            if shown is None:
                unrated[arm] += 1
                continue
            rows.append((r, *shown))
    rows.sort(
        key=lambda t: hashlib.sha256(
            f"{seed}:rating:{t[0]['case_id']}:{t[0]['arm']}".encode()
        ).hexdigest()
    )
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    with open(out / "rating_items.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RATING_COLUMNS)
        for number, (r, top3, text) in enumerate(rows, start=1):
            item = f"R-{number:03d}"
            key[item] = {
                "arm": r["arm"],
                "case_id": r["case_id"],
                "template_stood_in": r.get("fallback_reason") is not None,
            }
            writer.writerow(
                [item, "\n".join(r["case_text"]), "; ".join(top3), text, "", "", "", ""]
            )
    (out / "rating_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return {"items": len(rows), **{f"unrated_{arm}": n for arm, n in unrated.items()}}


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="the folder with results.jsonl")
    parser.add_argument("--out", type=Path, help="summary.json (default: in the folder)")
    parser.add_argument("--sheet", type=Path, help="write the rating sheet and its key here")
    args = parser.parse_args(argv)
    records = load_results(args.folder)
    summary = summarise(records)
    if args.sheet:
        summary["rating_sheet"] = rating_sheet(records, args.sheet)
    out = args.out or args.folder / "summary.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(status_lines(summary)))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
