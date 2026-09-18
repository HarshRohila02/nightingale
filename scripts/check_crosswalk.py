"""Validate the crosswalk and show what it changes (task 1b).

    python scripts/check_crosswalk.py

1. Checks src/medical_kg/crosswalk.py against data/raw/ddxplus/release_evidences.json.
2. Concept prevalence on the validate parquet: the share of each condition's patients who have
   each concept. A sanity check of the mapping: chest pain should be common wherever DDXPlus
   puts pain in the chest.
3. The red-flag rules on the validate patients, a preview of EXP-008 (task 2d): for each rule,
   the share of its own condition's patients it flags, and of everyone else (false alarms).
4. The golden cases on the real knowledge graph: each case expanded through the crosswalk and
   run through the pipeline with the NetworkX store in place of the stub.

Validate only: the test split stays closed until Phase 4. Runs in seconds. Writes
data/interim/crosswalk_check.json. Exits 1 if the crosswalk is invalid.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.conditions import BY_ID, CONDITIONS  # noqa: E402
from src.contracts import Finding, PatientCase  # noqa: E402
from src.medical_kg.crosswalk import (  # noqa: E402
    CROSSWALK,
    Match,
    concepts_from_evidences,
    expand_case,
    validate_crosswalk,
)
from src.medical_kg.networkx_store import NetworkXGraphStore  # noqa: E402
from src.pipeline import DiagnosisPipeline  # noqa: E402
from src.reasoning.red_flags import RULES, evaluate_red_flags  # noqa: E402
from src.stubs import ConstantRanker, EmptyRetriever, TemplateExplainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "ddxplus"
INTERIM = REPO_ROOT / "data" / "interim"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"


def prevalence(records: list[tuple[str, list[str]]]) -> dict[str, dict[str, float]]:
    """{concept: {condition: share of that condition's patients with the concept}}."""
    patients = Counter(label for label, _ in records)
    counts: dict[str, Counter] = defaultdict(Counter)
    for label, concepts in records:
        for concept in concepts:
            counts[concept][label] += 1
    return {
        e.concept_id: {
            cid: round(counts[e.concept_id][cid] / n, 3) for cid, n in sorted(patients.items())
        }
        for e in CROSSWALK
    }


def red_flag_rates(records: list[tuple[str, list[str]]]) -> dict:
    """Per rule: the share of its condition's patients flagged, and of all other patients.

    This is not docs/05's "red-flag sensitivity", which counts cases matching a rule's pattern
    and is 1.0 by construction for a deterministic rule. It asks how many of the condition's
    own patients the rule reaches. A flag on another condition's patient is not always wrong:
    the MI rule flagging unstable angina is an appropriate ischaemic alert.
    """
    patients = Counter(label for label, _ in records)
    flagged: dict[str, Counter] = defaultdict(Counter)  # rule condition -> label -> patients
    any_flag = 0
    for label, concepts in records:
        case = PatientCase(
            case_id="validate",
            age=50,
            sex="F",
            findings=[Finding(concept_id=c, label=c) for c in concepts],
        )
        fired = evaluate_red_flags(case)
        any_flag += bool(fired)
        for condition_id in fired:
            flagged[condition_id][label] += 1

    total = sum(patients.values())
    rules = {}
    for condition_id in dict.fromkeys(r.condition_id for r in RULES):
        own = patients.get(condition_id, 0)
        hits = flagged[condition_id]
        others = total - own
        rules[condition_id] = {
            "own_patients": own,
            "own_patients_flagged": round(hits[condition_id] / own, 3) if own else None,
            "other_patients_flagged": round((sum(hits.values()) - hits[condition_id]) / others, 3),
            "flagged_by_condition": dict(hits.most_common()),
        }
    return {"patients": total, "any_flag_rate": round(any_flag / total, 3), "rules": rules}


def golden_on_real_graph(store: NetworkXGraphStore) -> list[dict]:
    """Run each golden case, expanded through the crosswalk, on the real graph."""
    labels = dict(store.graph.nodes(data="label"))
    pipeline = DiagnosisPipeline(
        ranker=ConstantRanker(),
        graph=store,
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
    )
    rows = []
    for golden in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]:
        expect = golden["expect"]
        result = pipeline.run(expand_case(PatientCase(**golden["case"]), labels))
        top_k = expect.get("top_k", 3)
        top_ids = [c.condition_id for c in result.candidates[:top_k]]
        flagged = {c.condition_id for c in result.candidates if c.red_flag}
        checks = {f"{cid} in top {top_k}": cid in top_ids for cid in expect.get("must_include", [])}
        checks |= {f"red flag for {cid}": cid in flagged for cid in expect.get("red_flag_for", [])}
        if expect.get("no_red_flag"):
            checks["no red flag"] = not flagged
        rows.append(
            {
                "id": golden["id"],
                "checks": checks,
                "top5": [
                    [c.condition_id, round(c.kg_score, 3), c.red_flag]
                    for c in result.candidates[:5]
                ],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--interim", type=Path, default=INTERIM)
    args = parser.parse_args(argv)

    release_path = args.raw_dir / "release_evidences.json"
    if not release_path.exists():
        raise SystemExit(f"Missing {release_path}. Run: python scripts/download_data.py")
    problems = validate_crosswalk(json.loads(release_path.read_text(encoding="utf-8")))
    matches = Counter(e.match.value for e in CROSSWALK)
    summary: dict = {"entries": len(CROSSWALK), "matches": dict(matches), "problems": problems}

    print(f"Crosswalk : {len(CROSSWALK)} concepts  {dict(matches)}")
    for match in (Match.NONE, Match.RELATED):
        names = [e.concept_id for e in CROSSWALK if e.match is match]
        print(f"  {match.value:<8}: {', '.join(names)}")
    for problem in problems:
        print(f"PROBLEM: {problem}")

    parquet = args.interim / "ddxplus_chestpain_validate.parquet"
    if parquet.exists():
        import pandas as pd  # local import: pandas is only needed for this section

        frame = pd.read_parquet(parquet, columns=["evidences", "label_condition_id"])
        records = [
            (label, concepts_from_evidences(evidences))
            for evidences, label in zip(
                frame["evidences"], frame["label_condition_id"], strict=True
            )
        ]
        summary["prevalence"] = prevalence(records)
        summary["red_flags"] = red_flag_rates(records)

        print(f"\nConcept prevalence on validate ({len(records)} patients): top conditions")
        for concept, shares in summary["prevalence"].items():
            top = sorted(shares.items(), key=lambda kv: -kv[1])[:3]
            if top and top[0][1] > 0:
                shown = "  ".join(f"{BY_ID[c].label[:14]} {s:.0%}" for c, s in top if s > 0)
                print(f"  {concept:<38} {shown}")

        flags = summary["red_flags"]
        print(f"\nRed-flag rules on validate: {flags['any_flag_rate']:.0%} of patients get a flag")
        for cid, row in flags["rules"].items():
            own = row["own_patients_flagged"]
            own_text = "no DDXPlus patients" if own is None else f"{own:.0%}"
            print(
                f"  {BY_ID[cid].label:<26} flags its own patients: {own_text:<20} "
                f"and {row['other_patients_flagged']:.0%} of everyone else"
            )
        uncovered = [
            c.label for c in CONDITIONS if c.is_must_not_miss and c.id not in flags["rules"]
        ]
        print(f"  Must-not-miss conditions without a rule: {', '.join(uncovered)}")
    else:
        print(f"\n(skipped prevalence and red flags: {parquet} not found)")

    kg_files = [
        args.interim / "ddxplus_chestpain_conditions.json",
        args.interim / "ddxplus_evidences.json",
    ]
    if all(p.exists() for p in kg_files):
        summary["golden_on_real_graph"] = golden_on_real_graph(
            NetworkXGraphStore.from_files(*kg_files)
        )
        print("\nGolden cases on the real graph (expanded through the crosswalk):")
        for row in summary["golden_on_real_graph"]:
            failed = [name for name, ok in row["checks"].items() if not ok]
            verdict = "all expectations hold" if not failed else f"FAILS: {'; '.join(failed)}"
            print(f"  {row['id']}: {verdict}")
            for cid, kg, flagged in row["top5"]:
                print(f"      {BY_ID[cid].label:<26} kg {kg:.3f}{'  RED FLAG' if flagged else ''}")
    else:
        print("\n(skipped golden cases: run scripts/decode_ddxplus.py first)")

    out = args.interim / "crosswalk_check.json"
    if args.interim.exists():
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
