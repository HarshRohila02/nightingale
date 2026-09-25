"""EXP-019: every system that exists, on validate at 100%, 50% and 25% evidence (docs/05 §3.7).

    python scripts/evaluate_reduced_evidence.py --digest-only   # draw the masks and print their digests
    python scripts/evaluate_reduced_evidence.py                 # score

Amendment 3 (approved 2026-09-25) scores every system at three evidence levels. The masks come
from src/ml/evidence_masks.py, the question rule of docs/05 §3.7, drawn once with seed 42. Their
digest must reproduce the one recorded in docs/08 before anything is scored on them, or the run
is refused (§3.7: "a run whose masks do not reproduce the digest is invalid"). A second digest
covers the questions each masked patient was asked (decision A-9), which the "asked" channel reads.

The systems, by their docs/05 §4 labels:

* **B0**, the prevalence prior (EXP-003): the same at every level.
* **B1-LR, B1-XGB** (EXP-004), and **B1-LR+aug, B1-XGB+aug, B1-LR′+aug, B1-XGB′+aug** (EXP-018).
* **B2**, the knowledge graph alone (EXP-005). **Circular on DDXPlus (R-12).**
* **The red-flag layer** alone: red-flag sensitivity (amendment 2) and the burden on patients
  without a must-not-miss condition (EXP-008).

Every §3.1–§3.3 metric with its 95% bootstrap interval, at each level, and McNemar on top-3, over
all patients and over must-not-miss patients only (§6 as amended), for the pairs that isolate one
change: the model (LR against XGB on the same rows), the masked copies (+aug against none) and the
"asked" channel (′+aug against +aug). The reduced-evidence figures of B0, B1 and B2 are descriptive,
not pre-registered (§9). Validate only: the test split stays closed until Phase 4. Scoring only,
nothing is trained. Writes data/interim/exp019_reduced_evidence.json.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditions import must_not_miss_ids  # noqa: E402
from src.contracts import Finding, PatientCase  # noqa: E402
from src.ddxplus import (
    code_sort_key,
    load_evidence_specs,
    parse_token,
    positive_codes,
)  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    CaseOutcome,
    bootstrap_ratio,
    evaluate,
    mcnemar,
    outcome_from_record,
    ranking_from_scores,
    red_flag_sensitivity,
)
from src.eval.reports import score, top3_hits  # noqa: E402
from src.medical_kg.crosswalk import case_from_ddxplus, concepts_from_evidences  # noqa: E402
from src.ml.baselines import PrevalencePrior  # noqa: E402
from src.ml.evidence_masks import LEVELS, mask_frame  # noqa: E402
from src.ml.features import evidence_codes_in_scope  # noqa: E402
from src.ml.ranker import ModelRanker, open_ranker  # noqa: E402
from src.reasoning.red_flags import evaluate_red_flags  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_DIGEST = "e99a7a8fbf792675"
"""docs/05 §3.7's digest of the validation masks, as the proposal and EXP-019 record it."""
ASKED_DIGEST = "af3a357f6454838a"
"""The digest of the asked sets (A-9), recorded in docs/08 (EXP-019) before anything was scored."""

B1 = REPO_ROOT / "models" / "nightingale_b0_b1"
AUG = REPO_ROOT / "models" / "nightingale_b1_asked" / "b1_aug"
ASKED_AUG = REPO_ROOT / "models" / "nightingale_b1_asked" / "b1_asked_aug"
MODELS = (
    ("B1-LR", B1, "logreg"),
    ("B1-XGB", B1, "xgboost"),
    ("B1-LR+aug", AUG, "logreg"),
    ("B1-XGB+aug", AUG, "xgboost"),
    ("B1-LR′+aug", ASKED_AUG, "logreg"),
    ("B1-XGB′+aug", ASKED_AUG, "xgboost"),
)
PAIRS = (
    ("B1-XGB", "B1-LR", "model"),
    ("B1-XGB+aug", "B1-LR+aug", "model"),
    ("B1-XGB′+aug", "B1-LR′+aug", "model"),
    ("B1-LR+aug", "B1-LR", "masked copies"),
    ("B1-XGB+aug", "B1-XGB", "masked copies"),
    ("B1-LR′+aug", "B1-LR+aug", "asked channel"),
    ("B1-XGB′+aug", "B1-XGB+aug", "asked channel"),
)
COLUMNS = [
    "case_id",
    "initial_evidence",
    "age",
    "sex",
    "evidences",
    "positive_codes",
    "label_condition_id",
    "label_differential",
]
ENCODED = ["age", "sex", "evidences", "asked"]
SHOWN = (
    "top-1 accuracy",
    "top-3 accuracy",
    "MRR",
    "Precision@3",
    "Recall@5",
    "must-not-miss recall@3",
    "dangerous false-negative rate",
)


def digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()[:16]


def draw(validate: pd.DataFrame, codes: list[str], specs: dict) -> dict[float, pd.DataFrame]:
    """{level: frame}, 1.0 the patients as listed; the reduced levels masked, with the
    positive codes recomputed from the kept tokens for the graph and the red flags."""
    frames = {1.0: validate}
    for level in LEVELS:
        masked = mask_frame(validate, level, codes=codes)
        masked["positive_codes"] = [
            positive_codes([parse_token(t) for t in tokens], specs)
            for tokens in masked["evidences"]
        ]
        frames[level] = masked
    return frames


def digests(frames: dict[float, pd.DataFrame]) -> dict[str, str]:
    tokens, asked = [], []
    for level in LEVELS:
        frame = frames[level]
        for case_id, kept, questions in zip(
            frame["case_id"], frame["evidences"], frame["asked"], strict=True
        ):
            tokens.append(f"{case_id}\t{level}\t{' '.join(kept)}")
            ordered = sorted(questions, key=code_sort_key)
            asked.append(f"{case_id}\t{level}\t{' '.join(ordered)}")
    return {"tokens": digest(tokens), "asked": digest(asked)}


def realised(frames: dict[float, pd.DataFrame]) -> dict[str, dict[str, float]]:
    out = {}
    for level, frame in frames.items():
        questions = [len({t.partition("_@_")[0] for t in e}) for e in frame["evidences"]]
        asked = [len(a) for a in frame["asked"]] if "asked" in frame else [None]
        out[str(level)] = {
            "tokens": round(float(np.mean([len(e) for e in frame["evidences"]])), 2),
            "questions": round(float(np.mean(questions)), 2),
            "asked": None if asked == [None] else round(float(np.mean(asked)), 2),
        }
    return out


def _metrics(cases: list[CaseOutcome]) -> list[dict[str, Any]]:
    return [
        {"name": m.name, "value": m.value, "ci": [m.ci_low, m.ci_high], "cases": m.cases}
        for m in evaluate(cases)
    ]


def graph_only(frame: pd.DataFrame, store: Any, labels: dict) -> list[CaseOutcome]:
    records = frame[
        COLUMNS[:1]
        + ["age", "sex", "evidences", "positive_codes", "label_condition_id", "label_differential"]
    ].to_dict("records")
    return [
        outcome_from_record(
            r, ranking_from_scores(store.score_by_connectivity(case_from_ddxplus(r, {}, labels)))
        )
        for r in records
    ]


def red_flag_layer(frame: pd.DataFrame) -> dict[str, Any]:
    critical = must_not_miss_ids()
    outcomes = []
    for case_id, evidences, label in zip(
        frame["case_id"], frame["evidences"], frame["label_condition_id"], strict=True
    ):
        case = PatientCase(
            case_id=str(case_id),
            age=50,
            sex="F",
            findings=[Finding(concept_id=c, label=c) for c in concepts_from_evidences(evidences)],
        )
        outcomes.append(
            CaseOutcome(
                case_id=str(case_id),
                true_condition=label,
                ranking=(),
                differential=(),
                differential_size=0,
                flagged=tuple(evaluate_red_flags(case)),
            )
        )
    ratio = red_flag_sensitivity(outcomes)
    low, high = bootstrap_ratio(ratio)
    per = {}
    for condition in sorted(critical):
        r = red_flag_sensitivity(outcomes, condition=condition)
        if r.cases:
            per[condition] = round(r.value, 4)
    benign = [o for o in outcomes if o.true_condition not in critical]
    return {
        "sensitivity": {"value": round(ratio.value, 4), "ci": [round(low, 4), round(high, 4)]},
        "sensitivity_by_condition": per,
        "other_patients_flagged": round(sum(bool(o.flagged) for o in benign) / len(benign), 4),
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument("--digest-only", action="store_true", help="draw the masks; score nothing")
    parser.add_argument("--no-graph", action="store_true", help="skip B2, the slowest system")
    args = parser.parse_args(argv)
    started = time.perf_counter()

    conditions = args.interim / "ddxplus_chestpain_conditions.json"
    evidences = args.raw_dir / "release_evidences.json"
    specs = load_evidence_specs(evidences)
    codes = sorted(evidence_codes_in_scope(conditions), key=code_sort_key)
    validate = pd.read_parquet(args.interim / "ddxplus_chestpain_validate.parquet", columns=COLUMNS)
    frames = draw(validate, codes, specs)
    drawn = digests(frames)
    sizes = realised(frames)
    print(f"validate: {len(validate):,} patients; numpy {np.__version__}")
    print(
        f"mask digest (docs/05 §3.7): {drawn['tokens']}   asked-set digest (A-9): {drawn['asked']}"
    )
    for level, s in sizes.items():
        print(f"  {level}: {s['tokens']} tokens, {s['questions']} questions, asked {s['asked']}")
    if drawn["tokens"] != TOKEN_DIGEST:
        raise SystemExit(f"the masks do not reproduce {TOKEN_DIGEST}: the run is invalid (§3.7)")
    if args.digest_only:
        return 0
    if ASKED_DIGEST is None or drawn["asked"] != ASKED_DIGEST:
        raise SystemExit(
            f"the asked sets give {drawn['asked']}, not the recorded {ASKED_DIGEST}: record the "
            "digests in docs/08 and pin them here before scoring anything"
        )

    results: dict[str, Any] = {
        "numpy": np.__version__,
        "digests": drawn,
        "realised": sizes,
        "systems": {},
        "comparisons": [],
    }
    critical = must_not_miss_ids()
    prior = PrevalencePrior.from_json(
        json.loads((B1 / "b0_prevalence.json").read_text(encoding="utf-8"))
    )
    cases: dict[tuple[str, float], list[CaseOutcome]] = {}
    for level, frame in frames.items():
        key = str(level)
        report, cases[("B0", level)] = score("B0", frame, prior.predict_proba(len(frame)))
        results["systems"].setdefault("B0", {})[key] = report
        for label, bundle, backend in MODELS:
            ranker = open_ranker(
                backend, model_dir=bundle, evidences_path=evidences, conditions_path=conditions
            )
            if not isinstance(ranker, ModelRanker):
                raise SystemExit(f"{label}: {ranker.reason}")
            X = ranker.encoder.transform_sparse(frame[[c for c in ENCODED if c in frame]])
            report, cases[(label, level)] = score(label, frame, ranker.model.predict_proba(X))
            results["systems"].setdefault(label, {})[key] = report
        print(f"  scored the ML systems at {key} ({time.perf_counter() - started:.0f} s)")

    if not args.no_graph:
        from src.medical_kg.networkx_store import NetworkXGraphStore

        store = NetworkXGraphStore.from_files(
            conditions,
            args.interim / "ddxplus_evidences.json",
            REPO_ROOT / "data" / "raw" / "bodhi_s",
        )
        labels = dict(store.graph.nodes(data="label"))
        for level, frame in frames.items():
            cases[("B2", level)] = graph_only(frame, store, labels)
            results["systems"].setdefault("B2", {})[str(level)] = {
                "model": "B2",
                "circular": "R-12",
                "metrics": _metrics(cases[("B2", level)]),
            }
            print(f"  scored B2 at {level} ({time.perf_counter() - started:.0f} s)")

    results["red_flag_layer"] = {str(level): red_flag_layer(f) for level, f in frames.items()}

    for level in frames:
        for a, b, change in PAIRS:
            ca, cb = cases[(a, level)], cases[(b, level)]
            everyone = mcnemar(top3_hits(ca), top3_hits(cb))
            serious = [i for i, o in enumerate(ca) if o.true_condition in critical]
            hits_a, hits_b = top3_hits(ca), top3_hits(cb)
            must = mcnemar([hits_a[i] for i in serious], [hits_b[i] for i in serious])
            results["comparisons"].append(
                {
                    "level": level,
                    "a": a,
                    "b": b,
                    "change": change,
                    "top3": everyone,
                    "top3_must_not_miss": must,
                }
            )

    out = args.interim / "exp019_reduced_evidence.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    names = ["B0", *(m[0] for m in MODELS), *([] if args.no_graph else ["B2"])]
    for level in frames:
        print(f"\nEvidence {level:.0%}:")
        print(f"  {'system':<13}" + "".join(f"{n[:14]:>16}" for n in SHOWN))
        for name in names:
            by = {m["name"]: m for m in results["systems"][name][str(level)]["metrics"]}
            print(f"  {name:<13}" + "".join(f"{by[n]['value']:>16.4f}" for n in SHOWN))
    print(f"\nWrote {out} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
