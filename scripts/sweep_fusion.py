"""EXP-006: the fusion's one weight, swept on validation at 100%, 50% and 25% evidence (2c).

    python scripts/sweep_fusion.py

**This is tuning on project data, so it runs only where the owner says (D-7), and only after
the team answers E-2's first item** (which level the reduced-evidence claims are judged at). The
team settled the errata to amendment 3 (E-1) on 2026-10-02, as `docs/05` amendment 3a. By its item
17 (b) the reduced-evidence claims must also hold against the `+aug` B1 variants, the very
components this fuses and compares with; by its item 4, MRR has a paired test beside McNemar's.
Nothing is trained. It reuses EXP-019's masks (``scripts/evaluate_reduced_evidence.py``) and
refuses to run unless both recorded digests reproduce. It writes
``data/interim/exp006_fusion.json``.

The rule that picks the weight was recorded in ``docs/08`` (EXP-006) before this ever ran. It
applies to the unrounded values this script computes:

* **The weight.** α, the graph's share of the logarithmic opinion pool (``src/fusion/pool.py``),
  from 0.05 to 1 in steps of 0.05. The model floor ε = 0.01 and the rest of the design are fixed.
  α = 0 is left out: it is not a fusion (the floor is off there), and it is not the model alone
  either, because aortic dissection takes the graph's posterior at every α. The ML component alone
  is scored separately and is the comparator. α = 1 is exactly B2.
* **The golden cases first.** An α is admissible only if all four golden cases keep their
  expectations with red flags off, for the component swept; for B1-LR′+aug, whose α the pipeline
  adopts, also for B1-LR, the other model the golden tests run. These use hand-written cases only
  and are recorded in EXP-006 before the sweep. If no α passes, none is adopted: the pipeline keeps
  0.5 / 0.5, and no golden expectation or weight is changed by hand.
* **What is scored.** The fused ranking alone, red flags off: red flags bypass fusion
  (``docs/02`` §6) and are never tuned. Fusion plus red flags is reported beside it.
* **The objective.** Among admissible α, the mean over the three levels of MRR, one α for every
  level (``docs/05`` §3.7). Top-3 and must-not-miss recall@3 saturate on DDXPlus (R-16); MRR is
  H1's other metric.
* **The constraint.** At every level, the fused ranking's must-not-miss recall@3 is at least the
  ML component's own, from this run (equal to EXP-019's). Dissection, which no DDXPlus patient
  has, can push a true must-not-miss condition out of the top 3 at every α, so the constraint may
  fail everywhere; then, among the golden-admissible α, the highest mean must-not-miss recall@3 is
  taken, then MRR. Ties go to the smaller α. The branch taken is reported.
* **The components.** The four B1 variants trained like A0's component (``+aug``, and ′ for the
  primed systems), seed 42, each swept on its own. The pipeline takes B1-LR′+aug's α, written to
  ``ranking.ml_weight`` = 1 − α and ``ranking.kg_weight`` = α in configs/config.yaml. A0's own
  component, B1-DL+aug, is swept the same way when it exists. **The configured component stays
  B1-LR′+aug:** these fused results do not choose it. Revisiting it (EXP-019 point 9) would be a
  separate, dated decision after the sweep, disclosed as post hoc.
* **Seeds.** α is chosen on the seed-42 components only (``docs/05`` §6). When EXP-020's seeds
  43–46 return, each is fused at its arm's seed-42 α, never re-swept, and reported as mean ± std
  beside the seed-42 figures.
* **Recorded at every α,** not only the chosen one: Precision@3, how often dissection enters the
  top 3, and how many must-not-miss patients dissection alone pushes out of it.

The α is chosen on these patients, so the fused figures here are optimistic; the test split
(Phase 4) gives the honest ones. Every figure is on DDXPlus, which is synthetic and where the graph
is circular (R-12), and the system is closed-world (R-13).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_reduced_evidence as exp019  # noqa: E402

from src.conditions import CONDITIONS, must_not_miss_ids  # noqa: E402
from src.contracts import Finding, PatientCase  # noqa: E402
from src.ddxplus import code_sort_key, load_evidence_specs  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    evaluate,
    expected_calibration_error,
    f1_by_condition,
    mcnemar,
    outcome_from_record,
    paired_bootstrap_difference,
    reliability_table,
)
from src.fusion import ML_FLOOR, fuse_arrays  # noqa: E402
from src.medical_kg.crosswalk import (  # noqa: E402
    case_from_ddxplus,
    concepts_from_evidences,
    expand_case,
)
from src.ml.baselines import LABELS  # noqa: E402
from src.ml.features import evidence_codes_in_scope  # noqa: E402
from src.ml.ranker import ModelRanker, open_ranker  # noqa: E402
from src.pipeline import DiagnosisPipeline  # noqa: E402
from src.reasoning.red_flags import evaluate_red_flags  # noqa: E402
from src.stubs import EmptyRetriever, TemplateExplainer  # noqa: E402

GRID = tuple(round(0.05 * i, 2) for i in range(1, 21))
COMPONENTS = tuple(m for m in exp019.MODELS if m[0].endswith("+aug"))
CONFIGURED = "B1-LR′+aug"
ALSO_GOLDEN = {CONFIGURED: ("B1-LR", exp019.B1, "logreg")}
"""The other models the golden-case ranking tests run beside the configured one."""
IDS = tuple(c.id for c in CONDITIONS)
MODEL_COLUMNS = tuple(IDS.index(label) for label in LABELS)
DISSECTION = IDS.index("COND:aortic_dissection")
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"


def graph_matrix(frame: pd.DataFrame, store: Any, labels: dict) -> np.ndarray:
    """The graph's log-likelihood of every condition, one row per patient, in registry order."""
    rows = []
    for record in frame.to_dict("records"):
        scores = store.score_by_connectivity(case_from_ddxplus(record, {}, labels))
        rows.append([scores[c] for c in IDS])
    return np.asarray(rows)


def flag_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Which conditions each patient's red flags name, as EXP-019's red-flag layer computes them."""
    flagged = np.zeros((len(frame), len(IDS)), dtype=bool)
    for i, evidences in enumerate(frame["evidences"]):
        case = PatientCase(
            case_id=str(i),
            age=50,
            sex="F",
            findings=[Finding(concept_id=c, label=c) for c in concepts_from_evidences(evidences)],
        )
        for condition in evaluate_red_flags(case):
            flagged[i, IDS.index(condition)] = True
    return flagged


def differential_matrix(frame: pd.DataFrame) -> np.ndarray:
    """D_in: which in-scope conditions each patient's differential lists (docs/05 amendment 1)."""
    in_d = np.zeros((len(frame), len(IDS)), dtype=bool)
    for i, differential in enumerate(frame["label_differential"]):
        for entry in differential:
            if entry["condition_id"] in IDS:
                in_d[i, IDS.index(entry["condition_id"])] = True
    return in_d


def orders(scores: np.ndarray, flagged: np.ndarray | None = None) -> np.ndarray:
    """Column indices by descending score, ties by registry order (``ranking_from_scores``).

    With ``flagged``, flagged conditions come first, as the pipeline's (red_flag, fused_score)
    sort puts them: two keys, so no precision is lost.
    """
    if flagged is None:
        return np.argsort(-scores, axis=1, kind="stable")
    return np.lexsort((-scores, ~flagged), axis=1)


def true_ranks(order: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """1-based rank of each patient's true condition, which every order must contain."""
    found = order == truth[:, None]
    if not found.any(axis=1).all():
        raise ValueError("a true condition is missing from its order, so it has no rank")
    return np.argmax(found, axis=1) + 1


def point_metrics(ranks: np.ndarray, critical: np.ndarray) -> dict[str, float]:
    return {
        "MRR": float(np.mean(1.0 / ranks)),
        "top-3 accuracy": float(np.mean(ranks <= 3)),
        "must-not-miss recall@3": float(np.mean(ranks[critical] <= 3)),
    }


def dissection_effects(
    order: np.ndarray, truth: np.ndarray, critical: np.ndarray, in_d: np.ndarray
) -> dict[str, float]:
    """What dissection's slot costs a ranking of DDXPlus patients, none of whom has it."""
    ranks = true_ranks(order, truth)
    above = np.argmax(order == DISSECTION, axis=1) + 1 < ranks
    rank_without = ranks - above
    top3 = order[:, :3]
    return {
        "Precision@3": float(np.mean(np.take_along_axis(in_d, top3, axis=1).sum(1) / 3)),
        "dissection in the top 3": float(np.mean((top3 == DISSECTION).any(1))),
        "must-not-miss patients pushed out of the top 3 by dissection": int(
            np.sum(critical & (ranks > 3) & (rank_without <= 3))
        ),
    }


def golden_ranks(ranker: Any, store: Any, labels: dict, alpha: float) -> dict[str, Any]:
    """Each golden case's required conditions, ranked with red flags off at this α."""
    pipeline = DiagnosisPipeline(
        ranker=ranker,
        graph=store,
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
        enable_red_flags=False,
        ml_weight=1 - alpha,
        kg_weight=alpha,
    )
    out = {}
    for golden in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]:
        case = expand_case(PatientCase(**golden["case"]), labels)
        ranking = [c.condition_id for c in pipeline.run(case).candidates]
        top_k = golden["expect"].get("top_k", 3)
        required = golden["expect"].get("must_include", [])
        out[golden["id"]] = {
            "ranks": {c: ranking.index(c) + 1 for c in required},
            "top_k": top_k,
            "holds": all(ranking.index(c) < top_k for c in required),
        }
    return out


def choose(
    sweep: list[dict[str, Any]], floor: dict[str, float], golden: dict[float, bool]
) -> dict[str, Any]:
    """The pre-registered rule (docs/08 EXP-006), on unrounded values.

    Args:
        sweep: One entry per α: ``{"alpha", "levels": {level: point metrics}}``.
        floor: The ML component's own must-not-miss recall@3 at each level.
        golden: Whether every golden case holds at each α (red flags off).
    """

    def mean(entry: dict[str, Any], name: str) -> float:
        return float(np.mean([m[name] for m in entry["levels"].values()]))

    def safe(entry: dict[str, Any]) -> bool:
        return all(
            m["must-not-miss recall@3"] >= floor[level] for level, m in entry["levels"].items()
        )

    kept = [e for e in sweep if golden.get(e["alpha"], False)]
    if not kept:
        return {
            "alpha": None,
            "rule": "no α keeps every golden case: none adopted; the pipeline keeps 0.5 / 0.5",
        }
    admissible = [e for e in kept if safe(e)]
    if admissible:
        best = max(admissible, key=lambda e: (mean(e, "MRR"), -e["alpha"]))
        return {"alpha": best["alpha"], "rule": "highest mean MRR meeting the constraint"}
    best = max(kept, key=lambda e: (mean(e, "must-not-miss recall@3"), mean(e, "MRR"), -e["alpha"]))
    return {"alpha": best["alpha"], "rule": "no α met the constraint: highest mean must-not-miss"}


def outcomes(records: list[dict], order: np.ndarray, probabilities: np.ndarray | None) -> list:
    cases = []
    for i, record in enumerate(records):
        ranking = [IDS[j] for j in order[i]]
        p = None if probabilities is None else dict(zip(IDS, probabilities[i], strict=True))
        cases.append(outcome_from_record(record, ranking, probabilities=p))
    return cases


def _metrics(cases: list) -> list[dict[str, Any]]:
    return [
        {"name": m.name, "value": m.value, "ci": [m.ci_low, m.ci_high], "cases": m.cases}
        for m in evaluate(cases)
    ]


def _open(label: str, bundle: Path, backend: str, evidences: Path, conditions: Path) -> ModelRanker:
    ranker = open_ranker(
        backend, model_dir=bundle, evidences_path=evidences, conditions_path=conditions
    )
    if not isinstance(ranker, ModelRanker):
        raise SystemExit(f"{label}: {ranker.reason}")
    if tuple(ranker.model.labels) != LABELS:
        raise SystemExit(f"{label}: its classes are not in LABELS order, so its columns are not")
    return ranker


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument(
        "--golden-only",
        action="store_true",
        help="only the golden-case gate (hand-written cases; no validation patient is ranked)",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()

    def say(message: str) -> None:
        print(f"{message} ({time.perf_counter() - started:.0f} s)", flush=True)

    conditions = args.interim / "ddxplus_chestpain_conditions.json"
    evidences = args.raw_dir / "release_evidences.json"

    from src.medical_kg.networkx_store import NetworkXGraphStore

    store = NetworkXGraphStore.from_files(
        conditions, args.interim / "ddxplus_evidences.json", REPO_ROOT / "data" / "raw" / "bodhi_s"
    )
    labels = dict(store.graph.nodes(data="label"))
    rankers = {label: _open(label, b, k, evidences, conditions) for label, b, k in COMPONENTS}

    # The golden-case gate first: hand-written cases only, before any validation patient.
    golden: dict[str, Any] = {}
    for label, ranker in rankers.items():
        others = [ALSO_GOLDEN[label]] if label in ALSO_GOLDEN else []
        extra = {name: _open(name, b, k, evidences, conditions) for name, b, k in others}
        golden[label] = {}
        for alpha in GRID:
            per_model = {label: golden_ranks(ranker, store, labels, alpha)}
            per_model |= {name: golden_ranks(r, store, labels, alpha) for name, r in extra.items()}
            holds = all(c["holds"] for cases in per_model.values() for c in cases.values())
            golden[label][alpha] = {"holds": holds, "models": per_model}
        kept = [a for a in GRID if golden[label][a]["holds"]]
        say(f"{label}: the golden cases hold at α in {kept}")
    if args.golden_only:
        for label, by_alpha in golden.items():
            for name in by_alpha[GRID[0]]["models"]:
                for case_id, first in by_alpha[GRID[0]]["models"][name].items():
                    for condition in first["ranks"]:
                        ranks = " ".join(
                            f"{a:.2f}:{by_alpha[a]['models'][name][case_id]['ranks'][condition]}"
                            for a in GRID
                        )
                        print(f"{label} [{name}] {case_id} {condition} rank by α: {ranks}")
        print(
            json.dumps({k: {str(a): g["holds"] for a, g in v.items()} for k, v in golden.items()})
        )
        return 0

    specs = load_evidence_specs(evidences)
    codes = sorted(evidence_codes_in_scope(conditions), key=code_sort_key)
    validate = pd.read_parquet(
        args.interim / "ddxplus_chestpain_validate.parquet", columns=exp019.COLUMNS
    )
    frames = exp019.draw(validate, codes, specs)
    drawn = exp019.digests(frames)
    if drawn != {"tokens": exp019.TOKEN_DIGEST, "asked": exp019.ASKED_DIGEST}:
        raise SystemExit(f"the masks give {drawn}, not EXP-019's: the run is invalid (§3.7)")
    say(f"masks reproduce EXP-019's digests {drawn['tokens']} / {drawn['asked']}")

    critical_ids = must_not_miss_ids()
    graph, flags, in_d, ml, records, truth, critical = {}, {}, {}, {}, {}, {}, {}
    for level, frame in frames.items():
        records[level] = frame.to_dict("records")
        truth[level] = np.asarray([IDS.index(c) for c in frame["label_condition_id"]])
        critical[level] = np.asarray([c in critical_ids for c in frame["label_condition_id"]])
        graph[level] = graph_matrix(frame, store, labels)
        flags[level] = flag_matrix(frame)
        in_d[level] = differential_matrix(frame)
        for label, ranker in rankers.items():
            X = ranker.encoder.transform_sparse(frame[[c for c in exp019.ENCODED if c in frame]])
            ml[(label, level)] = np.asarray(ranker.model.predict_proba(X))
        say(f"scored the graph, the red flags and the models at {level}")

    results: dict[str, Any] = {
        "numpy": np.__version__,
        "digests": drawn,
        "realised": exp019.realised(frames),
        "grid": list(GRID),
        "ml_floor": ML_FLOOR,
        "components": {},
    }
    for label, _, _ in COMPONENTS:
        own = {}
        for level in frames:
            # The model's columns are LABELS, the registry's first 13 (checked in _open), so its
            # column indices are registry indices too.
            ranks = true_ranks(orders(ml[(label, level)]), truth[level])
            own[str(level)] = point_metrics(ranks, critical[level])
        sweep = []
        for alpha in GRID:
            entry: dict[str, Any] = {"alpha": alpha, "levels": {}}
            for level in frames:
                fused = fuse_arrays(
                    ml[(label, level)], graph[level], MODEL_COLUMNS, 1 - alpha, alpha
                )
                order = orders(fused)
                entry["levels"][str(level)] = {
                    **point_metrics(true_ranks(order, truth[level]), critical[level]),
                    **dissection_effects(order, truth[level], critical[level], in_d[level]),
                }
            sweep.append(entry)
        floor = {level: m["must-not-miss recall@3"] for level, m in own.items()}
        gate = {a: g["holds"] for a, g in golden[label].items()}
        chosen = choose(sweep, floor, gate)
        results["components"][label] = {
            "ml_alone": own,
            "golden": {str(a): g for a, g in golden[label].items()},
            "sweep": sweep,
            "chosen": chosen,
        }
        say(f"{label}: α = {chosen['alpha']} ({chosen['rule']})")

    critical_rows = {level: np.flatnonzero(critical[level]) for level in frames}
    for label, _, _ in COMPONENTS:
        entry = results["components"][label]
        alpha = entry["chosen"]["alpha"]
        if alpha is None:
            continue
        entry["at_chosen"] = {}
        for level in frames:
            fused = fuse_arrays(ml[(label, level)], graph[level], MODEL_COLUMNS, 1 - alpha, alpha)
            with_flags = orders(fused, flags[level])
            alone = orders(fused)
            ranks = {
                "fused": true_ranks(alone, truth[level]),
                "fused + red flags": true_ranks(with_flags, truth[level]),
                "ML alone": true_ranks(orders(ml[(label, level)]), truth[level]),
                "B2": true_ranks(orders(graph[level]), truth[level]),
            }
            hits = {name: (r <= 3).tolist() for name, r in ranks.items()}
            serious = critical_rows[level].tolist()
            tests = {}
            for a, b in (
                ("fused", "ML alone"),
                ("fused", "B2"),
                ("fused + red flags", "ML alone"),
            ):
                tests[f"{a} vs {b}"] = {
                    "top3": mcnemar(hits[a], hits[b]),
                    "top3_must_not_miss": mcnemar(
                        [hits[a][i] for i in serious], [hits[b][i] for i in serious]
                    ),
                    # docs/05 §6, amendment 3a (errata item 4): MRR's own paired test.
                    "mrr": paired_bootstrap_difference(1.0 / ranks[a], 1.0 / ranks[b]),
                }
            # Red flags reorder but change no probability, so the calibration is the fused
            # ranking's; the flag layer's own sensitivity and precision do not depend on α
            # (EXP-019's).
            fused_cases = outcomes(records[level], alone, np.exp(fused))
            flag_cases = outcomes(records[level], with_flags, None)
            entry["at_chosen"][str(level)] = {
                "fused": {
                    "metrics": _metrics(fused_cases),
                    "f1": f1_by_condition(fused_cases),
                    "ece": expected_calibration_error(fused_cases),
                    "reliability": reliability_table(fused_cases),
                },
                "fused + red flags": {
                    "metrics": _metrics(flag_cases),
                    "f1": f1_by_condition(flag_cases),
                },
                **dissection_effects(alone, truth[level], critical[level], in_d[level]),
                "paired_tests": tests,
            }
        say(f"{label}: scored at α = {alpha} with intervals")

    out = args.interim / "exp006_fusion.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")
    for label, entry in results["components"].items():
        alpha = entry["chosen"]["alpha"]
        print(f"\n{label}: α = {alpha} ({entry['chosen']['rule']})")
        print("   α    mean MRR   must-not-miss@3 per level     dissection in top 3 per level")
        for e in entry["sweep"]:
            lv = e["levels"]
            mrr = np.mean([m["MRR"] for m in lv.values()])
            mnm = " ".join(f"{m['must-not-miss recall@3']:.4f}" for m in lv.values())
            dis = " ".join(f"{m['dissection in the top 3']:.3f}" for m in lv.values())
            print(f"  {e['alpha']:.2f}   {mrr:.4f}    {mnm}      {dis}")
        own = " ".join(f"{m['must-not-miss recall@3']:.4f}" for m in entry["ml_alone"].values())
        print(f"  ML alone must-not-miss@3 per level: {own}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
