"""EXP-007: calibration, Platt scaling against isotonic regression, fitted on validation (2b).

    python scripts/calibrate.py                     # stage 1: B1-LR′+aug, B1-LR+aug and B2
    python scripts/calibrate.py --no-graph --model "B1-DL′+aug=models/nightingale_b1_dl/b1_dl_asked_aug/seed42:deep"
    python scripts/calibrate.py --fused "B1-LR′+aug"   # stage 2: refused until EXP-006 has chosen α

**Fitting a calibrator is tuning on project data:** it runs on the laptop's CPU, where the owner
placed it (D-7, 2026-10-12), under the rule recorded in ``docs/08`` (EXP-007) before it first ran:

* **The methods** (``src/fusion/calibration.py``): none, temperature scaling, one-vs-rest Platt
  scaling and one-vs-rest isotonic regression. A condition without fitting patients (aortic
  dissection) keeps its raw probability, and calibration never changes the ranking.
* **Fitted at 100%** on validate (``docs/05`` §3.7), applied unchanged at 50% and 25%; a calibrator
  fitted at each reduced level is reported beside, at its own level only.
* **Out of sample by cross-fitting:** five folds of the patients (a permutation with seed 42). Each
  fold is calibrated by the calibrator fitted on the other four, at every level, so no patient is
  scored by a calibrator fitted on any copy of itself. The calibrator saved is refitted on everyone.
* **The figures** at each level: the Brier score (14 conditions) and the top-label ECE (10 bins),
  each with its 95% bootstrap interval; the ECE per condition (one-vs-rest) and its mean; the log
  loss; the reliability table; and how often the calibrated probabilities' largest is not the
  system's first-ranked condition. The golden cases' probabilities, raw and calibrated, beside.
* **The rule, for each system on its own:** the method with the lowest mean over the three levels
  of the cross-fitted Brier score; a method that fits less, within 0.0005 of it, is preferred.
* **Stage 1** calibrates systems whose validation rankings EXP-019 already published: the two
  logistic regressions trained like A0's component, and the graph alone (B2, circular, R-12).
  **Stage 2** (``--fused``) calibrates A0's fused score. It needs D-13's component and EXP-006's α,
  and refuses to run until ``data/interim/exp006_fusion.json`` records that α.

Validate only: the test split stays closed until Phase 4. It reuses EXP-019's masks and refuses to
run unless both recorded digests reproduce. Writes ``data/interim/exp007_calibration.json`` and each
system's chosen calibrator to ``models/calibration/`` (gitignored). Every figure is on DDXPlus,
which is synthetic; a probability is conditional on one of the 14 conditions (R-13).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import subprocess
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
import sweep_fusion as exp006  # noqa: E402

from src.conditions import CONDITIONS  # noqa: E402
from src.contracts import PatientCase  # noqa: E402
from src.ddxplus import code_sort_key, load_evidence_specs  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    bootstrap_ratio,
    brier_score,
    calibration_error_by_condition,
    calibration_error_interval,
    expected_calibration_error,
    outcome_from_record,
    reliability_table,
)
from src.fusion import fuse_arrays, fuse_scores  # noqa: E402
from src.fusion.calibration import METHODS, Calibrator  # noqa: E402
from src.medical_kg.crosswalk import expand_case  # noqa: E402
from src.ml.features import evidence_codes_in_scope  # noqa: E402
from src.reasoning.red_flags import evaluate_red_flags  # noqa: E402

IDS = tuple(c.id for c in CONDITIONS)
MODEL_COLUMNS = exp006.MODEL_COLUMNS
FOLDS = 5
FOLD_SEED = 42
MARGIN = 0.0005
"""A method that fits less is preferred when its mean Brier score is within this of the best."""
STAGE1 = (
    ("B1-LR′+aug", exp019.ASKED_AUG, "logreg"),
    ("B1-LR+aug", exp019.AUG, "logreg"),
)
EXP006 = REPO_ROOT / "data" / "interim" / "exp006_fusion.json"
OUT_DIR = REPO_ROOT / "models" / "calibration"


def folds(n: int, k: int = FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Each patient's fold, 0..k−1: a permutation with ``seed``, dealt out in turn."""
    assignment = np.empty(n, dtype=int)
    assignment[np.random.default_rng(seed).permutation(n)] = np.arange(n) % k
    return assignment


def cross_fitted(
    method: str,
    fit_p: np.ndarray,
    truth: np.ndarray,
    apply: dict[Any, np.ndarray],
    fold: np.ndarray,
) -> dict[Any, np.ndarray]:
    """Each fold's rows of every matrix in ``apply``, calibrated by ``method`` fitted on the
    other folds' rows of ``fit_p``. The rows of every matrix are the same patients."""
    out = {key: np.empty_like(p) for key, p in apply.items()}
    for k in np.unique(fold):
        inside = fold == k
        calibrator = Calibrator.fit(method, fit_p[~inside], truth[~inside], IDS)
        for key, p in apply.items():
            out[key][inside] = calibrator.transform(p[inside])
    return out


def choose(mean_brier: dict[str, float], margin: float = MARGIN) -> dict[str, Any]:
    """The pre-registered rule: the lowest mean Brier score, unless a method that fits less
    (earlier in METHODS) is within ``margin`` of it."""
    best = min(mean_brier.values())
    for method in METHODS:
        if method in mean_brier and mean_brier[method] <= best + margin:
            return {
                "method": method,
                "mean_brier": mean_brier[method],
                "best_mean_brier": best,
                "rule": "lowest mean cross-fitted Brier score over the three levels, a method "
                f"that fits less preferred within {margin}",
            }
    raise ValueError("no method to choose from")


def figures(records: list[dict], order: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    """Every §3.3 figure for one system's calibrated probabilities ``q`` under its ranking."""
    cases = [
        outcome_from_record(
            record,
            [IDS[j] for j in order[i]],
            probabilities=dict(zip(IDS, map(float, q[i]), strict=True)),
        )
        for i, record in enumerate(records)
    ]
    brier = brier_score(cases)
    by_condition = calibration_error_by_condition(cases)
    truth = np.asarray([IDS.index(r["label_condition_id"]) for r in records])
    given = q[np.arange(len(q)), truth]
    return {
        "brier": {"value": brier.value, "ci": list(bootstrap_ratio(brier))},
        "ece": {
            "value": expected_calibration_error(cases),
            "ci": list(calibration_error_interval(cases)),
        },
        "ece_by_condition": by_condition,
        "ece_by_condition_mean": float(np.mean(list(by_condition.values()))),
        "log_loss": float(-np.mean(np.log(np.clip(given, 1e-15, 1.0)))),
        "top_is_not_most_probable": float(np.mean(q.argmax(axis=1) != order[:, 0])),
        "reliability": reliability_table(cases),
    }


def ml_matrix(ranker: Any, frame: pd.DataFrame) -> np.ndarray:
    """A model's probabilities as 14 columns in registry order; aortic dissection 0."""
    X = ranker.encoder.transform_sparse(frame[[c for c in exp019.ENCODED if c in frame]])
    p = np.zeros((len(frame), len(IDS)))
    p[:, MODEL_COLUMNS] = np.asarray(ranker.model.predict_proba(X))
    return p


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def golden_scores(system: dict[str, Any], store: Any, labels: dict) -> dict[str, dict]:
    """Each golden case through the system as the pipeline runs it: the raw probabilities over
    the 14 conditions and the order (red flags first for a fused system with flags)."""
    out = {}
    for golden in yaml.safe_load(exp006.GOLDEN.read_text(encoding="utf-8"))["cases"]:
        case = expand_case(PatientCase(**golden["case"]), labels)
        kg = store.score_by_connectivity(case) if system["kind"] != "ml" else {}
        ml = system["ranker"].score(case) if system["kind"] != "graph" else {}
        if system["kind"] == "ml":
            total = sum(ml.values())
            p = {c: ml.get(c, 0.0) / total for c in IDS}
        elif system["kind"] == "graph":
            row = softmax_rows(np.array([[kg[c] for c in IDS]]))[0]
            p = dict(zip(IDS, row.tolist(), strict=True))
        else:
            fused = fuse_scores(ml, kg, 1 - system["alpha"], system["alpha"])
            p = {c: math.exp(fused[c]) for c in IDS}
        flagged = set(evaluate_red_flags(case)) if system.get("red_flags") else set()
        order = sorted(IDS, key=lambda c: (c not in flagged, -p[c], IDS.index(c)))
        out[golden["id"]] = {
            "order": order,
            "raw": p,
            "required": golden["expect"].get("must_include", []),
        }
    return out


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def chosen_alpha(label: str, path: Path = EXP006) -> float:
    """The α EXP-006 chose for ``label``; stage 2 runs only once it exists."""
    if not path.exists():
        raise SystemExit(
            f"--fused needs EXP-006's choice of α ({path} does not exist): the sweep waits for "
            "E-2's first item, then the owner's choice of where (D-7)"
        )
    entry = json.loads(path.read_text(encoding="utf-8")).get("components", {}).get(label)
    alpha = (entry or {}).get("chosen", {}).get("alpha")
    if alpha is None:
        raise SystemExit(
            f"EXP-006 chose no α for {label}, so there is no fused system to calibrate"
        )
    return float(alpha)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument(
        "--model",
        action="append",
        type=exp019.parse_model,
        metavar="LABEL=DIR:BACKEND",
        help="calibrate this component alone instead of stage 1's two (repeat it)",
    )
    parser.add_argument("--no-graph", action="store_true", help="leave B2 out")
    parser.add_argument(
        "--fused",
        metavar="LABEL",
        help="stage 2: the fused system with this component, at EXP-006's α (red flags on)",
    )
    parser.add_argument(
        "--out", type=Path, help="the JSON written (default: data/interim/exp007_calibration.json)"
    )
    parser.add_argument("--models-out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    def say(message: str) -> None:
        print(f"{message} ({time.perf_counter() - started:.0f} s)", flush=True)

    components = tuple(args.model) if args.model else STAGE1
    alpha = chosen_alpha(args.fused) if args.fused else None
    if args.fused and args.fused not in [label for label, _, _ in components]:
        parser.error(f"--fused {args.fused}: give its bundle with --model")

    conditions = args.interim / "ddxplus_chestpain_conditions.json"
    evidences = args.raw_dir / "release_evidences.json"
    from src.medical_kg.cardiac_kg import build_cardiac_kg
    from src.medical_kg.neo4j_store import kg_fingerprint
    from src.medical_kg.networkx_store import NetworkXGraphStore

    kg = build_cardiac_kg(
        conditions, args.interim / "ddxplus_evidences.json", REPO_ROOT / "data" / "raw" / "bodhi_s"
    )
    store = NetworkXGraphStore(kg)
    graph_print = kg_fingerprint(kg)[:12]
    labels = dict(store.graph.nodes(data="label"))
    rankers = {
        label: exp006._open(label, b, k, evidences, conditions) for label, b, k in components
    }

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
    levels = list(frames)
    for level in levels:
        if list(frames[level]["case_id"]) != list(validate["case_id"]):
            raise SystemExit("the masked frames are not in the patients' order")
    records = {level: frames[level].to_dict("records") for level in levels}
    truth = np.asarray([IDS.index(c) for c in validate["label_condition_id"]])
    fold = folds(len(validate))

    graph = {level: exp006.graph_matrix(frames[level], store, labels) for level in levels}
    say("scored the graph at every level")

    # Each system: its raw probabilities and its ranking, at every level.
    systems: dict[str, dict[str, Any]] = {}
    for label, bundle, backend in components:
        ranker = rankers[label]
        probabilities = {level: ml_matrix(ranker, frames[level]) for level in levels}
        meta = {
            "kind": "ml",
            "ml": {
                "backend": backend,
                "model_dir": bundle.relative_to(REPO_ROOT).as_posix(),
                "feature_fingerprint": ranker.encoder.fingerprint,
            },
        }
        if args.fused != label:
            systems[label] = {
                **meta,
                "ranker": ranker,
                "p": probabilities,
                "order": {lv: exp006.orders(p) for lv, p in probabilities.items()},
            }
        else:
            fused = {
                lv: fuse_arrays(
                    probabilities[lv][:, MODEL_COLUMNS], graph[lv], MODEL_COLUMNS, 1 - alpha, alpha
                )
                for lv in levels
            }
            flags = {lv: exp006.flag_matrix(frames[lv]) for lv in levels}
            p = {lv: softmax_rows(f) for lv, f in fused.items()}
            systems[f"A0 ({label}, α = {alpha})"] = {
                **meta,
                "kind": "fused",
                "alpha": alpha,
                "graph": {"fingerprint": graph_print},
                "red_flags": True,
                "ranker": ranker,
                "p": p,
                "order": {lv: exp006.orders(fused[lv], flags[lv]) for lv in levels},
                "order_without_flags": {lv: exp006.orders(fused[lv]) for lv in levels},
            }
        say(f"scored {label} at every level")
    if not args.no_graph and not args.fused:
        systems["B2"] = {
            "kind": "graph",
            "graph": {"fingerprint": graph_print},
            "circular": "R-12",
            "p": {lv: softmax_rows(g) for lv, g in graph.items()},
            "order": {lv: exp006.orders(g) for lv, g in graph.items()},
        }

    results: dict[str, Any] = {
        "experiment": "EXP-007",
        "commit": git_commit(),
        "numpy": np.__version__,
        "digests": drawn,
        "folds": {"k": FOLDS, "seed": FOLD_SEED},
        "margin": MARGIN,
        "systems": {},
    }
    args.models_out.mkdir(parents=True, exist_ok=True)
    for name, system in systems.items():
        p, order = system["p"], system["order"]
        entry: dict[str, Any] = {
            k: v for k, v in system.items() if k in ("kind", "ml", "alpha", "graph", "circular")
        }
        entry["methods"] = {}
        for method in METHODS:
            applied = cross_fitted(method, p[1.0], truth, p, fold)
            own = {
                lv: cross_fitted(method, p[lv], truth, {lv: p[lv]}, fold)[lv]
                for lv in levels
                if lv != 1.0
            }
            entry["methods"][method] = {
                "fitted_at_100": {
                    str(lv): figures(records[lv], order[lv], applied[lv]) for lv in levels
                },
                "fitted_at_level": {
                    str(lv): figures(records[lv], order[lv], q) for lv, q in own.items()
                },
            }
            if "order_without_flags" in system:
                entry["methods"][method]["fused_ranking_without_flags"] = {
                    str(lv): figures(records[lv], system["order_without_flags"][lv], applied[lv])
                    for lv in levels
                }
            say(f"{name}: {method} cross-fitted")
        mean_brier = {
            m: float(np.mean([f["brier"]["value"] for f in e["fitted_at_100"].values()]))
            for m, e in entry["methods"].items()
        }
        entry["mean_brier"] = mean_brier
        entry["chosen"] = choose(mean_brier)

        # The calibrators fitted on every patient at 100%: the one saved, and the golden cases.
        finals = {m: Calibrator.fit(m, p[1.0], truth, IDS) for m in METHODS}
        golden = golden_scores(system, store, labels)
        entry["golden"] = {
            case_id: {
                "first": g["order"][0],
                "required": {
                    c: {"rank": g["order"].index(c) + 1, "raw": g["raw"][c]} for c in g["required"]
                },
                "first_probability": {
                    m: finals[m].calibrate(g["raw"])[g["order"][0]] for m in METHODS
                },
                "required_probability": {
                    m: {c: finals[m].calibrate(g["raw"])[c] for c in g["required"]} for m in METHODS
                },
            }
            for case_id, g in golden.items()
        }
        method = entry["chosen"]["method"]
        about = {
            "experiment": "EXP-007",
            "system": name,
            **{k: v for k, v in entry.items() if k in ("kind", "ml", "alpha", "graph")},
            "fitted_on": {
                "split": "validate",
                "level": 1.0,
                "patients": len(validate),
                "digests": drawn,
            },
            "commit": results["commit"],
            "numpy": np.__version__,
            "rule": entry["chosen"]["rule"],
        }
        saved = Calibrator.fit(method, p[1.0], truth, IDS, about=about)
        slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()
        path = args.models_out / f"exp007_{slug}.json"
        path.write_text(json.dumps(saved.to_json(), indent=1, ensure_ascii=False), encoding="utf-8")
        entry["saved"] = (
            path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)
        )
        results["systems"][name] = entry
        say(f"{name}: {method} chosen; saved {entry['saved']}")

    out = args.out or args.interim / "exp007_calibration.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    for name, entry in results["systems"].items():
        print(f"\n{name}: chosen {entry['chosen']['method']}")
        print(
            f"  {'method':<12}{'level':>7}{'Brier':>10}{'ECE':>9}  {'95% CI':<17}{'ECE/cond':>9}{'own-level ECE':>15}"
        )
        for method, e in entry["methods"].items():
            for lv in levels:
                f = e["fitted_at_100"][str(lv)]
                own = e["fitted_at_level"].get(str(lv))
                ci = f"[{f['ece']['ci'][0]:.4f}, {f['ece']['ci'][1]:.4f}]"
                own_text = f"{own['ece']['value']:.4f}" if own else "—"
                print(
                    f"  {method:<12}{lv:>7.0%}{f['brier']['value']:>10.4f}{f['ece']['value']:>9.4f}"
                    f"  {ci:<17}{f['ece_by_condition_mean']:>9.4f}{own_text:>15}"
                )
        for case_id, g in entry["golden"].items():
            first = g["first_probability"]
            print(
                f"  {case_id} first {g['first']}: "
                + ", ".join(f"{m} {first[m]:.3f}" for m in METHODS)
            )
    print(f"\nWrote {out} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
