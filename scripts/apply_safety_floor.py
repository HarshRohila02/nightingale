"""D-13 on the laptop: may the transformer replace logistic regression? (EXP-021)

    python scripts/apply_safety_floor.py              # seed 42: the verdict
    python scripts/apply_safety_floor.py --seed 43    # another seed, for the seeds' agreement

The rule was fixed before any transformer was trained (``src/eval/safety_floor.py``, EXP-021):
the safety targets at 100% and 25%, non-inferiority to the logistic regression trained like it at
25% (must-not-miss recall@3 and MRR, margin 0.005), and the golden cases with red flags on and off.
It is judged for both arms, each against its own comparator:

* **B1-DL′+aug** (with the "asked" channel) against **B1-LR′+aug**: A0′'s component, and the
  configured ranker if it passes;
* **B1-DL+aug** against **B1-LR+aug**: A0's.

Scoring only, nothing is trained or tuned. Validate only, on EXP-019's masks, and both digests must
reproduce. The golden cases run through the pipeline with the configured graph weights. Writes
``data/interim/exp021_d13_seed<N>.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_reduced_evidence as exp019  # noqa: E402
from _config import fusion_weights_from_config, load_config  # noqa: E402

from src.contracts import PatientCase  # noqa: E402
from src.ddxplus import code_sort_key, load_evidence_specs  # noqa: E402
from src.eval.reports import score  # noqa: E402
from src.eval.safety_floor import safety_floor  # noqa: E402
from src.medical_kg.crosswalk import expand_case  # noqa: E402
from src.ml.features import evidence_codes_in_scope  # noqa: E402
from src.ml.ranker import ModelRanker, open_ranker  # noqa: E402
from src.pipeline import DiagnosisPipeline  # noqa: E402
from src.stubs import EmptyRetriever, TemplateExplainer  # noqa: E402

GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"
DEEP = REPO_ROOT / "models" / "nightingale_b1_dl"
LEVELS = (1.0, 0.25)


def arms(seed: int) -> list[dict[str, Any]]:
    """Each transformer arm with the logistic regression trained like it, at ``seed``."""
    if seed == 42:
        aug, asked = exp019.AUG, exp019.ASKED_AUG
    else:
        seeds = REPO_ROOT / "models" / "nightingale_b1_seeds"
        aug, asked = seeds / "b1_aug" / f"seed{seed}", seeds / "b1_asked_aug" / f"seed{seed}"
    return [
        {
            "candidate": ("B1-DL′+aug", DEEP / "b1_dl_asked_aug" / f"seed{seed}", "deep"),
            "comparator": ("B1-LR′+aug", asked, "logreg"),
        },
        {
            "candidate": ("B1-DL+aug", DEEP / "b1_dl_aug" / f"seed{seed}", "deep"),
            "comparator": ("B1-LR+aug", aug, "logreg"),
        },
    ]


def golden(ranker: Any, store: Any, labels: dict, weights: dict) -> dict[str, Any]:
    """Every golden case with this ranker, red flags off and on, at the configured weights."""
    cases = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]
    out: dict[str, Any] = {}
    for flags in (False, True):
        pipeline = DiagnosisPipeline(
            ranker=ranker,
            graph=store,
            retriever=EmptyRetriever(),
            explainer=TemplateExplainer(),
            enable_red_flags=flags,
            **weights,
        )
        for g in cases:
            ranking = [
                c.condition_id
                for c in pipeline.run(expand_case(PatientCase(**g["case"]), labels)).candidates
            ]
            top_k = g["expect"].get("top_k", 3)
            required = g["expect"].get("must_include", [])
            out[f"{g['id']}, red flags {'on' if flags else 'off'}"] = {
                "ranks": {c: ranking.index(c) + 1 for c in required},
                "top_k": top_k,
                "passed": all(ranking.index(c) < top_k for c in required),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    def say(message: str) -> None:
        print(f"{message} ({time.perf_counter() - started:.0f} s)", flush=True)

    conditions = args.interim / "ddxplus_chestpain_conditions.json"
    evidences = args.raw_dir / "release_evidences.json"
    validate = pd.read_parquet(
        args.interim / "ddxplus_chestpain_validate.parquet", columns=exp019.COLUMNS
    )
    frames = exp019.draw(
        validate,
        sorted(evidence_codes_in_scope(conditions), key=code_sort_key),
        load_evidence_specs(evidences),
    )
    drawn = exp019.digests(frames)
    if drawn != {"tokens": exp019.TOKEN_DIGEST, "asked": exp019.ASKED_DIGEST}:
        raise SystemExit(f"the masks give {drawn}, not EXP-019's: the run is invalid (§3.7)")
    say(f"masks reproduce EXP-019's digests {drawn['tokens']} / {drawn['asked']}")

    from src.medical_kg.networkx_store import NetworkXGraphStore

    store = NetworkXGraphStore.from_files(
        conditions, args.interim / "ddxplus_evidences.json", REPO_ROOT / "data" / "raw" / "bodhi_s"
    )
    labels = dict(store.graph.nodes(data="label"))
    weights = fusion_weights_from_config(load_config())

    def opened(label: str, bundle: Path, backend: str) -> ModelRanker:
        ranker = open_ranker(
            backend, model_dir=bundle, evidences_path=evidences, conditions_path=conditions
        )
        if not isinstance(ranker, ModelRanker):
            raise SystemExit(f"{label}: {ranker.reason}")
        return ranker

    def scored(ranker: ModelRanker, label: str, levels: tuple[float, ...]) -> dict:
        reports, cases = {}, {}
        for level in levels:
            frame = frames[level]
            X = ranker.encoder.transform_sparse(frame[[c for c in exp019.ENCODED if c in frame]])
            reports[str(level)], cases[level] = score(
                label, frame, ranker.model.predict_proba(X), labels=tuple(ranker.model.labels)
            )
        return {"reports": reports, "cases": cases}

    results: dict[str, Any] = {
        "rule": "D-13, src/eval/safety_floor.py (pre-registered in EXP-021)",
        "seed": args.seed,
        "digests": drawn,
        "weights": weights,
        "verdicts": {},
    }
    for arm in arms(args.seed):
        name, bundle, backend = arm["candidate"]
        rival, rival_bundle, rival_backend = arm["comparator"]
        candidate = opened(name, bundle, backend)
        comparator = opened(rival, rival_bundle, rival_backend)
        ours = scored(candidate, name, LEVELS)
        theirs = scored(comparator, rival, (0.25,))
        say(f"scored {name} at 100% and 25%, {rival} at 25%")
        golden_runs = golden(candidate, store, labels, weights)
        verdict = safety_floor(
            ours["cases"],
            theirs["cases"],
            golden_cases_pass=all(g["passed"] for g in golden_runs.values()),
        )
        results["verdicts"][name] = {
            "comparator": rival,
            "bundle": bundle.relative_to(REPO_ROOT).as_posix(),
            **verdict,
            "golden": golden_runs,
            "metrics": {
                name: {lv: r["metrics"] for lv, r in ours["reports"].items()},
                rival: {lv: r["metrics"] for lv, r in theirs["reports"].items()},
            },
        }
        say(f"{name} against {rival}: {'PASSES' if verdict['passed'] else 'does not pass'} D-13")
        for check in verdict["checks"]:
            ci = f" [{check['ci'][0]:+.4f}, {check['ci'][1]:+.4f}]" if "ci" in check else ""
            value = check["value"]
            shown = f"{value:+.4f}" if "ci" in check else f"{value}"
            print(f"    {'pass' if check['passed'] else 'FAIL'}  {check['check']}: {shown}{ci}")

    out = args.out or args.interim / f"exp021_d13_seed{args.seed}.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
