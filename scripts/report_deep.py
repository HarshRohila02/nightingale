"""EXP-021's full report: the transformer against every comparator, at every level and seed.

    python scripts/report_deep.py --no-golden          # every seed and level (Kaggle; ~25 min here)
    python scripts/report_deep.py --seeds 43 44 --no-golden
    python scripts/report_deep.py --golden-only        # the golden cases for all ten networks (seconds)

D-13 decided the ranker on seed 42 (``scripts/apply_safety_floor.py``). EXP-021 also promised
every figure, for both arms and all five seeds, at every level with its interval (``docs/08``):

* **Each system's metrics** at 100%, 50% and 25% on the recorded masks (both digests must
  reproduce): the two transformer arms and the four `+aug` B1 variants trained like them,
  B1-LR(′)+aug and B1-XGB(′)+aug, at the same seed.
* **The paired tests** (``docs/05`` §6 as amended), transformer against each comparator of its own
  prime: McNemar on top-3, over everyone and over must-not-miss patients, and the paired
  bootstrap of the per-case difference for MRR and for the must-not-miss top-3 hit.
* **D-13 at every seed** (the seeds' agreement): its first two checks here; the golden cases,
  which need the knowledge graph, with ``--golden-only`` on the laptop.

Scoring only, nothing is trained or tuned. Validate only: the test split stays closed until Phase 4.
Writes ``data/interim/exp021_report.json`` (or ``--out``).

On Kaggle the models come from a private dataset (``--models-from``): :func:`place_models` finds
the three bundles by name and puts them where this script reads them (``docs/11`` §4.5).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_reduced_evidence as exp019  # noqa: E402

from src.conditions import must_not_miss_ids  # noqa: E402
from src.ddxplus import code_sort_key, load_evidence_specs  # noqa: E402
from src.eval.metrics import mcnemar, paired_bootstrap_difference, reciprocal_rank  # noqa: E402
from src.eval.reports import score, top3_hits  # noqa: E402
from src.eval.safety_floor import safety_floor  # noqa: E402
from src.ml.features import evidence_codes_in_scope  # noqa: E402
from src.ml.ranker import ModelRanker, open_ranker  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
LEVELS = (1.0, 0.5, 0.25)
BUNDLES = ("nightingale_b1_asked", "nightingale_b1_seeds", "nightingale_b1_dl")
"""The model folders this report reads, under ``models/``."""


def systems(seed: int, models: Path) -> dict[str, tuple[Path, str]]:
    """{label: (bundle, backend)}: both transformer arms and the `+aug` B1 variants, at ``seed``."""
    if seed == 42:
        aug, asked = (
            models / "nightingale_b1_asked" / "b1_aug",
            models / "nightingale_b1_asked" / "b1_asked_aug",
        )
    else:
        aug = models / "nightingale_b1_seeds" / "b1_aug" / f"seed{seed}"
        asked = models / "nightingale_b1_seeds" / "b1_asked_aug" / f"seed{seed}"
    deep = models / "nightingale_b1_dl"
    return {
        "B1-DL′+aug": (deep / "b1_dl_asked_aug" / f"seed{seed}", "deep"),
        "B1-DL+aug": (deep / "b1_dl_aug" / f"seed{seed}", "deep"),
        "B1-LR′+aug": (asked, "logreg"),
        "B1-XGB′+aug": (asked, "xgboost"),
        "B1-LR+aug": (aug, "logreg"),
        "B1-XGB+aug": (aug, "xgboost"),
    }


PAIRS = (
    ("B1-DL′+aug", "B1-LR′+aug"),
    ("B1-DL′+aug", "B1-XGB′+aug"),
    ("B1-DL+aug", "B1-LR+aug"),
    ("B1-DL+aug", "B1-XGB+aug"),
)
"""Each transformer against the comparators of its own prime (``docs/05`` §4: same encoder, same rows)."""


def place_models(source: Path, models: Path) -> list[str]:
    """Put the three bundles under ``models``, from wherever a Kaggle dataset left them.

    Kaggle may keep an uploaded zip as it is or unpack it, and may nest it in a folder of its own
    name. So each bundle is found by its folder name anywhere under ``source``, or inside a zip
    there, and copied in. Returns the bundles found.
    """
    found = []
    models.mkdir(parents=True, exist_ok=True)
    for archive in sorted(source.rglob("*.zip")):
        with zipfile.ZipFile(archive) as z:
            for name in BUNDLES:
                members = [m for m in z.namelist() if f"{name}/" in m]
                if members and not (models / name).exists():
                    for member in members:
                        relative = member[member.index(f"{name}/") :]
                        target = models / relative
                        if member.endswith("/"):
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(z.read(member))
                    found.append(name)
    for name in BUNDLES:
        if (models / name).exists():
            continue
        for folder in sorted(p for p in source.rglob(name) if p.is_dir()):
            shutil.copytree(folder, models / name)
            found.append(name)
            break
    return found


def paired(cases_a: list, cases_b: list, critical: frozenset[str]) -> dict[str, Any]:
    """McNemar on top-3 (everyone, must-not-miss patients) and the paired bootstraps."""
    if [o.case_id for o in cases_a] != [o.case_id for o in cases_b]:
        raise ValueError("a paired comparison needs the same patients in the same order")
    hits_a, hits_b = top3_hits(cases_a), top3_hits(cases_b)
    serious = [i for i, o in enumerate(cases_a) if o.true_condition in critical]
    return {
        "top3": mcnemar(hits_a, hits_b),
        "top3_must_not_miss": mcnemar([hits_a[i] for i in serious], [hits_b[i] for i in serious]),
        "mrr_difference": paired_bootstrap_difference(
            reciprocal_rank(cases_a).numerator, reciprocal_rank(cases_b).numerator
        ),
        "must_not_miss_hit_difference": paired_bootstrap_difference(
            [float(hits_a[i]) for i in serious], [float(hits_b[i]) for i in serious]
        ),
    }


def golden_only(seeds: tuple[int, ...], models: Path, interim: Path, raw: Path) -> dict[str, Any]:
    """The golden cases with every transformer network, red flags off and on."""
    import apply_safety_floor
    from _config import fusion_weights_from_config, load_config

    from src.medical_kg.networkx_store import NetworkXGraphStore

    conditions = interim / "ddxplus_chestpain_conditions.json"
    evidences = raw / "release_evidences.json"
    store = NetworkXGraphStore.from_files(
        conditions, interim / "ddxplus_evidences.json", REPO_ROOT / "data" / "raw" / "bodhi_s"
    )
    labels = dict(store.graph.nodes(data="label"))
    weights = fusion_weights_from_config(load_config())
    out: dict[str, Any] = {}
    for seed in seeds:
        for label in ("B1-DL′+aug", "B1-DL+aug"):
            bundle, backend = systems(seed, models)[label]
            ranker = open_ranker(
                backend, model_dir=bundle, evidences_path=evidences, conditions_path=conditions
            )
            if not isinstance(ranker, ModelRanker):
                raise SystemExit(f"{label} seed {seed}: {ranker.reason}")
            runs = apply_safety_floor.golden(ranker, store, labels, weights)
            out[f"{label} seed {seed}"] = {
                "passed": all(r["passed"] for r in runs.values()),
                "runs": runs,
            }
            print(
                f"{label} seed {seed}: golden cases {'pass' if out[f'{label} seed {seed}']['passed'] else 'FAIL'}"
            )
    return out


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument("--models", type=Path, default=REPO_ROOT / "models")
    parser.add_argument(
        "--models-from", type=Path, help="copy the bundles from here first (a Kaggle dataset)"
    )
    parser.add_argument(
        "--no-golden",
        action="store_true",
        help="say so in the report: the golden cases are run separately (--golden-only)",
    )
    parser.add_argument("--golden-only", action="store_true", help="only the golden cases")
    parser.add_argument(
        "--limit", type=int, help="a smoke test: score only the first N patients (not a result)"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    seeds = tuple(args.seeds)

    def say(message: str) -> None:
        print(f"{message} ({time.perf_counter() - started:.0f} s)", flush=True)

    if args.models_from:
        say(f"placed {place_models(args.models_from, args.models)} under {args.models}")
    if args.golden_only:
        report = golden_only(seeds, args.models, args.interim, args.raw_dir)
        out = args.out or args.interim / "exp021_golden.json"
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out}")
        return 0 if all(r["passed"] for r in report.values()) else 1

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
    if args.limit:
        frames = {level: frame.iloc[: args.limit] for level, frame in frames.items()}
        say(f"SMOKE TEST: the first {args.limit} patients only; these figures are not results")
    critical = frozenset(must_not_miss_ids())

    report: dict[str, Any] = {
        "experiment": "EXP-021",
        "digests": drawn,
        "realised": exp019.realised(frames),
        "golden": "not run here: scripts/report_deep.py --golden-only" if args.no_golden else None,
        "smoke_test_patients": args.limit,
        "seeds": {},
    }
    for seed in seeds:
        cases: dict[tuple[str, float], list] = {}
        entry: dict[str, Any] = {"metrics": {}, "comparisons": [], "d13": {}}
        for label, (bundle, backend) in systems(seed, args.models).items():
            ranker = open_ranker(
                backend, model_dir=bundle, evidences_path=evidences, conditions_path=conditions
            )
            if not isinstance(ranker, ModelRanker):
                raise SystemExit(f"{label} seed {seed}: {ranker.reason}")
            for level in LEVELS:
                frame = frames[level]
                X = ranker.encoder.transform_sparse(
                    frame[[c for c in exp019.ENCODED if c in frame]]
                )
                result, cases[(label, level)] = score(
                    label, frame, ranker.model.predict_proba(X), labels=tuple(ranker.model.labels)
                )
                entry["metrics"].setdefault(label, {})[str(level)] = result["metrics"]
            say(f"seed {seed}: scored {label} at every level")
        for a, b in PAIRS:
            for level in LEVELS:
                entry["comparisons"].append(
                    {
                        "level": level,
                        "a": a,
                        "b": b,
                        **paired(cases[(a, level)], cases[(b, level)], critical),
                    }
                )
        for candidate, comparator in (("B1-DL′+aug", "B1-LR′+aug"), ("B1-DL+aug", "B1-LR+aug")):
            verdict = safety_floor(
                {lv: cases[(candidate, lv)] for lv in (1.0, 0.25)},
                {0.25: cases[(comparator, 0.25)]},
                golden_cases_pass=True,  # checked separately (--golden-only); see "golden" above
            )
            checks = [c for c in verdict["checks"] if "golden" not in c["check"]]
            entry["d13"][candidate] = {
                "comparator": comparator,
                "checks_1_and_2_pass": all(c["passed"] for c in checks),
                "checks": checks,
            }
            say(
                f"seed {seed}: {candidate} against {comparator}: D-13 checks 1-2 "
                + ("pass" if entry["d13"][candidate]["checks_1_and_2_pass"] else "FAIL")
            )
        report["seeds"][str(seed)] = entry
        out = args.out or args.interim / "exp021_report.json"
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
