"""EXP-012: every system and ablation of docs/05 §5, at 100%, 50% and 25% evidence, with the verdicts.

    python scripts/evaluate_ablations.py --digest-only                       # draw the masks; print the digests
    python scripts/evaluate_ablations.py --limit 2000 --untuned-alpha 0.5    # a short check of the plumbing
    python scripts/evaluate_ablations.py --alpha-file data/interim/exp006_fusion.json   # the dry run (5.2)
    python scripts/evaluate_ablations.py --split test --confirm-single-opening \\
        --expect-digests TOKENS,ASKED --alpha-file data/interim/exp006_fusion.json \\
        --reduced-claims both                                               # the single opening (5.5)

**This is a scoring run, not a training run, but it is long** (about 20-40 minutes on the full
validation split, most of it the transformer's forward pass and the bootstrap), so it runs only
where the owner says (D-7). Nothing is trained or tuned here.

What it does, for each seed 42-46 (``docs/05`` §6):

* draws the evaluation masks (``scripts/evaluate_reduced_evidence.py``'s, via ``src/ml/evidence_masks.py``)
  and refuses to score unless both digests reproduce (§3.7);
* scores the graph and the red-flag rules on the masked patients and the models' probabilities;
* ranks every system with ``src/eval/ablation_run.py`` (B0, B1-XGB, B1-LR, their ``+aug`` and
  ``′+aug`` variants, B1-DL, B2, and A0/A0′ with A1, A1b, A2, A4 and A7), each fused configuration
  at the α EXP-006 chose for its component (a seed's models are fused at the **seed-42 α**, never
  re-swept);
* compares them with McNemar's test and the paired MRR bootstrap, and judges H1-H6, the Minimum,
  Target and Stretch (``src/eval/verdicts.py``) under **both** readings of E-2, recording which is
  in force; the seed-42 model's verdict is the verdict, and the others say how many seeds agree;
* writes every system's metrics with 95% intervals (seed 42), the seed means and spreads, and the
  breakdowns by condition and by sex, to ``data/interim/exp012_<split>.json``.

**The test split.** ``--split test`` refuses unless ``v0.9-freeze`` is an ancestor of ``HEAD``,
nothing the freeze fixed has changed since, the tree is clean, ``--confirm-single-opening`` is
given, the masks' digests match ``--expect-digests`` (pre-registered in EXP-012), the α comes from
EXP-006 and the reading of E-2 is named. The opening is recorded before the labels are read; a
second one needs ``--reopen-because`` and is disclosed (``src/eval/freeze.py``).

**A run that is not a result** says so in its output (``valid_as_a_result``): a ``--limit`` subset,
an untuned α, or fewer than 1,000 resamples.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_reduced_evidence as exp019  # noqa: E402
import sweep_fusion as exp006  # noqa: E402

from src.conditions import must_not_miss_ids  # noqa: E402
from src.ddxplus import code_sort_key, load_evidence_specs  # noqa: E402
from src.eval import ablation, ablation_run, freeze  # noqa: E402
from src.eval.metrics import DEFAULT_RESAMPLES, evaluate, outcome_from_record  # noqa: E402
from src.eval.verdicts import FAMILIES, POLICIES  # noqa: E402
from src.fusion.calibration import Calibrator  # noqa: E402
from src.ml.baselines import LABELS, PrevalencePrior  # noqa: E402
from src.ml.features import evidence_codes_in_scope  # noqa: E402
from src.ml.ranker import ModelRanker, open_ranker  # noqa: E402

IDS = ablation.IDS
LABEL_COLUMNS = ("label_condition_id", "label_differential")
SEEDS_DIR = REPO_ROOT / "models" / "nightingale_b1_seeds"
DEEP = REPO_ROOT / "models" / "nightingale_b1_dl"
MARKER = REPO_ROOT / "data" / "interim" / "test_split_opened.json"
MODELS: dict[str, tuple[str, str]] = {
    "B1-XGB": ("b1", "xgboost"),
    "B1-LR": ("b1", "logreg"),
    "B1-XGB+aug": ("b1_aug", "xgboost"),
    "B1-LR+aug": ("b1_aug", "logreg"),
    "B1-XGB′+aug": ("b1_asked_aug", "xgboost"),
    "B1-LR′+aug": ("b1_asked_aug", "logreg"),
    "B1-DL+aug": ("b1_dl_aug", "deep"),
    "B1-DL′+aug": ("b1_dl_asked_aug", "deep"),
}
"""model label: (arm, backend). The arm and the seed give the bundle (:func:`bundle`)."""
BREAKDOWN = (
    "A0",
    "A0′",
    "B1-XGB",
    "B1-LR",
    "B1-DL+aug",
    "B1-DL′+aug",
    "B2",
)
DEFAULT_ALPHA = 0.5


def bundle(label: str, seed: int) -> Path:
    """The directory holding one model: seed 42's are the ones the earlier experiments scored."""
    arm, _ = MODELS[label]
    if arm.startswith("b1_dl"):
        return DEEP / arm / f"seed{seed}"
    if label == "B1-LR":
        return exp019.B1  # deterministic on fixed rows: fitted once (docs/05 §6)
    if seed == 42:
        return {"b1": exp019.B1, "b1_aug": exp019.AUG, "b1_asked_aug": exp019.ASKED_AUG}[arm]
    return SEEDS_DIR / arm / f"seed{seed}"


def file_hashes(directory: Path) -> dict[str, str]:
    """The SHA-256 (first 16 hex digits) of every file in a bundle, to say which models scored."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


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


def jsonable(value: Any) -> Any:
    """json.dumps' default: numpy scalars and arrays, tuple keys joined."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def key(*parts: Any) -> str:
    return " | ".join(str(p) for p in parts)


def chosen_alphas(path: Path | None, untuned: float | None) -> dict[str, dict[str, Any]]:
    """{component label: {"alpha", "source"}} from EXP-006, or the untuned dry-run value."""
    if untuned is not None:
        return {m: {"alpha": untuned, "source": "UNTUNED (dry run)"} for m in MODELS}
    assert path is not None
    components = json.loads(path.read_text(encoding="utf-8")).get("components", {})
    out = {}
    for label in MODELS:
        entry = components.get(label)
        alpha = (entry or {}).get("chosen", {}).get("alpha")
        if alpha is None:
            out[label] = {
                "alpha": DEFAULT_ALPHA,
                "source": (
                    f"{DEFAULT_ALPHA} (EXP-006 chose none for {label})"
                    if entry
                    else f"{DEFAULT_ALPHA} (not swept: {label} is not an EXP-006 component)"
                ),
            }
        else:
            out[label] = {"alpha": float(alpha), "source": f"EXP-006 ({path.name})"}
    return out


def parse_pairs(items: list[str] | None, kind: type) -> dict[str, Any]:
    out = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or name not in FAMILIES:
            raise SystemExit(f"{item!r}: expected FAMILY=VALUE with FAMILY one of {list(FAMILIES)}")
        out[name] = kind(value)
    return out


def load_calibrators(paths: dict[str, Path], alphas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, path in paths.items():
        calibrator = Calibrator.from_json(json.loads(path.read_text(encoding="utf-8")))
        if tuple(calibrator.labels) != IDS:
            raise SystemExit(f"{path}: its columns are not the registry's")
        why = calibrator.mismatch(
            {"kind": "fused", "alpha": alphas[FAMILIES[name].component]["alpha"]}
        )
        if why:
            raise SystemExit(f"{path} was not fitted for {name}'s system: {why}")
        out[name] = calibrator
    return out


def outcomes_for(records: list[dict], order: np.ndarray, flagged: np.ndarray | None) -> list:
    cases = []
    for i, record in enumerate(records):
        fired = [IDS[j] for j in np.flatnonzero(flagged[i])] if flagged is not None else ()
        cases.append(outcome_from_record(record, [IDS[j] for j in order[i]], flagged=fired))
    return cases


def uses_flags(label: str) -> bool:
    return label.startswith(("A0", "A1b", "A2", "A7"))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("validate", "test"), default="validate")
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--alpha-file", type=Path, help="EXP-006's result (exp006_fusion.json)")
    parser.add_argument(
        "--untuned-alpha", type=float, help="a dry run's α for every component: not a result"
    )
    parser.add_argument(
        "--reduced-claims",
        choices=sorted(POLICIES),
        help="the reading of E-2 in force (default: both, as approved); verdicts under both are kept",
    )
    parser.add_argument(
        "--calibrator", action="append", metavar="FAMILY=PATH", help="EXP-007 stage 2's calibrator"
    )
    parser.add_argument(
        "--unsupported-rate", action="append", metavar="FAMILY=RATE", help="from EXP-009/010"
    )
    parser.add_argument("--limit", type=int, help="score only the first N patients: not a result")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--digest-only", action="store_true", help="draw the masks; score nothing")
    parser.add_argument(
        "--no-full-metrics", action="store_true", help="skip the per-system intervals"
    )
    parser.add_argument("--confirm-single-opening", action="store_true", help="test split only")
    parser.add_argument("--expect-digests", help="TOKENS,ASKED, pre-registered; test split only")
    parser.add_argument("--reopen-because", help="why the test split is opened again")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    def say(message: str) -> None:
        print(f"{message} ({time.perf_counter() - started:.0f} s)", flush=True)

    test = args.split == "test"
    invalid: list[str] = []
    gate: dict[str, Any] | None = None
    if test:
        if args.limit or args.untuned_alpha is not None:
            parser.error("the test split is scored whole and with EXP-006's α")
        if not args.digest_only:
            for flag, given in (
                ("--confirm-single-opening", args.confirm_single_opening),
                ("--expect-digests", args.expect_digests),
                ("--alpha-file", args.alpha_file),
                ("--reduced-claims", args.reduced_claims),
            ):
                if not given:
                    parser.error(f"scoring the test split needs {flag}")
        try:
            gate = freeze.require_freeze(REPO_ROOT)
        except freeze.NotFrozen as why:
            raise SystemExit(f"THE TEST SPLIT STAYS CLOSED: {why}") from None
        say(f"frozen at {gate['tag']} ({gate['tag_commit'][:8]}); scoring {gate['commit'][:8]}")
    elif args.alpha_file is None and args.untuned_alpha is None and not args.digest_only:
        parser.error("give --alpha-file (EXP-006) or --untuned-alpha for a dry run")
    if args.untuned_alpha is not None:
        invalid.append("untuned α (a dry run)")
    if args.limit:
        invalid.append(f"only the first {args.limit} patients")
    if args.resamples != DEFAULT_RESAMPLES:
        invalid.append(f"{args.resamples} bootstrap resamples, not {DEFAULT_RESAMPLES}")
    if args.alpha_file is not None and not args.alpha_file.exists():
        parser.error(f"{args.alpha_file} does not exist: the sweep (EXP-006) has not run")

    conditions = args.interim / "ddxplus_chestpain_conditions.json"
    evidences = args.raw_dir / "release_evidences.json"
    specs = load_evidence_specs(evidences)
    codes = sorted(evidence_codes_in_scope(conditions), key=code_sort_key)
    parquet = args.interim / f"ddxplus_chestpain_{args.split}.parquet"
    # The masks and their digests need no label, so the labels are not read until they reproduce.
    inputs = [c for c in exp019.COLUMNS if c not in LABEL_COLUMNS]
    frame = pd.read_parquet(parquet, columns=inputs)
    if args.limit:
        frame = frame.head(args.limit).reset_index(drop=True)
    frames = exp019.draw(frame, codes, specs)
    drawn = exp019.digests(frames)
    sizes = exp019.realised(frames)
    print(f"{args.split}: {len(frame):,} patients; numpy {np.__version__}")
    print(
        f"mask digest (docs/05 §3.7): {drawn['tokens']}   asked-set digest (A-9): {drawn['asked']}"
    )
    for level, size in sizes.items():
        print(
            f"  {level}: {size['tokens']} tokens, {size['questions']} questions, asked {size['asked']}"
        )
    expected = (
        tuple(args.expect_digests.split(","))
        if args.expect_digests
        else (exp019.TOKEN_DIGEST, exp019.ASKED_DIGEST) if not test else None
    )
    reproduces = expected is not None and (drawn["tokens"], drawn["asked"]) == expected
    if args.digest_only:
        return 0
    if not reproduces:
        if args.limit:
            invalid.append("the masks of a subset cannot reproduce the digests")
        else:
            raise SystemExit(
                f"the masks give {tuple(drawn.values())}, not the pre-registered {expected}: the "
                "run is invalid (docs/05 §3.7)"
            )

    if test:
        # The digests reproduce: only now is the split opened, and the opening recorded before
        # the first label is read (src/eval/freeze.py).
        assert gate is not None
        openings = freeze.record_opening(MARKER, gate["commit"], reopen_because=args.reopen_because)
        gate["openings"] = openings
        say(f"the test split is opened (opening {len(openings)})")
    truth = pd.read_parquet(parquet, columns=["case_id", *LABEL_COLUMNS]).set_index("case_id")
    for lv_frame in frames.values():
        for column in LABEL_COLUMNS:
            lv_frame[column] = lv_frame["case_id"].map(truth[column])
    frame = frames[1.0]

    from src.medical_kg.networkx_store import NetworkXGraphStore

    store = NetworkXGraphStore.from_files(
        conditions, args.interim / "ddxplus_evidences.json", REPO_ROOT / "data" / "raw" / "bodhi_s"
    )
    graph_labels = dict(store.graph.nodes(data="label"))
    critical_ids = must_not_miss_ids()
    levels: dict[float, ablation_run.Level] = {}
    records: dict[float, list[dict]] = {}
    for level, lv_frame in frames.items():
        records[level] = lv_frame.to_dict("records")
        levels[level] = ablation_run.Level(
            truth=np.asarray([IDS.index(c) for c in lv_frame["label_condition_id"]]),
            critical=np.asarray([c in critical_ids for c in lv_frame["label_condition_id"]]),
            graph=exp006.graph_matrix(lv_frame, store, graph_labels),
            flagged=exp006.flag_matrix(lv_frame),
        )
        say(f"scored the graph and the red flags at {level}")

    alphas = chosen_alphas(args.alpha_file, args.untuned_alpha)
    alpha_values = {label: entry["alpha"] for label, entry in alphas.items()}
    calibrators = load_calibrators(parse_pairs(args.calibrator, Path), alphas)
    unsupported = parse_pairs(args.unsupported_rate, float)
    prior_path = exp019.B1 / "b0_prevalence.json"
    prior = PrevalencePrior.from_json(json.loads(prior_path.read_text(encoding="utf-8"))).priors

    runs: dict[int, ablation_run.SeedRun] = {}
    bundles: dict[str, Any] = {}
    for seed in args.seeds:
        ml: dict[str, dict[float, np.ndarray]] = {}
        for label, (_, backend) in MODELS.items():
            directory = bundle(label, seed)
            bundles[key(label, f"seed{seed}")] = {
                "dir": directory.relative_to(REPO_ROOT).as_posix(),
                "files": file_hashes(directory),
            }
            ranker = open_ranker(
                backend, model_dir=directory, evidences_path=evidences, conditions_path=conditions
            )
            if not isinstance(ranker, ModelRanker):
                raise SystemExit(f"{label} seed {seed}: {ranker.reason}")
            if tuple(ranker.model.labels) != LABELS:
                raise SystemExit(f"{label}: its classes are not in LABELS order")
            ml[label] = {}
            for level, lv_frame in frames.items():
                X = ranker.encoder.transform_sparse(
                    lv_frame[[c for c in exp019.ENCODED if c in lv_frame]]
                )
                ml[label][level] = np.asarray(ranker.model.predict_proba(X))
        say(f"seed {seed}: scored the models")
        runs[seed] = ablation_run.run_seed(
            levels,
            ml,
            alpha_values,
            prior=prior,
            calibrators=calibrators,
            unsupported_rate=unsupported,
            resamples=args.resamples,
        )
        say(f"seed {seed}: ranked, compared and judged")

    primary = runs[42] if 42 in runs else runs[min(runs)]
    results = summarise(args, primary, runs, records, levels, frames, say)
    in_force = args.reduced_claims or "both"
    results.update(
        {
            "experiment": "EXP-012",
            "split": args.split,
            "patients": len(frame),
            "commit": git_commit(),
            "numpy": np.__version__,
            "digests": drawn,
            "realised": sizes,
            "seeds": list(args.seeds),
            "alphas": alphas,
            "bundles": bundles,
            "reduced_claims_in_force": in_force,
            "identities": ablation.SAME_AS_A0,
            "freeze": gate,
            "valid_as_a_result": not invalid,
            "not_a_result_because": invalid,
        }
    )
    out = args.out or args.interim / f"exp012_{args.split}{'' if not invalid else '_dryrun'}.json"
    out.write_text(
        json.dumps(results, indent=1, ensure_ascii=False, default=jsonable), encoding="utf-8"
    )
    report(results, primary, in_force)
    print(f"\nWrote {out} in {time.perf_counter() - started:.0f} s")
    return 0


def summarise(args, primary, runs, records, levels, frames, say) -> dict[str, Any]:
    """Everything the JSON holds beyond the run's metadata."""
    out: dict[str, Any] = {"systems": {}, "comparisons": [], "verdicts": {}, "seed_summary": {}}
    if not args.no_full_metrics:
        for (label, level), order in primary.orders.items():
            flagged = levels[level].flagged if uses_flags(label) else None
            cases = outcomes_for(records[level], order, flagged)
            metrics = [
                {"name": m.name, "value": m.value, "ci": [m.ci_low, m.ci_high], "cases": m.cases}
                for m in evaluate(cases, resamples=args.resamples)
            ]
            out["systems"].setdefault(label, {})[str(level)] = {
                "metrics": metrics,
                "point": primary.points[(label, level)],
            }
        say("scored every system with its intervals")
    else:
        for (label, level), point in primary.points.items():
            out["systems"].setdefault(label, {})[str(level)] = {"point": point}
    for (a, b, level), comparison in primary.comparisons.items():
        out["comparisons"].append({"a": a, "b": b, "level": level, **comparison})
    out["ece"] = {key(a, lv): v for (a, lv), v in primary.ece.items()}
    if primary.ece_raw:
        out["ece_uncalibrated_A5"] = {key(a, lv): v for (a, lv), v in primary.ece_raw.items()}
    out["h4"] = {key(a, b): v for (a, b), v in primary.h4.items()}
    out["verdicts"] = primary.claims

    # Seeds: mean and spread of every system's headline metrics, and how many seeds agree with
    # each verdict (docs/05 §6: the seed-42 model's verdict is the verdict).
    labels = sorted({label for label, _ in primary.points})
    for label in labels:
        for level in levels:
            per_seed = {
                s: r.points[(label, level)] for s, r in runs.items() if (label, level) in r.points
            }
            out["seed_summary"].setdefault(label, {})[str(level)] = ablation.seed_summary(per_seed)
    agreement: dict[str, Any] = {}
    for family in primary.claims:
        for policy in POLICIES:
            for claim, verdict in primary.claims[family][policy].items():
                holding = [
                    s
                    for s, r in runs.items()
                    if r.claims.get(family, {}).get(policy, {}).get(claim, {}).get("status")
                    in ("supported", "met")
                ]
                agreement.setdefault(family, {}).setdefault(policy, {})[claim] = {
                    "verdict_seed_42": verdict["status"],
                    "seeds_where_it_holds": holding,
                    "of": len(runs),
                }
    out["seed_agreement"] = agreement

    sex = {level: frames[level]["sex"].to_numpy() for level in levels}
    by_condition: dict[str, Any] = {}
    by_sex: dict[str, Any] = {}
    for label in BREAKDOWN:
        for level in levels:
            cases = primary.cases.get((label, level))
            if cases is None:
                continue
            names = ablation_run.condition_names(levels[level].truth)
            by_condition.setdefault(label, {})[str(level)] = ablation_run.breakdown(
                cases, names, resamples=args.resamples
            )
            by_sex.setdefault(label, {})[str(level)] = ablation_run.breakdown(
                cases, sex[level], resamples=args.resamples
            )
    out["by_condition"], out["by_sex"] = by_condition, by_sex
    say("breakdowns by condition and by sex")
    return out


SHOWN = ("top-3 accuracy", "MRR", "must-not-miss recall@3")


def report(results: dict[str, Any], primary: ablation_run.SeedRun, in_force: str) -> None:
    if not results["valid_as_a_result"]:
        print("\n*** NOT A RESULT:", "; ".join(results["not_a_result_because"]), "***")
    for level in sorted({lv for _, lv in primary.points}, reverse=True):
        print(f"\nEvidence {level:.0%} (seed 42):")
        print(f"  {'system':<22}" + "".join(f"{n:>26}" for n in SHOWN))
        for label in sorted({label for label, lv in primary.points if lv == level}):
            p = primary.points[(label, level)]
            keys = ("top-3 accuracy", "MRR", "must-not-miss recall@3")
            print(f"  {label:<22}" + "".join(f"{p[k]:>26.4f}" for k in keys))
    for family in primary.claims:
        for policy in POLICIES:
            mark = " <- in force" if policy == in_force else ""
            print(f"\nVerdicts, {family}, E-2 reading '{policy}'{mark}:")
            for claim, verdict in primary.claims[family][policy].items():
                failed = verdict.get("failed")
                note = f"  ({failed} comparison(s) failed)" if failed else ""
                print(f"  {claim:<28} {verdict['status']}{note}")


if __name__ == "__main__":
    sys.exit(main())
