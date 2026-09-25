"""Train and evaluate B0 and B1 on DDXPlus (task 1c: EXP-003 and EXP-004).

    python scripts/train_baselines.py                 # CPU
    python scripts/train_baselines.py --device cuda   # XGBoost on a cloud GPU
    python scripts/train_baselines.py --augment --out models/b1_aug                  # B1 +aug
    python scripts/train_baselines.py --asked-channel --augment --out models/b1_asked_aug  # B1′+aug

This is a training run on project data, so it runs only where the owner chooses (D-7). For B0 and
B1 that is Google Colab, through notebooks/colab_b0_b1.ipynb (docs/11 §4). It reads the train and
validate parquets only, never the test split (docs/05 §2).

1. Encode both splits with EvidenceEncoder, as sparse matrices.
2. B0: the train split's class frequencies.
3. B1: logistic regression on all of train; XGBoost on 90% of train, stopped early on the other
   10%. Validate is kept for the final scores alone, so they are an honest estimate.
4. Score all three on validate with src/eval/metrics.py: every docs/05 ranking and safety metric
   with its 95% bootstrap interval, F1 by condition, the calibration of the raw (uncalibrated)
   scores, and McNemar's test on top-3 correctness.
5. Write the bundle to --out (default models/b0_b1): the three models, metrics.json, run.json
   (commit, versions, device, timings), features.json and pip-freeze.txt.

**The R-18 retrain** (EXP-017): ``--augment`` adds one masked copy of every train patient
(src/ml/evidence_masks.py, a share drawn per patient from [0.1, 0.9]), and ``--asked-channel``
encodes which questions were asked (src/ml/features.py). The channel is constant at full evidence,
so it is only tried with ``--augment``: the two runs above give B1 +aug and B1′+aug, whose
comparison separates what the channel does from what the masked copies do. The XGBoost holdout is
split by patient, so no patient's copy is on the other side. **Validate is scored at full evidence
only here**; ``scripts/evaluate_reduced_evidence.py`` scores the 50% and 25% levels on the masks
docs/05 §3.7 fixes (EXP-019). Each bundle also records what every model answers to a patient with
no findings at all, where B1 XGBoost said atrial fibrillation with probability 1.000.

**Seeds** (docs/05 §6): ``--seed`` sets XGBoost's subsampling, the early-stopping holdout and the
masked copies. notebooks/colab_b1_seeds.ipynb trains seeds 43–46 of every XGBoost arm; the
seed-42 model stays the one used.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ddxplus import INPUT_COLUMNS  # noqa: E402
from src.eval.metrics import mcnemar  # noqa: E402
from src.eval.reports import score, top3_hits  # noqa: E402
from src.ml.baselines import (  # noqa: E402
    LogisticBaseline,
    PrevalencePrior,
    XGBoostBaseline,
    label_index,
)
from src.ml.evidence_masks import augment  # noqa: E402
from src.ml.features import EvidenceEncoder  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "case_id",
    "initial_evidence",  # defines a mask (--augment); never a model input
    *INPUT_COLUMNS,
    "label_condition_id",
    "label_differential",
]
ENCODED = ["age", "sex", "evidences", "asked"]
"""What the encoder reads: ``asked`` exists only on masked rows, and means everything elsewhere."""
EMPTY_ROWS = (
    {"age": 50, "sex": "M", "evidences": [], "asked": []},
    {"age": 50, "sex": "F", "evidences": [], "asked": []},
)
"""Patients with no findings, every question unasked: EXP-017's atrial-fibrillation probe."""


def load_split(interim: Path, split: str) -> pd.DataFrame:
    path = interim / f"ddxplus_chestpain_{split}.parquet"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Build it first: python scripts/build_ddxplus_chestpain.py --split {split}"
        )
    return pd.read_parquet(path, columns=COLUMNS)


def _run(command: list[str]) -> str | None:
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=REPO_ROOT)
        return result.stdout.strip()
    return None


def provenance(device: str) -> dict[str, Any]:
    """What docs/11 §4 asks every cloud run to record."""
    import scipy
    import sklearn
    import xgboost

    gpu = (
        _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if device == "cuda"
        else None
    )
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "uncommitted_changes": bool(_run(["git", "status", "--porcelain"])),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "device": device,
        "gpu": gpu,
    }


def variant(asked_channel: bool, augmented: bool) -> str:
    """The proposal's suffix for a model: ′ for the asked channel, +aug for masked copies."""
    return ("′" if asked_channel else "") + ("+aug" if augmented else "")


def empty_row_answers(encoder: EvidenceEncoder, models: dict[str, Any]) -> dict[str, Any]:
    """Each model's top condition, and its probability, for a patient with no findings."""
    rows = encoder.transform(list(EMPTY_ROWS))
    answers = {}
    for key, model in models.items():
        probabilities = np.asarray(model.predict_proba(rows))
        answers[key] = [
            {
                "sex": patient["sex"],
                "top": model.labels[int(p.argmax())],
                "probability": round(float(p.max()), 4),
            }
            for patient, p in zip(EMPTY_ROWS, probabilities, strict=True)
        ]
    return answers


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "b0_b1")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout", type=float, default=0.1, help="share of train for early stopping"
    )
    parser.add_argument("--max-rounds", type=int, default=1000)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument(
        "--resamples", type=int, default=1000, help="bootstrap resamples (docs/05 §6)"
    )
    parser.add_argument(
        "--asked-channel", action="store_true", help="encode which questions were asked (R-18)"
    )
    parser.add_argument(
        "--augment", action="store_true", help="add one masked copy of every train patient"
    )
    args = parser.parse_args(argv)
    if args.asked_channel and not args.augment:
        parser.error("--asked-channel is constant at full evidence; use it with --augment")
    suffix = variant(args.asked_channel, args.augment)

    from sklearn.model_selection import train_test_split

    started = datetime.now(timezone.utc)
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(step: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[step] = round(now - clock, 2)
        clock = now
        print(f"  {step}: {timings[step]:.1f} s", flush=True)

    print("Loading the train and validate parquets")
    train, validate = load_split(args.interim, "train"), load_split(args.interim, "validate")
    encoder = EvidenceEncoder.from_release(
        args.raw_dir / "release_evidences.json",
        args.interim / "ddxplus_chestpain_conditions.json",
        asked_channel=args.asked_channel,
    )
    lap("load")

    patients = len(train)
    fit_patients, holdout_patients = train_test_split(
        np.arange(patients),
        test_size=args.holdout,
        stratify=label_index(train["label_condition_id"]),
        random_state=args.seed,
    )
    if args.augment:
        print(f"Drawing one masked copy of each of {patients:,} train patients")
        train = augment(train, codes=encoder.codes, seed=args.seed)
        # Row i and row i + patients are the same patient: keep them on the same side.
        fit_rows = np.concatenate([fit_patients, fit_patients + patients])
        holdout_rows = np.concatenate([holdout_patients, holdout_patients + patients])
        lap("augment")
    else:
        fit_rows, holdout_rows = fit_patients, holdout_patients

    print(
        f"Encoding {len(train):,} + {len(validate):,} rows into {len(encoder.feature_names)} columns"
    )
    X_train = encoder.transform_sparse(train[[c for c in ENCODED if c in train]])
    X_validate = encoder.transform_sparse(validate[list(INPUT_COLUMNS)])
    y_train = label_index(train["label_condition_id"])
    lap("encode")

    print("B0: prevalence prior")
    b0 = PrevalencePrior.fit(y_train)
    lap("b0_fit")
    print("B1: logistic regression")
    logreg = LogisticBaseline.fit(
        X_train, y_train, feature_fingerprint=encoder.fingerprint, seed=args.seed
    )
    lap("logreg_fit")
    print(f"B1: XGBoost on {args.device}, early stopping on {len(holdout_rows):,} held-out rows")
    xgb = XGBoostBaseline.fit(
        X_train[fit_rows],
        y_train[fit_rows],
        X_train[holdout_rows],
        y_train[holdout_rows],
        feature_fingerprint=encoder.fingerprint,
        device=args.device,
        seed=args.seed,
        max_rounds=args.max_rounds,
        early_stopping=args.early_stopping,
    )
    lap("xgboost_fit")

    print(f"Scoring on validate ({args.resamples} bootstrap resamples per metric)")
    reports, hits = {}, {}
    for key, name, probabilities in (
        ("b0", "B0 prevalence prior", b0.predict_proba(len(validate))),
        ("b1_logreg", f"B1 logistic regression{suffix}", logreg.predict_proba(X_validate)),
        ("b1_xgboost", f"B1 XGBoost{suffix}", xgb.predict_proba(X_validate)),
    ):
        reports[key], cases = score(
            name, validate, probabilities, resamples=args.resamples, seed=args.seed
        )
        hits[key] = top3_hits(cases)
    comparisons = {
        "b1_xgboost_vs_b0": mcnemar(hits["b1_xgboost"], hits["b0"]),
        "b1_xgboost_vs_b1_logreg": mcnemar(hits["b1_xgboost"], hits["b1_logreg"]),
    }
    empty = empty_row_answers(encoder, {"b1_logreg": logreg, "b1_xgboost": xgb})
    lap("evaluate")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "b0_prevalence.json").write_text(json.dumps(b0.to_json(), indent=2), encoding="utf-8")
    (out / "b1_logreg.json").write_text(json.dumps(logreg.to_json()), encoding="utf-8")
    xgb.save(out)
    (out / "features.json").write_text(
        json.dumps(
            {"fingerprint": encoder.fingerprint, "names": list(encoder.feature_names)}, indent=1
        ),
        encoding="utf-8",
    )
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "split": "validate",
                "caveats": [
                    "closed-world: 13 conditions only (R-13)",
                    "synthetic DDXPlus patients",
                    "raw scores, not calibrated probabilities",
                    "full evidence only here: scripts/evaluate_reduced_evidence.py scores 50% and 25%",
                ],
                "variant": suffix,
                "models": reports,
                "mcnemar_top3": comparisons,
                "empty_row": empty,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    freeze = _run([sys.executable, "-m", "pip", "freeze"]) or ""
    (out / "pip-freeze.txt").write_text(freeze + "\n", encoding="utf-8")
    run = {
        **provenance(args.device),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "variant": {
            "asked_channel": args.asked_channel,
            "augment": {"low": 0.1, "high": 0.9} if args.augment else None,
        },
        "rows": {
            "train_patients": patients,
            "train": len(train),
            "train_fit": len(fit_rows),
            "train_holdout": len(holdout_rows),
            "validate": len(validate),
        },
        "class_counts": {
            "train": dict(Counter(train["label_condition_id"])),
            "validate": dict(Counter(validate["label_condition_id"])),
        },
        "features": {"fingerprint": encoder.fingerprint, "columns": len(encoder.feature_names)},
        "xgboost": {"best_iteration": xgb.best_iteration, "params": dict(xgb.params)},
        "logreg": dict(logreg.params),
        "timings_seconds": timings,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    print(f"\nValidate ({len(validate):,} patients), 95% bootstrap intervals:")
    wanted = ["top-1 accuracy", "top-3 accuracy", "MRR", "must-not-miss recall@3", "Recall@5"]
    for report in reports.values():
        by_name = {m["name"]: m for m in report["metrics"]}
        cells = [f"{n} {by_name[n]['value']:.3f}" for n in wanted]
        print(f"  {report['model']:<24} " + " · ".join(cells))
    print(f"  McNemar, top-3, XGBoost vs B0: p = {comparisons['b1_xgboost_vs_b0']['p_value']:.3g}")
    print("\nA patient with no findings (EXP-017: B1 XGBoost said atrial fibrillation, 1.000):")
    for key, answers in empty.items():
        cells = [f"{a['sex']}: {a['top']} {a['probability']:.3f}" for a in answers]
        print(f"  {key + suffix:<22} " + " · ".join(cells))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
