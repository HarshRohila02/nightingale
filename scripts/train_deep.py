"""Train and evaluate the deep ranker B1-DL, a small transformer (decision D-12, EXP-021).

    python scripts/train_deep.py --device cuda --seed 42 --out models/b1_dl   # both arms, one seed
    python scripts/train_deep.py --device cuda --smoke --out /tmp/smoke      # 1% of rows, 1 epoch

A training run on project data, so it runs only where the owner chooses (D-7): a cloud GPU,
through notebooks/colab_b1_dl.ipynb (docs/11 §4.4). It reads the train and validate parquets
only, never the test split (docs/05 §2).

**Trained exactly like the comparators** (docs/05 §4: "trained like it"). At each seed it uses the
same rows as that seed's B1-LR+aug / B1-LR′+aug and B1-XGB+aug / B1-XGB′+aug:

1. ``augment(train, seed)``: every train patient plus one masked copy (src/ml/evidence_masks.py).
2. ``holdout_split``: XGBoost's patient-level 90/10 split, so the transformer stops early on the
   same held-out patients. (Logistic regression fits on all rows; the transformer, like XGBoost, on
   90%. That can only disadvantage it, and is disclosed.)
3. Two arms from one encoding: **B1-DL′+aug** with the "asked" channel (691 columns,
   ``ddd14019c66eff10``) and **B1-DL+aug** without it (the first 607 of them, ``3a0d5a5e01d7f427``,
   which the script checks are exactly the plain encoder's columns).

Then, per arm: :meth:`DeepRanker.fit` (PyTorch; AdamW, early stopping on the held-out
cross-entropy, nothing tuned, T-6); the NumPy forward pass checked against PyTorch's (``parity``);
validate scored at full evidence with the NumPy forward pass, the deployed one, with every docs/05
metric and interval; and the no-findings probe of EXP-017. The 50% and 25% levels are scored by
``scripts/evaluate_reduced_evidence.py --model ...:deep`` on docs/05 §3.7's masks.

Each arm writes ``<out>/b1_dl_asked_aug/seed<N>/`` or ``<out>/b1_dl_aug/seed<N>/``:
``b1_dl.json`` (the model, plain JSON), ``metrics.json``, ``run.json`` (commit, versions, GPU,
timings, the epoch history, parity), ``features.json`` and ``pip-freeze.txt``. ``run.json`` is
written last. An arm whose ``run.json`` exists is
skipped, so a dropped session resumes where it stopped.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Before torch is imported anywhere: deterministic cuBLAS (torch.use_deterministic_algorithms).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Then torch itself, before SciPy: on Windows, SciPy's OpenBLAS loaded first can stop torch's DLLs
# from initialising (WinError 1114).
with contextlib.suppress(ImportError):
    import torch  # noqa: E402, F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_baselines import (  # noqa: E402
    EMPTY_ROWS,
    ENCODED,
    _run,
    holdout_split,
    load_split,
    variant,
)

from src.ddxplus import INPUT_COLUMNS  # noqa: E402
from src.eval.reports import score  # noqa: E402
from src.ml.baselines import label_index  # noqa: E402
from src.ml.deep import DeepRanker, TrainingConfig, TransformerConfig, parity  # noqa: E402
from src.ml.evidence_masks import augment  # noqa: E402
from src.ml.features import EvidenceEncoder  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS = {"b1_dl_asked_aug": True, "b1_dl_aug": False}
"""Output folder → whether the arm has the "asked" channel."""
PARITY_ROWS = 2000


def provenance(device: str) -> dict[str, Any]:
    """What docs/11 §4 asks every cloud run to record."""
    versions: dict[str, Any] = {"numpy": np.__version__, "pandas": pd.__version__}
    for module in ("torch", "scipy", "sklearn"):
        with contextlib.suppress(ImportError):
            versions[module] = __import__(module).__version__
    gpu = None
    if device == "cuda":
        import torch

        gpu = torch.cuda.get_device_name(0)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "uncommitted_changes": bool(_run(["git", "status", "--porcelain"])),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": versions,
        "device": device,
        "gpu": gpu,
    }


def augmentation_digest(frame: pd.DataFrame) -> str:
    """A digest of the masked copies (their tokens and asked sets), so another run can check it
    trained on the same rows."""
    copies = frame[frame["mask_share"] < 1.0]
    lines = sorted(
        f"{c}\t{' '.join(e)}\t{' '.join(a)}"
        for c, e, a in zip(copies["case_id"], copies["evidences"], copies["asked"], strict=True)
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def no_findings(model: DeepRanker, encoder: EvidenceEncoder) -> list[dict[str, Any]]:
    """The model's top condition for a patient with no findings (EXP-017's probe)."""
    probabilities = model.predict_proba(encoder.transform(list(EMPTY_ROWS)))
    return [
        {
            "sex": p["sex"],
            "top": model.labels[int(row.argmax())],
            "probability": round(float(row.max()), 4),
        }
        for p, row in zip(EMPTY_ROWS, probabilities, strict=True)
    ]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "b1_dl")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=0.1, help="share of train for stopping")
    parser.add_argument(
        "--arms", default=",".join(ARMS), help="comma-separated: " + ", ".join(ARMS)
    )
    parser.add_argument("--max-epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1%% of train patients and one epoch: checks the job and times an epoch",
    )
    args = parser.parse_args(argv)
    arms = [a for a in args.arms.split(",") if a]
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        parser.error(f"unknown arms {unknown}; expected {list(ARMS)}")
    outputs = {arm: args.out / arm / f"seed{args.seed}" for arm in arms}
    todo = [arm for arm in arms if not (outputs[arm] / "run.json").exists()]
    for arm in arms:
        if arm not in todo:
            print(f"{arm} seed {args.seed}: already done, skipped")
    if not todo:
        return 0

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
    if args.smoke:
        keep = np.random.default_rng(args.seed).permutation(len(train))[
            : max(len(train) // 100, 200)
        ]
        train = train.iloc[np.sort(keep)].reset_index(drop=True)
        validate = validate.sample(n=min(2000, len(validate)), random_state=args.seed)
        validate = validate.reset_index(drop=True)
    release = (
        args.raw_dir / "release_evidences.json",
        args.interim / "ddxplus_chestpain_conditions.json",
    )
    asked_encoder = EvidenceEncoder.from_release(*release, asked_channel=True)
    plain_encoder = EvidenceEncoder.from_release(*release, asked_channel=False)
    width = len(plain_encoder.feature_names)
    if asked_encoder.feature_names[:width] != plain_encoder.feature_names:
        raise SystemExit("the asked encoder's first columns are not the plain encoder's")
    lap("load")

    patients = len(train)
    fit_rows, holdout_rows = holdout_split(
        label_index(train["label_condition_id"]), args.holdout, args.seed, augmented=True
    )
    print(f"Drawing one masked copy of each of {patients:,} train patients (seed {args.seed})")
    train = augment(train, codes=asked_encoder.codes, seed=args.seed)
    digest = augmentation_digest(train)
    print(f"  augmentation digest {digest}")
    lap("augment")

    print(f"Encoding {len(train):,} + {len(validate):,} rows")
    X_train = asked_encoder.transform_sparse(train[[c for c in ENCODED if c in train]])
    X_validate = asked_encoder.transform_sparse(validate[list(INPUT_COLUMNS)])
    y_train = label_index(train["label_condition_id"])
    lap("encode")

    training = TrainingConfig(max_epochs=1 if args.smoke else args.max_epochs)
    config = TransformerConfig()
    for arm in todo:
        asked = ARMS[arm]
        encoder = asked_encoder if asked else plain_encoder
        columns = slice(None) if asked else slice(0, width)
        suffix = variant(asked, True)
        X, X_val = X_train[:, columns], X_validate[:, columns]
        print(f"\n=== B1-DL{suffix}, seed {args.seed}: {X.shape[1]} columns, {args.device} ===")
        model = DeepRanker.fit(
            X[fit_rows],
            y_train[fit_rows],
            X[holdout_rows],
            y_train[holdout_rows],
            feature_names=encoder.feature_names,
            feature_fingerprint=encoder.fingerprint,
            config=config,
            training=training,
            seed=args.seed,
            device=args.device,
        )
        lap(f"{arm}_fit")
        checked = parity(model, X_val[:PARITY_ROWS])
        print(
            f"  parity with PyTorch on {checked['rows']} rows: max |Δp| "
            f"{checked['max_abs_difference']:.2e}, top-1 disagreements {checked['top1_disagreements']}"
        )
        probabilities = model.predict_proba(X_val)
        report, _ = score(
            f"B1-DL{suffix}",
            validate,
            probabilities,
            resamples=args.resamples,
            seed=args.seed,
            extra={"evidence_level": 1.0, "seed": args.seed},
        )
        empty = no_findings(model, encoder)
        lap(f"{arm}_evaluate")
        if args.smoke:
            epoch = model.history[0]["seconds"]
            steps_full = 2 * 255_900 * (1 - args.holdout) / training.batch_size
            steps_here = len(fit_rows) / training.batch_size
            projected = epoch * steps_full / max(steps_here, 1)
            print(
                f"  smoke: one epoch took {epoch:.1f} s here, so about {projected:.0f} s per epoch on "
                f"the full data, at most {projected * args.max_epochs / 60:.0f} min per arm"
            )
            continue

        out = outputs[arm]
        out.mkdir(parents=True, exist_ok=True)
        (out / "b1_dl.json").write_text(json.dumps(model.to_json()), encoding="utf-8")
        (out / "features.json").write_text(
            json.dumps({"fingerprint": encoder.fingerprint, "names": list(encoder.feature_names)}),
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
                        "scored with the NumPy forward pass, the one the laptop runs",
                    ],
                    "variant": suffix,
                    "models": {"b1_dl": report},
                    "empty_row": {"b1_dl": empty},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        run = {
            **provenance(args.device),
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": args.seed,
            "variant": {"asked_channel": asked, "augment": {"low": 0.1, "high": 0.9}},
            "augmentation_digest": digest,
            "rows": {
                "train_patients": patients,
                "train": len(train),
                "train_fit": len(fit_rows),
                "train_holdout": len(holdout_rows),
                "validate": len(validate),
            },
            "class_counts": {"validate": dict(Counter(validate["label_condition_id"]))},
            "features": {"fingerprint": encoder.fingerprint, "columns": len(encoder.feature_names)},
            "complexity": model.complexity(),
            "training": dict(model.params),
            "history": model.history,
            "parity": checked,
            "timings_seconds": dict(timings),
        }
        freeze = _run([sys.executable, "-m", "pip", "freeze"]) or ""
        (out / "pip-freeze.txt").write_text(freeze + "\n", encoding="utf-8")
        (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        by_name = {m["name"]: m["value"] for m in report["metrics"]}
        print(
            f"  validate, full evidence: top-1 {by_name['top-1 accuracy']:.4f}, MRR "
            f"{by_name['MRR']:.4f}, must-not-miss recall@3 {by_name['must-not-miss recall@3']:.4f}; "
            f"best epoch {model.params['best_epoch']} of {model.params['epochs_run']}"
        )
        print(
            "  no findings: "
            + " · ".join(f"{a['sex']} {a['top']} {a['probability']:.3f}" for a in empty)
        )
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
