"""EXP-023: where each component goes wrong, at 100%, 50% and 25% evidence (3e) (validation split).

    python scripts/error_analysis.py                 # score, write the JSON, draw the figures
    python scripts/error_analysis.py --figures-only  # redraw docs/figures/ from the saved JSON

**Descriptive only.** Nothing is trained, tuned, fitted or chosen here, and no figure feeds a
decision. It reads the validation split only, at the three evidence levels of ``docs/05`` §3.7, on
EXP-019's recorded masks, and refuses to run unless both digests reproduce. The test split stays
closed until Phase 4.

**The components alone, never a fused ranking.** The team has not answered E-2, and the fused
validation rankings must not be read before it does (EXP-006). So the systems are:

* **B1-DL′+aug**, the configured ranker since D-13 (EXP-021; seed 42, backend ``deep``);
* **B1-LR′+aug**, its comparison and backup, and **B1-XGB′+aug** (EXP-018; seed 42);
* **B2**, the knowledge graph alone (EXP-005), **circular on DDXPlus (R-12)**: the graph is built
  from the definitions that generated these patients.

For each system and level: the overall figures it is read against; top-1 and top-3 recall per
condition; the top-1 confusion matrix (true → first-ranked) and the most frequent confusions;
where the must-not-miss misses (true condition outside the top 3) go; how the errors of one level
relate to the next (the masks are nested); and §3.3's calibration of the raw scores, top-label and
one-vs-rest per condition. Every share and every ECE has its 95% bootstrap interval (1,000
resamples, seed 42). The overall figures must reproduce EXP-019's and the transformer's own run,
and the top-label ECEs EXP-007's: the run records the largest difference.

Writes ``data/interim/exp023_error_analysis.json`` (aggregates only: counts, shares, tables; no
patient) and the figures in ``docs/figures/`` (SVG, drawn by :mod:`src.eval.svg`). Every figure is
on DDXPlus, which is synthetic, and the system is closed-world (R-13): a patient whose condition is
not among the 13 would be forced into one of them.
"""

from __future__ import annotations

import argparse
import contextlib
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

from src.conditions import BY_ID, CONDITIONS, must_not_miss_ids  # noqa: E402
from src.ddxplus import code_sort_key, load_evidence_specs  # noqa: E402
from src.eval import errors, svg  # noqa: E402
from src.ml.baselines import LABELS  # noqa: E402
from src.ml.features import evidence_codes_in_scope  # noqa: E402

IDS = tuple(c.id for c in CONDITIONS)
ROWS = LABELS
"""The true conditions: DDXPlus has no aortic dissection, so it is a column but never a row."""
LEVELS = ("1.0", "0.5", "0.25")
DEEP = REPO_ROOT / "models" / "nightingale_b1_dl"
MODELS = (
    ("B1-DL′+aug", DEEP / "b1_dl_asked_aug" / "seed42", "deep"),
    ("B1-LR′+aug", exp019.ASKED_AUG, "logreg"),
    ("B1-XGB′+aug", exp019.ASKED_AUG, "xgboost"),
)
SYSTEMS = (*(label for label, _, _ in MODELS), "B2")
OUT = REPO_ROOT / "data" / "interim" / "exp023_error_analysis.json"
FIGURES = REPO_ROOT / "docs" / "figures"
REFERENCES = {
    "B1-DL′+aug": DEEP / "reduced_evidence_seed42.json",
    "B1-LR′+aug": REPO_ROOT / "data" / "interim" / "exp019_reduced_evidence.json",
    "B1-XGB′+aug": REPO_ROOT / "data" / "interim" / "exp019_reduced_evidence.json",
    "B2": REPO_ROOT / "data" / "interim" / "exp019_reduced_evidence.json",
}
"""Where each system's overall figures were first recorded, which this run must reproduce."""
CALIBRATION_REFERENCES = {
    "B1-DL′+aug": REPO_ROOT / "data" / "interim" / "exp007_calibration_dl.json",
    "B1-LR′+aug": REPO_ROOT / "data" / "interim" / "exp007_calibration.json",
    "B2": REPO_ROOT / "data" / "interim" / "exp007_calibration.json",
}
"""EXP-007's raw ("none") top-label ECEs, which this run must reproduce."""
TOLERANCE = 1e-9

CODES = {
    "COND:nstemi_stemi": "MI",
    "COND:unstable_angina": "UA",
    "COND:stable_angina": "SA",
    "COND:pericarditis": "Peri",
    "COND:myocarditis": "Myo",
    "COND:acute_pulmonary_edema": "APE",
    "COND:atrial_fibrillation": "AF",
    "COND:psvt": "PSVT",
    "COND:pulmonary_embolism": "PE",
    "COND:spontaneous_pneumothorax": "PTX",
    "COND:boerhaave": "Boer",
    "COND:gerd": "GERD",
    "COND:panic_attack": "Panic",
    "COND:aortic_dissection": "AoD",
}
"""The heatmap's column codes; the rows carry the full names with these codes."""
DISPLAY = {
    "B1-DL′+aug": "B1-DL′+aug (the configured ranker)",
    "B1-LR′+aug": "B1-LR′+aug",
    "B1-XGB′+aug": "B1-XGB′+aug",
    "B2": "B2, the graph alone",
}
LEVEL_NAMES = {"1.0": "100%", "0.5": "50%", "0.25": "25%"}


def slug(label: str) -> str:
    """A file-name form of a model label: ``B1-DL′+aug`` → ``b1_dl_prime_aug``."""
    text = label.replace("′", "_prime").replace("+", "_").replace("-", "_").lower()
    return "_".join(part for part in text.split("_") if part)


def row_label(condition: str, *, code: bool = True) -> str:
    """A condition's printed name, its heatmap code, and * if it is must-not-miss."""
    name = BY_ID[condition].label
    if code:
        name += f" ({CODES[condition]})"
    return name + (" *" if condition in must_not_miss_ids() else "")


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


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_systems(
    frames: dict[float, pd.DataFrame], evidences: Path, conditions: Path, say: Any
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Every system's 14-column probabilities and its order, at every level.

    A model gives aortic dissection 0, so it ranks last and changes no top-k figure; B2 gives every
    condition its posterior (a softmax of the graph's log-likelihoods, uniform prior), as EXP-007
    computes it, and ranks by the log-likelihoods. Ties break by the registry order, as
    ``ranking_from_scores`` does.
    """
    import calibrate as exp007

    from src.medical_kg.cardiac_kg import build_cardiac_kg
    from src.medical_kg.networkx_store import NetworkXGraphStore

    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for label, bundle, backend in MODELS:
        ranker = exp006._open(label, bundle, backend, evidences, conditions)
        for level, frame in frames.items():
            p = exp007.ml_matrix(ranker, frame)
            out[(label, str(level))] = (p, exp006.orders(p))
        say(f"scored {label} at every level")
    kg = build_cardiac_kg(
        conditions,
        conditions.parent / "ddxplus_evidences.json",
        REPO_ROOT / "data" / "raw" / "bodhi_s",
    )
    store = NetworkXGraphStore(kg)
    labels = dict(store.graph.nodes(data="label"))
    for level, frame in frames.items():
        g = exp006.graph_matrix(frame, store, labels)
        out[("B2", str(level))] = (exp007.softmax_rows(g), exp006.orders(g))
    say("scored B2 at every level")
    return out


def analyse(cases: list) -> dict[str, Any]:
    """Every EXP-023 figure for one system at one level."""
    matrix = errors.confusion_matrix(cases, ROWS, IDS)
    return {
        "headline": errors.headline(cases),
        "recall_top1": errors.recall_by_condition(cases, 1, ROWS),
        "recall_top3": errors.recall_by_condition(cases, 3, ROWS),
        "confusion": {"rows": list(ROWS), "columns": list(IDS), "counts": matrix.tolist()},
        "top_confusions": errors.top_confusions(cases, 12),
        "must_not_miss_misses": errors.must_not_miss_misses(cases),
        "calibration": errors.calibration(cases, ROWS),
    }


def check_references(results: dict[str, Any]) -> dict[str, Any]:
    """The largest difference from the figures first recorded for each system (EXP-019, the
    transformer's own run, EXP-007). A difference above :data:`TOLERANCE` means this run does not
    score the systems as they were scored, and the run says so."""
    names = {
        "top-1 accuracy",
        "top-3 accuracy",
        "MRR",
        "must-not-miss recall@3",
        "dangerous false-negative rate",
    }
    out: dict[str, Any] = {}
    for system in SYSTEMS:
        worst, compared = 0.0, 0
        reference = json.loads(REFERENCES[system].read_text(encoding="utf-8"))["systems"][system]
        for level in LEVELS:
            recorded = {m["name"]: m for m in reference[level]["metrics"] if m["name"] in names}
            mine = results["systems"][system]["levels"][level]["headline"]
            for name, m in recorded.items():
                worst = max(worst, abs(m["value"] - mine[name]["value"]))
                worst = max(worst, *(abs(a - b) for a, b in zip(m["ci"], mine[name]["ci"], strict=True)))
                compared += 1
            raw = reference[level].get("calibration_of_raw_scores")
            if raw is not None:
                ece = results["systems"][system]["levels"][level]["calibration"]["top_label"]["ece"]
                worst = max(worst, abs(raw["ece"] - ece))
                compared += 1
        entry: dict[str, Any] = {
            "against": REFERENCES[system].relative_to(REPO_ROOT).as_posix(),
            "figures": compared,
            "largest_difference": worst,
        }
        if system in CALIBRATION_REFERENCES:
            path = CALIBRATION_REFERENCES[system]
            saved = json.loads(path.read_text(encoding="utf-8"))["systems"][system]["methods"]
            ece_worst = 0.0
            for level in LEVELS:
                recorded = saved["none"]["fitted_at_100"][level]["ece"]
                mine = results["systems"][system]["levels"][level]["calibration"]["top_label"]
                ece_worst = max(
                    ece_worst,
                    abs(recorded["value"] - mine["ece"]),
                    *(abs(a - b) for a, b in zip(recorded["ci"], mine["ci"], strict=True)),
                )
            entry["ece_against"] = path.relative_to(REPO_ROOT).as_posix()
            entry["ece_largest_difference"] = ece_worst
        entry["reproduced"] = all(
            entry.get(k, 0.0) <= TOLERANCE for k in ("largest_difference", "ece_largest_difference")
        )
        out[system] = entry
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()

    def say(message: str) -> None:
        print(f"{message} ({time.perf_counter() - started:.0f} s)", flush=True)

    conditions = args.interim / "ddxplus_chestpain_conditions.json"
    evidences = args.raw_dir / "release_evidences.json"
    specs = load_evidence_specs(evidences)
    codes = sorted(evidence_codes_in_scope(conditions), key=code_sort_key)
    validate = pd.read_parquet(
        args.interim / "ddxplus_chestpain_validate.parquet", columns=exp019.COLUMNS
    )
    frames = exp019.draw(validate, codes, specs)
    drawn = exp019.digests(frames)
    if drawn != {"tokens": exp019.TOKEN_DIGEST, "asked": exp019.ASKED_DIGEST}:
        raise SystemExit(f"the masks give {drawn}, not EXP-019's: the run is invalid (§3.7)")
    for frame in frames.values():
        if list(frame["case_id"]) != list(validate["case_id"]):
            raise SystemExit("the masked frames are not in the patients' order")
    say(f"masks reproduce EXP-019's digests {drawn['tokens']} / {drawn['asked']}")

    scored = score_systems(frames, evidences, conditions, say)
    # The labels are the same at every level: a mask removes answers, never the diagnosis.
    records = validate[["case_id", "label_condition_id", "label_differential"]].to_dict("records")
    results: dict[str, Any] = {
        "experiment": "EXP-023",
        "commit": git_commit(),
        "numpy": np.__version__,
        "split": "validate",
        "patients": len(validate),
        "digests": drawn,
        "realised": exp019.realised(frames),
        "bootstrap": {"resamples": 1000, "seed": 42, "level": 0.95},
        "fused": "none: E-2 is unanswered, so no fused ranking is scored (EXP-006)",
        "systems": {},
    }
    for label, bundle, backend in MODELS:
        results["systems"][label] = {
            "kind": "ml",
            "backend": backend,
            "model_dir": bundle.relative_to(REPO_ROOT).as_posix(),
            "levels": {},
        }
    results["systems"]["B2"] = {"kind": "graph", "circular": "R-12", "levels": {}}

    for system in SYSTEMS:
        by_level = {}
        for level in LEVELS:
            p, order = scored[(system, level)]
            by_level[level] = exp006.outcomes(records, order, p)
            results["systems"][system]["levels"][level] = analyse(by_level[level])
        results["systems"][system]["error_overlap"] = {
            f"top-{k}": {
                f"{LEVEL_NAMES[a]} → {LEVEL_NAMES[b]}": errors.error_overlap(
                    by_level[a], by_level[b], k
                )
                for a, b in (("1.0", "0.5"), ("0.5", "0.25"))
            }
            for k in (1, 3)
        }
        say(f"analysed {system}")
    results["reproduction"] = check_references(results)
    results["seconds"] = round(time.perf_counter() - started, 1)
    return results


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

COMMON = (
    "Validation split, 33,963 patients. DDXPlus is synthetic, so these figures describe its "
    "generator, not real patients. Closed world: only 13 conditions can be named (R-13). Evidence "
    "masked by docs/05 §3.7 (digests e99a7a8fbf792675 / af3a357f6454838a): the same patients at "
    "every level, each 25% history inside its 50% one; the masks remove answers at random, which "
    "a clinician's questions do not. The component alone, not fused. EXP-023."
)
CIRCULAR = (
    " B2 is the knowledge graph alone, built from the DDXPlus definitions that generated these "
    "patients, so its figures are circular (R-12)."
)


def _common(system: str) -> str:
    return COMMON + (CIRCULAR if system == "B2" else "")


def _scores_note(system: str) -> str:
    if system == "B2":
        return (
            "Raw scores: the graph's posterior, a softmax of its log-likelihoods. EXP-007's rule "
            "chose Platt scaling for B2; the raw scores are drawn here."
        )
    return "Raw scores, which EXP-007's rule keeps uncalibrated for this model."


def figures(results: dict[str, Any], out_dir: Path) -> list[Path]:
    """Every EXP-023 figure, from the results alone, so ``--figures-only`` needs no scoring."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def write(name: str, text: str) -> None:
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    for system in SYSTEMS:
        entry = results["systems"][system]["levels"]
        name = DISPLAY[system]
        panels = [
            {
                "level": level,
                "rows": entry[level]["calibration"]["top_label"]["reliability"],
                "ece": entry[level]["calibration"]["top_label"]["ece"],
                "ci": entry[level]["calibration"]["top_label"]["ci"],
            }
            for level in LEVELS
        ]
        write(
            f"exp023_reliability_{slug(system)}.svg",
            svg.reliability_figure(
                panels,
                title=f"{name}: top-label reliability at 100%, 50% and 25% evidence",
                subtitle=_scores_note(system),
                notes=(
                    "Each point is a 0.1-wide bin of the probability given to the first-ranked "
                    "condition: across, the bin's mean probability; up, the share of its patients "
                    "whose first-ranked condition is the true one. The diagonal is perfect "
                    "calibration. Hollow points hold fewer than 20 patients; the bars below count "
                    "each bin's patients. ECE over 10 equal-width bins, with its 95% bootstrap "
                    "interval (1,000 resamples, seed 42). " + _common(system)
                ),
            ),
        )
        grid = [
            {
                "label": row_label(condition, code=False),
                "series": {
                    level: {
                        "rows": one["reliability"],
                        "ece": one["ece"],
                        "ci": one["ci"],
                    }
                    for level in LEVELS
                    for one in (entry[level]["calibration"]["one_vs_rest"][condition],)
                },
            }
            for condition in ROWS
        ]
        write(
            f"exp023_reliability_by_condition_{slug(system)}.svg",
            svg.reliability_grid(
                grid,
                title=f"{name}: one-vs-rest reliability per condition",
                subtitle=_scores_note(system),
                notes=(
                    "For each condition, every patient's probability for it, binned in tenths: "
                    "across, the bin's mean probability; up, the share of the bin's patients who "
                    "have the condition. The diagonal is perfect calibration; hollow points hold "
                    "fewer than 20 patients (their counts are in exp023_error_analysis.json and on "
                    "hover). Under each panel, the condition's ECE at each level with its 95% "
                    "bootstrap interval. * must-not-miss. " + _common(system)
                ),
            ),
        )
        for level in LEVELS:
            write(
                f"exp023_confusion_{slug(system)}_{LEVEL_NAMES[level].rstrip('%')}.svg",
                svg.confusion_figure(
                    entry[level]["confusion"]["counts"],
                    rows=ROWS,
                    columns=IDS,
                    row_labels={c: row_label(c) for c in ROWS},
                    column_codes=CODES,
                    title=f"{name}: top-1 confusions at {LEVEL_NAMES[level]} evidence",
                    subtitle=(
                        "Each row is the patients of one true condition; each column, the "
                        "condition ranked first. AoD, aortic dissection, has no DDXPlus patient, "
                        "so it is a column only. * must-not-miss."
                    ),
                    notes=_common(system),
                ),
            )
    for k in (3, 1):
        panels = [
            {
                "label": system if system != "B2" else "B2 (circular)",
                "rows": {
                    condition: {
                        level: results["systems"][system]["levels"][level][f"recall_top{k}"][
                            condition
                        ]
                        for level in LEVELS
                    }
                    for condition in ROWS
                },
            }
            for system in SYSTEMS
        ]
        write(
            f"exp023_recall_top{k}_by_level.svg",
            svg.recall_figure(
                panels,
                conditions=ROWS,
                labels={c: row_label(c, code=False) for c in ROWS},
                title=f"Top-{k} recall per condition at 100%, 50% and 25% evidence",
                subtitle=(
                    f"The share of each condition's patients whose true condition is ranked in "
                    f"the top {k}, with its 95% bootstrap interval (1,000 resamples, seed 42). "
                    "Each component alone; B1-DL′+aug is the configured ranker."
                ),
                notes="* must-not-miss. " + COMMON + CIRCULAR,
            ),
        )
    return written


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument("--out", type=Path, default=OUT, help="the JSON written (or read)")
    parser.add_argument("--figures", type=Path, default=FIGURES, help="where the SVGs go")
    parser.add_argument(
        "--figures-only", action="store_true", help="redraw the figures from the saved JSON"
    )
    parser.add_argument("--no-figures", action="store_true", help="write the JSON only")
    args = parser.parse_args(argv)

    if args.figures_only:
        results = json.loads(args.out.read_text(encoding="utf-8"))
    else:
        results = run(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.out} ({results['seconds']} s)")
        for system, check in results["reproduction"].items():
            status = "reproduced" if check["reproduced"] else "DOES NOT REPRODUCE"
            print(
                f"  {system}: {status} ({check['figures']} figures, largest difference "
                f"{check['largest_difference']:.1e}"
                + (
                    f"; ECE {check['ece_largest_difference']:.1e}"
                    if "ece_largest_difference" in check
                    else ""
                )
                + ")"
            )
    if not args.no_figures:
        written = figures(results, args.figures)
        size = sum(p.stat().st_size for p in written)
        print(f"Drew {len(written)} figures in {args.figures} ({size / 1024:.0f} KiB)")
    if not args.figures_only:
        for system in SYSTEMS:
            print(f"\n{system}")
            for level in LEVELS:
                entry = results["systems"][system]["levels"][level]
                head = entry["headline"]
                top = ", ".join(
                    f"{CODES[c['true']]}→{CODES[c['predicted']]} {c['count']}"
                    for c in entry["top_confusions"][:4]
                )
                misses = entry["must_not_miss_misses"]["all"]
                print(
                    f"  {LEVEL_NAMES[level]:>4}: top-1 {head['top-1 accuracy']['value']:.4f} "
                    f"top-3 {head['top-3 accuracy']['value']:.4f}; must-not-miss misses "
                    f"{misses['misses']} (first: {misses['first']}); confusions: {top}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
