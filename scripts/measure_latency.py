"""EXP-022: how long one ranking takes on this laptop's CPU, and what each model costs (docs/05 §3.6).

    python scripts/measure_latency.py                     # the models and the pipeline, one thread
    python scripts/measure_latency.py --deep models/nightingale_b1_dl/b1_dl_asked_aug/seed42

The supervisor asked for the models' complexity, "as real-time models are preferred" (2026-10-04),
and the owner decided that real time means the ranking updating as the findings are entered,
used if it is fast enough and helps (D-14). This measures whether it is.

* **The models**, one patient at a time (batch 1, the dashboard's case): B1-LR′+aug (the
  configured ranker), B1-XGB′+aug and the transformer B1-DL′+aug. Their size on disk, their
  parameters or tree nodes, multiply-adds per patient, the time to load, and the latency of
  ``predict_proba`` on single validate rows at full evidence: 100 warm-up calls, then 1,000 timed
  calls, in three passes with the models in a shuffled order, reported as p50 / p95 / p99 / max.
  Without ``--deep`` the transformer is an untrained one of the same architecture: its cost is
  fixed by the architecture, not the weights, so the timing holds for the trained model too.
* **The pipeline**, per golden case, stage by stage (``DiagnosisPipeline.run`` with the NetworkX
  graph, the template explainer and no retrieval; the LLM is not part of it): the ranker, the
  graph, the fusion, the red flags, the explanation, the safety check, and the rest. Stages are
  timed by wrapping the pipeline's collaborators in this script; no pipeline code changes.

Soft real time only: Python on Windows cannot guarantee a deadline, so p99 and the maximum are
reported, never promised. One laptop, one thread by default (``--threads``), plugged in. Writes
data/interim/exp022_latency.json. Scoring only, nothing is trained.
"""

from __future__ import annotations

import os
import sys

# Thread counts must be set before NumPy loads its BLAS.
_THREADS = "1"
for i, arg in enumerate(sys.argv):
    if arg == "--threads" and i + 1 < len(sys.argv):
        _THREADS = sys.argv[i + 1]
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, _THREADS)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _config import fusion_weights_from_config, load_config, open_graph_from_config  # noqa: E402

import src.pipeline as pipeline_module  # noqa: E402
from src.contracts import PatientCase  # noqa: E402
from src.ddxplus import INPUT_COLUMNS  # noqa: E402
from src.medical_kg.crosswalk import expand_case  # noqa: E402
from src.ml.deep import DeepRanker  # noqa: E402
from src.ml.ranker import ModelRanker, open_ranker  # noqa: E402
from src.pipeline import DiagnosisPipeline  # noqa: E402
from src.stubs import EmptyRetriever, TemplateExplainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ASKED_AUG = REPO_ROOT / "models" / "nightingale_b1_asked" / "b1_asked_aug"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"
PERCENTILES = (50, 95, 99)


def summary(seconds: list[float]) -> dict[str, float]:
    """Milliseconds: p50, p95, p99 and max."""
    ms = np.asarray(seconds) * 1000
    out = {f"p{q}": round(float(np.percentile(ms, q)), 4) for q in PERCENTILES}
    out["max"] = round(float(ms.max()), 4)
    out["mean"] = round(float(ms.mean()), 4)
    out["calls"] = len(ms)
    return out


def size_on_disk(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.exists())


def xgboost_nodes(model: Any) -> dict[str, int]:
    """Trees kept and their nodes, as the booster dumps them (up to the best iteration)."""
    trees = model.booster.get_dump()[: (model.best_iteration + 1) * len(model.labels)]
    nodes = sum(len([line for line in tree.splitlines() if line.strip()]) for tree in trees)
    return {"trees": len(trees), "nodes": nodes, "depth_max": int(model.params.get("max_depth", 0))}


def load_models(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """{label: {"ranker", "load_seconds", "size_bytes", "cost"}}."""
    models: dict[str, dict[str, Any]] = {}
    for label, backend, files in (
        ("B1-LR′+aug", "logreg", ["b1_logreg.json"]),
        ("B1-XGB′+aug", "xgboost", ["b1_xgboost.ubj", "b1_xgboost.meta.json"]),
    ):
        started = time.perf_counter()
        ranker = open_ranker(backend, model_dir=ASKED_AUG, **paths)
        seconds = time.perf_counter() - started
        if not isinstance(ranker, ModelRanker):
            raise SystemExit(f"{label}: {ranker.reason}")
        model = ranker.model
        if backend == "logreg":
            cost = {
                "parameters": int(model.coef.size + model.intercept.size),
                "macs_per_patient": int(model.coef.size),
            }
        else:
            cost = xgboost_nodes(model)
        models[label] = {
            "ranker": ranker,
            "load_seconds": round(seconds, 4),
            "size_bytes": size_on_disk([ASKED_AUG / f for f in files]),
            "cost": cost,
        }

    if args.deep:
        started = time.perf_counter()
        ranker = open_ranker("deep", model_dir=args.deep, **paths)
        seconds = time.perf_counter() - started
        if not isinstance(ranker, ModelRanker):
            raise SystemExit(f"B1-DL′+aug: {ranker.reason}")
        size, trained = size_on_disk([args.deep / "b1_dl.json"]), True
    else:
        encoder = models["B1-LR′+aug"]["ranker"].encoder
        untrained = DeepRanker.initialised(
            encoder.feature_names, feature_fingerprint=encoder.fingerprint
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "b1_dl.json"
            path.write_text(json.dumps(untrained.to_json()), encoding="utf-8")
            size = path.stat().st_size
            started = time.perf_counter()
            ranker = open_ranker("deep", model_dir=Path(folder), **paths)
            seconds = time.perf_counter() - started
        trained = False
    if not isinstance(ranker, ModelRanker):
        raise SystemExit(f"B1-DL′+aug: {ranker.reason}")
    models["B1-DL′+aug"] = {
        "ranker": ranker,
        "load_seconds": round(seconds, 4),
        "size_bytes": size,
        "cost": ranker.model.complexity(),
        "trained": trained,
    }
    return models


def time_models(models: dict[str, dict[str, Any]], rows: np.ndarray, args: argparse.Namespace):
    """Batch-1 ``predict_proba`` latency, three passes in a shuffled model order."""
    timings: dict[str, list[float]] = {label: [] for label in models}
    per_pass: dict[str, list[float]] = {label: [] for label in models}
    order = list(models)
    shuffle = random.Random(42)
    for _ in range(args.passes):
        shuffle.shuffle(order)
        for label in order:
            model = models[label]["ranker"].model
            for i in range(args.warmup):
                model.predict_proba(rows[i % len(rows)][None, :])
            seconds = []
            for i in range(args.calls):
                row = rows[i % len(rows)][None, :]
                started = time.perf_counter()
                model.predict_proba(row)
                seconds.append(time.perf_counter() - started)
            timings[label] += seconds
            per_pass[label].append(round(float(np.percentile(seconds, 95)) * 1000, 4))
    return {label: {**summary(t), "p95_by_pass": per_pass[label]} for label, t in timings.items()}


class Timer:
    """Accumulates the time spent in wrapped callables, by stage."""

    def __init__(self) -> None:
        self.spent: dict[str, float] = {}

    def wrap(self, stage: str, function: Any) -> Any:
        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.spent[stage] = self.spent.get(stage, 0.0) + time.perf_counter() - started

        return timed


def time_pipeline(ranker: Any, graph: Any, weights: dict, cases: dict, args) -> dict[str, Any]:
    """Each golden case through the pipeline, ``args.pipeline_calls`` times, stage by stage."""
    timer = Timer()
    originals = {
        name: getattr(pipeline_module, name)
        for name in ("fuse_scores", "evaluate_red_flags", "safety_check")
    }
    explainer = TemplateExplainer(ranker=ranker)  # the model's reasons included (2b)
    explainer.explain = timer.wrap("explanation", explainer.explain)  # type: ignore[method-assign]
    pipeline = DiagnosisPipeline(
        ranker=ranker, graph=graph, retriever=EmptyRetriever(), explainer=explainer, **weights
    )
    pipeline.ranker = _Proxy(ranker, "score", timer.wrap("ranker", ranker.score))
    pipeline.graph = _Proxy(
        graph, "score_by_connectivity", timer.wrap("graph", graph.score_by_connectivity)
    )
    labels = dict(graph.graph.nodes(data="label"))
    results: dict[str, Any] = {}
    try:
        pipeline_module.fuse_scores = timer.wrap("fusion", originals["fuse_scores"])
        pipeline_module.evaluate_red_flags = timer.wrap(
            "red_flags", originals["evaluate_red_flags"]
        )
        pipeline_module.safety_check = timer.wrap("safety_check", originals["safety_check"])
        for case_id, spec in cases.items():
            case = expand_case(PatientCase(**spec["case"]), labels)
            for _ in range(args.warmup):
                pipeline.run(case)
            timer.spent.clear()
            totals = []
            for _ in range(args.pipeline_calls):
                started = time.perf_counter()
                result = pipeline.run(case)
                totals.append(time.perf_counter() - started)
            stages = {k: round(v / args.pipeline_calls * 1000, 4) for k, v in timer.spent.items()}
            stages["rest"] = round(float(np.mean(totals)) * 1000 - sum(stages.values()), 4)
            results[case_id] = {
                "total_ms": summary(totals),
                "mean_ms_by_stage": stages,
                "top": result.candidates[0].condition_id,
                "degraded": list(result.degraded_components),
            }
    finally:
        for name, function in originals.items():
            setattr(pipeline_module, name, function)
    return results


class _Proxy:
    """``target`` with one method replaced; every other attribute passes through."""

    def __init__(self, target: Any, method: str, replacement: Any) -> None:
        self._target, self._method, self._replacement = target, method, replacement

    def __getattr__(self, name: str) -> Any:
        if name == self._method:
            return self._replacement
        return getattr(self._target, name)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument(
        "--deep", type=Path, help="a trained B1-DL′+aug bundle (default: untrained)"
    )
    parser.add_argument("--threads", default=_THREADS, help="BLAS threads (default 1)")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--pipeline-calls", type=int, default=300)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    paths = {
        "evidences_path": args.raw_dir / "release_evidences.json",
        "conditions_path": args.interim / "ddxplus_chestpain_conditions.json",
    }
    models = load_models(args, paths)
    validate = pd.read_parquet(
        args.interim / "ddxplus_chestpain_validate.parquet", columns=list(INPUT_COLUMNS)
    ).iloc[: args.calls]
    rows = models["B1-LR′+aug"]["ranker"].encoder.transform(validate)
    print(
        f"Timing {len(models)} models on {len(rows)} validate rows, batch 1, {args.threads} thread(s)"
    )
    latency = time_models(models, rows, args)

    config = load_config()
    graph = open_graph_from_config(config, "networkx")
    cases = {c["id"]: c for c in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]}
    weights = fusion_weights_from_config(config)
    print(f"Timing the pipeline on {len(cases)} golden cases, {args.pipeline_calls} runs each")
    pipelines = {
        label: time_pipeline(models[label]["ranker"], graph, weights, cases, args)
        for label in ("B1-LR′+aug", "B1-DL′+aug")
    }

    report = {
        "experiment": "EXP-022",
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blas_threads": args.threads,
            "cpu_count": os.cpu_count(),
        },
        "settings": {k: getattr(args, k) for k in ("warmup", "calls", "passes", "pipeline_calls")},
        "models": {
            label: {
                "load_seconds": m["load_seconds"],
                "size_bytes": m["size_bytes"],
                "cost": m["cost"],
                "trained": m.get("trained", True),
                "predict_proba_batch1_ms": latency[label],
            }
            for label, m in models.items()
        },
        "pipeline": pipelines,
        "graph": "networkx (the local build; Aura is timed separately, if at all)",
        "llm": "not part of the ranking path (Phase 3); it explains after the ranking is shown",
    }
    out = args.out or args.interim / "exp022_latency.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    print(
        f"\n{'model':<13}{'size':>10}{'load s':>9}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'max ms':>9}"
    )
    for label, m in report["models"].items():
        t = m["predict_proba_batch1_ms"]
        print(
            f"{label:<13}{m['size_bytes'] / 1e6:>8.2f}MB{m['load_seconds']:>9.3f}"
            f"{t['p50']:>9.3f}{t['p95']:>9.3f}{t['p99']:>9.3f}{t['max']:>9.2f}"
        )
    for label, per_case in pipelines.items():
        print(f"\nThe pipeline with {label} (ms; p95 of the whole run, then the mean by stage):")
        for case_id, r in per_case.items():
            stages = " · ".join(f"{k} {v:.2f}" for k, v in r["mean_ms_by_stage"].items())
            print(
                f"  {case_id}: p95 {r['total_ms']['p95']:.2f}, max {r['total_ms']['max']:.2f} | {stages}"
            )
    print(f"\nWrote {out} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
