"""Tests for the B0/B1 baselines and their training job (task 1c).

Everything here is synthetic: toy matrices, and a made-up mini release of DDXPlus in the real file
formats. Fitting toy models on synthetic rows is testing, not a training run on project data (D-7).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.conditions import CONDITIONS, DDXPLUS_LABELS
from src.eval.metrics import ranking_from_scores
from src.ml.baselines import (
    LABELS,
    LogisticBaseline,
    PrevalencePrior,
    XGBoostBaseline,
    label_index,
    outcomes,
    rankings,
)
from src.ml.features import EvidenceEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_ddxplus_chestpain.py"
TRAINER = REPO_ROOT / "scripts" / "train_baselines.py"


def toy_data(rows_per_class: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Each class lights up its own column (95% of the time) over light noise (2%)."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(len(LABELS)), rows_per_class)
    X = (rng.random((len(y), len(LABELS) + 3)) < 0.02).astype(np.float32)
    X[np.arange(len(y)), y] = rng.random(len(y)) < 0.95
    X[:, -1] = rng.integers(18, 90, len(y))  # an unscaled column, like age
    return X, y


# --------------------------------------------------------------------------- #
# Labels and rankings
# --------------------------------------------------------------------------- #


def test_the_labels_are_the_13_trainable_conditions_in_registry_order():
    assert tuple(c.id for c in CONDITIONS if c.in_training_data) == LABELS
    assert len(LABELS) == 13 and "COND:aortic_dissection" not in LABELS


def test_label_index_refuses_an_untrainable_condition():
    assert label_index([LABELS[2], LABELS[0]]).tolist() == [2, 0]
    with pytest.raises(ValueError, match="aortic_dissection"):
        label_index(["COND:aortic_dissection"])


def test_rankings_break_ties_like_the_metrics_do():
    rng = np.random.default_rng(1)
    probabilities = np.round(rng.random((50, len(LABELS))), 1)  # plenty of ties
    for row, ranking in zip(probabilities, rankings(probabilities), strict=True):
        assert ranking == ranking_from_scores(dict(zip(LABELS, row, strict=True)))


# --------------------------------------------------------------------------- #
# The models
# --------------------------------------------------------------------------- #


def test_b0_ranks_every_patient_by_training_prevalence():
    y = label_index([LABELS[3]] * 5 + [LABELS[0]] * 3 + list(LABELS))
    b0 = PrevalencePrior.fit(y)
    assert b0.counts[3] == 6 and b0.counts[0] == 4 and sum(b0.counts) == 21
    ranked = rankings(b0.predict_proba(4))
    assert all(r == ranked[0] for r in ranked) and ranked[0][:2] == (LABELS[3], LABELS[0])
    assert PrevalencePrior.from_json(json.loads(json.dumps(b0.to_json()))) == b0


def test_the_logistic_baseline_learns_and_survives_its_json():
    X, y = toy_data()
    model = LogisticBaseline.fit(X, y, feature_fingerprint="toy")
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(y), 13)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities.argmax(axis=1) == y).mean() > 0.8
    reloaded = LogisticBaseline.from_json(
        json.loads(json.dumps(model.to_json())), feature_fingerprint="toy"
    )
    assert np.allclose(reloaded.predict_proba(sparse.csr_matrix(X)), probabilities)


def test_the_logistic_baseline_matches_scikit_learn():
    """The JSON model is the same function scikit-learn fitted, without scikit-learn."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MaxAbsScaler

    X, y = toy_data(seed=2)
    ours = LogisticBaseline.fit(X, y, feature_fingerprint="toy")
    scaler = MaxAbsScaler().fit(X)
    theirs = LogisticRegression(max_iter=1000, random_state=42).fit(scaler.transform(X), y)
    assert np.allclose(ours.predict_proba(X), theirs.predict_proba(scaler.transform(X)), atol=1e-8)


def test_xgboost_stops_early_and_survives_save_and_load(tmp_path):
    X, y = toy_data(rows_per_class=20)
    held_x, held_y = toy_data(rows_per_class=5, seed=3)
    model = XGBoostBaseline.fit(
        X, y, held_x, held_y, feature_fingerprint="toy", max_rounds=40, early_stopping=5
    )
    assert 0 <= model.best_iteration < 40
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(y), 13) and (probabilities.argmax(axis=1) == y).mean() > 0.8
    model.save(tmp_path)
    reloaded = XGBoostBaseline.load(tmp_path, feature_fingerprint="toy")
    assert np.allclose(reloaded.predict_proba(sparse.csr_matrix(X)), probabilities)


def test_xgboost_answers_the_same_for_dense_and_sparse_input():
    """XGBoost reads a sparse matrix's absent entries as missing but a dense array's zeros as
    values. The wrapper always hands it CSR, so the input's form cannot change an answer."""
    X, y = toy_data(rows_per_class=20)
    model = XGBoostBaseline.fit(X, y, X, y, feature_fingerprint="toy", max_rounds=20)
    assert np.array_equal(model.predict_proba(X), model.predict_proba(sparse.csr_matrix(X)))
    assert np.array_equal(model.predict_proba(X[:1]), model.predict_proba(X)[:1])


def test_a_model_refuses_another_feature_set(tmp_path):
    X, y = toy_data(rows_per_class=10)
    data = LogisticBaseline.fit(X, y, feature_fingerprint="toy").to_json()
    with pytest.raises(ValueError, match="retrain"):
        LogisticBaseline.from_json(data, feature_fingerprint="other")
    XGBoostBaseline.fit(X, y, X, y, feature_fingerprint="toy", max_rounds=3).save(tmp_path)
    with pytest.raises(ValueError, match="retrain"):
        XGBoostBaseline.load(tmp_path, feature_fingerprint="other")


def test_every_class_must_have_training_rows():
    X, y = toy_data(rows_per_class=5)
    keep = y != 4
    with pytest.raises(ValueError, match=LABELS[4]):
        LogisticBaseline.fit(X[keep], y[keep], feature_fingerprint="toy")


def test_outcomes_carry_the_differential_and_the_raw_scores():
    frame = pd.DataFrame(
        {
            "case_id": ["a"],
            "label_condition_id": [LABELS[1]],
            "label_differential": [
                [
                    {"pathology": "x", "condition_id": LABELS[1], "probability": 0.6},
                    {"pathology": "Anemia", "condition_id": None, "probability": 0.4},
                ]
            ],
        }
    )
    scores = np.linspace(0.0, 1.0, 13)[None, :]
    (outcome,) = outcomes(frame, scores)
    assert outcome.ranking[0] == LABELS[-1] and outcome.differential == {LABELS[1]}
    assert outcome.differential_size == 2 and outcome.probabilities[LABELS[-1]] == 1.0


# --------------------------------------------------------------------------- #
# The training job, end to end, on a synthetic mini release
# --------------------------------------------------------------------------- #

CODES = [f"E_{k}" for k in range(1, 14)]  # one tell-tale yes/no evidence per condition


def write_mini_release(root: Path, seed: int = 7) -> tuple[Path, Path]:
    """release_evidences.json, train.csv and validate.csv in DDXPlus's formats, plus the interim
    conditions file. Returns (raw dir, interim dir)."""
    raw, interim = root / "raw", root / "interim"
    raw.mkdir(parents=True)
    interim.mkdir()
    evidences = {
        code: {"data_type": "B", "default_value": 0, "is_antecedent": False, "possible-values": []}
        for code in CODES
    }
    evidences["E_56"] = {
        "data_type": "C",
        "default_value": 0,
        "is_antecedent": False,
        "possible-values": list(range(11)),
    }
    evidences["E_204"] = {
        "data_type": "C",
        "default_value": "V_10",
        "is_antecedent": True,
        "possible-values": ["V_10", "V_0"],
    }
    (raw / "release_evidences.json").write_text(json.dumps(evidences), encoding="utf-8")
    ddxplus_name = {c.id: name for name, c in DDXPLUS_LABELS.items()}
    conditions = {
        label: {
            "condition_id": label,
            "symptoms": [{"code": CODES[i]}, {"code": "E_56"}],
            "antecedents": [{"code": "E_204"}],
        }
        for i, label in enumerate(LABELS)
    }
    (interim / "ddxplus_chestpain_conditions.json").write_text(
        json.dumps(conditions), encoding="utf-8"
    )
    rng = random.Random(seed)
    for split, per_class in (("train", 40), ("validate", 15)):
        rows = []
        for i, label in enumerate(list(LABELS) * per_class + ["out-of-scope"] * 10):
            name = ddxplus_name.get(label, "Anemia")
            k = LABELS.index(label) if label in LABELS else rng.randrange(13)
            tokens = [CODES[k]] if rng.random() < 0.9 else []
            tokens += [rng.choice(CODES)] if rng.random() < 0.3 else []
            tokens += [f"E_56_@_{rng.randrange(11)}", rng.choice(["E_204_@_V_10", "E_204_@_V_0"])]
            other = ddxplus_name[LABELS[(k + 1) % 13]]
            rows.append(
                {
                    "AGE": 18 + (i * 7) % 70,
                    "SEX": "MF"[i % 2],
                    "PATHOLOGY": name,
                    "EVIDENCES": str(tokens),
                    "INITIAL_EVIDENCE": tokens[0].split("_@_")[0],
                    "DIFFERENTIAL_DIAGNOSIS": str([[name, 0.5], [other, 0.3], ["Anemia", 0.2]]),
                }
            )
        pd.DataFrame(rows).to_csv(raw / f"{split}.csv", index=False)
    return raw, interim


def _run(*command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_the_training_job_runs_end_to_end_on_a_mini_release(tmp_path):
    raw, interim = write_mini_release(tmp_path)
    for split in ("train", "validate"):
        built = _run(
            str(BUILDER), "--split", split, "--raw-dir", str(raw), "--out-dir", str(interim)
        )
        assert built.returncode == 0, built.stderr
    # The job must never read the test split: a corrupt file in its place changes nothing.
    (interim / "ddxplus_chestpain_test.parquet").write_bytes(b"not a parquet file")
    out = tmp_path / "out"
    trained = _run(
        str(TRAINER),
        "--interim",
        str(interim),
        "--raw-dir",
        str(raw),
        "--out",
        str(out),
        "--max-rounds",
        "30",
        "--early-stopping",
        "5",
        "--resamples",
        "50",
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr

    names = {p.name for p in out.iterdir()}
    assert names == {
        "b0_prevalence.json",
        "b1_logreg.json",
        "b1_xgboost.ubj",
        "b1_xgboost.meta.json",
        "features.json",
        "metrics.json",
        "run.json",
        "pip-freeze.txt",
    }
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    top1 = {
        key: next(m["value"] for m in report["metrics"] if m["name"] == "top-1 accuracy")
        for key, report in metrics["models"].items()
    }
    assert set(top1) == {"b0", "b1_logreg", "b1_xgboost"}
    assert top1["b1_xgboost"] > top1["b0"] and top1["b1_logreg"] > top1["b0"]
    assert set(metrics["mcnemar_top3"]) == {"b1_xgboost_vs_b0", "b1_xgboost_vs_b1_logreg"}

    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["rows"]["train"] == 13 * 40 and run["rows"]["validate"] == 13 * 15
    assert run["rows"]["train_fit"] + run["rows"]["train_holdout"] == 13 * 40
    features = json.loads((out / "features.json").read_text(encoding="utf-8"))
    encoder = EvidenceEncoder.from_release(
        raw / "release_evidences.json", interim / "ddxplus_chestpain_conditions.json"
    )
    assert features["fingerprint"] == encoder.fingerprint == run["features"]["fingerprint"]
    XGBoostBaseline.load(out, feature_fingerprint=encoder.fingerprint)  # loads, or raises


def test_the_training_job_explains_a_missing_parquet(tmp_path):
    result = _run(str(TRAINER), "--interim", str(tmp_path), "--raw-dir", str(tmp_path))
    assert result.returncode != 0
    assert "build_ddxplus_chestpain.py --split train" in result.stdout + result.stderr
