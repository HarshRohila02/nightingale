"""Calibration (2b, EXP-007): the four methods, their two rules, saving, and the pipeline's use.

Everything here runs on synthetic probabilities: fitting a calibrator on project data is
EXP-007's run (scripts/calibrate.py), not a test.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from src.conditions import CONDITIONS
from src.contracts import Finding, PatientCase
from src.eval.metrics import (
    CaseOutcome,
    calibration_error_by_condition,
    calibration_error_interval,
    expected_calibration_error,
)
from src.fusion.calibration import METHODS, Calibrator, RefusedCalibrator, _fit_isotonic
from src.pipeline import DiagnosisPipeline
from src.stubs import ConstantRanker, EmptyRetriever, InMemoryGraphStore, TemplateExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _config  # noqa: E402
import calibrate  # noqa: E402

IDS = tuple(c.id for c in CONDITIONS)
LABELS = ("a", "b", "c", "d")


def tempered_sample(n: int, temperature: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores that are p ∝ exp(logits), with the truth drawn from exp(logits / temperature): a
    model overconfident by exactly ``temperature``."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, len(LABELS))) * 3
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    q = np.exp(logits / temperature)
    q /= q.sum(axis=1, keepdims=True)
    truth = np.array([rng.choice(len(LABELS), p=row) for row in q])
    return p, truth


def log_loss(q: np.ndarray, truth: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(q[np.arange(len(truth)), truth], 1e-15, 1))))


# --------------------------------------------------------------------------- #
# The methods
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_probabilities(method):
    p, truth = tempered_sample(2000, 2.0, seed=0)
    q = Calibrator.fit(method, p, truth, LABELS).transform(p)
    assert q.shape == p.shape
    assert np.all(q >= 0)
    assert np.allclose(q.sum(axis=1), 1.0)


def test_none_leaves_the_scores_as_they_are():
    p, truth = tempered_sample(500, 2.0, seed=1)
    raw = p * 3.0  # rows need not sum to 1: they are divided by their sum
    assert np.allclose(Calibrator.fit("none", raw, truth, LABELS).transform(raw), p)


def test_temperature_scaling_recovers_a_known_overconfidence():
    p, truth = tempered_sample(20_000, 2.0, seed=2)
    calibrator = Calibrator.fit("temperature", p, truth, LABELS)
    assert calibrator.params["temperature"] == pytest.approx(2.0, rel=0.05)


def test_temperature_scaling_keeps_each_row_s_order():
    p, truth = tempered_sample(2000, 2.0, seed=3)
    q = Calibrator.fit("temperature", p, truth, LABELS).transform(p)
    assert np.array_equal(np.argsort(-p, axis=1), np.argsort(-q, axis=1))


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic"])
def test_each_method_improves_an_overconfident_model_on_new_patients(method):
    p, truth = tempered_sample(8000, 2.5, seed=4)
    fit, held = slice(0, 4000), slice(4000, None)
    calibrator = Calibrator.fit(method, p[fit], truth[fit], LABELS)
    assert log_loss(calibrator.transform(p[held]), truth[held]) < log_loss(p[held], truth[held])


def test_isotonic_is_pool_adjacent_violators():
    xs, ys = _fit_isotonic(np.array([0.1, 0.2, 0.2, 0.3, 0.4]), np.array([0, 1, 0, 0, 1.0]))
    # 0.2 averages to 0.5, then pools with 0.3's 0 into 1/3 over [0.2, 0.3].
    assert xs == [0.1, 0.2, 0.3, 0.4]
    assert ys == pytest.approx([0.0, 1 / 3, 1 / 3, 1.0])


def test_isotonic_is_monotone_in_its_own_column_before_renormalising():
    p, truth = tempered_sample(3000, 2.0, seed=5)
    calibrator = Calibrator.fit("isotonic", p, truth, LABELS)
    for xs, ys in zip(calibrator.params["x"], calibrator.params["y"], strict=True):
        assert np.all(np.diff(xs) > 0)
        assert np.all(np.diff(ys) >= 0)


def test_platt_stays_finite_when_a_condition_is_perfectly_separated():
    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    truth = np.array([0, 0, 1, 1])
    calibrator = Calibrator.fit("platt", p, truth, ("x", "y"))
    assert all(math.isfinite(v) for v in calibrator.params["a"] + calibrator.params["b"])
    assert np.all(np.isfinite(calibrator.transform(p)))


def test_a_row_isotonic_sends_to_zero_keeps_its_raw_shares():
    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
    truth = np.array([0, 0, 0, 1])
    calibrator = Calibrator.fit("isotonic", p, truth, ("x", "y"))
    # Isotonic gives 0 where no fitting patient with such a score had the condition; here,
    # made so for both columns, every row maps to 0 and 0.
    for j in (0, 1):
        calibrator.params["y"][j] = [0.0] * len(calibrator.params["y"][j])
    row = np.array([[0.2, 0.05]])
    assert np.allclose(calibrator.transform(row), row / row.sum())


# --------------------------------------------------------------------------- #
# Rule 1: a condition without fitting patients keeps its raw probability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", METHODS)
def test_a_condition_no_fitting_patient_had_keeps_its_raw_probability(method):
    """Aortic dissection on DDXPlus: no patient has it, so no method may learn to zero it."""
    p, truth = tempered_sample(3000, 2.0, seed=6)
    truth = np.where(truth == 3, 0, truth)  # nobody has "d"
    calibrator = Calibrator.fit(method, p, truth, LABELS)
    assert calibrator.fitted == (True, True, True, False)
    q = calibrator.transform(p)
    assert np.allclose(q[:, 3], p[:, 3])
    assert np.allclose(q[:, :3].sum(axis=1), 1 - p[:, 3])


# --------------------------------------------------------------------------- #
# Checks and saving
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", METHODS)
def test_a_calibrator_survives_json_unchanged(method):
    p, truth = tempered_sample(1000, 2.0, seed=7)
    calibrator = Calibrator.fit(method, p, truth, LABELS, about={"system": "toy"})
    text = json.dumps(calibrator.to_json())
    again = Calibrator.from_json(json.loads(text))
    assert again.about == {"system": "toy"}
    assert np.array_equal(again.transform(p), calibrator.transform(p))


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(format="pickle"),
        lambda d: d.update(version=2),
        lambda d: d.update(method="magic"),
        lambda d: d.update(fitted=[True]),
        lambda d: d["params"].update(temperature=-1.0),
    ],
)
def test_a_damaged_file_is_refused(change):
    p, truth = tempered_sample(200, 2.0, seed=8)
    data = Calibrator.fit("temperature", p, truth, LABELS).to_json()
    change(data)
    with pytest.raises(ValueError):
        Calibrator.from_json(data)


def test_isotonic_breakpoints_out_of_order_are_refused():
    p, truth = tempered_sample(500, 2.0, seed=9)
    data = Calibrator.fit("isotonic", p, truth, LABELS).to_json()
    data["params"]["x"][0] = list(reversed(data["params"]["x"][0]))
    with pytest.raises(ValueError):
        Calibrator.from_json(data)


@pytest.mark.parametrize(
    "rows",
    [np.array([[0.5, -0.1, 0.3, 0.3]]), np.array([[np.nan, 0.1, 0.1, 0.1]]), np.zeros((1, 4))],
)
def test_scores_that_are_not_probabilities_are_refused(rows):
    p, truth = tempered_sample(200, 2.0, seed=10)
    calibrator = Calibrator.fit("temperature", p, truth, LABELS)
    with pytest.raises(ValueError):
        calibrator.transform(rows)
    with pytest.raises(ValueError):
        calibrator.transform(p[:, :3])


def test_calibrating_one_case_by_name():
    p, truth = tempered_sample(500, 2.0, seed=11)
    calibrator = Calibrator.fit("temperature", p, truth, LABELS)
    out = calibrator.calibrate({"a": 0.7, "b": 0.3})
    assert set(out) == set(LABELS)
    assert out["c"] == pytest.approx(0.0, abs=1e-12) and out["d"] == pytest.approx(0.0, abs=1e-12)
    assert sum(out.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        calibrator.calibrate({"z": 1.0})


def test_a_calibrator_names_what_it_was_not_fitted_for():
    about = {"kind": "fused", "ml": {"backend": "logreg", "model_dir": "models/x/y"}, "alpha": 0.4}
    calibrator = Calibrator("none", LABELS, (True,) * 4, about=about)
    expected = {"kind": "fused", "backend": "logreg", "model_dir": "models\\x\\y", "alpha": 0.4}
    assert calibrator.mismatch(expected) is None
    assert "alpha" in calibrator.mismatch({**expected, "alpha": 0.5})
    assert "backend" in calibrator.mismatch({**expected, "backend": "deep"})
    assert "kind" in calibrator.mismatch({**expected, "kind": "ml"})


# --------------------------------------------------------------------------- #
# The metrics it is judged by
# --------------------------------------------------------------------------- #


def outcome(i: int, truth: str, p: dict[str, float]) -> CaseOutcome:
    ranking = tuple(sorted(p, key=lambda c: -p[c]))
    return CaseOutcome(str(i), truth, ranking, probabilities=p)


def test_the_ece_interval_is_zero_for_a_perfectly_calibrated_certain_system():
    cases = [outcome(i, "COND:x", {"COND:x": 1.0, "COND:y": 0.0}) for i in range(50)]
    assert calibration_error_interval(cases, resamples=100) == (0.0, 0.0)


def test_the_ece_interval_reproduces_with_its_seed():
    rng = np.random.default_rng(12)
    cases = []
    for i in range(400):
        x = float(rng.uniform(0.5, 1.0))
        cases.append(
            outcome(
                i, "COND:x" if rng.uniform() < 0.7 else "COND:y", {"COND:x": x, "COND:y": 1 - x}
            )
        )
    first = calibration_error_interval(cases, resamples=200, seed=3)
    assert first == calibration_error_interval(cases, resamples=200, seed=3)
    assert first[0] <= first[1]
    assert 0 <= expected_calibration_error(cases) <= 1


def test_the_ece_per_condition_is_one_vs_rest_and_skips_absent_conditions():
    # x gets 0.8 in every case and is true in half of them: |0.8 − 0.5| = 0.3 in one bin.
    cases = [
        outcome(0, "COND:x", {"COND:x": 0.8, "COND:y": 0.2}),
        outcome(1, "COND:y", {"COND:x": 0.8, "COND:y": 0.2}),
    ]
    by_condition = calibration_error_by_condition(cases, ["COND:x", "COND:y", "COND:z"])
    assert by_condition == pytest.approx({"COND:x": 0.3, "COND:y": 0.3})


# --------------------------------------------------------------------------- #
# EXP-007's script: folds, cross-fitting, the rule, and stage 2's guard
# --------------------------------------------------------------------------- #


def test_the_folds_are_balanced_and_fixed_by_their_seed():
    fold = calibrate.folds(1003)
    assert np.array_equal(fold, calibrate.folds(1003))
    counts = np.bincount(fold)
    assert len(counts) == calibrate.FOLDS and counts.max() - counts.min() <= 1


def test_cross_fitting_never_calibrates_a_patient_with_itself(monkeypatch):
    n = 50
    p = np.full((n, len(IDS)), 1.0)
    p[:, 0] = np.arange(n) + 2.0  # the row's own number, to see which rows a fit used
    truth = np.zeros(n, dtype=int)
    fold = calibrate.folds(n)
    seen: list[set[float]] = []
    real_fit = Calibrator.fit

    def spy(method, probabilities, truth, labels, **kwargs):
        seen.append(set(np.round(probabilities[:, 0] / probabilities.sum(axis=1), 12)))
        return real_fit("none", probabilities, truth, labels)

    monkeypatch.setattr(calibrate.Calibrator, "fit", staticmethod(spy))
    calibrate.cross_fitted("isotonic", p, truth, {1.0: p}, fold)
    shares = np.round(p[:, 0] / p.sum(axis=1), 12)
    for k, used in zip(np.unique(fold), seen, strict=True):
        assert not used & set(shares[fold == k])
        assert used == set(shares[fold != k])


def test_the_rule_takes_the_lowest_mean_brier_score():
    chosen = calibrate.choose({"none": 0.30, "temperature": 0.20, "platt": 0.10, "isotonic": 0.12})
    assert chosen["method"] == "platt"


def test_the_rule_prefers_the_method_that_fits_less_within_the_margin():
    chosen = calibrate.choose(
        {"none": 0.1004, "temperature": 0.1001, "platt": 0.1000, "isotonic": 0.0999}
    )
    assert chosen["method"] == "none"
    chosen = calibrate.choose({"none": 0.2, "temperature": 0.1003, "isotonic": 0.0999})
    assert chosen["method"] == "temperature"


def test_stage_two_waits_for_exp006(tmp_path):
    with pytest.raises(SystemExit, match="EXP-006"):
        calibrate.chosen_alpha("B1-LR′+aug", tmp_path / "missing.json")
    path = tmp_path / "exp006.json"
    path.write_text(json.dumps({"components": {"B1-LR′+aug": {"chosen": {"alpha": None}}}}))
    with pytest.raises(SystemExit, match="no α"):
        calibrate.chosen_alpha("B1-LR′+aug", path)
    path.write_text(json.dumps({"components": {"B1-LR′+aug": {"chosen": {"alpha": 0.35}}}}))
    assert calibrate.chosen_alpha("B1-LR′+aug", path) == 0.35


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


def toy_case() -> PatientCase:
    return PatientCase(
        case_id="toy",
        age=60,
        sex="M",
        findings=[Finding(concept_id="SYM:chest_pain", label="Chest pain")],
    )


def toy_pipeline(**kwargs) -> DiagnosisPipeline:
    return DiagnosisPipeline(
        ranker=kwargs.pop("ranker", ConstantRanker()),
        graph=InMemoryGraphStore(),
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
        **kwargs,
    )


def fitted_over_every_condition(method: str = "temperature") -> Calibrator:
    rng = np.random.default_rng(13)
    p = rng.dirichlet(np.ones(len(IDS)), size=500)
    truth = rng.integers(0, len(IDS) - 1, size=500)  # nobody has the last condition
    return Calibrator.fit(method, p, truth, IDS)


def test_without_a_calibrator_no_probability_is_shown():
    result = toy_pipeline().run(toy_case())
    assert all(c.calibrated_probability is None for c in result.candidates)
    assert "calibration" not in result.degraded_components


@pytest.mark.parametrize("method", METHODS)
def test_a_calibrator_adds_probabilities_and_never_reorders(method):
    plain = toy_pipeline().run(toy_case())
    calibrated = toy_pipeline(calibrator=fitted_over_every_condition(method)).run(toy_case())
    assert [c.condition_id for c in calibrated.candidates] == [
        c.condition_id for c in plain.candidates
    ]
    shown = [c.calibrated_probability for c in calibrated.candidates]
    assert all(v is not None and v >= 0 for v in shown)
    assert sum(shown) == pytest.approx(1.0)
    assert "calibration" not in calibrated.degraded_components


class _Degraded(ConstantRanker):
    degraded = True


def test_a_result_without_its_model_gets_no_probability():
    result = toy_pipeline(ranker=_Degraded(), calibrator=fitted_over_every_condition()).run(
        toy_case()
    )
    assert "ml" in result.degraded_components
    assert "calibration" in result.degraded_components
    assert all(c.calibrated_probability is None for c in result.candidates)


def test_a_refused_calibrator_is_reported():
    result = toy_pipeline(calibrator=RefusedCalibrator("fitted for another α")).run(toy_case())
    assert "calibration" in result.degraded_components
    assert all(c.calibrated_probability is None for c in result.candidates)


def config_with(file: str | None, alpha: float = 0.5) -> dict:
    return {
        "ml": {"backend": "logreg", "model_dir": "models/nightingale_b1_asked/b1_asked_aug"},
        "ranking": {"ml_weight": 1 - alpha, "kg_weight": alpha},
        "calibration": {"file": file},
    }


def test_the_configuration_names_no_calibrator_until_stage_two():
    assert _config.calibrator_from_config(config_with(None)) is None
    assert _config.load_config()["calibration"]["file"] is None


def test_the_configuration_refuses_a_calibrator_for_another_system(tmp_path):
    about = {
        "kind": "fused",
        "ml": {"backend": "logreg", "model_dir": "models/nightingale_b1_asked/b1_asked_aug"},
        "alpha": 0.5,
    }
    path = tmp_path / "calibrator.json"
    path.write_text(json.dumps(Calibrator("none", IDS, (True,) * len(IDS), about=about).to_json()))
    assert isinstance(_config.calibrator_from_config(config_with(str(path))), Calibrator)
    refused = _config.calibrator_from_config(config_with(str(path), alpha=0.3))
    assert isinstance(refused, RefusedCalibrator) and "alpha" in refused.reason
    missing = _config.calibrator_from_config(config_with(str(tmp_path / "none.json")))
    assert isinstance(missing, RefusedCalibrator)
