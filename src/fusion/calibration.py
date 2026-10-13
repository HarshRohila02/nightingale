"""Calibration: a system's scores as probabilities that mean what they say (2b, EXP-007).

``docs/05`` §3.3 and RQ5 (H5: post-calibration ECE < 0.10). No score is shown as a probability
until it passes calibration (§3.3, FR-4.5, ``docs/04`` F-2). A :class:`Calibrator` maps one
system's probability vector over the conditions to a calibrated one. It is fitted on validation
only (§2, §3.7), for one system: a calibrator fitted to one model, or to one fusion weight, says
nothing about another, so it records what it was fitted for (``about``) and the pipeline applies
it to that system only.

Four methods, fixed before any was fitted on project data (``docs/08`` EXP-007):

* ``none``: the scores as they are (ablation A5's).
* ``temperature``: temperature scaling (Guo et al. 2017), Platt scaling's one-parameter
  multi-class form: q ∝ p^(1/T), with T fitted by log loss. It keeps the conditions' order.
* ``platt``: Platt scaling one-vs-rest (Platt 1999; Zadrozny and Elkan 2002): for each condition,
  σ(a · logit p + b), fitted to Platt's smoothed targets, then renormalised.
* ``isotonic``: isotonic regression one-vs-rest (Zadrozny and Elkan 2002): for each condition,
  the pool-adjacent-violators fit of P(condition | p), interpolated linearly between its blocks,
  then renormalised. A row whose calibrated values are all 0 keeps its raw shares.

Two rules hold for every method:

1. **A condition with no patient in the fitting data keeps its raw probability**, and the
   calibrated conditions share the rest. DDXPlus has no aortic dissection, so a one-vs-rest fit
   would learn to give it 0 whatever the findings: for a must-not-miss condition, the one mistake
   the system exists to avoid.
2. **Calibration never changes the ranking.** The pipeline still orders the candidates by the fused
   score, red flags first; the calibrated probability is the number beside the rank. A one-vs-rest
   method can reorder the probabilities, so the two can disagree, and EXP-007 counts how often.

A calibrated probability is conditional on the patient having one of the conditions the system
knows (R-13, closed world): DDXPlus patients always do, real patients may not. Calibrators are
saved as JSON, never as pickles.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import numpy as np

__all__ = ["FORMAT", "METHODS", "Calibrator", "RefusedCalibrator"]

METHODS = ("none", "temperature", "platt", "isotonic")
"""In order of how much they fit: the order EXP-007's rule prefers when two are as good."""
FORMAT = "nightingale-calibrator"
VERSION = 1
EPS = 1e-12
"""Probabilities are clipped to [EPS, 1 − EPS] before a logit (Platt)."""
TINY = 1e-300
"""The floor before a logarithm (temperature): a tempered 1e-30 must stay negligible, and a floor
of EPS would raise it to EPS^(1/T), 0.25 at T = 20."""
TEMPERATURE_BOUNDS = (0.05, 20.0)


def _normalised(probabilities: Any, width: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != width:
        raise ValueError(f"expected one row per case and {width} columns, got shape {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = p.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("a row that sums to 0 has no probabilities to calibrate")
    return p / total


def _sigmoid(f: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * f))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def _tempered(log_p: np.ndarray, temperature: float) -> np.ndarray:
    z = log_p / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_temperature(p: np.ndarray, truth: np.ndarray) -> float:
    """T minimising the log loss of softmax(log p / T), searched on log T."""
    from scipy.optimize import minimize_scalar

    log_p = np.log(np.clip(p, TINY, 1.0))
    rows = np.arange(len(truth))

    def loss(log_t: float) -> float:
        z = log_p / math.exp(log_t)
        top = z.max(axis=1)
        return float(np.mean(top + np.log(np.exp(z - top[:, None]).sum(axis=1)) - z[rows, truth]))

    low, high = (math.log(b) for b in TEMPERATURE_BOUNDS)
    result = minimize_scalar(loss, bounds=(low, high), method="bounded", options={"xatol": 1e-7})
    return float(math.exp(result.x))


def _fit_platt(z: np.ndarray, positive: np.ndarray) -> tuple[float, float]:
    """(a, b) of σ(a z + b), by Newton's method on the log loss with Platt's (1999) targets:
    (N₊ + 1) / (N₊ + 2) for a positive and 1 / (N₋ + 2) for a negative. Starts at the identity
    (a = 1, b = 0), with step halving, so the loss never rises."""
    n_pos = float(positive.sum())
    n_neg = float(len(positive) - n_pos)
    target = np.where(positive, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    X = np.column_stack([z, np.ones_like(z)])

    def loss(w: np.ndarray) -> float:
        f = X @ w
        return float(np.sum(np.logaddexp(0.0, f) - target * f))

    w = np.array([1.0, 0.0])
    current = loss(w)
    for _ in range(100):
        s = _sigmoid(X @ w)
        grad = X.T @ (s - target)
        hessian = (X * (s * (1 - s))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, grad)
        rate = 1.0
        while rate > 1e-10:
            candidate = w - rate * step
            value = loss(candidate)
            if value <= current:
                break
            rate /= 2
        else:
            break
        w, previous, current = candidate, current, value
        if previous - current <= 1e-12 * max(1.0, abs(previous)):
            break
    return float(w[0]), float(w[1])


def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    """The non-decreasing least-squares fit of y on x (pool adjacent violators), as the
    breakpoints of a piecewise-linear function: each block's lowest and highest x at its value."""
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order].astype(float)
    unique, start, counts = np.unique(x, return_index=True, return_counts=True)
    sums = np.add.reduceat(y, start)
    values: list[float] = []
    weights: list[float] = []
    low: list[float] = []
    high: list[float] = []
    for xv, total, n in zip(unique.tolist(), sums.tolist(), counts.tolist(), strict=True):
        values.append(total / n)
        weights.append(float(n))
        low.append(xv)
        high.append(xv)
        while len(values) > 1 and values[-2] >= values[-1]:
            w = weights[-2] + weights[-1]
            values[-2] = (values[-2] * weights[-2] + values[-1] * weights[-1]) / w
            weights[-2] = w
            high[-2] = high[-1]
            del values[-1], weights[-1], low[-1], high[-1]
    xs: list[float] = []
    ys: list[float] = []
    for lo, hi, value in zip(low, high, values, strict=True):
        xs.append(lo)
        ys.append(value)
        if hi > lo:
            xs.append(hi)
            ys.append(value)
    return xs, ys


@dataclass(frozen=True)
class Calibrator:
    """One system's calibration map, fitted on validation.

    Attributes:
        method: One of :data:`METHODS`.
        labels: The conditions, in column order.
        fitted: For each column, whether the fitting data had a patient with that condition.
            A column without one keeps its raw probability (rule 1 above).
        params: The method's parameters: ``temperature``; ``a`` and ``b`` per column (None where
            not fitted); or isotonic breakpoints ``x`` and ``y`` per column.
        about: What it was fitted for and on: the system, the split, the level, the patients, the
            mask digests, the commit. The pipeline refuses a calibrator fitted for another system.
    """

    method: str
    labels: tuple[str, ...]
    fitted: tuple[bool, ...]
    params: dict[str, Any] = field(default_factory=dict)
    about: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown calibration method {self.method!r}; expected {METHODS}")
        if len(self.fitted) != len(self.labels) or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be distinct, with one fitted flag each")

    # -- fitting -------------------------------------------------------------- #

    @classmethod
    def fit(
        cls,
        method: str,
        probabilities: Any,
        truth: Sequence[int] | np.ndarray,
        labels: Sequence[str],
        *,
        about: Mapping[str, Any] | None = None,
    ) -> Calibrator:
        """Fit ``method`` to one system's scores on the fitting patients.

        Args:
            method: One of :data:`METHODS`.
            probabilities: One row per patient, one column per label; each row is divided by
                its sum first.
            truth: Each patient's true condition, as a column index into ``labels``.
            labels: The conditions, in column order.
            about: Recorded with the calibrator (see the class).

        Raises:
            ValueError: for an unknown method, a malformed matrix or a truth outside the labels.
        """
        labels = tuple(labels)
        p = _normalised(probabilities, len(labels))
        truth = np.asarray(truth, dtype=int)
        if truth.shape != (len(p),) or len(p) == 0:
            raise ValueError("one true condition per row, and at least one row")
        if truth.min() < 0 or truth.max() >= len(labels):
            raise ValueError("a true condition is outside the labels")
        fitted = tuple(bool(np.any(truth == j)) for j in range(len(labels)))
        columns = np.flatnonzero(fitted)
        params: dict[str, Any] = {}
        if method == "temperature":
            within = np.searchsorted(columns, truth)
            params["temperature"] = _fit_temperature(p[:, columns], within)
        elif method == "platt":
            a: list[float | None] = [None] * len(labels)
            b: list[float | None] = [None] * len(labels)
            for j in columns:
                a[j], b[j] = _fit_platt(_logit(p[:, j]), truth == j)
            params = {"a": a, "b": b}
        elif method == "isotonic":
            xs: list[list[float] | None] = [None] * len(labels)
            ys: list[list[float] | None] = [None] * len(labels)
            for j in columns:
                xs[j], ys[j] = _fit_isotonic(p[:, j], (truth == j).astype(float))
            params = {"x": xs, "y": ys}
        elif method != "none":
            raise ValueError(f"unknown calibration method {method!r}; expected {METHODS}")
        return cls(method, labels, fitted, params, dict(about or {}))

    # -- applying ------------------------------------------------------------- #

    def transform(self, probabilities: Any) -> np.ndarray:
        """Calibrated probabilities, one row per row of ``probabilities``, each summing to 1.

        A column without fitting patients keeps its raw (normalised) probability; the others
        share the rest. A row with nothing on the calibrated columns is returned unchanged.
        """
        p = _normalised(probabilities, len(self.labels))
        columns = np.flatnonzero(self.fitted)
        if self.method == "none" or columns.size == 0:
            return p
        raw = p[:, columns]
        share = raw.sum(axis=1)
        if self.method == "temperature":
            g = _tempered(np.log(np.clip(raw, TINY, 1.0)), self.params["temperature"])
        elif self.method == "platt":
            a = np.array([self.params["a"][j] for j in columns], dtype=float)
            b = np.array([self.params["b"][j] for j in columns], dtype=float)
            g = _sigmoid(a * _logit(raw) + b)
        else:
            g = np.column_stack(
                [
                    np.interp(raw[:, i], self.params["x"][j], self.params["y"][j])
                    for i, j in enumerate(columns)
                ]
            )
        total = g.sum(axis=1)
        empty = total <= 0
        g[empty] = raw[empty]
        total[empty] = share[empty]
        out = p.copy()
        live = share > 0
        out[np.ix_(live, columns)] = g[live] / total[live, None] * share[live, None]
        return out

    def calibrate(self, scores: Mapping[str, float]) -> dict[str, float]:
        """:meth:`transform` for one case: {condition: probability} in, the same keys out.

        A label the mapping leaves out counts as 0.

        Raises:
            ValueError: for a condition this calibrator was not fitted over.
        """
        unknown = set(scores) - set(self.labels)
        if unknown:
            raise ValueError(f"not fitted over {sorted(unknown)}")
        row = np.array([[float(scores.get(label, 0.0)) for label in self.labels]])
        calibrated = self.transform(row)[0]
        return {label: float(v) for label, v in zip(self.labels, calibrated, strict=True)}

    def mismatch(self, expected: Mapping[str, Any]) -> str | None:
        """Why this calibrator was not fitted for the system ``expected`` describes, or None.

        Args:
            expected: Any of ``kind`` (``fused``, ``ml``, ``graph``), ``backend`` and
                ``model_dir`` (the ML component), and ``alpha`` (the fusion's weight).
        """
        ml = self.about.get("ml") or {}
        found = {
            "kind": self.about.get("kind"),
            "backend": ml.get("backend"),
            "model_dir": ml.get("model_dir"),
            "alpha": self.about.get("alpha"),
        }
        for key, want in expected.items():
            have = found.get(key)
            if key == "alpha":
                same = (
                    have is not None
                    and want is not None
                    and math.isclose(float(have), float(want), abs_tol=1e-9)
                )
            elif key == "model_dir":
                same = (
                    have is not None
                    and want is not None
                    and (
                        PurePosixPath(str(have).replace("\\", "/"))
                        == PurePosixPath(str(want).replace("\\", "/"))
                    )
                )
            else:
                same = have == want
            if not same:
                return f"fitted for {key} {have!r}, not {want!r}"
        return None

    # -- saving --------------------------------------------------------------- #

    def to_json(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "method": self.method,
            "labels": list(self.labels),
            "fitted": list(self.fitted),
            "params": self.params,
            "about": self.about,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Calibrator:
        if data.get("format") != FORMAT or data.get("version") != VERSION:
            raise ValueError(f"not a {FORMAT} v{VERSION} file")
        calibrator = cls(
            str(data["method"]),
            tuple(data["labels"]),
            tuple(bool(f) for f in data["fitted"]),
            dict(data.get("params") or {}),
            dict(data.get("about") or {}),
        )
        calibrator._check_params()
        return calibrator

    def _check_params(self) -> None:
        width = len(self.labels)
        if self.method == "temperature":
            t = self.params.get("temperature")
            if not (isinstance(t, (int, float)) and math.isfinite(t) and t > 0):
                raise ValueError("temperature must be a positive number")
        elif self.method in ("platt", "isotonic"):
            keys = ("a", "b") if self.method == "platt" else ("x", "y")
            for key in keys:
                values = self.params.get(key)
                if not isinstance(values, list) or len(values) != width:
                    raise ValueError(f"{self.method}: {key} needs one entry per label")
                for j, flag in enumerate(self.fitted):
                    if flag != (values[j] is not None):
                        raise ValueError(f"{self.method}: {key}[{j}] does not match fitted")
            if self.method == "isotonic":
                for xs, ys in zip(self.params["x"], self.params["y"], strict=True):
                    if xs is not None and (
                        len(xs) != len(ys) or not xs or np.any(np.diff(xs) <= 0)
                    ):
                        raise ValueError("isotonic breakpoints must be strictly increasing")


class RefusedCalibrator:
    """Stands in for a configured calibrator that cannot be used: its file is missing or
    unreadable, or it was fitted for another system. The pipeline then shows no probabilities and
    reports ``calibration`` in ``degraded_components``, like a degraded ranker (docs/02 §7)."""

    degraded = True

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def calibrate(self, scores: Mapping[str, float]) -> dict[str, float]:  # noqa: ARG002
        raise RuntimeError(self.reason)
