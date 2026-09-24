"""The trained model behind the pipeline's ranker interface (task 1c).

:class:`ModelRanker` implements :class:`~src.contracts.ConditionRanker` over anything with
``labels`` and ``predict_proba``: the B1 models today, the deep ranker later. It owns the two steps
between a case and a probability — the concept → token inversion (:mod:`src.ml.case_tokens`) and
the feature encoding (:mod:`src.ml.features`) — so that the pipeline keeps knowing nothing about
either.

:func:`open_ranker` is the counterpart of
:func:`~src.medical_kg.neo4j_store.open_graph_store`: it returns *something* that ranks, whatever is
missing. Without the release files, without a trained model, or with a model trained on different
features, it returns a :class:`DegradedRanker` whose flat scores the pipeline's ``_normalise``
collapses to zeros — so the ranking is then provably the knowledge graph's alone, and
``degraded_components`` says ``"ml"``. Both release files are gitignored, so **a fresh clone and CI
are always degraded**, which is the correct default and is exactly what the tests assert.

**The backend is logistic regression, not XGBoost** (EXP-017, R-18). The two are 0.0002 apart on
full-evidence validate, and 0.59 top-1 apart at a quarter of the history: XGBoost answers atrial
fibrillation for 76% of patients there, and with probability 1.000 when given no evidence at all,
because the encoder gives a default "no" answer no column and AF is the condition whose DDXPlus
patients answer fewest questions. Every hand-authored case is short — the golden cases carry 5 to 13
tokens — so that is the regime this ranker actually runs in. ``backend="xgboost"`` stays available
for experiments; it must not be the configured default until the encoder has an "asked" channel.

Reference: docs/02-architecture.md §7 (degradation), docs/03-data-management.md §2.2 (features),
docs/08-experiment-log.md EXP-017.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.contracts import PatientCase
from src.ml.case_tokens import CaseTokens, encode_case
from src.ml.features import EvidenceEncoder

logger = logging.getLogger(__name__)

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "DEFAULT_MODEL_DIR",
    "DegradedRanker",
    "ModelRanker",
    "describe",
    "ProbabilityModel",
    "open_ranker",
]

DEFAULT_BACKEND = "logreg"
"""EXP-017 / R-18. Not ``xgboost``: see the module docstring."""

BACKENDS: tuple[str, ...] = ("logreg", "xgboost", "none")
"""``none`` asks for the degraded ranker deliberately, for a knowledge-graph-only ablation."""

DEFAULT_MODEL_DIR = Path("models/nightingale_b0_b1")
DEFAULT_EVIDENCES = Path("data/raw/ddxplus/release_evidences.json")
DEFAULT_CONDITIONS = Path("data/interim/ddxplus_chestpain_conditions.json")
LOGREG_FILE = "b1_logreg.json"
XGBOOST_STEM = "b1_xgboost"


@runtime_checkable
class ProbabilityModel(Protocol):
    """What :class:`ModelRanker` needs of a model: its classes, and probabilities for a row."""

    labels: tuple[str, ...]

    def predict_proba(self, X: Any) -> np.ndarray: ...


class DegradedRanker:
    """Stands in when there is no usable model. Implements ``ConditionRanker``.

    Its scores are flat, which ``src.pipeline._normalise`` turns into zeros, so the fused ranking
    is the graph's alone. That is a stronger guarantee than returning ``{}``: the ablation is
    testable by comparing the two pipelines' rankings for equality.
    """

    backend = "none"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    @property
    def degraded(self) -> bool:
        return True

    def score(self, case: PatientCase) -> dict[str, float]:  # noqa: ARG002
        from src.conditions import CONDITIONS

        return {c.id: 1.0 for c in CONDITIONS if c.in_training_data}


class ModelRanker:
    """A trained model as a :class:`~src.contracts.ConditionRanker`.

    Args:
        model: Anything with ``labels`` and ``predict_proba`` — a B1 baseline, or the deep ranker.
        encoder: The :class:`~src.ml.features.EvidenceEncoder` the model was trained with. Its
            fingerprint is checked when the model is loaded, not here.
        backend: The name this was opened as, for reporting.
        include_narrower: Passed to :func:`~src.ml.case_tokens.tokens_for_case` (decision A-8).

    Attributes:
        last_tokens: What the most recent :meth:`score` actually fed the model. The audit trail
            R-17 asks for: it names every concept that produced a token, every one that did not,
            and why.
    """

    def __init__(
        self,
        model: ProbabilityModel,
        encoder: EvidenceEncoder,
        *,
        backend: str = DEFAULT_BACKEND,
        include_narrower: bool = True,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.backend = backend
        self.include_narrower = include_narrower
        self.last_tokens: CaseTokens | None = None

    @property
    def degraded(self) -> bool:
        """False: a real model is answering. The pipeline checks this duck-typed."""
        return False

    def score(self, case: PatientCase) -> dict[str, float]:
        """{condition id: probability} over the conditions the model was trained on.

        Aortic dissection is absent by construction — DDXPlus has no cases of it — so the graph
        and the red-flag rules are its only route into a ranking. That is the architecture, not a
        gap (docs/04 §3).

        Raises:
            UnrepresentableCase: for a case the feature space cannot hold. The pipeline catches
                it and continues with the graph alone.
        """
        row, tokens = encode_case(case, self.encoder, include_narrower=self.include_narrower)
        self.last_tokens = tokens
        if not tokens.tokens:
            logger.warning(
                "case %s: no finding survived the crosswalk, so the model sees age and sex only",
                case.case_id,
            )
        probabilities = np.asarray(self.model.predict_proba(row.reshape(1, -1)))[0]
        return {
            label: float(value)
            for label, value in zip(self.model.labels, probabilities, strict=True)
        }


def _load_encoder(evidences_path: Path, conditions_path: Path) -> EvidenceEncoder | str:
    missing = [str(p) for p in (evidences_path, conditions_path) if not p.exists()]
    if missing:
        return f"the release files are not on this machine: {', '.join(missing)}"
    try:
        return EvidenceEncoder.from_release(evidences_path, conditions_path)
    except (OSError, ValueError, KeyError) as exc:
        return f"the release files cannot be read: {exc}"


def _load_model(
    backend: str, model_dir: Path, fingerprint: str, stem: str | None
) -> ProbabilityModel | str:
    """The trained model, or a sentence saying why there is none."""
    from src.ml.baselines import LogisticBaseline, XGBoostBaseline

    try:
        if backend == "logreg":
            path = model_dir / (stem or LOGREG_FILE)
            if not path.exists():
                return f"no trained model at {path}"
            return LogisticBaseline.from_json(
                json.loads(path.read_text(encoding="utf-8")), feature_fingerprint=fingerprint
            )
        path = model_dir / f"{stem or XGBOOST_STEM}.meta.json"
        if not path.exists():
            return f"no trained model at {path}"
        return XGBoostBaseline.load(
            model_dir, stem or XGBOOST_STEM, feature_fingerprint=fingerprint
        )
    except ValueError as exc:  # a fingerprint mismatch says "re-encode or retrain"
        return str(exc)
    except (OSError, KeyError, ImportError) as exc:
        return f"the model at {model_dir} cannot be loaded: {exc}"


def open_ranker(
    backend: str = DEFAULT_BACKEND,
    *,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    evidences_path: Path | str = DEFAULT_EVIDENCES,
    conditions_path: Path | str = DEFAULT_CONDITIONS,
    stem: str | None = None,
    include_narrower: bool = True,
) -> ModelRanker | DegradedRanker:
    """The ranker the configuration asks for, degrading instead of failing (docs/02 §7).

    Args:
        backend: ``ml.backend`` in configs/config.yaml, one of :data:`BACKENDS`. ``logreg`` is the
            default and the only one fit for short input (R-18); ``xgboost`` is for experiments;
            ``none`` asks for the degraded ranker, which is how a knowledge-graph-only ablation is
            configured.
        model_dir: The trained bundle. Gitignored, so it is absent in CI.
        evidences_path: ``release_evidences.json``, which fixes the feature columns.
        conditions_path: ``ddxplus_chestpain_conditions.json``, which fixes which evidences count.
        stem: A non-default file name inside ``model_dir``.
        include_narrower: Decision A-8; see :mod:`src.ml.case_tokens`.

    Returns:
        A :class:`ModelRanker`, or a :class:`DegradedRanker` naming what was missing. **Never
        raises for missing data**: a prototype with no model still has to rank.

    Raises:
        ValueError: for an unknown backend, which is a configuration typo rather than a missing
            file, exactly as :func:`~src.medical_kg.neo4j_store.open_graph_store` treats one.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown ml backend {backend!r}; expected one of {BACKENDS}")
    if backend == "none":
        return DegradedRanker("the ml backend is switched off in the configuration")

    encoder = _load_encoder(Path(evidences_path), Path(conditions_path))
    if isinstance(encoder, str):
        logger.warning("ranking without a model: %s", encoder)
        return DegradedRanker(encoder)

    model = _load_model(backend, Path(model_dir), encoder.fingerprint, stem)
    if isinstance(model, str):
        logger.warning("ranking without a model: %s", model)
        return DegradedRanker(model)

    logger.info("ranker: %s from %s, %d features", backend, model_dir, len(encoder.feature_names))
    return ModelRanker(model, encoder, backend=backend, include_narrower=include_narrower)


def describe(ranker: Any) -> str:
    """One line naming what is ranking, for a report or a demo header."""
    if getattr(ranker, "degraded", False):
        return f"ranker: none ({getattr(ranker, 'reason', 'degraded')})"
    backend = getattr(ranker, "backend", type(ranker).__name__)
    labels: Sequence[str] = getattr(getattr(ranker, "model", None), "labels", ())
    return f"ranker: {backend} over {len(labels)} conditions"
