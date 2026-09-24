"""Patients with part of their history: reduced-evidence masks (decision D-10, risk R-18).

A mask keeps a share of the questions a patient answered, as though the interview had stopped
early. It serves two purposes:

* **Evaluation** at 50% and 25% of the history, as ``docs/proposals/05-amendment-3.md`` §3.7
  proposes (:func:`mask_frame`, :func:`evaluation_rng`). **That amendment is not approved.** Until
  the team picks a mask rule, no model is scored on masked validate patients (PROGRESS.md §1).
* **Training** on masked copies of *train* patients, ``+aug`` in the proposal's labels
  (:func:`augment`, :func:`augmentation_rng`), so that a model learns what an unasked question
  looks like. Without masked training rows the encoder's "asked" channel is constant and teaches
  nothing (src/ml/features.py).

**The rule for the questions a patient listed** is the proposal's question rule, unchanged: the
initial evidence's question is always kept; the other listed questions, in first-appearance order,
are permuted once, and a share keeps the first ``round(n * share)`` of them, so every level is a
prefix of the same permutation and a patient's 25% history lies inside their 50% history. A kept
follow-up brings its parent (``E_54``-``E_59`` follow ``E_53``, ``E_152`` follows ``E_151``) when the
patient has it. A kept question keeps every token; a dropped one loses every token.
``scripts/check_mask_rules.py`` reproduces the proposal's figures and digest from this code.

**The rule for the questions a patient did not list**, which the proposal leaves to the encoder,
is decided here (open decision A-9, docs/02 §9). In DDXPlus an unlisted question means "no", so at
full evidence every question was asked. Under a mask, the unlisted questions are asked at the same
share: a second permutation, drawn from the same generator after the first, so the listed tokens
and the proposal's digest do not depend on it, and the levels nest here too. An unlisted follow-up
counts as asked only when its parent was: nobody asks where the pain is without asking about pain.
The result feeds the encoder's ``asked`` argument; at share 1.0 every question is asked.

A masked frame loses its ``positive_codes`` column, which would otherwise describe the full
history; recompute it from the kept tokens if needed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ddxplus import TOKEN_SEP
from src.medical_kg.crosswalk import PARENT_QUESTION

__all__ = [
    "LEVELS",
    "SEED",
    "Mask",
    "augment",
    "augmentation_rng",
    "draw_masks",
    "evaluation_rng",
    "mask_frame",
]

SEED = 42
LEVELS = (0.5, 0.25)
"""The reduced levels the proposal names, beside full evidence."""


@dataclass(frozen=True)
class Mask:
    """One patient's history at one share.

    Attributes:
        tokens: The kept EVIDENCES tokens, in their original order.
        asked: Every question asked, listed or not: the encoder's ``asked`` argument.
    """

    tokens: tuple[str, ...]
    asked: frozenset[str]


def _generator(key: str) -> np.random.Generator:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def evaluation_rng(case_id: str, seed: int = SEED) -> np.random.Generator:
    """The generator for a validate or test patient's masks, exactly as the proposal pins it."""
    return _generator(f"{seed}:{case_id}")


def augmentation_rng(case_id: str, seed: int) -> np.random.Generator:
    """The generator for a train patient's masked copy: a stream of its own, never the
    evaluation masks' (proposal §3.7)."""
    return _generator(f"{seed}:augment:{case_id}")


def _question(token: str) -> str:
    return token.partition(TOKEN_SEP)[0]


def draw_masks(
    initial_evidence: str,
    evidences: Sequence[str],
    shares: Iterable[float],
    *,
    codes: Sequence[str],
    rng: np.random.Generator,
) -> dict[float, Mask]:
    """One patient's masks at several shares, drawn from one pair of permutations so they nest.

    Args:
        initial_evidence: The patient's ``initial_evidence``, always kept. It defines the mask and
            is never a model input.
        evidences: The patient's EVIDENCES tokens.
        shares: Shares of the history to keep, each in [0, 1].
        codes: Every question the encoder carries (``EvidenceEncoder.codes``), in its order: the
            pool the unlisted questions come from.
        rng: :func:`evaluation_rng` or :func:`augmentation_rng`, fresh for this patient.
    """
    questions = list(dict.fromkeys(_question(t) for t in evidences))
    listed = set(questions)
    others = [q for q in questions if q != initial_evidence]
    order = rng.permutation(len(others))
    unlisted = [code for code in codes if code not in listed]
    unlisted_order = rng.permutation(len(unlisted))
    pool = set(codes)

    masks: dict[float, Mask] = {}
    for share in shares:
        if not 0.0 <= share <= 1.0:
            raise ValueError(f"a share is between 0 and 1, got {share}")
        kept = {initial_evidence} | {others[i] for i in order[: round(len(others) * share)]}
        kept |= {
            PARENT_QUESTION[q]
            for q in list(kept)
            if q in PARENT_QUESTION and PARENT_QUESTION[q] in listed
        }
        asked_unlisted = {unlisted[i] for i in unlisted_order[: round(len(unlisted) * share)]}
        asked = kept | asked_unlisted
        asked -= {
            q for q in asked_unlisted if q in PARENT_QUESTION and PARENT_QUESTION[q] not in asked
        }
        masks[share] = Mask(
            tokens=tuple(t for t in evidences if _question(t) in kept),
            asked=frozenset(asked & pool),
        )
    return masks


def mask_frame(
    frame: pd.DataFrame, share: float, *, codes: Sequence[str], seed: int = SEED
) -> pd.DataFrame:
    """Validate or test patients at one level: ``evidences`` cut, ``asked`` added.

    **Not to be scored before amendment 3 is approved** (module docstring). Reads ``case_id``,
    ``initial_evidence`` and ``evidences``; every other column is carried over, except
    ``positive_codes``.
    """
    masked = [
        draw_masks(i, list(e), [share], codes=codes, rng=evaluation_rng(c, seed))[share]
        for c, i, e in zip(
            frame["case_id"], frame["initial_evidence"], frame["evidences"], strict=True
        )
    ]
    out = frame.drop(columns=["positive_codes"], errors="ignore").copy()
    out["evidences"] = [list(m.tokens) for m in masked]
    out["asked"] = [sorted(m.asked) for m in masked]
    return out


def augment(
    frame: pd.DataFrame,
    *,
    codes: Sequence[str],
    seed: int,
    low: float = 0.1,
    high: float = 0.9,
) -> pd.DataFrame:
    """The training frame followed by one masked copy of every patient (``+aug``).

    Each copy keeps a share drawn uniformly from ``[low, high]``, per patient, from
    :func:`augmentation_rng`: a spread of history lengths rather than the evaluation levels
    themselves. The original rows keep ``asked`` empty (None), meaning full evidence; the copies
    record their share in ``mask_share``. Columns as :func:`mask_frame`.
    """
    copies = []
    for case_id, initial, evidences in zip(
        frame["case_id"], frame["initial_evidence"], frame["evidences"], strict=True
    ):
        rng = augmentation_rng(case_id, seed)
        share = float(rng.uniform(low, high))
        mask = draw_masks(initial, list(evidences), [share], codes=codes, rng=rng)[share]
        copies.append((share, mask))
    original = frame.drop(columns=["positive_codes"], errors="ignore").copy()
    original["asked"] = None
    original["mask_share"] = 1.0
    masked = original.copy()
    masked["evidences"] = [list(m.tokens) for _, m in copies]
    masked["asked"] = [sorted(m.asked) for _, m in copies]
    masked["mask_share"] = [s for s, _ in copies]
    return pd.concat([original, masked], ignore_index=True)
