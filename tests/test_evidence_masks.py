"""Tests for reduced-evidence masks and the asked channel they feed (src/ml/evidence_masks.py).

Synthetic patients, except the digest check, which reproduces the proposal's figure on the real
validate split and skips when data/ is absent (as in CI). No model is scored on masked validate
patients here or anywhere until docs/05 amendment 3 is approved; the toy model at the end is a
test on synthetic rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ddxplus import EvidenceSpec, code_sort_key
from src.ml.baselines import LogisticBaseline, label_index
from src.ml.evidence_masks import (
    LEVELS,
    augment,
    augmentation_rng,
    draw_masks,
    evaluation_rng,
    mask_frame,
)
from src.ml.features import EvidenceEncoder, evidence_codes_in_scope
from tests.sklearn_guard import needs_sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CONDITIONS = REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json"
REAL_PARQUET = REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_validate.parquet"

CODES = ["E_53", "E_54", "E_55", "E_79", "E_91", "E_151", "E_152", "E_204", "E_220"]
PATIENT = (
    "E_91",  # initial evidence
    ["E_53", "E_54_@_V_71", "E_55_@_V_16", "E_55_@_V_14", "E_79", "E_91", "E_204_@_V_10"],
)


def _q(token: str) -> str:
    return token.partition("_@_")[0]


def proposal_rule(case_id: str, initial: str, tokens: list[str], share: float) -> list[str]:
    """The question rule exactly as docs/proposals/05-amendment-3.md §3.7 writes it, re-derived
    here independently of the module."""
    digest = hashlib.sha256(f"42:{case_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    questions = list(dict.fromkeys(_q(t) for t in tokens))
    others = [q for q in questions if q != initial]
    order = rng.permutation(len(others))
    kept = {initial} | {others[i] for i in order[: round(len(others) * share)]}
    parents = {f"E_{n}": "E_53" for n in range(54, 60)} | {"E_152": "E_151"}
    kept |= {parents[q] for q in list(kept) if q in parents and parents[q] in questions}
    return [t for t in tokens if _q(t) in kept]


def masks(case_id="C-1", shares=(1.0, 0.5, 0.25, 0.0), patient=PATIENT, codes=CODES):
    initial, tokens = patient
    return draw_masks(initial, tokens, shares, codes=codes, rng=evaluation_rng(case_id))


# --------------------------------------------------------------------------- #
# The listed questions: the proposal's rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case_id", [f"C-{i}" for i in range(40)])
@pytest.mark.parametrize("share", [0.5, 0.25])
def test_the_listed_tokens_follow_the_proposal_exactly(case_id, share):
    initial, tokens = PATIENT
    assert list(masks(case_id)[share].tokens) == proposal_rule(case_id, initial, tokens, share)


def test_full_evidence_keeps_everything_and_asks_everything():
    full = masks()[1.0]
    assert list(full.tokens) == PATIENT[1]
    assert full.asked == set(CODES)


def test_the_initial_evidence_is_always_kept():
    nothing = masks()[0.0]
    assert nothing.tokens == ("E_91",) and nothing.asked == {"E_91"}


@pytest.mark.parametrize("case_id", [f"C-{i}" for i in range(40)])
def test_the_levels_nest(case_id):
    m = masks(case_id)
    assert set(m[0.25].tokens) <= set(m[0.5].tokens) <= set(m[1.0].tokens)
    assert m[0.25].asked <= m[0.5].asked <= m[1.0].asked


@pytest.mark.parametrize("case_id", [f"C-{i}" for i in range(40)])
def test_a_question_is_kept_whole_and_brings_its_parent(case_id):
    for mask in masks(case_id).values():
        kept = {_q(t) for t in mask.tokens}
        assert [t for t in PATIENT[1] if _q(t) in kept] == list(mask.tokens), "all or nothing"
        if kept & {"E_54", "E_55"}:
            assert "E_53" in kept, "pain that has a character is pain"


# --------------------------------------------------------------------------- #
# The unlisted questions: open decision A-9
# --------------------------------------------------------------------------- #


def test_every_listed_question_kept_is_asked_and_no_dropped_one_is():
    for mask in masks().values():
        kept = {_q(t) for t in mask.tokens}
        listed = {_q(t) for t in PATIENT[1]}
        assert mask.asked & listed == kept


def test_unlisted_questions_are_asked_at_about_the_same_share():
    unlisted = [c for c in CODES if c not in {_q(t) for t in PATIENT[1]}]
    for share in (0.5, 0.25):
        counts = [len(masks(f"C-{i}")[share].asked & set(unlisted)) for i in range(200)]
        assert np.mean(counts) == pytest.approx(len(unlisted) * share, abs=0.6)


def test_an_unlisted_follow_up_is_asked_only_after_its_parent():
    """E_152 (where is the swelling?) follows E_151 (swelling?); neither is listed here."""
    for i in range(200):
        for mask in masks(f"C-{i}").values():
            if "E_152" in mask.asked:
                assert "E_151" in mask.asked


def test_the_unlisted_draw_does_not_move_the_listed_tokens():
    """A second permutation after the first: the proposal's tokens and digest do not depend on
    which unlisted questions exist."""
    for i in range(40):
        a = masks(f"C-{i}", codes=CODES)
        b = masks(f"C-{i}", codes=CODES + ["E_300", "E_301"])
        assert all(a[s].tokens == b[s].tokens for s in a)


def test_the_streams_are_deterministic_and_separate():
    first, again = evaluation_rng("C-1").random(), evaluation_rng("C-1").random()
    assert first == again
    assert augmentation_rng("C-1", 42).random() != first


def test_a_worked_example_is_pinned():
    """NumPy promises no stream across versions: if this moves, the recorded masks, not the code,
    are the reference (proposal §3.7)."""
    m = masks("case-000123")
    # Five questions besides E_91: round(2.5) = 2 kept at 50%, round(1.25) = 1 at 25%.
    assert m[0.5].tokens == ("E_53", "E_91", "E_204_@_V_10")
    assert m[0.25].tokens == ("E_91", "E_204_@_V_10")
    # Unlisted E_151, E_152, E_220: two drawn at 50%; at 25% the one drawn is E_152, whose
    # parent E_151 was not asked, so it is not asked either.
    assert m[0.5].asked == {"E_53", "E_91", "E_204", "E_151", "E_152"}
    assert m[0.25].asked == {"E_91", "E_204"}


def test_a_share_outside_0_1_is_an_error():
    with pytest.raises(ValueError, match="between 0 and 1"):
        masks(shares=(1.5,))


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #


def frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [f"P-{i}" for i in range(n)],
            "initial_evidence": ["E_91"] * n,
            "age": [50] * n,
            "sex": ["M", "F"] * (n // 2),
            "evidences": [PATIENT[1]] * n,
            "positive_codes": [["E_53", "E_91"]] * n,
            "label_condition_id": ["COND:gerd"] * n,
        }
    )


def test_a_masked_frame_cuts_evidences_adds_asked_and_drops_positive_codes():
    masked = mask_frame(frame(), 0.25, codes=CODES)
    assert "positive_codes" not in masked
    assert list(masked["label_condition_id"]) == ["COND:gerd"] * 6, "other columns carry over"
    for case_id, tokens, asked in zip(
        masked["case_id"], masked["evidences"], masked["asked"], strict=True
    ):
        expected = masks(case_id, shares=(0.25,))[0.25]
        assert tuple(tokens) == expected.tokens and set(asked) == expected.asked


def test_augment_appends_one_masked_copy_per_patient():
    original = frame()
    rows = augment(original, codes=CODES, seed=42)
    assert len(rows) == 2 * len(original)
    head, tail = rows.iloc[: len(original)], rows.iloc[len(original) :]
    assert head["asked"].isna().all() and (head["mask_share"] == 1.0).all()
    assert list(head["evidences"]) == list(original["evidences"])
    assert tail["mask_share"].between(0.1, 0.9).all()
    assert list(tail["case_id"]) == list(original["case_id"])
    assert rows.equals(augment(original, codes=CODES, seed=42)), "deterministic"
    assert not rows.equals(augment(original, codes=CODES, seed=7))


@pytest.mark.skipif(
    not (REAL_PARQUET.exists() and REAL_CONDITIONS.exists()), reason="data/ is not committed (CI)"
)
def test_the_proposals_digest_is_reproduced_on_validate():
    """docs/proposals/05-amendment-3.md §3.3 / §3.7: digest e99a7a8fbf792675 (numpy 1.26.4).
    Mask properties only: nothing is scored."""
    validate = pd.read_parquet(REAL_PARQUET, columns=["case_id", "initial_evidence", "evidences"])
    codes = sorted(evidence_codes_in_scope(REAL_CONDITIONS), key=code_sort_key)
    lines = []
    for level in LEVELS:
        masked = mask_frame(validate, level, codes=codes)
        lines += [
            f"{c}\t{level}\t{' '.join(t)}"
            for c, t in zip(masked["case_id"], masked["evidences"], strict=True)
        ]
    digest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
    assert digest[:16] == "e99a7a8fbf792675"


# --------------------------------------------------------------------------- #
# Why the channel needs the masks: R-18 in miniature
# --------------------------------------------------------------------------- #

TOY_SPECS = {c: EvidenceSpec(c, "B", None, False) for c in ("E_1", "E_2", "E_3")}


def toy_patients(n: int = 400) -> pd.DataFrame:
    """Condition A always reports E_1, E_2 and E_3; condition B reports E_1 only. Everyone
    presents with E_1."""
    rows = []
    for i in range(n):
        a = i % 2 == 0
        rows.append(
            {
                "case_id": f"T-{i}",
                "initial_evidence": "E_1",
                "age": 50,
                "sex": "M",
                "evidences": ["E_1", "E_2", "E_3"] if a else ["E_1"],
                "label": "A" if a else "B",
            }
        )
    return pd.DataFrame(rows)


def toy_model(encoder: EvidenceEncoder, rows: pd.DataFrame) -> LogisticBaseline:
    return LogisticBaseline.fit(
        encoder.transform(rows),
        label_index(rows["label"], ("A", "B")),
        feature_fingerprint=encoder.fingerprint,
        labels=("A", "B"),
    )


def p_b(model: LogisticBaseline, encoder: EvidenceEncoder, asked: list[str] | None) -> float:
    row = encoder.encode(50, "M", ["E_1"], asked=asked).reshape(1, -1)
    return float(model.predict_proba(row)[0][1])


@needs_sklearn
def test_the_channel_with_masked_copies_tells_unasked_from_denied():
    """Without the channel, "E_1, nothing else asked" and "E_1, E_2 and E_3 denied" are one row,
    and the model is sure it is B. With the channel and masked copies, only the denial says B;
    an unfinished interview stays open. That is R-18, and the AF answer, in two conditions."""
    patients = toy_patients()
    plain = EvidenceEncoder(TOY_SPECS, TOY_SPECS)
    asked = EvidenceEncoder(TOY_SPECS, TOY_SPECS, asked_channel=True)
    codes = list(asked.codes)

    before = toy_model(plain, patients)
    assert p_b(before, plain, ["E_1"]) == p_b(before, plain, None) > 0.9

    after = toy_model(asked, augment(patients, codes=codes, seed=42))
    denied, unfinished = p_b(after, asked, None), p_b(after, asked, ["E_1"])
    assert denied > 0.9
    assert unfinished < 0.75, f"an unfinished interview is not a denial: P(B) = {unfinished:.2f}"


@pytest.mark.skipif(
    not (REAL_PARQUET.exists() and REAL_CONDITIONS.exists()), reason="data/ is not committed (CI)"
)
def test_the_exp019_script_reproduces_both_recorded_digests():
    """EXP-019 recorded the mask digest and the asked-set digest before anything was scored;
    the script refuses to score unless they reproduce."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_reduced_evidence.py"),
            "--digest-only",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "e99a7a8fbf792675" in result.stdout and "af3a357f6454838a" in result.stdout
