"""EXP-012's runner (scripts/evaluate_ablations.py): its helpers, and the gate on the test split.

The scoring itself needs data/ and models/ and is checked by the dry run on validation; these
tests need neither. What they hold is the part that must never depend on a data file: the test
split stays closed until the freeze, and the opening is recorded and confirmed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval import ablation, freeze
from src.eval.verdicts import FAMILIES, PLAIN

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_ddxplus_chestpain as builder  # noqa: E402
import evaluate_ablations as ev  # noqa: E402


def not_frozen(*_args, **_kwargs):
    raise freeze.NotFrozen("no v0.9-freeze tag")


def frozen(*_args, **_kwargs):
    return {
        "commit": "c" * 40,
        "tag": "v0.9-freeze",
        "tag_commit": "b" * 40,
        "changed_since_freeze": [],
    }


# --------------------------------------------------------------------------- #
# Models and α
# --------------------------------------------------------------------------- #


def test_every_model_a_claim_names_has_a_bundle():
    needed = {*PLAIN}
    for family in FAMILIES.values():
        needed |= {family.component, *family.like}
    assert needed == set(ev.MODELS)


def test_bundles_follow_the_layout_the_earlier_experiments_used():
    assert ev.bundle("B1-XGB", 42) == ev.exp019.B1
    assert ev.bundle("B1-LR", 45) == ev.exp019.B1, "the logistic regression is fitted once"
    assert ev.bundle("B1-XGB", 44) == ev.SEEDS_DIR / "b1" / "seed44"
    assert ev.bundle("B1-LR+aug", 42) == ev.exp019.AUG
    assert ev.bundle("B1-LR+aug", 43) == ev.SEEDS_DIR / "b1_aug" / "seed43"
    assert ev.bundle("B1-XGB′+aug", 42) == ev.exp019.ASKED_AUG
    assert ev.bundle("B1-XGB′+aug", 46) == ev.SEEDS_DIR / "b1_asked_aug" / "seed46"
    assert ev.bundle("B1-DL+aug", 44) == ev.DEEP / "b1_dl_aug" / "seed44"
    assert ev.bundle("B1-DL′+aug", 42) == ev.DEEP / "b1_dl_asked_aug" / "seed42"


def test_alphas_come_from_exp006_and_a_missing_choice_says_so(tmp_path):
    path = tmp_path / "exp006.json"
    path.write_text(
        json.dumps(
            {
                "components": {
                    "B1-DL′+aug": {"chosen": {"alpha": 0.35}},
                    "B1-LR+aug": {"chosen": {"alpha": None}},
                }
            }
        ),
        encoding="utf-8",
    )
    out = ev.chosen_alphas(path, None)
    assert out["B1-DL′+aug"] == {"alpha": 0.35, "source": "EXP-006 (exp006.json)"}
    assert out["B1-LR+aug"]["alpha"] == 0.5 and "chose none" in out["B1-LR+aug"]["source"]
    assert out["B1-XGB"]["alpha"] == 0.5 and "not swept" in out["B1-XGB"]["source"]


def test_an_untuned_alpha_is_marked_untuned():
    out = ev.chosen_alphas(None, 0.4)
    assert {e["alpha"] for e in out.values()} == {0.4}
    assert all("UNTUNED" in e["source"] for e in out.values())


def test_family_arguments_name_a_family():
    assert ev.parse_pairs(["unprimed=0.12"], float) == {"unprimed": 0.12}
    with pytest.raises(SystemExit):
        ev.parse_pairs(["nonsense=0.12"], float)
    with pytest.raises(SystemExit):
        ev.parse_pairs(["0.12"], float)


def test_flags_are_scored_only_for_the_systems_that_keep_them():
    kept = ["A0", "A0′", "A1b", "A1b′", "A2", "A7-LR", "A7-XGB′ (A0's α)"]
    dropped = ["A1", "A1′", "A4", "A4′", "B2", "B1-LR", "B0"]
    assert all(ev.uses_flags(label) for label in kept)
    assert not any(ev.uses_flags(label) for label in dropped)
    # the same split the library makes: the specs that keep the flags
    assert {n for n, s in ablation.SPECS.items() if s.flags} == {"A0", "A1b", "A2", "A7"}


def test_numpy_values_serialise():
    payload = {"a": np.float64(0.5), "b": np.int64(3), "c": np.array([1, 2]), "d": {"x", "y"}}
    assert json.loads(json.dumps(payload, default=ev.jsonable)) == {
        "a": 0.5,
        "b": 3,
        "c": [1, 2],
        "d": ["x", "y"],
    }
    with pytest.raises(TypeError):
        json.dumps({"e": object()}, default=ev.jsonable)


# --------------------------------------------------------------------------- #
# The gate on the test split
# --------------------------------------------------------------------------- #


def test_the_test_split_stays_closed_without_the_freeze(monkeypatch):
    monkeypatch.setattr(ev.freeze, "require_freeze", not_frozen)
    with pytest.raises(SystemExit) as stopped:
        ev.main(["--split", "test", "--digest-only"])
    assert "STAYS CLOSED" in str(stopped.value) and "v0.9-freeze" in str(stopped.value)


@pytest.mark.parametrize(
    "missing",
    ["--confirm-single-opening", "--expect-digests", "--alpha-file", "--reduced-claims"],
)
def test_scoring_the_test_split_needs_every_confirmation(monkeypatch, capsys, missing):
    monkeypatch.setattr(ev.freeze, "require_freeze", frozen)
    given = {
        "--confirm-single-opening": [],
        "--expect-digests": ["aaaa,bbbb"],
        "--alpha-file": ["exp006.json"],
        "--reduced-claims": ["both"],
    }
    argv = ["--split", "test"]
    for flag, value in given.items():
        if flag != missing:
            argv += [flag, *value]
    with pytest.raises(SystemExit) as stopped:
        ev.main(argv)
    assert stopped.value.code == 2
    assert missing in capsys.readouterr().err


def test_the_test_split_is_never_scored_on_a_subset_or_with_an_untuned_alpha(monkeypatch, capsys):
    monkeypatch.setattr(ev.freeze, "require_freeze", frozen)
    for extra in (["--limit", "100"], ["--untuned-alpha", "0.5"]):
        with pytest.raises(SystemExit) as stopped:
            ev.main(["--split", "test", *extra])
        assert stopped.value.code == 2
    assert "scored whole" in capsys.readouterr().err


def test_a_dry_run_needs_an_alpha_source(monkeypatch, capsys):
    with pytest.raises(SystemExit) as stopped:
        ev.main(["--split", "validate"])
    assert stopped.value.code == 2
    assert "--alpha-file" in capsys.readouterr().err


def test_a_missing_sweep_is_reported_not_guessed(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        ev.main(["--alpha-file", str(tmp_path / "nope.json")])
    assert stopped.value.code == 2
    assert "EXP-006" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The parquet builder's freeze-gated flag
# --------------------------------------------------------------------------- #


def write_config(tmp_path: Path, allow: bool = False) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(f"evaluation:\n  allow_test_split: {str(allow).lower()}\n", encoding="utf-8")
    return path


def test_the_builder_flag_runs_the_freeze_gate(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(builder.freeze, "require_freeze", not_frozen)
    built = []
    monkeypatch.setattr(builder, "build", lambda *a, **k: built.append(k) or {})
    code = builder.main(
        ["--split", "test", "--open-test-split", "--config", str(write_config(tmp_path))]
    )
    assert code == 2 and not built
    assert "stays closed" in capsys.readouterr().err


def test_the_builder_flag_opens_only_the_test_split_and_only_after_the_freeze(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(builder.freeze, "require_freeze", frozen)
    seen = []

    def stop_after_recording(split, raw_dir, out_dir, *, allow_test_split):
        seen.append((split, allow_test_split))
        raise PermissionError("stop here")

    monkeypatch.setattr(builder, "build", stop_after_recording)
    config = str(write_config(tmp_path, allow=False))
    assert builder.main(["--split", "test", "--open-test-split", "--config", config]) == 2
    assert seen == [("test", True)]
    with pytest.raises(SystemExit):
        builder.main(["--split", "validate", "--open-test-split", "--config", config])


def test_the_test_split_is_still_refused_without_the_flag_or_the_switch(monkeypatch, tmp_path):
    monkeypatch.setattr(builder.freeze, "require_freeze", frozen)
    seen = []

    def stop_after_recording(split, raw_dir, out_dir, *, allow_test_split):
        seen.append(allow_test_split)
        raise PermissionError("closed")

    monkeypatch.setattr(builder, "build", stop_after_recording)
    assert builder.main(["--split", "test", "--config", str(write_config(tmp_path))]) == 2
    assert seen == [False]
