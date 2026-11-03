"""The rehearsal's check against EXP-019, EXP-021 and EXP-006 (scripts/check_rehearsal.py)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_rehearsal as check  # noqa: E402

LEVELS = check.LEVELS
DIGESTS = {"tokens": "e99a7a8fbf792675", "asked": "af3a357f6454838a"}


def report(base: float = 0.9) -> dict:
    return {
        "metrics": [
            {
                "name": name,
                "value": base + 0.001 * i,
                "ci": [base - 0.01, base + 0.01],
                "cases": 1000,
            }
            for i, name in enumerate(sorted(check.REQUIRED))
        ]
    }


def levels(base: float = 0.9) -> dict:
    return {level: report(base - 0.1 * i) for i, level in enumerate(LEVELS)}


def records():
    """A rehearsal and the three records it must match, all agreeing."""
    exp019 = {"digests": DIGESTS, "systems": {"B1-LR": levels(0.9), "B2": levels(0.8)}}
    exp021 = {
        "digests": DIGESTS,
        "systems": {"B1-DL+aug": levels(0.95), "B1-DL′+aug": levels(0.96)},
    }
    components = {}
    systems = {**exp019["systems"], **exp021["systems"]}
    alphas = {}
    for label, (plain, flagged) in check.FUSED_AS.items():
        fused, with_flags = levels(0.97), levels(0.98)
        components[label] = {
            "chosen": {"alpha": 0.3},
            "at_chosen": {
                level: {"fused": fused[level], "fused + red flags": with_flags[level]}
                for level in LEVELS
            },
        }
        if plain:
            systems[plain] = fused
        systems[flagged] = with_flags
        alphas[label] = {"alpha": 0.3}
    rehearsal = {
        "valid_as_a_result": True,
        "digests": DIGESTS,
        "systems": copy.deepcopy(systems),
        "alphas": alphas,
    }
    return rehearsal, exp019, exp021, {"components": components}


def run(rehearsal, exp019, exp021, exp006):
    return check.check(rehearsal, exp019, exp021, exp006)


def test_a_rehearsal_that_matches_every_record_is_clean():
    rows, problems = run(*records())
    assert problems == []
    assert rows and {r["status"] for r in rows} == {"exact"}


def test_every_shared_figure_is_compared_for_each_level_and_part():
    rows, _ = run(*records())
    # 8 required metrics x 4 parts (value, ci low, ci high, cases) x 3 levels per comparison
    assert len(rows) % (8 * 4 * 3) == 0
    assert any("EXP-021 B1-DL′+aug @ 0.25, MRR, value" in r["what"] for r in rows)
    assert any("EXP-006 B1-LR+aug fused + flags @ 1.0 vs A7-LR" in r["what"] for r in rows)
    assert any("EXP-006 B1-DL+aug fused @ 0.5 vs A4" in r["what"] for r in rows)


def test_one_patient_of_difference_is_caught():
    rehearsal, *rest = records()
    top3 = next(
        m for m in rehearsal["systems"]["B1-LR"]["0.25"]["metrics"] if m["name"] == "top-3 accuracy"
    )
    top3["value"] -= 1 / 33963
    rows, _ = run(rehearsal, *rest)
    differing = [r for r in rows if r["status"] == "DIFFERS"]
    assert [r["what"] for r in differing] == ["EXP-019 B1-LR @ 0.25, top-3 accuracy, value"]


def test_a_rounding_level_difference_is_within_tolerance_not_a_difference():
    rehearsal, *rest = records()
    mrr = next(m for m in rehearsal["systems"]["B2"]["1.0"]["metrics"] if m["name"] == "MRR")
    mrr["value"] += 1e-13
    rows, problems = run(rehearsal, *rest)
    assert problems == [] and "close" in {r["status"] for r in rows}
    assert "DIFFERS" not in {r["status"] for r in rows}


def test_a_different_number_of_cases_is_always_a_difference():
    rehearsal, *rest = records()
    rehearsal["systems"]["B1-LR"]["1.0"]["metrics"][0]["cases"] = 999
    rows, _ = run(rehearsal, *rest)
    assert any(r["status"] == "DIFFERS" and r["what"].endswith("cases") for r in rows)


def test_a_system_missing_from_the_rehearsal_is_a_problem():
    rehearsal, *rest = records()
    del rehearsal["systems"]["B2"]
    _, problems = run(rehearsal, *rest)
    assert any("B2" in p for p in problems)


def test_other_digests_or_an_invalid_run_are_problems():
    rehearsal, *rest = records()
    rehearsal["digests"] = {"tokens": "0000000000000000", "asked": "1111111111111111"}
    rehearsal["valid_as_a_result"] = False
    rehearsal["not_a_result_because"] = ["only the first 300 patients"]
    _, problems = run(rehearsal, *rest)
    assert any("not valid as a result" in p for p in problems)
    assert any("digests" in p for p in problems)


def test_the_alpha_the_rehearsal_used_must_be_the_one_the_sweep_chose():
    rehearsal, exp019, exp021, exp006 = records()
    rehearsal["alphas"]["B1-DL′+aug"]["alpha"] = 0.5
    _, problems = run(rehearsal, exp019, exp021, exp006)
    assert any(
        "B1-DL′+aug: the rehearsal fused at α = 0.5, EXP-006 chose 0.3" in p for p in problems
    )


def test_a_sweep_that_chose_no_alpha_is_reported():
    rehearsal, exp019, exp021, exp006 = records()
    exp006 = copy.deepcopy(exp006)
    exp006["components"]["B1-LR+aug"]["chosen"]["alpha"] = None
    _, problems = run(rehearsal, exp019, exp021, exp006)
    assert any("chose no α for B1-LR+aug" in p for p in problems)


def test_records_with_no_metrics_in_common_cannot_pass():
    rehearsal, exp019, exp021, exp006 = records()
    for level in LEVELS:
        for system in (exp019["systems"]["B1-LR"],):
            system[level]["metrics"] = [
                {"name": "Brier score", "value": 0.1, "ci": [0, 1], "cases": 1000}
            ]
    _, problems = run(rehearsal, exp019, exp021, exp006)
    assert any("not compared" in p for p in problems)


def test_the_command_line_exit_codes(tmp_path, capsys):
    rehearsal, exp019, exp021, exp006 = records()
    paths = {}
    for name, data in (("r", rehearsal), ("a", exp019), ("b", exp021), ("c", exp006)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    argv = [
        str(paths["r"]),
        "--exp019",
        str(paths["a"]),
        "--exp021",
        str(paths["b"]),
        "--exp006",
        str(paths["c"]),
    ]
    assert check.main(argv) == 0
    assert "0 DIFFER; 0 other problem" in capsys.readouterr().out
    rehearsal["systems"]["B1-LR"]["1.0"]["metrics"][0]["value"] += 0.5
    paths["r"].write_text(json.dumps(rehearsal, ensure_ascii=False), encoding="utf-8")
    assert check.main(argv) == 1
    assert "DIFFERS" in capsys.readouterr().out
    with pytest.raises(SystemExit) as stopped:
        check.main([str(paths["r"]), "--exp019", str(tmp_path / "missing.json")])
    assert stopped.value.code == 2
