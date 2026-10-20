"""The scorer of the LLM trial (scripts/score_llm_baselines.py; EXP-009, EXP-010).

It applies the analysis the experiment log registered to a ``results.jsonl``. The records here are
invented, in the shape the runner writes, so that every rule can be checked against numbers worked
out by hand: the pooled rate, the four H3 comparisons that must all pass, the checkpoint's four
ways to fail, and a rating sheet that does not say which arm wrote what.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import score_llm_baselines as score  # noqa: E402

from src.llm.baselines import A0_LLM, B3, B3R, B4  # noqa: E402

GERD, MI, PE = "COND:gerd", "COND:nstemi_stemi", "COND:pulmonary_embolism"


def claims(n: int, unsupported: int, **extra) -> dict:
    return {
        "sentences": n,
        "claims": n,
        "unsupported": unsupported,
        "untraced": ["x"] * unsupported,
        "cited": [],
        "invalid_citations": [],
        "cited_sentences": 0,
        "cited_unsupported": 0,
        "treatment_language": False,
        **extra,
    }


def record(
    arm, i, *, level=1.0, source="validation", ranking=(GERD,), stats=None, error=None, **more
):
    truth = {"true_condition": GERD, "differential": [GERD, MI], "differential_size": 3}
    base = {
        "arm": arm,
        "case_id": f"p{i:02d}",
        "level": level,
        "source": source,
        "truth": truth,
        "case_text": ["Patient: 40 years old, female.", "Findings present:", "- Burning pain"],
        "parsed": {
            "ranking": list(ranking),
            "explanation": "x",
            "unmatched": [],
            "duplicates": 0,
            "flags": [],
        },
        "claims": stats,
        "reply": {"error": error, "elapsed_s": 5.0, "done_reason": "stop", "context_full": False},
    }
    if arm == A0_LLM:
        base |= {
            "ranking": list(ranking),
            "delivered": {"text": "Delivered text.", "grounded": True},
            "delivered_claims": claims(2, 0),
            "fallback_reason": None,
            "degraded": [],
            "passages": {GERD: ["PMCID:PMC1"]},
        }
        del base["parsed"]
    return base | more


def cohort(n=40, *, alone=3, grounded=1, arms=(B3, B3R, B4, A0_LLM)):
    """n patients; each arm's explanation has 4 claims, of which that many are unsupported."""
    rate = {B3: alone, B3R: alone, B4: grounded, A0_LLM: grounded}
    return [record(arm, i, stats=claims(4, rate[arm])) for arm in arms for i in range(n)]


# --------------------------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------------------------- #


def test_ranking_metrics_count_a_failed_call_and_an_unparsed_answer_as_misses():
    rows = [
        record(B3, 0, ranking=(GERD,)),  # right
        record(B3, 1, ranking=(PE, GERD)),  # right, second
        record(B3, 2, ranking=()),  # did not parse
        record(B3, 3, ranking=(), error="LLMError: busy"),  # the call failed
    ]
    block = score.ranking_block(rows)
    metrics = {m["name"]: m["value"] for m in block["metrics"]}
    assert metrics["top-1 accuracy"] == pytest.approx(0.25)
    assert metrics["top-3 accuracy"] == pytest.approx(0.5)
    assert metrics["MRR"] == pytest.approx((1 + 0.5 + 0 + 0) / 4)
    assert (block["cases"], block["call_failed"], block["no_ranking"]) == (4, 1, 2)
    assert all(len(m["ci"]) == 2 for m in block["metrics"])  # an interval for every figure


# --------------------------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------------------------- #


def test_the_pooled_rate_counts_claims_across_patients_and_notes_failures():
    rows = [
        record(B3, 0, stats=claims(4, 2)),
        record(B3, 1, stats=claims(2, 0)),
        record(B3, 2, stats=None, error="LLMError: busy"),
    ]
    block = score.explanation_block(rows)
    assert (block["cases"], block["with_explanation"], block["call_failed"]) == (3, 2, 1)
    assert (block["claims"], block["unsupported"]) == (6, 2)
    assert block["unsupported_rate"]["value"] == pytest.approx(2 / 6)
    assert block["cases_with_unsupported"] == 1
    assert block["mean_call_s"] == 5.0 and block["slower_than_30s"] == 0


def test_an_arm_with_no_claim_has_no_rate():
    assert score.explanation_block([record(B3, 0, stats=None)])["unsupported_rate"] is None


def test_citations_and_treatment_language_are_totalled():
    rows = [
        record(
            B4,
            0,
            stats=claims(
                3,
                1,
                cited=["a", "b"],
                invalid_citations=["b"],
                cited_sentences=2,
                cited_unsupported=1,
                treatment_language=True,
            ),
        ),
        record(B4, 1, stats=claims(2, 0)),
    ]
    block = score.explanation_block(rows)
    assert (block["citations"], block["invalid_citations"]) == (2, 1)
    assert (block["cited_sentences"], block["cited_unsupported"]) == (2, 1)
    assert block["treatment_language"] == 1


# --------------------------------------------------------------------------------------------- #
# H3
# --------------------------------------------------------------------------------------------- #


def per_arm(records):
    out = {}
    for r in records:
        out.setdefault(r["arm"], []).append(r)
    return out


def test_h3_is_supported_only_when_all_four_comparisons_pass():
    result = score.h3(per_arm(cohort(alone=3, grounded=1)))
    assert result["supported"] and len(result["comparisons"]) == 4
    assert all(c["passed"] and c["difference"] == pytest.approx(0.5) for c in result["comparisons"])
    assert [(c["alone"], c["grounded"]) for c in result["comparisons"]] == list(score.H3_PAIRS)


def test_h3_fails_when_one_grounded_arm_does_no_better_than_an_llm_alone():
    rows = [r for r in cohort(alone=3, grounded=1) if r["arm"] != B4]
    rows += [record(B4, i, stats=claims(4, 3)) for i in range(40)]  # B4 as unsupported as B3
    result = score.h3(per_arm(rows))
    assert not result["supported"]
    assert {c["grounded"]: c["passed"] for c in result["comparisons"]} == {
        B4: False,
        A0_LLM: True,
    } or all("passed" in c for c in result["comparisons"])
    assert [c["passed"] for c in result["comparisons"] if c["grounded"] == B4] == [False, False]


def test_h3_fails_in_the_wrong_direction():
    result = score.h3(per_arm(cohort(alone=1, grounded=3)))
    assert not result["supported"]
    assert all(c["excludes_zero"] and not c["passed"] for c in result["comparisons"])


def test_h3_says_so_when_an_arm_did_not_run():
    result = score.h3(per_arm(cohort(arms=(B3, B3R))))
    assert not result["supported"]
    assert all(c["status"] == "not run" for c in result["comparisons"])


def test_h3_pairs_only_the_patients_both_arms_have():
    rows = cohort(arms=(B3, B3R, B4)) + [record(A0_LLM, i, stats=claims(4, 1)) for i in range(20)]
    comparisons = score.h3(per_arm(rows))["comparisons"]
    assert {c["cases"] for c in comparisons if c["grounded"] == A0_LLM} == {20}
    assert {c["cases"] for c in comparisons if c["grounded"] == B4} == {40}


# --------------------------------------------------------------------------------------------- #
# The checkpoint
# --------------------------------------------------------------------------------------------- #


def summary_of(rows):
    return score.summarise(rows)


def test_the_checkpoint_passes_when_the_raw_rate_is_under_twenty_percent():
    rows = [record(A0_LLM, i, stats=claims(10, 1 if i < 10 else 0)) for i in range(40)]
    verdict = summary_of(rows)["checkpoint"]
    assert verdict == {"passed": True, "reasons": []}


def test_the_checkpoint_fails_at_exactly_twenty_percent():
    rows = [record(A0_LLM, i, stats=claims(5, 1)) for i in range(40)]
    verdict = summary_of(rows)["checkpoint"]
    assert not verdict["passed"] and "not under 0.2" in verdict["reasons"][0]


def test_the_checkpoint_fails_when_more_than_a_tenth_of_the_calls_failed():
    ok = [record(A0_LLM, i, stats=claims(10, 0)) for i in range(36)]
    failed = [record(A0_LLM, 40 + i, stats=None, error="LLMError: busy") for i in range(5)]
    verdict = summary_of(ok + failed)["checkpoint"]  # 5 of 41 is more than a tenth
    assert not verdict["passed"] and any("5 of 41 calls failed" in r for r in verdict["reasons"])
    # exactly a tenth (4 of 40) is not more than a tenth
    assert summary_of(ok + failed[:4])["checkpoint"]["passed"]


def test_the_checkpoint_fails_if_a_delivered_explanation_is_not_clean():
    rows = [record(A0_LLM, i, stats=claims(10, 0)) for i in range(10)]
    rows[0]["delivered_claims"] = claims(2, 1)
    verdict = summary_of(rows)["checkpoint"]
    assert not verdict["passed"] and "delivered" in verdict["reasons"][0]


def test_the_checkpoint_fails_when_a0_llm_did_not_run():
    verdict = summary_of(cohort(arms=(B3,)))["checkpoint"]
    assert not verdict["passed"] and "did not run" in verdict["reasons"][0]


# --------------------------------------------------------------------------------------------- #
# The summary, the golden cases and the status
# --------------------------------------------------------------------------------------------- #


def test_golden_cases_are_listed_apart_and_never_pooled():
    rows = cohort(n=5, arms=(B3,)) + [
        record(B3, 99, source="golden", ranking=(MI, GERD), stats=claims(3, 3))
    ]
    summary = summary_of(rows)
    assert summary["cases"] == 5
    assert summary["explanations"][B3]["claims"] == 20  # the golden case's 3 are not in it
    assert summary["golden"][0]["top3"] == ["Possible NSTEMI / STEMI", "GERD"]
    assert summary["golden"][0]["unsupported"] == 3


def test_a_reduced_level_is_reported_for_the_arms_that_ran_at_it():
    rows = cohort(n=6, arms=(B3, B3R))
    rows += [record(B3, i, level=0.5, stats=claims(4, 1)) for i in range(6)]
    ranking = summary_of(rows)["ranking"]
    assert set(ranking[B3]) == {"1.0", "0.5"} and set(ranking[B3R]) == {"1.0"}


def test_the_status_lines_carry_the_headline_and_the_verdicts():
    rows = cohort(alone=3, grounded=0)
    lines = score.status_lines(summary_of(rows))
    text = "\n".join(lines)
    assert "40 validation patients" in lines[0]
    assert "B3 at 100%: unsupported-claim rate 0.750" in text
    assert "H3 SUPPORTED" in text and "CHECKPOINT (2026-11-03): PASS" in text
    failing = score.status_lines(summary_of(cohort(alone=1, grounded=3)))
    assert any("H3 NOT supported" in line for line in failing)
    assert any(line.startswith("CHECKPOINT") and "FAIL" in line for line in failing)


# --------------------------------------------------------------------------------------------- #
# The rating sheet
# --------------------------------------------------------------------------------------------- #


def test_the_rating_sheet_hides_the_arm_and_keeps_a_key(tmp_path):
    rows = cohort(n=6, arms=(B3, B3R, B4, A0_LLM))
    rows[0]["parsed"]["explanation"] = ""  # one B3 answer has no explanation: not rated
    rows[-1]["fallback_reason"] = "the LLM's answer is not grounded (1 claim(s))"
    counts = score.rating_sheet(rows, tmp_path)
    assert counts == {"items": 17, "unrated_B3": 1, "unrated_B4": 0, "unrated_A0+LLM": 0}
    sheet = (tmp_path / "rating_items.csv").read_text(encoding="utf-8")
    assert "B3" not in sheet and "B4" not in sheet and "A0" not in sheet  # no arm is named
    with open(tmp_path / "rating_items.csv", encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert [r["item"] for r in table] == [f"R-{n:03d}" for n in range(1, 18)]
    assert all(r["reasoning_validity_1_5"] == "" for r in table)
    assert "Patient: 40 years old, female." in table[0]["patient"]
    key = json.loads((tmp_path / "rating_key.json").read_text(encoding="utf-8"))
    assert sorted(v["arm"] for v in key.values()).count(B3) == 5
    assert B3R not in {v["arm"] for v in key.values()}  # B3r is not rated
    assert sum(v["template_stood_in"] for v in key.values()) == 1
    # a fixed shuffle: the same again, and the arms are mixed
    again = tmp_path / "again"
    score.rating_sheet(rows, again)
    assert (again / "rating_items.csv").read_bytes() == (tmp_path / "rating_items.csv").read_bytes()
    assert len({v["arm"] for v in list(key.values())[:6]}) > 1


def test_a0_llm_is_rated_on_the_text_it_delivers_and_the_others_on_their_explanation(tmp_path):
    rows = [record(A0_LLM, 0, stats=claims(2, 0)), record(B4, 0, stats=claims(2, 0))]
    rows[1]["parsed"]["explanation"] = "B4 text."
    score.rating_sheet(rows, tmp_path)
    with open(tmp_path / "rating_items.csv", encoding="utf-8", newline="") as handle:
        texts = {r["explanation"] for r in csv.DictReader(handle)}
    assert texts == {"Delivered text.", "B4 text."}


# --------------------------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------------------------- #


def test_the_command_line_writes_the_summary_and_the_sheet(tmp_path, capsys):
    (tmp_path / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in cohort(n=8)) + "\n", encoding="utf-8"
    )
    assert score.main([str(tmp_path), "--sheet", str(tmp_path / "sheet")]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["h3"]["supported"] and summary["rating_sheet"]["items"] == 24
    out = capsys.readouterr().out
    assert "CHECKPOINT (2026-11-03)" in out and "summary.json" in out
    assert (tmp_path / "sheet" / "rating_key.json").exists()
