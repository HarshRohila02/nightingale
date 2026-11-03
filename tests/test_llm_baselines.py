"""The LLM baselines' prompts, sample and parser (EXP-009, EXP-010; src/llm/baselines.py).

No model is called and no test opens a network connection. What must hold: the prompts the code
sends are the ones the experiment log pre-registered, character for character; the sample is the
one the log records; the parser reads the answers it is meant to and counts every way it fails.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.conditions import BY_ID, CONDITIONS
from src.contracts import Assertion, Evidence, Finding, PatientCase
from src.llm.baselines import (
    A0_LLM,
    ALLOWED_LABELS,
    ARMS,
    B3,
    B3R,
    B4,
    FORMAT_INSTRUCTION,
    SYSTEM_PROMPTS,
    build_prompt,
    case_text,
    demographics,
    given_result,
    parse_answer,
    reference_case,
    sample_digest,
    sample_ids,
)
from src.llm.explainer import SYSTEM as EXPLAINER_SYSTEM

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = (REPO_ROOT / "docs" / "08-experiment-log.md").read_text(encoding="utf-8")
GOLDEN = yaml.safe_load((REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml").read_text("utf-8"))
VALIDATE = REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_validate.parquet"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def golden(case_id: str) -> PatientCase:
    return PatientCase(**next(c["case"] for c in GOLDEN["cases"] if c["id"] == case_id))


def passage(n: int = 1, text: str = "An excerpt about chest pain.") -> Evidence:
    return Evidence(source="PMC", title=f"Article {n}", passage=text, citation=f"PMCID:PMC{n:07d}")


# --------------------------------------------------------------------------------------------- #
# The log and the code say the same thing
# --------------------------------------------------------------------------------------------- #


def test_every_prompt_in_the_code_is_in_the_experiment_log_verbatim():
    """EXP-009 and EXP-010 pre-register the prompts; a prompt edited after that must fail here."""
    for arm, system in SYSTEM_PROMPTS.items():
        assert f"\n{system}\n" in LOG, f"{arm}'s system prompt differs from docs/08"
    assert f"\n{EXPLAINER_SYSTEM}\n" in LOG, "the grounded explainer's system prompt differs"
    assert f"\n{build_prompt(B3, golden('GC-001'))}\n" in LOG, "the user prompt differs"
    assert FORMAT_INSTRUCTION in LOG


def test_the_sample_digest_in_the_log_is_the_sample_drawn_from_the_validation_labels():
    if not VALIDATE.exists():
        pytest.skip("no data/interim parquet (gitignored)")
    ids = sample_ids(pd.read_parquet(VALIDATE, columns=["case_id"])["case_id"])
    assert len(ids) == 50
    assert sample_digest(ids) == "022bdcfa4641cb81"
    assert "`022bdcfa4641cb81`" in LOG


def test_the_arms_are_the_four_the_log_names():
    assert ARMS == (B3, B3R, B4, A0_LLM) == ("B3", "B3r", "B4", "A0+LLM")
    assert set(SYSTEM_PROMPTS) == {B3, B3R, B4}


# --------------------------------------------------------------------------------------------- #
# The sample
# --------------------------------------------------------------------------------------------- #


def test_the_sample_is_a_hash_order_and_does_not_depend_on_the_input_order():
    ids = [f"p{i:04d}" for i in range(500)]
    drawn = sample_ids(ids, 50)
    assert len(drawn) == len(set(drawn)) == 50
    assert drawn == sample_ids(reversed(ids), 50) == sample_ids(ids + ids[:10], 50)
    assert drawn != sorted(drawn)  # not the first fifty by id
    assert sample_ids(ids, 50, seed=43) != drawn
    assert sample_ids(ids, 10) == drawn[:10]  # a prefix: asking for fewer gives the first of them
    assert sample_digest(drawn) == sample_digest(reversed(drawn))
    assert len(sample_digest(drawn)) == 16


# --------------------------------------------------------------------------------------------- #
# What the model is shown
# --------------------------------------------------------------------------------------------- #


def test_allowed_labels_are_the_fourteen_conditions_in_alphabetical_order():
    assert len(ALLOWED_LABELS) == 14
    assert set(ALLOWED_LABELS) == {c.label for c in CONDITIONS}
    assert list(ALLOWED_LABELS) == sorted(ALLOWED_LABELS, key=str.casefold)


def test_the_case_text_holds_labels_only_and_leaves_out_vitals_and_qualifiers():
    text = "\n".join(case_text(golden("GC-001")))
    assert "Patient: 58 years old, male." in text
    assert "Pressure-type pain" in text and "Diabetes mellitus" in text
    for left_out in ("hr", "sbp", "98", "radiate", "to jaw and left arm"):
        assert left_out not in text


def test_a_denied_finding_is_listed_apart_and_an_unknown_one_is_left_out():
    case = golden("GC-004")
    case.findings.append(
        Finding(concept_id="SYM:fever", label="Fever", assertion=Assertion.UNKNOWN)
    )
    lines = case_text(case)
    assert "Findings denied:" in lines
    denied = lines[lines.index("Findings denied:") + 1 :]
    assert "- Exertional onset" in denied and "- Diaphoresis" in denied
    assert "Fever" not in "\n".join(lines)


def test_interview_questions_answered_yes_are_listed_apart_from_findings():
    case = PatientCase(
        case_id="ddxplus-validate-0000001",
        age=10,
        sex="F",
        findings=[
            Finding(concept_id="DDX:E_53", label="Do you have pain?", source="ddxplus"),
            Finding(concept_id="SYM:chest_pain", label="Chest pain", source="crosswalk"),
        ],
        risk_factors=[
            Finding(concept_id="DDX:E_9", label="Do you smoke?", source="ddxplus"),
            Finding(concept_id="RF:smoking", label="Smoking", source="crosswalk"),
        ],
    )
    lines = case_text(case)
    asked = lines.index("Interview questions answered yes:")
    assert lines[asked + 1 : asked + 3] == ["- Do you have pain?", "- Do you smoke?"]
    assert lines[lines.index("Findings present:") + 1] == "- Chest pain"
    assert lines[lines.index("Risk factors:") + 1] == "- Smoking"


def test_demographics_name_the_sex_in_words_and_the_reference_case_credits_them():
    case = golden("GC-002")
    assert demographics(case) == "42 years old, female"
    ref = reference_case(case)
    assert ref.findings[-1].label == "42 years old female"
    assert len(case.findings) == len(ref.findings) - 1  # the original is untouched
    assert reference_case(case).case_id == case.case_id


def test_b3_and_b3r_are_given_no_passages_and_b4_is_given_one_block():
    case = golden("GC-001")
    evidence = {"COND:pericarditis": [passage(1)], "COND:gerd": [passage(2)], "COND:psvt": []}
    with pytest.raises(ValueError):
        build_prompt(B3, case, evidence)
    with pytest.raises(ValueError):
        build_prompt("B9", case)
    plain = build_prompt(B3R, case)
    assert "Passages:" not in plain and "<passage" not in plain
    prompt = build_prompt(B4, case, evidence)
    assert prompt.count("<passage") == 2 and prompt.count("Passages:") == 1
    assert "<passage cite=[PMCID:PMC0000001] about='Pericarditis'>" in prompt
    assert prompt.index("Passages:") < prompt.index("Conditions to choose from:")
    assert prompt.endswith(FORMAT_INSTRUCTION)


def test_the_passages_come_in_an_order_that_depends_on_the_case_and_not_on_the_condition_order():
    case = golden("GC-001")
    forward = {c.id: [passage(i)] for i, c in enumerate(CONDITIONS, start=1)}
    backward = dict(reversed(list(forward.items())))
    assert build_prompt(B4, case, forward) == build_prompt(B4, case, backward)
    other = case.model_copy(update={"case_id": "GC-009"})
    first = [line for line in build_prompt(B4, case, forward).splitlines() if "<passage" in line]
    second = [line for line in build_prompt(B4, other, forward).splitlines() if "<passage" in line]
    assert sorted(first) == sorted(second) and first != second


def test_given_result_has_the_fourteen_conditions_and_only_the_passages_shown():
    case = golden("GC-001")
    result = given_result(case, {"COND:gerd": [passage(5)]})
    assert [c.condition_id for c in result.candidates] == [c.id for c in CONDITIONS]
    assert result.candidates[[c.id for c in CONDITIONS].index("COND:gerd")].evidence[
        0
    ].citation == ("PMCID:PMC0000005")
    assert sum(len(c.evidence) for c in result.candidates) == 1
    assert not result.red_flags and not result.suggested_next_tests
    assert all(not c.assessments for c in result.candidates)


# --------------------------------------------------------------------------------------------- #
# Reading the answer
# --------------------------------------------------------------------------------------------- #

GOOD = """RANKING:
1. Possible NSTEMI / STEMI
2. Unstable angina
3. Stable angina
4. Pericarditis
5. GERD
EXPLANATION:
Pressure-type pain with diaphoresis fits an ischaemic cause. Exertional onset favours angina."""


def test_a_well_formed_answer_is_read_in_full():
    parsed = parse_answer(GOOD)
    assert parsed.ranking == (
        "COND:nstemi_stemi",
        "COND:unstable_angina",
        "COND:stable_angina",
        "COND:pericarditis",
        "COND:gerd",
    )
    assert parsed.explanation.startswith("Pressure-type pain with diaphoresis")
    assert parsed.explanation.endswith("favours angina.")
    assert not parsed.flags and not parsed.unmatched and not parsed.duplicates
    assert not parsed.failed


def test_markdown_numbering_styles_aliases_and_slashes_are_read():
    text = (
        "**RANKING:**\n"
        "1) **Possible NSTEMI/STEMI** (the pain is typical)\n"
        "2: acute myocardial infarction\n"  # an alias of the first: a repeat
        "3. Unstable angina - rest pain\n"
        "4. Heart attack\n"  # also an alias of the first
        "5. `GERD`\n"
        "**EXPLANATION:** It fits."
    )
    parsed = parse_answer(text)
    assert parsed.ranking == ("COND:nstemi_stemi", "COND:unstable_angina", "COND:gerd")
    assert parsed.duplicates == 2
    assert parsed.explanation == "It fits."
    assert not parsed.flags


def test_an_item_outside_the_registry_is_kept_and_not_ranked():
    parsed = parse_answer("RANKING:\n1. Pneumonia\n2. GERD\n3. Costochondritis\nEXPLANATION:\nx")
    assert parsed.ranking == ("COND:gerd",)
    assert parsed.unmatched == ("Pneumonia", "Costochondritis")


@pytest.mark.parametrize(
    "text, flag",
    [
        ("", "empty_answer"),
        ("   \n  ", "empty_answer"),
        ("I cannot answer that.", "no_ranking_header"),
        ("RANKING:\n1. Pneumonia\nEXPLANATION:\nNo.", "no_valid_condition"),
    ],
)
def test_an_answer_with_no_ranking_is_a_failure_and_says_why(text, flag):
    parsed = parse_answer(text)
    assert parsed.failed and flag in parsed.flags


def test_a_missing_explanation_header_gives_no_explanation_and_a_flag():
    parsed = parse_answer("RANKING:\n1. GERD\n2. Panic attack\n")
    assert parsed.ranking == ("COND:gerd", "COND:panic_attack")
    assert parsed.explanation == "" and "no_explanation_header" in parsed.flags


def test_without_a_ranking_header_the_numbered_lines_before_the_explanation_count():
    parsed = parse_answer("1. GERD\n2. Pericarditis\nEXPLANATION: Burning pain.\n3. Panic attack")
    assert parsed.ranking == ("COND:gerd", "COND:pericarditis")  # the 3rd is after the header
    assert "no_ranking_header" in parsed.flags
    assert parsed.explanation == "Burning pain. 3. Panic attack"


def test_a_word_that_only_starts_with_explanation_is_not_the_header():
    parsed = parse_answer("RANKING:\n1. GERD\nEXPLANATION:\nExplanations vary. They are brief.")
    assert parsed.explanation == "Explanations vary. They are brief."


def test_more_than_five_conditions_are_all_read_in_order():
    names = [c.label for c in CONDITIONS[:7]]
    text = (
        "RANKING:\n" + "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1)) + "\nEXPLANATION:\nx"
    )
    assert parse_answer(text).ranking == tuple(c.id for c in CONDITIONS[:7])


def test_every_registry_label_is_read_back_to_its_own_id():
    for condition in CONDITIONS:
        parsed = parse_answer(f"RANKING:\n1. {condition.label}\nEXPLANATION:\nx")
        assert parsed.ranking == (condition.id,), condition.label
        assert BY_ID[condition.id].label == condition.label


# --------------------------------------------------------------------------------------------- #
# The client's two new options
# --------------------------------------------------------------------------------------------- #


def test_the_client_sends_a_seed_and_a_context_only_when_asked():
    from src.llm.client import OllamaClient

    plain = OllamaClient("m").payload("p", "s")
    assert plain["options"] == {"temperature": 0.1, "num_predict": 512}  # as before
    pinned = OllamaClient("m", temperature=0.0, seed=42, num_ctx=8192).payload("p", "s")
    assert pinned["options"] == {
        "temperature": 0.0,
        "num_predict": 512,
        "seed": 42,
        "num_ctx": 8192,
    }
    assert pinned["model"] == "m" and pinned["stream"] is False
