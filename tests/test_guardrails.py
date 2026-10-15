"""Guardrails v1 (task 3c, src/reasoning/guardrails.py): claim support and PMC-only citations.

The first test is the one docs/17 T5 names: every sentence the template explainer writes traces to
the result, for every golden case, with red flags on and off, so the pipeline's own explanations
stay grounded. The rest check the guardrail on invented explanations and results: an untraced
sentence is caught and a traced one kept, a sentence must trace to *one* source rather than to
words gathered from several, a citation the retriever never returned is removed, and no link
leads anywhere but PMC.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.conditions import BY_ID, CONDITIONS
from src.contracts import (
    Candidate,
    DiagnosisResult,
    Evidence,
    EvidenceRole,
    Explanation,
    Finding,
    FindingAssessment,
    PatientCase,
)
from src.ml.reasons import Reason, Reasons
from src.pipeline import DiagnosisPipeline
from src.rag.retriever import open_retriever
from src.reasoning.guardrails import (
    UNCITED,
    UNTRACED,
    check_explanation,
    citation_link,
    cited_sources,
    sentences,
    untraced_sentences,
)
from src.stubs import ConstantRanker, EmptyRetriever, InMemoryGraphStore, TemplateExplainer
from tests.rag_fixture import write_fake_bundle

GOLDEN = Path(__file__).parent / "fixtures" / "golden_cases.yaml"
CASES = {c["id"]: c for c in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]}
PE, PERI, MI = "COND:pulmonary_embolism", "COND:pericarditis", "COND:nstemi_stemi"


class ReasoningRanker:
    """A ranker with the model's reasons (2b), so the template writes its reasons sentence: each
    condition rests on the case's first finding and is held back by its last."""

    degraded = False

    def score(self, case: PatientCase) -> dict[str, float]:
        trained = [c.id for c in CONDITIONS if c.in_training_data]
        return {cid: 1 / len(trained) for cid in trained}

    def reasons(self, case: PatientCase) -> dict[str, Reasons]:
        first, last = case.findings[0], case.findings[-1]
        return {
            c.id: Reasons(
                c.id,
                supporting=(Reason(first.concept_id, first.label, False, 1.0),),
                against=(Reason(last.concept_id, last.label, True, -1.0),),
            )
            for c in CONDITIONS
        }


def golden(case_id: str) -> PatientCase:
    return PatientCase(**CASES[case_id]["case"])


# --------------------------------------------------------------------------- #
# The template's explanations trace, so the pipeline's stay grounded
# --------------------------------------------------------------------------- #


@pytest.mark.golden
@pytest.mark.parametrize("ranker", [ConstantRanker, ReasoningRanker])
@pytest.mark.parametrize("red_flags", [True, False])
@pytest.mark.parametrize("case_id", list(CASES))
def test_every_golden_explanation_traces_to_its_result(case_id, red_flags, ranker):
    ranker = ranker()
    pipeline = DiagnosisPipeline(
        ranker,
        InMemoryGraphStore(),
        EmptyRetriever(),
        TemplateExplainer(ranker=ranker),
        enable_red_flags=red_flags,
    )
    result = pipeline.run(golden(case_id))
    explanation = result.explanation
    assert explanation.grounded, explanation.unsupported_claims
    assert explanation.unsupported_claims == []
    assert explanation.citations == []
    if isinstance(ranker, ReasoningRanker):
        assert "The model's strongest reasons for" in explanation.text
    if red_flags and CASES[case_id]["expect"].get("red_flag_for"):
        assert "Red flags raised" in explanation.text


def test_the_guardrail_changes_nothing_but_the_explanation(tmp_path):
    """With evidence attached too: the candidates, their order, the flags and the tests are the
    same objects before and after the check."""
    ranker = ReasoningRanker()
    pipeline = DiagnosisPipeline(
        ranker,
        InMemoryGraphStore(),
        open_retriever(write_fake_bundle(tmp_path / "bundle")),
        TemplateExplainer(ranker=ranker),
        top_k_evidence=14,
    )
    for case_id in CASES:
        result = pipeline.run(golden(case_id))
        checked = check_explanation(result, golden(case_id))
        assert checked is result, "a grounded template explanation needs no change"
        assert result.explanation.grounded and result.explanation.unsupported_claims == []


# --------------------------------------------------------------------------- #
# Invented explanations
# --------------------------------------------------------------------------- #

CASE = PatientCase(
    case_id="T-1",
    age=40,
    sex="F",
    findings=[
        Finding(concept_id="SYM:chest_pain", label="Chest pain"),
        Finding(concept_id="SYM:pleuritic", label="Pleuritic pain"),
    ],
)
PASSAGE = Evidence(
    source="PMC",
    title="Pericarditis with chest pain: an invented case",
    passage="Acute pericarditis typically causes sharp pleuritic chest pain.",
    citation="PMCID:PMC99990001",
    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC99990001/",
)


def result_with(text: str, citations: list[str] | None = None) -> DiagnosisResult:
    """PE (flagged) and pericarditis (with one passage), their tests, and ``text`` to check."""
    pe = Candidate(
        condition_id=PE,
        label=BY_ID[PE].label,
        red_flag=True,
        red_flag_reason="Pleuritic pain with a Wells risk factor.",
        assessments=[
            FindingAssessment(
                finding_id="SYM:pleuritic", label="Pleuritic pain", role=EvidenceRole.SUPPORTING
            )
        ],
    )
    peri = Candidate(condition_id=PERI, label=BY_ID[PERI].label, evidence=[PASSAGE])
    return DiagnosisResult(
        case_id="T-1",
        candidates=[pe, peri],
        red_flags=[f"{pe.label}: {pe.red_flag_reason}"],
        suggested_next_tests=[
            "Pulmonary embolism: consider D-dimer if that probability is not high (ESC 2019 PE)",
            "Pericarditis: consider a 12-lead ECG; CRP and troponin (ESC 2015 pericardial)",
        ],
        explanation=Explanation(text=text, grounded=True, citations=citations or []),
    )


def test_an_untraced_sentence_is_flagged_and_a_traced_one_kept():
    text = (
        "Top candidate for case T-1: Pulmonary embolism. Supporting findings: Pleuritic pain. "
        "The patient most likely has pneumonia."
    )
    checked = check_explanation(result_with(text), CASE).explanation
    assert checked.grounded is False
    assert checked.unsupported_claims == [UNTRACED + "The patient most likely has pneumonia."]
    assert checked.text == text, "flagged, not suppressed"


@pytest.mark.parametrize(
    "sentence",
    [
        "Pulmonary embolism: consider D-dimer.",  # a suggested test
        "Red flag raised: Pulmonary embolism, pleuritic pain with a Wells risk factor.",
        "The model's strongest reasons for Pericarditis: Chest pain, Pleuritic pain (denied).",
        "Pericarditis: consider CRP and troponin.",
        "The clinician decides.",  # claims nothing
    ],
)
def test_sentences_built_from_one_source_trace(sentence):
    assert untraced_sentences(sentence, result_with(""), CASE) == []


@pytest.mark.parametrize(
    "sentence",
    [
        # Each word is in the result, but no one source holds them together: D-dimer is PE's test.
        "Pericarditis: consider D-dimer.",
        "Pulmonary embolism: consider a 12-lead ECG; CRP and troponin.",
        # A finding the case never recorded, a test and a drug the result never names.
        "The model's strongest reasons for Pericarditis: Fever.",
        "Consider a coronary angiogram.",
        "Aspirin is indicated.",
    ],
)
def test_words_gathered_from_several_sources_or_none_do_not_trace(sentence):
    assert untraced_sentences(sentence, result_with(""), CASE) == [sentence]


def test_a_sentence_that_traces_only_to_a_passage_cites_it():
    text = "Acute pericarditis typically causes sharp pleuritic chest pain."
    checked = check_explanation(result_with(text), CASE).explanation
    assert checked.grounded and checked.unsupported_claims == []
    assert checked.citations == ["PMCID:PMC99990001"]


def test_an_invented_citation_is_removed_and_a_retrieved_one_kept():
    result = result_with(
        "Top candidate for case T-1: Pulmonary embolism.",
        citations=["PMCID:PMC99990001", "PMCID:PMC12345678", "PMCID:PMC99990001"],
    )
    checked = check_explanation(result, CASE).explanation
    assert checked.citations == ["PMCID:PMC99990001"]
    assert checked.unsupported_claims == [UNCITED + "PMCID:PMC12345678"]
    assert checked.grounded is False


def test_earlier_claims_and_an_ungrounded_verdict_are_kept():
    result = result_with("Top candidate for case T-1: Pulmonary embolism.")
    result.explanation.grounded = False
    result.explanation.unsupported_claims = ["listed by the explainer"]
    checked = check_explanation(result, CASE).explanation
    assert checked.grounded is False
    assert checked.unsupported_claims == ["listed by the explainer"]


def test_no_explanation_is_left_alone():
    result = result_with("")
    result.explanation = None
    assert check_explanation(result, CASE) is result


def test_the_pipeline_flags_an_untraced_sentence_and_still_removes_treatment():
    """The guardrail runs before the safety layer: an invented claim is flagged and stays; a
    treatment sentence is flagged here and then removed there (FR-6.5)."""

    class Inventing(TemplateExplainer):
        def explain(self, case, candidates):
            base = super().explain(case, candidates)
            return Explanation(
                text=base.text + " It is probably pneumonia. Give aspirin 300 mg.",
                grounded=True,
            )

    pipeline = DiagnosisPipeline(
        ConstantRanker(), InMemoryGraphStore(), EmptyRetriever(), Inventing()
    )
    explanation = pipeline.run(golden("GC-001")).explanation
    assert explanation.grounded is False
    assert UNTRACED + "It is probably pneumonia." in explanation.unsupported_claims
    assert UNTRACED + "Give aspirin 300 mg." in explanation.unsupported_claims
    assert any(c.startswith("treatment advice removed") for c in explanation.unsupported_claims)
    assert "aspirin" not in explanation.text and "pneumonia" in explanation.text


def test_sentences_split_as_the_safety_layer_splits_them():
    """At "." and "!" only: DDXPlus's questions, which label findings on the real graph, end in
    "?" inside a sentence."""
    assert sentences("One. Two! Is it? 0.5 stays") == ["One.", "Two!", "Is it? 0.5 stays"]


# --------------------------------------------------------------------------- #
# Citations: PMC only, each once
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("citation", "link"),
    [
        ("PMCID:PMC99990001", "https://pmc.ncbi.nlm.nih.gov/articles/PMC99990001/"),
        ("PMC1234567", "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/"),
        (" PMCID:pmc1234567 ", "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/"),
        ("PMID:12345678", None),
        ("https://example.com/PMC1234567", None),
        ("PMCID:PMC12/../../x", None),
        ("PMCID:javascript:alert(1)", None),
        ("", None),
    ],
)
def test_a_citation_links_to_pmc_or_to_nothing(citation, link):
    assert citation_link(citation) == link


def test_cited_sources_are_listed_once_in_rank_order_with_pmc_links_only():
    elsewhere = PASSAGE.model_copy(update={"url": "https://example.com/phishing"})
    second = Evidence(source="PMC", title="Another", passage="x", citation="PMCID:PMC99990002")
    unlinked = Evidence(source="KG", title="A graph fact", passage="y", citation="KG:edge")
    candidates = [
        Candidate(condition_id=PE, label="PE", evidence=[elsewhere, second]),
        Candidate(condition_id=PERI, label="Pericarditis", evidence=[PASSAGE, unlinked]),
    ]
    sources = cited_sources(candidates)
    assert [s.citation for s in sources] == ["PMCID:PMC99990001", "PMCID:PMC99990002", "KG:edge"]
    assert sources[0].url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC99990001/"
    assert sources[2].url is None
    assert all(s.url is None or s.url.startswith("https://pmc.ncbi.nlm.nih.gov/") for s in sources)
