"""The model's reasons (2b): leave-one-finding-out occlusion, the explanation and the dashboard.

The toy tests use a linear softmax model over one column per finding, so every reason's direction
is known in advance. The data-gated tests run the configured B1-LR′+aug on the golden cases and
check properties, not clinical expectations: the inversion from hand-written findings to DDXPlus
answers has no ground truth (R-17).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from app.view import candidate_rows, model_reasons
from src.contracts import Assertion, Candidate, DiagnosisResult, Finding, PatientCase
from src.ml.case_tokens import UnrepresentableCase
from src.ml.reasons import Reason, Reasons, log_odds, occlusion, reasons_sentence
from src.stubs import TemplateExplainer

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("COND:nstemi_stemi", "COND:gerd", "COND:psvt")
CONCEPTS = ("SYM:a", "SYM:b", "SYM:c", "SYM:d", "DDX:E_1")
WEIGHTS = np.array(
    # nstemi, gerd, psvt: each finding's push on each condition's logit
    [
        [2.0, 0.0, 0.0],  # SYM:a supports MI
        [-3.0, 1.0, 0.0],  # SYM:b argues against MI
        [1.0, 0.0, 0.0],  # SYM:c: presence supports MI, so its denial argues against
        [0.0, 0.0, 0.0],  # SYM:d: the model cannot see it
        [0.5, 0.0, 0.0],  # DDX:E_1: derived from SYM:a for the graph
    ]
)


class ToyModel:
    labels = LABELS

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        self.calls.append(len(rows))
        logits = np.asarray(rows) @ WEIGHTS
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def toy_encode(case: PatientCase, encoder, *, include_narrower: bool = True):  # noqa: ARG001
    row = np.zeros(len(CONCEPTS))
    for finding in [*case.findings, *case.risk_factors]:
        if finding.concept_id in CONCEPTS:
            row[CONCEPTS.index(finding.concept_id)] = (
                1.0 if finding.assertion is Assertion.PRESENT else -1.0
            )
    return row, None


def toy_case(**extra) -> PatientCase:
    findings = [
        Finding(concept_id="SYM:a", label="A"),
        Finding(concept_id="SYM:b", label="B"),
        Finding(concept_id="SYM:c", label="C", assertion=Assertion.ABSENT),
        Finding(concept_id="SYM:d", label="D"),
        *extra.get("derived", []),
    ]
    return PatientCase(case_id="toy", age=60, sex="M", findings=findings)


def names(reasons: tuple[Reason, ...]) -> list[str]:
    return [r.concept_id for r in reasons]


def test_each_finding_s_direction_is_the_model_s():
    found = occlusion(ToyModel(), None, toy_case(), encode=toy_encode)
    mi = found["COND:nstemi_stemi"]
    assert names(mi.supporting) == ["SYM:a"]
    assert names(mi.against) == ["SYM:b", "SYM:c"]  # strongest first
    assert all(r.effect > 0 for r in mi.supporting) and all(r.effect < 0 for r in mi.against)
    assert mi.against[1].denied and mi.against[1].text() == "C (denied)"


def test_a_finding_the_model_cannot_see_is_no_reason():
    found = occlusion(ToyModel(), None, toy_case(), encode=toy_encode)
    for entry in found.values():
        assert "SYM:d" not in names(entry.supporting) + names(entry.against)


def test_the_case_and_every_variant_are_scored_in_one_batch():
    model = ToyModel()
    occlusion(model, None, toy_case(), encode=toy_encode)
    assert model.calls == [1 + 4]


def test_a_finding_derived_for_the_graph_goes_with_the_finding_it_came_from():
    derived = Finding(
        concept_id="DDX:E_1",
        label="Question 1",
        source="crosswalk",
        qualifiers={"crosswalk_from": "SYM:a"},
    )
    plain = occlusion(ToyModel(), None, toy_case(), encode=toy_encode)
    expanded = occlusion(ToyModel(), None, toy_case(derived=[derived]), encode=toy_encode)
    mi, mi_expanded = plain["COND:nstemi_stemi"], expanded["COND:nstemi_stemi"]
    every = names(mi_expanded.supporting) + names(mi_expanded.against)
    assert "DDX:E_1" not in every
    # Removing SYM:a also removes the 0.5 its derived question added.
    assert mi_expanded.supporting[0].effect > mi.supporting[0].effect


def test_a_case_without_findings_has_no_reasons():
    case = PatientCase(case_id="empty", age=40, sex="F")
    found = occlusion(ToyModel(), None, case, encode=toy_encode)
    assert set(found) == set(LABELS)
    assert all(not e.supporting and not e.against for e in found.values())


def test_a_case_the_model_cannot_hold_gives_none():
    def refuse(*args, **kwargs):
        raise UnrepresentableCase("no row for Sex.OTHER")

    assert occlusion(ToyModel(), None, toy_case(), encode=refuse) is None


def test_small_changes_are_no_reason_either_way():
    found = occlusion(ToyModel(), None, toy_case(), encode=toy_encode, min_effect=100.0)
    assert all(not e.supporting and not e.against for e in found.values())


def test_log_odds_keep_their_size_when_the_model_is_near_certain():
    p = np.array([[0.9998, 0.0001, 0.0001], [0.9990, 0.0005, 0.0005], [1.0, 1e-20, 1e-20]])
    odds = log_odds(p)
    assert np.all(np.isfinite(odds))
    assert odds[0, 0] - odds[1, 0] == pytest.approx(np.log(0.9998 / 0.0002) - np.log(0.999 / 0.001))
    assert odds[2, 0] > odds[0, 0]


def test_the_sentence_names_findings_never_numbers():
    reasons = {
        "COND:gerd": Reasons(
            "COND:gerd",
            supporting=(Reason("SYM:x", "Burning pain", False, 2.5),),
            against=(Reason("SYM:y", "Exertional onset", True, -1.2),),
        ),
        "COND:psvt": Reasons("COND:psvt", against=(Reason("SYM:z", "Palpitations", True, -0.4),)),
        "COND:nstemi_stemi": Reasons("COND:nstemi_stemi"),
    }
    text = reasons_sentence(reasons, "COND:gerd", "GERD", 3)
    assert "Burning pain" in text and "Exertional onset (denied)" in text
    assert not any(ch.isdigit() for ch in text)
    assert reasons_sentence(reasons, "COND:psvt", "PSVT", 3).startswith(
        "The model's strongest reasons against PSVT"
    )
    assert reasons_sentence(reasons, "COND:nstemi_stemi", "MI", 3) == ""
    assert reasons_sentence(reasons, "COND:aortic_dissection", "Aortic dissection", 3) == ""


# --------------------------------------------------------------------------- #
# The explanation and the dashboard
# --------------------------------------------------------------------------- #


class ToyRanker:
    degraded = False

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def reasons(self, case: PatientCase):
        if self.fail:
            raise RuntimeError("broken model")
        return occlusion(ToyModel(), None, case, encode=toy_encode)


def candidates(ml_score: float = 0.9) -> list[Candidate]:
    return [
        Candidate(condition_id="COND:nstemi_stemi", label="Acute MI", ml_score=ml_score),
        Candidate(condition_id="COND:gerd", label="GERD", ml_score=ml_score / 10),
    ]


def test_the_explanation_gives_the_model_s_reasons_for_its_top_candidate():
    text = TemplateExplainer(ranker=ToyRanker()).explain(toy_case(), candidates()).text
    assert "The model's strongest reasons for Acute MI: A; against it: B, C (denied)" in text


@pytest.mark.parametrize(
    "ranker, scores",
    [(None, 0.9), (ToyRanker(fail=True), 0.9), (ToyRanker(), 0.0)],
    ids=["no ranker", "the reasons fail", "the model did not answer"],
)
def test_the_explanation_goes_without_reasons_it_cannot_give(ranker, scores):
    explanation = TemplateExplainer(ranker=ranker).explain(toy_case(), candidates(scores))
    assert "The model's" not in explanation.text
    assert explanation.grounded


def test_a_degraded_ranker_gives_no_reasons():
    ranker = ToyRanker()
    ranker.degraded = True
    assert (
        "The model's" not in TemplateExplainer(ranker=ranker).explain(toy_case(), candidates()).text
    )
    assert model_reasons(ranker, toy_case()) is None


def test_the_dashboard_rows_carry_the_reasons_unless_the_model_failed():
    found = model_reasons(ToyRanker(), toy_case())
    result = DiagnosisResult(case_id="toy", candidates=candidates())
    row = candidate_rows(result, None, found)[0]
    assert row["model_for"] == ["A"] and row["model_against"] == ["B", "C (denied)"]
    failed = DiagnosisResult(case_id="toy", candidates=candidates(), degraded_components=["ml"])
    assert candidate_rows(failed, None, found)[0]["model_for"] == []
    assert model_reasons(ToyRanker(fail=True), toy_case()) is None
    assert model_reasons(object(), toy_case()) is None


# --------------------------------------------------------------------------- #
# The configured model, on the golden cases (data-gated)
# --------------------------------------------------------------------------- #

EVIDENCES = REPO_ROOT / "data" / "raw" / "ddxplus" / "release_evidences.json"
CONDITIONS_FILE = REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json"
CONFIGURED = REPO_ROOT / "models" / "nightingale_b1_asked" / "b1_asked_aug"
GOLDEN = yaml.safe_load(
    (REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml").read_text("utf-8")
)["cases"]


@pytest.mark.skipif(
    not all(p.exists() for p in (EVIDENCES, CONDITIONS_FILE, CONFIGURED)),
    reason="data/ and models/ are not committed (CI)",
)
@pytest.mark.parametrize("golden", GOLDEN, ids=[g["id"] for g in GOLDEN])
def test_the_configured_model_s_reasons_are_its_own(golden):
    from src.ml.ranker import open_ranker
    from src.ml.reasons import _own_findings, _without

    ranker = open_ranker(
        model_dir=CONFIGURED, evidences_path=EVIDENCES, conditions_path=CONDITIONS_FILE
    )
    case = PatientCase(**golden["case"])
    scores = ranker.score(case)
    before = ranker.last_tokens
    found = ranker.reasons(case)
    assert ranker.last_tokens is before  # nothing stored: a cached ranker stays truthful
    recorded = {f.concept_id for _, _, f in _own_findings(case)}
    for entry in found.values():
        assert {r.concept_id for r in [*entry.supporting, *entry.against]} <= recorded
        effects = [r.effect for r in entry.supporting]
        assert effects == sorted(effects, reverse=True)
    top = max(scores, key=scores.get)
    if found[top].supporting:
        strongest = found[top].supporting[0]
        name, index = next(
            (n, i) for n, i, f in _own_findings(case) if f.concept_id == strongest.concept_id
        )
        assert ranker.score(_without(case, name, index))[top] < scores[top]
