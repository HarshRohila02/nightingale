"""The ablation matrix as rankings (src/eval/ablation.py): the vectorised twin of the pipeline.

The pipeline is the reference. Every ablation that changes the ranking is run through the real
``DiagnosisPipeline`` with fake components and through ``ablation.ranking``, on the same arrays,
and the two orders must be the same. The comparisons are checked against hand-counted cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.conditions import CONDITIONS
from src.contracts import PatientCase
from src.eval import ablation
from src.eval.ablation import IDS, MODEL_COLUMNS, MODEL_LABELS, SPECS, CaseArrays
from src.pipeline import DiagnosisPipeline
from src.stubs import EmptyRetriever, TemplateExplainer

N = 14
RNG = np.random.default_rng(7)


def arrays(patients: int = N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model probabilities (13 columns), graph log-likelihoods (14) and flags (14).

    Row 0 has tied graph scores (as ``scoring.ROUND`` makes common), row 1 is a model that
    abstains (all zero), row 2 is a flat model, the rest are random.
    """
    ml = RNG.dirichlet(np.ones(len(MODEL_COLUMNS)) * 0.4, size=patients)
    ml[1, :] = 0.0
    ml[2, :] = 1.0 / len(MODEL_COLUMNS)
    graph = -np.abs(RNG.normal(size=(patients, len(IDS))) * 4)
    graph[0, :] = np.round(graph[0, :])
    flagged = RNG.random((patients, len(IDS))) < 0.2
    return ml, graph, flagged


class Fake:
    """A ranker and a graph store that answer from the arrays, one patient at a time."""

    def __init__(self, ml: np.ndarray | None, graph: np.ndarray | None) -> None:
        self.ml, self.graph = ml, graph

    def score(self, case: PatientCase) -> dict[str, float]:
        if self.ml is None:
            raise RuntimeError("no model")
        row = self.ml[int(case.case_id)]
        return {label: float(v) for label, v in zip(MODEL_LABELS, row, strict=True)}

    def score_by_connectivity(self, case: PatientCase) -> dict[str, float]:
        if self.graph is None:
            raise RuntimeError("no graph")
        return {c: float(v) for c, v in zip(IDS, self.graph[int(case.case_id)], strict=True)}

    def paths_for(self, case: PatientCase, condition_id: str) -> list:  # noqa: ARG002
        return []

    def expected_findings(self, condition_id: str) -> list:  # noqa: ARG002
        return []


def pipeline_orders(
    spec_name: str,
    ml: np.ndarray,
    graph: np.ndarray,
    flagged: np.ndarray,
    alpha: float,
    monkeypatch,
) -> list[list[str]]:
    """What the real pipeline ranks, for each patient, under an ablation."""
    spec = SPECS[spec_name]

    def fake_flags(case: PatientCase) -> dict[str, str]:
        row = flagged[int(case.case_id)]
        return {IDS[j]: "flag" for j in np.flatnonzero(row)}

    monkeypatch.setattr("src.pipeline.evaluate_red_flags", fake_flags)
    store = Fake(ml if spec.model else None, graph if spec.graph else None)
    pipe = DiagnosisPipeline(
        ranker=store,
        graph=store,
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
        enable_red_flags=spec.flags,
        ml_weight=1 - alpha,
        kg_weight=alpha,
    )
    ranked = []
    for i in range(len(ml)):
        result = pipe.run(PatientCase(case_id=str(i), age=50, sex="F"))
        ranked.append([c.condition_id for c in result.candidates])
    return ranked


@pytest.mark.parametrize("name", sorted(SPECS))
@pytest.mark.parametrize("alpha", [0.15, 0.5, 0.9])
def test_the_vectorised_ranking_is_the_pipelines(name, alpha, monkeypatch):
    ml, graph, flagged = arrays()
    mine = ablation.ranking(SPECS[name], ml, graph, flagged, alpha)
    theirs = pipeline_orders(name, ml, graph, flagged, alpha, monkeypatch)
    for i, expected in enumerate(theirs):
        assert [IDS[j] for j in mine[i]] == expected, f"{name} α={alpha} patient {i}"


def test_a7_ranks_like_a0_and_the_identities_are_named():
    assert SPECS["A7"] == SPECS["A0"]
    assert set(ablation.SAME_AS_A0) == {"A3", "A5", "A6"}
    assert all(ablation.SAME_AS_A0.values())


def test_a_spec_with_neither_model_nor_graph_is_refused():
    ml, graph, _ = arrays(3)
    with pytest.raises(ValueError):
        ablation.ablation_scores(ablation.Spec(False, False, True), ml, graph, 0.5)


def test_negative_model_scores_are_refused():
    ml, graph, _ = arrays(3)
    ml[0, MODEL_COLUMNS[0]] = -0.1
    with pytest.raises(ValueError):
        ablation.ablation_scores(SPECS["A1"], ml, graph, 0.5)


def test_flagged_conditions_come_first_then_score_then_registry_order():
    scores = np.array([[0.1, 0.9, 0.9, 0.5]])
    flagged = np.array([[False, False, False, True]])
    assert ablation.orders(scores).tolist() == [[1, 2, 3, 0]]
    assert ablation.orders(scores, flagged).tolist() == [[3, 1, 2, 0]]


def test_true_ranks_are_one_based_and_a_missing_condition_is_an_error():
    order = np.array([[2, 0, 1], [1, 2, 0]])
    assert ablation.true_ranks(order, np.array([0, 1])).tolist() == [2, 1]
    with pytest.raises(ValueError):
        ablation.true_ranks(np.array([[0, 1]]), np.array([2]))


def test_dissection_is_last_for_a_model_alone():
    ml, graph, flagged = arrays(5)
    order = ablation.ranking(SPECS["A1"], ml, graph, flagged, 0.5)
    dissection = IDS.index("COND:aortic_dissection")
    for i in (0, 3, 4):  # the rows where the model speaks
        assert order[i, -1] == dissection


def test_an_abstaining_model_falls_back_to_registry_order_without_a_graph():
    ml, graph, flagged = arrays(3)
    order = ablation.ranking(SPECS["A1"], ml, graph, flagged, 0.5)
    assert order[1].tolist() == list(range(len(IDS)))


# --------------------------------------------------------------------------- #
# Comparisons, with hand-counted cases
# --------------------------------------------------------------------------- #


def cases(ranks, critical) -> CaseArrays:
    return CaseArrays(ranks=np.array(ranks), critical=np.array(critical, dtype=bool))


def test_point_metrics_by_hand():
    c = cases([1, 2, 4, 6], [True, True, True, False])
    m = ablation.point_metrics(c)
    assert m["top-1 accuracy"] == 0.25
    assert m["top-3 accuracy"] == 0.5
    assert m["top-5 accuracy"] == 0.75
    assert m["MRR"] == pytest.approx((1 + 0.5 + 0.25 + 1 / 6) / 4)
    assert m["must-not-miss recall@3"] == pytest.approx(2 / 3)
    assert m["dangerous false-negative rate"] == 0.0


def test_a_clear_win_is_significant_and_in_a_s_favour():
    n = 200
    critical = [True] * 100 + [False] * 100
    a = cases([1] * n, critical)
    b = cases([4] * 60 + [1] * 140, critical)  # b misses the top 3 for 60 patients
    out = ablation.compare(a, b, resamples=200)
    assert out["top3"]["only_a"] == 60 and out["top3"]["only_b"] == 0
    assert out["top3"]["a_significantly_better"] is True
    assert out["must_not_miss_top3"]["cases"] == 100
    assert out["must_not_miss_top3"]["only_a"] == 60
    assert out["mrr"]["a_significantly_better"] is True
    assert out["mrr"]["difference"] > 0


def test_the_same_system_is_never_better_than_itself():
    a = cases([1, 2, 3, 5] * 30, [True, False] * 60)
    out = ablation.compare(a, a, resamples=100)
    for key in ("top3", "must_not_miss_top3", "mrr"):
        assert out[key]["a_significantly_better"] is False
    assert out["top3"]["p_value"] == 1.0


def test_a_significant_loss_is_not_a_win():
    n = 200
    critical = [True] * n
    a = cases([4] * 60 + [1] * 140, critical)
    b = cases([1] * n, critical)
    out = ablation.compare(a, b, resamples=100)
    assert out["top3"]["p_value"] < 0.05
    assert out["top3"]["a_significantly_better"] is False
    assert out["mrr"]["a_significantly_better"] is False


def test_a_comparator_that_misses_five_or_fewer_cannot_be_beaten_significantly():
    # docs/05 §1: McNemar cannot reach 0.05 when the comparator misses five cases or fewer.
    n = 33_963
    a = cases([1] * n, [True] * n)
    b = cases([4] * 5 + [1] * (n - 5), [True] * n)
    out = ablation.compare(a, b, resamples=50)
    assert out["top3"]["only_a"] == 5
    assert out["top3"]["p_value"] == pytest.approx(0.0625)
    assert out["top3"]["a_significantly_better"] is False


def test_systems_on_different_patients_are_refused():
    with pytest.raises(ValueError):
        ablation.compare(cases([1, 1], [True, True]), cases([1], [True]))
    with pytest.raises(ValueError):
        ablation.compare(cases([1, 1], [True, False]), cases([1, 1], [False, True]))


def test_the_degradation_contrast_by_hand():
    critical = [True] * 50 + [False] * 150
    full = cases([1] * 200, critical)
    # The ablated system loses the top 3 for 25 of the 50 must-not-miss patients (-0.5) and the
    # top 1 for those 25 plus 10 others (-35/200 = -0.175 of all patients).
    ablated = cases([5] * 25 + [1] * 25 + [2] * 10 + [1] * 140, critical)
    out = ablation.degradation_contrast(full, ablated, resamples=300)
    assert out["degradation_must_not_miss"] == pytest.approx(0.5)
    assert out["degradation_top1"] == pytest.approx(35 / 200)
    assert out["contrast"] == pytest.approx(0.5 - 35 / 200)
    assert out["more_on_must_not_miss"] is True
    low, high = out["contrast_ci"]
    assert low > 0 and high < 1


def test_no_degradation_is_not_more_on_must_not_miss():
    c = cases([1, 2, 7, 1] * 50, [True, False, True, False] * 50)
    out = ablation.degradation_contrast(c, c, resamples=100)
    assert out["contrast"] == 0 and out["more_on_must_not_miss"] is False


def test_seed_summary_has_a_sample_standard_deviation():
    per_seed = {42: {"MRR": 0.9}, 43: {"MRR": 0.8}, 44: {"MRR": 1.0}}
    out = ablation.seed_summary(per_seed)["MRR"]
    assert out["mean"] == pytest.approx(0.9)
    assert out["std"] == pytest.approx(0.1)
    assert out["seeds"] == 3 and out["min"] == 0.8 and out["max"] == 1.0
    assert math.isclose(ablation.seed_summary({1: {"x": 2.0}})["x"]["std"], 0.0)


def test_the_registry_has_fourteen_conditions_and_thirteen_model_columns():
    assert len(IDS) == len(CONDITIONS) == 14
    assert len(MODEL_COLUMNS) == 13
    assert IDS.index("COND:aortic_dissection") not in MODEL_COLUMNS
