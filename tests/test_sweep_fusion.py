"""EXP-006's weight sweep: the pre-registered rule and the ranking it scores (scripts/sweep_fusion.py).

The sweep itself needs data/ and runs only where the owner says (D-7); these tests need neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import ranking_from_scores

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import sweep_fusion  # noqa: E402


def entry(alpha: float, mrr: tuple[float, float, float], mnm: tuple[float, float, float]):
    levels = ("1.0", "0.5", "0.25")
    return {
        "alpha": alpha,
        "levels": {
            lv: {"MRR": r, "must-not-miss recall@3": m}
            for lv, r, m in zip(levels, mrr, mnm, strict=True)
        },
    }


FLOOR = {"1.0": 1.0, "0.5": 0.99, "0.25": 0.95}


def every(sweep, holds=True):
    return {e["alpha"]: holds for e in sweep}


def test_the_rule_takes_the_best_mrr_that_keeps_must_not_miss_recall():
    sweep = [
        entry(0.1, (0.90, 0.90, 0.90), (1.0, 0.99, 0.95)),
        entry(0.5, (0.95, 0.95, 0.95), (1.0, 0.99, 0.94)),  # best MRR, but loses safety at 25%
        entry(0.3, (0.92, 0.92, 0.92), (1.0, 0.995, 0.96)),
    ]
    assert sweep_fusion.choose(sweep, FLOOR, every(sweep))["alpha"] == 0.3


def test_an_alpha_that_breaks_a_golden_case_is_never_taken():
    sweep = [
        entry(0.1, (0.90, 0.90, 0.90), (1.0, 0.99, 0.95)),
        entry(0.3, (0.99, 0.99, 0.99), (1.0, 0.995, 0.96)),  # best, but a golden case fails
    ]
    golden = {0.1: True, 0.3: False}
    assert sweep_fusion.choose(sweep, FLOOR, golden)["alpha"] == 0.1


def test_when_no_alpha_keeps_the_golden_cases_none_is_adopted():
    sweep = [entry(0.1, (0.9, 0.9, 0.9), (1.0, 0.99, 0.95))]
    chosen = sweep_fusion.choose(sweep, FLOOR, every(sweep, holds=False))
    assert chosen["alpha"] is None
    assert "none adopted" in chosen["rule"]


def test_ties_go_to_the_smaller_weight():
    sweep = [entry(a, (0.9, 0.9, 0.9), (1.0, 0.99, 0.95)) for a in (0.4, 0.2, 0.6)]
    assert sweep_fusion.choose(sweep, FLOOR, every(sweep))["alpha"] == 0.2


def test_when_nothing_is_safe_enough_the_safest_golden_admissible_alpha_is_taken():
    sweep = [
        entry(0.1, (0.99, 0.99, 0.99), (0.9, 0.9, 0.9)),
        entry(0.5, (0.80, 0.80, 0.80), (0.95, 0.95, 0.94)),
        entry(0.7, (0.70, 0.70, 0.70), (0.99, 0.99, 0.94)),  # safest, but a golden case fails
    ]
    golden = {0.1: True, 0.5: True, 0.7: False}
    chosen = sweep_fusion.choose(sweep, FLOOR, golden)
    assert chosen["alpha"] == 0.5
    assert "no α met the constraint" in chosen["rule"]


def test_the_sweep_breaks_ties_like_the_metrics_do():
    rng = np.random.default_rng(3)
    scores = rng.integers(0, 3, size=(50, len(sweep_fusion.IDS))).astype(float)  # many ties
    order = sweep_fusion.orders(scores)
    for i in range(50):
        expected = ranking_from_scores(dict(zip(sweep_fusion.IDS, scores[i], strict=True)))
        assert [sweep_fusion.IDS[j] for j in order[i]] == list(expected)


def test_true_ranks_are_one_based():
    order = np.array([[2, 0, 1], [0, 1, 2]])
    assert sweep_fusion.true_ranks(order, np.array([0, 2])).tolist() == [2, 3]


def test_the_model_s_columns_are_the_registry_s_first():
    """The sweep reads a model's column index as a registry index (and checks each model)."""
    assert tuple(range(len(sweep_fusion.MODEL_COLUMNS))) == sweep_fusion.MODEL_COLUMNS
    assert sweep_fusion.IDS[sweep_fusion.DISSECTION] == "COND:aortic_dissection"


def test_flagged_conditions_sort_first_without_losing_precision():
    """Two keys, as the pipeline's (red_flag, fused_score) sort: a 1e-13 gap still orders."""
    fused = np.array([[-0.1, -700.0, -3.0, -0.1 + 1e-13]])
    flagged = np.array([[True, True, False, True]])
    assert sweep_fusion.orders(fused, flagged)[0].tolist() == [3, 0, 1, 2]


def test_the_grid_runs_from_005_to_the_graph_alone():
    """α = 0 is left out: not a fusion (the floor is off), and not the model alone either."""
    assert tuple(round(0.05 * i, 2) for i in range(1, 21)) == sweep_fusion.GRID


def test_dissection_s_cost_is_counted():
    ids = sweep_fusion.IDS
    d = sweep_fusion.DISSECTION
    mi, gerd = ids.index("COND:nstemi_stemi"), ids.index("COND:gerd")
    rest = [j for j in range(len(ids)) if j not in (mi, gerd, d)]
    # Patient 0: MI (must-not-miss) is 4th behind dissection; patient 1: GERD 1st, dissection 2nd.
    order = np.array([[rest[0], d, rest[1], mi, gerd, *rest[2:]], [gerd, d, mi, *rest]])
    truth = np.array([mi, gerd])
    critical = np.array([True, False])
    in_d = np.zeros((2, len(ids)), dtype=bool)
    in_d[0, [mi, rest[0]]] = True
    in_d[1, [gerd, mi]] = True
    effects = sweep_fusion.dissection_effects(order, truth, critical, in_d)
    assert effects["dissection in the top 3"] == 1.0
    assert effects["must-not-miss patients pushed out of the top 3 by dissection"] == 1
    assert effects["Precision@3"] == (1 / 3 + 2 / 3) / 2


def test_the_differential_matrix_keeps_only_in_scope_conditions():
    frame = pd.DataFrame(
        {
            "label_differential": [
                [{"condition_id": "COND:gerd"}, {"condition_id": None}],
                [{"condition_id": "COND:psvt"}],
            ]
        }
    )
    in_d = sweep_fusion.differential_matrix(frame)
    ids = sweep_fusion.IDS
    assert in_d[0].sum() == 1 and in_d[0, ids.index("COND:gerd")]
    assert in_d[1, ids.index("COND:psvt")]
