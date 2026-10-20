"""scripts/judge_retrieval.py: EXP-011's blind judging pool and its scoring (task 3a).

Everything runs on invented data. The pool is built over a synthetic bundle (random words and
random vectors, ``scripts/measure_retrieval.py``), where every condition has dozens of articles, so
the three methods disagree and a query's pool is a real mix; a few edge cases use the tiny invented
bundle of ``tests/rag_fixture.py``. The judges are simulated by filling the sheets with a rule.
What is checked:

* the 26 queries of EXP-011, and the pool built from them: reproducible with seed 42 (the shuffle
  re-derived here independently), at most 30 passages a query, exactly what each method shows;
* the blind: no method name, rank or score in any judge sheet or in the instructions;
* the sheets: safe to open in a spreadsheet, never written inside the repository outside data/,
  never over a pool a judge may have started on;
* the scale, word for word as ``docs/08`` EXP-011 states it;
* the scoring: Cohen's kappa and every metric against values worked out by hand, the rule, the
  reading of a judge's sheet (any order, any delimiter, Excel's encoding) and every refusal.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from src.contracts import PatientCase
from src.pipeline import DiagnosisPipeline
from src.rag.bundle import load_bundle
from src.rag.retriever import HybridRetriever, present_concepts
from src.stubs import ConstantRanker, InMemoryGraphStore, TemplateExplainer
from tests.rag_fixture import write_fake_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import judge_retrieval as jr  # noqa: E402
import measure_retrieval  # noqa: E402

METHOD_WORDS = ("bm25", "dense", "hybrid", "rrf", "fused", "score")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def stub_pipeline() -> DiagnosisPipeline:
    return DiagnosisPipeline(ConstantRanker(), InMemoryGraphStore(), None, TemplateExplainer())


@pytest.fixture(scope="module")
def queries() -> list[dict]:
    """EXP-011's 26 queries, with the stub pipeline standing in for the configured one."""
    case_part, degraded = jr.case_queries(jr.load_golden(), stub_pipeline())
    assert not set(degraded) - jr.IGNORABLE_DEGRADATION
    return jr.condition_queries() + case_part


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory) -> Path:
    return measure_retrieval.write_synthetic(tmp_path_factory.mktemp("synthetic") / "bundle", 2000)


@pytest.fixture(scope="module")
def retriever(bundle_dir: Path) -> HybridRetriever:
    return HybridRetriever(load_bundle(bundle_dir), depth=jr.POOL_DEPTH)


@pytest.fixture(scope="module")
def pooled(retriever: HybridRetriever, queries: list[dict]):
    return jr.build_pool(retriever, queries)


@pytest.fixture
def queries_file(tmp_path: Path, queries: list[dict]) -> Path:
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"queries": queries, "degraded": []}), encoding="utf-8")
    return path


def run_pool(bundle: Path, queries_file: Path, out_dir: Path, *extra: str) -> int:
    return jr.main(
        ["pool", "--bundle", str(bundle), "--queries", str(queries_file), "--out-dir", str(out_dir)]
        + list(extra)
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fill(
    source: Path,
    destination: Path,
    grade,
    *,
    treatment: set[str] = frozenset(),
    reverse: bool = False,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
) -> None:
    """A judge's work: the blank sheet with a ``grade(index, row)`` on every row."""
    rows = read_csv(source)
    for i, row in enumerate(rows):
        row["grade"] = str(grade(i, row))
        row["treatment"] = "1" if row["row_id"] in treatment else ""
    if reverse:
        rows.reverse()
    with open(destination, "w", encoding=encoding, newline="", errors="replace") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# The queries
# --------------------------------------------------------------------------- #


def test_the_queries_are_the_26_of_exp011(queries):
    assert len(queries) == 26
    conditions = [q for q in queries if q["kind"] == "condition"]
    cases = [q for q in queries if q["kind"] == "case"]
    assert len(conditions) == 14 and all(q["concepts"] == [] for q in conditions)
    assert len({q["condition_id"] for q in conditions}) == 14
    golden = [g["id"] for g in jr.load_golden()]
    assert len(cases) == 3 * len(golden) == 12
    assert [q["case_id"] for q in cases] == [g for g in golden for _ in range(3)]
    assert [q["candidate_rank"] for q in cases] == [1, 2, 3] * len(golden)
    assert len({q["id"] for q in queries}) == 26
    jr.check_queries(queries)


def test_case_queries_are_the_pipelines_top_three_with_the_present_findings(queries):
    pipeline = stub_pipeline()
    for entry in jr.load_golden():
        case = PatientCase(**entry["case"])
        top = [c.condition_id for c in pipeline.run(case).candidates[:3]]
        mine = [q for q in queries if q["case_id"] == entry["id"]]
        assert [q["condition_id"] for q in mine] == top
        assert all(q["concepts"] == present_concepts(case) for q in mine)
        assert all(len(q["findings"]) == len(q["concepts"]) for q in mine)


def test_a_golden_case_added_before_judging_adds_three_queries():
    golden = jr.load_golden()
    extra = {**golden[0], "id": "GC-005"}
    case_part, _ = jr.case_queries([*golden, extra], stub_pipeline())
    assert len(case_part) == 15


def test_check_queries_refuses_unusable_lists(queries):
    for bad in ([], [queries[0], queries[0]], [{**queries[0], "condition_id": "COND:nope"}]):
        with pytest.raises(ValueError):
            jr.check_queries(bad)
    with pytest.raises(ValueError):
        jr.check_queries([{k: v for k, v in queries[0].items() if k != "concepts"}])


def test_a_degraded_pipeline_is_refused_but_not_without_retrieval(queries_file):
    class Broken:
        def score(self, case):
            raise RuntimeError("the model is gone")

    def build(ranker):
        pipeline = DiagnosisPipeline(ranker, InMemoryGraphStore(), None, TemplateExplainer())
        return lambda config: {"pipeline": pipeline, "labels": None}

    args = type("Args", (), {"queries": None, "config": None})()
    with pytest.raises(SystemExit, match="degraded"):
        jr.queries_for_pool(args, build_pipeline=build(Broken()))
    # Without a retriever the pipeline reports `rag`; that does not change what ranks first.
    built, degraded = jr.queries_for_pool(args, build_pipeline=build(ConstantRanker()))
    assert len(built) == 26 and "rag" in degraded


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #


def test_the_pool_is_reproducible_with_seed_42(retriever, queries, pooled):
    rows, returned = pooled
    again_rows, again_returned = jr.build_pool(retriever, queries)
    assert rows == again_rows and returned == again_returned
    assert [r["row_id"] for r in rows] == [f"R{n:03d}" for n in range(1, len(rows) + 1)]

    # The documented recipe, re-derived: each query's distinct passages in index order, shuffled by
    # one generator seeded 42 and consumed query by query.
    rng = random.Random(42)
    expected = []
    for query in queries:
        shown = jr.run_methods(retriever, query)
        order = sorted({index for lst in shown.values() for index, _ in lst})
        rng.shuffle(order)
        expected += [(query["id"], index) for index in order]
    assert [(r["query_id"], r["passage_index"]) for r in rows] == expected

    other, _ = jr.build_pool(retriever, queries, seed=7)
    assert [r["passage_index"] for r in other] != [r["passage_index"] for r in rows]
    assert sorted((r["query_id"], r["passage_index"]) for r in other) == sorted(expected)


def test_the_pool_holds_exactly_what_each_method_shows(retriever, queries, pooled):
    rows, returned = pooled
    by_query: dict[str, list] = {q["id"]: [] for q in queries}
    for row in rows:
        by_query[row["query_id"]].append(row)
    for query in queries:
        pool = by_query[query["id"]]
        assert 0 < len(pool) <= 3 * jr.POOL_DEPTH
        shown = jr.run_methods(retriever, query)
        for method, listed in shown.items():
            assert len(listed) <= jr.POOL_DEPTH
            assert returned[query["id"]][method] == len(listed)
            ranked = sorted(
                (r for r in pool if r["ranks"][method] is not None),
                key=lambda r, m=method: r["ranks"][m],
            )
            assert [r["passage_index"] for r in ranked] == [index for index, _ in listed]
            assert [r["ranks"][method] for r in ranked] == list(range(1, len(listed) + 1))
            # One passage per article, within a method.
            assert len({r["pmcid"] for r in ranked}) == len(ranked)
        assert all(any(r["ranks"][m] is not None for m in jr.METHODS) for r in pool)
        assert len({r["passage_index"] for r in pool}) == len(pool)


def test_the_methods_really_differ_on_the_synthetic_bundle(pooled):
    """Guards the fixture: if every row were found by all three methods, a leak test would pass
    on a pool with nothing to leak."""
    rows, _ = pooled
    assert any(r["ranks"]["bm25"] is None for r in rows)
    assert any(r["ranks"]["dense"] is None for r in rows)
    assert any(r["ranks"]["hybrid"] is None for r in rows)
    assert any(all(v is not None for v in r["ranks"].values()) for r in rows)


def test_a_passage_the_retriever_would_not_show_is_never_pooled(tmp_path, queries):
    """The tiny bundle holds a passage with treatment language and an article whose title names a
    drug: neither can reach the dashboard, so neither may reach a judge."""
    bundle = write_fake_bundle(tmp_path / "bundle")
    retriever = HybridRetriever(load_bundle(bundle), depth=jr.POOL_DEPTH)
    pericarditis = next(q for q in queries if q["condition_id"] == "COND:pericarditis")
    rows, _ = jr.build_pool(retriever, [pericarditis])
    texts = " ".join(r["passage"] for r in rows).lower()
    titles = " ".join(r["title"] for r in rows).lower()
    assert rows and "ibuprofen" not in texts and "colchicine" not in titles


# --------------------------------------------------------------------------- #
# The judges' sheets: blind, safe, protected
# --------------------------------------------------------------------------- #


def test_no_method_name_leaks_into_a_judge_sheet(bundle_dir, queries_file, tmp_path):
    out = tmp_path / "exp011"
    assert run_pool(bundle_dir, queries_file, out) == 0
    sheets = out / "sheets"
    assert sorted(p.name for p in sheets.iterdir()) == [
        "judge_A.csv",
        "judge_B.csv",
        "judge_instructions.md",
    ]
    a, b = (sheets / "judge_A.csv").read_bytes(), (sheets / "judge_B.csv").read_bytes()
    assert a == b  # the same pool in the same order for both, so nothing differs to leak
    for name in ("judge_A.csv", "judge_B.csv", "judge_instructions.md"):
        text = (sheets / name).read_text(encoding="utf-8-sig").lower()
        for word in METHOD_WORDS:
            assert word not in text, f"{word!r} in {name}"
    header = read_csv(sheets / "judge_A.csv")[0]
    assert tuple(header) == jr.SHEET_COLUMNS
    # The key, kept apart from the sheets, is what names the methods.
    assert "bm25" in (out / jr.KEY_NAME).read_text(encoding="utf-8")


def test_the_sheets_are_blank_and_cover_the_whole_pool(bundle_dir, queries_file, tmp_path, pooled):
    out = tmp_path / "exp011"
    run_pool(bundle_dir, queries_file, out, "--judges", "Asha", "Ben")
    key = json.loads((out / jr.KEY_NAME).read_text(encoding="utf-8"))
    assert key["judges"] == ["Asha", "Ben"] and key["seed"] == 42 and key["depth"] == 10
    rows = read_csv(out / "sheets" / "judge_Asha.csv")
    assert [r["row_id"] for r in rows] == [r["row_id"] for r in key["rows"]]
    assert [r["pmcid"] for r in rows] == [r["pmcid"] for r in key["rows"]]
    assert all(r["grade"] == r["treatment"] == r["note"] == "" for r in rows)
    assert len(rows) == len(pooled[0]) >= jr.MIN_PAIRS
    condition_row = next(r for r in rows if r["query_id"] == "Q-cond-01")
    assert condition_row["patient_findings"] == jr.NO_FINDINGS
    case_row = next(r for r in rows if r["query_id"].startswith("Q-case-GC-001"))
    assert "Chest pain" in case_row["patient_findings"]
    # The key holds no article text.
    assert rows[0]["passage"] not in (out / jr.KEY_NAME).read_text(encoding="utf-8")


def test_corpus_text_starting_like_a_formula_is_neutralised():
    assert jr._cell("=HYPERLINK(1)") == " =HYPERLINK(1)"
    assert jr._cell("+1 case") == " +1 case"
    assert jr._cell("-5 mmHg") == " -5 mmHg"
    assert jr._cell("@x") == " @x"
    assert jr._cell("A normal title") == "A normal title"
    assert jr._cell("") == "" and jr._cell(None) == ""


def test_nothing_is_written_inside_the_repository_outside_data(tmp_path):
    with pytest.raises(SystemExit, match="not under data"):
        jr.ensure_private(REPO_ROOT / "docs" / "exp011")
    with pytest.raises(SystemExit):
        jr.ensure_private(REPO_ROOT)
    assert jr.ensure_private(REPO_ROOT / "data" / "interim" / "exp011")
    assert jr.ensure_private(tmp_path / "anywhere")


def test_an_existing_pool_is_never_overwritten_without_force(bundle_dir, queries_file, tmp_path):
    out = tmp_path / "exp011"
    run_pool(bundle_dir, queries_file, out)
    sheet = out / "sheets" / "judge_A.csv"
    sheet.write_text("a judge's work", encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        run_pool(bundle_dir, queries_file, out)
    assert sheet.read_text(encoding="utf-8") == "a judge's work"
    assert run_pool(bundle_dir, queries_file, out, "--force") == 0
    assert sheet.read_text(encoding="utf-8-sig") != "a judge's work"


def test_pool_refuses_bad_judge_labels_and_a_missing_bundle(bundle_dir, queries_file, tmp_path):
    with pytest.raises(SystemExit, match="two different"):
        run_pool(bundle_dir, queries_file, tmp_path / "a", "--judges", "A", "A")
    with pytest.raises(SystemExit, match="letters"):
        run_pool(bundle_dir, queries_file, tmp_path / "b", "--judges", "A/..", "B")
    with pytest.raises(SystemExit, match="cannot be used"):
        run_pool(tmp_path / "nothing-here", queries_file, tmp_path / "c")


def test_a_pool_under_100_pairs_and_an_empty_query_are_reported(tmp_path, queries, capsys):
    bundle = write_fake_bundle(tmp_path / "bundle")
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"queries": jr.condition_queries()}), encoding="utf-8")
    assert run_pool(bundle, path, tmp_path / "exp011") == 0
    out = capsys.readouterr().out
    assert "under docs/05" in out and "no passage can be shown" in out


# --------------------------------------------------------------------------- #
# The scale, word for word
# --------------------------------------------------------------------------- #


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_the_scale_is_exp011s_word_for_word():
    log = squash((REPO_ROOT / "docs" / "08-experiment-log.md").read_text(encoding="utf-8"))
    docstring = squash(jr.__doc__)
    instructions = squash(jr.JUDGE_INSTRUCTIONS)
    for grade, wording in jr.SCALE.items():
        assert f"**{grade}**: {wording}" in log, grade
        assert f"{grade}: {wording}" in docstring, grade
        assert f"**{grade}**: {wording}" in instructions, grade


# --------------------------------------------------------------------------- #
# Cohen's kappa, by hand
# --------------------------------------------------------------------------- #

# Eight rows, judge A then judge B: (0,0) (0,0) (0,1) (1,1) (1,2) (2,2) (2,2) (2,0).
A = [0, 0, 0, 1, 1, 2, 2, 2]
B = [0, 0, 1, 1, 2, 2, 2, 0]


def test_kappa_matches_the_hand_computation():
    counts = jr.confusion(A, B)
    assert counts.tolist() == [[2, 1, 0], [0, 1, 1], [1, 0, 2]]
    # Row sums (3, 2, 3), column sums (3, 2, 3).
    # Unweighted: po = 5/8, pe = (9 + 4 + 9)/64 = 22/64, kappa = (5/8 − 22/64) / (1 − 22/64) = 3/7.
    assert float(jr.kappa_from_matrix(counts, "none")) == pytest.approx(3 / 7)
    # Linear, w = |i − j|/2: Σ w·O = (0.5 + 0.5 + 1)/8 = 1/4; Σ w·E = 30/64; kappa = 1 − 16/30 = 7/15.
    assert float(jr.kappa_from_matrix(counts, "linear")) == pytest.approx(7 / 15)
    # Quadratic, w = ((i − j)/2)²: Σ w·O = 1.5/8 = 3/16; Σ w·E = 24/64 = 3/8; kappa = 1 − 1/2.
    assert float(jr.kappa_from_matrix(counts, "quadratic")) == pytest.approx(0.5)


def test_kappa_edges():
    perfect = jr.confusion([0, 1, 2, 1], [0, 1, 2, 1])
    for weights in ("none", "linear", "quadratic"):
        assert float(jr.kappa_from_matrix(perfect, weights)) == pytest.approx(1.0)
    # One grade only: chance agreement is already total, so kappa is undefined, not 1.
    assert math.isnan(float(jr.kappa_from_matrix(jr.confusion([1, 1, 1], [1, 1, 1]), "none")))
    assert math.isnan(float(jr.kappa_from_matrix(np.zeros((3, 3)), "linear")))
    with pytest.raises(ValueError):
        jr.kappa_from_matrix(perfect, "cubic")


# --------------------------------------------------------------------------- #
# The metrics, by hand
# --------------------------------------------------------------------------- #


def tiny_key() -> tuple[dict, dict[str, int]]:
    """Two queries. q1's pool: r1..r5 with grades 2, 0, 1, 0, 2. q2's: r6..r8, all 0.

    BM25 shows r1 r2 r3 r5; dense shows r2 r4 r5 r1; the hybrid shows r3 r5 r1. Every method shows
    r6 r7 r8 for q2.
    """
    ranks = {
        "R001": (1, 4, 3),
        "R002": (2, 1, None),
        "R003": (3, None, 1),
        "R004": (None, 2, None),
        "R005": (4, 3, 2),
        "R006": (1, 1, 1),
        "R007": (2, 2, 2),
        "R008": (3, 3, 3),
    }
    rows = [
        {
            "row_id": rid,
            "query_id": "Q-1" if n <= 5 else "Q-2",
            "pmcid": f"PMC9{n:07d}",
            "passage_index": n,
            "ranks": dict(zip(jr.METHODS, r, strict=True)),
        }
        for n, (rid, r) in enumerate(ranks.items(), start=1)
    ]
    grades = dict(zip(ranks, [2, 0, 1, 0, 2, 0, 0, 0], strict=True))
    return {"queries": [{"id": "Q-1"}, {"id": "Q-2"}], "rows": rows, "judges": ["A", "B"]}, grades


def test_the_metrics_match_the_hand_computation():
    key, grades = tiny_key()
    scores = jr.score_methods(key, grades)
    m = scores["methods"]
    log2 = math.log2
    ideal10 = 2 / log2(2) + 2 / log2(3) + 1 / log2(4)  # pool grades 2, 2, 1, 0, 0
    ideal3 = ideal10

    # Precision@3 divides by 3 and averages the two queries; q2 has nothing relevant.
    assert m["bm25"]["p3"]["value"] == pytest.approx((2 / 3 + 0) / 2)  # 2, 0, 1
    assert m["dense"]["p3"]["value"] == pytest.approx((1 / 3 + 0) / 2)  # 0, 0, 2
    assert m["hybrid"]["p3"]["value"] == pytest.approx((3 / 3 + 0) / 2)  # 1, 2, 2
    assert m["bm25"]["p3_strict"]["value"] == pytest.approx((1 / 3 + 0) / 2)
    assert m["dense"]["p3_strict"]["value"] == pytest.approx((1 / 3 + 0) / 2)
    assert m["hybrid"]["p3_strict"]["value"] == pytest.approx((2 / 3 + 0) / 2)

    # nDCG: gain = grade, discount log2(rank + 1), over the pool's ideal; q2 has no relevant
    # passage, so only q1 counts, for every method.
    assert m["bm25"]["ndcg3"]["queries"] == 1
    assert m["bm25"]["ndcg3"]["value"] == pytest.approx((2 / log2(2) + 1 / log2(4)) / ideal3)
    assert m["bm25"]["ndcg10"]["value"] == pytest.approx(
        (2 / log2(2) + 0 + 1 / log2(4) + 2 / log2(5)) / ideal10
    )
    assert m["dense"]["ndcg3"]["value"] == pytest.approx((2 / log2(4)) / ideal3)
    assert m["dense"]["ndcg10"]["value"] == pytest.approx((2 / log2(4) + 2 / log2(5)) / ideal10)
    assert m["hybrid"]["ndcg10"]["value"] == pytest.approx(
        (1 / log2(2) + 2 / log2(3) + 2 / log2(4)) / ideal10
    )

    # Pooled recall: q1's relevant passages are r1, r3, r5.
    assert m["bm25"]["recall10"]["value"] == pytest.approx(3 / 3)
    assert m["dense"]["recall10"]["value"] == pytest.approx(2 / 3)
    assert m["hybrid"]["recall10"]["value"] == pytest.approx(3 / 3)

    assert scores["queries"] == 2 and scores["queries_without_relevant_pool"] == 1
    assert all(m[x]["queries_showing_under_3"] == 0 for x in jr.METHODS)
    assert all(m[x]["queries_showing_under_10"] == 2 for x in jr.METHODS)
    for method in jr.METHODS:
        for name, _, _ in jr.METRICS:
            e = m[method][name]
            assert e["ci_low"] <= e["value"] <= e["ci_high"]

    difference = (
        (2 / log2(2) + 0 + 1 / log2(4) + 2 / log2(5)) - (1 / log2(2) + 2 / log2(3) + 2 / log2(4))
    ) / ideal10
    bm25 = scores["comparison_alone_minus_hybrid_ndcg10"]["bm25"]
    assert bm25["difference"] == pytest.approx(difference) and bm25["cases"] == 1
    assert bm25["better"] == "a"


def test_a_method_that_shows_fewer_than_three_is_counted_and_penalised():
    key, grades = tiny_key()
    short = {**key, "rows": [r for r in key["rows"] if r["query_id"] == "Q-1"]}
    short["queries"] = [{"id": "Q-1"}]
    # The hybrid shows r3, r5, r1; take its last one away: two passages shown.
    short["rows"] = [
        {
            **r,
            "ranks": {
                **r["ranks"],
                "hybrid": None if r["row_id"] == "R001" else r["ranks"]["hybrid"],
            },
        }
        for r in short["rows"]
    ]
    scores = jr.score_methods(short, grades)
    assert scores["methods"]["hybrid"]["queries_showing_under_3"] == 1
    assert scores["methods"]["hybrid"]["p3"]["value"] == pytest.approx(2 / 3)  # r3, r5 over 3


def test_the_rule_as_pre_registered():
    def scores(better: str | None, p3: float) -> dict:
        comparison = {
            m: {"better": "a" if m == better else None, "difference": 0.1}
            for m in ("bm25", "dense")
        }
        methods = {m: {"p3": {"value": 0.8}} for m in jr.METHODS}
        methods["hybrid"]["p3"]["value"] = p3
        methods["dense"]["p3"]["value"] = 0.3
        return {"comparison_alone_minus_hybrid_ndcg10": comparison, "methods": methods}

    stays = jr.apply_rule(scores(None, 0.6))
    assert stays["configured"] == "hybrid" and not stays["evidence_decision_for_the_team"]
    swapped = jr.apply_rule(scores("dense", 0.9))
    assert swapped["configured"] == "dense" and "replaces the hybrid" in swapped["verdict"]
    assert swapped["evidence_decision_for_the_team"]  # dense's Precision@3 is 0.3: under 0.5
    assert jr.apply_rule(scores(None, 0.49))["evidence_decision_for_the_team"]
    assert not jr.apply_rule(scores(None, 0.5))["evidence_decision_for_the_team"]


# --------------------------------------------------------------------------- #
# A whole round: sheets filled by two simulated judges
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def round_dir(tmp_path_factory, bundle_dir, queries) -> Path:
    """A pool written, and both judges' sheets filled: A grades by row number, B differs from A on
    every seventh row, in reverse order, saved the way a European Excel would."""
    out = tmp_path_factory.mktemp("round") / "exp011"
    path = out.parent / "queries.json"
    path.write_text(json.dumps({"queries": queries, "degraded": []}), encoding="utf-8")
    assert run_pool(bundle_dir, path, out) == 0
    sheets = out / "sheets"
    fill(
        sheets / "judge_A.csv",
        out / "a_filled.csv",
        lambda i, r: i % 3,
        treatment={"R002", "R005"},
    )
    fill(
        sheets / "judge_B.csv",
        out / "b_filled.csv",
        lambda i, r: (i % 3 + 1) % 3 if i % 7 == 0 else i % 3,
        treatment={"R005", "R009"},
        reverse=True,
        delimiter=";",
        encoding="cp1252",
    )
    return out


def score_args(round_dir: Path, *extra: str) -> list[str]:
    return [
        "score",
        "--key",
        str(round_dir / jr.KEY_NAME),
        "--sheet-a",
        str(round_dir / "a_filled.csv"),
        "--sheet-b",
        str(round_dir / "b_filled.csv"),
        "--judgements-out",
        str(round_dir / "judgements.csv"),
        *extra,
    ]


def test_a_round_gives_kappa_the_disagreements_and_provisional_figures(round_dir, capsys):
    key = json.loads((round_dir / jr.KEY_NAME).read_text(encoding="utf-8"))
    n = len(key["rows"])
    wrong = [i for i in range(n) if i % 7 == 0]
    assert jr.main(score_args(round_dir)) == 0
    out = capsys.readouterr().out
    assert "PROVISIONAL" in out and "kappa" in out
    assert "The pre-registered rule" not in out and "labels: CONSENSUS" not in out

    scores = json.loads((round_dir / jr.SCORES_NAME).read_text(encoding="utf-8"))
    assert scores["labels"] == "provisional" and "rule" not in scores
    assert scores["disagreements"] == [key["rows"][i]["row_id"] for i in wrong]
    assert scores["agreement"]["observed_agreement"] == pytest.approx(1 - len(wrong) / n)
    a = [i % 3 for i in range(n)]
    b = [(g + 1) % 3 if i % 7 == 0 else g for i, g in enumerate(a)]
    assert scores["agreement"]["kappa_none"]["value"] == pytest.approx(
        float(jr.kappa_from_matrix(jr.confusion(a, b), "none"))
    )
    assert {f["row_id"]: f["flagged_by"] for f in scores["treatment_flags"]} == {
        "R002": ["A"],
        "R005": ["A", "B"],
        "R009": ["B"],
    }
    assert set(scores["scores_by_judge"]) == {"A", "B"}

    # What the judges settle: row ids, both grades, a blank `consensus`, and the text to discuss it.
    rows = read_csv(round_dir / jr.DISAGREEMENTS_NAME)
    assert [r["row_id"] for r in rows] == scores["disagreements"]
    assert all(r["consensus"] == "" and r["excerpt"] for r in rows)
    assert {"grade_A", "grade_B"} <= set(rows[0])

    # exp011_judgements.csv: PMCIDs and labels only, and no label where they disagree yet.
    text = (round_dir / "judgements.csv").read_text(encoding="utf-8-sig")
    judged = read_csv(round_dir / "judgements.csv")
    assert list(judged[0]) == [
        "row_id",
        "query_id",
        "pmcid",
        "grade_a",
        "grade_b",
        "final",
        "treatment_a",
        "treatment_b",
    ]
    assert [r["final"] == "" for r in judged] == [i % 7 == 0 for i in range(n)]
    sheet = read_csv(round_dir / "sheets" / "judge_A.csv")
    assert all(r["passage"] not in text and r["title"] not in text for r in sheet[:50])
    assert not any(w in text.lower() for w in METHOD_WORDS)


def test_a_consensus_gives_the_final_figures_and_the_rule(round_dir, capsys):
    assert jr.main(score_args(round_dir)) == 0  # the provisional run writes the file to settle
    rows = read_csv(round_dir / jr.DISAGREEMENTS_NAME)
    consensus = round_dir / "consensus.csv"
    # Everything settled at judge A's grade.
    consensus.write_text(
        "row_id,consensus\n" + "".join(f"{r['row_id']},{r['grade_A']}\n" for r in rows),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert jr.main(score_args(round_dir, "--consensus", str(consensus))) == 0
    out = capsys.readouterr().out
    assert "labels: CONSENSUS" in out and "The pre-registered rule" in out

    scores = json.loads((round_dir / jr.SCORES_NAME).read_text(encoding="utf-8"))
    key = json.loads((round_dir / jr.KEY_NAME).read_text(encoding="utf-8"))
    by_a = {r["row_id"]: i % 3 for i, r in enumerate(key["rows"])}
    expected = jr.score_methods(key, by_a)
    assert scores["scores"]["methods"] == json.loads(json.dumps(expected["methods"]))
    assert scores["rule"]["configured"] in jr.METHODS
    # Every row now has a final label.
    assert all(r["final"] != "" for r in read_csv(round_dir / "judgements.csv"))


def test_a_judge_may_sort_a_sheet_and_the_scores_do_not_change(round_dir):
    key = json.loads((round_dir / jr.KEY_NAME).read_text(encoding="utf-8"))
    ordered = round_dir / "b_in_pool_order.csv"
    fill(
        round_dir / "sheets" / "judge_B.csv",
        ordered,
        lambda i, r: (i % 3 + 1) % 3 if i % 7 == 0 else i % 3,
        treatment={"R005", "R009"},
    )
    from_reversed, problems_r = jr.read_judgements(round_dir / "b_filled.csv", key)
    from_ordered, problems_o = jr.read_judgements(ordered, key)
    assert not problems_r and not problems_o
    assert {k: v["grade"] for k, v in from_reversed.items()} == {
        k: v["grade"] for k, v in from_ordered.items()
    }


def test_every_problem_in_a_sheet_is_reported_and_nothing_is_scored(round_dir, tmp_path, capsys):
    key = json.loads((round_dir / jr.KEY_NAME).read_text(encoding="utf-8"))
    rows = read_csv(round_dir / "a_filled.csv")
    rows[0]["grade"] = ""  # blank
    rows[1]["grade"] = "3"  # not a grade
    rows[2]["grade"] = "high"  # not a number
    rows[3]["treatment"] = "maybe"  # not a flag
    rows[4]["pmcid"] = "PMC1"  # a changed identity
    rows.append({**rows[5]})  # a row twice
    rows.append({**rows[6], "row_id": "R9999"})  # a row the pool does not have
    dropped = rows.pop(10)["row_id"]  # a row missing
    bad = tmp_path / "bad.csv"
    with open(bad, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _, problems = jr.read_judgements(bad, key)
    text = "\n".join(problems)
    for expected in (
        f"{rows[0]['row_id']} has no grade",
        "is not 0, 1 or 2",
        "is not a grade",
        "is not a treatment flag",
        "different pmcid",
        "appears twice",
        "'R9999' is not in the pool",
        f"row {dropped} is missing",
    ):
        assert expected in text, expected

    args = score_args(round_dir)
    args[args.index("--sheet-a") + 1] = str(bad)
    before = (round_dir / jr.SCORES_NAME).read_bytes()
    assert jr.main(args) == 1
    assert "cannot score" in capsys.readouterr().err
    assert (round_dir / jr.SCORES_NAME).read_bytes() == before  # nothing was rewritten


def test_a_consensus_can_only_settle_disagreements(round_dir, tmp_path):
    key = json.loads((round_dir / jr.KEY_NAME).read_text(encoding="utf-8"))
    rows = key["rows"]
    disagreement, agreed = rows[0]["row_id"], rows[1]["row_id"]  # i = 0 differs; i = 1 agrees
    sheet_a, sheet_b = round_dir / "a_filled.csv", round_dir / "b_filled.csv"

    def run(content: str):
        path = tmp_path / "consensus.csv"
        path.write_text(content, encoding="utf-8")
        return jr.score_round(key, sheet_a, sheet_b, path)

    everything = "".join(
        f"{r['row_id']},{r['grade_A']}\n" for r in read_csv(round_dir / jr.DISAGREEMENTS_NAME)
    )
    with pytest.raises(ValueError, match=f"no grade for the disagreement at row {disagreement}"):
        run("row_id,consensus\n" + "".join(everything.splitlines(True)[1:]))
    with pytest.raises(ValueError, match="not in the pool"):
        run("row_id,consensus\nR9999,1\n" + everything)
    with pytest.raises(ValueError, match="not a disagreement to settle"):
        run(f"row_id,consensus\n{agreed},{(1 % 3 + 1) % 3}\n" + everything)  # agreed at 1
    with pytest.raises(ValueError, match="not a grade"):
        run("row_id,consensus\nR001,good\n" + everything)
    with pytest.raises(ValueError, match="needs `row_id`"):
        run("something,else\n1,2\n")
    # Repeating an agreed grade is harmless, so a full labels file works too.
    assert run(f"row_id,consensus\n{agreed},1\n" + everything)["labels"] == "consensus"


def test_excels_encoding_and_delimiters_are_read(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes("row_id;grade;note\r\nR001;2;café\r\nR002;1;\r\n".encode("cp1252"))
    rows = jr.read_table(path)
    assert [(r["row_id"], r["grade"]) for r in rows] == [("R001", "2"), ("R002", "1")]
    assert rows[0]["note"] == "café"
    path.write_text("Row_ID\tGrade\nR001\t0.0\n", encoding="utf-8-sig")
    assert jr.parse_grade(jr.read_table(path)[0]["grade"]) == 0
    assert jr.parse_grade("") is None and jr.parse_grade(" 2 ") == 2
    for bad in ("3", "-1", "1.5", "x"):
        with pytest.raises(ValueError):
            jr.parse_grade(bad)
    assert jr.parse_flag("") is False and jr.parse_flag("X") and jr.parse_flag("1")
    assert jr.parse_flag("no") is False


def test_score_refuses_a_file_that_is_not_a_pool_key(tmp_path, round_dir):
    notkey = tmp_path / "k.json"
    notkey.write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a pool key"):
        jr.main(
            [
                "score",
                "--key",
                str(notkey),
                "--sheet-a",
                str(round_dir / "a_filled.csv"),
                "--sheet-b",
                str(round_dir / "b_filled.csv"),
                "--judgements-out",
                str(tmp_path / "j.csv"),
            ]
        )


def test_kappa_interval_resamples_queries_and_is_reproducible(round_dir):
    key = json.loads((round_dir / jr.KEY_NAME).read_text(encoding="utf-8"))
    a, _ = jr.read_judgements(round_dir / "a_filled.csv", key)
    b, _ = jr.read_judgements(round_dir / "b_filled.csv", key)
    ga, gb = ({k: v["grade"] for k, v in s.items()} for s in (a, b))
    first, second = jr.agreement(key, ga, gb), jr.agreement(key, ga, gb)
    assert first == second
    k = first["kappa_linear"]
    assert k["ci_low"] <= k["value"] <= k["ci_high"]


def test_the_disagreements_file_is_filled_in_place_and_never_overwritten(
    round_dir, tmp_path, capsys
):
    import shutil

    work = tmp_path / "copy"
    shutil.copytree(round_dir, work)
    (work / jr.DISAGREEMENTS_NAME).unlink(missing_ok=True)
    assert jr.main(score_args(work)) == 0
    path = work / jr.DISAGREEMENTS_NAME
    rows = read_csv(path)
    assert rows
    for row in rows:
        row["consensus"] = row["grade_B"]  # the judges' discussion, settled at B's grade
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = path.read_bytes()
    capsys.readouterr()
    # Passing the very file they filled in is the intended flow, and it must survive the run.
    assert jr.main(score_args(work, "--consensus", str(path))) == 0
    assert path.read_bytes() == before
    assert "kept" in capsys.readouterr().out
    scores = json.loads((work / jr.SCORES_NAME).read_text(encoding="utf-8"))
    assert scores["labels"] == "consensus"
    final = {r["row_id"]: r["final"] for r in read_csv(work / "judgements.csv")}
    assert all(final[r["row_id"]] == r["grade_B"] for r in rows)


def test_expanding_a_case_through_the_crosswalk_does_not_change_its_queries(queries):
    """With the real graph the demo expands each case first (``DDX:E_nn`` questions are added to
    its findings). The retriever ignores those, so the queries are the hand-written findings
    either way; this is the part of the real-pipeline path that can run without data/."""
    expanded, _ = jr.case_queries(jr.load_golden(), stub_pipeline(), labels={})
    assert expanded == [q for q in queries if q["kind"] == "case"]
