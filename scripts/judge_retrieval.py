"""EXP-011's blind relevance judging: build the pool, then score the judgements (task 3a).

    python scripts/judge_retrieval.py pool  --bundle data/raw/retrieval_corpus
    python scripts/judge_retrieval.py score --sheet-a data/interim/exp011/sheets/judge_A_filled.csv \\
                                            --sheet-b data/interim/exp011/sheets/judge_B_filled.csv
    python scripts/judge_retrieval.py score ... --consensus data/interim/exp011/exp011_disagreements.csv

This implements the design pre-registered in ``docs/08`` (EXP-011, 2026-10-14) and ``docs/05``
§3.4. **It retrieves, pools, shuffles and scores; it never judges** (Claude does not judge). It
reads the real bundle only through the retriever the prototype uses, fetches nothing, loads no
model and changes no retrieval behaviour.

**Who runs what.** The owner (or Claude) runs ``pool`` once, on the laptop, where the bundle, the
graph and the ranker are. It writes, under ``data/interim/exp011/`` (gitignored: the sheets hold
corpus text, which is never committed):

* ``sheets/judge_A.csv`` and ``sheets/judge_B.csv``: the same pool, in the same order, one blank
  sheet per judge, with **no method named anywhere**, and ``sheets/judge_instructions.md``. Send
  each judge their own sheet and the instructions, and nothing else.
* ``exp011_pool_key.json``: **private**. Which methods retrieved each row, and at which rank. Keep it
  away from both judges until they have finished.
* ``exp011_queries.json``: the 26 queries as run, so a later ``pool --queries`` reproduces them
  without the graph or the model.

**The pool** (EXP-011, "Pooling and judging"). For each query, the top 10 of BM25 alone, dense search
alone and the hybrid, one passage per article, exactly as the dashboard would show them (a passage
whose excerpt or article title fails the treatment screen is never shown, so never pooled): at most
30 distinct passages, shuffled with seed 42, the method hidden. The shuffle starts from the passages
in index order, so the order cannot depend on how any method ranked them. **Every pooled passage is
judged by both judges**, which covers ``docs/05`` §3.4's 100 sampled pairs several times over (the
script says if a pool is under 100).

**The 26 queries.**

* ``Q-cond-01``…``14``: each condition alone, ``rank(condition, [])``.
* ``Q-case-GC-001-1``…: each golden case × the three candidates the configured pipeline
  (``configs/config.yaml``, red flags on) ranks first for it, with the case's present findings. Any
  golden case added before judging begins adds three queries. ``pool`` refuses a pipeline that is
  degraded in the model or the graph, because the pre-registered queries are the configured
  pipeline's.

**The judges' scale** (EXP-011, verbatim), for one passage against one query:

    2: it says something about diagnosing this condition that bears on this patient's findings,
       such as a finding's value, the presentation, or an investigation's diagnostic role
    1: it is about diagnosing this condition, but not these findings
    0: it is not about diagnosing this condition: another condition, epidemiology alone, a study's
       logistics, or treatment

Also mark ``treatment`` = 1 on any row whose passage or excerpt contains treatment language (a drug,
a dose, a procedure, advice on management), whatever grade it gets: treatment that reached a judge
is counted on its own as a failure of the screen (the target is none). The two judges work
independently and blind; disagreements are settled afterwards by discussion.

**Scoring.** ``score`` joins the judges' two filled sheets to the key and reports, for BM25, dense
search and the hybrid, each with a 95% bootstrap interval over queries (1,000 resamples, seed 42,
``src/eval/metrics.py``): Precision@3 (judged at least 1, and strictly 2), nDCG@3, nDCG@10 and
pooled Recall@10 (passages judged at least 1, relative to the pool, not to the corpus); the weighted
Cohen's kappa **before discussion**; the rows the judges disagree on, in
``exp011_disagreements.csv`` with a blank ``consensus`` column to fill; the rows flagged as
treatment; and ``data/interim/exp011_judgements.csv`` (PMCIDs and labels only, no article text).
Without ``--consensus`` the figures are provisional, one set for each judge's own labels, and the
rule is not applied. The judges then fill the ``consensus`` column of ``exp011_disagreements.csv``
(``score`` never overwrites that file once it exists) and pass it back as ``--consensus``; any CSV
with ``row_id`` and ``consensus`` columns covering every disagreement works, and a consensus grade
for a row the judges agreed on in another grade is refused, since only disagreements are settled
by discussion. With it, the final figures, the hybrid's paired comparison with each method alone
on nDCG@10 (``paired_bootstrap_difference``) and the pre-registered rule are reported.

Definitions the pre-registration leaves open, fixed here and printed in every report: Precision@k
divides by k (a method that shows fewer is penalised; the report counts those queries); nDCG's gain
is the grade itself, its ideal ordering the query's judged pool; a query whose pool holds no
passage judged at least 1 has no nDCG or recall, and is left out of those figures for every method
alike; every per-query figure is averaged over queries. These and the kappa's weights are listed in
the pull request for the team to confirm before scoring.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditions import BY_ID  # noqa: E402
from src.contracts import PatientCase  # noqa: E402
from src.eval.metrics import Ratio, bootstrap_ratio, paired_bootstrap_difference  # noqa: E402
from src.rag.bundle import load_bundle, sha256_of  # noqa: E402
from src.rag.retriever import DEFAULT_BUNDLE, DEPTH, HybridRetriever, present_concepts  # noqa: E402
from src.rag.screening import SCREEN_DIGEST  # noqa: E402
from src.rag.vocabulary import CONDITION_TERMS, compose_query_text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT_DIR = DATA_DIR / "interim" / "exp011"
DEFAULT_JUDGEMENTS = DATA_DIR / "interim" / "exp011_judgements.csv"
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"

SEED = 42
RESAMPLES = 1_000
POOL_DEPTH = DEPTH
"""How far down each method's list enters the pool: 10 (EXP-011)."""
CASE_CANDIDATES = 3
"""Candidates per golden case that become queries: the three ranked first (EXP-011)."""
MIN_PAIRS = 100
"""docs/05 §3.4: relevance judgements on 100 sampled candidate-evidence pairs."""
GRADES = (0, 1, 2)
METHODS: Mapping[str, tuple[str, ...]] = {
    "bm25": ("bm25",),
    "dense": ("dense",),
    "hybrid": ("bm25", "dense"),
}
"""The three methods pooled, and the ``rank(..., methods=)`` each runs."""
METHOD_NAMES = {"bm25": "BM25 alone", "dense": "dense alone", "hybrid": "hybrid (BM25 + dense)"}
KEY_FORMAT = "nightingale-exp011-pool-key"
KEY_VERSION = 1
KEY_NAME = "exp011_pool_key.json"
QUERIES_NAME = "exp011_queries.json"
DISAGREEMENTS_NAME = "exp011_disagreements.csv"
SCORES_NAME = "exp011_scores.json"
INSTRUCTIONS_NAME = "judge_instructions.md"
IGNORABLE_DEGRADATION = {"rag", "graph_backend", "calibration"}
"""Components whose absence does not change which candidates a pipeline ranks first: retrieval is
attached after ranking, a fallback graph store answers identically (docs/02 §7), and a calibrator
never reorders (docs/05, EXP-007)."""

SHEET_COLUMNS = (
    "row_id",
    "query_id",
    "condition",
    "patient_findings",
    "pmcid",
    "title",
    "section",
    "excerpt",
    "passage",
    "grade",
    "treatment",
    "note",
)
NO_FINDINGS = "(none: the query is the condition alone)"

SCALE = {
    2: (
        "it says something about diagnosing this condition that bears on this patient's "
        "findings, such as a finding's value, the presentation, or an investigation's "
        "diagnostic role"
    ),
    1: "it is about diagnosing this condition, but not these findings",
    0: (
        "it is not about diagnosing this condition: another condition, epidemiology alone, "
        "a study's logistics, or treatment"
    ),
}
"""EXP-011's scale, word for word (docs/08); a test compares it with the log."""

JUDGE_INSTRUCTIONS = f"""\
# Judging evidence for Nightingale (EXP-011)

You are one of two team members who judge, **independently and blind**, whether passages from
open-access medical articles bear on the diagnosis of a condition. Do not talk to the other judge
about any row until you have both finished. This sheet does not say how any passage was found, and
you cannot tell. This is a research task: it is not a clinical decision, and nothing here is advice.

## What is in a row

Each row is one passage judged against one query. The query is the **condition** and, for some
rows, the **patient's findings** (a hand-written case, never a real patient). Judge the **passage**
(the `passage` column) against that query. The `excerpt` column is the part of it a clinician
would be shown beside the condition; use it to find the sentence that matters.

## The scale (fill the `grade` column with 0, 1 or 2)

- **2**: {SCALE[2]};
- **1**: {SCALE[1]};
- **0**: {SCALE[0]}.

Judge diagnosis only. A passage about how to treat the condition is a 0 whatever else it says.

## The `treatment` column

Put **1** in `treatment` on every row whose passage or excerpt contains treatment language (a drug,
a dose, a procedure, advice on management), **whatever grade you give it**. Leave it blank
otherwise. Treatment is never supposed to reach you, so each one you find is a failure of the
screen, and the team needs to know. Use `note` for anything else worth telling the team (optional).

## How to fill the sheet

1. Fill `grade` on **every** row. A blank or other value stops the scoring.
2. Do not change any other column (`row_id` is what the scoring joins on). Sorting or filtering is
   fine. Do not delete rows.
3. Save as **CSV UTF-8** under the name you were given plus `_filled`, and send it only to the
   person who gave you the sheet. Do not share your grades before both of you are done.
4. Judge what the text says, not what you already know about the article.
"""

Query = dict[str, Any]


# --------------------------------------------------------------------------- #
# The queries
# --------------------------------------------------------------------------- #


def condition_queries() -> list[Query]:
    """Q-cond: one query per condition, the condition with no findings (EXP-011)."""
    return [
        {
            "id": f"Q-cond-{i:02d}",
            "kind": "condition",
            "case_id": None,
            "candidate_rank": None,
            "condition_id": cid,
            "condition": BY_ID[cid].label,
            "concepts": [],
            "findings": [],
        }
        for i, cid in enumerate(CONDITION_TERMS, start=1)
    ]


def case_queries(
    golden: Sequence[Mapping[str, Any]],
    pipeline: Any,
    labels: Mapping[str, str] | None = None,
    top: int = CASE_CANDIDATES,
) -> tuple[list[Query], list[str]]:
    """Q-case: each golden case × the ``top`` candidates the pipeline ranks first for it, with the
    case's present findings (EXP-011).

    Args:
        golden: The golden cases as ``tests/fixtures/golden_cases.yaml`` holds them.
        pipeline: A ``DiagnosisPipeline`` with red flags on, as configured.
        labels: The real graph's node labels; when given, each case is expanded through the
            crosswalk first, as the demo and the dashboard do.
        top: How many candidates per case.

    Returns:
        The queries, and the degraded components any run reported.
    """
    from src.medical_kg.crosswalk import expand_case

    queries: list[Query] = []
    degraded: set[str] = set()
    for entry in golden:
        case = PatientCase(**entry["case"])
        if labels is not None:
            case = expand_case(case, labels)
        result = pipeline.run(case)
        degraded.update(result.degraded_components)
        concepts = present_concepts(case)
        names = {f.concept_id: f.label for f in [*case.findings, *case.risk_factors]}
        for rank, candidate in enumerate(result.candidates[:top], start=1):
            queries.append(
                {
                    "id": f"Q-case-{entry['id']}-{rank}",
                    "kind": "case",
                    "case_id": entry["id"],
                    "candidate_rank": rank,
                    "condition_id": candidate.condition_id,
                    "condition": candidate.label,
                    "concepts": list(concepts),
                    "findings": [names.get(c, c) for c in concepts],
                }
            )
    return queries, sorted(degraded)


def load_golden(path: Path = GOLDEN) -> list[dict[str, Any]]:
    import yaml

    return list(yaml.safe_load(path.read_text(encoding="utf-8"))["cases"])


def check_queries(queries: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a query list the retriever cannot run (a hand-edited ``--queries`` file)."""
    ids = [q.get("id") for q in queries]
    if not queries or len(set(ids)) != len(ids):
        raise ValueError("the queries need distinct ids")
    for q in queries:
        for field in ("id", "condition_id", "condition", "concepts", "findings"):
            if field not in q:
                raise ValueError(f"query {q.get('id')!r} has no {field!r}")
        if q["condition_id"] not in CONDITION_TERMS:
            raise ValueError(f"query {q['id']!r}: {q['condition_id']!r} is not a known condition")


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #


def run_methods(
    retriever: HybridRetriever, query: Query, depth: int = POOL_DEPTH
) -> dict[str, list[tuple[int, Any]]]:
    """``{method: [(passage index, Evidence)]}``, best first, at most ``depth`` each.

    Each list is what ``retrieve`` would show for that method: one passage per article, in rank
    order, skipping a passage whose excerpt or title fails the treatment screen (the retriever's
    own last check, ``_evidence``). The three methods share one excerpt per passage, since the
    excerpt depends on the query's words only.
    """
    cid, concepts = query["condition_id"], list(query["concepts"])
    terms = retriever.bm25.query_terms(compose_query_text(cid, concepts))
    found: dict[str, list[tuple[int, Any]]] = {}
    for name, methods in METHODS.items():
        shown: list[tuple[int, Any]] = []
        for index, score in retriever.rank(cid, concepts, methods=methods):
            evidence = retriever._evidence(index, score, terms)
            if evidence is not None:
                shown.append((index, evidence))
            if len(shown) == depth:
                break
        found[name] = shown
    return found


def build_pool(
    retriever: HybridRetriever,
    queries: Sequence[Query],
    *,
    depth: int = POOL_DEPTH,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """The pooled rows, in sheet order, and how many passages each method showed per query.

    Per query: the distinct passages any method showed, sorted by passage index (an order no
    method decided), then shuffled with ``random.Random(seed)``, one generator for the whole pool,
    consumed query by query in order. Rows are numbered ``R001``… after the shuffle.
    """
    rng = random.Random(seed)
    bundle = retriever.bundle
    rows: list[dict[str, Any]] = []
    returned: dict[str, dict[str, int]] = {}
    for query in queries:
        found = run_methods(retriever, query, depth)
        returned[query["id"]] = {name: len(shown) for name, shown in found.items()}
        ranks: dict[int, dict[str, int | None]] = {}
        evidence_of: dict[int, Any] = {}
        for name, shown in found.items():
            for rank, (index, evidence) in enumerate(shown, start=1):
                ranks.setdefault(index, dict.fromkeys(METHODS))[name] = rank
                evidence_of[index] = evidence
        order = sorted(ranks)
        rng.shuffle(order)
        for index in order:
            passage = bundle.passages[index]
            rows.append(
                {
                    "query_id": query["id"],
                    "passage_index": index,
                    "pmcid": passage["pmcid"],
                    "title": evidence_of[index].title,
                    "section": passage.get("section", ""),
                    "excerpt": evidence_of[index].passage,
                    "passage": passage["text"],
                    "ranks": ranks[index],
                }
            )
    for number, row in enumerate(rows, start=1):
        row["row_id"] = f"R{number:03d}"
    return rows, returned


def _cell(value: Any) -> str:
    """A text cell safe to open in a spreadsheet: corpus text is untrusted, and a cell starting
    with = + - @ would be run as a formula (CSV injection), so it gets a leading space."""
    text = "" if value is None else str(value)
    return " " + text if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text


def sheet_rows(rows: Sequence[Mapping[str, Any]], queries: Sequence[Query]) -> list[dict[str, str]]:
    """The judges' rows: the query's description and the passage, ``grade`` / ``treatment`` /
    ``note`` blank. **No method, rank or score is in any column.**"""
    by_id = {q["id"]: q for q in queries}
    out = []
    for row in rows:
        query = by_id[row["query_id"]]
        out.append(
            {
                "row_id": row["row_id"],
                "query_id": query["id"],
                "condition": _cell(query["condition"]),
                "patient_findings": _cell("; ".join(query["findings"]) or NO_FINDINGS),
                "pmcid": row["pmcid"],
                "title": _cell(row["title"]),
                "section": _cell(row["section"]),
                "excerpt": _cell(row["excerpt"]),
                "passage": _cell(row["passage"]),
                "grade": "",
                "treatment": "",
                "note": "",
            }
        )
    return out


def write_sheet(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """A UTF-8 CSV with a byte-order mark, which Excel needs to read the accents."""
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def make_key(
    rows: Sequence[Mapping[str, Any]],
    queries: Sequence[Query],
    returned: Mapping[str, Mapping[str, int]],
    *,
    judges: Sequence[str],
    bundle_info: Mapping[str, Any],
    seed: int = SEED,
    depth: int = POOL_DEPTH,
    degraded: Sequence[str] = (),
) -> dict[str, Any]:
    """The private key: which methods retrieved each row, at which rank (1-based, None if the
    method did not show it). Holds no corpus text."""
    return {
        "format": KEY_FORMAT,
        "version": KEY_VERSION,
        "seed": seed,
        "depth": depth,
        "judges": list(judges),
        "bundle": dict(bundle_info),
        "queries_degraded": list(degraded),
        "queries": [dict(q) for q in queries],
        "returned": {qid: dict(counts) for qid, counts in returned.items()},
        "rows": [
            {
                "row_id": row["row_id"],
                "query_id": row["query_id"],
                "pmcid": row["pmcid"],
                "passage_index": row["passage_index"],
                "ranks": dict(row["ranks"]),
            }
            for row in rows
        ],
    }


def pool_summary(rows: Sequence[Mapping[str, Any]], queries: Sequence[Query]) -> dict[str, Any]:
    """Counts only, nothing that says which method found what."""
    per_query = [sum(1 for r in rows if r["query_id"] == q["id"]) for q in queries]
    return {
        "queries": len(queries),
        "rows": len(rows),
        "per_query_min": min(per_query, default=0),
        "per_query_median": float(np.median(per_query)) if per_query else 0.0,
        "per_query_max": max(per_query, default=0),
        "empty_queries": [q["id"] for q, n in zip(queries, per_query, strict=True) if n == 0],
    }


# --------------------------------------------------------------------------- #
# Reading the judges' sheets
# --------------------------------------------------------------------------- #

_TRUE = {"1", "x", "y", "yes", "true"}
_FALSE = {"0", "n", "no", "false"}


def _decode(raw: bytes) -> str:
    """UTF-8 (with or without a mark), else Windows-1252: what Excel's plain "CSV" writes."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_table(path: Path) -> list[dict[str, str]]:
    """A CSV as dicts keyed by lower-cased, stripped header names. Reads a comma, semicolon or tab
    separated file (spreadsheets in some locales write semicolons)."""
    text = _decode(Path(path).read_bytes())
    header = text.split("\n", 1)[0]
    delimiter = max(",;\t", key=header.count)
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    out = []
    for raw in reader:
        row = {str(k).strip().lower(): (v or "").strip() for k, v in raw.items() if k is not None}
        if any(row.values()):
            out.append(row)
    return out


def parse_grade(value: str) -> int | None:
    """0, 1 or 2 from a cell (``1.0`` is accepted); None for blank; ValueError for anything else."""
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{value!r} is not a grade") from None
    if number not in (0.0, 1.0, 2.0):
        raise ValueError(f"{value!r} is not 0, 1 or 2")
    return int(number)


def parse_flag(value: str) -> bool:
    text = value.strip().lower()
    if text in ("", *_FALSE):
        return False
    if text in _TRUE:
        return True
    raise ValueError(f"{value!r} is not a treatment flag (use 1 or leave it blank)")


def read_judgements(path: Path, key: Mapping[str, Any]) -> tuple[dict[str, dict], list[str]]:
    """One judge's filled sheet: ``({row_id: {"grade", "treatment", "note", "title", "excerpt"}},
    problems)``. Joined by ``row_id``, so the judge may have sorted or filtered the rows. Every
    problem is collected, not just the first: a missing or invalid grade, a row twice, a row the
    key does not know, a row of the key that is missing, a ``query_id`` or ``pmcid`` that does not
    match the key's (an edited sheet)."""
    expected = {r["row_id"]: r for r in key["rows"]}
    table = read_table(path)
    problems: list[str] = []
    got: dict[str, dict] = {}
    if table and ("row_id" not in table[0] or "grade" not in table[0]):
        return {}, [f"{path}: needs `row_id` and `grade` columns"]
    for row in table:
        row_id = row.get("row_id", "")
        if row_id not in expected:
            problems.append(f"{path.name}: row {row_id!r} is not in the pool")
            continue
        if row_id in got:
            problems.append(f"{path.name}: row {row_id} appears twice")
            continue
        for column, truth in (("query_id", "query_id"), ("pmcid", "pmcid")):
            if row.get(column) and row[column] != expected[row_id][truth]:
                problems.append(f"{path.name}: row {row_id} has a different {column} than the pool")
        try:
            grade = parse_grade(row.get("grade", ""))
        except ValueError as error:
            problems.append(f"{path.name}: row {row_id}: {error}")
            grade = None
        else:
            if grade is None:
                problems.append(f"{path.name}: row {row_id} has no grade")
        try:
            flag = parse_flag(row.get("treatment", ""))
        except ValueError as error:
            problems.append(f"{path.name}: row {row_id}: {error}")
            flag = False
        got[row_id] = {
            "grade": grade,
            "treatment": flag,
            "note": row.get("note", ""),
            "title": row.get("title", ""),
            "excerpt": row.get("excerpt", ""),
        }
    for row_id in expected:
        if row_id not in got:
            problems.append(f"{path.name}: row {row_id} is missing")
    return got, problems


def read_consensus(path: Path) -> tuple[dict[str, int], list[str]]:
    """``{row_id: grade}`` from a file with a ``row_id`` column and a ``consensus`` (or ``grade`` /
    ``final``) column. Rows with a blank grade are skipped here, and reported by the caller as
    unresolved if they were disagreements."""
    problems: list[str] = []
    out: dict[str, int] = {}
    for row in read_table(path):
        column = next((c for c in ("consensus", "grade", "final") if c in row), None)
        row_id = row.get("row_id", "")
        if column is None or not row_id:
            return {}, [f"{path}: needs `row_id` and `consensus` columns"]
        try:
            grade = parse_grade(row[column])
        except ValueError as error:
            problems.append(f"{path.name}: row {row_id}: {error}")
            continue
        if grade is not None:
            out[row_id] = grade
    return out, problems


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


def kappa_from_matrix(counts: np.ndarray, weights: str = "none") -> np.ndarray:
    """Cohen's kappa from confusion matrices of shape (..., 3, 3) (rows judge A, columns judge B).

    ``weights``: ``none`` (every disagreement counts 1), ``linear`` (|i − j| / 2) or ``quadratic``
    ((i − j)² / 4). κ = 1 − Σ w·O / Σ w·E with O the observed and E the chance proportions.
    NaN when chance already disagrees on nothing (one grade only), or with no rows.
    """
    k = len(GRADES)
    i, j = np.indices((k, k))
    if weights == "none":
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / (k - 1)
    elif weights == "quadratic":
        w = ((i - j) / (k - 1)) ** 2
    else:
        raise ValueError(f"weights must be none, linear or quadratic, not {weights!r}")
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=(-2, -1))
    rows = counts.sum(axis=-1)
    cols = counts.sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = (w * counts).sum(axis=(-2, -1)) / total
        expected = (w * rows[..., :, None] * cols[..., None, :]).sum(axis=(-2, -1)) / total**2
        return np.where(expected > 0, 1.0 - observed / expected, np.nan)


def confusion(a: Sequence[int], b: Sequence[int]) -> np.ndarray:
    matrix = np.zeros((len(GRADES), len(GRADES)), dtype=np.int64)
    for x, y in zip(a, b, strict=True):
        matrix[x, y] += 1
    return matrix


def agreement(
    key: Mapping[str, Any], grades_a: Mapping[str, int], grades_b: Mapping[str, int]
) -> dict[str, Any]:
    """Cohen's kappa (unweighted, linear- and quadratic-weighted: the pre-registration says
    "weighted" without saying how, so all three are shown), each with a 95% interval from resampling
    queries (the unit the metrics resample), the observed agreement and the confusion matrix."""
    query_ids = [q["id"] for q in key["queries"]]
    index = {qid: n for n, qid in enumerate(query_ids)}
    per_query = np.zeros((len(query_ids), len(GRADES), len(GRADES)), dtype=np.int64)
    for row in key["rows"]:
        per_query[index[row["query_id"]], grades_a[row["row_id"]], grades_b[row["row_id"]]] += 1
    total = per_query.sum(axis=0)
    n = int(total.sum())
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0, len(query_ids), size=(RESAMPLES, len(query_ids)))
    resampled = per_query[sample].sum(axis=1)
    out: dict[str, Any] = {
        "rows": n,
        "observed_agreement": float(np.trace(total) / n) if n else math.nan,
        "confusion": total.tolist(),
    }
    for weights in ("none", "linear", "quadratic"):
        value = float(kappa_from_matrix(total, weights))
        draws = kappa_from_matrix(resampled, weights)
        low, high = (
            np.nanquantile(draws, [0.025, 0.975]) if np.isfinite(draws).any() else (math.nan,) * 2
        )
        out[f"kappa_{weights}"] = {"value": value, "ci_low": float(low), "ci_high": float(high)}
    return out


# --------------------------------------------------------------------------- #
# The metrics
# --------------------------------------------------------------------------- #


def _dcg(gains: Sequence[int], k: int) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def per_query_metrics(key: Mapping[str, Any], grades: Mapping[str, int]) -> dict[str, Any]:
    """Each method's figure for each query, in ``key['queries']`` order.

    ``p3``/``p3_strict``: the share of the first 3 passages shown judged ≥ 1 / = 2, over 3.
    ``ndcg3``/``ndcg10``: gains = the grade, discount log2(rank + 1), the ideal ordering the
    query's pool by grade. ``recall10``: passages judged ≥ 1 in the first 10, over the pool's.
    ``valid`` marks the queries whose pool holds a passage judged ≥ 1: only there nDCG and recall
    exist, the same queries for every method.
    """
    queries = [q["id"] for q in key["queries"]]
    by_query: dict[str, list[dict]] = {qid: [] for qid in queries}
    for row in key["rows"]:
        by_query[row["query_id"]].append(row)
    valid = np.zeros(len(queries), dtype=bool)
    values = {
        m: {
            name: np.zeros(len(queries))
            for name in ("p3", "p3_strict", "ndcg3", "ndcg10", "recall10", "shown")
        }
        for m in METHODS
    }
    for q, qid in enumerate(queries):
        pool = by_query[qid]
        pool_grades = sorted((grades[r["row_id"]] for r in pool), reverse=True)
        relevant = sum(1 for g in pool_grades if g >= 1)
        valid[q] = relevant > 0
        for method in METHODS:
            listed = sorted(
                (r for r in pool if r["ranks"].get(method) is not None),
                key=lambda r, m=method: r["ranks"][m],
            )
            shown = [grades[r["row_id"]] for r in listed]
            top3, top10 = shown[:3], shown[:10]
            v = values[method]
            v["shown"][q] = len(shown)
            v["p3"][q] = sum(1 for g in top3 if g >= 1) / 3
            v["p3_strict"][q] = sum(1 for g in top3 if g >= 2) / 3
            if relevant:
                v["ndcg3"][q] = _dcg(top3, 3) / _dcg(pool_grades, 3)
                v["ndcg10"][q] = _dcg(top10, 10) / _dcg(pool_grades, 10)
                v["recall10"][q] = sum(1 for g in top10 if g >= 1) / relevant
    return {"queries": queries, "valid": valid, "values": values}


def interval(values: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    """The mean over the valid queries, with its 95% bootstrap interval over queries (1,000
    resamples, seed 42, ``bootstrap_ratio``)."""
    ratio = Ratio(np.where(valid, values, 0.0), valid.astype(float))
    low, high = bootstrap_ratio(ratio, resamples=RESAMPLES, seed=SEED)
    return {"value": ratio.value, "ci_low": low, "ci_high": high, "queries": ratio.cases}


METRICS = (
    ("p3", "Precision@3 (judged ≥ 1)", False),
    ("p3_strict", "Precision@3 (judged 2)", False),
    ("ndcg3", "nDCG@3", True),
    ("ndcg10", "nDCG@10", True),
    ("recall10", "pooled Recall@10 (judged ≥ 1)", True),
)
"""(key, label, needs a relevant passage in the pool)."""


def score_methods(key: Mapping[str, Any], grades: Mapping[str, int]) -> dict[str, Any]:
    """Every method's metrics with intervals, and the hybrid's paired comparisons (nDCG@10)."""
    per = per_query_metrics(key, grades)
    valid, values = per["valid"], per["values"]
    everyone = np.ones(len(per["queries"]), dtype=bool)
    methods: dict[str, Any] = {}
    for method in METHODS:
        v = values[method]
        entry = {
            name: interval(v[name], valid if needs_relevant else everyone)
            for name, _, needs_relevant in METRICS
        }
        entry["queries_showing_under_3"] = int((v["shown"] < 3).sum())
        entry["queries_showing_under_10"] = int((v["shown"] < POOL_DEPTH).sum())
        methods[method] = entry
    comparison: dict[str, Any] = {}
    if valid.any():
        for alone in ("bm25", "dense"):
            comparison[alone] = paired_bootstrap_difference(
                values[alone]["ndcg10"][valid],
                values["hybrid"]["ndcg10"][valid],
                resamples=RESAMPLES,
                seed=SEED,
            )
    return {
        "queries": len(per["queries"]),
        "queries_without_relevant_pool": int((~valid).sum()),
        "methods": methods,
        "comparison_alone_minus_hybrid_ndcg10": comparison,
    }


def apply_rule(scores: Mapping[str, Any]) -> dict[str, Any]:
    """EXP-011's rule, as written.

    The hybrid stays configured unless BM25 alone or dense alone is better on nDCG@10 with a paired
    interval that excludes 0 (then the better method alone is configured). If the configured
    method's Precision@3 (judged ≥ 1) is under 0.5, the team decides whether evidence stays.
    """
    comparison = scores["comparison_alone_minus_hybrid_ndcg10"]
    better = [m for m in ("bm25", "dense") if comparison.get(m, {}).get("better") == "a"]
    if better:
        configured = max(better, key=lambda m: comparison[m]["difference"])
        verdict = f"{METHOD_NAMES[configured]} replaces the hybrid (better on nDCG@10, interval excludes 0)"
    else:
        configured = "hybrid"
        verdict = "the hybrid stays configured (neither method alone is better on nDCG@10)"
    p3 = scores["methods"][configured]["p3"]["value"]
    return {
        "configured": configured,
        "verdict": verdict,
        "configured_precision_at_3": p3,
        "evidence_decision_for_the_team": bool(p3 < 0.5),
    }


# --------------------------------------------------------------------------- #
# Putting a judging round together
# --------------------------------------------------------------------------- #


def resolve_labels(
    key: Mapping[str, Any],
    a: Mapping[str, dict],
    b: Mapping[str, dict],
    consensus: Mapping[str, int] | None,
) -> tuple[dict[str, int], list[str], list[str]]:
    """``(final grades, disagreeing row ids, problems)``.

    A row both judges graded alike keeps that grade. A disagreement takes its consensus grade; with
    no consensus yet it has none, and is left out of ``final``. A consensus grade for a row the
    judges agreed on in another grade, or for a row not in the pool, is a problem: only
    disagreements are settled by discussion.
    """
    final: dict[str, int] = {}
    disagreements: list[str] = []
    problems: list[str] = []
    for row in key["rows"]:
        rid = row["row_id"]
        if a[rid]["grade"] == b[rid]["grade"]:
            final[rid] = a[rid]["grade"]
        else:
            disagreements.append(rid)
    if consensus is not None:
        known = {r["row_id"] for r in key["rows"]}
        for rid, grade in consensus.items():
            if rid not in known:
                problems.append(f"consensus: row {rid!r} is not in the pool")
            elif rid in final and final[rid] != grade:
                problems.append(
                    f"consensus: row {rid} was agreed at {final[rid]}, not a disagreement to settle"
                )
            elif rid not in final:
                final[rid] = grade
        for rid in disagreements:
            if rid not in final:
                problems.append(f"consensus: no grade for the disagreement at row {rid}")
    return final, disagreements, problems


def score_round(
    key: Mapping[str, Any],
    sheet_a: Path,
    sheet_b: Path,
    consensus_path: Path | None = None,
) -> dict[str, Any]:
    """Everything ``score`` reports, as data. Raises ``ValueError`` listing every problem found in
    the sheets or the consensus file."""
    a, problems_a = read_judgements(sheet_a, key)
    b, problems_b = read_judgements(sheet_b, key)
    problems = problems_a + problems_b
    consensus = None
    if consensus_path is not None:
        consensus, problems_c = read_consensus(consensus_path)
        problems += problems_c
    if problems:
        raise ValueError("\n".join(problems))
    final, disagreements, problems = resolve_labels(key, a, b, consensus)
    if problems:
        raise ValueError("\n".join(problems))
    grades_a = {rid: v["grade"] for rid, v in a.items()}
    grades_b = {rid: v["grade"] for rid, v in b.items()}
    result: dict[str, Any] = {
        "judges": list(key["judges"]),
        "labels": "consensus" if consensus is not None else "provisional",
        "agreement": agreement(key, grades_a, grades_b),
        "disagreements": disagreements,
        "treatment_flags": [
            {
                "row_id": r["row_id"],
                "query_id": r["query_id"],
                "pmcid": r["pmcid"],
                "flagged_by": [
                    name
                    for name, sheet in zip(key["judges"], (a, b), strict=True)
                    if sheet[r["row_id"]]["treatment"]
                ],
            }
            for r in key["rows"]
            if a[r["row_id"]]["treatment"] or b[r["row_id"]]["treatment"]
        ],
        "final": final,
        "a": a,
        "b": b,
    }
    if consensus is not None:
        scores = score_methods(key, final)
        result["scores"] = scores
        result["rule"] = apply_rule(scores)
    else:
        result["scores_by_judge"] = {
            name: score_methods(key, grades)
            for name, grades in zip(key["judges"], (grades_a, grades_b), strict=True)
        }
    return result


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def _fmt(entry: Mapping[str, float]) -> str:
    return (
        f"{entry['value']:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}] "
        f"(n={entry['queries']})"
    )


def _methods_table(scores: Mapping[str, Any], heading: str) -> list[str]:
    lines = [heading]
    for name, label, _ in METRICS:
        lines.append(f"  {label}")
        for method in METHODS:
            lines.append(f"    {METHOD_NAMES[method]:<24}{_fmt(scores['methods'][method][name])}")
    short = {
        METHOD_NAMES[m]: (
            scores["methods"][m]["queries_showing_under_3"],
            scores["methods"][m]["queries_showing_under_10"],
        )
        for m in METHODS
    }
    lines.append(
        "  queries where a method showed fewer than 3 / fewer than 10 passages: "
        + "; ".join(f"{name} {u3} / {u10}" for name, (u3, u10) in short.items())
    )
    if scores["queries_without_relevant_pool"]:
        lines.append(
            f"  {scores['queries_without_relevant_pool']} of {scores['queries']} queries have no "
            "passage judged ≥ 1 in their pool: left out of nDCG and Recall, for every method"
        )
    return lines


def render_report(key: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    """The plain-text report ``score`` prints."""
    ag = result["agreement"]
    out = [
        "EXP-011 — evidence retrieval: the judging, scored",
        f"labels: {result['labels'].upper()}"
        + (
            ""
            if result["labels"] == "consensus"
            else " (one set per judge, and no rule applied, until a consensus file settles the "
            "disagreements)"
        ),
        f"queries {len(key['queries'])} · pooled rows {len(key['rows'])} · "
        f"judges {', '.join(result['judges'])}",
        "",
        "Agreement before discussion (docs/08 EXP-011: the weighted kappa; the weights are not "
        "pre-registered, so three are shown)",
        f"  observed agreement {ag['observed_agreement']:.3f} over {ag['rows']} rows, "
        f"{len(result['disagreements'])} disagreements",
    ]
    for weights, label in (
        ("none", "unweighted"),
        ("linear", "linear"),
        ("quadratic", "quadratic"),
    ):
        k = ag[f"kappa_{weights}"]
        out.append(
            f"  kappa, {label:<11}{k['value']:.3f} [{k['ci_low']:.3f}, {k['ci_high']:.3f}]"
            "  (95%, resampling queries)"
        )
    out.append(
        "  confusion (rows: " + result["judges"][0] + ", columns: " + result["judges"][1] + ")"
    )
    for grade, line in zip(GRADES, ag["confusion"], strict=True):
        out.append(f"    {grade}: " + "  ".join(f"{n:4d}" for n in line))
    flags = result["treatment_flags"]
    out += [
        "",
        f"Treatment that reached a judge: {len(flags)} row(s) flagged (the target is none; each "
        "one extends the screen's pattern, and the bundle is rebuilt)",
    ]
    out += [
        f"  {f['row_id']}  {f['query_id']}  PMC{f['pmcid'].removeprefix('PMC')}  "
        f"flagged by {', '.join(f['flagged_by'])}"
        for f in flags
    ]
    out.append("")
    if result["labels"] == "consensus":
        out += _methods_table(result["scores"], "Metrics on the consensus labels, per method")
        out += ["", "Hybrid against each method alone, nDCG@10 (paired bootstrap, alone − hybrid)"]
        for alone, c in result["scores"]["comparison_alone_minus_hybrid_ndcg10"].items():
            out.append(
                f"  {METHOD_NAMES[alone]:<24}{c['difference']:+.3f} "
                f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  "
                f"{'excludes 0, favours ' + ('it' if c['better'] == 'a' else 'the hybrid') if c['excludes_zero'] else 'includes 0'}"
            )
        rule = result["rule"]
        out += [
            "",
            "The pre-registered rule",
            f"  {rule['verdict']}",
            f"  the configured method's Precision@3 (judged ≥ 1): "
            f"{rule['configured_precision_at_3']:.3f}"
            + (
                " — UNDER 0.5: the team decides whether evidence stays in the prototype"
                if rule["evidence_decision_for_the_team"]
                else " (not under 0.5)"
            ),
        ]
    else:
        for name, scores in result["scores_by_judge"].items():
            out += _methods_table(scores, f"PROVISIONAL, on judge {name}'s labels alone")
            out.append("")
        out.append(
            f"Next: the judges settle the {len(result['disagreements'])} disagreement(s) in "
            f"{DISAGREEMENTS_NAME} (fill `consensus`), then run `score --consensus` on that file."
        )
    out += [
        "",
        "Definitions: Precision@k divides by k; nDCG's gain is the grade, its ideal ordering the "
        "query's judged pool; each figure is the mean over queries; 95% intervals resample queries "
        f"({RESAMPLES} resamples, seed {SEED}). Pooled recall is relative to what the three "
        "methods found, not to the corpus. Closed world (R-13): 14 conditions. No DDXPlus "
        "patient is read: the queries are the conditions and the hand-written golden cases.",
    ]
    return "\n".join(out)


def scores_json(key: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """The machine-readable report: figures only, no article text."""
    keep = ("judges", "labels", "agreement", "disagreements", "treatment_flags")
    out = {k: result[k] for k in keep}
    out["queries"] = len(key["queries"])
    out["rows"] = len(key["rows"])
    for name in ("scores", "rule", "scores_by_judge"):
        if name in result:
            out[name] = result[name]
    return out


def judgement_rows(key: Mapping[str, Any], result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``exp011_judgements.csv``'s rows: PMCIDs and labels only, no article text (EXP-011)."""
    return [
        {
            "row_id": r["row_id"],
            "query_id": r["query_id"],
            "pmcid": r["pmcid"],
            "grade_a": result["a"][r["row_id"]]["grade"],
            "grade_b": result["b"][r["row_id"]]["grade"],
            "final": result["final"].get(r["row_id"], ""),
            "treatment_a": int(result["a"][r["row_id"]]["treatment"]),
            "treatment_b": int(result["b"][r["row_id"]]["treatment"]),
        }
        for r in key["rows"]
    ]


def disagreement_rows(key: Mapping[str, Any], result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The rows to settle by discussion, with a blank ``consensus`` to fill. Holds the title and
    excerpt (from the judges' sheet) so the discussion has the text, so it stays under data/."""
    by_id = {r["row_id"]: r for r in key["rows"]}
    queries = {q["id"]: q for q in key["queries"]}
    a, b = result["a"], result["b"]
    out = []
    for rid in result["disagreements"]:
        query = queries[by_id[rid]["query_id"]]
        out.append(
            {
                "row_id": rid,
                "query_id": query["id"],
                "condition": _cell(query["condition"]),
                "patient_findings": _cell("; ".join(query["findings"]) or NO_FINDINGS),
                "pmcid": by_id[rid]["pmcid"],
                "title": _cell(a[rid]["title"]),
                "excerpt": _cell(a[rid]["excerpt"]),
                f"grade_{key['judges'][0]}": a[rid]["grade"],
                f"grade_{key['judges'][1]}": b[rid]["grade"],
                f"note_{key['judges'][0]}": _cell(a[rid]["note"]),
                f"note_{key['judges'][1]}": _cell(b[rid]["note"]),
                "consensus": "",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Files and the command line
# --------------------------------------------------------------------------- #


def ensure_private(path: Path) -> Path:
    """Refuse an output path inside the repository but outside ``data/``: the sheets hold corpus
    text and the key holds the answer, and neither is ever committed (docs/03 §1.4)."""
    resolved = Path(path).resolve()
    inside_repo = resolved == REPO_ROOT or REPO_ROOT in resolved.parents
    inside_data = resolved == DATA_DIR or DATA_DIR in resolved.parents
    if inside_repo and not inside_data:
        raise SystemExit(
            f"refusing to write to {path}: it is inside the repository but not under data/, "
            "which is gitignored. Corpus text is never committed."
        )
    return resolved


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def queries_for_pool(
    args: argparse.Namespace, build_pipeline: Callable[..., dict[str, Any]] | None = None
) -> tuple[list[Query], list[str]]:
    """The 26 queries: from ``--queries`` (a file an earlier ``pool`` wrote), or by running the
    configured pipeline on the golden cases."""
    if args.queries:
        data = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        queries = data["queries"] if isinstance(data, dict) else data
        check_queries(queries)
        return queries, list(data.get("degraded", [])) if isinstance(data, dict) else []

    from _config import build_pipeline as default_build
    from _config import load_config

    from src.medical_kg.neo4j_store import GraphUnavailable

    config = load_config(args.config)
    # Retrieval is attached after ranking and never changes it, so the pipeline runs without a
    # bundle here: only the candidates it ranks first matter.
    config.setdefault("retrieval", {})["bundle"] = None
    try:
        built = (build_pipeline or default_build)(config)
    except GraphUnavailable as error:
        raise SystemExit(
            f"No knowledge graph ({error}). The case queries are the configured pipeline's top "
            f"three; run this where data/ and models/ are, or pass --queries FILE from an earlier "
            "run."
        ) from error
    queries, degraded = case_queries(load_golden(), built["pipeline"], built["labels"])
    blocking = sorted(set(degraded) - IGNORABLE_DEGRADATION)
    if blocking:
        raise SystemExit(
            f"the configured pipeline is degraded ({', '.join(blocking)}), so the case queries "
            "would not be the pre-registered ones. Fix it (models/, data/, Aura) and run again."
        )
    return condition_queries() + queries, degraded


def cmd_pool(args: argparse.Namespace) -> int:
    out_dir = ensure_private(args.out_dir)
    sheets_dir = out_dir / "sheets"
    judges = list(args.judges)
    if len(judges) != 2 or judges[0] == judges[1]:
        raise SystemExit("--judges needs two different labels, e.g. A B")
    for label in judges:
        if not label.replace("_", "").replace("-", "").isalnum():
            raise SystemExit(f"judge label {label!r}: use letters, digits, - and _ only")
    targets = [
        *(sheets_dir / f"judge_{label}.csv" for label in judges),
        sheets_dir / INSTRUCTIONS_NAME,
        out_dir / KEY_NAME,
        out_dir / QUERIES_NAME,
    ]
    existing = [t for t in targets if t.exists()]
    if existing and not args.force:
        raise SystemExit(
            "already exists (a judge may have started on it): "
            + ", ".join(str(t) for t in existing)
            + ". Choose another --out-dir, or --force to overwrite."
        )

    queries, degraded = queries_for_pool(args)
    from src.rag.bundle import BundleError

    try:
        bundle = load_bundle(args.bundle)
    except BundleError as error:
        raise SystemExit(f"the bundle at {args.bundle} cannot be used: {error}") from error
    retriever = HybridRetriever(bundle, depth=POOL_DEPTH)
    rows, returned = build_pool(retriever, queries)
    summary = pool_summary(rows, queries)

    sheets_dir.mkdir(parents=True, exist_ok=True)
    sheet = sheet_rows(rows, queries)
    for label in judges:
        write_sheet(sheets_dir / f"judge_{label}.csv", sheet)
    (sheets_dir / INSTRUCTIONS_NAME).write_text(JUDGE_INSTRUCTIONS, encoding="utf-8")
    manifest = Path(args.bundle) / "manifest.json"
    key = make_key(
        rows,
        queries,
        returned,
        judges=judges,
        bundle_info={
            "path": str(args.bundle),
            "manifest_sha256": sha256_of(manifest) if manifest.is_file() else None,
            "passages": len(bundle.passages),
            "articles": len(bundle.articles),
            "screen_digest_now": SCREEN_DIGEST,
            "screen_digest_bundle": (bundle.manifest.get("screening") or {}).get("digest"),
            "rescreened": retriever.rescreened,
        },
        degraded=degraded,
    )
    _write_json(out_dir / KEY_NAME, key)
    _write_json(out_dir / QUERIES_NAME, {"queries": queries, "degraded": degraded})

    print(f"pool: {summary['queries']} queries, {summary['rows']} rows to judge")
    print(
        f"  per query: min {summary['per_query_min']}, median {summary['per_query_median']:g}, "
        f"max {summary['per_query_max']}"
    )
    if summary["queries"] != 26:
        print(
            f"  note: {summary['queries']} queries, not 26 (EXP-011 allows golden cases added "
            "before judging begins: 3 queries each)"
        )
    if summary["empty_queries"]:
        print(f"  WARNING: no passage can be shown for {', '.join(summary['empty_queries'])}")
    if summary["rows"] < MIN_PAIRS:
        print(f"  WARNING: {summary['rows']} rows is under docs/05 §3.4's {MIN_PAIRS} pairs")
    else:
        print(f"  covers docs/05 §3.4's {MIN_PAIRS} judged pairs")
    print(f"  send each judge their own sheet and {INSTRUCTIONS_NAME}, from {sheets_dir}:")
    for label in judges:
        print(f"    judge_{label}.csv")
    print(f"  KEEP PRIVATE until both have finished: {out_dir / KEY_NAME}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    key_path = Path(args.key)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if key.get("format") != KEY_FORMAT or key.get("version") != KEY_VERSION:
        raise SystemExit(f"{key_path} is not a pool key this script wrote")
    out_dir = ensure_private(args.out_dir or key_path.parent)
    judgements_out = ensure_private(args.judgements_out)
    try:
        result = score_round(key, Path(args.sheet_a), Path(args.sheet_b), args.consensus)
    except ValueError as error:
        lines = str(error).splitlines()
        shown = "\n  ".join(lines[:40])
        more = f"\n  … and {len(lines) - 40} more" if len(lines) > 40 else ""
        print(f"cannot score:\n  {shown}{more}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    judgements_out.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(
        judgements_out,
        (
            "row_id",
            "query_id",
            "pmcid",
            "grade_a",
            "grade_b",
            "final",
            "treatment_a",
            "treatment_b",
        ),
        judgement_rows(key, result),
    )
    rows = disagreement_rows(key, result)
    disagreements = out_dir / DISAGREEMENTS_NAME
    # The judges fill this file's `consensus` column and may pass it back as --consensus, so a
    # later run must never write over it.
    keep = bool(rows) and disagreements.exists()
    if rows and not keep:
        _write_csv(disagreements, list(rows[0]), rows)
    _write_json(out_dir / SCORES_NAME, scores_json(key, result))
    print(render_report(key, result))
    print()
    print(f"wrote {judgements_out} (PMCIDs and labels only)")
    if keep:
        print(f"kept {disagreements} as it is (delete it first to write a fresh one)")
    elif rows:
        print(f"wrote {disagreements}: {len(rows)} row(s) to settle")
    print(f"wrote {out_dir / SCORES_NAME}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    pool = sub.add_parser("pool", help="build the blind judging pool and the judges' sheets")
    pool.add_argument("--bundle", type=Path, default=REPO_ROOT / DEFAULT_BUNDLE)
    pool.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    pool.add_argument("--judges", nargs=2, default=["A", "B"], metavar="LABEL")
    pool.add_argument(
        "--queries", type=Path, help=f"reuse the queries of an earlier {QUERIES_NAME}"
    )
    pool.add_argument("--config", type=Path, default=None, help="configs/config.yaml")
    pool.add_argument("--force", action="store_true", help="overwrite an existing pool")
    pool.set_defaults(run=cmd_pool)

    score = sub.add_parser("score", help="score the judges' filled sheets")
    score.add_argument("--key", type=Path, default=DEFAULT_OUT_DIR / KEY_NAME)
    score.add_argument("--sheet-a", type=Path, required=True, help="the first judge's filled sheet")
    score.add_argument(
        "--sheet-b", type=Path, required=True, help="the second judge's filled sheet"
    )
    score.add_argument(
        "--consensus", type=Path, help="row_id, consensus: settles every disagreement"
    )
    score.add_argument("--out-dir", type=Path, help="default: the key's folder")
    score.add_argument("--judgements-out", type=Path, default=DEFAULT_JUDGEMENTS)
    score.set_defaults(run=cmd_score)

    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
