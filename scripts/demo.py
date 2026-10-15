"""Run a case through the pipeline and print the result.

This is the walking skeleton: it exercises the full path from a patient case to a
ranked, explained, red-flagged differential. The ranker is the trained model when
configs/config.yaml's ml: block points at one that is on this machine, and degrades
to knowledge-graph-only ranking when it does not (src/ml/ranker.py). The graph is the
real cardiac knowledge graph the graph: block names: the copy in AuraDB, or the local
build when Aura is paused (src/medical_kg/neo4j_store.py). The case is expanded
through the crosswalk first, because the real graph knows DDXPlus questions. The demo
never writes to Aura.

The pipeline is the one the dashboard and the API run (``build_pipeline`` in
scripts/_config.py): the configured retriever too, which gives no evidence and reports
``rag`` until the corpus bundle is on this machine (3a; docs/11 §4.6). Below the
differential the demo prints what the dashboard shows beside it, from the same helpers
(app/view.py): the evidence with its citations, the questions the graph would learn
most from and the cited tests to consider (3d). None of them changes the ranking.

    python scripts/demo.py                    # the anchor ACS case
    python scripts/demo.py --case GC-003      # aortic dissection (KG-only red flag)
    python scripts/demo.py --graph networkx   # the local build, without trying Aura
    python scripts/demo.py --graph stub       # the toy graph, where data/ is absent
    python scripts/demo.py --ranker none      # force graph-only ranking
    python scripts/demo.py --list
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _config import GRAPH_BACKENDS, build_pipeline, load_config  # noqa: E402

from app.view import evidence_note, evidence_rows, next_steps  # noqa: E402
from src.conditions import BY_ID  # noqa: E402
from src.contracts import DiagnosisResult, PatientCase  # noqa: E402
from src.medical_kg.crosswalk import expand_case, recorded_answers  # noqa: E402
from src.medical_kg.neo4j_store import GraphUnavailable  # noqa: E402
from src.stubs import MISSING_SHOWN  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden_cases.yaml"


def _init_console() -> bool:
    """Try to put stdout into UTF-8 and report whether symbols are safe to print.

    Windows consoles default to cp1252, which cannot encode the warning sign or
    the tick/cross markers. Rather than crash on output, fall back to ASCII.
    """
    with contextlib.suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    try:
        "⚠ ✓ ✗ ⓘ".encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


UNICODE_OK = _init_console()

# (warning, supporting, contradicting, missing, info)
MARKS = ("⚠", "✓", "✗", "?", "ⓘ") if UNICODE_OK else ("!", "+", "-", "?", "i")
WARN, TICK, CROSS, QUERY, INFO = MARKS


def _ascii_safe(text: str) -> str:
    """Drop characters the console cannot render, so output never crashes."""
    if UNICODE_OK:
        return text
    return text.encode("ascii", errors="ignore").decode("ascii").strip()


def load_cases() -> dict[str, dict]:
    data = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    return {c["id"]: c for c in data["cases"]}


answered_by = recorded_answers
"""{question the crosswalk added: the recorded finding answering it} (src/medical_kg/crosswalk.py)."""


def render(
    result: DiagnosisResult,
    top_k: int = 5,
    case: PatientCase | None = None,
    graph: Any = None,
    retriever: Any = None,
) -> str:
    """The result as the console shows it.

    Args:
        result: The pipeline's result.
        top_k: How many candidates the differential lists.
        case: The case the pipeline ranked (after ``expand_case`` on the real graph): its
            recorded answers label the graph's questions, and the questions to ask next are
            computed for it.
        graph: The graph store that ranked it, for the questions to ask next. Without one, or
            with one that cannot answer, no question is suggested and the block says why.
        retriever: The retriever that gave the evidence, for each article's licence and
            attribution (its bundle's ``source_of``); without one the citation stands alone.
    """
    width = 74
    answers = answered_by(case) if case else {}
    out: list[str] = [
        "=" * width,
        "  NIGHTINGALE — CLINICAL DECISION SUPPORT".ljust(width),
        "=" * width,
    ]
    out.append(f"  Case: {result.case_id}")

    if result.red_flags:
        out += ["", "  " + "-" * (width - 4), f"  {WARN} RED FLAGS", "  " + "-" * (width - 4)]
        out += [f"    * {_ascii_safe(flag)}" for flag in result.red_flags]

    out += ["", "  " + "-" * (width - 4), "  DIFFERENTIAL (ranked)", "  " + "-" * (width - 4)]
    out += [
        "   score: ml and kg pooled, a log-probability (not calibrated);",
        "   ml: the model's raw score, not a calibrated probability (— : not trained on it);",
        "   kg: the graph's score (a log-likelihood on the real graph, at most 0);",
        f"   {WARN} red-flagged ones come first",
        "",
    ]
    for i, candidate in enumerate(result.candidates[:top_k], start=1):
        marker = f" {WARN}" if candidate.red_flag else "  "
        # The model is silent about a condition it was not trained on, not against it (2c).
        trained = BY_ID[candidate.condition_id].in_training_data
        ml = f"{candidate.ml_score:.2f}" if trained else "  — "
        out.append(
            f"   {i}.{marker} {candidate.label:<34} "
            f"score {candidate.fused_score:.3f}  (ml {ml} / kg {candidate.kg_score:.2f})"
        )

    top = result.candidates[0] if result.candidates else None
    if top:
        out += [
            "",
            "  " + "-" * (width - 4),
            f"  WHY {top.label.upper()}?",
            "  " + "-" * (width - 4),
        ]
        for mark, assessments in ((TICK, top.supporting()), (CROSS, top.contradicting())):
            for a in assessments:
                answer = answers.get(a.finding_id)
                out.append(f"    {mark} {a.label}" + (f"  <- {answer}" if answer else ""))
        missing = top.missing()
        for a in missing[:MISSING_SHOWN]:
            out.append(f"    {QUERY} {a.label}  (not recorded)")
        if len(missing) > MISSING_SHOWN:
            out.append(
                f"    {QUERY} ...and {len(missing) - MISSING_SHOWN} more the graph links to "
                f"{top.label}, not recorded"
            )

    out += _evidence_block(result, retriever, width)
    out += _next_steps_block(next_steps(result, case, graph), width)

    if result.explanation:
        out += ["", "  " + "-" * (width - 4), "  EXPLANATION", "  " + "-" * (width - 4)]
        for line in _wrap(_ascii_safe(result.explanation.text), width - 6):
            out.append(f"    {line}")

    if result.degraded_components:
        out += ["", f"  {INFO} Degraded components: {', '.join(result.degraded_components)}"]

    out += ["", "  " + "-" * (width - 4)]
    for line in _wrap(_ascii_safe(result.disclaimer), width - 6):
        out.append(f"    {line}")
    out.append("=" * width)
    return "\n".join(out)


def _section(title: str, width: int) -> list[str]:
    rule = "  " + "-" * (width - 4)
    return ["", rule, f"  {title}", rule]


def _bullet(text: str, width: int, first: str, rest: str) -> list[str]:
    """``text`` wrapped to ``width``, its first line after ``first`` and the others after ``rest``."""
    lines = _wrap(_ascii_safe(text), width - max(len(first), len(rest)))
    return [(rest if i else first) + line for i, line in enumerate(lines)]


def _evidence_block(result: DiagnosisResult, retriever: Any, width: int) -> list[str]:
    """The evidence for the leading candidates (3a), as the dashboard shows it in each candidate's
    row: the excerpt verbatim, its citation and PMC link, the article's title, and, from the
    bundle, its attribution and licence (app/view.py). One quiet line when retrieval is off."""
    out = _section("EVIDENCE (excerpts from PubMed Central Open Access articles)", width)
    note = evidence_note(result)
    if note:
        return out + _bullet(note, width, "    ", "    ")
    rows = evidence_rows(result, retriever)
    if not rows:
        return out + ["    No passage was found for the leading candidates."]
    for candidate in result.candidates:
        passages = rows.get(candidate.condition_id, [])
        if passages:
            out.append(f"    {candidate.label}")
        for row in passages:
            out += _bullet(f'"{row["snippet"]}"', width, "      - ", "        ")
            out += _bullet(f"{row['citation']}  {row['url']}", width, "        ", "        ")
            about = [row["title"], row.get("attribution"), row.get("licence")]
            out += _bullet(" | ".join(p for p in about if p), width, "        ", "        ")
    return out


def _next_steps_block(steps: dict[str, Any], width: int) -> list[str]:
    """3d's "What to ask next", as the dashboard shows it (app/view.py ``next_steps``): the
    findings the knowledge graph would learn most from, then the cited tests to consider."""
    out = _section("WHAT TO ASK NEXT", width)
    out.append("    Questions the knowledge graph would learn most from:")
    if steps["questions"]:
        for q in steps["questions"]:
            why = "; ".join(
                part
                for part in (
                    "a yes favours " + ", ".join(q["if_present"]) if q["if_present"] else "",
                    "a no favours " + ", ".join(q["if_denied"]) if q["if_denied"] else "",
                )
                if part
            )
            text = f"{q['verb']} {q['label']}" + (f": {why}" if why else "")
            out += _bullet(text, width, f"      {QUERY} ", "        ")
    elif steps["entropy_bits"] is None:
        out.append("      The knowledge graph is unavailable, so no questions are suggested.")
    else:
        out.append("      No single answer would tell the graph anything more.")
    out += ["", "    Tests to consider (investigations only, from the guideline in brackets):"]
    for test in steps["tests"]:
        out += _bullet(test, width, "      - ", "        ")
    if not steps["tests"]:
        out.append("      none suggested")
    out.append("")
    out += _bullet(
        "Questions are ordered by how much their answer would reduce the knowledge graph's "
        "uncertainty over the 14 conditions; the model is not consulted. Tests cover the top "
        "three candidates and every red flag. The clinician decides.",
        width,
        "    ",
        "    ",
    )
    return out


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="GC-001", help="Golden case id (default: GC-001)")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument("--config", type=Path, default=None, help="configs/config.yaml")
    parser.add_argument(
        "--ranker", default=None, help="Override ml.backend: deep | logreg | xgboost | none"
    )
    parser.add_argument(
        "--graph", choices=GRAPH_BACKENDS, default=None, help="Override graph.backend"
    )
    args = parser.parse_args(argv)

    cases = load_cases()

    if args.list:
        for case_id, case in cases.items():
            print(f"{case_id}: {case['description'].strip().splitlines()[0]}")
        return 0

    if args.case not in cases:
        print(f"Unknown case {args.case!r}. Available: {', '.join(cases)}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    if args.ranker:
        config.setdefault("ml", {})["backend"] = args.ranker
    try:
        built = build_pipeline(config, args.graph)
    except GraphUnavailable as exc:
        print(
            f"No knowledge graph: {exc}. Build it with scripts/decode_ddxplus.py, "
            "or run with --graph stub.",
            file=sys.stderr,
        )
        return 1

    case = PatientCase(**cases[args.case]["case"])
    if built["labels"] is not None:
        # The real graph knows DDXPlus questions; the golden cases are written in SYM:/RF: ids.
        case = expand_case(case, built["labels"])

    result = built["pipeline"].run(case)
    for line in ("ranker", "graph", "retriever"):
        print(_ascii_safe(built["about"][line]))
    print(render(result, case=case, graph=built["graph"], retriever=built["retriever"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
