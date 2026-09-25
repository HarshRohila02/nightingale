"""Run a case through the pipeline and print the result.

This is the walking skeleton: it exercises the full path from a patient case to a
ranked, explained, red-flagged differential. The ranker is the trained model when
configs/config.yaml's ml: block points at one that is on this machine, and degrades
to knowledge-graph-only ranking when it does not (src/ml/ranker.py). The graph is the
real cardiac knowledge graph the graph: block names: the copy in AuraDB, or the local
build when Aura is paused (src/medical_kg/neo4j_store.py). The case is expanded
through the crosswalk first, because the real graph knows DDXPlus questions. The demo
never writes to Aura.

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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _config import (  # noqa: E402
    GRAPH_BACKENDS,
    describe_graph,
    load_config,
    open_graph_from_config,
    open_ranker_from_config,
)

from src.contracts import DiagnosisResult, PatientCase  # noqa: E402
from src.medical_kg.crosswalk import DERIVED_SOURCE, expand_case  # noqa: E402
from src.medical_kg.neo4j_store import GraphUnavailable  # noqa: E402
from src.ml.ranker import describe  # noqa: E402
from src.pipeline import DiagnosisPipeline  # noqa: E402
from src.stubs import MISSING_SHOWN, EmptyRetriever, TemplateExplainer  # noqa: E402

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


def answered_by(case: PatientCase) -> dict[str, str]:
    """{question the crosswalk added: label of the recorded finding that answers it}.

    The graph links a condition to the *question*, so that stays the claim; the answer is shown
    beside it. "Pressure-type pain" answers "Characterize your pain", which GERD is linked to, but
    the graph never says GERD presents with pressure-type pain (docs/02 §5.2).
    """
    everything = [*case.findings, *case.risk_factors]
    recorded = {f.concept_id: f.label for f in everything if f.source != DERIVED_SOURCE}
    return {
        f.concept_id: recorded[origin]
        for f in everything
        if f.source == DERIVED_SOURCE and (origin := f.qualifiers.get("crosswalk_from")) in recorded
    }


def render(result: DiagnosisResult, top_k: int = 5, case: PatientCase | None = None) -> str:
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
        "   score: ml and kg fused; ml: the model's raw score, not a calibrated probability;",
        "   kg: the graph's score (a log-likelihood on the real graph, at most 0);",
        f"   {WARN} red-flagged ones come first",
        "",
    ]
    for i, candidate in enumerate(result.candidates[:top_k], start=1):
        marker = f" {WARN}" if candidate.red_flag else "  "
        out.append(
            f"   {i}.{marker} {candidate.label:<34} "
            f"score {candidate.fused_score:.3f}  (ml {candidate.ml_score:.2f} / "
            f"kg {candidate.kg_score:.2f})"
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
        "--ranker", default=None, help="Override ml.backend: logreg | xgboost | none"
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
    ranker = open_ranker_from_config(config)
    try:
        graph = open_graph_from_config(config, args.graph)
    except GraphUnavailable as exc:
        print(
            f"No knowledge graph: {exc}. Build it with scripts/decode_ddxplus.py, "
            "or run with --graph stub.",
            file=sys.stderr,
        )
        return 1

    case = PatientCase(**cases[args.case]["case"])
    if hasattr(graph, "graph"):
        # The real graph knows DDXPlus questions; the golden cases are written in SYM:/RF: ids.
        case = expand_case(case, dict(graph.graph.nodes(data="label")))

    pipeline = DiagnosisPipeline(
        ranker=ranker,
        graph=graph,
        retriever=EmptyRetriever(),
        explainer=TemplateExplainer(),
    )
    result = pipeline.run(case)
    print(describe(ranker))
    print(describe_graph(graph))
    print(render(result, case=case))
    return 0


if __name__ == "__main__":
    sys.exit(main())
