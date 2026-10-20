"""EXP-009 / EXP-010: the LLM baselines B3 and B4, and the grounded explainer (A0+LLM), on a sample.

    python scripts/run_llm_baselines.py --out DIR [--bundle DIR] [--smoke]

**This calls a language model through Ollama, and so it runs on Kaggle, never on the laptop**
(D-5, D-7, D-9; ``notebooks/kaggle_llm_explain.ipynb``, ``docs/11`` §4.8). Nothing is trained.
The protocol is pre-registered in ``docs/08`` (EXP-009); what each arm is given, the prompts, the
parser, the claim check and the sample are in ``src/llm/baselines.py`` and ``src/eval/claims.py``.

The sample: the four golden cases and 50 validation patients drawn by hash order (seed 42). The
patients are read at 100%, 50% and 25% of their history on the approved masks of ``docs/05`` §3.7,
whose two digests must reproduce or the run stops. The arms:

* **B3**, the model alone: the case in the prompt, no retrieval, no graph, no ranker; 100%, 50%, 25%.
* **B3r**, B3 told what the grounded systems are told (use only the facts given); 100%.
* **B4**, text retrieval and the model: the retriever's best passage for each of the 14 conditions;
  100%, 50%, 25%. Needs ``--bundle``.
* **A0+LLM**, the pipeline as configured with the LLM explainer on: the model phrases the
  explanation. Its raw answer is kept beside what the guardrails let through; 100%. Needs
  ``--bundle``.

One call per (arm, case, level), temperature 0, seed 42; the results are appended to
``results.jsonl`` as they come, so a run that stops resumes where it stopped. Every failure is
recorded (a call that errors, an answer that does not parse) and counted, never dropped.
``scripts/score_llm_baselines.py`` turns the file into the pre-registered figures.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from src.conditions import BY_ID, CONDITIONS  # noqa: E402
from src.contracts import Evidence, PatientCase  # noqa: E402
from src.eval.claims import claim_stats  # noqa: E402
from src.llm.baselines import (  # noqa: E402
    A0_LLM,
    ARMS,
    B3,
    B3R,
    B4,
    PASSAGES_PER_CONDITION,
    SAMPLE_SEED,
    SAMPLE_SIZE,
    SYSTEM_PROMPTS,
    ParsedAnswer,
    build_prompt,
    case_text,
    given_result,
    parse_answer,
    reference_case,
    sample_digest,
    sample_ids,
)
from src.llm.client import LLMError, OllamaClient, urllib_transport  # noqa: E402
from src.llm.explainer import SYSTEM as EXPLAINER_SYSTEM  # noqa: E402
from src.llm.explainer import LLMExplainer  # noqa: E402
from src.rag.bundle import REQUIRED_FILES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"

MODEL = "llama3.1:8b"
TEMPERATURE = 0.0
SEED = 42
MAX_TOKENS = 512
NUM_CTX = 8192
TIMEOUT_SECONDS = 120.0
MAX_CONSECUTIVE_FAILURES = 5
"""A run that makes this many failed calls in a row has lost the model, not found a result."""
RETRIES = 2
"""A call that fails (an error, a timeout) is tried again up to this many times; every attempt
is counted in the record."""

LEVELS_BY_ARM: dict[str, tuple[float, ...]] = {
    B3: (1.0, 0.5, 0.25),
    B4: (1.0, 0.5, 0.25),
    B3R: (1.0,),
    A0_LLM: (1.0,),
}
"""docs/05 §3.7: B3 and B4 are scored at 50% and 25% on the same sample as at 100%; §3.4 and §3.5,
the explanation measures, are taken at 100% only, so B3r and A0+LLM are run there only."""

__all__ = [
    "Caller",
    "Experiment",
    "Item",
    "Reply",
    "find_bundle",
    "load_golden",
    "main",
    "place_bodhi_s",
    "previous_results",
    "server_info",
]


# --------------------------------------------------------------------------------------------- #
# Calling the model
# --------------------------------------------------------------------------------------------- #


@dataclass
class Reply:
    """One call's outcome: the text, or why there is none, and what Ollama reported."""

    text: str | None
    error: str | None
    attempts: int
    elapsed_s: float
    ollama: dict[str, Any] = field(default_factory=dict)
    prompt: str | None = None


class RecordingTransport:
    """The standard transport, remembering the last reply (``eval_count``, ``done_reason``…)."""

    def __init__(self, inner: Callable[[str, dict[str, Any], float], Any] = urllib_transport):
        self.inner = inner
        self.last: dict[str, Any] | None = None

    def __call__(self, url: str, payload: dict[str, Any], timeout: float) -> Any:
        self.last = None
        reply = self.inner(url, payload, timeout)
        self.last = reply if isinstance(reply, dict) else None
        return reply


_KEPT = ("done_reason", "eval_count", "prompt_eval_count", "total_duration", "load_duration")


class Caller:
    """A client that retries, times each call and keeps the last :class:`Reply`.

    It satisfies ``LLMClient``, so the grounded explainer can be given it: the explainer then asks
    through it, and the raw answer stays in ``last`` even when the guardrails send the explainer
    back to the template. ``generate`` raises after the last attempt, as a client does.
    """

    def __init__(
        self,
        client: Any,
        transport: RecordingTransport | None = None,
        *,
        timeout: float = TIMEOUT_SECONDS,
        retries: int = RETRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.transport = transport
        self.timeout = timeout
        self.retries = retries
        self.clock = clock
        self.last: Reply | None = None
        self.calls = 0

    def generate(self, prompt: str, *, system: str, timeout: float | None = None) -> str:
        del timeout  # the explainer's own limit; this caller has one for the whole run
        started = self.clock()
        error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            self.calls += 1
            try:
                text = self.client.generate(prompt, system=system, timeout=self.timeout)
            except (LLMError, TimeoutError, OSError) as failure:
                error = failure
                continue
            reply = (self.transport.last if self.transport else None) or {}
            self.last = Reply(
                text,
                None,
                attempt,
                self.clock() - started,
                {k: reply[k] for k in _KEPT if k in reply},
                prompt,
            )
            return text
        self.last = Reply(
            None,
            f"{type(error).__name__}: {error}",
            self.retries + 1,
            self.clock() - started,
            prompt=prompt,
        )
        raise LLMError(self.last.error or "no answer")

    def ask(self, prompt: str, system: str) -> Reply:
        """The reply to ``prompt``, never raising: a failure is a :class:`Reply` with an error."""
        with contextlib.suppress(LLMError):
            self.generate(prompt, system=system)
        assert self.last is not None
        return self.last


class StatusExplainer:
    """The grounded explainer, remembering why the template stood in (the pipeline keeps only
    that it did)."""

    def __init__(self, inner: LLMExplainer) -> None:
        self.inner = inner
        self.last: tuple[Any, str | None] | None = None

    def explain(self, case: PatientCase, candidates: list) -> Any:
        return self.explain_with_status(case, candidates)[0]

    def explain_with_status(self, case: PatientCase, candidates: list) -> tuple[Any, str | None]:
        self.last = self.inner.explain_with_status(case, candidates)
        return self.last


# --------------------------------------------------------------------------------------------- #
# The work
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Item:
    """One call to make: an arm, a case at a level, and what is known about the truth."""

    arm: str
    case: PatientCase
    level: float
    source: str  # "golden" | "validation"
    truth: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.arm}|{self.case.case_id}|{self.level}"


def load_golden(path: Path = GOLDEN) -> list[tuple[PatientCase, dict[str, Any]]]:
    """The golden cases and their expectations, from the fixture the tests use."""
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
    return [(PatientCase(**c["case"]), {"expected": c.get("expect", {})}) for c in cases]


def work_items(
    arms: Sequence[str],
    golden: Sequence[tuple[PatientCase, dict[str, Any]]],
    validation: dict[float, Sequence[tuple[PatientCase, dict[str, Any]]]],
) -> Iterator[Item]:
    """Every call to make, the 100% level first (all arms, case by case, so a run that stops has
    paired data), then 50%, then 25%."""
    for level in (1.0, 0.5, 0.25):
        pairs: list[tuple[str, list[tuple[PatientCase, dict[str, Any]]]]] = []
        if level == 1.0:
            pairs.append(("golden", list(golden)))
        pairs.append(("validation", list(validation.get(level, ()))))
        for source, cases in pairs:
            for case, truth in cases:
                for arm in ARMS:
                    if arm in arms and level in LEVELS_BY_ARM[arm]:
                        yield Item(arm, case, level, source, truth)


def server_info(
    base_url: str, model: str, fetch: Callable[[str], Any] | None = None
) -> dict[str, Any]:
    """Ollama's version and the model's digest, quantisation and size, for ``run.json``.

    Raises:
        SystemExit: if the server does not answer or does not have ``model``.
    """

    def get(path: str) -> Any:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=30) as reply:
            return json.loads(reply.read().decode("utf-8"))

    fetch = fetch or get
    try:
        version = fetch("/api/version").get("version")
        models = fetch("/api/tags").get("models", [])
    except (OSError, ValueError) as error:
        raise SystemExit(f"Ollama does not answer at {base_url}: {error}") from error
    found = [m for m in models if model in (m.get("name"), m.get("model"))]
    if not found:
        names = [m.get("name") for m in models]
        raise SystemExit(f"Ollama has no model {model!r} (it has {names}): `ollama pull {model}`")
    entry = found[0]
    return {
        "version": version,
        "model": model,
        "digest": entry.get("digest"),
        "size_bytes": entry.get("size"),
        "details": entry.get("details", {}),
    }


class Experiment:
    """Runs the arms over the items, appending a record per call to ``results.jsonl``.

    Args:
        caller: Asks the model (:class:`Caller`).
        pipeline: A ``DiagnosisPipeline`` (or None): A0+LLM runs it with the grounded explainer.
        retriever: The evidence retriever B4 asks; None, or a degraded one, means no B4.
        ranker: The pipeline's ranker, for the template inside the grounded explainer.
        out: The folder the results go to.
    """

    def __init__(
        self,
        caller: Caller,
        *,
        pipeline: Any | None,
        retriever: Any | None,
        ranker: Any | None,
        out: Path,
        timeout: float = TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.caller = caller
        self.pipeline = pipeline
        self.retriever = retriever
        self.ranker = ranker
        self.out = out
        self.clock = clock
        self.explainer = StatusExplainer(
            # The explainer discards an answer that took longer than its limit; the caller has
            # its own (with retries), so this one only has to be longer than all of them.
            LLMExplainer(caller, ranker=ranker, timeout_seconds=timeout * (RETRIES + 2) + 10)
        )
        if pipeline is not None:
            pipeline.explainer = self.explainer
        self.results = out / "results.jsonl"

    # -- the arms' availability ------------------------------------------------------------- #

    @property
    def has_retrieval(self) -> bool:
        return self.retriever is not None and not getattr(self.retriever, "degraded", False)

    def available(self, arms: Iterable[str]) -> list[str]:
        """The arms that can run: B4 needs the retriever; A0+LLM needs that, the pipeline and the
        configured ML ranker (without it the pipeline is not A0)."""
        out = []
        for arm in arms:
            if arm == B4 and not self.has_retrieval:
                continue
            if arm == A0_LLM and (
                self.pipeline is None
                or not self.has_retrieval
                or getattr(self.ranker, "degraded", False)
            ):
                continue
            out.append(arm)
        return out

    # -- one record ------------------------------------------------------------------------- #

    def evidence_for(self, case: PatientCase) -> dict[str, list[Evidence]]:
        """B4's retrieval: the best passage for each of the 14 conditions in this case."""
        return {
            c.id: self.retriever.retrieve(c.id, case, k=PASSAGES_PER_CONDITION) for c in CONDITIONS
        }

    def run_ranking(self, item: Item) -> dict[str, Any]:
        evidence = self.evidence_for(item.case) if item.arm == B4 else None
        started = self.clock()
        prompt = build_prompt(item.arm, item.case, evidence)
        system = SYSTEM_PROMPTS[item.arm]
        reply = self.caller.ask(prompt, system)
        if reply.text is None:
            parsed = ParsedAnswer((), "", flags=("llm_error",))
        else:
            parsed = parse_answer(reply.text)
        stats = None
        if parsed.explanation:
            stats = claim_stats(
                parsed.explanation, given_result(item.case, evidence), reference_case(item.case)
            ).as_dict()
        return {
            "prompt": prompt,
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "raw": reply.text,
            "parsed": {
                "ranking": list(parsed.ranking),
                "explanation": parsed.explanation,
                "unmatched": list(parsed.unmatched),
                "duplicates": parsed.duplicates,
                "flags": list(parsed.flags),
            },
            "claims": stats,
            "passages": {
                cid: [e.citation for e in found] for cid, found in (evidence or {}).items()
            },
            "prep_s": round(self.clock() - started - reply.elapsed_s, 3),
            "reply": _reply_dict(reply),
        }

    def run_grounded(self, item: Item) -> dict[str, Any]:
        assert self.pipeline is not None
        self.caller.last = None
        self.explainer.last = None
        result = self.pipeline.run(item.case)
        reply = self.caller.last
        fallback = self.explainer.last[1] if self.explainer.last else "no call"
        ref = reference_case(item.case)
        raw = " ".join(reply.text.split()) if reply and reply.text else None
        delivered = result.explanation
        return {
            "prompt": reply.prompt if reply else None,
            "raw": raw,
            "claims": claim_stats(raw, result, ref).as_dict() if raw else None,
            "delivered": {
                "text": delivered.text if delivered else None,
                "grounded": bool(delivered and delivered.grounded),
                "unsupported_claims": list(delivered.unsupported_claims) if delivered else [],
                "citations": list(delivered.citations) if delivered else [],
            },
            "delivered_claims": (
                claim_stats(delivered.text, result, ref).as_dict()
                if delivered and delivered.text
                else None
            ),
            "fallback_reason": fallback,
            "degraded": list(result.degraded_components),
            "ranking": [c.condition_id for c in result.candidates],
            "ranking_via": "crosswalk inversion (R-17): not EXP-012's A0 ranking",
            "red_flags": list(result.red_flags),
            "tests": list(result.suggested_next_tests),
            "passages": {
                c.condition_id: [e.citation for e in c.evidence]
                for c in result.candidates
                if c.evidence
            },
            "reply": _reply_dict(reply) if reply else None,
        }

    def run_item(self, item: Item) -> dict[str, Any]:
        started = self.clock()
        body = self.run_grounded(item) if item.arm == A0_LLM else self.run_ranking(item)
        return {
            "arm": item.arm,
            "case_id": item.case.case_id,
            "level": item.level,
            "source": item.source,
            "truth": item.truth,
            "case_text": case_text(item.case),
            **body,
            "wall_s": round(self.clock() - started, 3),
        }

    # -- the run ---------------------------------------------------------------------------- #

    def done(self) -> set[str]:
        """The keys already in ``results.jsonl`` (resuming)."""
        if not self.results.exists():
            return set()
        keys = set()
        for line in self.results.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(ValueError, KeyError):
                row = json.loads(line)
                keys.add(f"{row['arm']}|{row['case_id']}|{row['level']}")
        return keys

    def run(
        self, items: Iterable[Item], *, progress: Callable[[str], None] = print
    ) -> dict[str, int]:
        """Make every call not already in ``results.jsonl``.

        A run that cannot reach the model is not a result: after :data:`MAX_CONSECUTIVE_FAILURES`
        failed calls in a row it stops, takes those records out of the file (into
        ``failed_tail.jsonl``, for the log) so that a resume asks them again, and reports
        ``stopped``. Failures that are isolated stay in the file as results, and are counted.
        """
        self.out.mkdir(parents=True, exist_ok=True)
        done = self.done()
        counts = {"planned": 0, "skipped": 0, "run": 0, "stopped": 0}
        streak: list[str] = []
        with open(self.results, "a", encoding="utf-8") as handle:
            for item in items:
                counts["planned"] += 1
                if item.key in done:
                    counts["skipped"] += 1
                    continue
                record = self.run_item(item)
                line = json.dumps(record, ensure_ascii=False)
                handle.write(line + "\n")
                handle.flush()
                counts["run"] += 1
                streak = [*streak, line] if (record.get("reply") or {}).get("error") else []
                if len(streak) >= MAX_CONSECUTIVE_FAILURES:
                    counts["stopped"] = 1
                    break
                if counts["run"] % 10 == 0:
                    progress(f"  {counts['run']} calls made ({item.arm} {item.case.case_id})")
        if streak and counts["stopped"]:
            self._take_out(streak)
        return counts

    def _take_out(self, tail: list[str]) -> None:
        """Remove the last ``len(tail)`` lines of ``results.jsonl`` (they are ``tail``) and keep
        them in ``failed_tail.jsonl``."""
        lines = self.results.read_text(encoding="utf-8").splitlines()
        assert lines[-len(tail) :] == tail
        self.results.write_text(
            "".join(line + "\n" for line in lines[: -len(tail)]), encoding="utf-8"
        )
        with open(self.out / "failed_tail.jsonl", "a", encoding="utf-8") as handle:
            handle.write("".join(line + "\n" for line in tail))


def _reply_dict(reply: Reply) -> dict[str, Any]:
    return {
        "error": reply.error,
        "attempts": reply.attempts,
        "elapsed_s": round(reply.elapsed_s, 3),
        **reply.ollama,
        "context_full": _context_full(reply.ollama),
    }


# --------------------------------------------------------------------------------------------- #
# Finding what Kaggle put in /kaggle/input
# --------------------------------------------------------------------------------------------- #


def find_bundle(source: Path, scratch: Path) -> Path | None:
    """The retrieval bundle under ``source``: a folder with ``manifest.json`` and the bundle's
    files, or a zip of one (unzipped under ``scratch``). None if there is none: B4 and A0+LLM then
    do not run, and B3 and B3r do.

    Kaggle may hand a notebook's output over as the folder it wrote or as the zip, and a dataset
    unpacked or not, so both are looked for. Whether the bundle is *sound* is ``load_bundle``'s
    to say (its SHA-256 checks), not this function's.
    """
    for manifest in sorted(source.rglob("manifest.json")):
        if all((manifest.parent / name).exists() for name in REQUIRED_FILES):
            return manifest.parent
    for archive in sorted(source.rglob("*.zip")):
        try:
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
                if not any(n.endswith("manifest.json") for n in names) or not any(
                    n.endswith("passages.jsonl") for n in names
                ):
                    continue
                target = scratch / archive.stem
                z.extractall(target)
        except (zipfile.BadZipFile, OSError):
            continue
        found = find_bundle(target, scratch)
        if found is not None:
            return found
    return None


def place_bodhi_s(source: Path, target: Path) -> bool:
    """Copy BODHI-S's two files (``triples.jsonl``, ``nl_facts.jsonl``) from a private dataset
    under ``source`` into ``target``. BODHI-S is CC-BY-NC and never in the repository (``docs/03``
    §1.2), so the graph on Kaggle has its enrichment only if the owner adds it. True if both
    files are now in ``target``."""
    for triples in sorted(source.rglob("triples.jsonl")):
        facts = triples.parent / "nl_facts.jsonl"
        if facts.exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy(triples, target / "triples.jsonl")
            shutil.copy(facts, target / "nl_facts.jsonl")
            return True
    return False


def previous_results(source: Path, out: Path) -> int:
    """Resume: put the fullest ``results.jsonl`` that an earlier version of the notebook left
    under ``source`` (with its ``run.json`` beside it) into ``out``, unless ``out`` has as many
    lines already. Returns the number of lines in ``out``'s file afterwards."""

    def lines(path: Path) -> int:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    target = out / "results.jsonl"
    have = lines(target) if target.exists() else 0
    for found in sorted(source.rglob("results.jsonl")):
        if (found.parent / "run.json").exists() and lines(found) > have:
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(found, target)
            have = lines(target)
    return have


# --------------------------------------------------------------------------------------------- #
# The validation patients
# --------------------------------------------------------------------------------------------- #


def validation_cases(
    interim: Path, raw: Path, labels: dict[str, str] | None, size: int
) -> tuple[dict[float, list[tuple[PatientCase, dict[str, Any]]]], dict[str, Any]]:
    """The sample's patients at each level, as cases with their truth, and what to record about
    the draw. Stops if the masks do not reproduce docs/05 §3.7's digests."""
    import evaluate_reduced_evidence as ere
    import pandas as pd

    from src.ddxplus import load_evidence_specs
    from src.medical_kg.crosswalk import case_from_ddxplus
    from src.ml.features import evidence_codes_in_scope

    specs = load_evidence_specs(raw / "release_evidences.json")
    conditions = interim / "ddxplus_chestpain_conditions.json"
    from src.ddxplus import code_sort_key

    codes = sorted(evidence_codes_in_scope(conditions), key=code_sort_key)
    validate = pd.read_parquet(interim / "ddxplus_chestpain_validate.parquet", columns=ere.COLUMNS)
    frames = ere.draw(validate, codes, specs)
    drawn = ere.digests(frames)
    if drawn["tokens"] != ere.TOKEN_DIGEST or drawn["asked"] != ere.ASKED_DIGEST:
        raise SystemExit(
            f"the masks do not reproduce their digests ({drawn}): the run is invalid (docs/05 §3.7)"
        )
    ids = sample_ids(validate["case_id"], size)
    by_level: dict[float, list[tuple[PatientCase, dict[str, Any]]]] = {}
    for level, frame in frames.items():
        rows = frame.set_index("case_id").loc[ids].reset_index().to_dict("records")
        by_level[level] = [
            (
                case_from_ddxplus(row, specs, labels),
                {
                    "true_condition": row["label_condition_id"],
                    "differential": sorted(
                        e["condition_id"] for e in row["label_differential"] if e["condition_id"]
                    ),
                    "differential_size": len(row["label_differential"]),
                },
            )
            for row in rows
        ]
    truth = {c.case_id: t["true_condition"] for c, t in by_level[1.0]}
    composition: dict[str, int] = {}
    for condition in truth.values():
        composition[condition] = composition.get(condition, 0) + 1
    info = {
        "ids": ids,
        "digest": sample_digest(ids),
        "composition": dict(sorted(composition.items())),
        "must_not_miss": sum(BY_ID[c].is_must_not_miss for c in truth.values()),
        "mask_digests": drawn,
    }
    return by_level, info


# --------------------------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------------------------- #


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the folder for results.jsonl")
    parser.add_argument("--bundle", type=Path, help="the retrieval bundle (B4 and A0+LLM need it)")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--config", type=Path, help="configs/config.yaml (default)")
    parser.add_argument(
        "--graph", default="networkx", help="the graph backend (networkx on Kaggle)"
    )
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--golden-only", action="store_true", help="no validation patients")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--smoke", action="store_true", help="2 golden cases, 2 patients, fail fast"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # cp1252 consoles (CLAUDE.md gotcha)
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    args = parse_args(argv)
    from _config import build_pipeline, load_config

    info = server_info(args.base_url, args.model)
    print(f"Ollama {info['version']}, {args.model} {str(info['digest'])[:12]}")

    config = load_config(args.config)
    config.setdefault("llm", {})["enabled"] = False  # the experiment builds its own explainer
    if args.bundle:
        config.setdefault("retrieval", {})["bundle"] = str(args.bundle.resolve())
    built = build_pipeline(config, args.graph)
    print("\n".join(f"  {line}" for line in built["about"].values()))

    transport = RecordingTransport()
    client = OllamaClient(
        args.model,
        base_url=args.base_url,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        transport=transport,
        seed=SEED,
        num_ctx=NUM_CTX,
    )
    caller = Caller(client, transport)
    experiment = Experiment(
        caller,
        pipeline=built["pipeline"],
        retriever=built["retriever"],
        ranker=built["ranker"],
        out=args.out,
    )
    arms = experiment.available(args.arms)
    skipped = [a for a in args.arms if a not in arms]
    if skipped:
        print(f"  not run, no retrieval bundle: {skipped}")

    golden = load_golden()
    validation: dict[float, list] = {}
    sample: dict[str, Any] = {}
    if not args.golden_only:
        validation, sample = validation_cases(
            args.interim, args.raw_dir, built["labels"], args.sample_size
        )
        print(f"sample {sample['digest']}: {sample['composition']}")
    if args.smoke:
        golden = [g for g in golden if g[0].case_id in ("GC-001", "GC-004")]
        validation = {lv: cases[:2] for lv, cases in validation.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = getattr(getattr(built["retriever"], "bundle", None), "manifest", None) or {}
    run_info = {
        "experiment": "EXP-009/EXP-010",
        "commit": _git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ollama": info,
        "decoding": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_predict": MAX_TOKENS,
            "num_ctx": NUM_CTX,
            "timeout_s": TIMEOUT_SECONDS,
            "retries": RETRIES,
        },
        "arms_planned": list(args.arms),
        "arms_run": arms,
        "prompts_sha256": {
            **{a: hashlib.sha256(s.encode()).hexdigest() for a, s in SYSTEM_PROMPTS.items()},
            A0_LLM: hashlib.sha256(EXPLAINER_SYSTEM.encode()).hexdigest(),
        },
        "sample": {k: v for k, v in sample.items() if k != "ids"} | {"seed": SAMPLE_SEED},
        "pipeline": built["about"],
        "bundle": {
            k: manifest.get(k) for k in ("created", "commit", "counts", "licences", "screening")
        },
        "smoke": args.smoke,
    }
    (args.out / "run.json").write_text(
        json.dumps(run_info, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    if sample:
        (args.out / "sample_ids.json").write_text(json.dumps(sample["ids"]), encoding="utf-8")

    items = list(work_items(arms, golden, validation))
    print(f"{len(items)} calls planned for {arms}")
    started = time.monotonic()
    counts = experiment.run(items)
    print(f"{counts} in {(time.monotonic() - started) / 60:.1f} min; {caller.calls} requests")
    if counts["stopped"]:
        print(
            f"STOPPED: {MAX_CONSECUTIVE_FAILURES} calls failed in a row, so the model is not "
            "answering. Those records were set aside (failed_tail.jsonl); run again to resume."
        )
        return 3
    if args.smoke:
        return smoke_verdict(experiment, caller, args.out)
    return 0


def smoke_verdict(experiment: Experiment, caller: Caller, out: Path) -> int:
    """Fail fast: a model that errors, or whose ranking never parses, stops the notebook here."""
    rows = [
        json.loads(line) for line in experiment.results.read_text(encoding="utf-8").splitlines()
    ]
    problems = []
    errors = [r for r in rows if (r.get("reply") or {}).get("error")]
    if errors:
        problems.append(f"{len(errors)} call(s) failed: {errors[0]['reply']['error']}")
    ranking = [r for r in rows if r["arm"] in (B3, B3R, B4)]
    if ranking and not any(r["parsed"]["ranking"] for r in ranking):
        problems.append("no ranking answer parsed: " + repr((ranking[0].get("raw") or "")[:300]))
    cut = [r for r in rows if (r.get("reply") or {}).get("context_full")]
    if cut:
        problems.append(f"{len(cut)} prompt(s) filled the context window ({NUM_CTX} tokens)")
    probe = next((r for r in rows if r["arm"] == B3 and r["raw"]), None)
    same = None
    if probe is not None:
        again = caller.ask(probe["prompt"], SYSTEM_PROMPTS[B3])
        same = again.text == probe["raw"]
    report = {"rows": len(rows), "problems": problems, "same_text_twice": same}
    (out / "smoke.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report))
    return 2 if problems else 0


def _context_full(reply: dict[str, Any]) -> bool:
    return (reply.get("prompt_eval_count") or 0) + (reply.get("eval_count") or 0) >= NUM_CTX


if __name__ == "__main__":
    sys.exit(main())
