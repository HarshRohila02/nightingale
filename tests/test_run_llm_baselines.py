"""The runner of the LLM trial (scripts/run_llm_baselines.py; EXP-009, EXP-010).

No model is called and no test opens a network connection: the model is a scripted fake, the
pipeline is the stub pipeline, and the retriever hands out one invented passage per condition. What
must hold: every arm gets the prompt the log pre-registered; every failure is recorded and counted;
a run that stops resumes; the grounded explainer's raw answer survives the guardrails that send it
back to the template; and the smoke test stops a run that cannot work.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_llm_baselines as run  # noqa: E402

from src.conditions import BY_ID, CONDITIONS  # noqa: E402
from src.contracts import Evidence, PatientCase  # noqa: E402
from src.llm.baselines import A0_LLM, B3, B3R, B4, SYSTEM_PROMPTS  # noqa: E402
from src.llm.client import LLMError, OllamaClient  # noqa: E402
from src.llm.explainer import SYSTEM as EXPLAINER_SYSTEM  # noqa: E402
from src.pipeline import DiagnosisPipeline  # noqa: E402
from src.stubs import ConstantRanker, InMemoryGraphStore  # noqa: E402

ANSWER = (
    "RANKING:\n1. Possible NSTEMI / STEMI\n2. Unstable angina\n3. GERD\n"
    "EXPLANATION:\nChest pain and diaphoresis. Pneumonia is unlikely."
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


class Scripted:
    """A client that answers by system prompt, keeps every prompt, and can fail on demand."""

    def __init__(self, answers=None, fail_first: int = 0):
        self.answers = answers or {}
        self.fail_first = fail_first
        self.calls: list[tuple[str, str]] = []

    def generate(self, prompt: str, *, system: str, timeout: float) -> str:
        self.calls.append((prompt, system))
        if self.fail_first:
            self.fail_first -= 1
            raise LLMError("busy")
        return self.answers.get(system, ANSWER)


class Passages:
    """A retriever with one invented passage under every condition."""

    degraded = False

    def retrieve(self, condition: str, case: PatientCase, k: int = 5) -> list[Evidence]:
        cid = (
            condition
            if condition in BY_ID
            else next(c.id for c in CONDITIONS if c.label == condition)
        )
        number = [c.id for c in CONDITIONS].index(cid) + 1
        return [
            Evidence(
                source="PMC",
                title=f"Article {number}",
                passage=f"{BY_ID[cid].label} is assessed with an ECG.",
                citation=f"PMCID:PMC{number:07d}",
            )
        ][:k]


class Degraded(Passages):
    degraded = True


def experiment(tmp_path, client, retriever=None, *, ranker=None):
    ranker = ranker or ConstantRanker()
    pipeline = DiagnosisPipeline(ranker, InMemoryGraphStore(), retriever)
    return run.Experiment(
        run.Caller(client, retries=2),
        pipeline=pipeline,
        retriever=retriever,
        ranker=ranker,
        out=tmp_path,
    )


def golden_items(arms, level=1.0):
    return [
        run.Item(arm, case, level, "golden", truth)
        for case, truth in run.load_golden()[:1]
        for arm in arms
    ]


# --------------------------------------------------------------------------------------------- #
# Calling the model
# --------------------------------------------------------------------------------------------- #


def test_a_call_that_fails_is_tried_again_and_the_attempts_are_counted():
    caller = run.Caller(Scripted(fail_first=2), retries=2)
    reply = caller.ask("p", "s")
    assert reply.text == ANSWER and reply.error is None and reply.attempts == 3
    assert caller.calls == 3


def test_a_call_that_never_succeeds_is_a_reply_with_an_error_and_generate_raises():
    caller = run.Caller(Scripted(fail_first=99), retries=1)
    reply = caller.ask("p", "s")
    assert reply.text is None and reply.attempts == 2 and "LLMError" in reply.error
    with pytest.raises(LLMError):
        caller.generate("p", system="s")


def test_what_ollama_reports_is_kept_and_a_full_context_is_flagged():
    def transport(reply):
        return lambda url, payload, timeout: reply

    for prompt_tokens, full in ((1000, False), (8100, True)):
        recording = run.RecordingTransport(
            transport(
                {
                    "response": "x",
                    "done": True,
                    "done_reason": "stop",
                    "eval_count": 100,
                    "prompt_eval_count": prompt_tokens,
                    "unrelated": 1,
                }
            )
        )
        client = OllamaClient("m", transport=recording, seed=42, num_ctx=run.NUM_CTX)
        caller = run.Caller(client, recording)
        reply = caller.ask("p", "s")
        assert reply.ollama == {
            "done_reason": "stop",
            "eval_count": 100,
            "prompt_eval_count": prompt_tokens,
        }
        assert run._reply_dict(reply)["context_full"] is full


# --------------------------------------------------------------------------------------------- #
# The work
# --------------------------------------------------------------------------------------------- #


def test_the_golden_cases_load_with_what_they_expect():
    cases = run.load_golden()
    assert [c.case_id for c, _ in cases] == ["GC-001", "GC-002", "GC-003", "GC-004"]
    assert cases[0][1]["expected"]["must_include"] == ["COND:nstemi_stemi"]


def test_work_is_planned_at_each_arms_levels_with_100_percent_first():
    golden = run.load_golden()
    template = golden[0][0]
    patients = {
        lv: [(template.model_copy(update={"case_id": f"P{lv}-{i}"}), {}) for i in range(2)]
        for lv in (1.0, 0.5, 0.25)
    }
    items = list(run.work_items([B3, B3R, B4, A0_LLM], golden, patients))
    by_arm = {a: [i for i in items if i.arm == a] for a in (B3, B3R, B4, A0_LLM)}
    assert {a: len(v) for a, v in by_arm.items()} == {
        B3: 4 + 6,
        B3R: 4 + 2,
        B4: 4 + 6,
        A0_LLM: 4 + 2,
    }
    assert [i.level for i in items] == sorted((i.level for i in items), reverse=True)
    assert {i.source for i in by_arm[B3R]} == {"golden", "validation"}
    assert all(i.level == 1.0 for i in by_arm[B3R] + by_arm[A0_LLM])
    # case by case, so a run that stops has paired data: one case's arms are next to each other
    first = [i for i in items if i.case.case_id == "GC-001"]
    assert [i.arm for i in first] == [B3, B3R, B4, A0_LLM]
    assert len({i.key for i in items}) == len(items)


def test_the_arms_that_need_retrieval_are_left_out_without_it(tmp_path):
    arms = [B3, B3R, B4, A0_LLM]
    assert experiment(tmp_path, Scripted(), None).available(arms) == [B3, B3R]
    assert experiment(tmp_path, Scripted(), Degraded()).available(arms) == [B3, B3R]
    assert experiment(tmp_path, Scripted(), Passages()).available(arms) == arms


def test_without_the_configured_ranker_a0_llm_is_not_run(tmp_path):
    class Broken:
        degraded = True

        def score(self, case):
            return {}

    assert experiment(tmp_path, Scripted(), Passages(), ranker=Broken()).available(
        [B3, A0_LLM]
    ) == [B3]


# --------------------------------------------------------------------------------------------- #
# The records
# --------------------------------------------------------------------------------------------- #


def test_a_ranking_arm_sends_the_registered_prompt_and_records_the_reading(tmp_path):
    client = Scripted()
    exp = experiment(tmp_path, client)
    case, truth = run.load_golden()[0]
    record = exp.run_item(run.Item(B3, case, 1.0, "golden", truth))
    prompt, system = client.calls[0]
    assert system == SYSTEM_PROMPTS[B3] and prompt == record["prompt"]
    assert record["parsed"]["ranking"] == ["COND:nstemi_stemi", "COND:unstable_angina", "COND:gerd"]
    assert record["claims"]["claims"] == 2 and record["claims"]["unsupported"] == 1
    assert record["claims"]["untraced"] == ["Pneumonia is unlikely."]
    assert record["case_text"][0] == "Patient: 58 years old, male."
    assert record["reply"]["error"] is None and record["passages"] == {}
    assert record["system_sha256"] and record["wall_s"] >= 0


def test_a_failed_call_is_recorded_not_dropped(tmp_path):
    exp = experiment(tmp_path, Scripted(fail_first=99))
    case, truth = run.load_golden()[0]
    record = exp.run_item(run.Item(B3, case, 1.0, "golden", truth))
    assert record["raw"] is None and record["claims"] is None
    assert record["parsed"]["ranking"] == [] and "llm_error" in record["parsed"]["flags"]
    assert record["reply"]["error"] and record["reply"]["attempts"] == 3


def test_an_answer_that_does_not_parse_is_recorded_with_its_text(tmp_path):
    exp = experiment(tmp_path, Scripted({SYSTEM_PROMPTS[B3]: "I would rather not say."}))
    case, truth = run.load_golden()[0]
    record = exp.run_item(run.Item(B3, case, 1.0, "golden", truth))
    assert record["raw"] == "I would rather not say."
    assert record["parsed"]["ranking"] == [] and "no_ranking_header" in record["parsed"]["flags"]
    assert record["claims"] is None


def test_b4_is_given_one_passage_for_each_condition_and_cites_only_those(tmp_path):
    answer = (
        "RANKING:\n1. GERD\nEXPLANATION:\nGERD is assessed with an ECG [PMCID:PMC0000012]. "
        "Chest pain [PMCID:PMC9999999]."
    )
    client = Scripted({SYSTEM_PROMPTS[B4]: answer})
    exp = experiment(tmp_path, client, Passages())
    case, truth = run.load_golden()[0]
    record = exp.run_item(run.Item(B4, case, 1.0, "golden", truth))
    assert len(record["passages"]) == 14 and all(len(v) == 1 for v in record["passages"].values())
    assert record["prompt"].count("<passage") == 14
    assert record["claims"]["cited"] == ["PMCID:PMC0000012", "PMCID:PMC9999999"]
    assert record["claims"]["invalid_citations"] == ["PMCID:PMC9999999"]


def test_a0_llm_keeps_the_raw_answer_when_the_guardrails_send_the_template_back(tmp_path):
    ungrounded = "Pneumonia is the most likely cause."
    exp = experiment(tmp_path, Scripted({EXPLAINER_SYSTEM: ungrounded}), Passages())
    case, truth = run.load_golden()[0]
    record = exp.run_item(run.Item(A0_LLM, case, 1.0, "golden", truth))
    assert record["raw"] == ungrounded
    assert record["prompt"].startswith("Case GC-001. The candidates, best first:")  # kept to audit
    assert record["claims"]["unsupported"] == 1
    assert record["fallback_reason"] and "not grounded" in record["fallback_reason"]
    assert "llm" in record["degraded"]
    assert record["delivered"]["text"] != ungrounded
    assert record["delivered_claims"]["unsupported"] == 0  # the template always traces
    assert not record["delivered_claims"]["treatment_language"]
    assert record["ranking"] and len(record["ranking"]) == 14
    assert "not EXP-012's A0 ranking" in record["ranking_via"]


def test_a0_llm_delivers_a_grounded_answer_as_it_was_written(tmp_path):
    grounded = "Possible NSTEMI / STEMI. Chest pain."
    exp = experiment(tmp_path, Scripted({EXPLAINER_SYSTEM: grounded}), Passages())
    case, truth = run.load_golden()[0]
    record = exp.run_item(run.Item(A0_LLM, case, 1.0, "golden", truth))
    assert record["fallback_reason"] is None and "llm" not in record["degraded"]
    assert record["delivered"]["text"] == grounded and record["delivered"]["grounded"]
    assert record["claims"]["unsupported"] == 0
    assert {c for c in record["passages"]} <= {c.id for c in CONDITIONS}


def test_a_run_writes_one_line_per_call_and_resumes_where_it_stopped(tmp_path):
    arms = [B3, B3R]
    items = golden_items(arms)
    first = experiment(tmp_path, Scripted())
    assert first.run(items, progress=lambda _: None) == {
        "planned": 2,
        "skipped": 0,
        "run": 2,
        "stopped": 0,
    }
    assert first.done() == {i.key for i in items}
    lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["arm"] for line in lines] == arms

    client = Scripted()
    again = experiment(tmp_path, client)
    more = golden_items([B3, B3R], level=0.5)[:1]
    counts = again.run([*items, *more], progress=lambda _: None)
    assert counts == {"planned": 3, "skipped": 2, "run": 1, "stopped": 0}
    assert len(client.calls) == 1
    assert len((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 3


# --------------------------------------------------------------------------------------------- #
# The server and the smoke test
# --------------------------------------------------------------------------------------------- #


def fetcher(models, version="0.99.0"):
    def fetch(path):
        if path == "/api/version":
            return {"version": version}
        return {"models": models}

    return fetch


def test_the_server_info_names_the_version_and_the_models_digest():
    info = run.server_info(
        "http://x",
        "llama3.1:8b",
        fetcher(
            [
                {
                    "name": "llama3.1:8b",
                    "digest": "abc",
                    "size": 5,
                    "details": {"quantization_level": "Q4_K_M"},
                }
            ]
        ),
    )
    assert info["version"] == "0.99.0" and info["digest"] == "abc"
    assert info["details"]["quantization_level"] == "Q4_K_M"


def test_a_missing_model_or_server_stops_the_run_with_the_fix_to_try():
    with pytest.raises(SystemExit, match="ollama pull llama3.1:8b"):
        run.server_info("http://x", "llama3.1:8b", fetcher([{"name": "other:1b"}]))

    def down(path):
        raise OSError("refused")

    with pytest.raises(SystemExit, match="does not answer"):
        run.server_info("http://x", "llama3.1:8b", down)


def smoke(tmp_path, client, retriever=None):
    exp = experiment(tmp_path, client, retriever)
    exp.run(golden_items([B3, B3R], 1.0), progress=lambda _: None)
    return exp, run.smoke_verdict(exp, exp.caller, tmp_path)


def test_a_smoke_test_that_works_passes_and_records_whether_the_text_repeats(tmp_path):
    exp, code = smoke(tmp_path, Scripted())
    report = json.loads((tmp_path / "smoke.json").read_text(encoding="utf-8"))
    assert code == 0 and report["problems"] == [] and report["same_text_twice"] is True


def test_a_smoke_test_stops_a_model_that_errors(tmp_path):
    _, code = smoke(tmp_path, Scripted(fail_first=99))
    assert code == 2
    assert "failed" in json.loads((tmp_path / "smoke.json").read_text("utf-8"))["problems"][0]


def test_a_smoke_test_stops_when_no_answer_parses(tmp_path):
    _, code = smoke(tmp_path, Scripted({SYSTEM_PROMPTS[B3]: "no", SYSTEM_PROMPTS[B3R]: "no"}))
    assert code == 2
    assert (
        "no ranking answer parsed"
        in json.loads((tmp_path / "smoke.json").read_text("utf-8"))["problems"][0]
    )


def test_the_arguments_default_to_every_arm_and_refuse_an_unknown_one():
    args = run.parse_args(["--out", "x"])
    assert args.arms == ["B3", "B3r", "B4", "A0+LLM"] and args.model == "llama3.1:8b"
    assert args.bundle is None and not args.smoke and args.graph == "networkx"
    with pytest.raises(SystemExit):
        run.parse_args(["--out", "x", "--arms", "B9"])


# --------------------------------------------------------------------------------------------- #
# What Kaggle puts in /kaggle/input
# --------------------------------------------------------------------------------------------- #


def write_bundle(folder: Path, missing: str | None = None) -> Path:
    from src.rag.bundle import REQUIRED_FILES

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text("{}", encoding="utf-8")
    for name in REQUIRED_FILES:
        if name != missing:
            (folder / name).write_text("x", encoding="utf-8")
    return folder


def test_a_bundle_folder_is_found_wherever_kaggle_nested_it(tmp_path):
    bundle = write_bundle(tmp_path / "input" / "corpus-notebook" / "retrieval_corpus")
    assert run.find_bundle(tmp_path / "input", tmp_path / "scratch") == bundle


def test_a_folder_missing_one_of_the_files_is_not_a_bundle(tmp_path):
    write_bundle(tmp_path / "input" / "b", missing="passages.jsonl")
    assert run.find_bundle(tmp_path / "input", tmp_path / "scratch") is None
    assert run.find_bundle(tmp_path / "nothing", tmp_path / "scratch") is None


def test_a_zipped_bundle_is_unzipped_and_found(tmp_path):
    import shutil

    made = write_bundle(tmp_path / "made" / "retrieval_corpus")
    (tmp_path / "input").mkdir()
    shutil.make_archive(str(tmp_path / "input" / "retrieval_corpus"), "zip", tmp_path / "made")
    found = run.find_bundle(tmp_path / "input", tmp_path / "scratch")
    assert found is not None and found.name == made.name
    assert found.is_relative_to(tmp_path / "scratch")


def test_an_unrelated_or_broken_zip_is_ignored(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "broken.zip").write_bytes(b"not a zip")
    import zipfile

    with zipfile.ZipFile(tmp_path / "input" / "other.zip", "w") as z:
        z.writestr("readme.txt", "hello")
    assert run.find_bundle(tmp_path / "input", tmp_path / "scratch") is None


def test_bodhi_s_is_placed_only_when_both_its_files_are_there(tmp_path):
    source = tmp_path / "input" / "private" / "bodhi_s"
    source.mkdir(parents=True)
    (source / "triples.jsonl").write_text("t", encoding="utf-8")
    assert not run.place_bodhi_s(tmp_path / "input", tmp_path / "raw" / "bodhi_s")
    (source / "nl_facts.jsonl").write_text("f", encoding="utf-8")
    assert run.place_bodhi_s(tmp_path / "input", tmp_path / "raw" / "bodhi_s")
    assert (tmp_path / "raw" / "bodhi_s" / "nl_facts.jsonl").read_text(encoding="utf-8") == "f"


def test_a_resumed_run_takes_the_fullest_earlier_results_and_never_a_shorter_one(tmp_path):
    def results(folder: Path, n: int, with_run_json: bool = True) -> None:
        folder.mkdir(parents=True)
        (folder / "results.jsonl").write_text("{}\n" * n, encoding="utf-8")
        if with_run_json:
            (folder / "run.json").write_text("{}", encoding="utf-8")

    results(tmp_path / "input" / "v1" / "llm_explain", 5)
    results(tmp_path / "input" / "v2" / "llm_explain", 9)
    results(tmp_path / "input" / "other", 99, with_run_json=False)  # not a run's output
    out = tmp_path / "out"
    assert run.previous_results(tmp_path / "input", out) == 9
    assert run.previous_results(tmp_path / "input", out) == 9  # again: nothing changes
    (out / "results.jsonl").write_text("{}\n" * 12, encoding="utf-8")  # more lines already here
    assert run.previous_results(tmp_path / "input", out) == 12
    assert run.previous_results(tmp_path / "empty", tmp_path / "fresh") == 0


# --------------------------------------------------------------------------------------------- #
# A model that stops answering is not a result
# --------------------------------------------------------------------------------------------- #


def test_five_failed_calls_in_a_row_stop_the_run_and_are_set_aside_for_a_resume(tmp_path):
    cases = run.load_golden()
    items = [run.Item(B3, c, 1.0, "golden", t) for c, t in cases]
    more = [run.Item(B3R, c, 1.0, "golden", t) for c, t in cases]
    client = Scripted()
    exp = experiment(tmp_path, client)
    # three calls answered, then the model goes away
    state = {"n": 0}
    real = client.generate

    def flaky(prompt, *, system, timeout):
        state["n"] += 1
        if state["n"] > 3:
            raise LLMError("gone")
        return real(prompt, system=system, timeout=timeout)

    client.generate = flaky
    counts = exp.run([*items, *more], progress=lambda _: None)
    assert counts["stopped"] == 1 and counts["run"] == 3 + run.MAX_CONSECUTIVE_FAILURES
    kept = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(kept) == 3  # the failures that opened the streak are not results
    tail = (tmp_path / "failed_tail.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(tail) == run.MAX_CONSECUTIVE_FAILURES
    assert all(json.loads(line)["reply"]["error"] for line in tail)

    # the model is back: a resume asks them again, and nothing is asked twice
    client.generate = real
    again = experiment(tmp_path, client)
    counts = again.run([*items, *more], progress=lambda _: None)
    assert counts["stopped"] == 0 and counts["skipped"] == 3
    assert len((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 8


def test_isolated_failures_stay_in_the_results(tmp_path):
    client = Scripted()
    real = client.generate
    state = {"n": 0}

    def sometimes(prompt, *, system, timeout):
        state["n"] += 1
        if state["n"] in (
            3,
            4,
            5,
        ):  # the third item fails all three of its attempts, then it recovers
            raise LLMError("busy")
        return real(prompt, system=system, timeout=timeout)

    client.generate = sometimes
    exp = experiment(tmp_path, client)
    items = [run.Item(B3, c, 1.0, "golden", t) for c, t in run.load_golden()]
    counts = exp.run(items, progress=lambda _: None)
    assert counts["stopped"] == 0 and counts["run"] == 4
    rows = [
        json.loads(x) for x in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [bool(r["reply"]["error"]) for r in rows] == [False, False, True, False]
