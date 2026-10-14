# 17 · Work for a cloud session: safe, and needing nothing from the owner

**Version 1.0 · 2026-10-14 · written for a Claude Code cloud session, at the owner's request:** "make
a .md of all the stuff that can be done in cloud session without any risk and need to my input and
intervention". Read `CLAUDE.md`, `PROGRESS.md` §1–§2 and `docs/16-cloud-handoff.md` first. Where this
page and those disagree, **they win**, and the session says so in its PR.

> ⚠️ Nightingale is decision support, never a diagnosis. Nothing here may weaken the disclaimer or
> add a treatment or drug recommendation.

---

## 1. What "safe, with no input" means here

A task is on this list only if all six hold:

1. **It touches nothing the cloud does not have.** `data/`, `models/` and `.env` exist only on the
   owner's laptop. Never ask for them, recreate them or copy them.
2. **It downloads nothing beyond the repository and CI's packages.**
   - No DDXPlus, no model weights, no Synthea jar, no corpus.
   - No `ollama pull` and no Hugging Face model.
   - Reading a public web page (a guideline's page, a GitHub file) to check a fact is allowed.
3. **It trains, tunes and scores nothing on project data,** and runs no model on a GPU.
   - Unit tests on invented data are tests, not training.
   - Never compute a fused ranking over the validation split: E-2 is unanswered.
4. **It changes no decision record:**
   - `docs/05-evaluation-protocol.md` (frozen);
   - `src/contracts.py` (needs all four team members);
   - `PROGRESS.md` §3 and §6;
   - any golden case's expectation in `tests/fixtures/golden_cases.yaml` (if one fails, fix the
     component, never the expectation).
5. **It is reversible and reviewable.** Each task goes on its own branch and ends as a pull request.
   - Never push to `master`, merge, rebase someone else's branch, force-push, tag or delete
     anything.
6. **It needs no choice from the owner or the team.** A task that turns out to need one stops. It
   records the question in its PR, under "Needs the owner", and moves on.

**The one install this list assumes:** `pip install -r requirements-dev.txt` into the cloud
session's own environment, Python 3.11. CI installs exactly this, and it touches nothing of the
owner's. *(Owner: if you do not allow even this, delete this paragraph, and the session can only edit
documents.)*

## 2. How to work

- **Start:**
  1. Clone the public repository.
  2. Create a Python 3.11 environment and install `requirements-dev.txt`.
  3. Run `python -m pytest -q`.
  - Every data-gated test skips, as in CI. That is expected, not a failure.
- **One branch per task,** named `cloud/<task-id>-<slug>` (e.g. `cloud/T1-demo-docs`), from the
  latest `master`.
- **Commit early and often** on that branch. A session can stop at a usage limit, and pushed commits
  survive it.
  - Stage files by path, never `git add -A`.
  - End every commit message with the attribution line the session's instructions give.
- **Before every push:** run `python -m black app src tests scripts`,
  `python -m ruff check app src tests scripts` and the full test suite, all green.
- **Open one PR per task** against `master`, ready for review, and leave it open. Its description
  says:
  - what changed and why;
  - what was checked in the cloud;
  - **"Laptop checks needed"**: what only the real data can confirm;
  - **"Needs the owner"**: questions, if any;
  - **"Records for PROGRESS.md"**: the lines §5 and §8 should get. Do not edit `PROGRESS.md` in a
    task branch: parallel branches would conflict there. The laptop session writes the records when
    it merges.
- **Agents: at most 2–3 at a time** (the owner's rule). Tasks that touch the same files run one
  after another.
- **Style:** match the repository: plain docstrings that say why, cite the doc section, and disclose
  limits. Correct old text by strike-through (`~~old~~ new`), never by erasing it.

## 3. The tasks, in priority order (the prototype first)

**Two touch the dashboard or the demo: T1 and T6.** Run them one after another, not in parallel.

### T1 · The demo and its guide for `v0.2-prototype` (3d and 3a are merged)
- **`docs/15-prototype-demo.md`:**
  - add the "What to ask next" block to the demo script: GC-001's questions are rest pain and the
    crescendo pattern, and its tests are ECG and serial troponin, as the code gives them;
  - add the Evidence block, saying it stays empty until the owner's Kaggle corpus run (`docs/11`
    §4.6);
  - move both out of "Not in this version yet".
- **`scripts/demo.py`:** build its pipeline through `scripts/_config.py`'s `build_pipeline`, so it
  gets the configured retriever and degrades to `rag` without a bundle. Then print the suggested
  tests, the top questions (`src/reasoning/next_questions.py`) and any evidence with its citation.
  Keep its existing flags working.
- **`README.md`:** one line each for "what to ask next" and evidence in the feature list.
- **Done when:** the stub path (`scripts/demo.py --graph stub`) prints all three blocks; tests cover
  the new output; nothing in a ranking changes.
- **Laptop checks needed:** `scripts/demo.py --case GC-001` on the real graph.

### T2 · The retriever in the measuring scripts
- **`scripts/measure_latency.py`** (EXP-022): replace `EmptyRetriever` with the configured
  retriever, so that the laptop can re-measure the pipeline with evidence and the questions once
  the bundle exists. Report retrieval's time as its own stage. Do not change what the script
  measured before. Add a flag that keeps the old empty retriever, so EXP-022 can be reproduced.
- **Leave `apply_safety_floor.py`, `check_crosswalk.py` and `sweep_fusion.py` as they are.** They
  judge rankings, and evidence never changes a ranking.
- **Done when:** the script's tests pass, and a run without data degrades and says so.

### T3 · Small fixes the reviews of 3d and 3a found
- **Rename `tests_to_consider`** in `src/reasoning/investigations.py`. Pytest collects any
  `test*` name imported into a test module.
  - Use a name such as `investigations_to_consider`.
  - Update every import, and drop the alias in `tests/test_next_steps.py`.
- **Leave the two `verify=True` references unchanged.** T8 deals with them.
- **Done when:** the full suite is green, with the same count of passing tests.

### T4 · Stale passages, corrected by strike-through
Each is an old statement that the code or a later decision contradicts. Keep the old text struck
through, write the correction beside it, and cite what changed.

| File | Stale | Correct |
|---|---|---|
| `README.md` (the architecture diagram, ~line 125) | "LogReg · RandomForest · XGBoost" | No random forest was ever built. Today: logistic regression and XGBoost (B1), and the transformer B1-DL′+aug, the configured ranker since D-13 |
| `README.md` (the stack table, ~line 286) | "MLflow, DVC" | Neither is used: no DVC (`PROGRESS.md` §6, 2026-09-17), and runs are recorded in `docs/08` and each bundle's `run.json` |
| `docs/00-project-charter.md` O7 | "across 5 configurations" | A0–A7 and A1b, at 100%, 50% and 25% evidence (`docs/05` §5 as amended, v1.4) |
| `docs/01-srs.md` FR-10.2 | "all five ablation configurations" | The same as O7 |

- Do not touch `docs/05`.
- **Done when:** each row is corrected, and a search finds no other unstruck copy of the same claim.
  If one is found in `docs/report/`, correct it there too.

### T5 · 3c guardrails v1, for the template explanations and the evidence (no LLM)
Phase 3's 3c, the part that needs no model:
- **`src/reasoning/guardrails.py`:**
  - a **claim-support check**: each sentence of an `Explanation` must trace to something the result
    holds, namely a graph path, a red flag, a model reason, a suggested test or a cited passage.
    A sentence that traces to none goes into `Explanation.unsupported_claims`, and `grounded`
    becomes False. Both fields already exist in `src/contracts.py`, so the contract does not change;
  - **citation rendering**: one formatter for evidence citations (`PMCID:…` → its PMC link), with
    duplicates removed. It never makes a link to any host but PMC.
- **Where it runs:** in the pipeline, after the explainer and before `safety_check`. The template
  explainer's sentences must all pass it: they are built from those parts. That is the first test.
- **Tests:** on invented explanations and results, an unsupported sentence is caught, a supported
  one is kept, and a citation to another host is refused.
- **Done when:** every golden case's explanation passes, with red flags on and off, on the stub
  graph.
- **Laptop checks needed:** the same on the real graph and transformer.

### T6 · 3b scaffolding: the grounded LLM explainer, behind the template, off by default
Code only. **No model is pulled or run, nothing calls Ollama, and the GPU is never touched.** D-5
(which model) stays the owner's, so the code reads `llm.model` from `configs/config.yaml` and
assumes nothing about it.
- **`src/llm/`:**
  - an `LLMExplainer` that builds its prompt only from what the result holds (the top candidates,
    their graph paths, red flags, the model's reasons, suggested tests and cited passages);
  - asks a client for text;
  - passes the answer through T5's check;
  - falls back to `TemplateExplainer` when the client is absent, times out
    (`llm.timeout_seconds`), fails, or returns any unsupported sentence or any treatment language.
- **The client is an interface.** Write the Ollama HTTP client against Ollama's documented API, but
  test it only with a fake transport. No test opens a network connection.
- **The config** gains `llm.enabled: false`. With it false the pipeline is exactly as before, and a
  test asserts that equality.
- **Not now:** the B3/B4 baselines (EXP-009/010) and anything that measures an LLM. They need a
  model, D-5, and the owner's choice of where (D-7).
- **Done when:** the fake client's grounded answer is used, every ungrounded, treatment-bearing or
  timed-out answer falls back, and the suite is green.

### T7 · 1d: Synthea and the patient knowledge graph, as code (`docs/16` §1's plan)
The notebook is written but **not run**: the owner runs it on Kaggle (D-7).
- **`notebooks/kaggle_synthea.ipynb`:**
  - follow `notebooks/kaggle_retrieval_corpus.ipynb`'s pattern (repository at a commit, a smoke
    cell, `STATUS.txt`, output zipped);
  - download a **pinned** Synthea release on Kaggle and run it with seed 42 and a stated population
    size;
  - print every SNOMED code it meets that the table below does not map.
- **`src/patient_kg/`:** a per-patient graph from Synthea's CSV output (conditions, observations,
  medications as history only, never as advice). Plus a table from SNOMED codes to the existing
  `RF:*` concepts, returning `Finding`s with `source="synthea"`.
  - `"synthea"` is already an allowed `Finding.source` (`src/contracts.py`), so no contract
    change is needed.
  - Every code the table maps must be one that Synthea's public module files contain. Read them on
    GitHub to check, and mark any code that cannot be checked `verify=True`.
  - An unmapped code is kept with its reason.
  - A new `RF:*` concept needs a crosswalk entry, and a test enforces it.
- **`scripts/build_patient_kg.py`**, and tests on a **hand-made** fixture of a few invented patients.
- **Wiring:** do not add the patient KG to the pipeline's ranking. Whether it feeds the case is a
  design choice for the owner. Expose a function that turns a patient's graph into `Finding`s, and
  stop there.
- **Done when:** the builder and table are tested on the fixture, the notebook's cells parse, and
  `tests/test_notebooks.py` covers it.

### T8 · References and the report, from the repository's own records
- **The two references marked `verify=True`** in `src/reasoning/investigations.py`: BTS pleural
  disease 2023 (*Thorax*) and WSES oesophageal emergencies 2019 (*World J Emerg Surg*).
  - Check each against the publisher's own page.
  - Correct a detail only when that page shows it exactly, then clear the flag and cite the page in
    the PR.
  - If the page cannot be reached, leave the flag and say so.
- **`docs/report/`:** bring chapters 4 and 6 up to date from `docs/08` and the code, copying every
  figure with its interval and evidence level exactly as `docs/08` gives it:
  - 3d and 3a in chapter 4;
  - EXP-021's full report in chapter 6;
  - EXP-023 in chapter 6 once it is on `master`.
- **Never invent a citation.** A claim without a source in the repository keeps its
  `[citation needed]` marker.
- **Done when:** a figure check, like the one run on 2026-10-14 (every number quoted with an
  interval must appear in `docs/08`), finds no mismatch.

## 4. What a cloud session must not do (it needs the owner, the team or the laptop)

| Not in the cloud | Why | Who |
|---|---|---|
| Merge a PR, push to `master`, tag `v0.2-prototype` | The owner reviews every change; a tag marks a state checked on the real data | The owner, or a laptop session |
| Run any notebook or Kaggle job (the retrieval corpus, Synthea) | It runs under the owner's Kaggle account (D-7) | The owner |
| Download DDXPlus, models, the corpus, Synthea or an LLM; run Ollama; use a GPU | Downloads and model runs are asked one by one (D-4, D-5, D-9) | The owner |
| Train, tune or score anything on project data; open the test split | D-7; `docs/05` §2 (the test split is opened once, in Phase 4) | The owner |
| Anything that needs `data/` or `models/`: golden checks on the real graph and transformer, EXP-022's re-measure, checking the corpus bundle, the dashboard demo | Those files are on the laptop only | A laptop session |
| EXP-006's fusion sweep; EXP-007 stage 2 (the system's calibration) | They wait for E-2's first item | The team, then the owner |
| `src/contracts.py`, including the two proposals (a `suggested_questions` field; `licence` and `attribution` on `Evidence`) | Contract changes need all four team members | The team |
| `docs/05`, `PROGRESS.md` §3 and §6, a golden case's expectation, GC-005 | Frozen, or a decision, or clinical judgement | The team |
| Which LLM (D-5); the B3/B4 baselines | A choice, then a model run | The owner |
| Delete or move files, the agents' worktrees included | Needs the owner's approval | The owner |
| Open, copy or recreate `.env` | It holds the Aura login | Never |

## 5. Left for the owner (only you can do these)

1. **Run the retrieval corpus notebook on Kaggle.**
   - Steps: `notebooks/kaggle_retrieval_corpus.ipynb`, GPU T4 x2, internet on, *Save & Run All
     (Commit)* (`docs/11` §4.6).
   - Then unzip `retrieval_corpus.zip` into `data/raw/retrieval_corpus/`, and set
     `retrieval.bundle: data/raw/retrieval_corpus` in `configs/config.yaml`.
2. **Relay to the team:**
   - E-2 (`PROGRESS.md` §3);
   - the two contract proposals (PR #2's and PR #3's descriptions);
   - EXP-011's relevance judgements, which need two team members;
   - GC-005.
3. **Review and merge the cloud session's PRs.**
4. **Record the demo video** (Phase 2's exit criterion).

## 6. Back on the laptop, after the cloud session

1. Merge the reviewed PRs, then `git pull`.
2. `./.venv/Scripts/python.exe -m pytest` and `scripts/check_crosswalk.py`: every golden case on the
   real graph and transformer.
3. Each PR's "Laptop checks needed" list.
4. The records for `PROGRESS.md` from each PR (§4.3), in one commit.
5. With the corpus bundle in place: `scripts/measure_retrieval.py --bundle data/raw/retrieval_corpus`,
   the dashboard checked end to end, then tag `v0.2-prototype`.
