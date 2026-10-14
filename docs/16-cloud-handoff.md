# 16 · Handoff to a cloud session (2026-10-14)

**Read this, then `PROGRESS.md` §1–§2, before doing anything.** The owner moved the work from the
laptop session to a cloud Claude Code session on 2026-10-14. This page holds what a fresh session
needs that the repository does not show by itself: where each workstream stopped, what it still
needs, and what a cloud machine cannot do.

## 1. State of the repository

> **Update 2026-10-14 (later):** PR #1 is merged (`a7222ee`). 3d and 3a were finished on the
> laptop by two agents and reviewed with the real graph and transformer: PR #2 (3d) and PR #3 (3a,
> which contains 3d). The `wip/3d-next-steps` and `wip/3a-retrieval` rows below are history; 3e
> and 1d are still as described. `PROGRESS.md` §1 has the next action.

- **`master` = `v0.1-prototype`** (`06b6263`). The working prototype: the dashboard with the
  transformer B1-DL′+aug (D-13), the knowledge graph, the red flags and the model's reasons. To
  present it, see `docs/15-prototype-demo.md`.
- **PR #1** (`docs/report-draft-and-fixes` → `master`): the report's first draft (`docs/report/`)
  and stale-passage corrections; this page is on that branch too.
  - **Merge it first.** CI must be green, and the owner reviews it.
- **Unfinished work on WIP branches.** Three agents stopped at the account's usage limit. Never
  merge these as they are; finish them on their branch, review, test, then open a PR.

  | Branch | Workstream | What is there | What is left |
  |---|---|---|---|
  | `wip/3a-retrieval` | **3a, evidence retrieval** (EXP-011) | `src/rag/`: BM25, the bundle loader with SHA-256 checks, corpus and E-utilities helpers, the retriever, the query vocabulary; first pipeline/config wiring | Finish and review the code; `tests/test_retrieval.py` on a tiny invented corpus; `notebooks/kaggle_retrieval_corpus.ipynb`; a small "Evidence" block in each candidate's expander in the dashboard; the EXP-011 design note |
  | `wip/3d-next-steps` | **3d, "what to ask next"** | `src/reasoning/investigations.py`, the start of the cited table of investigations | The question ranking (e.g. expected entropy reduction over the graph's naive-Bayes likelihoods); `DiagnosisResult.suggested_next_tests`; the API; a "What to ask next" dashboard block; `tests/test_next_steps.py` |
  | `wip/3e-error-analysis` | **3e, calibration curves and errors by condition** (EXP-023) | `scripts/error_analysis.py`, `src/eval/errors.py`, an SVG renderer `src/eval/svg.py`, tests | Its last note: the figure grid expects `rows`, but the analysis stores `reliability`; fix it, then redraw with `--figures-only`. **The run needs `data/` and `models/`** (§3) |

- **1d (Synthea and the patient knowledge graph)** is not started. Its plan:
  - `notebooks/kaggle_synthea.ipynb`: a pinned Synthea jar, seed 42, run on Kaggle;
  - `src/patient_kg/`: a per-patient graph from Synthea's CSVs;
  - a SNOMED → `RF:*` table that returns `Finding`s with `source="synthea"`. An unmapped code is
    kept with its reason. A new `RF:*` concept needs a crosswalk entry;
  - `scripts/build_patient_kg.py`;
  - tests on a hand-made fixture.

## 2. The briefs, in short (each agent had these rules)

Each workstream had these rules:
- **Write tests**, and run black, ruff and the full suite.
- **Stage files by path**, never `git add -A`.
- **Leave the decision records alone.** Do not edit `docs/05` (frozen) or `src/contracts.py` (it
  needs all four team members; flag a change instead). The lead updates `PROGRESS.md`, `CLAUDE.md`
  and `docs/08`.
- **Safety:** keep the golden cases green, and fix the component, not the expectation. Never
  weaken the disclaimer. No treatment or drug advice; diagnostic investigations only, phrased as
  "consider".
- **No fused rankings over the validation split** until the team answers E-2 (`PROGRESS.md` §3).
  Hand-written golden cases are fine.

Each workstream also has its own rules:
- **3a:** the laptop's runtime must stay pure NumPy and the standard library: no faiss, no
  sentence-transformers, no torch. The Kaggle notebook fetches **PMC Open Access** articles only.
  It uses no API key, keeps to 3 requests a second, and records each article's licence. It embeds
  the passages, plus a fixed query vocabulary (one per condition, one per finding label), with
  `pritamdeka/S-PubMedBert-MS-MARCO`. At run time the query vector is a mix of precomputed vectors,
  so no model is needed. The hybrid is reciprocal rank fusion with k = 60. Never commit the corpus
  or the embeddings (`docs/03` §1.4).
- **3d:** every investigation carries a published, cited guideline; never invent a reference. If
  "questions to ask" needs a contract field, expose it another way and flag the change. Keep the
  whole pipeline fast enough for live re-ranking: about 10–13 ms now (EXP-022).
- **3e:** score the components alone (B1-DL′+aug, B1-LR′+aug, B1-XGB′+aug, B2), never the fusion.
  Draw the figures as SVG with no plotting library, readable on a slide.

## 3. What a cloud session cannot do

- **`data/` and `models/` exist only on the owner's laptop** (gitignored; `PROGRESS.md` §7). So, in
  the cloud:
  - every data-gated test skips, as in CI;
  - the dashboard ranks with the graph's stub, or not at all, and the configured transformer is
    absent;
  - 3e's analysis, golden checks on the real models and anything scored on validation cannot run.
  - Do code, tests on synthetic fixtures, notebooks and documents in the cloud. Leave the
    data-dependent runs to the laptop or to Kaggle.
- **Downloading DDXPlus or models in the cloud is a download, so ask the owner first** (D-1, D-4).
  The trained models sit in the owner's private Kaggle dataset `nightingale-models`, which a cloud
  session cannot read.
- **`.env` (the AuraDB credentials) is on the laptop only and must never be copied anywhere.**
  Without it the graph falls back to the local build; in the cloud there is no local build, so the
  graph is the stub.
- **No GPU work**, and no training or tuning, without the owner's choice of where (D-7: Kaggle for
  every cloud job). The laptop GPU is irrelevant in the cloud.

## 4. The owner's standing rules (2026-10-14)

- **At most 2–3 agents at a time.** Five at once hit the account's usage limit twice.
- **The working prototype comes first.** The next prototype version, `v0.2-prototype`, adds 3d
  (what to ask next) and 3a (evidence with citations). 3a's evidence needs the owner's Kaggle run of
  its corpus notebook.
- **Kaggle for every cloud job.** It runs as *Save & Run All (Commit)*, so the browser can be
  closed. Use `notebooks/kaggle_b1_dl.ipynb` and `notebooks/kaggle_exp021_reports.ipynb` as
  patterns, and `docs/11` §4.4–§4.5 for the runbook.

## 5. Suggested order for the cloud session

1. Merge PR #1 once CI is green and the owner approves it.
2. **Finish 3d on `wip/3d-next-steps`.** It needs no data beyond the stub graph for its tests;
   golden checks on the real graph wait for the laptop. Then open a PR.
3. **Finish 3a's code and Kaggle notebook on `wip/3a-retrieval`**; then open a PR. The owner then
   runs the notebook and unzips the bundle into `data/raw/retrieval_corpus/` on the laptop.
4. On the laptop, or a session with `data/`: 3e's run, the golden checks with 3d's and 3a's
   changes, then tag `v0.2-prototype`.
5. 1d when there is room. **E-2 is still with the team.** It blocks the fusion sweep (EXP-006)
   and the system's calibration (EXP-007 stage 2).
