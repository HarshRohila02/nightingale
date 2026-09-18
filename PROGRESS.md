# PROGRESS — Nightingale

> **The single source of truth for where this project is.** Auto-loaded into every Claude Code session
> through `CLAUDE.md`. Read §1 and §2 first. **Updating this file is part of every task** — the rules
> are in §4 and they are mandatory. If this file and the repository disagree, stop and reconcile
> (§4.1) before doing any new work.

**Last updated:** 2026-09-18 · by Claude (D-9 recorded: laptop-GPU tests run after the user's yes) · **Last verified commit:** `e956594`

---

## 1. 📍 Current position

| | |
|---|---|
| **Completed phase** | Phase 0 — Preparation & Documentation ✅ *(environment items carried over to 1.0 — see §5)* |
| **Current phase** | **Phase 1 — Data & Knowledge Foundations** |
| **Current sub-phase** | **1.0 — Environment bring-up** · 🔄 (Python 3.11 ✅ · torch pin ✅ · compute placement **D-7** + **D-4**/**D-6** set by the user ✅ · laptop-GPU test environment `.venv-gpu` ✅ (torch 2.11.0+cu128); the AuraDB instance awaits the user) · **1a — Data** · 🔄 (validate parquet ✅ + EXP-013; the train split waits, and the user picks where it is built) · **1b — KG** · 🔄 (loader + NetworkX store ✅; crosswalk, BODHI-S, aortic dissection, Neo4j store to go) |
| **Next action** | **Claude:** the 1b **crosswalk** (`DDX:E_nn` ↔ `SYM:*`). It needs no download and runs in seconds. **The user, when convenient:** create the **AuraDB Free** instance (`docs/11` §5). **The team:** **D-8** |
| **Blocked on** | The Neo4j store ← the user's AuraDB instance · any Precision@3 / Recall@5 number ← **D-8** (team) · Phase 3 LLM ← D-5 (deferred) · university GPU access (external, R-14). **Every training or tuning run, and any job over ~5 min, waits for the user to say where** (D-7) |
| **Schedule** | Week 1 of 8–10 · ahead of plan · next milestone 🎯 W3 walking skeleton, target 2026-10-07 |
| **Health** | 91 tests passing locally in `.venv` (88 in CI, where the 3 real-data tests skip). In `.venv-gpu`, 89 pass and the 2 torch-missing checks skip · CI green on GitHub at `e956594` |

**Handoff note for the next session:** Nothing is in flight. The chest-pain parquet exists for
**validate** only (`docs/03` §2.1). EXP-013 found two things everyone must know. First, **a listed
DDXPlus token can mean "no"** (`E_204_@_V_10` = "did not travel"), so use `positive_codes()`.
Second, **the ground-truth differentials are open-world**, so `Recall@5` as written tops out at
0.434 (**D-8**). The KG now loads into a NetworkX store, but read its card (`docs/02` §5.1) before
trusting a score. It links conditions to *questions*, not answers, and ranks **stable angina above
must-not-miss unstable angina** even when rest pain is present; 2a must fix this. The pipeline and
golden cases still use the stub store until the crosswalk exists. Before touching the knowledge graph,
read `docs/10-spike-r01-crosswalk.md` — the KG is built from DDXPlus `release_conditions.json`,
**not** BODHI-S, which creates circularity risk R-12. EXP-002 added two things every result must
respect: the system is **closed-world** (R-13 — only 13 conditions exist for it), and DDXPlus
demographics are synthetic (~50% female everywhere), so **by-sex parity is an artifact, not
fairness**. The evaluation protocol (`docs/05`) is **frozen**. Use `./.venv/Scripts/python.exe` —
plain `python` is 3.14, not the team's 3.11. **Compute (D-7, `docs/11`):** before every training or
tuning run on project data, and any job over ~5 minutes (CPU jobs too), ask the user **where** to
run it, suggesting the cloud. **The laptop GPU is for short tests only, and each one needs the
user's yes in chat first (D-9).** That includes any Ollama model, which uses the GPU automatically.
A yes covers one test. `.venv-gpu` is set up and verified (`docs/11` §2).

---

## 2. 🛫 In-flight (write-ahead marker)

> Filled in **before** a task starts; cleared **after** it is committed. If this section is not empty
> when a session begins, the previous session was interrupted — go to §4.4.

_Nothing in flight._

<!-- Template — copy above the line when a task starts:
- **Task:** <one line>
- **Sub-phase:** <e.g. 1b>
- **Started:** <YYYY-MM-DD HH:MM> by <Claude | P1–P4>
- **Files expected to change:** <paths>
- **Done so far:** <bullets — keep current as you work>
- **Remaining:** <bullets>
- **Safe to resume blindly?** <yes | no — verify first, because …>
-->

---

## 3. ❓ Awaiting user decisions

| ID | Decision needed | Options | Recommended | Blocks |
|---|---|---|---|---|
| **D-5** | Which LLM, and pulling it (`ollama pull llama3.1:8b`, ~4.9 GB) | llama3.1:8b · qwen2.5:7b-instruct · reuse the installed `qwen2.5-coder:7b` | **llama3.1:8b**, a general instruct model (the coder model is tuned for code, not clinical prose). **Deferred to Phase 3** (Week 6). Under D-7 and D-9, a laptop test runs only after the user's yes (`docs/11` §3, which uses the installed `qwen2.5-coder:7b`, so no download), and batch runs pull the model on the chosen cloud runtime | Phase 3 explainer |
| **D-8** 🆕 | **What the ground-truth differential D means** in Precision@3 and Recall@5 (`docs/05` §3.1). EXP-013: 91.8% of in-scope patients' D contain conditions we cannot output (33% of the probability mass), so **Recall@5 as written tops out at 0.434 for a perfect system**. `docs/05` is **frozen**, so any change is a dated amendment in its §9, approved by the team **before any model is evaluated** | Keep as written · **restrict D to the 13 in-scope conditions** (ceiling 0.753) · restrict and renormalise the probabilities | **Restrict D to the in-scope conditions** for the headline figure, and report the as-written figure alongside it for comparison with published DDXPlus results. Top-k accuracy, MRR and must-not-miss recall are unaffected | 1e metrics · any Precision@3 / Recall@5 number |

When the user decides, move the row to §6 with the date, and record it in §8.

---

## 4. 🔁 Update protocol — MANDATORY

These rules exist so that **nothing is lost or miscommunicated** when a session ends, a new session
starts, or work is interrupted in the middle of a message. They apply to Claude and to all four team
members.

### 4.1 At the start of every session
1. Read §1 Current position and §2 In-flight (already loaded via `CLAUDE.md`).
2. Run `git status --short` and `git log --oneline -5`.
3. Replace any `→ pending` commit hashes in §5 and §8 with the real hashes from `git log`, and advance
   **Last verified commit** (top of file) to the newest commit whose work is now recorded. Commit this
   as `chore(progress): record commit hashes`.
4. The state is **inconsistent** if any of these is true:
   - §2 In-flight is not empty;
   - `git status` shows any uncommitted or untracked changes;
   - `git log` shows commits after the **Last verified commit** that §5/§8 do not account for
     (commits touching only `PROGRESS.md` / `CLAUDE.md` are exempt).
5. **Inconsistent** → follow §4.4 before any new work. **Consistent** → tell the user in one line:
   *"We're at <phase / sub-phase>; the next action is <…>."*

### 4.2 Before starting a task — write-ahead
Fill in §2 In-flight **before editing any other file.** A *task* is a unit of work that ends in a
commit — **so every commit is a task boundary.** If one piece of work is split across several
commits, each commit must refresh §1 as well as §2; otherwise §1 goes stale between them. *(Lesson
learned 2026-09-18: §1 was left saying "blocked on D-1/D-2" across two intermediate commits.)* If the task will take a long time or download something large, ask the user first.

### 4.3 After finishing a task — in the same commit as the work
1. Tick the task in §5 and mark it `→ pending` (a commit cannot contain its own hash; §4.1 step 3
   fills it in next session).
2. If that finishes a **sub-phase** → mark it ✅ with the date. If it finishes a **phase** → check the
   phase's exit criteria first; mark ✅ only if they are met, otherwise record the gap as ⚠️ carried
   over, with where it went.
3. Update §1: current phase, sub-phase, next action, blocked-on, last updated, last verified commit.
4. Clear §2 In-flight.
5. Add one line to §8 Session log.
6. Commit the work **and** this file together, then push.

### 4.4 Recovering from an interruption
1. **Do not assume the in-flight task was finished — or that it wasn't.**
2. Inspect: `git status`, `git diff`, and run the tests.
3. Write what you find into §2 (*Done so far / Remaining*).
4. Tell the user what was interrupted and the state it is in, then offer: **finish it · revert the
   partial change · start over.** Wait for their choice unless the fix is trivially safe.

### 4.5 If you have to stop mid-task
Context running out, a user interrupt, or a blocker: **update §2 first** — what is done, what is
half-done, the exact next step. A half-finished task with an accurate In-flight entry is recoverable;
one without it is not.

### 4.6 Status rules
- ✅ only when **committed and verified** (tests pass, output checked). Never tick ahead of the commit.
- A sub-phase is ✅ only when all of its tasks are ✅. A phase is ✅ only when all of its sub-phases
  are ✅ **and** its exit criteria are met — or the gap is written down as carried over.
- Never erase history. Correct a wrong entry by striking it through (`~~wrong~~`) and adding the fix.
- When the user changes the timeline, update the targets in §5 and log it in §8 the same day.

**Legend:** ✅ done · 🔄 in progress · ⬜ not started · ⚠️ partial / carried over · ⛔ blocked ·
⏭️ deferred or cut

---

## 5. 🗺️ Phase tracker

Targets assume **Week 1 began 2026-09-17** with an 8–10 week budget. The user will say if the timeline
changes — update this table when they do.

| Phase | Weeks | Target end | Status |
|---|---|---|---|
| 0 — Preparation & documentation | 1 | 2026-09-23 | ✅ 2026-09-17 · env items → 1.0 |
| 1 — Data & knowledge foundations | 2–3 | 2026-10-07 | 🔄 1.0 + 1a + 1b in progress |
| 2 — Reasoning, fusion & prototype | 4–5 | 2026-10-21 | ⬜ |
| 3 — Evidence & explanation | 6–7 | 2026-11-04 | ⬜ |
| 4 — Evaluation, ablations & write-up | 8–9 | 2026-11-18 | ⬜ |
| Buffer | 10 | 2026-11-25 | ⬜ |

### Phase 0 — Preparation & Documentation · ✅ 2026-09-17
**Exit criteria:** docs drafted ✅ · repo scaffolded ✅ · spike answered with a recorded decision ✅ ·
evaluation protocol frozen ✅ · **Neo4j + Ollama running ⚠️ not met — carried over to 1.0**
*(2026-09-18: Neo4j → AuraDB Free, awaiting the user's instance; Ollama → 3b, by D-5 and D-7)*

**0a — Documentation set · ✅ 2026-09-17**
- [x] `docs/00`–`09` plus index — `26240f5`, `1a47a0e`
- [x] `CONTRIBUTING.md`, `LICENSE` — `1a47a0e`
- [x] README documentation section — `a842c0f`

**0b — Repo scaffold · ✅ 2026-09-17** *(code only — environment moved to 1.0)*
- [x] `src/contracts.py`, `conditions.py`, `pipeline.py`, `reasoning/red_flags.py`, `stubs.py` — `efc10e9`
- [x] requirements (core + NLP), `pyproject.toml`, `docker-compose.yml`, `configs/`, `.env.example`, CI — `efc10e9`
- [x] 33 tests incl. 4 golden clinical cases; `scripts/demo.py`, `scripts/download_data.py` — `efc10e9`
- [x] CI verified green on GitHub for `efc10e9` and `8b682e9`

**0c — Feasibility spike, R-01 · ✅ 2026-09-17**
- [x] `scripts/decode_ddxplus.py` — 919-entry vocabulary, 13/13 conditions found — `8b682e9`
- [x] `scripts/spike_crosswalk.py` + `docs/10` report — 31% vs 60% gate → **FALLBACK** — `8b682e9`
- [x] R-01 closed, R-12 opened, EXP-001 logged — `8b682e9`

**0d — Phase exit · ✅ 2026-09-18**
- [x] `docs/05` evaluation protocol **frozen** at v1.1, before any model was trained — `8b682e9`
- [x] Session continuity: `PROGRESS.md` + `CLAUDE.md` — `51b49ce`

### Phase 1 — Data & Knowledge Foundations → Walking Skeleton · 🔄
**Exit criteria:** 🎯 **W3 milestone** — a case produces a ranked differential from the **real** ML
ranker with **real** KG-matched supporting findings, end to end · CI green.

**1.0 — Environment bring-up · 🔄** *(carried over from Phase 0)*
- [x] Archive the reference docs to `docs/archive/`; delete the byte-identical duplicate (D-3) — `5ce3725`
- [x] Python 3.11 (D-2): `py install 3.11` → 3.11.9; `.venv` rebuilt; tests, black, ruff green — `1ccb06a`
- [x] Fix local-vs-CI tool drift: new `requirements-dev.txt` is the single source of truth for pytest/black/ruff; CI and `requirements.txt` both read it — `1ccb06a`
- [x] Fix the torch pin in `requirements-nlp.txt`: `~=2.4` → `~=2.7`. 2.7 is the first release that supports the RTX 5060; the CPU and every CUDA build satisfy it — `8dced77`
- [x] Record the GPU policy (**D-7**, set by the user): laptop GPU only with explicit permission; university GPU being sought. Now in `CLAUDE.md`, the setup guides, `configs/` (`compute.device: cpu`), charter, SRS, R-14 and `docs/09` §1.4 — `8dced77` *(refined the same day into D-7's compute placement, §6)*
- [x] Compute placement set by the user (**D-7**), plus **D-4** and **D-6**: runbook `docs/11`, `scripts/check_gpu.py` (the user's manual GPU check), `.venv-gpu` gitignored, and the rules in `CLAUDE.md` and the setup guides — `d3683a7`
- [ ] ~~Full `pip install -r requirements.txt` (D-4); commit `requirements.lock.txt`~~ → replaced by per-task installs (D-4). Each cloud run saves its `pip freeze` next to its outputs
- [ ] ~~Start Docker Desktop → `docker compose up -d` → confirm Neo4j Browser at `localhost:7474` (D-6)~~ → **the user creates the AuraDB Free instance** and fills `.env` (`docs/11` §5); Claude then connects from the 1b Neo4j store task
- [ ] ~~`ollama pull llama3.1:8b`; smoke test (D-5)~~ → moved to **3b** (D-5 deferred; laptop GPU tests are the user's, `docs/11` §3)
- [x] ~~Optional, the user, whenever convenient:~~ Laptop-GPU setup and the first checks (`docs/11` §2–§3), **run by Claude at the user's request** on 2026-09-18: `.venv-gpu` with torch 2.11.0+cu128 · `check_gpu.py --device cuda` ✅ (`sm_120`, 6.6 TFLOP/s) · the test suite passes inside it · Ollama smoke test ✅, 100% GPU. Also aligned one line of `check_gpu.py`'s report — `e956594`
- [x] Record **D-9** (set by the user): Claude runs each laptop-GPU test only after the user's yes in chat. Updated in `CLAUDE.md`, `docs/11`, and every doc that described the old rule (README, CONTRIBUTING, `docs/00`/`06`/`07`/`09`, `configs/`, `check_gpu.py`) — → pending
- [ ] Optional: pre-commit hooks for black + ruff

**1a — Data & class balance · 🔄** · P2
- [x] DDXPlus `validate.csv` downloaded (D-1, 87 MB); `train.csv` / `test.csv` deferred until training — `843c5fb`
- [x] EXP-002 on validate: R-03 not triggered (rarest ≈10,880 projected training cases; 2.7× imbalance); **R-13 opened** (closed-world) — `843c5fb`
- [x] Decode patient rows; filter to the 13 conditions → `data/interim/ddxplus_chestpain_validate.parquet` (33,963 rows; one file per split, and the builder refuses the test split). `src/ddxplus.py` + `scripts/build_ddxplus_chestpain.py` + 35 tests. Label audit **EXP-013** → R-13 extended, **D-8** opened — `4d640a8`
- [ ] Build the train parquet when `train.csv` is downloaded: `scripts/build_ddxplus_chestpain.py --split train`. **Ask where first** (D-7). The download is large, and the parquet is needed wherever training runs, probably the cloud
- [ ] Re-confirm EXP-002 counts on `train.csv` once it is downloaded

**1b — Cardiac medical KG · 🔄** · P1
- [x] KG loader from `data/interim/ddxplus_chestpain_conditions.json` → a backend-neutral `KnowledgeGraph` (`src/medical_kg/loader.py`): 14 conditions, 84 `DDX:E_nn` evidence nodes, 245 edges, **each with a `source`** (R-12). KG card in `docs/02` §5.1 — `6709697`
- [x] NetworkX `GraphStore` (`src/medical_kg/networkx_store.py`): a drop-in for `InMemoryGraphStore`, proven by running the pipeline on it in a test. 20 tests — `6709697`
- [ ] Crosswalk: `DDX:E_nn` ↔ the `SYM:*` / `RF:*` ids the red-flag rules and golden cases use. Needed before the pipeline can switch from the stub store
- [ ] BODHI-S enrichment for MI, pericarditis, PE, GERD (likelihood edge weights, `source = bodhi_s`)
- [ ] Hand-author aortic dissection into the KG (`source = hand_authored`)
- [ ] Neo4j-backed `GraphStore` from the same `KnowledgeGraph`, with NetworkX as the fallback (**D-6**)

**1c — Baseline ML ranker · ⬜** · P2
- [ ] Feature encoding from decoded evidence **codes** (not English labels), handling the three token
  kinds EXP-002 found: categorical codes · **numeric ordinal scales** (12.6% of tokens, e.g. pain
  intensity `E_56_@_4`) as ordered features, not one-hot · the `V_11` "NA" sentinel, explicitly
- [ ] EXP-003: B0 prevalence baseline
- [ ] EXP-004: B1 LogReg → XGBoost; `ConditionRanker` replacing `ConstantRanker`. **Every training or
  tuning run on project data, even a small sample, waits for the user's OK and a choice of where**
  (D-7). Unit tests that fit a toy model on a few synthetic rows are tests, not training runs

**1d — Patient KG + Synthea · ⬜** · P3
- [ ] Synthea cardiac-module generation (fixed seed)
- [ ] Patient KG builder
- [ ] Start the PubMed/PMC corpus fetch — slow, so begin now; used in Phase 3

**1e — Platform · ⬜** · P4
- [ ] FastAPI: `POST /diagnose`, `GET /conditions`, `GET /health`
- [ ] Evaluation harness scaffold (`src/eval/metrics.py`)
- [ ] Re-verify golden cases against real components — **fix the component, not the expectation**

### Phase 2 — Reasoning, Fusion & Prototype · ⬜
**Exit criteria:** 🎯 **W5 milestone** — working prototype in the UI, tagged `v0.1-prototype`, demo video recorded.
- **2a** · P1 — KG scoring (overlap → Personalised PageRank) + reasoning paths · EXP-005.
  **Must fix:** the current overlap ranks stable angina above must-not-miss unstable angina even with
  rest pain present (`docs/02` §5.1, limitation 2). Add that case as a regression test once fixed
- **2b** · P2 — Calibration (Platt vs isotonic) + SHAP · EXP-007
- **2c** · P4 — Fusion layer, ✓/?/✗ analysis, weight sweep on validation only · EXP-006
- **2d** · P3 — Red-flag rules on real concepts + safety layer v1 · EXP-008. Three
  must-not-miss conditions have **no rule yet**: unstable angina, myocarditis, acute pulmonary edema
- **2e** · P4 — Streamlit dashboard v1 (must render `degraded_components` and the disclaimer)

### Phase 3 — Evidence & Explanation · ⬜
**Exit criteria:** 🎯 **W7 milestone** — feature-complete against the README architecture.
- **3a** · P3 — Corpus → chunk → embed → FAISS; BM25 hybrid; rerank · EXP-011
- **3b** · P3 — Grounded Ollama explainer; baselines B3/B4 · EXP-009/010. *Carried from 1.0:*
  pull the model where it will run (D-5). The batch B3/B4 runs go to the cloud or the university
  GPU, and the user picks where (D-7)
- **3c** · P4 — Guardrails: claim-support checking, citation rendering
- **3d** · P1 — "What to ask next" (troponin / ECG / D-dimer)
- **3e** · P2 — Calibration curves + per-condition error analysis

### Phase 4 — Evaluation, Ablations & Write-up · ⬜
**Exit criteria:** every metric in `docs/05` reported with a CI · report submitted · `v1.0` tagged.
- **4a** — Open the test split **once**; full ablation A0–A6 · EXP-012
- **4b** — Bootstrap CIs, McNemar, per-condition and by-sex breakdowns
- **4c** — Final report / thesis, including the R-12 circularity disclosure
- **4d** — Review presentation + demo video
- **4e** — Compliance checklists (`docs/03` §7, `docs/04` §8); tag `v1.0`

### Buffer (Week 10) · ⬜
Contingency first; stretch only if genuinely ahead: UCI Heart risk sub-model · KG embeddings · PTB-XL ECG.
**Cut line if time runs short:** PTB-XL / embeddings → "what to ask next" → hybrid retrieval → LLM
explainer. **Never cut:** the W5 prototype, the red-flag layer, the ablation study.

---

## 6. 🔒 Locked decisions — do not re-litigate

| Date | Decision | Recorded in |
|---|---|---|
| 2026-09-17 | Scope: acute chest pain — 13 trainable conditions + aortic dissection (KG-only) | README, `src/conditions.py` |
| 2026-09-17 | Name: Nightingale — Clinical Intelligence & Reasoning System | README |
| 2026-09-17 | Open / synthetic data only — no MIMIC or credentialed data | `docs/03` |
| 2026-09-17 | Stack: Neo4j (NetworkX fallback), local Ollama LLM, FAISS, Streamlit, scikit-learn/XGBoost; no DVC | `docs/02`, approved plan |
| 2026-09-17 | KG backbone: DDXPlus `release_conditions.json`; BODHI-S enrichment only (R-01 fallback) | `docs/10` |
| 2026-09-17 | Headline metric: must-not-miss recall | `docs/05` |
| 2026-09-17 | Evaluation protocol frozen at v1.1 | `docs/05` |
| 2026-09-18 | **D-1:** download DDXPlus `validate.csv` only (87 MB) now; train/test CSVs when training starts | user · §8 |
| 2026-09-18 | **D-2:** the team standardises on **Python 3.11** (matches CI; safest for scispacy/medspaCy) | user · §8 |
| 2026-09-18 | **D-3:** reference docs archived to `docs/archive/`; the byte-identical duplicate prompt deleted | user · `docs/archive/README.md` |
| 2026-09-18 | ~~**GPU policy:** the laptop GPU (RTX 5060) is used **only with the user's explicit permission, asked before each use**; the team is seeking a university GPU. Which machine does GPU work stays open as D-7~~ → superseded the same day by **D-7** below | user · `CLAUDE.md`, R-14 |
| 2026-09-18 | **D-7, compute placement:** the **laptop GPU is for tests only**, ~~set up and run by the user by hand from Claude's steps~~ → who runs them is refined by **D-9** below (`docs/11` §2–§3). **Main training runs on the university GPU and the cloud** (Google Colab free · Kaggle · Lightning AI / Studio Lab). Claude **asks where, per job**, before every training or tuning run and any job expected to take **over ~5 minutes, CPU jobs included**, and suggests the cloud | user · `CLAUDE.md`, `docs/11` |
| 2026-09-18 | **D-4:** packages go onto the laptop **per task, as needed**, not the full ~1 GB stack. Next: scikit-learn + XGBoost (~200 MB) when 1c starts, asked at that time | user · `docs/11` §1 |
| 2026-09-18 | **D-6:** Neo4j runs on **AuraDB Free** (cloud). The user creates the account and keeps its credentials in `.env`; no Docker on the laptop. NetworkX stays the fallback | user · `docs/11` §5, R-06 |
| 2026-09-18 | **D-9, laptop-GPU tests:** Claude runs each short laptop-GPU test (a CUDA check, a short model load, any Ollama model) **only after the user says yes in chat**, and a yes covers one test. The user can still run tests by hand. Claude checks the laptop is on the charger, frees the GPU afterwards and reports the result | user · `CLAUDE.md`, `docs/11` §1–§3 |

---

## 7. 🧰 Environment state — verified 2026-09-18

| Item | State |
|---|---|
| Git | clean · the repository is **public** on GitHub, so cloud notebooks clone it without a token |
| CI (GitHub Actions) | ✅ green on every push so far (latest verified: `e956594`) · Python 3.11 |
| Local Python | **3.11.9** in `.venv` (D-2) — light deps only: `requirements-dev.txt` (now incl. pandas, numpy, pyarrow, **networkx 3.6.1**) + huggingface_hub. Full install is D-4 |
| Machine Pythons | 3.14 (**still the default** — plain `python` bypasses the venv), 3.12, 3.11 · always use `./.venv/Scripts/python.exe` |
| Docker | 29.7.2 installed, **not needed**: Neo4j runs on AuraDB Free (D-6). `docker-compose.yml` stays as the optional local route |
| Neo4j | **AuraDB Free**: instance not yet created (the user, `docs/11` §5) |
| Ollama | **0.34.2** installed (it updates itself; it was 0.34.1) · only `qwen2.5-coder:7b` pulled — a coding model, not the configured `llama3.1:8b` · runs models **on the GPU by default**. Smoke test on 2026-09-18 ✅: 100% GPU, ~71 tokens/s, first prompt 13.6 s, then 0.3 s (`docs/11` §3). Because it uses the GPU, **each run needs the user's yes first** (D-9) |
| GPU | NVIDIA RTX 5060 Laptop · 8 GB VRAM · Blackwell (`sm_120`; torch ≥ 2.7 / CUDA 12.8) · driver 591.91 (CUDA 13.1) · **`.venv-gpu` set up 2026-09-18** (4.3 GB on D:): torch 2.11.0+cu128, `check_gpu.py --device cuda` ✅ 6.6 TFLOP/s float32 (`docs/11` §2) · **tests only**, each run by Claude only after the user's yes (D-9) · `compute.device: cpu` in `configs/` |
| Heavy compute | **Google Colab (free), Kaggle Notebooks, Lightning AI / Studio Lab**, chosen per job by the user (`docs/11` §4) · university GPU not yet granted (R-14) |
| Data on disk (gitignored) | `data/raw/ddxplus/release_*.json` + **`validate.csv`** (87 MB) · `data/raw/bodhi_s/*.jsonl` · `data/interim/ddxplus_*.json`, `exp002_class_balance.json`, **`ddxplus_chestpain_validate.parquet`** (4.6 MB) + `.summary.json` · `train.csv` / `test.csv` **not downloaded** |
| CI dependency set | `requirements-dev.txt`: the tools plus pydantic, pyyaml, pandas, numpy, pyarrow and networkx. These moved from `requirements.txt` so CI runs the parquet-builder and KG tests (CI went from ~15 s to ~35 s) |

Re-verify this table whenever the environment changes, and date it.

---

## 8. 📓 Session log — newest first, one line per session

| Date | Who | What happened | Commits |
|---|---|---|---|
| 2026-09-18 | Claude | The user decided **D-9**: Claude runs each laptop-GPU test only after the user says yes in chat. Recorded in `CLAUDE.md`, `docs/11` (v1.1) and every doc that still said the user runs the tests by hand | → pending |
| 2026-09-18 | Claude | The user asked Claude to run the laptop-GPU setup and tests itself ("just run and complete the setup and test yourself"). Claude treated that as covering these tests only: `.venv-gpu` with torch 2.11.0+cu128 (2.75 GB download, no pip cache kept), `check_gpu.py --device cuda` ✅ (`sm_120`, 6.6 TFLOP/s), the test suite passes inside it, Ollama smoke test ✅ at 100% GPU, and the model was unloaded afterwards. `docs/11` now records the results. Whether Claude may run GPU tests from now on is **D-9** | `e956594` |
| 2026-09-18 | Claude | The user set compute placement (**D-7**): the laptop GPU for tests only, run by the user by hand; training runs, and any job over ~5 min (CPU included), go to the cloud (Colab free · Kaggle · Lightning AI / Studio Lab) or the university GPU, asked per job. Also **D-4** (per-task installs) and **D-6** (Neo4j on AuraDB Free). New runbook `docs/11` with the user's GPU setup and start/stop steps; `scripts/check_gpu.py`; R-14 re-scoped, R-06 mitigated | `d3683a7` |
| 2026-09-18 | Claude | Recorded the user's GPU policy as **D-7** (laptop GPU only with permission; university GPU sought) and **R-14**; D-4 re-framed to CPU torch, D-5 deferred to D-7; torch pin fixed. **1a:** validate parquet (33,963 rows) + label audit **EXP-013**. Found that a listed token can mean "no", and that the ground-truth differentials are open-world (Recall@5 ceiling 0.434) → **D-8**. **1b:** KG loader + NetworkX store, with edge provenance for R-12. KG card: questions not answers; stable ⊂ unstable angina, so overlap ranks the benign one first. Corrected the stale README/charter claims that the KG is built from BODHI-S. Protocol note: the session was interrupted once, mid-way through updating this file for task 2, and recovered from §2 | `8dced77` `4d640a8` `6709697` |
| 2026-09-18 | Claude | Applied user decisions D-1–D-3: archived the reference docs (D-3); Python 3.11 + `requirements-dev.txt`, fixing local/CI tool drift (D-2); `validate.csv` + EXP-002 — R-03 resolved, **R-13 opened** (D-1). Protocol lesson: §1 was not refreshed at the two intermediate commits — fixed, and §4.2 now requires it | `5ce3725` `1ccb06a` `843c5fb` |
| 2026-09-18 | Claude | Added `PROGRESS.md` + `CLAUDE.md` session protocol; verified the environment; corrected Phase 0 status — Neo4j/Ollama exit criteria were never met, now carried to 1.0 | `51b49ce` |
| 2026-09-17 | Claude | Phase 0c: R-01 spike (31% → FALLBACK), R-12 opened, eval protocol frozen | `8b682e9` |
| 2026-09-17 | Claude | Phase 0b: scaffold + walking skeleton, 33 tests, CI | `efc10e9` |
| 2026-09-17 | Claude | Phase 0a: 11 planning documents | `26240f5` `1a47a0e` `a842c0f` |
| 2026-09-15–17 | Claude | Scope locked to chest pain; renamed Nightingale; repo created | `f09be7f` `d612336` |
