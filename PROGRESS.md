# PROGRESS — Nightingale

> **The single source of truth for where this project is.** Auto-loaded into every Claude Code session
> through `CLAUDE.md`. Read §1 and §2 first. **Updating this file is part of every task** — the rules
> are in §4 and they are mandatory. If this file and the repository disagree, stop and reconcile
> (§4.1) before doing any new work.

**Last updated:** 2026-09-18 · by Claude (D-7 recorded; 1a validate parquet + EXP-013; 1b KG loader + NetworkX store) · **Last verified commit:** `6709697`

---

## 1. 📍 Current position

| | |
|---|---|
| **Completed phase** | Phase 0 — Preparation & Documentation ✅ *(environment items carried over to 1.0 — see §5)* |
| **Current phase** | **Phase 1 — Data & Knowledge Foundations** |
| **Current sub-phase** | **1.0 — Environment bring-up** · 🔄 (Python 3.11 ✅ · torch pin ✅ · GPU policy D-7 recorded; installs await D-4/D-6) · **1a — Data** · 🔄 (validate parquet ✅ + EXP-013; train awaits download) · **1b — KG** · 🔄 (loader + NetworkX store ✅; crosswalk, BODHI-S, aortic dissection, Neo4j to go) |
| **Next action** | Ask the user about **D-4** and **D-6** (downloads), and take **D-8** to the team. Unblocked meanwhile, with no downloads: the 1b **crosswalk** (`DDX:E_nn` ↔ `SYM:*`), then hand-authoring **aortic dissection** and **BODHI-S enrichment** (its data is on disk) |
| **Blocked on** | D-4 (full install, ~1 GB with CPU torch) · D-6 (Neo4j image, ~0.5 GB) — need permission · D-5 (LLM pull) waits on **D-7**: where GPU work runs. A university GPU is being sought; **the laptop GPU is used only with the user's permission** · **D-8** (definition of D) blocks any Precision@3 / Recall@5 number |
| **Schedule** | Week 1 of 8–10 · ahead of plan · next milestone 🎯 W3 walking skeleton, target 2026-10-07 |
| **Health** | 89 tests passing locally (86 in CI, where the 3 real-data tests skip) · CI green on GitHub at `6709697` |

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
plain `python` is 3.14, not the team's 3.11. **Never use the laptop GPU without asking the user
first** (D-7). That includes running any model through Ollama, which uses the GPU automatically.

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
| **D-4** | Full `pip install -r requirements.txt` into the 3.11 venv — **lengthy, needs permission** | **CPU-only torch** (~1 GB in all) · CUDA 12.8 torch (~3 GB in all, and only useful if D-7 approves the laptop GPU) | **CPU-only torch** (`--index-url …/whl/cpu`), then the rest. *Re-framed 2026-09-18:* nothing before Phase 3 needs a GPU, since XGBoost trains on the CPU. The CUDA build goes on whichever machine D-7 picks | 1.0 · 1c ranker onwards |
| **D-5** | `ollama pull llama3.1:8b` — **lengthy (~4.9 GB), needs permission** | llama3.1:8b · qwen2.5:7b-instruct · reuse the installed `qwen2.5-coder:7b` | **llama3.1:8b**, a general instruct model (the coder model is tuned for code, not clinical prose). **Defer until D-7:** the LLM is Phase 3 (Week 6), and Ollama runs any model on the GPU automatically, so even a smoke test on the laptop would use the laptop GPU. Pull it where it will be served | Phase 3 explainer |
| **D-6** | Start Docker Desktop and pull the `neo4j:5-community` image — **lengthy (~0.5 GB), needs permission** | Neo4j now · NetworkX only until 1b needs a real graph DB | **Neo4j now** — Docker setup problems are better found early (R-06). No GPU involved. The NetworkX backend now works (1b), so nothing is hard-blocked meanwhile | 1.0 · 1b Neo4j store |
| **D-7** 🆕 | **Where GPU work runs** (LLM + embeddings, Phase 3). *Set by the user 2026-09-18:* the team is seeking a **university GPU**, and the **laptop RTX 5060 is used only with the user's explicit permission, asked before each use** | University GPU, if granted · laptop RTX 5060, with permission · CPU only (slow LLM; the templated explainer is the fallback) | Waiting on the university. **Decide by 2026-10-14** (end of Week 4) so Phase 3 (from 2026-10-22) is not blocked. If there is no university GPU by then, ask the user about the laptop GPU for Phase 3 (risk R-14) | D-5 · 3a embeddings · 3b LLM · any CUDA install |
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
- [x] Record the GPU policy (**D-7**, set by the user): laptop GPU only with explicit permission; university GPU being sought. Now in `CLAUDE.md`, the setup guides, `configs/` (`compute.device: cpu`), charter, SRS, R-14 and `docs/09` §1.4 — `8dced77`
- [ ] Full `pip install -r requirements.txt` (D-4 — **ask first**; CPU torch recommended); commit `requirements.lock.txt`
- [ ] Start Docker Desktop → `docker compose up -d` → confirm Neo4j Browser at `localhost:7474` (**D-6**)
- [ ] `ollama pull llama3.1:8b` (the configured model; only `qwen2.5-coder:7b` is present); smoke test (**D-5**, deferred until **D-7** — a smoke test uses the GPU)
- [ ] Optional: pre-commit hooks for black + ruff

**1a — Data & class balance · 🔄** · P2
- [x] DDXPlus `validate.csv` downloaded (D-1, 87 MB); `train.csv` / `test.csv` deferred until training — `843c5fb`
- [x] EXP-002 on validate: R-03 not triggered (rarest ≈10,880 projected training cases; 2.7× imbalance); **R-13 opened** (closed-world) — `843c5fb`
- [x] Decode patient rows; filter to the 13 conditions → `data/interim/ddxplus_chestpain_validate.parquet` (33,963 rows; one file per split, and the builder refuses the test split). `src/ddxplus.py` + `scripts/build_ddxplus_chestpain.py` + 35 tests. Label audit **EXP-013** → R-13 extended, **D-8** opened — `4d640a8`
- [ ] Build the train parquet when `train.csv` is downloaded: `scripts/build_ddxplus_chestpain.py --split train`
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
- [ ] EXP-004: B1 LogReg → XGBoost; `ConditionRanker` replacing `ConstantRanker`

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
- **3b** · P3 — Grounded Ollama explainer; baselines B3/B4 · EXP-009/010
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
| 2026-09-18 | **GPU policy:** the laptop GPU (RTX 5060) is used **only with the user's explicit permission, asked before each use**; the team is seeking a university GPU. Which machine does GPU work stays open as D-7 | user · `CLAUDE.md`, R-14 |

---

## 7. 🧰 Environment state — verified 2026-09-18

| Item | State |
|---|---|
| Git | clean — reference docs archived to `docs/archive/` (D-3) |
| CI (GitHub Actions) | ✅ green on every push so far (`efc10e9`, `8b682e9`, `51b49ce`) · Python 3.11 |
| Local Python | **3.11.9** in `.venv` (D-2) — light deps only: `requirements-dev.txt` (now incl. pandas, numpy, pyarrow, **networkx 3.6.1**) + huggingface_hub. Full install is D-4 |
| Machine Pythons | 3.14 (**still the default** — plain `python` bypasses the venv), 3.12, 3.11 · always use `./.venv/Scripts/python.exe` |
| Docker | 29.7.2 installed · **daemon not running** — start Docker Desktop |
| Neo4j | never started |
| Ollama | 0.34.1 installed · only `qwen2.5-coder:7b` pulled — a coding model, not the configured `llama3.1:8b` · runs models **on the GPU by default**, so running one is laptop-GPU use |
| GPU | NVIDIA RTX 5060 Laptop · 8 GB VRAM · Blackwell (torch ≥ 2.7 / CUDA 12.8) · **use only with the user's permission** (D-7); a university GPU is being sought · `compute.device: cpu` in `configs/` |
| Data on disk (gitignored) | `data/raw/ddxplus/release_*.json` + **`validate.csv`** (87 MB) · `data/raw/bodhi_s/*.jsonl` · `data/interim/ddxplus_*.json`, `exp002_class_balance.json`, **`ddxplus_chestpain_validate.parquet`** (4.6 MB) + `.summary.json` · `train.csv` / `test.csv` **not downloaded** |
| CI dependency set | `requirements-dev.txt`: the tools plus pydantic, pyyaml, pandas, numpy, pyarrow and networkx. These moved from `requirements.txt` so CI runs the parquet-builder and KG tests (CI went from ~15 s to ~35 s) |

Re-verify this table whenever the environment changes, and date it.

---

## 8. 📓 Session log — newest first, one line per session

| Date | Who | What happened | Commits |
|---|---|---|---|
| 2026-09-18 | Claude | Recorded the user's GPU policy as **D-7** (laptop GPU only with permission; university GPU sought) and **R-14**; D-4 re-framed to CPU torch, D-5 deferred to D-7; torch pin fixed. **1a:** validate parquet (33,963 rows) + label audit **EXP-013**. Found that a listed token can mean "no", and that the ground-truth differentials are open-world (Recall@5 ceiling 0.434) → **D-8**. **1b:** KG loader + NetworkX store, with edge provenance for R-12. KG card: questions not answers; stable ⊂ unstable angina, so overlap ranks the benign one first. Corrected the stale README/charter claims that the KG is built from BODHI-S. Protocol note: the session was interrupted once, mid-way through updating this file for task 2, and recovered from §2 | `8dced77` `4d640a8` `6709697` |
| 2026-09-18 | Claude | Applied user decisions D-1–D-3: archived the reference docs (D-3); Python 3.11 + `requirements-dev.txt`, fixing local/CI tool drift (D-2); `validate.csv` + EXP-002 — R-03 resolved, **R-13 opened** (D-1). Protocol lesson: §1 was not refreshed at the two intermediate commits — fixed, and §4.2 now requires it | `5ce3725` `1ccb06a` `843c5fb` |
| 2026-09-18 | Claude | Added `PROGRESS.md` + `CLAUDE.md` session protocol; verified the environment; corrected Phase 0 status — Neo4j/Ollama exit criteria were never met, now carried to 1.0 | `51b49ce` |
| 2026-09-17 | Claude | Phase 0c: R-01 spike (31% → FALLBACK), R-12 opened, eval protocol frozen | `8b682e9` |
| 2026-09-17 | Claude | Phase 0b: scaffold + walking skeleton, 33 tests, CI | `efc10e9` |
| 2026-09-17 | Claude | Phase 0a: 11 planning documents | `26240f5` `1a47a0e` `a842c0f` |
| 2026-09-15–17 | Claude | Scope locked to chest pain; renamed Nightingale; repo created | `f09be7f` `d612336` |
