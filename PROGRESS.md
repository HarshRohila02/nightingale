# PROGRESS — Nightingale

> **The single source of truth for where this project is.** Auto-loaded into every Claude Code session
> through `CLAUDE.md`. Read §1 and §2 first. **Updating this file is part of every task** — the rules
> are in §4 and they are mandatory. If this file and the repository disagree, stop and reconcile
> (§4.1) before doing any new work.

**Last updated:** 2026-09-18 · by Claude (progress-tracker setup) · **Last verified commit:** `51b49ce`

---

## 1. 📍 Current position

| | |
|---|---|
| **Completed phase** | Phase 0 — Preparation & Documentation ✅ *(environment items carried over to 1.0 — see §5)* |
| **Current phase** | **Phase 1 — Data & Knowledge Foundations** |
| **Current sub-phase** | **1.0 — Environment bring-up** · ⬜ not started |
| **Next action** | Get the user's answers to **D-1** and **D-2** (§3), then start Docker Desktop and bring up Neo4j |
| **Blocked on** | User decisions D-1 (CSV download) and D-2 (Python version) |
| **Schedule** | Week 1 of 8–10 · ahead of plan · next milestone 🎯 W3 walking skeleton, target 2026-10-07 |
| **Health** | 33 tests passing locally · CI green on GitHub at `51b49ce` |

**Handoff note for the next session:** Phase 0 is done and nothing is in flight. Before touching the
knowledge graph, read `docs/10-spike-r01-crosswalk.md` — the KG is built from DDXPlus
`release_conditions.json`, **not** BODHI-S, and that choice creates a circularity risk (R-12) that
must be disclosed in results. The evaluation protocol (`docs/05`) is **frozen**.

---

## 2. 🛫 In-flight (write-ahead marker)

> Filled in **before** a task starts; cleared **after** it is committed. If this section is not empty
> when a session begins, the previous session was interrupted — go to §4.4.

- **Task:** Apply user decisions **D-1** (download DDXPlus `validate.csv`, run EXP-002),
  **D-2** (Python 3.11), **D-3** (archive the reference docs)
- **Sub-phase:** 1.0 (D-2, D-3) and 1a (D-1)
- **Started:** 2026-09-18 by Claude
- **Files expected to change:** `PROGRESS.md` · root reference docs → `docs/archive/` (+ delete the
  byte-identical duplicate) · `docs/archive/README.md` · `.venv` (rebuilt on 3.11, gitignored) ·
  `data/raw/ddxplus/validate.csv` (gitignored) · `scripts/class_balance.py` ·
  `docs/08-experiment-log.md` · `docs/07-risk-register.md`
- **Done so far:** decisions approved by the user 2026-09-18; duplicate re-verified identical
  (SHA-256 `fc0212c3…`); Python 3.11 confirmed not installed (only 3.14, 3.12) ·
  ① **D-3 done** — 3 docs archived and renamed, duplicate deleted, `docs/archive/README.md` written ·
  ② **D-2 done** — Python 3.11.9 installed, `.venv` rebuilt and verified; `requirements-dev.txt`
  added to stop local/CI tool drift
- **Remaining:** ③ D-1 download `validate.csv`, EXP-002 → commit · then clear this marker
- **Safe to resume blindly?** No — check `git log` for which of ①②③ committed, and run
  `./.venv/Scripts/python.exe --version` to see whether the venv is already 3.11.

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
| **D-4** | Full `pip install -r requirements.txt` into the 3.11 venv — **lengthy (~2–3 GB), needs permission** | CPU-only torch · CUDA 12.8 torch for the RTX 5060 | **CUDA 12.8 torch first** (`--index-url …/whl/cu128`), then the rest | 1.0 · all real modelling |
| **D-5** | `ollama pull llama3.1:8b` — **lengthy (~4.9 GB), needs permission** | llama3.1:8b · qwen2.5:7b-instruct · reuse the installed `qwen2.5-coder:7b` | **llama3.1:8b** — a general instruct model; the coder model is tuned for code, not clinical prose | 1.0 · Phase 3 explainer |

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
commit. If the task will take a long time or download something large, ask the user first.

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
| 1 — Data & knowledge foundations | 2–3 | 2026-10-07 | ⬜ |
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

### Phase 1 — Data & Knowledge Foundations → Walking Skeleton · ⬜
**Exit criteria:** 🎯 **W3 milestone** — a case produces a ranked differential from the **real** ML
ranker with **real** KG-matched supporting findings, end to end · CI green.

**1.0 — Environment bring-up · 🔄** *(carried over from Phase 0)*
- [x] Archive the reference docs to `docs/archive/`; delete the byte-identical duplicate (D-3) — `→ pending`
- [x] Python 3.11 (D-2): `py install 3.11` → 3.11.9; `.venv` rebuilt; tests, black, ruff green — `→ pending`
- [x] Fix local-vs-CI tool drift: new `requirements-dev.txt` is the single source of truth for pytest/black/ruff; CI and `requirements.txt` both read it — `→ pending`
- [ ] Full `pip install -r requirements.txt` (D-4 — **ask first**); commit `requirements.lock.txt`
- [ ] Fix the torch pin in `requirements-nlp.txt`: `~=2.4` cannot use the RTX 5060 (Blackwell needs torch ≥ 2.7 / CUDA 12.8)
- [ ] Start Docker Desktop → `docker compose up -d` → confirm Neo4j Browser at `localhost:7474`
- [ ] `ollama pull llama3.1:8b` (the configured model; only `qwen2.5-coder:7b` is present) — **lengthy, ask first**; smoke test
- [ ] Optional: pre-commit hooks for black + ruff

**1a — Data & class balance · ⬜** · P2
- [ ] DDXPlus patient CSVs (D-1) — **lengthy, ask first**
- [ ] Decode patient rows; filter to the 13 conditions → `data/interim/ddxplus_chestpain.parquet`
- [ ] EXP-002: per-condition counts per split; flag any condition with < 500 training cases (R-03)

**1b — Cardiac medical KG · ⬜** · P1
- [ ] KG loader from `data/interim/ddxplus_chestpain_conditions.json`
- [ ] BODHI-S enrichment for MI, pericarditis, PE, GERD (likelihood edge weights)
- [ ] Hand-author aortic dissection into the KG
- [ ] Neo4j-backed `GraphStore` (NetworkX fallback) replacing `InMemoryGraphStore`
- [ ] Crosswalk: DDXPlus evidence codes ↔ the `SYM:*` concept ids the red-flag rules use

**1c — Baseline ML ranker · ⬜** · P2
- [ ] Feature encoding from decoded evidence **codes** (not English labels)
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
- **2a** · P1 — KG scoring (overlap → Personalised PageRank) + reasoning paths · EXP-005
- **2b** · P2 — Calibration (Platt vs isotonic) + SHAP · EXP-007
- **2c** · P4 — Fusion layer, ✓/?/✗ analysis, weight sweep on validation only · EXP-006
- **2d** · P3 — Red-flag rules on real concepts + safety layer v1 · EXP-008
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

---

## 7. 🧰 Environment state — verified 2026-09-18

| Item | State |
|---|---|
| Git | clean — reference docs archived to `docs/archive/` (D-3) |
| CI (GitHub Actions) | ✅ green on every push so far (`efc10e9`, `8b682e9`, `51b49ce`) · Python 3.11 |
| Local Python | **3.11.9** in `.venv` (D-2) — light deps only: `requirements-dev.txt` + pandas, numpy, pyarrow, huggingface_hub. Full install is D-4 |
| Machine Pythons | 3.14 (**still the default** — plain `python` bypasses the venv), 3.12, 3.11 · always use `./.venv/Scripts/python.exe` |
| Docker | 29.7.2 installed · **daemon not running** — start Docker Desktop |
| Neo4j | never started |
| Ollama | 0.34.1 installed · only `qwen2.5-coder:7b` pulled — a coding model, not the configured `llama3.1:8b` |
| GPU | NVIDIA RTX 5060 Laptop · 8 GB VRAM · Blackwell (torch ≥ 2.7 / CUDA 12.8) |
| Data on disk (gitignored) | `data/raw/ddxplus/release_*.json`, `data/raw/bodhi_s/*.jsonl`, `data/interim/ddxplus_*.json` · **patient CSVs not downloaded** |

Re-verify this table whenever the environment changes, and date it.

---

## 8. 📓 Session log — newest first, one line per session

| Date | Who | What happened | Commits |
|---|---|---|---|
| 2026-09-18 | Claude | Added `PROGRESS.md` + `CLAUDE.md` session protocol; verified the environment; corrected Phase 0 status — Neo4j/Ollama exit criteria were never met, now carried to 1.0 | `51b49ce` |
| 2026-09-17 | Claude | Phase 0c: R-01 spike (31% → FALLBACK), R-12 opened, eval protocol frozen | `8b682e9` |
| 2026-09-17 | Claude | Phase 0b: scaffold + walking skeleton, 33 tests, CI | `efc10e9` |
| 2026-09-17 | Claude | Phase 0a: 11 planning documents | `26240f5` `1a47a0e` `a842c0f` |
| 2026-09-15–17 | Claude | Scope locked to chest pain; renamed Nightingale; repo created | `f09be7f` `d612336` |
