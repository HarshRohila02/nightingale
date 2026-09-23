# PROGRESS — Nightingale

> **The single source of truth for where this project is.** Auto-loaded into every Claude Code session
> through `CLAUDE.md`. Read §1 and §2 first. **Updating this file is part of every task** — the rules
> are in §4 and they are mandatory. If this file and the repository disagree, stop and reconcile
> (§4.1) before doing any new work.

**Last updated:** 2026-09-23 · by Claude (1c: the ranker seam — the real model now ranks behind the pipeline) · **Last verified commit:** `98c746b`

---

## 1. 📍 Current position

| | |
|---|---|
| **Completed phase** | Phase 0 — Preparation & Documentation ✅ *(environment items carried over to 1.0 — see §5)* |
| **Current phase** | **Phase 1 — Data & Knowledge Foundations** |
| **Current sub-phase** | **1.0 — Environment bring-up** ✅ 2026-09-19 · **1a — Data** ✅ 2026-09-20 (validate parquet + EXP-013 on the laptop; the train split built on Colab inside the B0/B1 job; EXP-002 confirmed on the train counts) · **1b — KG** ✅ 2026-09-19 (DDXPlus + hand-authored aortic dissection + BODHI-S, NetworkX and Neo4j stores, crosswalk) · **1c — Baseline ML ranker** · 🔄 (feature encoding ✅; B0 and B1 trained and scored ✅, EXP-003/004; `ConditionRanker` ✅ 2026-09-23 — the case → token inversion, then the model behind the pipeline) |
| **Next action** | **Claude:** the last piece of phase A — move the scoring helpers out of `scripts/train_baselines.py` into `src/eval/reports.py`, so the baseline job and the deep job produce comparable reports. **Then phase A is done and the deep work waits on the team** (gate 1: **D-10**, **D-11**, **A-8**, `docs/05` amendment 3, and the clinical review of the 17 representative answers). **For the user:** GC-005, a proposed fifth golden case (a benign-looking ischaemic presentation that raises no red flag), and whether the model bundle moves to `models/b0_b1/`. **The team:** decide **D-10** (§3) before the fusion work (2c): B1 already reaches the ceiling of DDXPlus's headline metrics (R-16, EXP-004). **The user, when convenient:** may the bundle move from `models/nightingale_b0_b1/` to `models/b0_b1/`, where the docs expect it? |
| **Blocked on** | The "asked" channel that R-18 needs ← a retraining run, so the user's choice of where (D-7) · how the fusion is evaluated (2c) ← **D-10**, the team · Phase 3 LLM ← D-5 (deferred) · university GPU access (external, R-14). **Every training or tuning run, and any job over ~5 min, waits for the user to say where** (D-7) |
| **Schedule** | Week 1 of 8–10 · ahead of plan · 🎯 **W3 walking skeleton substantially reached 2026-09-23** (target 2026-10-07): a case is ranked by the real model and the real graph end to end, `scripts/demo.py` prints it, and `scripts/check_crosswalk.py` runs all four golden cases on the real graph. What is left of it is the real graph behind the demo itself, which is 2a's |
| **Health** | 370 tests (268 + 79 for the inversion + 19 for the ranker + 4 golden cases re-run with red flags off), one strict xfail (GC-001 without its red flag, below). Locally 259 pass, and the 9 live Neo4j tests skip unless `NEO4J_TEST_DOTENV=1`; with it, all 9 passed against Aura on 2026-09-19. A data-free copy gives 247 passed and 21 skipped; CI adds a throwaway Neo4j, so 8 of the live tests run there · CI green on GitHub at `68c14bd`: 255 passed, 13 skipped, the 8 live Neo4j tests without real data running against CI's container · the Colab-trained B0/B1 models load on the laptop and reproduce their validate metrics to within 3 × 10⁻¹² |

**Handoff note for the next session:** Nothing is in flight. **B0 and B1 are trained** (the owner's
Colab run, 2026-09-19: EXP-003, EXP-004), and the bundle is in `models/nightingale_b0_b1/` (§7).
**B1 is near-perfect on DDXPlus validate** (XGBoost top-1 0.9985; top-3 and must-not-miss recall@3
both 1.000), because DDXPlus draws each patient's evidence only from their condition's list. On
full-evidence DDXPlus the fusion therefore cannot beat B1 on the headline metrics: **R-16, and
decision D-10 for the team** (§3). **The user asked for the next task (`ConditionRanker`) to wait
until they say go.** On 2026-09-19 the user created the
AuraDB instance (connected; see the §7 Neo4j row), chose **Colab** for the B0/B1 training job and
approved the next installs (§8); the team settled D-8 and four smaller points (§6). **1b is complete:** the KG
is on AuraDB (label `CardiacKG`); `scripts/load_neo4j.py` refreshes it, and `open_graph_store()`
falls back to NetworkX, reporting `graph_backend`, when Aura is paused. The KG now has three
sources:
DDXPlus, the hand-authored aortic dissection and BODHI-S. **The graph alone ranks DDXPlus
patients 88% top-1, and that is circularity (EXP-015, R-12)**, never a result to report on its
own. BODHI-S lowers it to 86%, and MI to 60%, because the overlap score punishes enriched
conditions (EXP-016): 2a must replace the score. The chest-pain parquet is on the laptop for
**validate** only; train was built on Colab (`docs/03` §2.1). EXP-013 found two things everyone must know. First, **a listed
DDXPlus token can mean "no"** (`E_204_@_V_10` = "did not travel"), so use `positive_codes()`.
Second, **the ground-truth differentials are open-world**, so `Recall@5` as first written tops
out at 0.434; **D-8** (decided 2026-09-19) makes the headline count only D's in-scope part. The KG now loads into a NetworkX store, but read its card (`docs/02` §5.1) before
trusting a score. It links conditions to *questions*, not answers, and ranks **stable angina above
must-not-miss unstable angina** even when rest pain is present; 2a must fix this. **The crosswalk
(`docs/02` §5.2) now links the `SYM:*`/`RF:*` concepts to DDXPlus.** It showed (EXP-014) that the
**red-flag rules over-fire** on DDXPlus: 59% of patients get a flag, and the aortic-dissection rule
flags 50% (R-15, a 2d fix). The golden cases pass on the real graph only because red flags rank
first. Before touching the knowledge graph,
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

---

## 3. ❓ Awaiting user decisions

| ID | Decision needed | Options | Recommended | Blocks |
|---|---|---|---|---|
| **D-5** | Which LLM, and pulling it (`ollama pull llama3.1:8b`, ~4.9 GB) | llama3.1:8b · qwen2.5:7b-instruct · reuse the installed `qwen2.5-coder:7b` | **llama3.1:8b**, a general instruct model (the coder model is tuned for code, not clinical prose). **Deferred to Phase 3** (Week 6). Under D-7 and D-9, a laptop test runs only after the user's yes (`docs/11` §3, which uses the installed `qwen2.5-coder:7b`, so no download), and batch runs pull the model on the chosen cloud runtime | Phase 3 explainer |
| **D-10** | How to evaluate the fusion now that B1 reaches the ceiling of DDXPlus's headline metrics (EXP-004, R-16). On full-evidence validate, B1 already scores 1.000 on top-3 and on must-not-miss recall@3, so H1 and H2 cannot be supported there, and `docs/05` §7's *Target* is out of reach | (a) keep the protocol and explain the ceiling in the report · (b) **amend `docs/05`: add a reduced-evidence condition**: each patient keeps their initial evidence plus a fixed random share of the rest (say 25% and 50%), the masks drawn once with seed 42, and every system is scored on full and reduced evidence · (c) add a small independent set of clinical vignettes written by the team, like the golden cases · (b) with (c) | **(b)**, plus (c) if time allows. A partial history is the realistic case, the metrics get room again, and the masks and levels can be fixed before any fusion result is seen, so the amendment stays pre-registered. Training the ranker on masked evidence is a new training run, so where it runs is asked first (D-7) | The fusion evaluation (2c, EXP-006) and the Phase 4 ablation (EXP-012): decide before 2c starts |

| **D-11** | Whether the deep model becomes the **primary** ranker, as the supervisor suggested (2026-09-20) | (a) ratify: a deep ranker replaces B1 as primary, measured against it at every evidence level · (b) keep B1 primary and add the deep model as one more baseline · (c) neither | **(a)**, with the honest caveat now measured: on *full-evidence* DDXPlus no model can beat B1 (R-16), so the deep model will tie there and the report must say so. The place it can win is the partial-history regime, where the two classical models already differ by 0.7 top-1 (EXP-017, R-18). The owner has chosen this direction; the row is here so the team ratifies it | Phase B of the deep-ranker plan. **Nothing deep is trained before this and D-10 are settled** |
| **A-8** | Whether the concept → token inversion may use `Match.NARROWER` crosswalk entries, and the clinical review of the 17 representative answers (docs/02 §9, R-17) | admit them (current setting) · exclude them | **Admit.** Excluding them drops sudden onset, exertional pain and relief by rest, so a golden case reaches the model with nothing to separate embolism, pneumothorax, dissection and the anginas. Every token so derived is recorded | Nothing today; it changes what the model sees |

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
| 1 — Data & knowledge foundations | 2–3 | 2026-10-07 | 🔄 1.0 ✅ · 1a ✅ · 1b ✅ · 1c in progress |
| 2 — Reasoning, fusion & prototype | 4–5 | 2026-10-21 | ⬜ |
| 3 — Evidence & explanation | 6–7 | 2026-11-04 | ⬜ |
| 4 — Evaluation, ablations & write-up | 8–9 | 2026-11-18 | ⬜ |
| Buffer | 10 | 2026-11-25 | ⬜ |

### Phase 0 — Preparation & Documentation · ✅ 2026-09-17
**Exit criteria:** docs drafted ✅ · repo scaffolded ✅ · spike answered with a recorded decision ✅ ·
evaluation protocol frozen ✅ · **Neo4j + Ollama running ⚠️ not met — carried over to 1.0**
*(2026-09-18: Neo4j → AuraDB Free, awaiting the user's instance; Ollama → 3b, by D-5 and D-7.
2026-09-19: the AuraDB instance runs and holds the KG ✅)*

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

**1.0 — Environment bring-up · ✅ 2026-09-19** *(carried over from Phase 0)*
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
- [x] Record **D-9** (set by the user): Claude runs each laptop-GPU test only after the user's yes in chat. Updated in `CLAUDE.md`, `docs/11`, and every doc that described the old rule (README, CONTRIBUTING, `docs/00`/`06`/`07`/`09`, `configs/`, `check_gpu.py`) — `4007807`
- [x] The user created the **AuraDB Free** instance and filled `.env` (`docs/11` §5). Claude checked the login without reading `.env`: connected to Neo4j 5.27 (Aura), empty database. **The home database is named after the instance id, not `neo4j`**, so the store must not hard-code a database name — `e6f6c60`
- [ ] ⏭️ Optional: pre-commit hooks for black + ruff. Deferred 2026-09-19: CI already runs black and ruff on every push; add the hooks if the team wants them

**1a — Data & class balance · ✅ 2026-09-20** · P2
- [x] DDXPlus `validate.csv` downloaded (D-1, 87 MB); `train.csv` / `test.csv` deferred until training — `843c5fb`
- [x] EXP-002 on validate: R-03 not triggered (rarest ≈10,880 projected training cases; 2.7× imbalance); **R-13 opened** (closed-world) — `843c5fb`
- [x] Decode patient rows; filter to the 13 conditions → `data/interim/ddxplus_chestpain_validate.parquet` (33,963 rows; one file per split, and the builder refuses the test split). `src/ddxplus.py` + `scripts/build_ddxplus_chestpain.py` + 35 tests. Label audit **EXP-013** → R-13 extended, **D-8** opened — `4d640a8`
- [x] Build the train parquet when `train.csv` is downloaded: `scripts/build_ddxplus_chestpain.py --split train`. **Ask where first** (D-7). The download is 670.6 MB, and the parquet is needed wherever training runs. **Where: Colab** (the user, 2026-09-19), inside the B0/B1 training job. **Built on Colab 2026-09-19** by the owner's run: 255,900 of 1,025,602 rows (25.0%). The parquet stays on Colab; its summary came back in the bundle (`docs/03` §2.1) — `98c746b`
- [x] Re-confirm EXP-002 counts on `train.csv` once it is downloaded: **confirmed** (EXP-003). The rarest condition, spontaneous pneumothorax, has 10,162 training cases (projected ≈10,880), the imbalance is 2.70×, and every condition is within 8% of its projection; R-03 stays resolved — `98c746b`

**1b — Cardiac medical KG · ✅ 2026-09-19** · P1
- [x] KG loader from `data/interim/ddxplus_chestpain_conditions.json` → a backend-neutral `KnowledgeGraph` (`src/medical_kg/loader.py`): 14 conditions, 84 `DDX:E_nn` evidence nodes, 245 edges, **each with a `source`** (R-12). KG card in `docs/02` §5.1 — `6709697`
- [x] NetworkX `GraphStore` (`src/medical_kg/networkx_store.py`): a drop-in for `InMemoryGraphStore`, proven by running the pipeline on it in a test. 20 tests — `6709697`
- [x] Crosswalk, `SYM:*`/`RF:*` ↔ DDXPlus evidence (`src/medical_kg/crosswalk.py`): 33 concepts with SKOS match types (11 exact, 12 close, 3 broader, 3 narrower, 1 related, 3 with no DDXPlus equivalent), validated against the release. `expand_case` (concept → graph) and `concepts_from_evidences` / `case_from_ddxplus` (DDXPlus → concept). `scripts/check_crosswalk.py`, 52 tests, card in `docs/02` §5.2. **EXP-014**: the red flags over-fire (**R-15**); the golden cases pass on the real graph only through red flags — `249b968`
- [x] BODHI-S enrichment for MI, pericarditis, PE, GERD (likelihood edge weights, `source = bodhi_s`): 107 facts, 98 mapped to 56 edges, 7 unmappable with reasons, 2 of zero strength (`src/medical_kg/bodhi_s.py`, keyed by BODHI-S id, no BODHI-S text committed); 17 new crosswalk concepts (64 in all). **EXP-016**: GC-001's MI climbs from 5th to 3rd on graph score alone, but on DDXPlus patients the graph-only top-1 falls to 0.856 and MI's to 0.60, because the overlap score punishes enriched conditions (a 2a fix) — `cc2a9cd`
- [x] Hand-author aortic dissection into the KG (`source = hand_authored`): 20 edges from the ADD-RS markers, weighted by IRAD frequencies (`src/medical_kg/hand_authored.py`), with 14 new crosswalk concepts. `cardiac_kg.build_cardiac_kg` merges the sources, `canonical_concept_id` puts a yes/no-equivalent concept on its DDXPlus node, and `only_sources` filters for ablations. GC-003 now ranks dissection first on graph score alone. **EXP-015**: the graph alone scores 88% top-1 on DDXPlus patients, which is circularity (R-12) — `d181586`
- [x] Neo4j-backed `GraphStore` from the same `KnowledgeGraph`, with NetworkX as the fallback (**D-6**): `src/medical_kg/neo4j_store.py`. It writes the KG to AuraDB under the label `CardiacKG` whenever the local build's fingerprint differs, then reads it back at start-up and scores it in memory, identically to NetworkX (checked on the golden cases). `open_graph_store()` falls back to NetworkX, and the pipeline reports `graph_backend` (docs/02 §7). `scripts/load_neo4j.py` loaded the real graph (129 nodes, 321 edges, fingerprint `de349201157a`). 32 new tests: 23 offline and 9 live (all 9 passed against Aura; CI runs them against a throwaway Neo4j). `neo4j` + `python-dotenv` moved into `requirements-dev.txt` — `3e9577a`

**1c — Baseline ML ranker · 🔄** · P2
- [x] Feature encoding from decoded evidence **codes** (not English labels), handling the three token
  kinds EXP-002 found: categorical codes · **numeric ordinal scales** (12.6% of tokens, e.g. pain
  intensity `E_56_@_4`) as ordered features, not one-hot · the `V_11` "NA" sentinel, explicitly.
  `src/ml/features.py` (`EvidenceEncoder`): 607 columns fixed by the release files, not by patients
  (fingerprint `3a0d5a5e01d7f427`). A default answer gets no column, so the question columns equal
  `positive_codes` for all 33,963 validate patients; an ordinal is its value plus an "answered"
  flag. It reads only `age`, `sex` and `evidences`. Feature card in `docs/03` §2.2; 30 tests — `2e7f1bb`
- [x] EXP-003: B0 prevalence baseline, the floor: top-1 0.110, top-3 0.306 and must-not-miss recall@3 0.220 on validate (the owner's Colab run, 2026-09-19) — `98c746b`
- [x] EXP-004: B1 LogReg → XGBoost~~; `ConditionRanker` replacing `ConstantRanker`~~ *(now its own item, below)*. **Every training or
  tuning run on project data, even a small sample, waits for the user's OK and a choice of where**
  (D-7). Unit tests that fit a toy model on a few synthetic rows are tests, not training runs.
  **B0 and B1 run on Colab** (the user, 2026-09-19). Installing scikit-learn + XGBoost (~170 MB) into
  `.venv` and CI is approved for when the training job is written (D-4).
  **The job is ready** (not yet run): `scripts/train_baselines.py` and `notebooks/colab_b0_b1.ipynb`
  (`docs/11` §4.1). `src/ml/baselines.py` holds B0, logistic regression and XGBoost, saved as JSON
  with the feature fingerprint; XGBoost always gets sparse input. Validate is used only for the
  final scores; XGBoost stops early on 10% of train. 14 tests, one running the whole job on a
  synthetic mini-release. scikit-learn 1.9.1 + XGBoost 2.1.4 installed — `68c14bd`.
  **Run 2026-09-19** by the owner on a Colab T4 (commit `68c14bd`; the job took about a minute):
  XGBoost top-1 0.9985, top-3 1.000, must-not-miss recall@3 1.000, with logistic regression almost
  as good. That is how DDXPlus was generated, not skill (**R-16**, decision **D-10**). The laptop
  reloads the models and reproduces every metric — `98c746b`
- [x] Translate hand-authored cases into DDXPlus tokens, the seam the ranker needs
  (`src/ml/case_tokens.py`): `ConditionRanker.score` is given `SYM:*` concepts, `EvidenceEncoder`
  needs `E_55_@_V_101`, and nothing inverted that. Two hand-curated tables choose a
  representative answer per concept (17 entries, each with the French); `Match.NARROWER` is
  admitted by default (**A-8**, for the team); denials are recorded but not encoded (the R-18
  defect); every dropped finding keeps its reason. 79 tests, round-tripping every concept back
  through `concepts_from_evidences`. **R-17 opened**: the inversion has no ground truth.
  Smoke-testing it produced **EXP-017** and **R-18** — → pending
- [x] `ConditionRanker` replacing `ConstantRanker` (`src/ml/ranker.py`): `ModelRanker` wraps
  anything with `labels` and `predict_proba`, so the deep model later drops into the same
  place; `open_ranker()` degrades the way `open_graph_store()` does, returning flat scores
  that normalise to zeros — a **provably** graph-only ranking, asserted by equality — when the
  release files, the model or a matching fingerprint are missing, which is CI's ordinary
  state. The pipeline reports `ml`; `configs/config.yaml` grew an `ml:` block and the scripts
  a `--config` / `--ranker` flag. **The backend is logistic regression, not XGBoost**
  (EXP-017, R-18). `scripts/demo.py` and `scripts/check_crosswalk.py` both run it, and all
  four golden cases hold on the real graph. 19 tests — → pending
- [x] The golden cases re-run with **red flags off**, so the ranking is under test rather than
  the flag sort: `pipeline.run` sorts by `(red_flag, fused_score)`, and three of the four
  cases are flagged, so they could not detect a ranking regression at all. All four pass on
  the stubs. On the real graph and model, GC-002/003/004 pass and **GC-001's infarction ranks
  4th** — recorded as a *strict* xfail naming EXP-014 (the graph's overlap score divides by
  each condition's evidence-set size) so that it fails loudly the day 2a fixes it — → pending
- [ ] **GC-005, for the team:** a fifth golden case — a benign-looking ischaemic presentation
  that raises no red flag — because only GC-004 currently depends on the ranker at all

**1d — Patient KG + Synthea · ⬜** · P3
- [ ] Synthea cardiac-module generation (fixed seed)
- [ ] Patient KG builder
- [ ] Start the PubMed/PMC corpus fetch — slow, so begin now; used in Phase 3

**1e — Platform · ⬜** · P4
- [ ] FastAPI: `POST /diagnose`, `GET /conditions`, `GET /health`
- [x] Evaluation harness scaffold (`src/eval/metrics.py`): Precision@3 and Recall@5 on `D_in`, as amended by D-8, with Recall@5 on the full D beside them (`docs/05` amendment 1). Also top-1/3/5, MRR, per-condition F1, must-not-miss recall@3, the dangerous false-negative rate, red-flag sensitivity against the true condition (amendment 2) and precision, ECE, Brier and the reliability table, a 95% bootstrap interval for every ratio metric (1,000 resamples, seed 42; 3.6 s for 11 metrics on 34k cases), and McNemar's test. 25 tests with hand-computed answers. Open decision **A-7** added (what makes a red flag "appropriate") — `fafabb2`
- [ ] Re-verify golden cases against real components — **fix the component, not the expectation**

### Phase 2 — Reasoning, Fusion & Prototype · ⬜
**Exit criteria:** 🎯 **W5 milestone** — working prototype in the UI, tagged `v0.1-prototype`, demo video recorded.
- **2a** · P1 — KG scoring (overlap → Personalised PageRank) + reasoning paths · EXP-005.
  **Must fix:** the current overlap ranks stable angina above must-not-miss unstable angina even with
  rest pain present (`docs/02` §5.1, limitation 2). Add that case as a regression test once fixed.
  **Also (EXP-014):** by graph score alone, GC-001's MI ranks **fifth**, because overlap is divided by
  the size of each condition's evidence set. Use GC-001 as a second regression case.
  **EXP-015:** by graph score alone, unstable angina is first for only 21% of its validate patients.
  Re-measure with `scripts/build_cardiac_kg.py`, and report any graph-only number on DDXPlus with
  the circularity caveat (88% top-1 comes from R-12, not skill).
  **EXP-016:** the overlap score punishes a condition for having more evidence, so BODHI-S drops
  MI to 60% top-1 on DDXPlus patients. The new score must not do that (naive-Bayes or
  likelihood-ratio scoring over the likelihood bands, or PPR); until then `only_sources()` can
  keep BODHI-S out of scoring
- **2b** · P2 — Calibration (Platt vs isotonic) + SHAP · EXP-007. B1's raw scores already have ECE 0.0001 on validate (EXP-004), so calibration matters mainly for fused and reduced-evidence scores (D-10)
- **2c** · P4 — Fusion layer, ✓/?/✗ analysis, weight sweep on validation only · EXP-006. **Decide D-10 first** (R-16): on full-evidence DDXPlus, B1 leaves no room on top-3 or must-not-miss recall
- **2d** · P3 — Red-flag rules on real concepts + safety layer v1 · EXP-008. Three
  must-not-miss conditions have **no rule yet**: unstable angina, myocarditis, acute pulmonary edema.
  **R-15 (EXP-014):** the aortic-dissection rule flags 50% of validate patients, because back
  radiation alone fires it. The MI rule flags 24% of non-MI patients, mostly unstable angina, which
  is appropriate. The pneumothorax rule reaches 32%, limited by the sudden-onset cut-off (A-5).
  Re-measure with `scripts/check_crosswalk.py`. Red-flag sensitivity is now each condition's
  own-patient rate (`docs/05` amendment 2)
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
| 2026-09-19 | **D-8:** Precision@3 and Recall@5 use `D_in`, the part of the ground-truth differential inside the 13 in-scope conditions (Recall@5 can reach 0.753); Recall@5 with the full D (ceiling 0.434) is reported beside it. `docs/05` amendment 1 | the team, relayed by the owner · `docs/05` §9 |
| 2026-09-19 | **Red-flag sensitivity** is measured against the true condition: of must-not-miss patients, the share flagged for their own condition, overall and per condition. A condition without a rule counts as missed; the target is unchanged (≥ 0.95). `docs/05` amendment 2 | the team · `docs/05` §9 |
| 2026-09-19 | **`Finding.source`** also lists `ddxplus` and `crosswalk` (its description only; a contracts change, agreed by the team) | the team · `src/contracts.py`, `docs/02` §4.2 |
| 2026-09-19 | **A-5** (sudden onset is `E_59` ≥ 8) and **A-6** (the likelihood band weights) accepted as working values; 2d and 2a re-check them | the team · `docs/02` §9 |

---

## 7. 🧰 Environment state — verified 2026-09-18 (Neo4j and Local Python rows: 2026-09-19; CI, Heavy compute and Data rows: 2026-09-20)

| Item | State |
|---|---|
| Git | clean · the repository is **public** on GitHub, so cloud notebooks clone it without a token |
| CI (GitHub Actions) | ✅ green on every push so far (latest verified: `68c14bd`) · Python 3.11 · since 2026-09-19 the job also starts a throwaway `neo4j:5-community` container for the live Neo4j tests |
| Local Python | **3.11.9** in `.venv` (D-2) — light deps only: `requirements-dev.txt` (now incl. pandas, numpy, pyarrow, **networkx 3.6.1**) + huggingface_hub + **`neo4j` 5.28.6, `python-dotenv` 1.2.3** (2026-09-19, for the Neo4j store) + **scikit-learn 1.9.1, XGBoost 2.1.4, SciPy 1.17.1** (2026-09-19, for the ML baselines); all now in `requirements-dev.txt`. Installs are per task (D-4) |
| Machine Pythons | 3.14 (**still the default** — plain `python` bypasses the venv), 3.12, 3.11 · always use `./.venv/Scripts/python.exe` |
| Docker | 29.7.2 installed, **not needed**: Neo4j runs on AuraDB Free (D-6). `docker-compose.yml` stays as the optional local route |
| Neo4j | **AuraDB Free**: instance created by the user on 2026-09-19; `.env` filled. Login verified from the laptop without reading `.env`: Neo4j 5.27 (Aura). **It holds the KG since 2026-09-19** (`scripts/load_neo4j.py`): label `CardiacKG`, 129 nodes, 321 edges, fingerprint `de349201157a`, and nothing else. **Its home database is named after the instance id, not `neo4j`.** A Free instance pauses after 72 h unused (resume with **Play** in the console) and is deleted after 30 days paused |
| Ollama | **0.34.2** installed (it updates itself; it was 0.34.1) · only `qwen2.5-coder:7b` pulled — a coding model, not the configured `llama3.1:8b` · runs models **on the GPU by default**. Smoke test on 2026-09-18 ✅: 100% GPU, ~71 tokens/s, first prompt 13.6 s, then 0.3 s (`docs/11` §3). Because it uses the GPU, **each run needs the user's yes first** (D-9) |
| GPU | NVIDIA RTX 5060 Laptop · 8 GB VRAM · Blackwell (`sm_120`; torch ≥ 2.7 / CUDA 12.8) · driver 591.91 (CUDA 13.1) · **`.venv-gpu` set up 2026-09-18** (4.3 GB on D:): torch 2.11.0+cu128, `check_gpu.py --device cuda` ✅ 6.6 TFLOP/s float32 (`docs/11` §2) · **tests only**, each run by Claude only after the user's yes (D-9) · `compute.device: cpu` in `configs/` |
| Heavy compute | **Google Colab (free), Kaggle Notebooks, Lightning AI / Studio Lab**, chosen per job by the user (`docs/11` §4) · university GPU not yet granted (R-14) · **first cloud job, 2026-09-19:** B0/B1 on a Colab T4 (Python 3.13), about a minute for the job itself (`docs/11` §4.1) |
| Data on disk (gitignored) | `data/raw/ddxplus/release_*.json` + **`validate.csv`** (87 MB) · `data/raw/bodhi_s/*.jsonl` · `data/interim/ddxplus_*.json`, `exp002_class_balance.json`, **`ddxplus_chestpain_validate.parquet`** (4.6 MB) + `.summary.json` · `cardiac_kg_summary.json`, `crosswalk_check.json` · `train.csv` / `test.csv` **not on the laptop** (train was downloaded and decoded on Colab, inside the B0/B1 job) · **`models/nightingale_b0_b1/`**, plus the zip it came in: the owner's Colab bundle (3 MB: the three models, `metrics.json`, `run.json`, both splits' summaries). The docs and `train_baselines.py` expect `models/b0_b1/`; moving it waits for the user's OK |
| CI dependency set | `requirements-dev.txt`: the tools plus pydantic, pyyaml, pandas, numpy, pyarrow, networkx, neo4j, python-dotenv, scikit-learn and xgboost. These moved from `requirements.txt` so CI runs the parquet-builder and KG tests (CI went from ~15 s to ~35 s) |

Re-verify this table whenever the environment changes, and date it.

---

## 8. 📓 Session log — newest first, one line per session

| Date | Who | What happened | Commits |
|---|---|---|---|
| 2026-09-23 | Claude | **The real model ranks behind the pipeline (1c) — the W3 walking skeleton.** `src/ml/ranker.py`: `ModelRanker` over anything with `labels` and `predict_proba`, and an `open_ranker()` that degrades exactly as `open_graph_store()` does. A degraded ranker returns *flat* scores rather than none, so the fused ranking provably equals a graph-only one, which a test asserts by equality. Three lines in the pipeline report `ml`; `configs/config.yaml` grew an `ml:` block, and the two scripts a `--config` flag. Then the golden cases were re-run **with the red flags off**, which is the first time their *ranking* has ever been tested: three of the four are flagged, and the pipeline sorts flagged candidates first, so until now a ranker returning noise would have passed them. All four pass on the stubs; on the real graph and model GC-002/003/004 pass and **GC-001's infarction ranks 4th**, behind Boerhaave and pericarditis, because the graph's overlap score divides by each condition's evidence-set size (EXP-014, a 2a fix). That is recorded as a strict xfail so it fails the day 2a repairs it. **A-8** opened (docs/02 §9) and **D-11** raised for the team | → pending |
| 2026-09-23 | Claude | **The ranker seam, and what it uncovered (1c).** `src/ml/case_tokens.py` turns a hand-authored case into the DDXPlus tokens a patient would have given — the inversion the crosswalk never had, and the piece every ranker needs. Smoke-testing it against the Colab bundle found something bigger: **B1's two models are indistinguishable on full evidence and far apart without it** (**EXP-017**). Keeping each patient's initial evidence plus half the rest, XGBoost's top-1 falls 0.9985 → 0.597 while logistic regression holds 0.976; at a quarter it is 0.267 against 0.856, and XGBoost answers atrial fibrillation for 76% of patients — with probability 1.000 when given no evidence at all. The encoder gives a default "no" answer no column, so it cannot tell a denied question from an unasked one, and AF is the condition whose DDXPlus patients answer fewest questions. At 25% evidence XGBoost's must-not-miss recall@3 is 0.935, **below the 0.95 target**. **R-18 opened**; the real ranker will be logistic regression, overriding the plan's "wire XGBoost first". **R-17 opened** (the inversion has no ground truth). This is also the measured answer to the supervisor's robustness point, and the strongest argument yet for **D-10** option (b) | → pending |
| 2026-09-20 | the user + Claude | **B0 and B1 trained (EXP-003, EXP-004); 1a ✅.** The owner ran the Colab notebook at `68c14bd` on a T4 on 2026-09-19; the job itself took about a minute. Claude checked the bundle on the laptop: the models load (the feature fingerprint matches), and re-scoring validate on the CPU reproduces every metric to within 3 × 10⁻¹². **B1 is near-perfect** (XGBoost top-1 0.9985; top-3 and must-not-miss recall@3 1.000), because DDXPlus draws each patient's evidence only from their condition's list: for 91.7% of patients, no other condition's evidence set holds their positive answers. So on full-evidence DDXPlus the fusion cannot beat B1 on the headline metrics: **R-16 opened, decision D-10 raised for the team.** Every B1 error puts unstable angina (or MI) below stable angina. EXP-002 confirmed on the real train counts (rarest 10,162, 2.70×). The user asked for the next task (`ConditionRanker`) to wait until they say go | `98c746b` |
| 2026-09-19 | Claude | **1c: the B0/B1 training job is ready for the owner's Colab run.** `src/ml/baselines.py` (B0 prevalence prior; B1 logistic regression and XGBoost, saved as JSON with the feature fingerprint), `scripts/train_baselines.py` (sparse features, early stopping on 10% of train, scored on validate with every `docs/05` metric and its interval) and `notebooks/colab_b0_b1.ipynb` (`docs/11` §4.1). Found and fixed: XGBoost reads a sparse matrix's absent entries as missing but a dense array's zeros as values, so the wrapper always hands it CSR. The whole job runs end to end in CI on a synthetic mini-release; no training on project data happened here (D-7). scikit-learn 1.9.1 and XGBoost 2.1.4 installed (approved) | `68c14bd` |
| 2026-09-19 | Claude | **1e evaluation metrics**: `src/eval/metrics.py` implements `docs/05` §3.1–§3.3 and §6 as amended, with every ratio metric carrying a 95% bootstrap interval. The hand-computed tests caught a bug before commit: must-not-miss recall counted the top-3 hits of cases that were not must-not-miss. Open decision **A-7** (docs/02 §9): red-flag precision needs a definition of "clinically appropriate"; until then the metric is strict. Protocol slip: the write-ahead marker was written just after the first file, not before | `fafabb2` |
| 2026-09-19 | Claude | **1c feature encoding**: `EvidenceEncoder` (`src/ml/features.py`) turns age, sex and the evidence tokens into 607 columns, fixed by the release files: binary evidences 0/1, categorical and multi-choice answers one-hot under their question (the default answer, meaning "no", gets no column), ordinals as a value plus an "answered" flag, and NA explicitly. On validate it encodes 33,963 patients in 1 s, and its question columns equal `positive_codes` for every one. Only 179 columns are ever nonzero, so the training job will use a sparse matrix. Feature card in `docs/03` §2.2 | `2e7f1bb` |
| 2026-09-19 | Claude | **1b Neo4j store; 1b ✅.** `src/medical_kg/neo4j_store.py` writes the KG to AuraDB (label `CardiacKG`) when its fingerprint differs from the local build, then reads it back at start-up and scores it in memory, identically to NetworkX; an edit made in Neo4j by hand is caught. `open_graph_store()` falls back to NetworkX when Aura is paused, offline or refuses the login, and the pipeline reports `graph_backend` (docs/02 §7). The real graph is loaded (129 nodes, 321 edges). 32 tests, 9 of them live: those passed against Aura, and CI runs them against a throwaway Neo4j container. Found: Aura's home database is named after the instance id, not `neo4j`. **1.0 ✅** too, with the optional pre-commit hooks deferred | `3e9577a` |
| 2026-09-19 | the user + Claude | **The team decided** the five points the owner sent them ("go with Claude's suggestion", relayed by the owner): **D-8** restricts D to the in-scope conditions for the headline Precision@3 / Recall@5 (`docs/05` amendment 1); **red-flag sensitivity** is measured against the true condition (amendment 2); `Finding.source` also lists `ddxplus` and `crosswalk` (a contracts change); A-5 and A-6 accepted as working values; R-12, R-13 and R-15 recoloured 🔴 (the key stays). The owner's checklist is done except the Colab run | `de357d8` |
| 2026-09-19 | the user + Claude | The owner is working through a checklist of their own tasks. **Done:** created the AuraDB Free instance and filled `.env` (Claude verified the login without reading it: Neo4j 5.27, empty; the home database is named after the instance id, not `neo4j`); chose **Colab** for the B0/B1 training job (D-7); approved installing `neo4j` + `python-dotenv` (now installed) and scikit-learn + XGBoost (when the training job is written) (D-4), and loading the KG into Aura. **Next:** the owner sends the team the decision list (D-8 first); then Claude builds the Neo4j store | `e6f6c60` |
| 2026-09-19 | Claude | **1b BODHI-S enrichment**: 107 facts about MI, pericarditis, PE and GERD; 98 mapped to 56 edges weighted by P(finding \| condition), keyed by BODHI-S id so that no BODHI-S text is committed (CC-BY-NC); 17 crosswalk concepts added (64). 21 facts restate DDXPlus and count once. **EXP-016**: GC-001's MI climbs from 5th to 3rd on graph score alone, but on DDXPlus patients graph-only top-1 falls to 0.856 and MI's to 0.60, because the overlap score punishes enriched conditions. 2a must replace it. 1b is done except the Neo4j store, which waits for AuraDB | `cc2a9cd` |
| 2026-09-19 | Claude | **1b aortic dissection hand-authored**: 20 edges from the ADD-RS markers with IRAD frequencies (Hagan 2000; Evangelista 2016) as likelihood bands, each with its reference; 14 crosswalk concepts added (47 in all); the graph is now merged from sources, and a fact two sources state is counted once. GC-003 ranks dissection first on graph score alone. **EXP-015**: the graph alone ranks DDXPlus validate patients 88% top-1, 99.8% top-3, which is circularity (R-12, now measured); unstable angina is first for only 21%. Open decision A-6 (the band weights) added | `d181586` |
| 2026-09-19 | Claude | **1b crosswalk**: 33 hand-authored concepts mapped to DDXPlus answers, with SKOS match types that allow only sound inferences. Validated against the release. `expand_case` lets the real graph score hand-authored cases, and `concepts_from_evidences` lets the red-flag rules run on DDXPlus patients. **EXP-014** on validate: the mapping behaves sensibly; the **red flags over-fire** (59% of patients; the aortic-dissection rule flags 50%) → **R-15**; the golden cases pass on the real graph, but only because red flags rank first (by graph score alone, GC-001's MI is fifth). Open decision A-5 (the sudden-onset cut-off) added to `docs/02` §9 | `249b968` |
| 2026-09-18 | Claude | The user decided **D-9**: Claude runs each laptop-GPU test only after the user says yes in chat. Recorded in `CLAUDE.md`, `docs/11` (v1.1) and every doc that still said the user runs the tests by hand | `4007807` |
| 2026-09-18 | Claude | The user asked Claude to run the laptop-GPU setup and tests itself ("just run and complete the setup and test yourself"). Claude treated that as covering these tests only: `.venv-gpu` with torch 2.11.0+cu128 (2.75 GB download, no pip cache kept), `check_gpu.py --device cuda` ✅ (`sm_120`, 6.6 TFLOP/s), the test suite passes inside it, Ollama smoke test ✅ at 100% GPU, and the model was unloaded afterwards. `docs/11` now records the results. Whether Claude may run GPU tests from now on is **D-9** | `e956594` |
| 2026-09-18 | Claude | The user set compute placement (**D-7**): the laptop GPU for tests only, run by the user by hand; training runs, and any job over ~5 min (CPU included), go to the cloud (Colab free · Kaggle · Lightning AI / Studio Lab) or the university GPU, asked per job. Also **D-4** (per-task installs) and **D-6** (Neo4j on AuraDB Free). New runbook `docs/11` with the user's GPU setup and start/stop steps; `scripts/check_gpu.py`; R-14 re-scoped, R-06 mitigated | `d3683a7` |
| 2026-09-18 | Claude | Recorded the user's GPU policy as **D-7** (laptop GPU only with permission; university GPU sought) and **R-14**; D-4 re-framed to CPU torch, D-5 deferred to D-7; torch pin fixed. **1a:** validate parquet (33,963 rows) + label audit **EXP-013**. Found that a listed token can mean "no", and that the ground-truth differentials are open-world (Recall@5 ceiling 0.434) → **D-8**. **1b:** KG loader + NetworkX store, with edge provenance for R-12. KG card: questions not answers; stable ⊂ unstable angina, so overlap ranks the benign one first. Corrected the stale README/charter claims that the KG is built from BODHI-S. Protocol note: the session was interrupted once, mid-way through updating this file for task 2, and recovered from §2 | `8dced77` `4d640a8` `6709697` |
| 2026-09-18 | Claude | Applied user decisions D-1–D-3: archived the reference docs (D-3); Python 3.11 + `requirements-dev.txt`, fixing local/CI tool drift (D-2); `validate.csv` + EXP-002 — R-03 resolved, **R-13 opened** (D-1). Protocol lesson: §1 was not refreshed at the two intermediate commits — fixed, and §4.2 now requires it | `5ce3725` `1ccb06a` `843c5fb` |
| 2026-09-18 | Claude | Added `PROGRESS.md` + `CLAUDE.md` session protocol; verified the environment; corrected Phase 0 status — Neo4j/Ollama exit criteria were never met, now carried to 1.0 | `51b49ce` |
| 2026-09-17 | Claude | Phase 0c: R-01 spike (31% → FALLBACK), R-12 opened, eval protocol frozen | `8b682e9` |
| 2026-09-17 | Claude | Phase 0b: scaffold + walking skeleton, 33 tests, CI | `efc10e9` |
| 2026-09-17 | Claude | Phase 0a: 11 planning documents | `26240f5` `1a47a0e` `a842c0f` |
| 2026-09-15–17 | Claude | Scope locked to chest pain; renamed Nightingale; repo created | `f09be7f` `d612336` |
