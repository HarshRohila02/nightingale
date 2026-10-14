# 19 · The work in four segments: one per team member

**Version 1.0 · 2026-10-14.** This page splits Nightingale into four segments, one per member.
Each segment is a coherent part of the system that one person owns, explains and answers questions
on. It covers the work done, the work left, the experiments, the decisions, the report sections and
the presentation slides.

**What it builds on:**
- the charter's roles (`docs/00` §9);
- the owners already recorded per sub-phase in `PROGRESS.md` §5.

**Where it departs from those, to balance the load:**
- **Fusion (2c)** moves from P4 to P3. It sits beside the red flags and the safety layer.
- **Error analysis (3e)** moves from P2 to P4. It is evaluation work, run with P4's harness.
- **The patient knowledge graph (1d)** moves from P3 to P1. It is a knowledge-graph build.
- **"What to ask next" (3d)** is split:
  - the cited table of tests goes to P3 (clinical safety);
  - the graph-computed questions stay with P1.
- **The case-to-model-token mapping** (A-8, A-10) goes to P1, as clinical knowledge curation.

**Fill in the names:**

| Segment | Member | Name |
|---|---|---|
| 1 · Medical knowledge graph and graph reasoning | P1 | [name] |
| 2 · Data and machine-learning rankers | P2 | [name] |
| 3 · Clinical safety, fusion, evidence and explanation | P3 | [name] |
| 4 · Platform, evaluation and integration | P4 | [name] |

> **Owning a segment means being able to explain and defend it:**
> - what it does and why;
> - its numbers, with their intervals;
> - its limits.
>
> Each segment's "Be ready to explain" list is the minimum. The project was built with AI
> assistance (Claude Code), directed and reviewed by the team. **How that is acknowledged** in the
> report and presentation follows the university's policy, and is still an open team decision
> (`PROGRESS.md` §5, 4c).

---

## The four segments at a glance

| | **P1 · Knowledge graph** | **P2 · Data and ML** | **P3 · Safety, fusion, evidence** | **P4 · Platform and evaluation** |
|---|---|---|---|---|
| **Sub-phases** | 1b, 2a, 3d (questions), 1d | 1a, 1c, 2b, 2f | 2c, 2d, 3d (tests), 3a, 3b, 3c | 0b, 1e, 2e, 3e, Phase 4 (4a, 4b, 4e) |
| **Main components built** | The graph from three sources; the crosswalk; the Neo4j and NetworkX stores; naive-Bayes scoring; case→tokens; the question ranking | DDXPlus decoding; the encoder with the "asked" channel; masked training; B0/B1; the transformer; the safety floor; calibration; the model's reasons | Red-flag rules; the safety layer; the opinion-pool fusion; the cited tests; the golden cases; evidence retrieval; the template explainer | Contracts and pipeline; the metrics harness; the evaluation protocol and masks; the API; the dashboard; latency; error analysis; CI |
| **Experiments run** | EXP-001, 005, 015, 016 | EXP-002, 003, 004, 007, 017, 018, 020, 021 | EXP-008, 014 | EXP-013, 019, 022, 023 |
| **Experiments left** | — (H4's reading of A1/A1b in Phase 4) | EXP-007 stage 2 | EXP-006, 011, 009/010 | EXP-012 |
| **Biggest piece left** | The patient graph (1d) | Calibration stage 2; models for the test read | The LLM explainer, guardrails, retrieval quality, the fusion sweep | The test split once, every ablation, the final report |
| **Test files** (tests) | `test_medical_kg` (21), `test_kg_sources` (24), `test_crosswalk` (52), `test_kg_scoring` (27), `test_neo4j_store` (34), `test_case_tokens` (87) | `test_ddxplus` (37), `test_features` (40), `test_evidence_masks` (174), `test_baselines` (19), `test_ranker` (27), `test_deep` (22), `test_reasons` (19), `test_calibration` (50), `test_safety_floor` (7), `test_report_deep` (4) | `test_red_flags` (52), `test_safety` (25), `test_sweep_fusion` (13), `test_next_steps` (75, the questions half shared with P1), `test_retrieval` (96) | `test_pipeline` (47), `test_contracts` (18), `test_metrics` (31), `test_api` (22), `test_dashboard` (9), `test_error_analysis` (20), `test_svg` (20), `test_notebooks` (20), `test_scripts_config` (5), `test_demo` (3), `test_check_gpu` (2) |

**Why this is a fair split.** Each segment owns:
- about seven built components;
- a share of the experiments;
- a share of what is left.

Where one segment has less done, it has more to do:
- **P3** has the largest remaining load: the LLM explanation, its guardrails and retrieval quality.
- **P2** has the most experiments already run.
- **P4** carries the final evaluation and the report's assembly.

The raw test counts are uneven (`test_evidence_masks` is heavily parametrised), so they are a
guide, not a measure.

---

## Segment 1 · P1 · Medical knowledge graph and graph reasoning

**In one sentence:** P1 built the cardiac knowledge graph from three sources, the vocabulary
crosswalk that lets clinicians' concepts reach it, the naive-Bayes score that ranks conditions from
it, and the information-gain ranking of what to ask next.

**What P1 built:**
- **The R-01 feasibility spike** (`scripts/spike_crosswalk.py`, `scripts/decode_ddxplus.py`,
  `docs/10`):
  - BODHI-S matched only 31% of our conditions, against a 60% gate;
  - so the graph's backbone became DDXPlus's own knowledge base (EXP-001).
- **The knowledge graph** (`src/medical_kg/`: `loader.py`, `cardiac_kg.py`, `bodhi_s.py`,
  `hand_authored.py`): **129 nodes, 321 edges**:
  - **245** from DDXPlus;
  - **56** from BODHI-S, for MI, pericarditis, PE and GERD, weighted by likelihood band;
  - **20** hand-authored for aortic dissection, from the ADD-RS and IRAD;
  - every edge keeps its source (R-12).
- **The crosswalk** (`crosswalk.py`): **67 concepts** mapped to DDXPlus answers with SKOS match
  types, so only sound inferences are drawn; `expand_case`, `concepts_from_evidences`.
- **Two graph stores:**
  - Neo4j AuraDB (`neo4j_store.py`, `scripts/load_neo4j.py`; label `CardiacKG`, fingerprint
    `de349201157a`);
  - NetworkX (`networkx_store.py`), with automatic fallback.
- **The graph score** (`scoring.py`, 2a):
  - a naive-Bayes log-likelihood over the graph closed under the crosswalk;
  - reasoning paths, and each finding's exact contribution;
  - chosen over Personalised PageRank (EXP-005).
- **Case → model tokens** (`src/ml/case_tokens.py`): the representative DDXPlus answer for each
  concept:
  - A-8, approved by the team;
  - **A-10**, the jaw-radiation answer corrected after the model's reasons exposed it.
- **What to ask next, the questions half** (`src/reasoning/next_questions.py`, 3d): the unrecorded
  findings ranked by expected information gain over the graph, through the crosswalk.
- **Scripts:** `build_cardiac_kg.py`, `check_crosswalk.py`.

**Experiments:**
- **EXP-001:** the crosswalk spike.
- **EXP-005:** graph scoring; every golden case ranks first by the graph alone.
- **EXP-015:** graph circularity. It measured how far the graph alone succeeds on the DDXPlus patients
  generated from its own source (R-12).
- **EXP-016:** BODHI-S's effect, and why the old overlap score punished enriched conditions.

**Decisions and risks P1 owns:**
- R-01 (the fallback);
- **R-12** (circularity, disclosed with every graph figure);
- A-6 (the likelihood bands);
- A-8 and A-10;
- R-17 (the token mapping has no ground truth).

**Left for P1:**
1. **The patient knowledge graph (1d):**
   - the Synthea notebook for Kaggle;
   - the per-patient graph;
   - the SNOMED → `RF:*` table (the cloud session's task T7 drafts the code);
   - how it feeds a case, a team decision.
2. **Phase 4:** reading ablations A1 and A1b (no medical graph) for hypothesis H4.
3. **Report:**
   - §2.4 (knowledge graphs);
   - §3.3 (the graph's sources);
   - §4.3–§4.5 (graph, crosswalk, score);
   - §4.8 (case → model input);
   - §6.5 (the graph alone, circular);
   - §7.2 (circularity) and §7.4 (R-17, A-10).

**Be ready to explain:**
- why the graph is built from DDXPlus and not BODHI-S;
- "the graph knows questions, not answers";
- why **not mentioned ≠ denied**;
- why stable and unstable angina tie in the graph;
- why the graph-alone 0.9234 [0.9206, 0.9259] top-1 on DDXPlus is **circular**;
- how aortic dissection, absent from the data, is still ranked;
- what the A-10 fix was;
- how information gain picks a question (GC-001: rest pain, a crescendo pattern).

**Mid-term slides:**
- literature strands A1 (computer-aided diagnosis) and A5 (knowledge graphs): slides 4 and 7.
  P1 presents all of slide 7, A6 included;
- methods slide 14 (the knowledge graph);
- the questions part of slide 16.

---

## Segment 2 · P2 · Data and machine-learning rankers

**In one sentence:** P2 turned DDXPlus into model-ready data, found and repaired the "not asked =
no" defect with an explicit "asked" channel and masked training, trained the classical rankers and
the transformer, and adopted the transformer only through a safety floor fixed before training.

**What P2 built:**
- **Data (1a)** (`src/ddxplus.py`, `scripts/build_ddxplus_chestpain.py`,
  `scripts/class_balance.py`):
  - the chest-pain subset: **255,900 training** and **33,963 validation** patients in 13
    conditions;
  - `positive_codes()`, because a listed token can mean "no";
  - the test split locked.
- **The feature encoder** (`src/ml/features.py`):
  - **607 columns** fixed by the release files, with ordinals encoded as ordinals;
  - the **"asked" channel**, **691 columns**, so that "not asked" ≠ "denied".
- **Masked training copies** (`src/ml/evidence_masks.py`, `augment()`), the `+aug` models. The same
  module draws the protocol's evaluation masks, which P4 runs.
- **The baselines** (`src/ml/baselines.py`, `scripts/train_baselines.py`, the Colab notebooks): B0
  (prevalence), B1-LR and B1-XGB (XGBoost always given sparse input).
- **The ranker seam** (`src/ml/ranker.py`): `ModelRanker` and `open_ranker`, which degrade to a
  provably graph-only ranking.
- **The deep ranker** (`src/ml/deep.py`, `scripts/train_deep.py`, `notebooks/kaggle_b1_dl.ipynb`):
  - a small FT-Transformer: 84 question tokens plus age, sex and [CLS]; width 96, 2 blocks, 4
    heads; **225,645 parameters**;
  - trained on a Kaggle GPU over 5 seeds;
  - a NumPy forward pass on the laptop.
- **The safety floor D-13** (`src/eval/safety_floor.py`, `scripts/apply_safety_floor.py`) and the
  full seed report (`scripts/report_deep.py`, `notebooks/kaggle_exp021_reports.ipynb`).
- **Calibration** (`src/fusion/calibration.py`, `scripts/calibrate.py`): none, temperature, Platt
  and isotonic, fitted per system (EXP-007).
- **The model's reasons** (`src/ml/reasons.py`): leave-one-finding-out occlusion.

**Experiments:**
- **EXP-002:** class balance, 2.70× at most.
- **EXP-003:** B0 top-1 0.110 [0.106, 0.113].
- **EXP-004:** B1 top-1 0.9985 [0.9980, 0.9989] on full histories. This is **R-16**, saturation.
- **EXP-017:** B1 on partial histories, which found **R-18**.
- **EXP-018:** the "asked" channel and masked copies.
- **EXP-020:** seeds 43–46.
- **EXP-021:** the transformer and D-13.
  - At 25%: must-not-miss recall@3 0.998 [0.997, 0.999].
  - MRR +0.0018 [+0.0009, +0.0026] over logistic regression.
  - Level with XGBoost.
- **EXP-007:** calibration, stage 1.

**Decisions and risks P2 owns:**
- **D-12** (a small transformer);
- **D-13** (safety first);
- T-6 (no tuning beyond early stopping);
- R-16, **R-18**, R-19 (the deep model no better than XGBoost);
- R-03 (class imbalance, resolved).

**Left for P2:**
1. **Calibration stage 2:** the fused score, after EXP-006's α.
2. **Phase 4:** the models scored on the test split; ablation A7 (classical in place of deep).
3. **Report:**
   - §2.6–§2.7 (calibration, tabular deep learning);
   - §3.2 (DDXPlus);
   - §4.6, §4.7, §4.9, §4.12, §4.13;
   - §6.2–§6.4, §6.7–§6.9;
   - §7.3 (synthetic demographics) and §7.5 (saturation).

**Be ready to explain:**
- why every model scores ~0.998 on full histories, and why that is the data, not skill;
- what the "asked" channel is, and why XGBoost answered atrial fibrillation for an empty history;
- what masked training copies are;
- why a transformer, and its size and cost (1.38 ms p95 per patient);
- what D-13 required, and that the transformer **passed** at all 5 seeds but is **level with
  XGBoost**;
- why the models stay uncalibrated (ECE ≤ 0.011);
- how occlusion names a model's reasons.

**Mid-term slides:**
- literature strands A3 (automatic diagnosis) and A4 (tabular deep learning, missing data):
  slide 6;
- slide 13 (dataset);
- slide 15 (the ML rankers);
- slide 19 (early results).

---

## Segment 3 · P3 · Clinical safety, fusion, evidence and explanation

**In one sentence:** P3 built everything that keeps the system safe and grounded:
- the red-flag rules after published clinical patterns, and the safety layer that enforces "no
  treatment, always the disclaimer";
- the opinion-pool fusion that lets the graph keep a condition the model never saw;
- the cited tests to consider;
- the golden cases;
- evidence retrieval from the open-access literature, screened of treatment advice.

**What P3 built:**
- **Red-flag rules** (`src/reasoning/red_flags.py`, `scripts/check_red_flags.py`, 2d):
  - one rule per must-not-miss condition (8), after the **ADD-RS**, **Wells** and **Braunwald**;
  - A-5 (sudden onset `E_59` ≥ 8);
  - **A-7**, which flags are clinically appropriate, settled by a three-judge panel.
- **The safety layer** (`src/reasoning/safety.py`), which runs last on every result:
  - the disclaimer always present;
  - red flags first;
  - no treatment, drug or dose in any sentence, test or evidence passage;
  - every failed component named.
- **Fusion** (`src/fusion/pool.py`, `scripts/sweep_fusion.py`, 2c):
  - a logarithmic opinion pool, with the model floored at 0.01;
  - aortic dissection takes the graph's posterior, so GC-003's dissection is first even with the
    red flags off;
  - EXP-006's weight sweep is pre-registered.
- **What to ask next, the tests half** (`src/reasoning/investigations.py`, 3d):
  - a cited table of the diagnostic investigations to consider per condition, after the ESC,
    ACC/AHA, BTS and other guidelines;
  - investigations only, and every line says "consider".
- **The golden clinical cases** (`tests/fixtures/golden_cases.yaml`): ACS, PE, aortic dissection
  and GERD. They must always pass, red flags on and off.
- **Evidence retrieval** (`src/rag/`, 3a):
  - BM25 plus dense search fused by reciprocal rank (k = 60), in NumPy on the laptop;
  - the PMC Open Access corpus built on Kaggle (`notebooks/kaggle_retrieval_corpus.ipynb`,
    `scripts/build_retrieval_corpus.py`);
  - the treatment-language screen (`screening.py`);
  - Creative Commons licences only;
  - `scripts/measure_retrieval.py`.
- **The template explainer** (`src/stubs.py`, `TemplateExplainer`): explanations derived from the
  graph and the model's reasons, never invented.
- **Safety and ethics** (`docs/04`).

**Experiments:**
- **EXP-008:** red flags.
  - Sensitivity rose from 0.449 [0.441, 0.456] to 0.823 [0.818, 0.829].
  - Precision is 0.848 [0.845, 0.852] by A-7's mapping (strictly, 0.512 [0.508, 0.516]).
  - The dissection rule's false alarms fell from 50% to 8% of patients.
- **EXP-014:** the crosswalk check that found the red flags over-firing (R-15).

**Decisions and risks P3 owns:**
- A-2 (the fusion design);
- A-5, A-7;
- **R-15** (alarm fatigue: measured on patients *without* a must-not-miss condition, 0.246 [0.240,
  0.253]);
- R-04 (LLM hallucination);
- FR-6.5 (no treatment advice);
- GC-005 (a fifth golden case, open).

**Left for P3 (the largest remaining load):**
1. **The evidence corpus:** the owner's Kaggle run, then the **EXP-011** retrieval-quality
   judgements, with a second team member.
2. **EXP-006, the fusion weight sweep,** once the team answers E-2.
3. **3b:** the grounded local-LLM explainer (model choice D-5), and baselines B3/B4
   (EXP-009/010).
4. **3c:** guardrails, the claim-support check and citation rendering. The cloud session's T5 is in
   review.
5. **Report:**
   - §2.2 (clinical rules), §2.5 (opinion pools), §2.7's retrieval and LLM part;
   - §3.6 (the golden cases);
   - §4.10 (fusion), §4.11 (red flags and safety), §4.14 (explanations);
   - §6.6 (red flags), §6.11 (golden cases), §6.12–§6.13;
   - §7.8–§7.9 (safety limits and design).

**Be ready to explain:**
- why each rule follows a published pattern;
- why red-flag sensitivity is 0.823, and why we do not loosen rules to chase 0.95;
- why the rules **collapse at 25%** (0.135 [0.130, 0.140]), and why the ranker must cover for them;
- how a logarithmic opinion pool works, and why a 0.01 floor stops an overconfident model from
  vetoing the graph;
- how dissection gets ranked;
- how treatment language is kept out of every output;
- how retrieval works without running a model on the laptop.

**Mid-term slides:**
- literature strands A2 (chest pain and clinical rules), A7 (retrieval and LLMs) and A8 (safety):
  slides 5 and 8;
- **slide 10 (the research gap, 5 marks)**;
- slide 16 (safety, explanation, tests, evidence).

---

## Segment 4 · P4 · Platform, evaluation and integration

**In one sentence:** P4 built the system's skeleton and everything that measures it:
- the typed contracts and the pipeline every component plugs into;
- the API and the live dashboard;
- the frozen evaluation protocol with its three evidence levels, the metrics harness with intervals
  on every figure, the latency study and the error analysis;
- the CI that keeps it all honest.

**What P4 built:**
- **The scaffold (0b):**
  - the contracts (`src/contracts.py`): present / denied / not-asked findings, `DiagnosisResult`,
    the disclaimer;
  - the pipeline (`src/pipeline.py`);
  - stubs, so it ran end to end from week 1;
  - `src/conditions.py`.
- **The walking skeleton** (`scripts/demo.py`, `scripts/_config.py`, `configs/config.yaml`): the
  real graph and model, end to end, on 2026-09-24.
- **The evaluation harness** (`src/eval/metrics.py`, `reports.py`):
  - every metric of `docs/05`;
  - a 95% bootstrap interval on each;
  - McNemar's test and paired bootstraps.
- **The evaluation protocol** (`docs/05`):
  - frozen before any model was trained;
  - **amendments 1, 2, 3 and 3a** applied and disclosed;
  - the **three evidence levels** run (`scripts/evaluate_reduced_evidence.py`,
    `check_mask_rules.py`), with the masks' digests committed before scoring.
- **The API** (`src/api/app.py`, `scripts/serve_api.py`): `/diagnose`, `/next-questions`,
  `/conditions`, `/concepts`, `/health`.
- **The dashboard** (`app/streamlit_app.py`, `app/view.py`, 2e):
  - present / denied / not asked, and the golden cases as examples;
  - ranks rather than scores;
  - ✓/✗/?, the degraded components, the disclaimer;
  - **live re-ranking** (D-14).
- **Latency** (`scripts/measure_latency.py`, EXP-022).
- **Error analysis and figures** (`scripts/error_analysis.py`, `src/eval/errors.py`,
  `src/eval/svg.py`, 3e): 22 SVG figures.
- **Engineering:**
  - CI (`.github/workflows/ci.yml`, with a throwaway Neo4j);
  - the test infrastructure (`tests/test_notebooks.py`);
  - `PROGRESS.md` and the experiment log;
  - the compute runbook (`docs/11`);
  - the demo guide (`docs/15`);
  - the `v0.1-prototype` tag.

**Experiments:**
- **EXP-013:** the label audit, which led to **D-8**.
- **EXP-019:** every system at 100%, 50% and 25%.
  - Standard XGBoost falls to top-1 0.245 [0.240, 0.250] at 25%.
  - Every masked-trained model meets the safety targets.
- **EXP-022:**
  - the transformer: p95 1.38 ms per patient;
  - the pipeline: p95 10–13 ms;
  - so the system works in soft real time.
- **EXP-023:** where every component goes wrong.
  - The main error is unstable angina ranked below stable angina, 0.222 [0.206, 0.238] at 25%.

**Decisions and risks P4 owns:**
- D-8;
- **D-10 / D-11** (amendment 3);
- E-1 (amendment 3a);
- **E-2** (open: how the reduced-level claims are judged);
- D-14 (real time);
- C-1 (contract changes);
- R-02 (the timeline), R-08 (test contamination), R-13 (closed world).

**Left for P4:**
1. **`v0.2-prototype`:** the dashboard with evidence and questions, checked end to end once the
   corpus is in.
2. **Phase 4:**
   - the freeze (`v0.9-freeze`);
   - **EXP-012**, pre-registered;
   - **the test split opened once**;
   - every ablation (A0–A7, A1b) at three levels;
   - the hypothesis verdicts (4a);
   - statistics and breakdowns (4b);
   - compliance checklists and `v1.0` (4e).
3. **Report assembly (4c), and the presentation and video coordination (4d).**
4. **Report:**
   - ch. 1 (introduction);
   - ch. 5 (evaluation protocol);
   - §2.8 (references, collected from everyone);
   - §3.4–§3.5, §3.7;
   - §4.1–§4.2, §4.15–§4.18;
   - §6.1, §6.10, §6.14;
   - §7.6–§7.7, §7.10–§7.11;
   - ch. 8 (conclusion, with everyone).

**Be ready to explain:**
- why the protocol was frozen before any model existed;
- **why three evidence levels**, and how the masks work (fixed, nested, digests recorded first);
- why must-not-miss recall@3 is the headline;
- what a bootstrap interval is, and McNemar's test;
- why the test split stays closed until Phase 4;
- how the system degrades instead of crashing;
- what "soft real time" means here;
- what E-2 is, and why the fusion sweep waits for it.

**Mid-term slides:**
- slides 2–3 (problem, scope);
- slide 9 (the literature summary table);
- slide 11 (problem definition, objectives);
- slide 12 (architecture);
- slide 17 (evaluation and tools);
- slide 18 (plan and progress);
- the live demo;
- slide 20 (limitations and next steps).

---

## Shared work

| Work | Split |
|---|---|
| Phase 0 documents | P1: `docs/10` and the graph cards in `docs/02` §5 · P2: `docs/03` (data) · P3: `docs/04` (safety and ethics) · P4: `docs/00`, `01`, `02` (architecture), `05`, `06`, `07`, `11` |
| Golden cases | Written by P3; every member keeps them green for their own components |
| The literature review | **Report ch. 2:** P1 A1, A5 · P2 A3, A4, and A6's calibration · P3 A2, A7, A8, and A6's fusion · P4 the references (§2.8). **Mid-term slides:** P1 4 and 7 · P2 6 · P3 5, 8 and the gap (10) · P4 the summary (9) |
| Checking the [V] references (`docs/presentation/midterm-prep.md` §9) | Each member checks the references of their own strands |
| The mid-term evaluation's questions (`midterm-prep.md` §7) | Q1, Q5, Q6 P2 · Q3, Q4 P1 · Q7, Q8 P3 · Q2, Q9–Q12 P4 |
| EXP-011's relevance judgements | P3 and one other member, blind (its pre-registration) |
| The final presentation and demo video (4d) | Each presents their segment; P4 coordinates and runs the demo |

## How the segments connect

```
P2 data ──► P2 encoder + "asked" channel ──► P2 ML rankers (transformer) ──┐
P1 knowledge graph ──► P1 crosswalk ──► P1 graph score ────────────────────┼──► P3 fusion ──► candidates
P3 red-flag rules ─────────────────────────────────────────────────────────┘        │
        P1 questions · P3 tests · P3 evidence · P2 model's reasons · P3 explanation ◄┘
                                   ──► P3 safety layer ──► P4 pipeline, API, dashboard
P4 contracts, metrics, protocol, masks, latency, error analysis, CI: across everything
```
