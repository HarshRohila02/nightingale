# 07 — Risk Register

**Version:** 1.0 · 2026-09-17 · **Owner:** P1
**Review cadence:** weekly, at the Monday standup

Scoring: Likelihood (L) and Impact (I) 1–5. **Score = L × I.**
🔴 ≥15 critical · 🟠 9–14 high · 🟡 4–8 medium · 🟢 ≤3 low · closed, resolved and mitigated
risks show 🟢 whatever their score

---

## Active risks

### 🟢 R-01 — DDXPlus and BODHI-S vocabularies do not align
**L 4 · I 5 · Score 20 · Owner P1 · Status: ✅ CLOSED 2026-09-17 — fallback adopted**

*Measured outcome* (full report: [10-spike-r01-crosswalk.md](10-spike-r01-crosswalk.md)):

| Measure | Result | Gate |
|---|---|---|
| Condition coverage, strict | **31%** (4/13 exact) | 60% |
| Condition coverage, generous | 77% (incl. approximate) | 60% |
| Symptom alignment, content-word | **28%** | — |
| Symptom alignment, naive string | 0% *(artifact — see report)* | — |

**Decision: FALLBACK**, and the decision is robust to the measurement method — every figure that
matters is below the gate. BODHI-S spans 555 conditions across all of medicine and is simply not
cardiac-deep: myocarditis, spontaneous pneumothorax and Boerhaave are absent entirely, and it
collapses stable/unstable angina into one `Angina` node and AF/PSVT into one `Arrhythmia` node.

*Resolution — better than the anticipated fallback.* The plan assumed we would fall back to
co-occurrence statistics mined from patient rows. Instead, DDXPlus ships
**`release_conditions.json`**: a *curated* condition↔symptom knowledge base with **13/13 coverage**,
explicit symptoms and antecedents, **ICD-10 codes**, and a **severity ranking** that independently
corroborates our must-not-miss set. BODHI-S is retained to enrich the 4 exactly-matched conditions
(MI, pericarditis, PE, GERD) with its `likelihood_*` edge qualifiers.

*Spawned risk:* **R-12 (circularity)** — see below.

---

### 🔴 R-12 — KG and ML ranker share a data source (circularity)
**L 5 · I 3 · Score 15 · Owner P4 · Status: OPEN — accepted, must be disclosed**

Consequence of the R-01 fallback: if the cardiac KG is derived from DDXPlus and the ML ranker is
trained on DDXPlus, the KG is no longer an *independent* knowledge source. The ablation question
"does the KG add value over ML alone?" is weakened, because both encode the same beliefs.

*Mitigation — keep the comparison meaningful:*
1. The KG contributes what the ranker cannot express: reasoning paths, supporting/missing/
   contradicting analysis, and rule-based red flags.
2. **Aortic dissection is in neither source** and is hand-authored — a contribution the ranker can
   never make.
3. Red-flag rules are hand-authored from clinical literature, not derived from DDXPlus.
4. BODHI-S enrichment supplies independent evidence for 4 conditions.
5. *(Added 2026-09-18.)* **Every KG edge records its `source`** (`ddxplus`, `bodhi_s` or
   `hand_authored`), so the KG's shared-source contribution can be separated from its independent
   ones mechanically. ~~Today all 245 edges are `ddxplus`: the KG is still entirely shared-source.~~
   *Updated 2026-09-19:* 76 of the 321 edges do not come from DDXPlus: 20 `hand_authored`
   (aortic dissection, from the ADD-RS and IRAD) and 56 `bodhi_s` (MI, pericarditis, PE, GERD).
   KG card: [02-architecture.md](02-architecture.md) §5.1.

*Measured 2026-09-19 (EXP-015).* On DDXPlus validate patients, the graph alone ranks the true
condition first for **88%** and in the top 3 for **99.8%**, because DDXPlus generated them from
the definitions the graph is built from. The hand-written golden cases show the other side: the
same graph puts GC-001's MI fifth. No KG number on DDXPlus data may be reported without this
caveat. The independent BODHI-S edges lower the graph-only score on DDXPlus patients (top-1
0.856; MI 0.60, EXP-016) while lifting GC-001's MI from fifth to third: circularity rewards
DDXPlus's own knowledge, so independent knowledge looks worse on DDXPlus data.

*Updated 2026-09-24 (EXP-005, 2a).* The overlap score has been replaced, and the circularity is
now starker: the graph alone reaches top-1 **0.923** (protocol tie rule), must-not-miss recall@3
**1.000** and Precision@3 **0.756** on validate — close to B1 XGBoost's 0.764, with no training at
all. The independent BODHI-S edges no longer lower MI (0.992 → 1.000). None of this is skill: the
golden cases, on which every expectation now holds by graph score alone, remain the evidence.

*Expected honest finding:* the KG earns its place on **safety and explainability**, not raw
accuracy. Predicting this in advance is better science than discovering it at the end. Recorded as
a limitation in [05-evaluation-protocol.md](05-evaluation-protocol.md) §8.

---

### 🔴 R-18 — B1's answer depends on how much was asked, not only on what was answered
**L 5 · I 4 · Score 20 · Owner P2 · Status: OPEN — opened 2026-09-23 from EXP-017**

Measured on validate, with the models unchanged from EXP-004. Keeping each patient's initial
evidence plus a share of the rest:

| Evidence kept | XGBoost top-1 | LogReg top-1 | XGBoost must-not-miss recall@3 |
|---|---|---|---|
| 100% | 0.9985 | 0.9983 | 1.0000 |
| 50% | **0.5972** | 0.9755 | 0.9969 |
| 25% | **0.2671** | 0.8558 | **0.9346** — below the 0.95 target |

At 25% evidence XGBoost answers **atrial fibrillation for 76%** of patients; given no evidence at
all it answers atrial fibrillation with probability **1.000**.

*Cause.* `EvidenceEncoder` gives a default ("no") answer no column (`docs/03` §2.2), so a question
answered "no" and a question never asked are the same row. Atrial fibrillation is the condition
whose DDXPlus patients answer fewest questions (median 9 positive codes, against 17–19 for
infarction and unstable angina), so *within DDXPlus* "few answers" predicts it almost perfectly.
XGBoost compounds this by reading a sparse matrix's absent entries as **missing** and taking each
split's default branch. It is the defect `docs/04` §3 item 4 names explicitly: absent must not be
read as denied.

*Why it matters beyond the dataset.* Every hand-authored case is short — the golden cases carry 5 to
13 tokens after the concept → token inversion — so this is not a hypothetical regime, it is the
deployed one. A clinician using the prototype has a partial history by definition.

*Mitigations:*
1. **Decided 2026-09-23:** the pipeline's real ranker will be logistic regression, not
   XGBoost (`src/ml/ranker.py`, landing with the ranker wrapper), and the choice is recorded here
   rather than left as a preference.
2. **Next:** give the encoder an explicit "asked" channel (one column per question), which changes
   the feature fingerprint and needs both models retrained on identical rows — a training run, so
   the owner chooses where (D-7). Until then no XGBoost figure on short input means anything.
3. Score every future model — the deep ranker included — at all three evidence levels from the
   start, never at 100% alone.
4. It is the strongest argument for **D-10** option (b): the headline protocol cannot see this.

*Trigger for review:* the "asked" channel lands, or D-10 is decided.

---

### 🟠 R-17 — The concept → DDXPlus-token inversion has no ground truth
**L 4 · I 3 · Score 12 · Owner P2 · Status: OPEN — opened 2026-09-23 with `src/ml/case_tokens.py`**

The ranker seam has to choose *which answer* a hand-authored concept stands for: a patient whose
pain "radiates to the jaw or arm" answered `E_57` with one specific location, and the model has a
column per location. No dataset of (hand-authored case → tokens) pairs exists, so these choices
cannot be validated against anything. A wrong one — *forearm* where the discriminating answer is
*jaw* — is silent, and downstream it looks exactly like a model error.

*Mitigations in place:*
- Every token records the concept that produced it (`CaseTokens.by_concept`), and every finding that
  produces nothing is kept with a reason (`CaseTokens.dropped`), so a ranking can always be traced
  back to what the model was actually shown.
- The tests round-trip every concept through `concepts_from_evidences`, the crosswalk's
  independently written forward direction, and fail if a chosen answer does not imply its own
  concept back. They also fail if a new crosswalk entry has no chosen answer, or if a chosen answer
  is one the release file does not allow.
- The two tables (`REPRESENTATIVE`, `ORDINAL_REPRESENTATIVE`) are 17 entries, each with the French
  meaning in a comment, and are small enough for a clinical review — which is open, for the team.
- **Open decision A-8:** the inversion admits `Match.NARROWER` entries, which `expand_case` excludes.
  Without them `SYM:sudden_onset`, `SYM:exertional` and `SYM:relieved_by_rest` produce nothing, and
  those are the discriminators for embolism, pneumothorax, dissection and the anginas. Every token
  derived that way is listed in `CaseTokens.narrower`.

*Trigger for review:* the team's clinical review of the two tables, or the first time a golden case
ranks wrongly for a reason traced to a chosen answer.

---

### 🔴 R-16 — B1 reaches the ceiling on DDXPlus, so the headline comparisons cannot separate the systems
**L 5 · I 4 · Score 20 · Owner P4 · Status: OPEN — opened 2026-09-20 from EXP-004; decision D-10 pending**

On validate, ML-only B1 (XGBoost) ranks the true condition first for **99.85%** of patients and in
the top 3 for **all** of them. Its must-not-miss recall@3 is **1.000** and its MRR 0.9992. The cause
is DDXPlus's generator: each patient's evidence is drawn only from their own condition's list, so
every validate patient's positive answers fit inside their condition's evidence set, and for 91.7%
inside no other condition's (EXP-004). On full-evidence DDXPlus, therefore:

- **H1** (fusion > ML-only on top-3 and MRR) and **H2** (fusion + red flags > ML-only on
  must-not-miss recall@3) cannot be supported: top-3 and must-not-miss recall have no room left,
  and MRR has 0.0008.
- `docs/05` §7's *Target* (A0 > B1 on top-3 and must-not-miss recall) is out of reach, and the §6
  McNemar test on top-3 between A0 and B1 can show only a tie or a loss.
- **H4** (removing the KG hurts must-not-miss recall more than top-1) will likely fail for the same
  reason: B1 alone already has the recall.

This differs from R-09, where the hypothesis may be false: here the test cannot tell whether it is.

*What still separates systems on DDXPlus:* agreement with its differential (B1: Precision@3 0.764
of a possible 0.929, Recall@5 0.569 of 0.753), though a KG gain there needs the R-12 caveat, since
D and the graph both come from DDXPlus. Beyond DDXPlus: aortic dissection, which no DDXPlus patient
has; the golden cases; and the explanations.

*Mitigation: decision D-10 for the team (`PROGRESS.md` §3), to settle before the fusion work (2c),
so that any protocol amendment is made before a fusion result exists:*
1. Keep the protocol and explain the ceiling in the report.
2. **Add a reduced-evidence condition** (a `docs/05` amendment): every system is also scored on
   patients who keep their initial evidence plus a fixed random share of the rest, with the masks
   drawn once (seed 42). It mirrors a consultation in progress and gives the metrics room again.
   *Recommended.*
3. Add an independent test set: clinical vignettes the team writes from textbooks and case reports,
   like the golden cases. It escapes R-12 too, but it is small, so its intervals are wide.

*Trigger for review:* D-10 decided.

---

### 🟠 R-02 — Ten-week timeline is too short for the full pipeline
**L 4 · I 3 · Score 12 · Owner P4 · Status: OPEN — mitigated by design**

*Mitigation:* the plan is built around a **Week-5 working prototype** that is demo-able without RAG
or LLM, and a documented cut line (PTB-XL → KG embeddings → "what to ask next" → hybrid retrieval →
LLM explainer). The prototype, red-flag layer, and ablation study are never cut.
*Trigger:* the Week-3 walking skeleton slips by more than 3 days.

---

### 🟢 R-03 — Severe class imbalance across the 13 conditions
**L 1 · I 3 · Score 3 · Owner P2 · Status: ✅ RESOLVED 2026-09-18 on projection · confirmed on the train counts 2026-09-20**

*Measured* (EXP-002, validate split): the rarest condition (spontaneous pneumothorax) projects to
**≈10,880 training cases** — about 22× the trigger — and the imbalance across all 13 is only
**2.7×**. Projection uses the exact train/validate ratio 7.743; re-confirm with real counts when
`train.csv` is downloaded. Original assessment kept below for the record.

*Confirmed 2026-09-20* (EXP-003, the real train split): the rarest condition has **10,162**
training cases, 20× the trigger; the imbalance is **2.70×**; every condition is within 8% of its
projection. B1 learns even the two rarest perfectly (F1 1.000 for myocarditis and pneumothorax,
EXP-004).

~~**L 4 · I 3 · Score 12 · Owner P2 · Status: OPEN**~~

Rare conditions (Boerhaave, myocarditis) may have too few DDXPlus cases to learn, producing a model
that never ranks them — directly harming the must-not-miss metric.

*Trigger:* any in-scope condition has < 500 training cases.
*Mitigation:* report per-condition counts before modelling; `class_weight="balanced"`; stratified
splits; per-condition metrics always reported. If a condition is unlearnable, cover it with a
**red-flag rule** (as with aortic dissection) and disclose it.

---

### 🟠 R-04 — LLM hallucinates clinical facts
**L 3 · I 4 · Score 12 · Owner P3 · Status: OPEN**

*Mitigation:* generation constrained to retrieved evidence; claim-support checking; output validated
against the 13-condition set; unsupported claims flagged. Fallback to **templated explanations from
KG paths**, which cannot hallucinate. Measured as unsupported-claim rate in the evaluation.

---

### 🔴 R-13 — Closed-world assumption: out-of-scope causes are forced into our 13
**L 5 · I 3 · Score 15 · Owner P3 · Status: OPEN — accepted, must be disclosed** · *found by EXP-002*

Only 25.6% of DDXPlus cases fall inside the 13 in-scope conditions, and the model is trained only on
those. It therefore has no notion of "none of the above": a chest-pain presentation caused by
something outside the set — pneumonia is the obvious example — will still be ranked as one of our 13,
possibly with confidence.

*Mitigation:*
1. **Disclose** it wherever results are reported, and in the UI ("ranks 13 conditions only — other
   causes are not considered").
2. The **must-not-miss red flags still fire independently** of the ranking, so the safety layer is
   not closed-world in the same way.
3. *Stretch:* DDXPlus's other 36 pathologies are a ready-made source of out-of-scope examples for an
   abstention / out-of-distribution signal.

*The labels are open-world too* (EXP-013, 2026-09-18). 91.8% of in-scope patients' ground-truth
differentials include out-of-scope conditions, which hold 33% of the probability mass on average.
As docs/05 §3.1 defines Recall@5, a perfect 13-condition system could not score above **0.434**.
The definition of D is therefore open as decision **D-8** in `PROGRESS.md`. docs/05 is frozen, so
any change goes through its amendment log.

*Decided 2026-09-19 (D-8, docs/05 §9 amendment 1):* the headline Precision@3 and Recall@5 count
only the in-scope part of D, so a perfect system can reach 0.753, and Recall@5 with the full D is
reported beside it.

---

### 🔴 R-15 — Red-flag rules fire so often that a flag stops meaning anything
**L 5 · I 3 · Score 15 · Owner P3 · Status: OPEN** · *found by EXP-014, 2026-09-18*

The crosswalk let the red-flag rules run on DDXPlus patients for the first time. **59% of validate
patients get at least one flag.** The aortic-dissection rule fires on **50% of all patients**,
although DDXPlus contains no aortic dissection: back radiation alone is enough to fire it, and
DDXPlus lists back radiation for 99% of PE, pericarditis and Boerhaave patients and 74% of MI
patients. docs/04 §3 accepts low red-flag precision, but not at this level. A flag on half of all
patients teaches users to ignore flags (alarm fatigue). The pipeline also ranks flagged candidates
first, so over-firing reorders the differential, and must-not-miss recall@3 could look better than
the ranking deserves.

*Mitigation:*
1. **2d (EXP-008)** tightens the rules and re-measures them on validate with
   `scripts/check_crosswalk.py`. The aortic-dissection rule should need more than back radiation,
   for example tearing pain with it; the ADD-RS score weighs pain features the same way.
2. Report red-flag rates per condition (own patients versus everyone else) next to must-not-miss
   recall, so a gain that comes from over-flagging is visible.
3. Treat the DDXPlus rates as an upper bound: DDXPlus records several radiation sites per patient,
   likely more than real patients report.
4. *(2026-09-19)* docs/05 now defines red-flag sensitivity as the share of each must-not-miss
   condition's own patients that its flag reaches (amendment 2). A rule that fires on everyone
   still scores well on that, so the rate on other patients (item 2) is always reported beside it.

---

### 🟡 R-05 — Local LLM too slow or too large for the available GPU
**L 3 · I 2 · Score 6 · Owner P3 · Status: OPEN**

*Trigger:* generation latency > 30 s or the model does not fit in VRAM.
*Mitigation:* use a quantised 7–8B model; cache explanations per candidate; explanation is
asynchronous in the UI; templated fallback always available.
*Related:* **R-14**, which GPU is available at all.

---

### 🟡 R-14 — Heavy jobs depend on free cloud tiers and university access *(was: No GPU secured for Phase 3)*
**L 2 · I 3 · Score 6 · Owner P3 · Status: OPEN** · *opened 2026-09-18, re-scoped the same day* ·
~~L 3 · I 3 · Score 9~~

*Update 2026-09-18, from the owner's compute policy (D-7,
[11-compute-runbook.md](11-compute-runbook.md)).* Heavy jobs run on the cloud (Colab free, Kaggle,
Lightning AI / Studio Lab) or the university GPU, and the owner chooses per job. The laptop GPU is
available for short tests, each run only after the owner says yes (D-9). It is set up and
verified (docs/11 §2). Free GPU tiers exist, so the risk is no longer "no GPU
at all". It is now the free tiers' limits: sessions that end without warning, weekly quotas, and
GPUs that are not guaranteed.

*Mitigation, current:*
1. Job scripts save outputs as they go and can resume (docs/06 §1). Kaggle's background runs suit
   long jobs.
2. There are three platforms, so a quota running out on one does not stop the work.
3. Nothing before Phase 3 needs a GPU. Long CPU jobs go to the cloud too.
4. The CPU fallbacks still exist: the templated explainer, and the LLM explainer being last in the
   cut line.
5. Start the university setup as soon as access is granted (docs/09 §1.4, docs/11 §6).

*Original assessment, kept for the record:*

The LLM explainer (3b), its baselines B3/B4 (EXP-009/010) and embedding the evidence corpus (3a) need
a GPU to run at a usable speed. The team is seeking access to a **university GPU**. The project
owner's laptop GPU (RTX 5060, 8 GB) may be used **only with their explicit permission** (decision
D-7 in `PROGRESS.md`). If neither is available when Phase 3 starts on 2026-10-22, the LLM runs on
the CPU and R-05 fires.

*Trigger:* university GPU access not confirmed by **2026-10-14** (end of Week 4).
*Mitigation:*
1. **Nothing before Phase 3 needs a GPU.** XGBoost, SHAP, the knowledge graph and the red-flag layer
   all run on the CPU, and `compute.device` in `configs/config.yaml` defaults to `cpu`.
2. At the trigger date, ask the owner whether the laptop GPU may be used for Phase 3.
3. The CPU fallbacks already exist. The templated explainer needs no LLM (R-04, R-05), and the LLM
   explainer is last in the cut line (R-02).
4. Setting up university access takes time: accounts, SSH, the job scheduler, a CUDA build to match
   its driver. Start as soon as access is granted. Learning items are in `docs/09` §1.4.

---

### 🟢 R-06 — Neo4j setup consumes disproportionate time
**L 1 · I 2 · Score 2 · Owner P1 · Status: MITIGATED 2026-09-18** · ~~L 2 · I 3 · Score 6 · OPEN~~

*Update 2026-09-18:* Neo4j will run on **AuraDB Free** (decision D-6), so no Docker is needed on the
laptop. Two small dependencies replace it: an internet connection, and a free instance that pauses
when unused (resume it from the Aura console). The NetworkX backend covers both (docs/02 §7), and it
already works.

*Update 2026-09-19:* the Neo4j store is built and the graph is on AuraDB. The store reads it at
start-up and falls back to NetworkX by itself, reporting `graph_backend`, when Aura is paused,
offline or refuses the login.

*Mitigation:* the `GraphStore` Protocol means a **NetworkX in-memory backend** is a drop-in
substitute. If Docker is not working by end of Week 2, switch and move on. Do not lose days to
infrastructure.

---

### 🟡 R-07 — PubMed corpus build is slow / rate-limited
**L 3 · I 2 · Score 6 · Owner P3 · Status: OPEN**

*Mitigation:* **start fetching in Week 2**, four weeks before it is needed. Register an NCBI API key
(10 req/s). Cache aggressively; corpus scoped to the 13 conditions only.

---

### 🟡 R-08 — Test-set contamination through repeated evaluation
**L 2 · I 4 · Score 8 · Owner P4 · Status: OPEN**

*Mitigation:* test data is loaded only by `src/eval/run_ablations.py`, only in Phase 4. All tuning
on validation. Policy stated in [05-evaluation-protocol.md](05-evaluation-protocol.md) §2.

---

### 🟡 R-09 — Fusion does not beat the ML-only baseline
**L 3 · I 2 · Score 6 · Owner P4 · Status: OPEN**

The hypothesis may simply be false.

*Mitigation:* this is a **legitimate research outcome**, not a failure, provided the ablation is
rigorous and the analysis explains why. The safety metric may still favour the hybrid even if
accuracy does not — which is itself the interesting finding. Report honestly; do not tune until the
desired answer appears.

*Update 2026-09-20:* on full-evidence DDXPlus this cannot even be tested: B1 already scores 1.000
on top-3 and on must-not-miss recall@3 (EXP-004). See **R-16** and decision D-10.

---

### 🟢 R-10 — BODHI-S non-commercial licence misunderstood
**L 2 · I 2 · Score 4 · Owner P2 · Status: MITIGATED**

*Mitigation:* documented in [03-data-management.md](03-data-management.md) §1.2; attribution and
non-commercial statement in README, report, and UI credits. *(2026-09-19)* The enrichment code
(`src/medical_kg/bodhi_s.py`) keys facts by BODHI-S ids with paraphrased notes, so no BODHI-S
text is committed.

---

### 🟢 R-11 — Team member unavailable (illness, exams)
**L 3 · I 1 · Score 3 · Owner P4 · Status: OPEN**

*Mitigation:* contracts + stubs mean no workstream hard-blocks another; cross-review spreads
knowledge; daily standup surfaces absence early.

---

## Closed risks

| ID | Risk | Resolution |
|---|---|---|
| R-00 | Datasets may require credentialing (MIMIC/PhysioNet), delaying the start by weeks | **Closed 2026-09-17** — dataset strategy switched to fully open/synthetic sources (DDXPlus, BODHI-S, Synthea), verified available. No DUA, no IRB, no wait. |

---

## Weekly review log

| Week | Date | Changes | Reviewed by |
|---|---|---|---|
| 1 | 2026-09-17 | Register created; R-01 spike scheduled | — |
| 1 | 2026-09-17 | **R-01 closed** (31% < 60% gate → fallback adopted); **R-12 opened** (circularity, spawned by the R-01 resolution) | — |
| 1 | 2026-09-18 | **R-03 resolved** on projection (EXP-002: min ≈10,880 train cases, 2.7× imbalance); **R-13 opened** (closed-world assumption, found by EXP-002) | — |
| 1 | 2026-09-18 | **R-14 opened** (no GPU secured for Phase 3): the laptop GPU is used only with the owner's permission, and a university GPU is being sought | — |
| 1 | 2026-09-18 | **R-13 extended** by EXP-013: the ground-truth differentials are open-world too (Recall@5 ceiling 0.434), which raises decision D-8 | — |
| 1 | 2026-09-18 | **R-14 re-scoped** (9 → 6) by the owner's compute policy: heavy jobs on the cloud or the university GPU, the laptop GPU for the owner's own tests. **R-06 mitigated** (6 → 2): Neo4j on AuraDB Free, no Docker | — |
| 1 | 2026-09-18 | R-14: the laptop GPU is set up and verified (`.venv-gpu` with torch 2.11.0+cu128; Ollama runs at 100% GPU). Laptop tests now run only after the owner says yes (D-9). Score unchanged | — |
| 1 | 2026-09-18 | **R-15 opened** (red flags over-fire, found by EXP-014 when the crosswalk let the rules run on DDXPlus): 59% of validate patients flagged; the aortic-dissection rule flags 50% | — |
| 1 | 2026-09-19 | R-12 measured by EXP-015: the graph alone scores 88% top-1 on DDXPlus patients, which is circularity. The first independent edges exist: 20 hand-authored for aortic dissection. Score unchanged | — |
| 1 | 2026-09-19 | R-12: 56 BODHI-S edges added (76 of 321 now independent of DDXPlus). EXP-016: they lower the graph-only score on DDXPlus and raise MI on GC-001. R-10: no BODHI-S text committed | — |
| 1 | 2026-09-19 | **R-12, R-13 and R-15 recoloured 🔴.** Each scores 15, which the key calls critical, and the team kept the key. The key now also says that closed, resolved and mitigated risks show 🟢, as R-01 and R-10 already did. R-13: D-8 decided (docs/05 amendment 1). R-15: red-flag sensitivity is now measured against the true condition (docs/05 amendment 2) | the team, relayed by the owner |
| 1 | 2026-09-19 | R-06: the Neo4j store is built, with the automatic fallback to NetworkX; the graph is on AuraDB. Score unchanged | — |
| 1 | 2026-09-24 | **R-12 updated** (EXP-005, 2a): the naive-Bayes score lifts the graph's circular top-1 on validate to 0.923 and its Precision@3 to 0.756, near B1's 0.764, so any KG figure on DDXPlus is now even less informative about skill. The overlap score's three defects — the angina inversion, GC-001's infarction, BODHI-S punishing MI — are fixed. Other scores unchanged | — |
| 1 | 2026-09-23 | **R-18 opened** (found by EXP-017, while smoke-testing the new ranker seam): B1's two models are 0.001 apart on full evidence and 0.38-0.59 top-1 apart once half the history is missing, because the encoder cannot tell a denied question from an unasked one. XGBoost falls below the 0.95 must-not-miss target at 25% evidence. **R-17 opened** with `src/ml/case_tokens.py`: the concept -> token inversion has no ground truth. R-16 unchanged: at 100% evidence every EXP-004 figure reproduces exactly | — |
| 1 | 2026-09-20 | **R-16 opened** (found by EXP-004): B1 reaches the ceiling of DDXPlus's top-3 and must-not-miss recall@3 (both 1.000), which raises decision D-10. **R-03 confirmed** on the real train counts (rarest 10,162, 2.70×; EXP-003). R-09 cannot be tested on full-evidence DDXPlus (see R-16). R-14: the first cloud job, B0/B1, ran on a Colab T4 in about a minute. Other scores unchanged | — |
