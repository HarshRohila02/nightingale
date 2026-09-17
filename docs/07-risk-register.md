# 07 — Risk Register

**Version:** 1.0 · 2026-09-17 · **Owner:** P1
**Review cadence:** weekly, at the Monday standup

Scoring: Likelihood (L) and Impact (I) 1–5. **Score = L × I.**
🔴 ≥15 critical · 🟠 9–14 high · 🟡 4–8 medium · 🟢 ≤3 low

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

### 🟠 R-12 — KG and ML ranker share a data source (circularity)
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

*Expected honest finding:* the KG earns its place on **safety and explainability**, not raw
accuracy. Predicting this in advance is better science than discovering it at the end. Recorded as
a limitation in [05-evaluation-protocol.md](05-evaluation-protocol.md) §8.

---

### 🟠 R-02 — Ten-week timeline is too short for the full pipeline
**L 4 · I 3 · Score 12 · Owner P4 · Status: OPEN — mitigated by design**

*Mitigation:* the plan is built around a **Week-5 working prototype** that is demo-able without RAG
or LLM, and a documented cut line (PTB-XL → KG embeddings → "what to ask next" → hybrid retrieval →
LLM explainer). The prototype, red-flag layer, and ablation study are never cut.
*Trigger:* the Week-3 walking skeleton slips by more than 3 days.

---

### 🟠 R-03 — Severe class imbalance across the 13 conditions
**L 4 · I 3 · Score 12 · Owner P2 · Status: OPEN**

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

### 🟡 R-05 — Local LLM too slow or too large for the available GPU
**L 3 · I 2 · Score 6 · Owner P3 · Status: OPEN**

*Trigger:* generation latency > 30 s or the model does not fit in VRAM.
*Mitigation:* use a quantised 7–8B model; cache explanations per candidate; explanation is
asynchronous in the UI; templated fallback always available.

---

### 🟡 R-06 — Neo4j setup consumes disproportionate time
**L 2 · I 3 · Score 6 · Owner P1 · Status: OPEN*

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

---

### 🟢 R-10 — BODHI-S non-commercial licence misunderstood
**L 2 · I 2 · Score 4 · Owner P2 · Status: MITIGATED**

*Mitigation:* documented in [03-data-management.md](03-data-management.md) §1.2; attribution and
non-commercial statement in README, report, and UI credits.

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
