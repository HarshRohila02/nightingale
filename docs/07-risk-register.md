# 07 — Risk Register

**Version:** 1.0 · 2026-09-17 · **Owner:** P1
**Review cadence:** weekly, at the Monday standup

Scoring: Likelihood (L) and Impact (I) 1–5. **Score = L × I.**
🔴 ≥15 critical · 🟠 9–14 high · 🟡 4–8 medium · 🟢 ≤3 low

---

## Active risks

### 🔴 R-01 — DDXPlus and BODHI-S vocabularies do not align
**L 4 · I 5 · Score 20 · Owner P1 · Status: OPEN — resolving in Week-1 spike**

DDXPlus stores symptoms as codes (`E_54_@_V_161`); BODHI-S as English sentences
(`Chest pain <radiate> to jaw`). If they cannot be reconciled, KG scoring cannot be computed over
patient findings and the fusion design collapses.

*Trigger:* crosswalk coverage < 60% at the Week-1 gate.
*Mitigation:* spike in Days 3–5 measures coverage before any dependent work starts.
*Fallback (pre-approved):* derive the medical KG from **DDXPlus condition↔evidence co-occurrence
statistics**; use BODHI-S only for qualifiers and red-flag rules. The KG remains real and
explainable; it simply carries less external knowledge. The report must disclose this.

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
