# 05 — Evaluation Protocol (Pre-Registered)

**Version:** 1.1 · 2026-09-17 · **Owner:** P4
**Status:** 🔒 **FROZEN** as of 2026-09-17, end of Phase 0 — before any model was trained

> Frozen means: no metric, baseline, ablation, or success threshold in this document may change
> without a dated entry in the amendment log (§9) **and** disclosure in the final report. No model
> had been trained at the time of freezing, so nothing here was chosen with knowledge of results.

> **Why this document is written first.** Choosing metrics after seeing results is how a project
> talks itself into a conclusion. Everything measurable is fixed here, in advance. After the freeze
> date, changes are permitted only via the amendment log in §9, with a dated justification.
>
> **The test set is opened once, in Phase 4.** All development uses train/validation only.

---

## 1. Research questions and hypotheses

| # | Research question | Hypothesis |
|---|---|---|
| **RQ1** | Does fusing KG reasoning with an ML ranker improve differential accuracy over either alone? | **H1:** Fusion > ML-only and > KG-only on Top-3 accuracy and MRR |
| **RQ2** | Does the system improve **safety** over an ML-only baseline? | **H2:** Fusion + red flags > ML-only on must-not-miss recall@3 |
| **RQ3** | Does evidence grounding reduce unsupported claims versus an unconstrained LLM? | **H3:** RAG-grounded explanation has a lower unsupported-claim rate than LLM-only |
| **RQ4** | Which components actually contribute? | **H4:** Removing the medical KG degrades must-not-miss recall more than it degrades Top-1 |
| **RQ5** | Are the confidence scores calibrated? | **H5:** Post-calibration ECE < 0.10 |

**Negative results are reportable results.** If H1 fails, that is a finding about hybrid
architectures, not a project failure. It must be reported honestly.

---

## 2. Data and splits

| Aspect | Decision |
|---|---|
| Dataset | DDXPlus, filtered to the 13 in-scope chest-pain conditions |
| Splits | DDXPlus-provided train / validate / test |
| Split level | Case level (each synthetic patient appears in exactly one split) |
| Tuning | **Validation only** — including fusion weights and calibration |
| Test access | **Once**, Phase 4, after all development is frozen |
| Inputs | `AGE`, `SEX`, `EVIDENCES` (decoded) |
| **Forbidden input** | `DIFFERENTIAL_DIAGNOSIS` — this is the label. Using it is leakage. |
| Ground truth (ranking) | `DIFFERENTIAL_DIAGNOSIS` (ranked list) |
| Ground truth (top-1) | `PATHOLOGY` |

Report per-split case counts and per-condition class balance before any modelling.

---

## 3. Metrics

### 3.1 Ranking (primary)

Let the system output ranked candidates `ĉ₁…ĉₙ`; `y` = true pathology; `D` = ground-truth
differential set.

| Metric | Definition | Reported as |
|---|---|---|
| **Top-1 accuracy** | fraction where `ĉ₁ = y` | % |
| **Top-3 accuracy** | fraction where `y ∈ {ĉ₁,ĉ₂,ĉ₃}` | % ← *primary ranking metric* |
| **Top-5 accuracy** | fraction where `y ∈ top 5` | % |
| **MRR** | mean of `1 / rank(y)` | 0–1 |
| **Precision@3** | \|top-3 ∩ D\| / 3 | 0–1 |
| **Recall@5** | \|top-5 ∩ D\| / \|D\| | 0–1 |
| **Per-condition F1** | macro and micro across 13 conditions | 0–1 |

### 3.2 Safety (headline)

| Metric | Definition | Target |
|---|---|---|
| **Must-not-miss recall@3** | Of cases whose true condition is must-not-miss, fraction where it appears in the top 3 | **≥ 0.95** |
| **Dangerous false-negative rate** | Fraction of must-not-miss cases ranked **outside** the top 5 | ≤ 0.02 |
| **Red-flag sensitivity** | Of cases matching a red-flag pattern, fraction where the flag fired | ≥ 0.95 |
| **Red-flag precision** | Of fired flags, fraction clinically appropriate | reported, not targeted |

> Red-flag precision is deliberately **not** targeted. Over-flagging is an accepted cost
> ([04-safety-ethics.md](04-safety-ethics.md) §3).

### 3.3 Calibration

ECE (10 bins), Brier score, reliability diagrams — per condition and overall. **Target ECE < 0.10.**
No score is reported as a probability until it passes calibration.

### 3.4 Evidence retrieval

Recall@k, Precision@k, nDCG@10, plus **supported-claim rate** (fraction of generated claims
traceable to a retrieved passage), **unsupported-claim rate**, and **citation correctness** (does
the cited passage actually support the claim?).

Relevance judgements: 100 sampled candidate–evidence pairs, labelled independently by two team
members; report Cohen's κ. Disagreements resolved by discussion.

### 3.5 Explainability (qualitative)

50 sampled cases scored 1–5 on reasoning validity, evidence relevance, and clarity by two team
members. If a clinically-trained reviewer is available, their judgement supersedes; state clearly in
the report which applies. **Do not describe team members as clinical experts.**

### 3.6 System

Latency (p50/p95) with and without the LLM; graceful-degradation coverage (each failure in
[02-architecture.md](02-architecture.md) §7 exercised).

---

## 4. Baselines

| # | Baseline | Purpose |
|---|---|---|
| **B0** | Majority class / prevalence prior | Floor — anything below this is broken |
| **B1** | ML-only (XGBoost on decoded evidences) | The serious competitor |
| **B2** | KG-only (symptom-overlap / PageRank) | Is the graph useful alone? |
| **B3** | LLM-only (local model, case in prompt, no retrieval, no KG) | Tests the "just use an LLM" position |
| **B4** | Text-RAG + LLM (retrieval, no KG, no ML) | Tests whether retrieval alone suffices |

**B1 is the baseline that matters.** If the full system cannot beat a well-tuned XGBoost, the
architecture is not justified and the report must say so.

---

## 5. Ablation matrix

| Config | Patient KG | Medical KG | ML | Fusion | Red flags | RAG | LLM |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **A0 — Full Nightingale** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| A1 — no medical KG | ✅ | ❌ | ✅ | — | ❌ | ✅ | ✅ |
| A2 — no ML | ✅ | ✅ | ❌ | — | ✅ | ✅ | ✅ |
| A3 — no RAG | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| A4 — no red flags | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| A5 — no calibration | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| A6 — no patient KG (flat features) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Each configuration is evaluated on **all** §3 metrics. The expected headline finding is that
**A1 and A4 degrade safety metrics disproportionately** — i.e. the knowledge graph earns its place
on safety rather than on raw accuracy.

---

## 6. Statistical testing

- **Bootstrap** 1,000 resamples of the test set → 95% confidence intervals on every headline metric.
- **McNemar's test** for paired Top-3 correctness between A0 and B1.
- Significance level α = 0.05; **report effect sizes and CIs, not just p-values.**
- Fixed seed (42); report mean ± std over 5 seeds for any stochastic component.
- **No metric is reported without a confidence interval.**

---

## 7. Success criteria

| Level | Criteria |
|---|---|
| **Minimum** | Pipeline runs end-to-end; A0 beats B0; all metrics computed with CIs; ablations complete |
| **Target** | A0 > B1 on Top-3 **and** must-not-miss recall; ECE < 0.10; unsupported-claim rate < 20% |
| **Stretch** | A0 > B1 with statistical significance; must-not-miss recall ≥ 0.95; clinician-reviewed explainability |

Failing "Target" while meeting "Minimum" is an acceptable, publishable outcome provided the analysis
explains *why*.

---

## 8. Reporting rules

1. Every metric reported with a 95% CI.
2. Every ablation row reported, including ones that contradict our hypotheses.
3. Per-condition results reported — aggregate numbers hide rare-condition failure.
4. The **synthetic-data limitation** stated alongside every accuracy claim.
5. Failed hypotheses stated plainly in the abstract/conclusion, not buried.
6. Fusion weights, hyperparameters, seeds, and model versions disclosed.
7. **The circularity caveat must accompany every RQ1/RQ4 result** (risk R-12). The cardiac KG is
   derived largely from DDXPlus `release_conditions.json`, the same source the ML ranker trains on,
   so the KG is not a fully independent knowledge source. State this wherever the value of the KG
   is claimed. The independent contributions that remain — hand-authored red-flag rules, aortic
   dissection (absent from the training data entirely), and BODHI-S enrichment for four conditions
   — should be reported separately from the shared-source component. See
   [10-spike-r01-crosswalk.md](10-spike-r01-crosswalk.md).

---

## 9. Amendment log

| Date | Change | Justification | Approved by |
|---|---|---|---|
| 2026-09-17 | Initial version | — | — |
| 2026-09-17 | Added reporting rule §8.7 (circularity caveat) | R-01 spike resolved to a fallback that derives the KG from DDXPlus, the ranker's own training source. Added **before** freezing and before any model was trained. | — |
| 2026-09-17 | **Document FROZEN** | End of Phase 0. No model trained to date. | — |

*Amendments after the freeze date require an entry here and must be disclosed in the final report.*
