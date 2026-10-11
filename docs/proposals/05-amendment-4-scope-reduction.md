# Proposal record — `docs/05` amendment 4, the scope reduction (drafted 2026-10-02)

**What this file is.** A verbatim record of an uncommitted draft. On 2026-10-02 Claude wrote it
straight into the working tree of the frozen `docs/05-evaluation-protocol.md`, during the
interrupted "simplify the project" task (`PROGRESS.md` §2). It was found uncommitted on 2026-10-11.
At the owner's choice ("keep, then revise"), it is recorded here exactly as drafted, before
`docs/05` is restored to its committed v1.4. Nothing below has been edited.

Two parts follow:
1. The drafted §0, copied from the working tree.
2. The complete uncommitted diff of `docs/05`, which also holds the header change, the seven
   "superseded" markers, the §8 rule 8 change and the §9 row.

---

## Part 1 — the drafted §0, verbatim

## 0. Amendment 4 (2026-10-02): the simplified evaluation, which is what is judged now

*Where §§1–8 differ from this section, this section governs; they are kept for the record. Approved
by the team on 2026-10-02 (relayed by the owner), with the results listed in its §9 entry known.*

**Why.** The owner and the team judged the project too large for a college major project and too
hard to explain to the mentor, so its scope was cut to what the work so far can answer, and the
evaluation with it. As amended through 3a, the evaluation could not have supported any claim
against B1 at any level (E-2: the comparators saturate at 100% and 50%), so its hypotheses had to
be restated in any case.

### 0.1 The systems

| Label | What it is | Role |
|---|---|---|
| B0 | Prevalence prior (EXP-003) | The floor |
| B1-LR | Logistic regression trained on full histories (EXP-004) | The ordinary model |
| **ML** = B1-LR′+aug | Logistic regression trained on full histories **and** masked copies, with the "asked" channel (EXP-018) | The robust model, and the system's ML component |
| **KG** = B2 | The knowledge graph's naive-Bayes score (EXP-005) | The graph alone |
| **Fusion** | KG and ML pooled by the logarithmic opinion pool (`src/fusion/pool.py`) at the weight EXP-006 chooses; red flags off | The fused ranking |
| **System** = A0 | Fusion, then the red flags and the safety check | The full system |
| B1-DL | A small multi-layer perceptron (MLP) trained like ML: masked copies, the "asked" channel | The deep-learning comparison, reported |
| B1-XGB | XGBoost (EXP-004) | Reported beside the others, never judged |

The four ablation rows are **ML, KG, Fusion and System**. B1-XGB+aug, B1-XGB′+aug, B1-LR+aug,
A1–A7, A1b, B3 and B4 are no longer part of the evaluation; figures already measured for them stay
in docs/08.

### 0.2 Evidence levels

Every system is judged at **100%** and **25%** of each patient's evidence, with §3.7's masks
unchanged (a run that does not reproduce both recorded digests is invalid). The 50% level is still
computed and reported in an appendix, and is not judged.

### 0.3 Research questions and hypotheses

Each is judged once, on the test split (Phase 4), for the seed-42 models.

| # | Question | Hypothesis | Tests (each must pass, in the stated direction) |
|---|---|---|---|
| **RQ1** | Does training on incomplete histories make the ranker safe when a history is incomplete? | **H1:** at 25%, ML > B1-LR on must-not-miss recall@3 and on top-3 accuracy | McNemar on top-3: over must-not-miss patients, and over all patients |
| **RQ2** | Does fusing the knowledge graph with the model improve the ranking? | **H2:** at 25%, Fusion > ML and Fusion > KG on MRR | The paired bootstrap of the per-case difference in reciprocal rank, against ML and against KG |
| **RQ3** | Is the full system safe? | **H3:** at 100% and at 25%, System meets the §3.2 targets: must-not-miss recall@3 ≥ 0.95 and dangerous false-negative rate ≤ 0.02 | The point estimates against the targets; their 95% intervals are reported beside them |

**Reported, not judged:** every system at both levels on every §3.1–§3.3 metric; Fusion against ML
and KG at 100% (R-16: ML saturates there, so a tie is expected); System against ML on must-not-miss
recall@3 (McNemar on must-not-miss patients); B1-DL against ML at 25% with H1's and H2's tests;
red-flag sensitivity and precision (§3.2's 0.95 sensitivity target is reported as missed wherever it
is); the expected calibration error of the calibrated fused scores against §3.3's 0.10 target at
100%, with the 25% figure beside it (one calibrator, fitted on validation at 100%); the four golden
cases, including aortic dissection with red flags off.

### 0.4 Statistical tests

- A 95% bootstrap interval for every ratio metric (1,000 resamples of the cases, seed 42).
- **McNemar's test** (exact binomial below 25 discordant pairs) **passes** when p < 0.05 **and** the
  claimed system has more of the discordant wins.
- **The MRR test passes** when the 95% interval of the paired difference (the claimed system minus
  its comparator) lies **entirely above 0** (`paired_bootstrap_difference`: `better == "a"`).
- A hypothesis is supported only when every one of its tests passes. A single failure defeats it,
  so no multiplicity correction is needed. Every test is reported, passing or not.

### 0.5 Seeds and tuning

Every model trained on masked copies (ML and B1-DL) is trained with seeds **42–46**: the seed-42
model is the one used and judged, and the mean ± standard deviation over the five is reported
beside it. B1-LR is deterministic and fitted once. XGBoost is not retrained. No ML model is tuned
beyond early stopping on a 10% training holdout (T-6), and the MLP's size is fixed before it is
trained. The only quantities tuned on validation are the fusion's weight (EXP-006) and the
calibrator.

### 0.6 Success criteria

| Level | Criteria |
|---|---|
| **Minimum** | The pipeline runs end to end; System beats B0; every metric with its interval; the four ablation rows at both judged levels |
| **Target** | H1 and H3 supported |
| **Stretch** | H2 supported, and the calibrated fused scores' ECE < 0.10 at 100% |

Failing the Target while meeting the Minimum is still a publishable outcome if the report explains
why (§7).

### 0.7 Explanations

An explanation is built from the graph's matched findings and the logistic regression's largest
contributions to the score, by a template. A local language model may paraphrase it, and may add
nothing that the template did not contain. §3.5's qualitative review applies (50 cases, two team
members). Evidence retrieval is not built, so §3.4 is not measured, and the first-written H3 (a
retrieval-grounded explanation against an LLM alone) is withdrawn.

### 0.8 Precision fixes (from E-2)

- A mask digest is the first 16 hexadecimal digits of the SHA-256 of §3.7's lines. In the second
  digest, the questions asked are sorted by the number after `E_` (E_2 before E_10).
  `scripts/check_mask_rules.py` computes the first, `scripts/evaluate_reduced_evidence.py` both, and
  both are recorded in docs/08 before anything is scored on the masks.
- EXP-006's rule binds as committed with this amendment; a later change needs an entry in §9.
- §8 rule 8's disclosure includes amendments 3a and 4.

### 0.9 What was known when this was approved

The validation results of B0, B1-LR, B1-XGB, every `+aug` variant and B2 at all three levels
(EXP-003–EXP-005, EXP-017–EXP-019), the red-flag layer's (EXP-008, EXP-019), and the fused ranks of
the four hand-written golden cases (EXP-006). **H1 restates a validation finding** (EXP-019: at 25%,
must-not-miss recall@3 0.997 for ML against 0.925 for B1-LR), so it is a confirmation on the test
split, and is reported as one. **H2 is genuinely open:** no fused ranking of a DDXPlus patient had
been scored. **H3's** ML component already meets its targets on validation. No MLP existed. The
test split had never been read.

---


## Part 2 — the complete uncommitted diff of `docs/05-evaluation-protocol.md`, verbatim

```diff
diff --git a/docs/05-evaluation-protocol.md b/docs/05-evaluation-protocol.md
index bef769b..aae6519 100644
--- a/docs/05-evaluation-protocol.md
+++ b/docs/05-evaluation-protocol.md
@@ -1,12 +1,13 @@
 # 05 — Evaluation Protocol (Pre-Registered)
 
-**Version:** 1.4 · 2026-10-02 (amendments 1–3 and 3a, §9) · **Owner:** P4
+**Version:** 2.0 · 2026-10-02 (amendments 1–4, §9; **amendment 4 simplified the evaluation: read §0 first**) · **Owner:** P4
 **Status:** 🔒 **FROZEN** as of 2026-09-17, end of Phase 0 — before any model was trained.
 Amended on 2026-09-19 through §9, still before any model was trained or evaluated.
 Amended again on 2026-09-25 (amendment 3), **after** B0 and B1 had been trained and scored on the
 validation split; its §9 entry lists what was known when it was written. Amended on 2026-10-02
 (amendment 3a, the errata to amendment 3), with EXP-019's descriptive results known; its §9
-entry says so
+entry says so. Amended the same day by amendment 4, which **simplified the project and its
+evaluation** (§0), with the results its §9 entry lists known
 
 > Frozen means: no metric, baseline, ablation, or success threshold in this document may change
 > without a dated entry in the amendment log (§9) **and** disclosure in the final report. No model
@@ -23,8 +24,121 @@ entry says so
 
 ---
 
+## 0. Amendment 4 (2026-10-02): the simplified evaluation, which is what is judged now
+
+*Where §§1–8 differ from this section, this section governs; they are kept for the record. Approved
+by the team on 2026-10-02 (relayed by the owner), with the results listed in its §9 entry known.*
+
+**Why.** The owner and the team judged the project too large for a college major project and too
+hard to explain to the mentor, so its scope was cut to what the work so far can answer, and the
+evaluation with it. As amended through 3a, the evaluation could not have supported any claim
+against B1 at any level (E-2: the comparators saturate at 100% and 50%), so its hypotheses had to
+be restated in any case.
+
+### 0.1 The systems
+
+| Label | What it is | Role |
+|---|---|---|
+| B0 | Prevalence prior (EXP-003) | The floor |
+| B1-LR | Logistic regression trained on full histories (EXP-004) | The ordinary model |
+| **ML** = B1-LR′+aug | Logistic regression trained on full histories **and** masked copies, with the "asked" channel (EXP-018) | The robust model, and the system's ML component |
+| **KG** = B2 | The knowledge graph's naive-Bayes score (EXP-005) | The graph alone |
+| **Fusion** | KG and ML pooled by the logarithmic opinion pool (`src/fusion/pool.py`) at the weight EXP-006 chooses; red flags off | The fused ranking |
+| **System** = A0 | Fusion, then the red flags and the safety check | The full system |
+| B1-DL | A small multi-layer perceptron (MLP) trained like ML: masked copies, the "asked" channel | The deep-learning comparison, reported |
+| B1-XGB | XGBoost (EXP-004) | Reported beside the others, never judged |
+
+The four ablation rows are **ML, KG, Fusion and System**. B1-XGB+aug, B1-XGB′+aug, B1-LR+aug,
+A1–A7, A1b, B3 and B4 are no longer part of the evaluation; figures already measured for them stay
+in docs/08.
+
+### 0.2 Evidence levels
+
+Every system is judged at **100%** and **25%** of each patient's evidence, with §3.7's masks
+unchanged (a run that does not reproduce both recorded digests is invalid). The 50% level is still
+computed and reported in an appendix, and is not judged.
+
+### 0.3 Research questions and hypotheses
+
+Each is judged once, on the test split (Phase 4), for the seed-42 models.
+
+| # | Question | Hypothesis | Tests (each must pass, in the stated direction) |
+|---|---|---|---|
+| **RQ1** | Does training on incomplete histories make the ranker safe when a history is incomplete? | **H1:** at 25%, ML > B1-LR on must-not-miss recall@3 and on top-3 accuracy | McNemar on top-3: over must-not-miss patients, and over all patients |
+| **RQ2** | Does fusing the knowledge graph with the model improve the ranking? | **H2:** at 25%, Fusion > ML and Fusion > KG on MRR | The paired bootstrap of the per-case difference in reciprocal rank, against ML and against KG |
+| **RQ3** | Is the full system safe? | **H3:** at 100% and at 25%, System meets the §3.2 targets: must-not-miss recall@3 ≥ 0.95 and dangerous false-negative rate ≤ 0.02 | The point estimates against the targets; their 95% intervals are reported beside them |
+
+**Reported, not judged:** every system at both levels on every §3.1–§3.3 metric; Fusion against ML
+and KG at 100% (R-16: ML saturates there, so a tie is expected); System against ML on must-not-miss
+recall@3 (McNemar on must-not-miss patients); B1-DL against ML at 25% with H1's and H2's tests;
+red-flag sensitivity and precision (§3.2's 0.95 sensitivity target is reported as missed wherever it
+is); the expected calibration error of the calibrated fused scores against §3.3's 0.10 target at
+100%, with the 25% figure beside it (one calibrator, fitted on validation at 100%); the four golden
+cases, including aortic dissection with red flags off.
+
+### 0.4 Statistical tests
+
+- A 95% bootstrap interval for every ratio metric (1,000 resamples of the cases, seed 42).
+- **McNemar's test** (exact binomial below 25 discordant pairs) **passes** when p < 0.05 **and** the
+  claimed system has more of the discordant wins.
+- **The MRR test passes** when the 95% interval of the paired difference (the claimed system minus
+  its comparator) lies **entirely above 0** (`paired_bootstrap_difference`: `better == "a"`).
+- A hypothesis is supported only when every one of its tests passes. A single failure defeats it,
+  so no multiplicity correction is needed. Every test is reported, passing or not.
+
+### 0.5 Seeds and tuning
+
+Every model trained on masked copies (ML and B1-DL) is trained with seeds **42–46**: the seed-42
+model is the one used and judged, and the mean ± standard deviation over the five is reported
+beside it. B1-LR is deterministic and fitted once. XGBoost is not retrained. No ML model is tuned
+beyond early stopping on a 10% training holdout (T-6), and the MLP's size is fixed before it is
+trained. The only quantities tuned on validation are the fusion's weight (EXP-006) and the
+calibrator.
+
+### 0.6 Success criteria
+
+| Level | Criteria |
+|---|---|
+| **Minimum** | The pipeline runs end to end; System beats B0; every metric with its interval; the four ablation rows at both judged levels |
+| **Target** | H1 and H3 supported |
+| **Stretch** | H2 supported, and the calibrated fused scores' ECE < 0.10 at 100% |
+
+Failing the Target while meeting the Minimum is still a publishable outcome if the report explains
+why (§7).
+
+### 0.7 Explanations
+
+An explanation is built from the graph's matched findings and the logistic regression's largest
+contributions to the score, by a template. A local language model may paraphrase it, and may add
+nothing that the template did not contain. §3.5's qualitative review applies (50 cases, two team
+members). Evidence retrieval is not built, so §3.4 is not measured, and the first-written H3 (a
+retrieval-grounded explanation against an LLM alone) is withdrawn.
+
+### 0.8 Precision fixes (from E-2)
+
+- A mask digest is the first 16 hexadecimal digits of the SHA-256 of §3.7's lines. In the second
+  digest, the questions asked are sorted by the number after `E_` (E_2 before E_10).
+  `scripts/check_mask_rules.py` computes the first, `scripts/evaluate_reduced_evidence.py` both, and
+  both are recorded in docs/08 before anything is scored on the masks.
+- EXP-006's rule binds as committed with this amendment; a later change needs an entry in §9.
+- §8 rule 8's disclosure includes amendments 3a and 4.
+
+### 0.9 What was known when this was approved
+
+The validation results of B0, B1-LR, B1-XGB, every `+aug` variant and B2 at all three levels
+(EXP-003–EXP-005, EXP-017–EXP-019), the red-flag layer's (EXP-008, EXP-019), and the fused ranks of
+the four hand-written golden cases (EXP-006). **H1 restates a validation finding** (EXP-019: at 25%,
+must-not-miss recall@3 0.997 for ML against 0.925 for B1-LR), so it is a confirmation on the test
+split, and is reported as one. **H2 is genuinely open:** no fused ranking of a DDXPlus patient had
+been scored. **H3's** ML component already meets its targets on validation. No MLP existed. The
+test split had never been read.
+
+---
+
 ## 1. Research questions and hypotheses
 
+*Amendment 4: where this section differs from §0, §0 governs; it is kept for the record.*
+
 | # | Research question | Hypothesis |
 |---|---|---|
 | **RQ1** | Does fusing KG reasoning with an ML ranker improve differential accuracy over either alone? | **H1:** Fusion > ML-only and > KG-only on Top-3 accuracy and MRR. **H1-R** *(amendment 3)*: the same, at **both** 50% and 25% evidence (§3.7), against every ML-only system (B1-XGB, B1-LR and B1-DL, §4, and the B1 variants trained like A0's ML component *(amendment 3a)*) |
@@ -129,6 +243,8 @@ No score is reported as a probability until it passes calibration.
 
 ### 3.4 Evidence retrieval
 
+*Amendment 4: where this section differs from §0, §0 governs; it is kept for the record.*
+
 Recall@k, Precision@k, nDCG@10, plus **supported-claim rate** (fraction of generated claims
 traceable to a retrieved passage), **unsupported-claim rate**, and **citation correctness** (does
 the cited passage actually support the claim?).
@@ -149,6 +265,8 @@ Latency (p50/p95) with and without the LLM; graceful-degradation coverage (each
 
 ### 3.7 Evidence levels *(amendment 3)*
 
+*Amendment 4: where this section differs from §0, §0 governs; it is kept for the record.*
+
 Every system in §4 and every configuration in §5 is scored at three evidence levels, except as
 stated below for B3 and B4 *(amendment 3a)*, on the same patients: **100%**, the patient as DDXPlus lists them (every result as first defined), **50%** and
 **25%**. Every §3.1–§3.3 metric, with its interval, is reported at every level, and the §3.2–§3.3
@@ -217,6 +335,8 @@ compute allows. If they are not, the report says so.
 
 ## 4. Baselines
 
+*Amendment 4: where this section differs from §0, §0 governs; it is kept for the record.*
+
 | # | Baseline | Purpose |
 |---|---|---|
 | **B0** | Majority class / prevalence prior | Floor — anything below this is broken |
@@ -260,6 +380,8 @@ encoding or data.
 
 ## 5. Ablation matrix
 
+*Amendment 4: where this section differs from §0, §0 governs; it is kept for the record.*
+
 | Config | Patient KG | Medical KG | ML | Fusion | Red flags | RAG | LLM |
 |---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
 | **A0 — Full Nightingale** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
@@ -296,6 +418,8 @@ DDXPlus patient's findings (`concepts_from_evidences`); without it no rule could
 
 ## 6. Statistical testing
 
+*Amendment 4: where this section differs from §0, §0 governs; it is kept for the record.*
+
 - **Bootstrap** 1,000 resamples of the test set → 95% confidence intervals on every headline metric.
 - **McNemar's test** for paired Top-3 correctness between A0 and B1.
 - Significance level α = 0.05; **report effect sizes and CIs, not just p-values.**
@@ -329,6 +453,8 @@ DDXPlus patient's findings (`concepts_from_evidences`); without it no rule could
 
 ## 7. Success criteria
 
+*Amendment 4: where this section differs from §0, §0 governs; it is kept for the record.*
+
 | Level | Criteria |
 |---|---|
 | **Minimum** | Pipeline runs end-to-end; A0 beats B0; all metrics computed with CIs; ablations complete |
@@ -372,7 +498,7 @@ explains *why*.
    [10-spike-r01-crosswalk.md](10-spike-r01-crosswalk.md).
 8. *(Amendment 3)* **Every figure names its evidence level and its model label (§4).** A
    reduced-evidence result is never reported without the same system's full-evidence result beside
-   it, and never without the amendment-3 disclosure (§9). Reduced evidence removes answers at
+   it, and never without the amendment-3 disclosure (§9), and those of amendments 3a and 4 *(amendment 4)*. Reduced evidence removes answers at
    random, and a clinician does not ask at random, so these results measure robustness to missing
    answers. They must not be presented as a simulation of history-taking. The mask cannot move a
    patient's kept answers outside their own condition's DDXPlus evidence list, so R-16's cause is
@@ -391,5 +517,6 @@ explains *why*.
 | 2026-09-19 | **Amendment 2, §3.2.** Red-flag sensitivity is measured against the true condition: of cases whose true condition is must-not-miss, the fraction where a flag fired for that condition, overall and per condition. A must-not-miss condition without a rule counts as missed. The target is unchanged (≥ 0.95) | As first written (of cases matching a red-flag pattern, the fraction where the flag fired), the metric is 1.0 by construction for fixed rules on structured input, so it could never fail. Counting any flag instead of the condition's own would reward the over-firing EXP-014 found (R-15). Disclosure: EXP-014 had already measured each rule on its own patients on validate (MI 91%, PE 79%, Boerhaave 75%, pneumothorax 32%; unstable angina, myocarditis and acute pulmonary edema have no rule yet). The amended metric is therefore harder to meet than the original, and the target was not changed | The team, 2026-09-19 (relayed by the owner) |
 | 2026-09-25 | **Amendment 3 (decisions D-10 and D-11): §§1, 2, 3.7 (new), 4, 5, 6, 7, 8.** Every system in §4 and every configuration in §5 is also scored at 50% and 25% of each patient's evidence. The mask is defined in §3.7: whole questions, the initial evidence always kept, a kept follow-up's parent kept, one permutation per patient keyed by `case_id` and seed 42, drawn once, digest recorded. `initial_evidence` defines the mask and is never an input. H1-R, H2-R, H6 (new, RQ6) and a reduced-evidence Target and Stretch are added **beside** the first-written H1, H2, Target and Stretch. Those are still judged at 100% against B1-XGB and are reported first. B1 is not redefined: it is reported as two variants (B1-XGB, as first written; B1-LR, EXP-004's linear reference) at every level, and a claim of beating B1 must hold against each. B1-DL, a deep ranker, is added as an ML-only baseline and as A0's ML component, and A7 swaps B1 back in. §6's five seeds are 42–46 for every stochastic model. The model used is the seed-42 model, fixed before ~~any such model exists~~ any B1-DL model exists *(corrected by amendment 3a)*. A prime (′) marks models retrained with the encoder's "asked" channel (R-18); the originals stay reported | **Unlike amendments 1 and 2, this amendment was written after models had been trained and evaluated.** When it was drafted (2026-09-23), B0 and both B1 models had been scored on the validation split (EXP-003, EXP-004). B1's saturation of full-evidence DDXPlus was known (R-16), so H1, H2 and the Target cannot be met at 100%. EXP-017 had scored both B1 models at these exact levels, under a token-level version of this mask: at 25%, XGBoost's top-1 was 0.27 and its must-not-miss recall@3 0.935, while logistic regression held 0.86 and 0.974. The KG-only graph had been scored at 100% (EXP-015, EXP-016, and with the 2a score EXP-005). The 50% and 25% levels themselves had been proposed in decision D-10 on 2026-09-20 (commit `98c746b`), before any model was scored at them; EXP-017 used them three days later. The causes are R-16 and the project supervisor's email of 2026-09-20, which suggested deep learning for a more robust model. The mask works by question rather than by token because, at 25%, EXP-017's token mask left 53% of validation patients with a positive follow-up answer whose parent "yes" had been dropped, which never happens at full evidence. This was measured on the masks alone; no model was scored under the new rule before it was fixed. **Between the proposal and its approval** (2026-09-24 to 2026-09-25): the red-flag rules were revised and scored on validation at 100% (EXP-008); and B1-LR+aug, B1-XGB+aug, B1-LR′+aug and B1-XGB′+aug were trained, with seed 42 only (EXP-018; B1-LR and B1-XGB themselves are unchanged). Those four were scored on validation at 100% only, and on two short inputs that use no mask: a patient with no findings, and the four golden cases. The short inputs showed that the `+aug` models no longer answer atrial fibrillation to short input, and that they are overconfident on it; this bears on H6 and on the `+aug` arm chosen for A0. The XGBoost `+aug` models chose their stopping round on held-out training patients, half of them masked copies (§3.7 allows masks on the train split). Before approval, models had been scored on masked **validation** patients only in EXP-017, under the token rule; EXP-017 also probed a patient with no findings. What keeps the amendment honest: the levels, the mask and the new hypotheses are fixed before any fusion result on DDXPlus data or any deep-model result exists; the test split has never been read; the first-written hypotheses and Target are kept and reported first; and every claim against B1 must hold against both variants, so knowing that XGBoost degrades at reduced evidence cannot flatter any system. The reduced-evidence validation figures of B0, B1 and B2 are descriptive, not pre-registered | The team, 2026-09-25 (relayed by the owner) |
 | 2026-10-02 | **Amendment 3a (decision E-1, the errata to amendment 3): the header and §§1, 3.7, 4, 5, 6 and 7, and amendment 3's row in §9 (item 7, struck through rather than replaced), as listed in `docs/proposals/05-amendment-3-errata.md`, items 1–18; each changed passage is marked *(amendment 3a)*.** Corrections of contradictions and gaps in amendment 3's approved wording, and of the B2 description: one verdict rule at 100% (item 2); what significance a hypothesis, a Stretch and a Target need (3); a test for MRR (4); `+aug` logistic regressions take the five seeds (5); what an arm is (6); a second mask digest for the 'asked' input (8); no ML model tuned beyond early stopping (11, decision T-6). **Item 17, option (b):** H1-R, H2-R and the reduced-evidence Target must also hold against the B1 variants trained like A0's ML component. **Item 18:** each fused configuration's weight is chosen for its own ML component, with A7 at A0's weight reported beside it. A1b (no medical KG, red flags kept) is added beside A1 (item 16). No metric, level, mask or threshold value changes; an ablation (A1b) is added, B2's definition corrected, the reduced-evidence criteria's comparators widened, and a second digest made a validity condition | Items 1–16 were found by an independent review of the applied text on 2026-09-25, the day amendment 3 was approved; item 17 was raised by EXP-019's results the same day, and said so; item 18 came from the review of 2c's fusion on 2026-09-30. **What was known when the errata were approved:** EXP-019 had scored B0, B1, B2, the four `+aug` B1 variants and the red-flag layer at the three levels (descriptive), so item 17's choice was made knowing that the `+aug` variants nearly saturate top-3 and must-not-miss recall@3 at 25%; it makes the claims harder to meet, not easier. Item 3 was approved knowing the same results, and it settles the Target's test in the easier of the two readings amendment 3 allowed (point estimates; the hypotheses and the Stretch need significance), as the first-written Target and Stretch wording implies. **Found after the approval, when it was applied:** at 50% the `+aug` comparators are perfect or within four misses on validation, and H1-R, H2-R, H6 and the reduced Target and Stretch each need both 50% and 25%, so on these figures none of them can be met as a whole, and with R-16 at 100% no comparative claim against B1 could be supported at any level (§1). EXP-019 had published the 50% figures before the approval, but the recommendation cited only the 25% ones, so the team is asked again (E-2). The fusion's design and the rule that picks its weight were recorded in EXP-006, with the fused ranks of the four hand-written golden cases; no fused ranking of a DDXPlus patient had been scored, and no deep model existed. **Not settled by the approval itself:** the team approved the recommendations as a whole ("go with recommendations"), but items 15 and 16 offered two options with no recommendation, so Claude chose (15: the note, not a calibration column; 16: A1b, with H4 still judged on A1 as first written, which applies both of item 16's options); the request sent to the team had named only items 17 and 18 as choices. Item 17 (b) had no drafted wording, so the §1 and §7 text implementing it is Claude's. Item 7's replacement text omits "never replaced by the best-scoring seed"; Claude kept the clause, since the item corrects only the "fixed before" claim and the errata change no rule. The team may change any of these by a further entry before a result depends on it | The team, 2026-10-02 (relayed by the owner) |
+| 2026-10-02 | **Amendment 4 (the scope reduction): a new §0, which governs where §§1–8 differ; §§1, 3.4, 3.7, 4, 5, 6 and 7 marked; §8 rule 8 extended.** The project is cut to three research questions: RQ1, does training on incomplete histories make the ranker safe at 25% (H1: ML > B1-LR on must-not-miss recall@3 and top-3); RQ2, does fusing the knowledge graph improve the ranking at 25% (H2: Fusion > ML and > KG on MRR); RQ3, does the full system meet the §3.2 safety targets at 100% and 25% (H3). Judged at 100% and 25%; 50% reported in an appendix. Systems: B0, B1-LR, ML (B1-LR′+aug), KG (B2), Fusion, System (A0), a small MLP (B1-DL) reported, B1-XGB reported; four ablation rows. McNemar and the MRR test pass only in the claimed direction. Masked-copy models take seeds 42–46; XGBoost is not retrained (EXP-020 cancelled). Explanations by template, optionally paraphrased by a local LLM; no retrieval, so §3.4 is not measured and the first-written H3 is withdrawn. E-2 is resolved by it: its item 1 because the reduced-evidence claims are now judged at 25% only, its items 2–11 by §0's definitions | The owner and the team judged the project too large for a college major project and too hard to explain to the mentor. And as amended through 3a, no claim against B1 could be supported at any level (E-2), so the hypotheses had to be restated anyway. **This amendment was written after results were known**, all of them listed in §0.9: the validation results of every model and of the graph at the three levels, the red-flag layer's, and the fused ranks of the four hand-written golden cases. **H1 restates a validation finding** (EXP-019) and is a confirmation on the test split; H2 is genuinely open, since no fused ranking of a DDXPlus patient had been scored; H3's ML component already meets its targets on validation. Choosing 25% as the judged reduced level was made knowing that the `+aug` comparators saturate at 50% (E-2). No MLP existed, and the test split had never been read. What keeps it honest: the first-written and amended text stays in place and is reported; every claim names its evidence level; the test split is still opened once | The team, 2026-10-02 (relayed by the owner) |
 
 *Amendments after the freeze date require an entry here and must be disclosed in the final report.*
```
