# 5 · Evaluation protocol

> **Draft 1, 2026-10-14.** This chapter summarises `docs/05-evaluation-protocol.md` at version 1.4
> (2026-10-02). Where this summary and `docs/05` differ, `docs/05` governs.

## 5.1 Pre-registration

The protocol was written first and **frozen on 2026-09-17, before any model was trained**. After
the freeze, a metric, baseline, ablation or threshold may change only through a dated entry in its
amendment log (`docs/05` §9), and every amendment must be disclosed in this report. Four amendments
have been made (Table 5.1); each entry records what was known when it was written. Chapter 7
(§7.10) lists every disclosure in date order.

**Table 5.1 — Amendments to the protocol** (`docs/05` §9).

| Version | Date | Amendment | What was known when it was made |
|---|---|---|---|
| 1.1 | 2026-09-17 | Reporting rule §8.7, the circularity caveat; then frozen | No model trained |
| — | 2026-09-19 | **Amendment 1 (D-8):** Precision@3 and Recall@5 use `D_in`; Recall@5 on the full `D` reported beside it | No model trained; the label audit (EXP-013) only |
| — | 2026-09-19 | **Amendment 2:** red-flag sensitivity measured against each patient's true condition | EXP-014's per-rule rates on validation; the amended metric is harder to meet, and the target was not changed |
| 1.3 | 2026-09-25 | **Amendment 3 (D-10, D-11):** every system also scored at 50% and 25% of each history; H1-R, H2-R, H6 and reduced-evidence criteria added beside the originals; B1 reported as B1-XGB and B1-LR; the deep ranker B1-DL added as A0's ML component; five seeds | **Written after B0 and B1 had been trained and scored** (EXP-003, EXP-004), with B1's saturation (R-16) and EXP-017's partial-history diagnostic known |
| 1.4 | 2026-10-02 | **Amendment 3a (E-1), 18 errata to amendment 3:** among them, the reduced-evidence claims must also hold against the `+aug` B1 variants; MRR compared by a paired bootstrap; `+aug` logistic regressions take five seeds; no model tuned beyond early stopping (T-6); A1b added | EXP-019's descriptive results at all three levels |

## 5.2 Data and splits

The protocol uses DDXPlus filtered to the 13 conditions, with DDXPlus's own train, validation and
test splits, split at the patient level. **Everything tuned (the fusion weight, the calibrators,
the choice of ranker) is tuned on validation only. The test split is opened once, in Phase 4,
after all development is frozen.** The inputs are age, sex and the decoded evidence;
`DIFFERENTIAL_DIAGNOSIS` is the ranking label and is never an input; `PATHOLOGY` is the top-1
label (`docs/05` §2). `INITIAL_EVIDENCE` decides what a mask keeps (§5.4) and is never an input.

## 5.3 Metrics

Let the system output ranked candidates ĉ₁…ĉₙ, let y be the true pathology and `D` the
ground-truth differential; `D_in` is the part of `D` inside the 13 in-scope conditions (amendment
1).

**Table 5.2 — Metrics and targets** (`docs/05` §3).

| Group | Metric | Definition | Target |
|---|---|---|---|
| Ranking | Top-1, Top-3, Top-5 accuracy | Share of patients with y at rank 1, in the top 3, in the top 5. **Top-3 is the primary ranking metric** | — |
| | MRR | Mean of 1 / rank(y) | — |
| | Precision@3 | \|top-3 ∩ `D_in`\| / 3 | — |
| | Recall@5 | \|top-5 ∩ `D_in`\| / \|`D_in`\|, with the full-`D` figure beside it | — |
| | Per-condition F1 | Macro and micro over the 13 conditions | — |
| **Safety (headline)** | **Must-not-miss recall@3** | Of patients whose true condition is must-not-miss, the share with it in the top 3 | **≥ 0.95** |
| | Dangerous false-negative rate | Share of must-not-miss patients ranked outside the top 5 | ≤ 0.02 |
| | Red-flag sensitivity | Of must-not-miss patients, the share flagged **for their own condition**; a condition without a rule counts as missed (amendment 2) | ≥ 0.95 |
| | Red-flag precision | Of fired flags, the share clinically appropriate (decision A-7's mapping, with the strict version beside it) | Reported, not targeted |
| Calibration | ECE (10 bins), Brier score, reliability diagrams | Per condition and overall | **ECE < 0.10** |
| Retrieval | Recall@k, Precision@k, nDCG@10; supported- and unsupported-claim rates; citation correctness | 100 candidate–evidence pairs labelled by two team members, with Cohen's κ | Unsupported claims < 20% (Target) |
| Explainability | Reasoning validity, evidence relevance, clarity, 1–5 | 50 cases scored by two team members; a clinically trained reviewer supersedes if available. Team members are never described as clinical experts | — |
| System | Latency p50 and p95, with and without the LLM; graceful-degradation coverage | Each failure of `docs/02` §7 exercised | NFR-1: < 5 s without the LLM, < 20 s with it |

Red-flag precision is deliberately not targeted: over-flagging is an accepted cost (`docs/04` §3).
Because half of DDXPlus's patients have a must-not-miss condition, we measure **alarm fatigue as the
share of patients without a must-not-miss condition who get any flag**, never as the share of all
patients (EXP-008), and we report each rule's rate on everyone else beside its sensitivity (R-15).

All ratio metrics are computed by `src/eval/metrics.py`, with a 95% bootstrap interval (1,000
resamples, seed 42). A percentile bootstrap cannot show uncertainty around a perfect score, so a
perfect score is reported as [1.000, 1.000], with the exact lower bound beside it where it matters
(EXP-004).

## 5.4 Evidence levels and the masks

**Why.** On full DDXPlus histories every ML model is near-perfect, which leaves the headline
comparisons no room (R-16). Amendment 3 therefore scores **every system and every ablation at
100%, 50% and 25% of each patient's evidence**, on the same patients, with every §5.3 metric and
its interval at every level, and with the safety and calibration targets applying unchanged at
every level. Retrieval and explainability are measured at 100% only (`docs/05` §3.7).

**The mask** (`docs/05` §3.7). Age, sex and the patient's initial evidence (with all its answers)
are always kept. The distinct question codes of the patient's other tokens, in order of first
appearance, form a list Q. Each patient gets one permutation of Q, from a generator seeded by the
patient's id:

```python
rng   = numpy.random.default_rng(
            int.from_bytes(hashlib.sha256(f"42:{case_id}".encode("utf-8")).digest()[:8], "big"))
order = rng.permutation(len(Q))
kept  = [Q[i] for i in order[: round(len(Q) * share)]]   # Python's round: halves go to even
```

- A kept question keeps every one of its tokens; a dropped question loses every one.
- A kept follow-up question brings its parent (the pain questions follow "pain related to the
  consultation?").
- Every level is a prefix of the same permutation, so a patient's 25% history lies inside their 50%
  history.
- **The whole system sees only the masked patient:** the model, the graph and the red-flag rules
  read the kept tokens only.
- For models with the "asked" channel, the unlisted questions count as asked at the same share,
  from a second permutation drawn after the first (decision A-9, kept by the team on 2026-10-02).

**Drawn once, recorded before use.** The validation masks were drawn once, by
`src/ml/evidence_masks.py` under NumPy 1.26.4, and their digests were committed (`0f47d67`) before
any system was scored on them (EXP-019):

| | Value |
|---|---|
| Mask digest (the kept tokens, `docs/05` §3.7) | **`e99a7a8fbf792675`** |
| Asked-set digest (the questions asked, for models with the channel; amendment 3a) | **`af3a357f6454838a`** |
| Realised size, 100% | 21.90 tokens, 15.43 questions per patient |
| Realised size, 50% | 11.87 tokens, 8.61 questions; 42.57 of the 84 questions asked |
| Realised size, 25% | 6.71 tokens, 5.10 questions; 21.99 of the 84 questions asked |

NumPy does not promise the same random stream across versions, so the recorded masks are the
reference: a run that does not reproduce both digests is invalid, and every scoring script refuses
to run unless they do. The cloud runs reproduced them under NumPy 2.1.3 (EXP-021). The same code, at
the same commit, will draw the test split's masks in Phase 4, within its single opening.

**Training on masked patients** is allowed on the training split only, with masks drawn from the
training seed, never from the evaluation masks; such a model carries the suffix `+aug`.

**What the masks are not.** They remove answers at random, and a clinician does not ask at random,
so these results measure robustness to missing answers. **They are not a simulation of
history-taking** (`docs/05` §8 rule 8; §7.6).

## 5.5 Baselines and model labels

**Table 5.3 — Baselines** (`docs/05` §4).

| # | Baseline | Purpose |
|---|---|---|
| B0 | Prevalence prior: the same ranking for everyone | The floor; anything below it is broken |
| B1 | ML only, in two variants: **B1-XGB** (XGBoost, as first written) and **B1-LR** (multinomial logistic regression) | The serious competitor |
| B1-DL | The deep ranker, ML only, on the same features | A0's ML component; measured against B1 at every level; never replaces B1 as the comparator |
| B2 | The graph alone: the naive-Bayes graph score (EXP-005) | Is the graph useful alone? |
| B3 | A local LLM alone: the case in the prompt, no retrieval, no graph | The "just use an LLM" position [pending: LLM baselines B3/B4, EXP-009/010] |
| B4 | Text retrieval + LLM: no graph, no ML | Does retrieval alone suffice? [pending: LLM baselines B3/B4, EXP-009/010] |

**B1 is the baseline that matters:** if the full system cannot beat it, the architecture is not
justified, and the report must say so. A claim that a system beats B1 must hold against **both**
variants, because EXP-017 found them 0.0002 apart in top-1 at full evidence and 0.59 apart at 25%
(under its token mask), so choosing one after the results would decide the outcome. No ML model is
tuned beyond early stopping (T-6).

**Labels.** A label names the model, its encoder and its training rows: `+aug` means trained with
masked copies; ′ (prime) means the encoder with the "asked" channel. Two models are compared as
models only when they share both. **B1-DL** in the hypotheses means **B1-DL+aug, seed 42**, the arm
A0 uses, and "trained like it" means B1-LR+aug and B1-XGB+aug; primed systems are compared with the
primed variants trained like them. The configured pipeline runs the primed arm, A0′.

## 5.6 Ablations

**Table 5.4 — The ablation matrix** (`docs/05` §5).

| Config | Patient KG | Medical KG | ML | Fusion | Red flags | RAG | LLM |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **A0, full Nightingale** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| A1, no medical KG | ✓ | ✗ | ✓ | n/a | ✗ | ✓ | ✓ |
| A1b, no medical KG, red flags kept | ✓ | ✗ | ✓ | n/a | ✓ | ✓ | ✓ |
| A2, no ML | ✓ | ✓ | ✗ | n/a | ✓ | ✓ | ✓ |
| A3, no RAG | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| A4, no red flags | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| A5, no calibration (A0 with uncalibrated scores) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| A6, no patient KG (flat features) | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| A7, classical ML in place of the deep ranker | ✓ | ✓ | B1 | ✓ | ✓ | ✓ | ✓ |

✓ present · ✗ removed · n/a not applicable (`docs/05` §5 marks it "—").

Every configuration is evaluated on every metric, at all three levels. A0's ML component is
**B1-DL+aug, seed 42** (decision D-11); A7 swaps in a B1 variant trained like it, reported twice
(A7-LR, A7-XGB), so the deep ranker's own contribution inside the full system is measured. A1 also
switches off the red flags, so it cannot isolate the medical KG; H4 is judged on A1 as first
written, and A1b, which keeps the red flags (and the crosswalk through which they read a patient's
findings), is reported beside it. Each fused configuration's weight α is chosen for its own ML
component by EXP-006's rule, with A7 at A0's α reported beside it. The expected headline finding,
written before any result, was that **A1 and A4 degrade the safety metrics disproportionately**:
the graph earns its place on safety rather than raw accuracy.

## 5.7 Statistics

At each evidence level separately (`docs/05` §6):

- a **95% bootstrap interval** (1,000 resamples, seed 42) on every headline metric; **no metric is
  reported without an interval**;
- **McNemar's test** on paired top-3 correctness between A0 and **each** B1 variant, and between
  B1-DL and each variant trained like it; for H2-R and H6 also on must-not-miss patients alone;
- **MRR** compared by a paired bootstrap of the per-case difference in reciprocal rank (1,000
  resamples, seed 42), passing when the 95% interval excludes 0;
- significance level α = 0.05, with effect sizes and intervals, not only p-values;
- a hypothesis is supported, and a Stretch met, only when **every one** of its comparisons passes
  (an intersection–union test, so no multiplicity correction is needed); a Target compares point
  estimates, with its tests reported beside it;
- **five seeds, 42–46**, for every stochastic model and for the `+aug` logistic regressions, whose
  rows depend on the seed. **The model used, everywhere, is the seed-42 model**, fixed before any
  B1-DL model existed and never replaced by the best-scoring seed; every seed's figures are kept,
  and each verdict says for how many of the five seeds it would also hold.

## 5.8 Success criteria and reporting rules

**Table 5.5 — Success criteria** (`docs/05` §7).

| Level | Criteria |
|---|---|
| **Minimum** | Pipeline runs end-to-end; A0 beats B0; all metrics computed with CIs; ablations complete |
| **Target** | A0 > B1 on Top-3 **and** must-not-miss recall; ECE < 0.10; unsupported-claim rate < 20% |
| **Stretch** | A0 > B1 with statistical significance; must-not-miss recall ≥ 0.95; clinician-reviewed explainability |
| **Target, reduced evidence** *(amendment 3)* | At **both** 50% and 25% evidence: A0 > **each** B1 variant on Top-3 **and** must-not-miss recall@3; the B1 variants include those trained like A0's ML component (§1) *(amendment 3a)* |
| **Stretch, reduced evidence** *(amendment 3)* | The reduced-evidence Target with every comparison statistically significant (§6); must-not-miss recall@3 ≥ 0.95 at both levels |

Minimum, Target and Stretch are judged at 100% exactly as first written, with B1 = B1-XGB, and the
B1-LR comparison is reported beside them. The reduced-evidence rows are reported **beside** them,
never in their place: meeting the reduced-evidence Target does not meet the Target. Each criterion
is judged separately for the unprimed and the primed systems. Failing the Target while meeting the
Minimum is an acceptable outcome, provided the analysis explains why.

**Reporting rules** (`docs/05` §8), which this report follows:

1. every metric with a 95% interval;
2. every ablation row, including those that contradict the hypotheses;
3. per-condition results;
4. the synthetic-data limitation beside every accuracy claim;
5. failed hypotheses stated plainly in the abstract and conclusion;
6. fusion weights, hyperparameters, seeds and model versions disclosed;
7. **the circularity caveat (R-12) with every result that claims value for the knowledge graph**,
   with its independent contributions (the hand-authored red-flag rules, aortic dissection and the
   BODHI-S enrichment) reported apart from its shared-source part;
8. every figure names its evidence level and its model label; a reduced-evidence result is never
   reported without the same system's full-evidence result beside it, and never as a simulation of
   history-taking.

## 5.9 The open decision E-2

Applying amendment 3a on 2026-10-02 showed a problem with the reduced-evidence claims as approved
(`docs/05` §1, §7; `docs/proposals/05-amendment-3b-errata.md`).

- **The facts.** H1-R, H2-R, H6 and the reduced-evidence Target and Stretch each need both 50% and
  25%. At 50%, on the validation patients, the comparators trained with masked copies are perfect
  or within four misses: top-3 misses B1-XGB+aug 1, B1-LR+aug 4, both ′+aug models 0, of 33,963;
  must-not-miss misses B1-XGB+aug 0, B1-LR+aug 3, both ′+aug models 0, of 16,943. A point estimate
  cannot exceed 1.000, and McNemar's test cannot reach α = 0.05 when the comparator misses five
  cases or fewer.
- **What follows, on these figures.** None of those claims can be met as a whole; only their 25%
  comparisons can still be won. With H1, H2 and the Target already out of reach at 100% (R-16), the
  text as approved leaves no comparative claim against B1 that can be supported at any level. The
  verdicts are taken on the test split, which has never been opened; there a significant win at 50%
  would need the comparators to miss at least six cases.
- **The question to the team (E-2, item 1),** with the options recorded:
  - (a) keep the text and report the verdicts as unreachable;
  - (b) judge those claims at 25% only, with 50% reported beside them;
  - (c) compare with the plain B1 at 50% (which does not help H6);
  - (d) declare a comparison untestable when its comparator misses five cases or fewer on the
    test split.
  Items 2–11 are smaller gaps in the amended text, to approve or amend.
- **Status.** A recommendation is on record in the proposal; the decision is the team's, and it had
  not been made on 2026-10-14. Until it is, the text stands as approved, and §8 rule 5 requires this
  report to state those failures plainly. EXP-006's fusion sweep waits for item 1, because which
  level the claims are judged at must be settled before any fused result on DDXPlus is known.

## 5.10 Rules fixed before their results, outside the protocol

Three later decisions were each written down, as code or in the experiment log, before the data
they act on were read.

- **The fusion weight (EXP-006).** α is chosen on validation from a grid of 0.05 to 1 in steps of
  0.05 (α = 0 is left out: it is neither a fusion nor the model alone, since dissection takes the
  graph's posterior at every α). An α is admissible only if all four golden cases keep their
  expectations with red flags off, for the component swept and for the other models the golden
  tests run. Among admissible α, the objective is **the mean over the three levels of MRR**, with
  one α for every level, scored with red flags off, under the constraint that the fused ranking's
  must-not-miss recall@3 is at least the ML component's own at every level. Ties go to the smaller
  α. Each component is swept on its own, and the pipeline adopts the configured component's α
  (since 2026-10-13, B1-DL′+aug's). Seeds 43–46 are fused at their arm's seed-42 α, never re-swept.
  On the hand-written cases alone, every α on the grid is admissible for all six components.
- **Calibration (EXP-007).** Four methods (none, temperature, one-vs-rest Platt, one-vs-rest
  isotonic), each fitted at 100% on validation, five-fold cross-fitted and applied unchanged at
  every level; the method with **the lowest mean cross-fitted Brier score over the three levels**
  wins, and a simpler method within 0.0005 of it is preferred. A system's calibrated probabilities
  may be shown when the chosen calibrator's cross-fitted ECE is below 0.10 at all three levels. A
  calibrator fitted at each reduced level is reported beside it, at its own level only.
- **Adopting the transformer (D-13, `src/eval/safety_floor.py`).** B1-DL′+aug replaces B1-LR′+aug
  as the configured ranker, and becomes A0′'s component, only if: (1) must-not-miss recall@3 ≥ 0.95
  and the dangerous false-negative rate ≤ 0.02 at 100% and 25%; (2) at 25% it is non-inferior to
  B1-LR′+aug, with the 95% lower bound of the paired bootstrap difference above −0.005 for both the
  must-not-miss top-3 hit and the reciprocal rank; (3) every golden case passes with it, red flags
  on and off. The same rule applies to B1-DL+aug against B1-LR+aug. It was applied once, to the
  seed-42 models (§6.8).
