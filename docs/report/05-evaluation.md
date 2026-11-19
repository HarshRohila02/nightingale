# 5 · Evaluation protocol

> **Draft 2, 2026-11-11.** This chapter summarises `docs/05-evaluation-protocol.md` at version 1.5
> (2026-11-07), the version in force when the test split was opened (2026-11-09). Where this summary
> and `docs/05` differ, `docs/05` governs. §5.9 records the answer to the decision E-2, and §5.11 the
> freeze, the rehearsal and the single opening of the test split.

## 5.1 Pre-registration

The protocol was written first and **frozen on 2026-09-17, before any model was trained**. After
the freeze, a metric, baseline, ablation or threshold may change only through a dated entry in its
amendment log (`docs/05` §9), and every amendment must be disclosed in this report. Five amendments
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
| 1.5 | 2026-11-07 | **Amendment 3b (E-2):** H1-R, H2-R, H6 and the reduced-evidence Target and Stretch are judged at **25% evidence**, their 50% comparisons run and reported beside them, labelled descriptive and not counted; H3 and H4 are point estimates with their intervals; the 100% Stretch includes the Target's criteria; a pass needs the claimed system to be the better one; ten smaller items (the comparators of each claim, the digests' form, what A1 removes, EXP-006's rule) | The validation results of EXP-019 to EXP-023, **the transformer's included**, and EXP-007's first stage; no fused ranking over the whole validation split had been scored and the test split had never been opened. The answer was "go with the recommendation", given by the owner; **the other team members' agreement is not recorded** (§5.9) |

## 5.2 Data and splits

The protocol uses DDXPlus filtered to the 13 conditions, with DDXPlus's own train, validation and
test splits, split at the patient level. **Everything tuned (the fusion weight, the calibrators,
the choice of ranker) is tuned on validation only. The test split was opened once, on 2026-11-09, after
all development was frozen** (§5.11). The inputs are age, sex and the decoded evidence;
`DIFFERENTIAL_DIAGNOSIS` is the ranking label and is never an input; `PATHOLOGY` is the top-1
label (`docs/05` §2). `INITIAL_EVIDENCE` decides what a mask keeps (§5.4) and is never an input.

## 5.3 Metrics

Let the system output ranked candidates ĉ₁…ĉₙ, let y be the true pathology and `D` the
ground-truth differential; `D_in` is the part of `D` inside the 13 in-scope conditions (amendment
1).

**Table 5.2 — Metrics and targets** (`docs/05` §3). Several rows were not measured as designed; §5.12
lists them.

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
any system was scored on them (EXP-019). The test split's masks were drawn once, by the same code,
after the freeze and before anything was scored on the split; they read no label and record no
opening, and their digests were recorded in `docs/08` on 2026-11-09 (EXP-012).

| | Validation | Test |
|---|---|---|
| Mask digest (the kept tokens, `docs/05` §3.7) | **`e99a7a8fbf792675`** | **`0136f67f1ca6542f`** |
| Asked-set digest (the questions asked, for models with the channel; amendment 3a) | **`af3a357f6454838a`** | **`795acf2689f1415e`** |
| Realised size, 100% | 21.90 tokens, 15.43 questions per patient | 21.85 tokens, 15.43 questions |
| Realised size, 50% | 11.87 tokens, 8.61 questions; 42.57 of the 84 questions asked | 11.83 tokens, 8.61 questions |
| Realised size, 25% | 6.71 tokens, 5.10 questions; 21.99 of the 84 questions asked | 6.68 tokens, 5.09 questions |

NumPy does not promise the same random stream across versions, so the recorded masks are the
reference: a run that does not reproduce both digests is invalid, and every scoring script refuses
to run unless they do. The cloud runs reproduced the validation digests under NumPy 2.1.3 (EXP-021,
and the rehearsal of §5.11). The test run's notebook drew the test masks again on Kaggle and stopped
unless both test digests equalled the two recorded; both reproduced before any label was read
(§5.11).

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
| B3 | A local LLM alone: the case in the prompt, no retrieval, no graph | The "just use an LLM" position. Run once, on 50 validation patients, with `llama3.1:8b` (EXP-009); **not carried into the evaluation** after D-5's checkpoint failed, and never run on the test split (§6.13.2) |
| B4 | Text retrieval + LLM: no graph, no ML | Does retrieval alone suffice? Run on the same 50 patients (EXP-010); not carried into the evaluation, for the same reason (§6.13.2) |

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
component by EXP-006's rule, with A7 at A0's α reported beside it.

**Three configurations are identical to A0 by construction, so they were not recomputed** (EXP-012;
a test holds each):

- **A3** (no retrieval): retrieval never ranks, so removing it cannot change an order.
- **A5** (no calibration): a calibrator never reorders, and since `calibration.file` stays null
  (§6.9.1) no calibrator ships, so A5 is A0 exactly and A0's ECE is the raw fused score's.
- **A6** (no patient KG): a DDXPlus patient has no history, so there is nothing to remove. The
  patient history of a synthetic Synthea patient is display-only and never an input (the owner's
  decision of 2026-10-20).

The expected headline finding, written before any result, was that **A1 and A4 degrade the safety
metrics disproportionately**: the graph earns its place on safety rather than raw accuracy. **The test
split did not bear it out** (§6.15): removing the graph leaves must-not-miss recall@3 exactly where it
was. On the test split the record holds A0, A0′, A1, A1b and A4 beside the baselines (Table 6.35);
A2 and A7 are not reproduced in it, and this report does not quote them.

## 5.7 Statistics

At each evidence level separately (`docs/05` §6):

- a **95% bootstrap interval** (1,000 resamples, seed 42) on every headline metric; **no metric is
  reported without an interval**;
- **McNemar's test**, two-sided, on paired top-3 correctness between A0 and **each** B1 variant, and
  between B1-DL and each variant trained like it; for H1 and H1-R against every comparator the
  hypothesis lists, including B2 and the deep ranker; for H2 and H2-R the test on must-not-miss
  patients alone is the one the verdict rests on, with the test on all patients beside it;
- **MRR** compared by a paired bootstrap of the per-case difference in reciprocal rank (1,000
  resamples, seed 42), passing when the 95% interval excludes 0;
- significance level α = 0.05, with effect sizes and intervals, not only p-values;
- a hypothesis is supported, and a Stretch met, only when **every one** of its comparisons passes
  (an intersection–union test, so no multiplicity correction is needed). **A pass has a direction**
  (amendment 3b): the test is significant **and** the system the claim favours is the better one; a
  significantly worse system fails the comparison, and the report says it is worse;
- a **Target** compares point estimates, with its tests reported beside it; **a tie fails** such a
  comparison (EXP-012, choice 9);
- **H3 and H4 have no test** (amendment 3b): they are judged on point estimates, each with its 95%
  bootstrap interval, and the report says whether the interval excludes 0 without calling it
  significance;
- **levels that count** (amendment 3b): for H1-R, H2-R, H6 and the reduced-evidence Target and
  Stretch, the comparisons at **25%**; those at 50% are run and reported, labelled descriptive, and
  not counted;
- **five seeds, 42–46**, for every stochastic model and for the `+aug` logistic regressions, whose
  rows depend on the seed. **The model used, everywhere, is the seed-42 model**, fixed before any
  B1-DL model existed and never replaced by the best-scoring seed; every seed's figures are kept,
  and each verdict says for how many of the five seeds it would also hold.

## 5.8 Success criteria and reporting rules

**Table 5.5 — Success criteria** (`docs/05` §7, as amended by 3b).

| Level | Criteria |
|---|---|
| **Minimum** | Pipeline runs end-to-end; A0 beats B0; all metrics computed with CIs; ablations complete |
| **Target** | A0 > B1 on Top-3 **and** must-not-miss recall; ECE < 0.10; unsupported-claim rate < 20% |
| **Stretch** | A0 > B1 with statistical significance; must-not-miss recall ≥ 0.95; clinician-reviewed explainability; **and, cumulatively, the Target's ECE and unsupported-claim criteria** *(amendment 3b)* |
| **Target, reduced evidence** *(amendment 3)* | At **25% evidence**, the 50% comparison reported beside it *(amendment 3b)*: A0 > **each** B1 variant on Top-3 **and** must-not-miss recall@3; the B1 variants include those trained like A0's ML component (§1) *(amendment 3a)* |
| **Stretch, reduced evidence** *(amendment 3)* | The reduced-evidence Target with every comparison statistically significant (§6); must-not-miss recall@3 ≥ 0.95 at 25% evidence *(amendment 3b)* |

Minimum, Target and Stretch are judged at 100% exactly as first written, with B1 = B1-XGB, and the
B1-LR comparison is reported beside them. The reduced-evidence rows are reported **beside** them,
never in their place: meeting the reduced-evidence Target does not meet the Target. Each criterion
is judged separately for the unprimed and the primed systems. A criterion that cannot be assessed
(the unsupported-claim rate, clinician-reviewed explainability) reads **not assessed**, never met
(EXP-012, choice 5). Failing the Target while meeting the Minimum is an acceptable outcome, provided
the analysis explains why.

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

## 5.9 The decision E-2, and its answer (amendment 3b)

Applying amendment 3a on 2026-10-02 showed a problem with the reduced-evidence claims as approved
(`docs/05` §1, §7; `docs/proposals/05-amendment-3b-errata.md`).

- **The facts.** H1-R, H2-R, H6 and the reduced-evidence Target and Stretch each needed both 50% and
  25%. At 50%, on the validation patients, the comparators trained with masked copies are perfect
  or within four misses: top-3 misses B1-XGB+aug 1, B1-LR+aug 4, both ′+aug models 0, of 33,963;
  must-not-miss misses B1-XGB+aug 0, B1-LR+aug 3, both ′+aug models 0, of 16,943. A point estimate
  cannot exceed 1.000, and McNemar's test cannot reach α = 0.05 when the comparator misses five
  cases or fewer.
- **What followed, on those figures.** None of those claims could have been met as a whole; only their
  25% comparisons could still be won. With H1, H2 and the Target already out of reach at 100% (R-16),
  the text as approved left no comparative claim against B1 that could be supported at any level.
- **The question to the team (E-2, item 1),** with the options recorded:
  - (a) keep the text and report the verdicts as unreachable;
  - (b) judge those claims at 25% only, with 50% reported beside them;
  - (c) compare with the plain B1 at 50% (which does not help H6);
  - (d) declare a comparison untestable when its comparator misses five cases or fewer on the
    test split.
  Items 2–11 were smaller gaps in the amended text, to approve or amend.
- **The answer.** On 2026-11-07 the owner took the recommendation, **option (b)** and items 2 to 11
  as drafted ("go with the recommendation"), and it was applied to `docs/05` as amendment 3b (v1.5).
  H1-R, H2-R, H6 and the reduced-evidence Target and Stretch are judged at **25%**; their 50%
  comparisons are run and reported beside them, labelled descriptive and not counted. **The other
  team members' agreement is not recorded, and this report says so.** Until the answer, the text as
  approved stood.
- **What was known when it was answered** (`docs/05` §9): the validation results of EXP-019 to
  EXP-023, including the transformer's figures, read on 2026-10-13; no fused ranking over the whole
  validation split; the test split never opened. The ablation runner's plumbing runs (2,000
  validation patients, an untuned α of 0.5) had already displayed the verdicts under both readings;
  they are stamped as not a result and nothing in them is quoted.
- **The character of the choice.** Choosing 25% as the only level at which H6 is judged is choosing the
  level at which EXP-021 already showed a small MRR gain over the logistic regression (+0.0018
  [+0.0009, +0.0026] for B1-DL′+aug): the kind of choice amendment 3 warns about. The recommendation
  itself was written on 2026-10-02, before any deep result existed.
- **What the test split showed.** At 50% the comparators trained on masked copies miss at most three of
  34,657 patients in the top 3, so no comparison there could reach significance; at 25% they miss
  41 to 97 (Table 6.37). The older reading, with 50% counted, is still computed and kept in the
  run's output: under it H1-R, H2-R and H6 fail with more comparisons, and the reduced Target is met
  by neither arm (EXP-012).

## 5.10 Rules fixed before their results, outside the protocol

Five later decisions were each written down, as code or in the experiment log, before the data
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
  On the hand-written cases alone, every α on the grid is admissible for all six components. The
  sweep ran on 2026-11-07 and chose α = 0.05 for the configured component (§6.12).
- **Calibration (EXP-007).** Four methods (none, temperature, one-vs-rest Platt, one-vs-rest
  isotonic), each fitted at 100% on validation, five-fold cross-fitted and applied unchanged at
  every level; the method with **the lowest mean cross-fitted Brier score over the three levels**
  wins, and a simpler method within 0.0005 of it is preferred. A system's calibrated probabilities
  may be shown when the chosen calibrator's cross-fitted ECE is below 0.10 at all three levels. A
  calibrator fitted at each reduced level is reported beside it, at its own level only. The rule
  chose temperature scaling for A0′'s fused score, and the owner decided on 2026-11-09 not to
  deploy it: ranks only, no probability shown (§6.9.1).
- **Adopting the transformer (D-13, `src/eval/safety_floor.py`).** B1-DL′+aug replaces B1-LR′+aug
  as the configured ranker, and becomes A0′'s component, only if: (1) must-not-miss recall@3 ≥ 0.95
  and the dangerous false-negative rate ≤ 0.02 at 100% and 25%; (2) at 25% it is non-inferior to
  B1-LR′+aug, with the 95% lower bound of the paired bootstrap difference above −0.005 for both the
  must-not-miss top-3 hit and the reciprocal rank; (3) every golden case passes with it, red flags
  on and off. The same rule applies to B1-DL+aug against B1-LR+aug. It was applied once, to the
  seed-42 models (§6.8).
- **The language-model checkpoint and H3's test (D-5, EXP-009).** Decided 2026-10-20 and
  pre-registered on 2026-10-21, before any model was called. The template explainer stays, and B3 and
  B4 are reported as not carried into the evaluation, if any of four things holds on A0+LLM's raw
  answers at 100% for the 50-patient sample: (1) the unsupported-claim rate is not under 0.20, as a
  point estimate; (2) more than 10% of its calls failed; (3) the run did not finish; (4) anything the
  guardrails delivered holds an unsupported claim or treatment language. H3 is supported only if all
  four of its paired differences, B3 and B3r each against B4 and against A0+LLM, have an interval
  above 0. The checkpoint failed, on its first rule (§6.13.2).
- **The evidence retrieval's design and rule (EXP-011).** Pre-registered on 2026-10-14, before any
  corpus existed: the hybrid stays configured unless BM25 alone or dense search alone is better by a
  paired interval that excludes 0; if the configured method's Precision@3 is under 0.5 the team decides
  whether evidence stays; and a median top-10 overlap under 0.5 between the laptop's mixed query
  vector and the full-sentence embedding counts as a poor stand-in, to be disclosed. On 2026-11-09,
  before any judging, the pool was lightened to each method's top 3 and nDCG@3 replaced nDCG@10 as the
  deciding metric. No judging followed (§6.13.1).

## 5.11 The freeze, the rehearsal and the single opening of the test split

**The runner and its gate.** The ablation runner (`scripts/evaluate_ablations.py`, with its computation
in `src/eval/`) scores every system of §5.5 and §5.6 at the three levels and every comparison of §5.7,
and judges every hypothesis and criterion for the unprimed system A0 and the primed A0′ separately.
Its rankings were tested against the real pipeline's order. The test split opens only through
`src/eval/freeze.py` (risk R-08): the tag `v0.9-freeze` must be an ancestor of the commit, nothing
in `configs/`, `src/` (outside `src/eval/`) or `tests/fixtures/` may have changed since it, the tree
must be clean, the opening is recorded, and a second opening needs a stated reason
(`--reopen-because`) and is disclosed. The runner reads no label until both mask digests reproduce.

**Choices the protocol left open.** EXP-012 was pre-registered on 2026-10-21, before any full scoring,
with ten choices where `docs/05` is silent, any of which the team could change by a dated note before
the freeze. Amendment 3b settled three: how H4 is read (a point estimate with its interval beside it),
the reading of E-2 and the cumulative 100% Stretch. **The team did not answer the others, so they stood
as written at the freeze** (a dated note of 2026-11-09), and the report says so:

- H5 is read on the point estimate of A0's top-label ECE at 100%, the top label being the fused
  ranking's first condition without the flags;
- the Minimum's "A0 beats B0" is top-3 at 100% as a point estimate, with the test beside it;
- ECE and Brier are computed for A0 and A0′ only;
- the unsupported-claim rate and clinician-reviewed explainability read **not assessed** until they
  exist;
- A7 has no verdict of its own, and is reported at both weights;
- the graph is the NetworkX build, which answers exactly as the Aura copy does (tested);
- a tie fails a point-estimate comparison.

**The rehearsal on validation (2026-11-07).** The owner ran the whole runner on Kaggle on the whole
validation split, with EXP-006's α, for all five seeds (45 minutes, commit `46f76eb`), as a rehearsal
that had to reproduce what earlier experiments recorded **exactly**. Both validation digests
reproduced, the output is stamped `valid_as_a_result`, and `scripts/check_rehearsal.py` compared 1,944
figures with EXP-019, EXP-021 and EXP-006: **1,944 were exact, none different** (§6.15.2).

**The freeze (2026-11-09).** `v0.9-freeze` (`808bceb`) was tagged after the owner's decisions of that
day (§6.9.1, §6.13.1, §6.13.2). The fusion weight, the ranker, the retrieval bundle, `llm.enabled`
false and `calibration.file` null are all in `configs/`; the golden cases held on the real components;
CI was green. Nothing under `configs/`, `src/` (outside `src/eval/`) or `tests/fixtures/` was changed
after it.

**The test split.** After the owner's yes, `test.csv` (88,582,473 bytes, a pinned revision of the
DDXPlus release) was downloaded to the laptop and built through the gate: 134,529 source rows, 34,657 in
scope, 16,972 of them with a must-not-miss condition. The masks were drawn with `--digest-only`, which
reads no label and records no opening, and **both digests were recorded in `docs/08` before anything was
scored** (§5.4). The scoring ran on Kaggle (the owner's choice of where, D-7), in a notebook that
rebuilds the parquet from the same pinned file, draws the masks again and **stops unless both digests
equal the recorded ones**, and then scores once, with the single opening confirmed on the command line.
A dress run of every cell up to the scoring cell, from a fresh clone, passed the gate and reproduced
both digests; the scoring cell was never run before the opening.

**The opening.** One opening, on 2026-11-09 (08:07:22 UTC), recorded at commit `f644faa`; it is not a
reopening, so no reason is recorded. The laptop recorded no opening of its own. Before any figure was
read, the returned files were checked: one opening; both digests equal to the two recorded; the result
file's hash equal to the one in the run's status file; the files changed since the freeze none of them
under `configs/`, `src/` or `tests/fixtures/`; the result stamped valid. The figures are in §6.15.
**Nothing was re-run, re-tuned or looked at twice, and the split must not be scored again.**

**Breakdowns** (`docs/05` §8 rule 3). For A0, A0′, B1-XGB, B1-LR, B1-DL+aug, B1-DL′+aug and B2, at
each level: top-1, top-3 and MRR (and must-not-miss recall@3 where it applies) with bootstrap
intervals, by true condition and by sex. **By-sex results are an artifact, not evidence of
fairness**: DDXPlus's demographics are synthetic (EXP-002).

**Limits that stay with the design.** DDXPlus patients are generated from the definitions the graph
is built from, so the graph's value on them is circular (R-12); the system knows 13 conditions plus
aortic dissection, which no DDXPlus patient has (R-13). The comparators are at or near 1.000 on top-3
and must-not-miss recall at 100% (R-16), and the `+aug` ones at 50%, so only the 25% comparisons can
still be won. α was chosen on validation, so fused validation figures are optimistic and the test
split gives the honest ones. The masks are random, so they measure robustness to missing answers,
not history-taking. The generated differentials are open-world, so Precision@3 and Recall@5 use `D_in`
(D-8). The test split is a second sample from the same generator as validation: it checks that
nothing was fitted to validation, not that the system works on real patients.

## 5.12 What was not assessed

The protocol asked for more than was measured. The following were **not carried out as designed**, and
the report does not present a figure for any of them.

- **Retrieval relevance.** The two-rater judging of the retrieved passages was not done: the team had
  no time. Precision@k, nDCG, pooled Recall and Cohen's κ for retrieval were never computed, and the
  evidence's relevance is unjudged (EXP-011, §6.13.1).
- **Explanation ratings.** The 150-item sheet (reasoning validity, evidence relevance and clarity,
  1 to 5) was not rated, by the owner's decision of 2026-11-09, and **no clinician-reviewed
  explainability exists**. Team members are not clinical experts and are not described as such.
- **The unsupported-claim rate for the explainer that ships.** It was measured for the
  language-model explainer only (0.741, which does not ship), on 50 validation patients with a lexical
  check; it was not measured for the template explainer. The Target's and the Stretch's criteria on
  it read not assessed (§6.15.4).
- **The language-model systems on the test split.** B3, B4 and A0+LLM were not run there.
- **Calibrated probabilities in the product.** No calibrator ships, so A5 is A0 and no probability is
  shown (§6.9.1).
- **A2 and A7 on the test split.** They are not reproduced in the record of the test results, so this
  report does not quote them (§6.15.5).
- **Real patients.** Every figure is on synthetic DDXPlus patients or four hand-written cases; no real
  patient has been seen and nothing here is clinical validation.
