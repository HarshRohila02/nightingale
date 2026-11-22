# Results

This chapter reports what we measured. Section 5.2 holds development results on the validation split: they shaped the system and they are not its verdicts. The verdicts are in Section 5.3, from the test split, which we opened once, on 2026-11-09, after the code was frozen. Every test figure is marked `[TEST]`; every other experimental figure is a validation figure. Each figure carries its 95% bootstrap interval in square brackets, its evidence level (100%, 50% or 25% of each patient's history) and its model label. Four caveats travel with every figure. The patients are synthetic, so nothing here describes real patients. The system is closed-world: it ranks 14 conditions only (risk R-13 in our risk register). The knowledge graph's figures on DDXPlus patients are circular, because DDXPlus generated its patients from the definitions the graph is built from (R-12). Reduced-evidence results measure robustness to answers removed at random, which is not how a clinician takes a history.

## 5.1 Experimental Set-up

### 5.1.1 The protocol, the data and the splits

Our evaluation protocol (version 1.5) was written and frozen on 2026-09-17, before any model was trained. After the freeze a metric, baseline, ablation or threshold could change only through a dated entry in its amendment log, and this report discloses every entry (Section 5.1.6). The protocol fixes the splits, metrics, comparisons, statistics and the rule that turns each result into a verdict, so that we could not choose a favourable reading after seeing the numbers. Experiments are numbered EXP-nnn in our experiment log, the single record of every result.

We use DDXPlus [@tchango2022], a synthetic patient dataset, filtered to 13 chest-pain conditions, with its own patient-level splits. Validation has 33,963 patients, 16,943 of them with a must-not-miss condition (one whose omission risks death, such as possible myocardial infarction or pulmonary embolism); test has 34,657 patients, 16,972 of them with one. Everything we tuned (the fusion weight, the calibrators, the choice of ranker) was tuned on validation only. We opened the test split once, on 2026-11-09, through a gate in code that checks that the code is frozen, records the opening and refuses a second. The system ranks 14 conditions: the 13 DDXPlus conditions and aortic dissection, which no DDXPlus patient has and which only the graph and the rules can rank.

### 5.1.2 Evidence levels and masks

On complete DDXPlus histories every machine-learning model is near-perfect, which leaves no room to separate systems (risk R-16, Section 5.2.1). The protocol therefore scores every system at three evidence levels: 100%, 50% and 25% of each patient's evidence. A mask keeps age, sex and the initial evidence, plus a random share of the other questions drawn from a generator seeded with the patient's identifier. A kept question keeps all its answers and brings its parent question, and each level is a prefix of the same random order, so a 25% history lies inside the 50% one. The whole system sees only the masked patient: the model, the graph and the red-flag rules (independent rules that raise an alarm for a dangerous pattern) read the kept answers only. We drew the masks once and recorded their digests (short fingerprints of the kept answers) before any system was scored on them; a run that does not reproduce both digests is invalid (Table 5.1). **The masks remove answers at random, and a clinician does not ask at random, so these results measure robustness to missing answers, not history-taking.**

**Table 5.1.** Mask digests and realised sizes of the evidence levels, validation and test splits.

| | Validation | Test |
|---|---|---|
| Mask digest (the kept answers) | `e99a7a8fbf792675` | `0136f67f1ca6542f` |
| Asked-set digest (the questions asked, for models with the "asked" channel) | `af3a357f6454838a` | `795acf2689f1415e` |
| Realised size at 100% | 21.90 answers, 15.43 questions per patient | 21.85 answers, 15.43 questions |
| Realised size at 50% | 11.87 answers, 8.61 questions | 11.83 answers, 8.61 questions |
| Realised size at 25% | 6.71 answers, 5.10 questions | 6.68 answers, 5.09 questions |

A model can be trained on masked copies of the training histories (the suffix `+aug`; training split only). A prime (′) marks a model whose encoder has an "asked" channel: one extra column per question recording whether it was asked, so the model can tell a denied question from one never asked.

### 5.1.3 The systems compared

The baseline that matters is B1: if the full system cannot beat it, the architecture is not justified. B1 has two forms, XGBoost (B1-XGB) and logistic regression (B1-LR), and a claim against B1 must hold against both. A0 is the full system with the deep ranker B1-DL+aug as its machine-learning component; A0′ uses B1-DL′+aug and is the shipped system. A3, A5 and A6 equal A0 by construction (retrieval never ranks, no calibrator ships, a DDXPlus patient has no history to remove), so we did not recompute them. Table 5.2 lists the systems.

**Table 5.2.** The systems compared (evaluation protocol §4 and §5).

| System | What it is | Purpose |
|---|---|---|
| B0 | Prevalence prior: the same ranking for everyone | The floor |
| B1-XGB, B1-LR | XGBoost and multinomial logistic regression on the encoded history | The serious competitor |
| B1-XGB+aug, B1-LR+aug and their ′ variants | The same models trained on masked copies, without and with the "asked" channel (′) | Comparators trained like A0's component |
| B1-DL+aug, B1-DL′+aug | A small transformer (225,645 parameters), trained on masked copies | A0's and A0′'s machine-learning component |
| B2 | The knowledge graph alone, scored by naive Bayes | Is the graph useful alone? Circular (R-12) |
| B3, B4 | A language model alone; retrieval plus a language model | Run once on 50 validation patients; not carried into the evaluation (Section 5.5) |
| A0, A0′ | Model fused with the graph, red flags ranked first | The full system |
| A1, A1b | No medical graph (A1 also drops the flags; A1b keeps them) | Isolates the graph |
| A2, A4 | No machine learning; no red flags | Isolates each component |
| A3, A5, A6 | No retrieval; no calibration; no patient graph | Equal A0 by construction |
| A7 | A classical model in place of the deep ranker | Not reproduced in the test record; not quoted |

### 5.1.4 Metrics and targets

All ratio metrics carry a 95% bootstrap interval: we resample the patients with replacement 1,000 times (seed 42) and report the middle 95% of the results. A percentile bootstrap cannot show uncertainty around a perfect score, so a perfect score is written [1.000, 1.000] and means that no error was observed. Table 5.3 lists the metrics.

**Table 5.3.** Metrics and targets (evaluation protocol §3).

| Metric | Meaning | Target |
|---|---|---|
| Top-1, top-3, top-5 accuracy | Share of patients whose true condition is ranked first, in the first 3, in the first 5. Top-3 is the primary ranking metric | None |
| MRR (mean reciprocal rank) | Mean of 1 divided by the rank of the true condition | None |
| Precision@3, Recall@5 | Agreement with DDXPlus's own differential, counting only its in-scope part; Recall@5 on the full differential is reported beside it | None |
| **Must-not-miss recall@3** | Of patients whose true condition is must-not-miss, the share with it in the top 3 | At least 0.95 |
| Dangerous false-negative rate | Share of must-not-miss patients whose condition is ranked outside the top 5 | At most 0.02 |
| Red-flag sensitivity | Of must-not-miss patients, the share flagged for their own condition | At least 0.95 |
| Red-flag precision | Of the flags fired, the share clinically appropriate by decision A-7's list (strict version: the flag names the true condition) | Reported, not a target |
| ECE (expected calibration error) [@guo2017] | How far the probability given to the first-ranked condition is from how often it is right, over 10 bins | Below 0.10 |
| Unsupported-claim rate | Share of an explanation's claims that trace to no source the system was given | Below 20% |
| Latency | Time per ranking, p95 (the time within which 95% of rankings finish) | Below 5 s without a language model |

### 5.1.5 Statistics and the rule behind each verdict

We compare systems on the same patients at each evidence level separately. McNemar's test (two-sided) asks whether two systems, paired on whether each ranks the true condition in the top 3, differ by more than chance; MRR is compared by a paired bootstrap of the per-patient difference in reciprocal rank. Every stochastic model has five seeds, 42 to 46. The model used everywhere is the seed-42 model, fixed before any deep model existed, and each verdict says for how many of the five seeds it holds. A hypothesis is supported only if every one of its comparisons passes, and a pass needs the system the claim favours to be the better one. A Target compares point estimates, and a tie fails. H3 and H4 are judged on point estimates with their intervals beside them, without a test. The six hypotheses are:

- **H1** (**H1-R** at 25% evidence): fusion beats machine learning alone and the graph alone on top-3 and MRR.
- **H2** (**H2-R** at 25%): fusion with red flags beats machine learning alone on must-not-miss recall@3.
- **H3**: a retrieval-grounded explanation has a lower unsupported-claim rate than a language model alone.
- **H4**: removing the graph hurts must-not-miss recall more than top-1.
- **H5**: ECE is below 0.10.
- **H6**: the deep ranker beats each classical model trained like it, at 25%.

Three success levels apply: the **Minimum** (the pipeline runs end to end and A0 beats B0), the **Target** (A0 beats B1 on top-3 and must-not-miss recall, ECE below 0.10, unsupported claims below 20%) and the **Stretch** (the Target with statistical significance, must-not-miss recall of at least 0.95 and clinician-reviewed explainability). The Target and Stretch are also judged at 25% evidence against every B1 variant, including those trained like A0's component. A criterion that cannot be assessed reads "not assessed", never "met".

### 5.1.6 Amendments, and the reading in force

Amendments 3 and 3a were written after B0 and B1 had been scored, so Table 5.4 states what was known at the time of each.

**Table 5.4.** Amendments to the frozen protocol.

| Version and date | Amendment | What was known when it was made |
|---|---|---|
| 1.1, 2026-09-17 | Reporting rule on circularity; protocol frozen | No model trained |
| 2026-09-19 | Amendment 1: Precision@3 and Recall@5 count only DDXPlus's in-scope differential | No model trained |
| 2026-09-19 | Amendment 2: red-flag sensitivity measured against each patient's true condition | Per-rule rates on validation; the target was not changed |
| 1.3, 2026-09-25 | Amendment 3: every system also scored at 50% and 25%; B1 reported as B1-XGB and B1-LR; deep ranker added; five seeds | Written after B0 and B1 were scored; B1's saturation (R-16) was known |
| 1.4, 2026-10-02 | Amendment 3a: 18 errata, including that reduced-evidence claims must also hold against the `+aug` B1 variants | The descriptive results at all three levels |
| 1.5, 2026-11-07 | Amendment 3b: H1-R, H2-R, H6 and the reduced-evidence Target and Stretch judged at 25% evidence only | Validation results including the transformer's; the test split never opened |

**The reading in force when the test split was opened is amendment 3b.** H1-R, H2-R, H6 and the reduced-evidence Target and Stretch are judged at 25% evidence only, with the 50% comparisons reported beside them, labelled descriptive and not counted, because at 50% the comparators trained on masked copies were perfect or within four misses on validation. The project owner [[name]] took the recommendation on 2026-11-07. **The other team members' agreement is not recorded, and we say so.** The choice also fell on the level at which validation had already shown a small gain for the transformer, which is the kind of choice an amendment process warns about. The older reading, with 50% counted, is still computed; under it H1-R, H2-R and H6 fail with more comparisons, and the reduced-evidence Target is met by neither arm.

Four later rules were fixed before the data they act on were read: the fusion weight's, the calibration method's, the transformer's safety floor (decision D-13) and the language-model checkpoint (decision D-5). A rehearsal of the whole scoring program on validation, before the test split was opened, reproduced 1,944 of 1,944 earlier recorded figures exactly.

## 5.2 Development Results on the Validation Split

Every figure in this section is a validation figure. We list the findings in the order in which they shaped the system.

### 5.2.1 Complete histories are saturated (R-16)

Table 5.5 shows that B1 is near-perfect on complete histories: both models rank the true condition in the top 3 for every validation patient. The cause is the data, not skill or leakage. No label column is read, and every patient's positive answers lie inside their own condition's evidence list, for 91.7% of patients inside no other condition's list, so which questions are answered "yes" nearly names the condition. The laptop also reproduces the cloud-trained models' metrics to within 3 × 10⁻¹².

**Table 5.5.** B0 and B1 at 100% evidence (validation; models as trained in EXP-004; B1-XGB and B1-LR).

| Model (100%) | Top-1 | Top-3 | MRR | Must-not-miss recall@3 | Dangerous false-negative rate |
|---|---|---|---|---|---|
| B0 | 0.1097 [0.1065, 0.1132] | 0.3059 [0.3013, 0.3108] | 0.2924 [0.2895, 0.2954] | 0.2199 [0.2137, 0.2264] | 0.6064 [0.5989, 0.6136] |
| B1-LR | 0.9983 [0.9978, 0.9987] | 1.0000 [1.0000, 1.0000] | 0.9991 [0.9989, 0.9994] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] |
| B1-XGB | 0.9985 [0.9980, 0.9989] | 1.0000 [1.0000, 1.0000] | 0.9992 [0.9990, 0.9994] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] |

So H1, H2 and the Target cannot be supported at 100%: no system can exceed 1.000, and a paired test between the full system and B1 can show only a tie or a loss. This is a property of the test, not evidence about fusion either way, and it is why amendment 3 added the reduced-evidence levels. B1's few errors are one confusion: B1-XGB ranks the wrong condition first for 51 patients and B1-LR for 59, almost all of them unstable angina placed below stable angina, and every one keeps the true condition in the top 3.

### 5.2.2 Short histories broke the original B1, and masked training repaired it (R-18)

A diagnostic run found a defect in how the original B1 reads an incomplete history: given no evidence at all, B1-XGB answered atrial fibrillation with probability 1.000. A default "no" answer had no column in the encoding, so a question answered "no" and a question never asked were the same row. Atrial fibrillation is the condition whose patients answer the fewest questions, so "few answers" predicted it (risk R-18). Every hand-written case is a short history, so this is the regime the prototype is used in.

On the protocol's masks at 25% evidence (Table 5.6), the original B1 fails both safety targets, and every model trained on masked copies meets them. The "asked" channel adds a small, significant gain: top-1 rises from 0.931 to 0.946 for logistic regression and from 0.933 to 0.948 for XGBoost, and top-3 is better for 79 patients against 42 (logistic regression, p = 0.0011). Seeds 43 to 46 left every conclusion unchanged. The retrained models' raw scores are calibrated against DDXPlus (ECE at most 0.011 at every level) but stay near-certain on hand-written cases (risk R-17, Section 5.7.2).

**Table 5.6.** B1 at 25% evidence, original and retrained (validation; seed 42; targets: must-not-miss recall@3 at least 0.95, dangerous false-negative rate at most 0.02).

| Model (25%) | Top-1 | Must-not-miss recall@3 | Dangerous false-negative rate |
|---|---|---|---|
| B1-XGB | 0.245 [0.240, 0.250] | 0.877 [0.872, 0.882] | 0.059 [0.056, 0.062] |
| B1-LR | 0.766 [0.761, 0.770] | 0.925 [0.921, 0.929] | 0.024 [0.021, 0.026] |
| B1-XGB+aug | 0.933 [0.931, 0.936] | 0.997 [0.996, 0.998] | 0.001 [0.000, 0.001] |
| B1-LR+aug | 0.931 [0.928, 0.934] | 0.996 [0.995, 0.997] | 0.001 [0.000, 0.001] |
| B1-XGB′+aug | 0.948 [0.946, 0.950] | 0.998 [0.997, 0.998] | 0.000 [0.000, 0.000] |
| B1-LR′+aug | 0.946 [0.944, 0.948] | 0.997 [0.997, 0.998] | 0.000 [0.000, 0.001] |

### 5.2.3 The deep ranker and its safety floor (D-13)

Our supervisor asked which deep model suits the approach, and at what cost. We chose a small transformer in the style of the FT-Transformer [@gorishniy2021]: one token per DDXPlus question plus age, sex and a summary token, with 225,645 parameters. Before training, rule D-13 fixed when it could replace logistic regression: it had to meet the safety targets at 100% and 25%, be non-inferior to the logistic regression trained like it at 25% (the lower bound of the paired 95% interval above −0.005, for must-not-miss recall@3 and for reciprocal rank), and pass the golden cases. Against B1-LR′+aug the paired difference is +0.0006 [−0.0001, +0.0013] on must-not-miss recall@3 and +0.0018 [+0.0009, +0.0026] on MRR (Table 5.7), so both arms pass, at all five seeds. B1-DL′+aug became the configured ranker on 2026-10-13, with B1-LR′+aug kept as the backup.

**Table 5.7.** The transformer and the logistic regressions trained like it at 25% evidence (validation; seed 42).

| Model (25%) | Top-1 | MRR | Must-not-miss recall@3 | Dangerous false-negative rate |
|---|---|---|---|---|
| B1-DL′+aug | 0.9487 [0.9464, 0.9509] | 0.9728 [0.9715, 0.9740] | 0.9981 [0.9973, 0.9987] | 0.0004 [0.0001, 0.0007] |
| B1-LR′+aug | 0.9459 [0.9436, 0.9482] | 0.9710 [0.9697, 0.9722] | 0.9975 [0.9967, 0.9982] | 0.0004 [0.0001, 0.0008] |
| B1-DL+aug | 0.9347 [0.9323, 0.9373] | 0.9649 [0.9635, 0.9663] | 0.9976 [0.9968, 0.9983] | 0.0004 [0.0001, 0.0008] |
| B1-LR+aug | 0.9311 [0.9285, 0.9337] | 0.9627 [0.9612, 0.9641] | 0.9958 [0.9948, 0.9967] | 0.0009 [0.0005, 0.0014] |

The gain is small, and against XGBoost trained like it the transformer is level. The trained transformer takes 1.38 ms at p95 for one patient on the laptop processor, and the whole pipeline with evidence retrieval takes 13.6 to 16.9 ms at p95, far inside the 5 s target and fast enough to re-rank as each finding is entered.

### 5.2.4 Fusion adds nothing measurable (EXP-006)

The system combines the model's probabilities with the graph's by a logarithmic opinion pool [@genest1986]: each source's probabilities are raised to a weight and multiplied, and α is the graph's weight. A condition the model cannot score (aortic dissection) takes the graph's probability. The rule for α was fixed before any fused ranking of a validation patient was scored: the best mean MRR over the three levels, provided fused must-not-miss recall@3 is not below the model's own. For the configured component no α met that constraint, so the rule's other branch chose the highest mean must-not-miss recall@3, and α = 0.05, the smallest on the grid, went into the configuration (0.30 for A0's component). The objective was flat: for B1-DL+aug the mean MRR is 0.9852 from α = 0.05 to α = 0.45.

Fusion adds nothing measurable (Table 5.8). The MRR difference at 25% is −0.00007 with an interval that includes 0. On top-3 the fused system is right alone for 1 patient and the model alone for 7 (exact p = 0.070), and at 100% and 50% there is no discordant patient; the fused MRR there is 0.9992 [0.9990, 0.9994] and 0.9932 [0.9926, 0.9938]. What fusion was designed for, a condition the model cannot score ranking first when the graph supports it, shows only on the hand-written case GC-003 (Appendix B).

**Table 5.8.** B1-DL′+aug alone and fused with the graph at α = 0.05 (25% evidence; validation; red flags off, as the selection rule scores it).

| System (25%) | MRR | Must-not-miss recall@3 |
|---|---|---|
| B1-DL′+aug alone | 0.9728 [0.9715, 0.9740] | 0.9981 [0.9973, 0.9987] |
| Fused with the graph, α = 0.05 | 0.9727 [0.9715, 0.9739] | 0.9978 [0.9971, 0.9985] |

### 5.2.5 The graph alone, and the red flags

The graph alone (B2) looks strong on DDXPlus, and that is circularity: at 100% on validation it reaches top-1 0.9234 [0.9206, 0.9259] and Precision@3 0.7559 [0.7532, 0.7588] with no training, because DDXPlus generated its patients from the definitions the graph is built from (R-12). It cannot recognise stable angina at all (top-1 0.000): with nothing denied, stable and unstable angina tie, and the tie rule puts the more dangerous one first. At 25% its must-not-miss recall@3 is 0.935 [0.931, 0.939], below the target. The red-flag rules, written after published patterns such as the aortic dissection detection risk score [@rogers2011] and the classification of unstable angina [@braunwald1989], raised red-flag sensitivity from 0.449 [0.441, 0.456] to 0.823 [0.818, 0.829] at 100%, still below the 0.95 target (Section 5.4).

### 5.2.6 The golden cases

All four hand-written golden cases pass with every model, red flags on and off, and their expectations were never changed (Appendix B). They also found a defect. The model's reasons (which finding an answer rests on) showed that "radiation to jaw or arm" had been translated into the one answer that only acute pulmonary edema patients give in DDXPlus, so the transformer alone answered pulmonary edema for the classic ischaemic case GC-001. We moved it to the answer ischaemic patients give (decision A-10). The other 16 hand-chosen representative answers have not been checked the same way.

## 5.3 Results on the Test Split

**Every figure in this section is `[TEST]`.** The test split was opened once, on 2026-11-09, in the project owner's run of the frozen code on Kaggle (commit `f644faa`; the tag `v0.9-freeze` is an ancestor, and nothing under `configs/`, `src/` outside `src/eval/`, or `tests/fixtures/` changed after it). Both mask digests reproduced before any label was read, one opening was recorded, and the result file's hash matched the run's status file. The run covered 34,657 patients, seeds 42 to 46 and 102 comparisons. Nothing was re-run, re-tuned or looked at twice, and the split must not be scored again. Of the seven hypotheses judged on it, **five are not supported**.

**Table 5.9.** Hypothesis and success-criterion verdicts (`[TEST]`; seed 42; reading in force, 25% evidence only for the reduced-evidence claims). They are the same as the rehearsal's on validation, row for row.

| Claim | A0 | A0′ | Seeds (of 5) | What stands behind it |
|---|---|---|---|---|
| H1, H2 (100%) | not supported | not supported | 0 | Every machine-learning comparator misses 0 of 34,657 patients in the top 3 (R-16); A0's MRR is lower than the model's |
| H1-R, H2-R (25%) | not supported | not supported | 0 | The models trained on masked copies are level on top-3 and ahead on MRR |
| H3 | not supported | n/a | n/a | Validation, 50 patients (Section 5.5); not carried to the test split |
| H4 | supported, hollow | supported, hollow | 5 | The contrast comes from the flags lowering top-1; the graph contributes nothing measurable |
| H5 | supported | supported | 5 | Raw ECE below 0.10 at 100% |
| H6 | not supported | not supported | 0 | Beats logistic regression, level with XGBoost |
| Minimum | met | met | 5 | A0's top-3 against B0's |
| Target, 100% | not met | not met | 0 | Ties at 1.000; the template explainer's unsupported-claim rate is not assessed |
| Stretch, 100% | not met | not met | 0 | Same ties |
| Target, reduced (25%) | not met | **met** | 0 (A0), 5 (A0′) | A0′ misses 40 of 34,657 patients in the top 3, a handful of patients |
| Stretch, reduced (25%) | not met | not met | 0 | Neither arm passes every test |

**Two verdicts need saying plainly.** H4's support is hollow: removing the graph leaves must-not-miss recall@3 exactly where it was, and the contrast the rule reads is the red-flag layer lowering top-1, so H4 must not be quoted as evidence that the graph helps. The reduced-evidence Target is met by one arm only, on a handful of patients, and the fused system equals its own component to within a patient, so it is not a robust win.

### 5.3.1 What stands behind each verdict

- **H1 and H2 at 100%.** Every machine-learning comparator (B1-XGB, B1-LR, B1-DL+aug and B1-DL′+aug) misses 0 of 34,657 patients in the top 3 and 0 of 16,972 must-not-miss patients, so no test can pass against them. The graph alone (B2) misses 27 patients (4 must-not-miss), so the one H1 check that passes, top-3 against B2 (1.0000 against 0.9992, p = 5.6 × 10⁻⁷), is circular. A0's MRR is lower than B1-XGB's, 0.9157 against 0.9994, a difference of −0.0837 [−0.0859, −0.0819], because the red-flag layer ranks flagged conditions first even when the flag names the wrong one: A1b, the model with flags and no graph, has the same MRR, 0.9157 [0.9135, 0.9175].
- **H1-R and H2-R at 25%.** Both arms beat the original B1 models and B2 on every check, but not the models trained on masked copies. A0 against B1-XGB+aug is −0.0001 on top-3 (p = 0.883) and −0.0052 on MRR [−0.0062, −0.0044]. A0′ ties B1-DL′+aug on top-3 (0.9988 both) and trails it on MRR (−0.0057 [−0.0063, −0.0051]). A0 fails 5 of 12 H1-R checks and 2 of 5 H2-R checks; A0′ fails 4 and 2.
- **H6 at 25%.** The transformer beats B1-LR+aug on all three metrics (MRR +0.0022 [+0.0015, +0.0028]) but is level with B1-XGB+aug (top-3 0.0000, p = 0.883; MRR +0.0002 [−0.0005, +0.0009]; must-not-miss +0.0003, p = 0.458). Primed, against B1-XGB′+aug the MRR passes (+0.0016 [+0.0007, +0.0025]) but top-3 (+0.0004, p = 0.064) and must-not-miss recall (+0.0005, p = 0.164) do not. H6 needs every comparator, so it fails.
- **H4.** Removing the graph and the flags (A1) leaves must-not-miss recall@3 where it was (a degradation of 0.0000), while top-1 rises by 0.1642 [0.1606, 0.1685], from 0.8346 (A0) to 0.9987 (A1), because the flags no longer push their conditions first. Against A1b, which keeps the flags and drops only the graph, both degradations are 0.0000 and the contrast is 0. The same holds on validation.
- **H5.** A0's raw top-label ECE at 100% is 0.0099 [0.0094, 0.0103] and A0′'s is 0.0398 [0.0394, 0.0402]; no calibrator ships (Section 5.4).
- **Minimum, Target and Stretch.** The Minimum holds: top-3 at 100%, A0 1.0000 against B0's 0.3061 [0.3013, 0.3109]. The Target and Stretch at 100% fail because the machine-learning comparators tie at 1.000 and a tie fails a point-estimate comparison. Two criteria are not assessed, the shipped template explainer's unsupported-claim rate and clinician-reviewed explainability, and neither is reported as met.
- **Target at 25%.** A0′ misses 40 of 34,657 patients in the top 3 (24 of 16,972 must-not-miss patients), B1-DL′+aug 41 (24), B1-XGB′+aug 56 (33) and B1-LR′+aug 80 (45). A0 misses 73 (32) against B1-XGB+aug's 71 (36), so A0 fails by two patients (−0.0001, p = 0.883). Against B1-XGB′+aug the must-not-miss difference for A0′ (+0.0005) is not significant (p = 0.164).

### 5.3.2 The systems at the three evidence levels

Tables 5.10 to 5.12 give ranking and safety at each level. Aortic dissection has no DDXPlus patient, so no row can show it. B2 is circular.

**Table 5.10.** Systems at 100% evidence (`[TEST]`; seed 42; 34,657 patients; B0 is the same at every level).

| System (100%) | Top-1 | Top-3 | MRR | Must-not-miss recall@3 |
|---|---|---|---|---|
| B0 | 0.1062 [0.1030, 0.1094] | 0.3061 [0.3013, 0.3109] | 0.2906 [0.2878, 0.2934] | 0.2168 [0.2105, 0.2231] |
| B1-XGB | 0.9987 [0.9983, 0.9991] | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9992, 0.9996] | 1.0000 [1.0000, 1.0000] |
| B1-LR | 0.9986 [0.9982, 0.9990] | 1.0000 [1.0000, 1.0000] | 0.9993 [0.9991, 0.9995] | 1.0000 [1.0000, 1.0000] |
| B1-DL+aug (A1) | 0.9987 [0.9983, 0.9991] | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9992, 0.9996] | 1.0000 [1.0000, 1.0000] |
| B1-DL′+aug | 0.9987 [0.9984, 0.9991] | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9992, 0.9996] | 1.0000 [1.0000, 1.0000] |
| B2 (circular) | 0.9237 [0.9209, 0.9264] | 0.9992 [0.9989, 0.9995] | 0.9612 [0.9598, 0.9626] | 0.9998 [0.9995, 0.9999] |
| A1b | 0.8346 [0.8303, 0.8382] | 1.0000 [1.0000, 1.0000] | 0.9157 [0.9135, 0.9175] | 1.0000 [1.0000, 1.0000] |
| A4 | 0.9987 [0.9983, 0.9991] | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9992, 0.9996] | 1.0000 [1.0000, 1.0000] |
| A0 | 0.8346 [0.8303, 0.8382] | 1.0000 [1.0000, 1.0000] | 0.9157 [0.9135, 0.9175] | 1.0000 [1.0000, 1.0000] |
| A0′ | 0.8346 [0.8303, 0.8382] | 1.0000 [1.0000, 1.0000] | 0.9156 [0.9134, 0.9175] | 1.0000 [1.0000, 1.0000] |

**Table 5.11.** Systems at 50% evidence (`[TEST]`; seed 42; 34,657 patients; descriptive, not counted in the verdicts).

| System (50%) | Top-1 | Top-3 | MRR | Must-not-miss recall@3 |
|---|---|---|---|---|
| B1-XGB | 0.5744 [0.5697, 0.5800] | 0.9899 [0.9887, 0.9909] | 0.7795 [0.7771, 0.7825] | 0.9858 [0.9840, 0.9875] |
| B1-LR | 0.9436 [0.9412, 0.9460] | 0.9958 [0.9951, 0.9964] | 0.9690 [0.9677, 0.9704] | 0.9930 [0.9917, 0.9942] |
| B1-DL+aug (A1) | 0.9841 [0.9827, 0.9854] | 1.0000 [0.9999, 1.0000] | 0.9919 [0.9912, 0.9925] | 1.0000 [1.0000, 1.0000] |
| B1-DL′+aug | 0.9875 [0.9862, 0.9885] | 1.0000 [0.9999, 1.0000] | 0.9936 [0.9930, 0.9942] | 0.9999 [0.9998, 1.0000] |
| B2 (circular) | 0.8673 [0.8639, 0.8708] | 0.9854 [0.9841, 0.9867] | 0.9244 [0.9223, 0.9265] | 0.9872 [0.9854, 0.9890] |
| A1b | 0.9291 [0.9262, 0.9318] | 1.0000 [0.9999, 1.0000] | 0.9639 [0.9624, 0.9653] | 1.0000 [1.0000, 1.0000] |
| A4 | 0.9843 [0.9829, 0.9856] | 0.9999 [0.9999, 1.0000] | 0.9920 [0.9913, 0.9927] | 0.9999 [0.9998, 1.0000] |
| A0 | 0.9293 [0.9265, 0.9320] | 0.9999 [0.9999, 1.0000] | 0.9640 [0.9626, 0.9654] | 0.9999 [0.9998, 1.0000] |
| A0′ | 0.9318 [0.9292, 0.9344] | 1.0000 [0.9999, 1.0000] | 0.9653 [0.9640, 0.9667] | 0.9999 [0.9998, 1.0000] |

**Table 5.12.** Systems at 25% evidence (`[TEST]`; seed 42; 34,657 patients; the level at which the reduced-evidence claims are judged).

| System (25%) | Top-1 | Top-3 | MRR | Must-not-miss recall@3 |
|---|---|---|---|---|
| B1-XGB | 0.2482 [0.2439, 0.2524] | 0.9035 [0.9003, 0.9068] | 0.5789 [0.5763, 0.5816] | 0.8710 [0.8659, 0.8764] |
| B1-LR | 0.7652 [0.7604, 0.7697] | 0.9436 [0.9412, 0.9461] | 0.8575 [0.8546, 0.8602] | 0.9198 [0.9156, 0.9238] |
| B1-DL+aug (A1) | 0.9328 [0.9298, 0.9353] | 0.9980 [0.9975, 0.9984] | 0.9639 [0.9623, 0.9652] | 0.9982 [0.9975, 0.9988] |
| B1-DL′+aug | 0.9481 [0.9456, 0.9503] | 0.9988 [0.9985, 0.9991] | 0.9723 [0.9711, 0.9735] | 0.9986 [0.9980, 0.9991] |
| B2 (circular) | 0.7676 [0.7632, 0.7717] | 0.9389 [0.9365, 0.9415] | 0.8540 [0.8512, 0.8567] | 0.9405 [0.9371, 0.9442] |
| A1b | 0.9223 [0.9192, 0.9249] | 0.9980 [0.9975, 0.9984] | 0.9585 [0.9568, 0.9599] | 0.9982 [0.9975, 0.9988] |
| A4 | 0.9328 [0.9299, 0.9354] | 0.9979 [0.9974, 0.9983] | 0.9639 [0.9623, 0.9652] | 0.9981 [0.9974, 0.9987] |
| A0 | 0.9223 [0.9192, 0.9249] | 0.9979 [0.9974, 0.9983] | 0.9584 [0.9568, 0.9599] | 0.9981 [0.9974, 0.9987] |
| A0′ | 0.9371 [0.9343, 0.9396] | 0.9988 [0.9985, 0.9992] | 0.9667 [0.9653, 0.9680] | 0.9986 [0.9980, 0.9991] |

The plain B1 variants trained on masked copies appear only in Table 5.13, because the record holds no other figure for them on the test split. The table counts the patients each system misses in the top 3 and, in brackets, the must-not-miss patients missed. A comparison against a system that misses five patients or fewer cannot reach significance, so the machine-learning comparators are at a ceiling at 100%, the ones trained on masked copies are at a ceiling at 50%, and 25% is the only level at which a comparison can still be won.

**Table 5.13.** Patients missed in the top 3, of 34,657, and must-not-miss patients missed, of 16,972, in brackets (`[TEST]`; seed 42).

| System | 100% | 50% | 25% |
|---|---|---|---|
| B1-XGB | 0 (0) | 351 (241) | 3,346 (2,190) |
| B1-LR | 0 (0) | 147 (119) | 1,954 (1,361) |
| B1-XGB+aug | 0 (0) | 2 (1) | 71 (36) |
| B1-LR+aug | 0 (0) | 3 (2) | 97 (62) |
| B1-DL+aug | 0 (0) | 1 (0) | 71 (31) |
| A0 | 0 (0) | 2 (1) | 73 (32) |
| B1-XGB′+aug | 0 (0) | 1 (1) | 56 (33) |
| B1-LR′+aug | 0 (0) | 0 (0) | 80 (45) |
| B1-DL′+aug | 0 (0) | 1 (1) | 41 (24) |
| A0′ | 0 (0) | 1 (1) | 40 (24) |
| B2 (circular) | 27 (4) | 507 (217) | 2,119 (1,009) |

The five seeds agree with seed 42: at 25%, A0's top-3 is 0.9976 (standard deviation 0.0002) and A0′'s is 0.9986 (0.0001) `[TEST]`. Every verdict that reads supported or met holds at 5 of 5 seeds, and every one that reads not supported or not met holds at 0 of 5, so no verdict is an accident of seed 42.

## 5.4 Safety, Red Flags and Calibration

### 5.4.1 The safety targets

The headline safety result is that the models trained on masked copies meet both safety targets at 25% evidence on the test split, and the original B1 does not (Table 5.14). This reproduces the validation finding (Table 5.6) on a split nobody tuned on. **The value is in training on masked copies, not in the fusion**: at 25% A4 (A0 without the flags) equals A1 (the transformer alone) to the fourth decimal on MRR, 0.9639 for both. Precision@3 and Recall@5 are capped by DDXPlus's open-world differentials: A0 at 100% has Precision@3 0.7430 [0.7402, 0.7459], Recall@5 on the in-scope differential 0.5632 [0.5610, 0.5654] and Recall@5 on the full differential 0.3303 [0.3284, 0.3322] `[TEST]`.

**Table 5.14.** Safety at 25% evidence (`[TEST]`; seed 42; targets: must-not-miss recall@3 at least 0.95, dangerous false-negative rate at most 0.02).

| System (25%) | Must-not-miss recall@3 | Dangerous false-negative rate | Both targets met |
|---|---|---|---|
| B1-XGB | 0.8710 [0.8659, 0.8764] | 0.0629 [0.0592, 0.0665] | No |
| B1-LR | 0.9198 [0.9156, 0.9238] | 0.0272 [0.0249, 0.0297] | No |
| A0 | 0.9981 [0.9974, 0.9987] | 0.0003 [0.0001, 0.0006] | Yes |
| A0′ | 0.9986 [0.9980, 0.9991] | 0.0002 [0.0000, 0.0004] | Yes |

### 5.4.2 The red-flag layer

Each red-flag rule follows a published pattern, and every must-not-miss condition has one. Sensitivity is below its 0.95 target at every level and collapses as the history shortens (Table 5.15). We did not loosen a rule to chase the target: DDXPlus often omits the defining finding (20% of Boerhaave patients never report the vomiting, 29% of myocarditis patients the viral illness), so a rule that fires without it would fit the generator, not medicine. On validation at 100%, three conditions limit sensitivity: pneumothorax 0.325 [0.301, 0.348], myocarditis 0.641 [0.617, 0.665] and Boerhaave syndrome 0.748 [0.730, 0.767]. At reduced evidence most rules, which need two or three findings together, cannot fire, because a 25% history keeps about 28% of the questions beyond the initial one. **So on partial histories, must-not-miss safety is carried by the ranker trained on masked copies, not by the rules.**

We measure alarm fatigue as the share of patients without a must-not-miss condition who get any flag, never as the share of all patients, because half of DDXPlus's patients have one. It is 0.246 [0.240, 0.253] at 100% on validation, mostly the infarction rule firing on stable angina, which decision A-7 counts as appropriate. The flags also have a cost: they rank a flagged condition first even when the flag names the wrong one, which lowers top-1 at full evidence (Section 5.3.1).

**Table 5.15.** The red-flag layer at three evidence levels (A0 and A0′ are identical here).

| Measure | 100% | 50% | 25% |
|---|---|---|---|
| Sensitivity, `[TEST]` | 0.8285 [0.8229, 0.8344] | 0.3318 [0.3245, 0.3386] | 0.1369 [0.1317, 0.1420] |
| Precision, strict, `[TEST]` | 0.5089 [0.5047, 0.5128] | 0.6469 [0.6379, 0.6550] | 0.8220 [0.8084, 0.8352] |
| Sensitivity, validation | 0.823 [0.818, 0.829] | 0.327 [0.320, 0.334] | 0.135 [0.130, 0.140] |
| Precision, appropriate by decision A-7, validation | 0.848 [0.845, 0.852] | 0.890 [0.884, 0.896] | 0.931 [0.922, 0.941] |
| Precision, strict, validation | 0.512 [0.508, 0.516] | 0.650 [0.640, 0.659] | 0.822 [0.808, 0.836] |
| Patients without a must-not-miss condition flagged, validation | 0.246 [0.240, 0.253] | 0.050 [0.046, 0.053] | 0.007 [0.006, 0.008] |

### 5.4.3 Calibration

A calibrated system gives a probability that matches how often it is right. We pre-registered four methods (none, temperature scaling, Platt scaling and isotonic regression [@zadrozny2002]), each fitted at 100% on validation and applied unchanged at every level, and chose by the lowest mean Brier score [@brier1950] over the three levels. The rule kept the raw scores for all four machine-learning components, whose ECE is at most 0.011 at every level. A calibrator fitted on full histories over-sharpens short ones: for B1-LR′+aug, Platt scaling raised the ECE at 25% from 0.011 to 0.098, and isotonic regression gave the true condition under 0.001 for 2.4% of patients (a count made after seeing the results, so post hoc). A displayed "0%" beside a must-not-miss condition is the failure the system exists to prevent.

For A0′'s fused score the rule chose temperature scaling (Table 5.16), and **we did not deploy it.** On hand-written cases it makes the near-certainty worse: the first condition's probability goes from 0.936 to 0.998 for GC-002 and from 0.915 to 0.997 for GC-004 (R-17). The project owner therefore decided on 2026-11-09 to keep ranks only: no calibrator ships, no probability is shown, A5 is A0 exactly, and H5 is judged on the raw score. Figure 5.1 is the reliability diagram of the configured ranker; each point is a bin of patients and the diagonal is perfect calibration.

**Table 5.16.** Top-label ECE of the raw and calibrated scores (the model's alone and A0′'s fused scores; validation unless marked `[TEST]`; target below 0.10).

| Score | ECE at 100% | ECE at 50% | ECE at 25% |
|---|---|---|---|
| B1-DL′+aug alone, raw | 0.0003 [0.0001, 0.0008] | 0.0024 [0.0016, 0.0036] | 0.0047 [0.0034, 0.0068] |
| A0′ fused, raw (this is A5) | 0.0327 [0.0324, 0.0329] | 0.0603 [0.0595, 0.0614] | 0.0773 [0.0757, 0.0793] |
| A0′ fused, temperature scaling (not deployed) | 0.0008 [0.0008, 0.0008] | 0.0051 [0.0041, 0.0062] | 0.0176 [0.0156, 0.0195] |
| A0 fused, raw, `[TEST]` | 0.0099 [0.0094, 0.0103] | 0.0355 [0.0342, 0.0368] | 0.0644 [0.0620, 0.0666] |
| A0′ fused, raw, `[TEST]` | 0.0398 [0.0394, 0.0402] | 0.0648 [0.0640, 0.0659] | 0.0785 [0.0766, 0.0806] |

![Figure 5.1. Reliability diagram of the configured ranker B1-DL′+aug, raw scores, at 100%, 50% and 25% evidence (validation; seed 42; the component alone; EXP-023). The diagonal is perfect calibration.](img/exp023_reliability_b1_dl_prime_aug.png)

One discrepancy is left unreconciled. For the same system on validation, the fusion experiment records the raw fused score's ECE as 0.040 / 0.064 / 0.078 at 100 / 50 / 25%, and the calibration experiment records the slightly lower figures in Table 5.16. The verdict against 0.10 is the same under both.

## 5.5 Evidence Retrieval and the Language-Model Trial

Two components of the explanation layer were measured, and neither measurement is a success. The evidence retrieval was built and costed, but its relevance was never judged. The language-model explainer failed its pre-registered checkpoint and does not ship. Neither affects the ranking: evidence is attached after ranking, a test checks that it never changes the order, and the dashboard shows the template explainer.

### 5.5.1 Evidence retrieval

The design follows the retrieval-augmented approach [@lewis2020] and was fixed on 2026-10-14, before any corpus existed: hybrid retrieval, in which a word-matching search (BM25) and a dense search over embeddings are fused by reciprocal rank, over open-access PubMed Central articles, with 26 queries. Two team members were to judge each retrieved passage 0 to 2, blind to the method. We lightened the load on 2026-11-09 from 489 to 159 rows a judge, and on 2026-11-10 the project owner reported that the team members had no time to judge even that, so **no filled sheet exists, and the relevance of the evidence shown beside a candidate is unjudged.** The hybrid stays configured by default, not by a result, and the dashboard's caption says the relevance has not been formally judged and asks the reader to take the passages as leads, not proof (Table 5.17).

**Table 5.17.** Evidence retrieval: what was measured and what was not.

| Item | Result |
|---|---|
| Corpus | 2,067 articles and 26,190 passages from PubMed Central Open Access (446 articles skipped); embeddings from S-PubMedBert-MS-MARCO |
| Licences | Every article carries a Creative Commons licence; 844 of 2,067 carry a non-commercial term, shown with each excerpt |
| Fidelity of the laptop's query vector (56 pairs) | The laptop runs no embedding model and mixes precomputed vectors. Cosine to the full-sentence embedding: median 0.962. Overlap of the dense top 10: **median 0.400**, under the pre-registered 0.5, so **a poor stand-in** |
| Cost on the laptop | Start-up 1.8 s; one candidate's retrieval p95 1.22 ms; the whole pipeline with evidence p95 4.9 ms against 0.9 ms without |
| Precision@3, nDCG@3, the hybrid compared with each method, Cohen's kappa | **Not computed: nothing was judged** |

Treatment language is excluded by design: a sentence with treatment language is never stored, embedded or shown. The screen is lexical, so it could miss a treatment written in words it does not list, and no judge counted any. The evidence is partly non-commercial (Appendix A).

### 5.5.2 The language-model trial

The trial (EXP-009 and EXP-010) used one local model, `llama3.1:8b`, at temperature 0, on 50 validation patients (the first 50 in the order of a hash of their identifier; 29 have a must-not-miss condition) and the four golden cases, in four arms. **B3** gives the model the case and the 14 condition names; **B3r** also tells it to use only the facts it is given; **B4** adds the best retrieved passage per condition; **A0+LLM** gives it the pipeline's result and asks for an explanation. An unsupported claim is a sentence that traces to no source the arm was given [@ji2023]. The check is lexical, so a correct paraphrase fails, and the rate measures how closely an explanation stays within its inputs, not whether it is correct. The run took 47 minutes on Kaggle, and no call failed.

**The checkpoint failed.** The rule, fixed before any model was called, was that if A0+LLM's unsupported-claim rate was not under 0.20, the template explainer stays and B3 and B4 are reported as not carried into the evaluation. The rate is 0.741 [0.663, 0.803], 163 of 220 claims (Table 5.18), so the template explainer stays, the language model is switched off, and B3 and B4 are not carried forward. **H3 is not supported**: two of its four paired comparisons pass (B3 minus A0+LLM and B3r minus A0+LLM are each +0.259 [+0.191, +0.333]), and two do not (B3 minus B4 and B3r minus B4 are each +0.004 [+0.000, +0.012], whose interval touches 0).

**Table 5.18.** Unsupported-claim rates (validation; 50 patients; 100% evidence; lexical check).

| Arm | Claims | Unsupported | Rate |
|---|---|---|---|
| B3, the model alone | 247 | 247 | 1.000 [1.000, 1.000] |
| B3r, told to use only the facts it is given | 248 | 248 | 1.000 [1.000, 1.000] |
| B4, retrieval and the model | 244 | 243 | 0.996 [0.984, 1.000] |
| A0+LLM, raw answer, before the guardrails | 220 | 163 | **0.741 [0.663, 0.803]** |

The guardrails did their job: in all 50 cases the grounded explainer's answer failed guardrails v1 (1 to 11 unsupported claims each), so the template's text was delivered, and none delivered an unsupported claim or treatment language. **The model's prose never reached a user.** Citations did not make claims supported: B4 cites 77 passages, none invented, but all 72 of its cited sentences were unsupported by the check. A language model asked to rank the 14 conditions ranked worse than the pipeline, and retrieved passages made it worse (Table 5.19). The limits are fifty patients (wide intervals), a lexical check, and one model, one run and one temperature. The 150-item rating sheet for the explanations was not rated, so no clinician-reviewed explainability exists. Nothing here ran on the test split.

**Table 5.19.** The ranking each arm produced (validation; 50 patients; 100% evidence; an interval appears only where the record gives one). A0+LLM's ranking is the pipeline's through the case-to-token translation and is not A0 of Section 5.3.

| Arm (100%) | Top-1 | Top-3 | MRR |
|---|---|---|---|
| A0+LLM | 0.80 [0.68, 0.90] | 1.00 [1.00, 1.00] | 0.88 [0.82, 0.94] |
| B3 | 0.44 | 0.78 [0.66, 0.90] | 0.62 [0.52, 0.72] |
| B3r | 0.50 [0.36, 0.64] | 0.84 [0.74, 0.94] | 0.67 [0.57, 0.76] |
| B4 | 0.20 | 0.58 [0.44, 0.72] | 0.40 [0.31, 0.50] |

## 5.6 Breakdowns by Condition, Sex and Age

**Near-equal results across sex or age are an artifact of DDXPlus's synthetic demographics, not evidence of fairness**: every condition is about half female in the data, and the generated ages follow the generator, not patients.

### 5.6.1 By condition

On the test split, A0′ at 25% ranks the true condition in the top 3 for at least 0.993 of the patients of every one of the 13 conditions (lowest: spontaneous pneumothorax, 0.9933 [0.9888, 0.9970], 1,343 patients; Figure 5.2 shows the same on validation for each component alone). Top-1 is lowest for unstable angina (Table 5.20), the confusion between unstable and stable angina that validation (EXP-023) found for every machine-learning component alone: at 25%, of 2,748 unstable-angina patients, 610 are ranked below stable angina by the transformer (0.222 [0.206, 0.238]; Figure 5.3). DDXPlus describes these two conditions with nested evidence, and what separates them clinically (rest pain, a crescendo pattern, the troponin) is what the "what to ask next" feature asks for. Among must-not-miss patients at 25%, the transformer misses 33 of 16,943 on validation (0.0019 [0.0014, 0.0027]), 19 of them at rank 4. These figures describe DDXPlus's generator on 13 conditions, not real patients.

**Table 5.20.** The lowest top-1 by true condition at 25% evidence (`[TEST]`; A0′; seed 42).

| True condition (25%) | Patients | Top-1 |
|---|---|---|
| Unstable angina | 2,880 | 0.6896 [0.6726, 0.7059] |
| Myocarditis | 1,478 | 0.8606 [0.8430, 0.8775] |
| Spontaneous pneumothorax | 1,343 | 0.8883 [0.8712, 0.9047] |
| Stable angina | 2,386 | 0.8927 [0.8793, 0.9049] |

![Figure 5.2. Top-3 recall per condition at 100%, 50% and 25% evidence for each component alone (validation; seed 42; EXP-023). Must-not-miss conditions are starred; B2 is the graph alone and is circular.](img/exp023_recall_top3_by_level.png)

![Figure 5.3. Top-1 confusions of the configured ranker B1-DL′+aug at 25% evidence (validation; seed 42; EXP-023). Each row is one true condition; the diagonal is the share ranked first.](img/exp023_confusion_b1_dl_prime_aug_25.png)

### 5.6.2 By sex

On the test split, A0′'s top-1 at 25% is 0.9375 [0.9338, 0.9410] for women (17,898 patients) and 0.9368 [0.9330, 0.9408] for men (16,759 patients). The intervals overlap, but DDXPlus's sex distribution is synthetic and near-equal for every condition, so this says nothing about fairness. The tables for seven systems at every level are in the run's output file.

### 5.6.3 By age (validation only)

The five age bands (0-17, 18-39, 40-59, 60-79 and 80+) were fixed in a dated note before any figure by age existed. We ran the breakdown on validation only, seed 42 (588 s on the laptop processor), because an age breakdown on the test split would have needed a second opening of it. A0′ keeps must-not-miss recall@3 at 0.9971 or above at 25% in all five bands (Table 5.21; the widest interval is the 80+ band's, 0.9971 [0.9927, 1.0000]), and at 1.0000 in every band at 100% and 50%. The plain models miss the target in most bands: B1-XGB is below 0.95 in all five, and B1-LR in four of five. Top-1 is highest in the youngest band at every level (A0′ at 25%: 0.9807 [0.9763, 0.9849] against 0.9260 to 0.9357 in the other bands). Why is not established: DDXPlus's ages and its condition mix by age are generated, so this is a property of the generator, not a finding about children. No verdict depends on this breakdown.

**Table 5.21.** Must-not-miss recall@3 at 25% evidence by age band (validation; seed 42; target at least 0.95).

| Band | Patients | B1-XGB | B1-LR | A0′ |
|---|---:|---|---|---|
| 0-17 | 4,093 | 0.9198 [0.9069, 0.9332] | 0.9705 [0.9621, 0.9782] | 0.9987 [0.9968, 1.0000] |
| 18-39 | 12,336 | 0.8791 [0.8712, 0.8867] | 0.9227 [0.9164, 0.9285] | 0.9978 [0.9965, 0.9989] |
| 40-59 | 10,133 | 0.8648 [0.8549, 0.8744] | 0.9205 [0.9124, 0.9277] | 0.9971 [0.9956, 0.9985] |
| 60-79 | 6,022 | 0.8747 [0.8622, 0.8860] | 0.9178 [0.9078, 0.9271] | 0.9987 [0.9974, 0.9997] |
| 80+ | 1,379 | 0.8620 [0.8355, 0.8869] | 0.9134 [0.8928, 0.9339] | 0.9971 [0.9927, 1.0000] |

## 5.7 Discussion

### 5.7.1 What the results mean

The comparative claims we pre-registered are mostly not supported, and we report them that way. Four findings carry.

1. **Training on masked copies produces the safety result.** At 25% evidence the original B1 models miss the safety targets and every model trained on masked copies meets them (Table 5.14). Neither the graph nor the flags caused this.
2. **The full system is not better than its own machine-learning component.** At 100% every machine-learning comparator is already perfect, and at 50% those trained on masked copies are perfect or within three patients (R-16), the graph's contribution is not measurable on DDXPlus, and the red-flag layer lowers top-1 at full evidence (0.8346 against 0.9987 `[TEST]`) while raising nothing in recall. The graph's argument is one DDXPlus cannot show, an explanation and a condition the model has never seen (aortic dissection, GC-003), and we state it as a design argument shown on one hand-written case, not as a measured gain.
3. **A small transformer is better than logistic regression trained the same way and level with XGBoost,** the expected result where the headroom is this small.
4. **The red-flag layer is below its own target and weakens as histories shorten,** and we show ranks, not probabilities, because a short hand-written case is near-certain under any calibration.

### 5.7.2 Threats to validity

- **Synthetic data.** Every figure describes DDXPlus's generator on 13 conditions. The test split is a second sample from the same generator: it shows that nothing was fitted to validation, not that the system works on real patients.
- **Circularity (R-12).** The graph is built from the definitions that generated the data, so any figure for the graph, or for a system that uses it, may flatter it. The graph's independent parts (the hand-authored dissection edges, the red-flag rules and the BODHI-S enrichment) are tested only on four hand-written cases.
- **Saturation (R-16).** On complete histories every machine-learning model is perfect, so the comparisons at 100% could not be won. A perfect score written [1.000, 1.000] means that no error was observed; on validation the exact lower bounds are 0.99989 for top-3 and 0.99978 for must-not-miss recall@3.
- **Closed world (R-13).** A patient with pneumonia is forced into one of the 14 conditions; the system cannot say "none of these".
- **Random masks.** They measure robustness to missing answers, not history-taking.
- **The case-to-token translation has no ground truth (R-17).** A hand-written case reaches the model through 17 hand-chosen representative answers. One was wrong (A-10) and was found when the model's reasons were examined; the other 16 are unchecked.
- **The choice of reading.** Judging the reduced-evidence claims at 25% was chosen after validation figures were known, and the other team members' agreement is not recorded. The older reading is also computed.
- **Seeds and intervals.** The seed-42 model is the one used, fixed beforehand, and every verdict holds at 5 of 5 or 0 of 5 seeds. Intervals come from 1,000 bootstrap resamples, and the fusion weight was chosen on validation, so fused validation figures are optimistic and the test figures are the honest ones.
- **What was not assessed.** Retrieval relevance, the template explainer's unsupported-claim rate and clinician-reviewed explainability. No clinician reviewed anything, no real patient was seen, and latency was measured on one laptop.

### 5.7.3 What the examiners should take away

1. We built a complete, working decision-support prototype and tested it with a pre-registered protocol and a test split opened once.
2. The honest result is mostly negative on the comparative claims: H1, H2, H1-R, H2-R, H3 and H6 are not supported, and H4's support is hollow.
3. The result that carries is practical: a small transformer trained on masked histories keeps must-not-miss recall@3 above 0.99 when a history is cut to a quarter, where the same models trained on complete histories do not.
4. The guardrails worked as designed in the language-model trial: they stopped every unsupported explanation before it reached a user.
5. Nothing here is clinical validation. The output is decision support only, never a diagnosis.

# Conclusion and Future Scope

## 6.1 Conclusion

### 6.1.1 What we built

Nightingale is a decision-support prototype for acute chest pain. It ranks 14 conditions (13 learned from DDXPlus, and aortic dissection, written into the knowledge graph) by combining a small transformer with a cardiac knowledge graph built from three sources. Independent rules, written after published clinical patterns, raise the must-not-miss conditions. The system explains each ranking, names the next questions and the investigations to consider, re-ranks as findings are entered, reports every component that degraded, and never recommends a treatment. It also holds evidence retrieval over 2,067 open-access articles, a display-only panel of 1,013 synthetic patient histories, and a language-model explainer that we built, tested and did not ship. The disclaimer appears on every output.

### 6.1.2 Answers to the research questions

Each verdict is the pre-registered rule's, applied once to the test split, and the same as on validation, row for row. Failures come first.

- **RQ1, does fusing graph reasoning with a machine-learning ranker improve accuracy over either alone? No (H1 and H1-R not supported; 0 of 5 seeds).** At 100% every machine-learning comparator ranks the true condition in the top 3 for all 34,657 patients `[TEST]`, and A0's MRR is lower than the model's (−0.0837 [−0.0859, −0.0819]) because the flags rank first.
- **RQ2, does the system improve safety over machine learning alone? No (H2 and H2-R not supported; 0 of 5 seeds).** The safety result is real, but it belongs to training on masked copies, not to the fusion.
- **RQ3, does evidence grounding reduce unsupported claims? Not supported (H3).** The grounded explainer's raw answers hold an unsupported-claim rate of 0.741 [0.663, 0.803] against a bar of under 0.20, so the template explainer ships (validation, with a lexical check).
- **RQ6, does a deep ranker hold up better on incomplete histories? Not supported (H6; 0 of 5 seeds).** It beats the logistic regression trained like it and is level with XGBoost.
- **RQ4, which components contribute? H4 is supported by the rule, and the support is hollow.** Removing the graph changes must-not-miss recall by exactly 0.0000; the contrast comes from the flags lowering top-1 by 0.1642 [0.1606, 0.1685].
- **RQ5, are the confidence scores calibrated? Yes, on the raw scores (H5 supported, 5 of 5 seeds):** A0's ECE at 100% is 0.0099 [0.0094, 0.0103] and A0′'s 0.0398 [0.0394, 0.0402] `[TEST]`. No probability is shown, only ranks.

Among the success criteria, the Minimum is met. The Target and Stretch at 100% are not met, because the machine-learning comparators tie at 1.000. The reduced-evidence Target is met by A0′ only, on a handful of patients (it misses 40 of 34,657 in the top 3 where XGBoost trained like it misses 56), and the reduced-evidence Stretch is not met. The unsupported-claim rate of the shipped explainer and clinician-reviewed explainability were not assessed.

### 6.1.3 What we found along the way

On DDXPlus, which questions were answered nearly names the condition, so full histories cannot separate systems. An encoding that cannot tell a denied question from an unasked one made a strong model answer atrial fibrillation to an empty history, and masked training with an explicit "asked" channel repaired it. On partial histories, must-not-miss safety is carried by the ranker, not by the rules: red-flag sensitivity falls from 0.8285 [0.8229, 0.8344] at 100% to 0.1369 [0.1317, 0.1420] at 25% `[TEST]`, while B1-DL′+aug holds must-not-miss recall@3 at 0.9986 [0.9980, 0.9991] `[TEST]`. A clinically sensible translation of a hand-written finding was wrong for the data, and the model's reasons found it.

### 6.1.4 Closing statement

On synthetic chest-pain data, a small transformer trained on masked copies of the histories keeps must-not-miss recall@3 above 0.99 when a history is cut to a quarter, and the graph and the rules around it add explanations and a safety net, not measurable accuracy. Nightingale is a research prototype with no clinical validation, no regulatory review and no approval by any authority. It provides decision support only, never a diagnosis, and a qualified clinician makes every decision.

## 6.2 Future Scope

These are concrete next steps, in the order we would take them. None was done in this project.

- **An independent test set of clinical vignettes,** which would escape the graph's circularity (R-12) at the price of wide intervals.
- **An abstention signal for the closed world (R-13):** a detector trained on DDXPlus's other pathologies, so that the system can say a presentation is outside its 14 conditions.
- **Checking the other 16 representative answers** against which conditions give each answer in DDXPlus (R-17), the way the jaw answer was checked.
- **Relevance judging of the retrieved passages.** The judging tool and its 159-row pool exist; two blind judges would make the evidence's Precision@3 and nDCG@3 measurable.
- **Clinician review** of the explanations and the golden cases, including our open question about how aortic dissection should rank for borderline presentations.
- **Red-flag rules that survive short histories.** Most rules need two or three findings together, which a 25% history rarely holds; whether rules can be designed for partial histories without fitting the generator is open.
- **Interactive history-taking:** an agent loop that asks the top "what to ask next" question and re-ranks, with a question ranking that weights must-not-miss conditions.
- **A new language-model trial** under a new pre-registered design, only if a reason arises; none is planned.
- **Real, de-identified data** under the proper approvals, and the stretch goals not pursued (a heart-risk sub-model, knowledge-graph embeddings and an ECG extension).

## Appendix A. Data Sources, Licences and Attribution

No real or identifiable patient data enters the project: everything is synthetic or openly licensed, and the `data/` folder is never committed. Table 6.1 lists every source we used, with its licence as re-verified on 2026-11-10 from public metadata.

**Table 6.1.** Data sources, licences and status.

| Source | Used for | Licence | Status |
|---|---|---|---|
| DDXPlus [@tchango2022; @ddxplus_card] | Training and test data, the ground truth, the graph's backbone | CC-BY-4.0 | Verified; the revision used is the card's current one |
| BODHI-S [@bodhis_card] (Eka Care) | Enrichment of four conditions in the graph | CC-BY-NC-4.0, non-commercial | Verified; no BODHI-S text is committed |
| Synthea [@walonoski2018] | Synthetic patient histories, display only | Apache-2.0 | Verified |
| PubMed Central Open Access | The evidence corpus (2,067 articles) | Per article, from its own XML: 15 Creative Commons licence strings; 844 articles non-commercial | From the corpus manifest; each excerpt shows its licence |
| S-PubMedBert-MS-MARCO (`pritamdeka`) | Passage vectors, computed once on Kaggle | CC-BY-NC-2.0, non-commercial | Verified; not in the register of 2026-09-17; the retrieval bundle inherits the restriction |
| Llama 3.1 8B (`llama3.1:8b`) | The language-model trial only; not shipped | Meta's Llama 3.1 Community Licence | **Not re-verified: the licence page is gated** |
| UCI Heart Disease, PTB-XL | Never used | n/a | n/a |

**The project is non-commercial in three independent ways**: BODHI-S, the embedding model, and 844 of the corpus's 2,067 articles allow non-commercial use only. Commercial use would need BODHI-S replaced (Hetionet, released under CC0 [@himmelstein2017], is the candidate), the embedding model replaced and the non-commercial articles removed. The graph we build from BODHI-S inherits its restriction and stays private; the corpus and the embeddings are never committed.

Attribution statement: this project uses DDXPlus (Tchango et al., NeurIPS 2022, CC-BY-4.0); BODHI-S by Eka Care (CC-BY-NC-4.0, used for non-commercial academic research); Synthea (MITRE, Apache-2.0); the S-PubMedBert-MS-MARCO embedding model (`pritamdeka` on Hugging Face, CC-BY-NC-2.0); for the language-model trial only, Llama 3.1 (Meta); and literature retrieved from PubMed and PubMed Central (NCBI), each excerpt under its own Creative Commons licence.

## Appendix B. Golden Clinical Cases

The four golden cases are executable clinical expectations: hand-written, synthetic, and not drawn from any patient record. They run in the test suite and must stay passing; if one fails, the component is fixed, not the expectation, and no expectation has been changed. They are the evidence for the graph that is not circular, but they reach the machine-learning model through a hand-curated translation that has no ground truth (R-17), and a short hand-written history is near-certain under the model. All four pass with every model, with the red flags on and off.

**Table 6.2.** The golden cases and how the system behaves on each.

| Case | Presentation (hand-written) | Expectation | Behaviour |
|---|---|---|---|
| GC-001 | A 58-year-old man with exertional, pressure-type central chest pain radiating to the jaw and left arm, with sweating and breathlessness; diabetic smoker | Possible myocardial infarction (NSTEMI or STEMI, without or with ST elevation on the ECG) in the top 3, with an ischaemic red flag | The graph alone ranks infarction first. The model alone answers unstable angina, because a history cannot separate it from infarction (only a troponin can). Fused with flags off, infarction is second; with flags on, the red flag lists it first |
| GC-002 | A 42-year-old woman with sudden pleuritic chest pain, breathlessness and a swollen leg after recent immobilisation | Pulmonary embolism in the top 3, with its red flag | Embolism is first by the graph and by the model; the embolism flag fires, with a pneumothorax flag beside it |
| GC-003 | A 64-year-old man with sudden tearing chest pain radiating to the back, a blood-pressure difference between the arms, and hypertension | Aortic dissection in the top 5, with its red flag | No model can score dissection (DDXPlus has none), and the transformer alone answers pneumothorax, because DDXPlus describes pneumothorax pain as tearing. The graph ranks dissection first, the fusion keeps it first with flags off, and the dissection flag fires. This case justifies the graph |
| GC-004 | A 35-year-old woman with burning pain after meals, worse lying flat, with exertional onset, jaw or arm radiation and sweating all denied | Gastro-oesophageal reflux disease (GERD) in the top 3, with no red flag | Reflux is first with no flag. It guards against over-flagging: a benign presentation must not raise a cardiac flag |

Each case is loadable in the dashboard as an example, where its findings can be removed, added or denied and the ranking updates at once.

## Appendix C. Reproducibility

The code, the configuration and the records are in the project repository. The data and the trained models are not: `data/` and `models/` are never committed, and the data are rebuilt from their public sources by scripts. The test split was opened once, and it must not be scored again.

**Table 6.3.** Tags and commits that fix the results.

| Tag or commit | Date | What it fixes |
|---|---|---|
| `0f47d67` | 2026-09-25 | Amendment 3 applied; the validation mask digests recorded before any system was scored on them |
| `v0.1-prototype` (`06b6263`) | 2026-10-14 | The first prototype: the dashboard with the transformer, the graph, the red flags and the model's reasons |
| `46f76eb` | 2026-11-07 (the run date) | The code the validation rehearsal ran on |
| `v0.2-prototype` (`172c421`) | 2026-11-08 | The evidence corpus wired in; the language-model trial scored |
| `v0.9-freeze` (`808bceb`) | 2026-11-09 | `configs/`, `src/` (outside `src/eval/`) and `tests/fixtures/` frozen before the test split |
| `f644faa` | 2026-11-09 | The test-scoring notebook and the test mask digests; the single opening ran on this commit |
| `04b5eb7` | 2026-11-10 | The test results recorded in the experiment log |
| `cc70ce9` | 2026-11-11 | The age breakdown's code and bands, fixed before any figure by age existed |

**Table 6.4.** Digests, fingerprints and the opening record.

| Item | Value |
|---|---|
| Validation mask digest; asked-set digest | `e99a7a8fbf792675`; `af3a357f6454838a` |
| Test mask digest; asked-set digest | `0136f67f1ca6542f`; `795acf2689f1415e` |
| Knowledge graph fingerprint | `de349201157a` |
| Encoder fingerprint, "asked" channel off; on (691 columns) | `3a0d5a5e01d7f427`; `ddd14019c66eff10` |
| Language-model trial sample (digest of the 50 identifiers) | `022bdcfa4641cb81` |
| The single opening of the test split | 2026-11-09T08:07:22Z, at `f644faa`, with no reopening reason; result file hash begins `f4100057ef462b94` |

**Table 6.5.** Software versions.

| Layer | Versions |
|---|---|
| Language and data | Python 3.11.9; NumPy 1.26.4 drew the evaluation masks; pandas; PyArrow |
| Machine learning | scikit-learn 1.9.1; XGBoost 2.1.4; PyTorch for training only, on Kaggle (a separate environment with torch 2.11.0+cu128 for short tests); the laptop runs the transformer as a NumPy forward pass |
| Graph | Neo4j AuraDB Free (Neo4j 5.27) with driver 5.28; NetworkX 3.6 as the fallback |
| Service and interface | FastAPI 0.143; uvicorn 0.54; Streamlit 1.65; Pydantic v2 |
| Cloud runs | Kaggle, Python 3.13.18, NumPy 2.1.3, pandas 2.3.3, XGBoost 2.1.4; Synthea v4.0.0; Ollama 0.40.15 with `llama3.1:8b` (Q4_K_M) |

Three checks support the results. The validation rehearsal reproduced 1,944 of 1,944 earlier recorded figures exactly. The laptop reproduces the cloud-trained B0 and B1 validation metrics to within 3 × 10⁻¹², and the transformer's seed-42 validation metrics to within 7 × 10⁻¹¹. The test run's scoring took 2,797 s. The test suite has 1,494 tests passing and 15 skipped (they need PyTorch or a live graph database), including the four golden cases; the code is formatted and linted with pinned tool versions, and continuous integration was green at the last check.

## Appendix D. How to Run the System

These commands run on Windows from the repository root with the project's Python 3.11 environment. Always call the environment's interpreter, because the machine's default Python is a different version. The data and the trained models must be in place; without the data, the demo runs on a small stub graph.

Run the tests, including the four golden cases:

```
./.venv/Scripts/python.exe -m pytest
```

Start the dashboard (it serves this machine only, at 127.0.0.1:8501, and re-ranks as each finding changes), the API (127.0.0.1:8000/docs), or the command-line demo for one golden case:

```
./.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
./.venv/Scripts/python.exe scripts/serve_api.py
./.venv/Scripts/python.exe scripts/demo.py --case GC-003
```

Use the stub graph when no data are present, and measure the model's cost and the pipeline's latency (about 15 s):

```
./.venv/Scripts/python.exe scripts/demo.py --case GC-003 --graph stub
./.venv/Scripts/python.exe scripts/measure_latency.py
```

Rebuild the validation parquet and the knowledge graph, check the crosswalk and the golden cases on the real data, and print the mask digests (a digest run reads no label and records no opening):

```
./.venv/Scripts/python.exe scripts/build_ddxplus_chestpain.py
./.venv/Scripts/python.exe scripts/build_cardiac_kg.py
./.venv/Scripts/python.exe scripts/check_crosswalk.py
./.venv/Scripts/python.exe scripts/evaluate_ablations.py --digest-only
```

Training and full scoring runs are long and ran on Kaggle, not on the laptop. The cloud graph database pauses after 72 hours unused; when it is paused, the system falls back to the local graph and reports `graph_backend` as degraded, with an identical ranking. Every screen and output carries the disclaimer: Clinical decision support — not a diagnosis. Nightingale is a research prototype. Its output is generated from synthetic training data and must not be used for real patient care. Rankings are not certainties and confidence values are estimates. A qualified clinician is responsible for all diagnostic and treatment decisions.

```abbreviations
A0 = The full system with B1-DL+aug as its machine-learning component (A0′: with B1-DL′+aug)
A1, A1b, A2, A3, A4, A5, A6, A7 = Ablations of the full system (Table 5.2)
A-7, A-10 = Decisions A-7 (which red flags are clinically appropriate) and A-10 (the answer used for radiation to the jaw or arm)
API = Application Programming Interface
B0, B1, B2, B3, B4 = Baselines (prevalence, machine learning, graph alone, language model alone, retrieval plus language model)
B1-DL = The small transformer used as the deep ranker
B1-LR = Logistic-regression form of B1
B1-XGB = XGBoost form of B1
BM25 = Best Matching 25 (a word-matching search function)
CC-BY = Creative Commons Attribution licence
CC-BY-NC = Creative Commons Attribution-NonCommercial licence
D-5, D-13 = Decisions D-5 (the language-model checkpoint) and D-13 (the transformer's safety floor)
DDXPlus = A synthetic differential-diagnosis dataset
ECE = Expected Calibration Error
ECG = Electrocardiogram
EXP = Experiment (a numbered entry of the experiment log)
GC = Golden Case
GERD = Gastro-Oesophageal Reflux Disease
H1 to H6 = Hypotheses 1 to 6 (H1-R, H2-R, H6: judged at 25% evidence)
LLM = Large Language Model
MRR = Mean Reciprocal Rank
NCBI = National Center for Biotechnology Information
nDCG = Normalised Discounted Cumulative Gain
NSTEMI = Non-ST-Elevation Myocardial Infarction
p95 = 95th percentile (the time within which 95% of rankings finish)
R-12, R-13, R-16, R-17, R-18 = Risks in the risk register (circularity, closed world, saturation, case-to-token translation, not-asked read as no)
RQ = Research Question
STEMI = ST-Elevation Myocardial Infarction
```
