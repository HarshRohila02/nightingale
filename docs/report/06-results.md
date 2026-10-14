# 6 · Results (validation only)

> **⚠ VALIDATION RESULTS ONLY — NOT THE STUDY'S VERDICTS.** Every figure in this chapter comes from
> the DDXPlus **validation** split (33,963 patients, 16,943 of them with a must-not-miss condition)
> or from the four hand-written golden cases. The test split has never been opened; it is read once,
> in Phase 4, and only that reading decides the hypotheses [pending: EXP-012, test split].
>
> **Read every figure with four caveats** (`docs/05` §8):
> 1. DDXPlus patients are **synthetic**; performance on real patients is unknown.
> 2. The system is **closed-world** (R-13): it ranks 14 conditions only.
> 3. Every figure for the graph alone (B2) on DDXPlus patients is **circular** (R-12, §6.5).
> 4. Reduced-evidence results measure robustness to answers **removed at random**; they are not a
>    simulation of history-taking.
>
> **Conventions.** Every figure names its **evidence level** (100%, 50% or 25% of each history,
> `docs/05` §3.7) and its **model label** (`docs/05` §4; `+aug`: trained with masked copies; ′: the
> "asked" channel). Brackets are 95% bootstrap intervals (1,000 resamples, seed 42). A figure marked
> **†** has no interval recorded in `docs/08`; one must be added before submission (`docs/05` §8 rule
> 1). Every figure is copied from the experiment log entry cited beside it. The models are the
> seed-42 models unless a seed is named.

## 6.1 What this chapter contains, and what is still to come

| Section | Experiment(s) | Status |
|---|---|---|
| 6.2 B0 and B1 on full histories; saturation (R-16) | EXP-003, EXP-004 | Done |
| 6.3 B1 on partial histories: the diagnostic (R-18) | EXP-017 | Done |
| 6.4 Masked training copies and the "asked" channel, at three levels | EXP-018, EXP-019 | Done |
| 6.5 The graph alone (circular) | EXP-005, EXP-015, EXP-016, EXP-019 | Done |
| 6.6 The red-flag rules, and their collapse at reduced evidence | EXP-008, EXP-019 | Done |
| 6.7 Seeds 43–46 of the classical models | EXP-020 | Done |
| 6.8 The transformer B1-DL and the safety floor D-13 | EXP-021 | Done: ~~seed 42 done; seeds 43–46 and the full comparison table pending~~ every seed and the full comparison table (2026-10-14) |
| 6.9 Calibration | EXP-007 | Stage 1 and the transformer arms done; stage 2 pending |
| 6.10 Complexity and latency | EXP-022 | Done |
| 6.11 The golden cases | EXP-005, EXP-006, EXP-018, EXP-021 | Done |
| 6.12 Fusion | EXP-006 | [pending: EXP-006 fusion sweep] |
| 6.13 Retrieval and the language model | EXP-009, EXP-010, EXP-011 | [pending: retrieval EXP-011] [pending: LLM baselines B3/B4, EXP-009/010] |
| 6.14 Where each component goes wrong, by condition *(added 2026-10-14)* | EXP-023 | Done |
| ~~6.14~~ 6.15 The test split: ablations and hypothesis verdicts | EXP-012 | [pending: EXP-012, test split] |

## 6.2 B0 and B1 on full histories, and saturation (R-16)

**Table 6.1 — B0 and B1 at 100% evidence** (EXP-003, EXP-004; validation; B1-XGB and B1-LR as
trained in EXP-004, feature fingerprint `3a0d5a5e01d7f427`). "Best possible" is the ceiling the
labels allow (EXP-013).

| Metric (100%) | B0 | B1-LR | B1-XGB | Best possible |
|---|---|---|---|---|
| Top-1 accuracy | 0.1097 [0.1065, 0.1132] | 0.9983 [0.9978, 0.9987] | 0.9985 [0.9980, 0.9989] | 1 |
| Top-3 accuracy | 0.3059 [0.3013, 0.3108] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1 |
| Top-5 accuracy | 0.4818 [0.4765, 0.4869] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1 |
| MRR | 0.2924 [0.2895, 0.2954] | 0.9991 [0.9989, 0.9994] | 0.9992 [0.9990, 0.9994] | 1 |
| Precision@3 (`D_in`) | 0.5638 [0.5604, 0.5674] | 0.7291 [0.7265, 0.7317] | 0.7639 [0.7613, 0.7667] | 0.929 |
| Recall@5 (`D_in`) | 0.4602 [0.4580, 0.4623] | 0.5533 [0.5508, 0.5556] | 0.5694 [0.5668, 0.5718] | 0.753 |
| Recall@5, full `D` | 0.2679 [0.2663, 0.2696] | 0.3195 [0.3176, 0.3213] | 0.3338 [0.3318, 0.3360] | 0.434 |
| Must-not-miss recall@3 | 0.2199 [0.2137, 0.2264] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1 |
| Dangerous false-negative rate | 0.6064 [0.5989, 0.6136] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0 |
| Brier score | 0.9183 [0.9179, 0.9187] | 0.0034 [0.0026, 0.0043] | 0.0030 [0.0022, 0.0039] | 0 |
| ECE, raw scores | 0.0023 † | 0.0007 † | 0.0001 † | 0 |
| Macro F1 | 0.0152 † | 0.9982 † | 0.9985 † | 1 |

**Findings.**

1. **B0 is the floor.** It ranks pulmonary embolism, GERD and panic attack first for everyone, so its
   must-not-miss recall@3 is simply pulmonary embolism's share of the must-not-miss patients
   (EXP-003).
2. **B1 saturates full-evidence DDXPlus, and the cause is the data, not skill or leakage** (EXP-004,
   R-16). No label column is read, the test split was never touched, and the laptop reproduces every
   metric of the cloud-trained models to within 3 × 10⁻¹². Every validation patient's positive
   answers lie inside their own condition's evidence set, and for 91.7% inside no other condition's,
   so which questions are answered "yes" nearly names the condition (§3.2.4). A percentile bootstrap
   cannot show uncertainty around a perfect score; the exact (Clopper–Pearson) 95% lower bounds are
   0.99989 for top-3 and 0.99978 for must-not-miss recall@3 (EXP-004).
3. **So H1, H2 and the Target cannot be supported at 100%.** Top-3 and must-not-miss recall@3 are
   already 1.000, and MRR leaves 0.0008; McNemar's test between A0 and B1 on top-3 can show only a
   tie or a loss (EXP-004, R-16). This is a property of the test, not evidence about fusion either
   way. It is why amendment 3 added the reduced-evidence levels (§5.4).
4. **What still separates systems at 100% is agreement with DDXPlus's differential.** B1-XGB
   reaches Precision@3 0.7639 of a possible 0.929 and Recall@5 0.5694 of 0.753, ahead of B1-LR with
   intervals that do not overlap. Any graph gain on these metrics carries the R-12 caveat, since `D`
   and the graph both come from DDXPlus (EXP-004).
5. **B1's only errors are the angina overlap.** B1-XGB ranks the wrong condition first for 51
   patients (50 with unstable angina and 1 with MI, each placed below stable angina); B1-LR for 59
   (56 unstable angina and 2 MI below stable angina, 1 stable angina below pericarditis). Every one
   keeps its true condition in the top 3. That is the dangerous direction, and the graph's blind spot
   too (§4.5). On top-1, B1-XGB beats B1-LR on 8 patients to 0 (exact McNemar p = 0.008) (EXP-004).
6. **Class balance holds on the real training split** (EXP-003): the rarest condition has 10,162
   training patients, and B1-XGB's per-condition F1 is 1.000 for 10 of the 13 conditions (stable
   angina 0.989, unstable angina 0.991, MI 0.9998) (EXP-004) †.

## 6.3 B1 on partial histories: the diagnostic that found R-18 (EXP-017)

EXP-017 scored the two B1 models of EXP-004, unchanged, on validation patients who keep their
initial evidence plus a share of the rest. **It used a token-level mask, not the protocol's
question-level mask** (§5.4), so it is a diagnostic; the protocol's figures are in §6.4.

**Table 6.2 — B1 under EXP-017's token mask** (EXP-017; validation; B1-XGB and B1-LR of EXP-004).

| Evidence kept (token mask) | Mean tokens | Model | Top-1 | Top-3 | Must-not-miss recall@3 | Most-predicted condition |
|---|---|---|---|---|---|---|
| 100% | 21.9 | B1-XGB | 0.9985 [0.9980, 0.9989] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | pulmonary embolism, 11% |
| 100% | 21.9 | B1-LR | 0.9983 [0.9978, 0.9987] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | pulmonary embolism, 11% |
| 50% | 11.5 | B1-XGB | 0.5972 [0.5916, 0.6022] | 0.9972 [0.9967, 0.9978] | 0.9969 [0.9960, 0.9977] | atrial fibrillation, 46% |
| 50% | 11.5 | B1-LR | 0.9755 [0.9740, 0.9772] | 0.9995 [0.9992, 0.9997] | 0.9990 [0.9985, 0.9995] | pulmonary embolism, 11% |
| 25% | 6.2 | B1-XGB | 0.2671 [0.2625, 0.2718] | 0.9463 [0.9438, 0.9488] | 0.9346 [0.9305, 0.9383] | atrial fibrillation, 76% |
| 25% | 6.2 | B1-LR | 0.8558 [0.8522, 0.8597] | 0.9818 [0.9804, 0.9832] | 0.9736 [0.9712, 0.9760] | pulmonary embolism, 10% |

**Findings.**

1. **The two models are indistinguishable where the original protocol looks, and 0.59 top-1 apart
   at 25% of the history** (0.38 at 50%). *(EXP-017 first said 0.73; that was an arithmetic slip,
   corrected on 2026-09-24: 0.8558 − 0.2671 = 0.589.)* At 100% they differ on 8 patients of 33,963;
   at 50% on 13,279, of which B1-LR is right on 13,064; at 25% on 21,124, of which B1-LR is right on
   20,560 (McNemar on top-1, EXP-017).
2. **Given no evidence at all, B1-XGB answers atrial fibrillation with probability 1.000**; B1-LR
   answers atrial fibrillation at 0.43 (EXP-017) †.
3. **The cause is the feature encoding.** A default "no" answer got no column, so a question
   answered "no" and a question never asked were the same row, and a short history read as a long
   list of denials. Atrial fibrillation is the condition whose DDXPlus patients answer the fewest
   questions (a median of 9 positive codes, against 17–19 for MI and unstable angina), so within
   DDXPlus "few answers" predicts it almost perfectly. XGBoost compounds this by sending a sparse
   matrix's absent entries down each split's default branch (EXP-017). This is **risk R-18**.
4. **It is a safety finding.** At 25% under this mask, B1-XGB's must-not-miss recall@3,
   0.9346 [0.9305, 0.9383], is below the 0.95 target with its whole interval (EXP-017). Every
   hand-written case is a short history, so this is the regime the prototype is used in.
5. **Consequences.** The pipeline's ranker became logistic regression (2026-09-23), the encoder
   gained the "asked" channel (§4.6), and the models were retrained on masked copies (§6.4).

## 6.4 Masked training copies and the "asked" channel (EXP-018, EXP-019)

### 6.4.1 At full evidence (EXP-018)

Retraining with masked copies (`+aug`), with and without the "asked" channel (′), costs nothing on
top-1 at full evidence (top-3 and must-not-miss recall@3 stay 1.000 for every model).

**Table 6.3 — The retrained B1 at 100% evidence** (EXP-018; validation; seed 42).

| Model (100%) | Top-1 | Precision@3 | Recall@5 |
|---|---|---|---|
| B1-LR (EXP-004) | 0.9983 [0.9978, 0.9987] | 0.7291 [0.7265, 0.7317] | 0.5533 [0.5508, 0.5556] |
| B1-LR+aug | 0.9982 [0.9977, 0.9986] | 0.7411 [0.7383, 0.7439] | 0.5555 [0.5531, 0.5577] |
| B1-LR′+aug | 0.9983 [0.9978, 0.9987] | 0.7163 [0.7138, 0.7191] | 0.5444 [0.5421, 0.5465] |
| B1-XGB (EXP-004) | 0.9985 [0.9980, 0.9989] | 0.7639 [0.7613, 0.7667] | 0.5694 [0.5668, 0.5718] |
| B1-XGB+aug | 0.9984 [0.9980, 0.9988] | 0.7340 [0.7313, 0.7369] | 0.5520 [0.5498, 0.5541] |
| B1-XGB′+aug | 0.9985 [0.9980, 0.9989] | 0.7441 [0.7413, 0.7470] | 0.5679 [0.5657, 0.5701] |

**The no-findings probe** (a patient of age 50 with no findings; EXP-018) †: B1-XGB answers atrial
fibrillation at 1.000; B1-LR+aug and B1-XGB+aug answer PSVT at 0.43 and 0.28; B1-LR′+aug answers
PSVT at 0.23 and B1-XGB′+aug pericarditis at 0.22. The masked copies alone remove the confident
atrial-fibrillation answer: without the channel, a short row is no longer a rare row. **This was
later narrowed** (EXP-020, §6.7): at seeds 43–46, several `+aug` models again put atrial
fibrillation first for a patient with no findings, at 0.18–0.45 rather than 1.000. The retraining
removed the certainty, not the answer, which may be a fair prior, since atrial-fibrillation
patients answer the fewest questions.

### 6.4.2 At three evidence levels, on the protocol's masks (EXP-019)

EXP-019 scored every system at 100%, 50% and 25% on the masks of §5.4, whose digests were committed
before any system was scored on them (`0f47d67`). Nothing was trained. The 100% column reproduces
EXP-003, EXP-004, EXP-005 and EXP-018 exactly. **B2 is circular (R-12).**

**Table 6.4 — Top-1 accuracy** (EXP-019; validation).

| Model | 100% | 50% | 25% |
|---|---|---|---|
| B0 | 0.110 [0.106, 0.113] | 0.110 [0.106, 0.113] | 0.110 [0.106, 0.113] |
| B1-LR | 0.998 [0.998, 0.999] | 0.947 [0.944, 0.949] | 0.766 [0.761, 0.770] |
| B1-XGB | 0.998 [0.998, 0.999] | 0.578 [0.573, 0.583] | 0.245 [0.240, 0.250] |
| B1-LR+aug | 0.998 [0.998, 0.999] | 0.982 [0.981, 0.984] | 0.931 [0.928, 0.934] |
| B1-XGB+aug | 0.998 [0.998, 0.999] | 0.983 [0.982, 0.985] | 0.933 [0.931, 0.936] |
| B1-LR′+aug | 0.998 [0.998, 0.999] | 0.986 [0.984, 0.987] | 0.946 [0.944, 0.948] |
| B1-XGB′+aug | 0.998 [0.998, 0.999] | 0.987 [0.985, 0.988] | 0.948 [0.946, 0.950] |
| B2 *(circular)* | 0.923 [0.921, 0.926] | 0.867 [0.863, 0.870] | 0.769 [0.764, 0.773] |

**Table 6.5 — Top-3 accuracy** (EXP-019; validation).

| Model | 100% | 50% | 25% |
|---|---|---|---|
| B0 | 0.306 [0.301, 0.311] | 0.306 [0.301, 0.311] | 0.306 [0.301, 0.311] |
| B1-LR | 1.000 [1.000, 1.000] | 0.997 [0.996, 0.997] | 0.945 [0.943, 0.947] |
| B1-XGB | 1.000 [1.000, 1.000] | 0.991 [0.990, 0.992] | 0.906 [0.903, 0.909] |
| B1-LR+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.996, 0.997] |
| B1-XGB+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.997, 0.998] |
| B1-LR′+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.998 [0.997, 0.998] |
| B1-XGB′+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.998 [0.998, 0.999] |
| B2 *(circular)* | 1.000 [0.999, 1.000] | 0.985 [0.984, 0.986] | 0.937 [0.935, 0.940] |

**Table 6.6 — MRR** (EXP-019; validation).

| Model | 100% | 50% | 25% |
|---|---|---|---|
| B0 | 0.292 [0.289, 0.295] | 0.292 [0.289, 0.295] | 0.292 [0.289, 0.295] |
| B1-LR | 0.999 [0.999, 0.999] | 0.971 [0.970, 0.972] | 0.858 [0.855, 0.861] |
| B1-XGB | 0.999 [0.999, 0.999] | 0.782 [0.779, 0.784] | 0.578 [0.575, 0.581] |
| B1-LR+aug | 0.999 [0.999, 0.999] | 0.991 [0.990, 0.992] | 0.963 [0.961, 0.964] |
| B1-XGB+aug | 0.999 [0.999, 0.999] | 0.992 [0.991, 0.992] | 0.964 [0.963, 0.966] |
| B1-LR′+aug | 0.999 [0.999, 0.999] | 0.993 [0.992, 0.993] | 0.971 [0.970, 0.972] |
| B1-XGB′+aug | 0.999 [0.999, 0.999] | 0.993 [0.993, 0.994] | 0.973 [0.971, 0.974] |
| B2 *(circular)* | 0.961 [0.960, 0.963] | 0.924 [0.922, 0.926] | 0.854 [0.851, 0.857] |

**Table 6.7 — Must-not-miss recall@3** (EXP-019; validation; 16,943 patients; target ≥ 0.95).

| Model | 100% | 50% | 25% |
|---|---|---|---|
| B0 | 0.220 [0.214, 0.226] | 0.220 [0.214, 0.226] | 0.220 [0.214, 0.226] |
| B1-LR | 1.000 [1.000, 1.000] | 0.994 [0.993, 0.995] | **0.925** [0.921, 0.929] |
| B1-XGB | 1.000 [1.000, 1.000] | 0.987 [0.985, 0.988] | **0.877** [0.872, 0.882] |
| B1-LR+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.996 [0.995, 0.997] |
| B1-XGB+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.996, 0.998] |
| B1-LR′+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.997, 0.998] |
| B1-XGB′+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.998 [0.997, 0.998] |
| B2 *(circular)* | 1.000 [1.000, 1.000] | 0.985 [0.983, 0.987] | **0.935** [0.931, 0.939] |

**Table 6.8 — Dangerous false-negative rate** (EXP-019; validation; target ≤ 0.02).

| Model | 100% | 50% | 25% |
|---|---|---|---|
| B0 | 0.606 [0.599, 0.614] | 0.606 [0.599, 0.614] | 0.606 [0.599, 0.614] |
| B1-LR | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | **0.024** [0.021, 0.026] |
| B1-XGB | 0.000 [0.000, 0.000] | 0.004 [0.003, 0.005] | **0.059** [0.056, 0.062] |
| B1-LR+aug | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.001] |
| B1-XGB+aug | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.001] |
| B1-LR′+aug | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] |
| B1-XGB′+aug | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| B2 *(circular)* | 0.000 [0.000, 0.000] | 0.007 [0.006, 0.008] | **0.039** [0.036, 0.042] |

**Table 6.9 — Precision@3 and Recall@5 (`D_in`)** (EXP-019; validation).

| Model | P@3 100% | P@3 50% | P@3 25% | R@5 100% | R@5 50% | R@5 25% |
|---|---|---|---|---|---|---|
| B1-LR | 0.729 [0.726, 0.732] | 0.703 [0.700, 0.706] | 0.684 [0.682, 0.687] | 0.553 [0.551, 0.556] | 0.539 [0.536, 0.541] | 0.527 [0.525, 0.530] |
| B1-XGB | 0.764 [0.761, 0.767] | 0.725 [0.722, 0.728] | 0.703 [0.700, 0.706] | 0.569 [0.567, 0.572] | 0.549 [0.547, 0.551] | 0.525 [0.522, 0.527] |
| B1-LR+aug | 0.741 [0.738, 0.744] | 0.724 [0.722, 0.727] | 0.711 [0.708, 0.713] | 0.556 [0.553, 0.558] | 0.546 [0.543, 0.548] | 0.536 [0.533, 0.538] |
| B1-XGB+aug | 0.734 [0.731, 0.737] | 0.730 [0.727, 0.732] | 0.718 [0.715, 0.721] | 0.552 [0.550, 0.554] | 0.551 [0.548, 0.553] | 0.544 [0.542, 0.546] |
| B1-LR′+aug | 0.716 [0.714, 0.719] | 0.715 [0.712, 0.717] | 0.708 [0.705, 0.711] | 0.544 [0.542, 0.547] | 0.537 [0.535, 0.539] | 0.533 [0.531, 0.535] |
| B1-XGB′+aug | 0.744 [0.741, 0.747] | 0.726 [0.723, 0.729] | 0.718 [0.715, 0.721] | 0.568 [0.566, 0.570] | 0.553 [0.551, 0.556] | 0.547 [0.545, 0.549] |
| B2 *(circular)* | 0.756 [0.753, 0.759] | 0.747 [0.744, 0.750] | 0.733 [0.730, 0.736] | 0.570 [0.568, 0.572] | 0.562 [0.560, 0.564] | 0.556 [0.554, 0.559] |

**Table 6.10 — ECE of the raw scores, 10 bins** (EXP-019; validation) †.

| Model | 100% | 50% | 25% |
|---|---|---|---|
| B1-LR | 0.001 | 0.005 | 0.035 |
| B1-XGB | 0.000 | 0.310 | 0.665 |
| B1-LR+aug | 0.001 | 0.003 | 0.010 |
| B1-XGB+aug | 0.000 | 0.004 | 0.006 |
| B1-LR′+aug | 0.000 | 0.002 | 0.011 |
| B1-XGB′+aug | 0.000 | 0.003 | 0.006 |

**Table 6.11 — Paired comparisons on top-3** (McNemar; EXP-019; validation). "A only / B only" counts
the patients only A, or only B, gets right. At 100% every pair is 0 / 0.

| Level | A vs B (the change isolated) | All patients: A only / B only, p | Must-not-miss patients only |
|---|---|---|---|
| 50% | B1-XGB vs B1-LR (model) | 89 / 279, p = 6.7e-23 | 73 / 197, p = 7.1e-14 |
| 50% | B1-XGB+aug vs B1-LR+aug (model) | 3 / 0, p = 0.25 | 3 / 0, p = 0.25 |
| 50% | B1-XGB′+aug vs B1-LR′+aug (model) | 0 / 0, p = 1 | 0 / 0, p = 1 |
| 50% | B1-LR+aug vs B1-LR (masked copies) | 118 / 4, p = 1.4e-24 | 98 / 3, p = 8.5e-21 |
| 50% | B1-XGB+aug vs B1-XGB (masked copies) | 308 / 1, p = 7.2e-68 | 222 / 0, p = 9e-50 |
| 50% | B1-LR′+aug vs B1-LR+aug (asked channel) | 4 / 0, p = 0.12 | 3 / 0, p = 0.25 |
| 50% | B1-XGB′+aug vs B1-XGB+aug (asked channel) | 1 / 0, p = 1 | 0 / 0, p = 1 |
| 25% | B1-XGB vs B1-LR (model) | 845 / 2162, p = 2.9e-127 | 547 / 1363, p = 1.3e-77 |
| 25% | B1-XGB+aug vs B1-LR+aug (model) | 49 / 25, p = 0.0075 | 35 / 11, p = 0.0007 |
| 25% | B1-XGB′+aug vs B1-LR′+aug (model) | 44 / 27, p = 0.058 | 23 / 18, p = 0.53 |
| 25% | B1-LR+aug vs B1-LR (masked copies) | 1811 / 47, p < 1e-300 | 1226 / 28, p = 1.8e-250 |
| 25% | B1-XGB+aug vs B1-XGB (masked copies) | 3138 / 33, p < 1e-300 | 2046 / 8, p < 1e-300 |
| 25% | B1-LR′+aug vs B1-LR+aug (asked channel) | 79 / 42, p = 0.0011 | 50 / 22, p = 0.0015 |
| 25% | B1-XGB′+aug vs B1-XGB+aug (asked channel) | 52 / 22, p = 0.00075 | 27 / 18, p = 0.23 |

**Findings** (EXP-019).

1. **Under the protocol's masks, the original B1 fails the safety targets at 25%.** B1-XGB reaches
   must-not-miss recall@3 0.877 and a dangerous false-negative rate of 0.059, with raw scores badly
   calibrated (ECE 0.665 †); B1-LR degrades less but still misses both targets (0.925, 0.024).
   **The mask rule matters:** under EXP-017's token mask B1-LR met the must-not-miss target at 25%
   (0.974) and B1-XGB reached 0.935, although the question mask keeps slightly more evidence (6.71
   tokens against 6.21); dropping whole questions hurts more than dropping tokens. The question rule
   was fixed before scoring, and EXP-017 stays a diagnostic.
2. **Masked training copies repair almost all of it.** Every `+aug` model keeps top-3 ≥ 0.997 and
   must-not-miss recall@3 ≥ 0.996 at 25%, with a dangerous false-negative rate ≤ 0.001 and raw ECE
   ≤ 0.011 †. Against its unaugmented self at 25%, each `+aug` model gets top-3 right where the
   plain model fails for 1,811 patients (LR) and 3,138 (XGB), and the reverse for 47 and 33.
3. **The "asked" channel adds a small, significant gain at 25%:** top-1 0.931 → 0.946 (LR) and
   0.933 → 0.948 (XGB); top-3 better for 79 patients against 42 (LR, p = 0.0011) and 52 against 22
   (XGB, p = 0.00075); on must-not-miss patients alone significant for LR (50 / 22, p = 0.0015) but
   not for XGB (27 / 18, p = 0.23). At 50% it makes no significant difference.
4. **Once both are trained on masked copies, LR and XGBoost are close.** At 25% their top-3
   disagreement falls from 845 / 2,162 patients (plain) to 49 / 25 (`+aug`, XGBoost slightly
   better, p = 0.0075) and 44 / 27 (′+aug, p = 0.058). Most of EXP-017's gap came from the training
   rows, not the model family.
5. **Saturation returns at the reduced levels for the augmented B1** (R-16). At 25% the `+aug`
   variants reach top-3 0.997–0.998 and must-not-miss recall@3 0.996–0.998, so a fused or deep
   system has almost no room above them on those two metrics; top-1, MRR, Precision@3 and Recall@5
   still leave some. At 50% they are perfect or within four misses (§5.9).
6. **Calibrated on DDXPlus, near-certain on hand-written cases.** On masked validation patients at
   25% the `+aug` models' raw scores are calibrated (ECE 0.006–0.011 †): on DDXPlus a short history
   often does settle the condition. That is calibration against DDXPlus's own generator. On GC-001's
   classic presentation the same models give unstable angina 0.98–1.00, though only a troponin can
   separate it from MI (EXP-018, EXP-019 point 7). Both hold.
7. **The configured ranker became B1-LR′+aug on 2026-09-25**, because every hand-written case is a
   partial history, B1-LR misses both safety targets at 25%, and B1-LR′+aug meets them and sees a
   case's denials; it was replaced by the transformer on 2026-10-13 (§6.8).

## 6.5 The graph alone (B2) — circular on DDXPlus (R-12)

> **Circularity caveat.** DDXPlus generated its patients from `release_conditions.json`, the file
> the graph is built from. Every B2 figure on DDXPlus patients therefore measures the graph
> recognising its own definitions, **not skill** (R-12, `docs/05` §8 rule 7). The hand-written golden
> cases are the evidence that counts (§6.11).

**The overlap score and its three defects** (EXP-014 to EXP-016, before 2a). The first graph score,
a weighted overlap, ranked DDXPlus patients 0.880 top-1 and 0.998 top-3 on the DDXPlus-only graph
(EXP-015, ties broken at random; circular), ranked unstable angina first for only 0.21 of its own
patients, and scored a stable-angina picture *plus rest pain* as stable 1.00, unstable 0.875.
Adding BODHI-S lowered its top-1 to 0.856, and MI's to 0.602, because the score divided by
everything a condition might show (EXP-016). On GC-001 it put MI fifth by graph score alone
(EXP-014).

**Table 6.12 — B2 at 100%: the overlap score against the naive-Bayes score of 2a** (EXP-005;
validation; full graph; the protocol's deterministic tie rule; **circular**).

| Metric (100%) | Overlap (before 2a) | Naive Bayes (2a) |
|---|---|---|
| Top-1 | 0.8583 [0.8549, 0.8620] | 0.9234 [0.9206, 0.9259] |
| Top-3 | 0.9957 [0.9950, 0.9964] | 0.9996 [0.9994, 0.9998] |
| MRR | 0.9246 [0.9227, 0.9266] | 0.9613 [0.9598, 0.9625] |
| Precision@3 (`D_in`) | 0.6224 [0.6199, 0.6253] | 0.7559 [0.7532, 0.7588] |
| Recall@5 (`D_in`) | 0.4959 [0.4933, 0.4983] | 0.5698 [0.5675, 0.5719] |
| Recall@5, full `D` | 0.2853 [0.2835, 0.2872] | 0.3333 [0.3315, 0.3353] |
| Must-not-miss recall@3 | 0.9926 [0.9914, 0.9940] | 0.9998 [0.9995, 1.0000] |
| Dangerous false-negative rate | 0.0006 [0.0002, 0.0010] | 0.0000 [0.0000, 0.0000] |

**Findings.**

1. **The naive-Bayes score fixes all three defects, on the hand-written checks** (EXP-005). With
   rest pain present, unstable angina outranks stable angina by a likelihood ratio of about 90
   (scores −2.21 against −6.71); with rest pain denied, stable angina comes first; GC-001's MI is
   first by graph score alone; and BODHI-S no longer lowers MI (its top-1 on validation 0.992 →
   1.000, circular). Per condition, its top-1 is 1.000 for seven conditions, 0.997 for unstable
   angina and pneumothorax, 0.979 for myocarditis, 0.972 for Boerhaave and 0.949 for pericarditis †.
2. **The graph alone cannot recognise stable angina: 0.000 top-1** †. With nothing denied, stable and
   unstable angina tie, and the tie rule puts unstable angina first every time (0.997). That is the
   safe direction and the honest answer, since not mentioned is not denied; telling the two apart is
   the ML ranker's job (EXP-005).
3. **The circular figures are high, which is the point of the caveat.** B2's Precision@3, 0.756,
   nearly matches B1-XGB's 0.764 with no training at all, and its must-not-miss recall@3 at 100% is
   0.9998 [0.9995, 1.0000] (EXP-005; 1.000 to three places in EXP-019). At 50% and 25% B2 has the best Precision@3 of any system and at every
   level the best Recall@5, except that at 100% B1-XGB's Precision@3 is higher (0.764 against 0.756)
   and its Recall@5 0.001 lower (EXP-019 point 8). None of this is skill.
4. **At reduced evidence B2 degrades between the plain and the augmented B1:** at 25%, top-1 0.769
   and must-not-miss recall@3 0.935 [0.931, 0.939], below the target (Tables 6.4, 6.7; circular).
5. **Hand-authored dissection costs DDXPlus patients almost nothing.** Adding it moved top-1 by
   0.001; dissection took a top-3 place for 3.1% of patients and first place for none (EXP-015,
   overlap score) †. No DDXPlus patient has it.

## 6.6 The red-flag rules, and their collapse at reduced evidence

**Table 6.13 — The red-flag layer at 100%, before and after 2d** (EXP-008; validation; the rules
alone, on each patient's concepts).

| Measure (100%) | Before 2d | After 2d |
|---|---|---|
| Rules; must-not-miss conditions without one | 5; 3 † | 10; 0 † |
| **Red-flag sensitivity** (target ≥ 0.95) | 0.449 [0.441, 0.456] | **0.823** [0.818, 0.829] |
| Red-flag precision, strict | 0.230 [0.226, 0.234] | 0.512 [0.508, 0.516] |
| Red-flag precision, decision A-7's mapping | — | 0.848 [0.845, 0.852] |
| Any flag, all patients | 59% † | 58% † |
| **Any flag, patients without a must-not-miss condition** (the false-alarm burden) | 31% † | **25%** † (0.246 [0.240, 0.253], EXP-019) |
| … of stable angina / pericarditis / AF, PSVT, GERD, panic | 98% / 98% / 0% † | 95% / 65% / 0% † |

**Table 6.14 — Each rule on its own patients and on everyone else, 100%** (EXP-008; validation).

| Rule | Own patients flagged, before → after | Everyone else flagged, before → after |
|---|---|---|
| Aortic dissection | no DDXPlus patients | **50% → 8%** † (all of it Boerhaave and pneumothorax, by "tearing") |
| NSTEMI / STEMI | 0.907 † | 24% † (unstable angina, stable angina, pulmonary edema) |
| Unstable angina | — → 0.892 [0.880, 0.903] | 0% † |
| Pulmonary embolism | 0.785 † → 0.922 [0.913, 0.930] | 0% → 2% † (pulmonary edema patients reporting calf pain, 502 of 599, or one swollen leg) |
| Spontaneous pneumothorax | 0.325 [0.301, 0.348] | 3% † |
| Myocarditis | — → 0.641 [0.617, 0.665] | 6% † (pericarditis, 62% of its patients) |
| Acute pulmonary edema | — → 0.958 [0.949, 0.966] | 0% † |
| Boerhaave | 0.748 [0.730, 0.767] | 0% † |

**Table 6.15 — The red-flag layer at three levels** (EXP-019; validation).

| Measure | 100% | 50% | 25% |
|---|---|---|---|
| Sensitivity (target ≥ 0.95) | 0.823 [0.818, 0.829] | 0.327 [0.320, 0.334] | 0.135 [0.130, 0.140] |
| Precision (A-7) | 0.848 [0.845, 0.852] | 0.890 [0.884, 0.896] | 0.931 [0.922, 0.941] |
| Precision (strict) | 0.512 [0.508, 0.516] | 0.650 [0.640, 0.659] | 0.822 [0.808, 0.836] |
| Patients without a must-not-miss condition flagged | 0.246 [0.240, 0.253] | 0.050 [0.046, 0.053] | 0.007 [0.006, 0.008] |

**Findings.**

1. **The main cause of over-firing is gone** (EXP-008). Following the ADD-RS, back radiation no
   longer fires the dissection rule: it flags 8% of patients instead of 50%, and the 8% is DDXPlus's
   vocabulary, since "tearing" is how DDXPlus describes pneumothorax and oesophageal-rupture pain,
   both must-not-miss emergencies themselves.
2. **Alarm fatigue is measured on the right patients.** Half of the validation patients have a
   must-not-miss condition, so a perfect rule set would flag half of everyone. On the others the
   burden fell from 31% to 25%, and almost all of what remains is the MI rule on stable angina
   (95%): exertional pressure pain radiating to the arm is the ischaemic pattern, and nothing in a
   history separates stable from unstable disease at presentation (EXP-008). Decision A-7 counts that
   flag as appropriate.
3. **Sensitivity, 0.823, misses the 0.95 target, and we did not loosen the rules to chase it.**
   Three conditions are the reason, pneumothorax (0.325), myocarditis (0.641) and Boerhaave (0.748),
   each capped by DDXPlus's sampling: 20% of Boerhaave patients never report the vomiting, 29% of
   myocarditis patients the viral illness, and pneumothorax's onset speed is drawn uniformly over a
   range. Rules that fire without the defining finding would fit the generator, not medicine
   (EXP-008). Lowering the sudden-onset cut-off from ≥ 8 to ≥ 6 would lift the pneumothorax rule
   from 32% to 54% of its patients for 3% → 5% of everyone else; it stays at ≥ 8 (decision A-5)
   (EXP-008) †.
4. **At reduced evidence the rules collapse** (EXP-019): sensitivity falls to 0.327 at 50% and 0.135
   at 25%. At 25% the masks keep about 28% of a patient's questions beyond the initial one, and most
   rules need two or three findings together, so the combination rules fall furthest (MI 0.907 →
   0.025, myocarditis 0.641 → 0.014, pneumothorax 0.325 → 0.009) †. The pulmonary-edema rule, which
   can fire on one finding, keeps the most (0.958 → 0.319), with PE next (0.922 → 0.282) †. Precision
   rises as the rules fire less.
5. **So on DDXPlus at reduced evidence, must-not-miss safety is carried by the ranker, not by the
   rules** (EXP-019 point 6). This bears directly on H2-R ("fusion + red flags > ML-only").
6. **Every rate here is an upper bound on what the rules do with real histories** (R-12): DDXPlus
   generated the patients from the definitions the crosswalk reads. The safety layer v1 found nothing
   to repair on 3,000 validation patients (EXP-008).

## 6.7 Seeds 43–46 of the classical models (EXP-020)

`docs/05` §6 requires five seeds (42–46) for every stochastic model and every `+aug` logistic
regression. EXP-020 trained seeds 43–46 of B1-XGB, B1-XGB+aug and B1-XGB′+aug, with the logistic
regressions trained alongside them, and seed 42 of plain B1 again as a check, on Colab in 33 min
25 s.

- **Seed 42 of plain B1 reproduces EXP-004 byte for byte** (identical SHA-256 of the model files).
  Plain B1-LR is deterministic: its seeds 43–46 have the same coefficients. The `+aug` logistic
  regressions differ by seed, because their masked copies are drawn from it (EXP-020).
- **At 100% the seed changes nothing that matters:** top-1 moves by at most 0.0001, and every run
  keeps top-3 and must-not-miss recall@3 at 1.000 (EXP-020) † (intervals are in each run's
  `metrics.json`, not in `docs/08`).

**Table 6.16 — The range over seeds 43–46 at 25%** (EXP-020; validation; point estimates †, the
intervals are in `data/interim/exp020_seed43.json`–`46` and must be copied).

| Model (25%) | Top-1 | MRR | Must-not-miss recall@3 | Dangerous FNR |
|---|---|---|---|---|
| B1-LR (deterministic) | 0.7656 | 0.8582 | 0.9251 | 0.0237 |
| B1-XGB | 0.2416–0.2643 | 0.5756–0.5848 | 0.8663–0.8797 | 0.0601–0.0740 |
| B1-LR+aug | 0.9308–0.9318 | 0.9625–0.9630 | 0.9959–0.9960 | 0.0008–0.0012 |
| B1-XGB+aug | 0.9333–0.9352 | 0.9642–0.9650 | 0.9968–0.9975 | 0.0002–0.0006 |
| B1-LR′+aug | 0.9447–0.9468 | 0.9703–0.9716 | 0.9970–0.9980 | 0.0002–0.0006 |
| B1-XGB′+aug | 0.9465–0.9470 | 0.9716–0.9719 | 0.9977–0.9979 | 0.0002–0.0004 |

**Every EXP-019 conclusion holds at every seed:** the plain models miss the safety targets at 25%;
every `+aug` model meets them; and the "asked" channel's gain (about +0.014 top-1 and +0.008 MRR) is
larger than the seed spread (≤ 0.002 for the `+aug` models). Seed 42 falls inside every `+aug` range.
The seed matters most for plain XGBoost: 0.58–0.68 top-1 at 50% (seed 42: 0.597, as EXP-017 records
it) †. At 50% every `+aug` model is at 0.982–0.986 top-1 and must-not-miss recall@3 ≥ 0.9999, at
every seed † (EXP-020).

## 6.8 The transformer B1-DL and the safety floor D-13 (EXP-021)

**What was fixed before training** (2026-10-12): the architecture and training of §4.9, the rule
D-13 as code (§5.10), and the list of figures to report. At that point no deep model existed, no
fused DDXPlus ranking had been scored, and no test data had been read. EXP-019's comparator figures
were known, so the headroom was known to be small (R-19) (EXP-021).

**The run.** Ten networks (two arms × seeds 42–46) trained on Kaggle (two T4 GPUs) in 2 h 10 min;
every run's provenance, fingerprints and training rows were checked; the NumPy pass matches PyTorch
to 4.3 × 10⁻⁷; early stopping chose epochs 18 to 43, none at the cap; the laptop reproduces seed 42's
validation metrics to 7 × 10⁻¹¹; and the cloud's scoring reproduced both mask digests at every seed
(EXP-021).

**Table 6.17 — D-13 at 25% of the history** (EXP-021; validation; seed 42; run on the laptop at
`ad4e2d4`, after A-10). The differences are paired, transformer minus logistic regression.

| Model (25%) | Top-1 | MRR | Must-not-miss recall@3 | Dangerous false negatives |
|---|---|---|---|---|
| **B1-DL′+aug** | 0.9487 [0.9464, 0.9509] | 0.9728 [0.9715, 0.9740] | 0.9981 [0.9973, 0.9987] | 0.0004 [0.0001, 0.0007] |
| B1-LR′+aug | 0.9459 [0.9436, 0.9482] | 0.9710 [0.9697, 0.9722] | 0.9975 [0.9967, 0.9982] | 0.0004 [0.0001, 0.0008] |
| **B1-DL+aug** | 0.9347 [0.9323, 0.9373] | 0.9649 [0.9635, 0.9663] | 0.9976 [0.9968, 0.9983] | 0.0004 [0.0001, 0.0008] |
| B1-LR+aug | 0.9311 [0.9285, 0.9337] | 0.9627 [0.9612, 0.9641] | 0.9958 [0.9948, 0.9967] | 0.0009 [0.0005, 0.0014] |

At 100% both transformers reach must-not-miss recall@3 1.000 and a dangerous false-negative rate of
0.000, with top-1 0.9985 (B1-DL′+aug) and 0.9984 (B1-DL+aug), and MRR 0.9992 † (EXP-021).

**Table 6.18 — The checks of D-13** (EXP-021; validation; seed 42).

| Check | B1-DL′+aug vs B1-LR′+aug | B1-DL+aug vs B1-LR+aug |
|---|---|---|
| Must-not-miss recall@3 ≥ 0.95 and dangerous rate ≤ 0.02, at 100% and 25% | pass | pass |
| Must-not-miss recall@3 at 25%: difference, lower bound > −0.005 | +0.0006 [−0.0001, +0.0013], pass | +0.0018 [+0.0011, +0.0026], pass |
| MRR at 25%: difference, lower bound > −0.005 | +0.0018 [+0.0009, +0.0026], pass | +0.0022 [+0.0016, +0.0029], pass |
| All four golden cases, red flags on and off | pass | pass |
| **D-13** | **passes** | **passes** |

**Findings.**

1. **Both arms pass the safety floor, so the configured ranker is now B1-DL′+aug** (2026-10-13), and
   A0's ML component is B1-DL+aug and A0′'s B1-DL′+aug, as `docs/05` §5 already said; no protocol
   change was needed. B1-LR′+aug stays as the comparison and the backup.
2. **Not only non-inferior on validation.** For both arms the MRR interval at 25% excludes 0
   (B1-DL′+aug +0.0018 [+0.0009, +0.0026]), and for the unprimed arm the must-not-miss interval does
   too (+0.0018 [+0.0011, +0.0026]). For the primed arm the must-not-miss difference includes 0
   (+0.0006 [−0.0001, +0.0013]). The effects are small: the transformer's MRR is 0.0018 higher at
   25%. **These are validation figures; only the test read counts** for H6
   [pending: EXP-012, test split].
3. **The comparators' figures equal EXP-019's at seed 42.** *(Correction: a check of the cloud bundle
   first quoted B1-LR′+aug's 25% figures as 0.9451, 0.9707, 0.9976 and 0.0005; those were seed 43's,
   EXP-020. Seed 42's, 0.9459, 0.9710, 0.9975 and 0.0004, are used here; EXP-021.)*
4. **The safety targets hold at every seed** on the cloud's own figures: at 25%, must-not-miss
   recall@3 ≥ 0.9969 and the dangerous rate ≤ 0.0007, for both arms; seeds 43–46 stay within 0.003
   of seed 42 on every figure at 25% (EXP-021) †.
5. **The no-findings probe:** B1-DL′+aug answers PSVT at 0.38 and B1-DL+aug atrial fibrillation at
   0.39–0.41 †. That is EXP-018's pattern: no longer certain, not gone (EXP-021).
6. **Disclosed:** these validation figures were read on 2026-10-13, before the team answered E-2.
   E-2's recommendation, written without any deep result, stands as written (EXP-021).

~~**Still to report** (EXP-021): non-inferiority at seeds 43–46, and the full comparison table
(McNemar on top-3 and the paired MRR test against B1-LR(′)+aug and B1-XGB(′)+aug at every level and
seed), with the seed spread and the training times [pending: EXP-021 seeds 43–46 and the full
comparison table].~~ *Reported 2026-10-14, below.*

**The full report** (EXP-021; validation; run on Kaggle's CPU by the owner,
`notebooks/kaggle_exp021_reports.ipynb`; both mask digests reproduced; seed 42 reproduces the
laptop's D-13 run exactly, e.g. MRR +0.0018 [+0.0009, +0.0026]).

**Table 6.19 — D-13 at every seed, at 25% of the history** (EXP-021; validation). The differences
are paired against the logistic regression trained like each arm, transformer minus logistic
regression. The four figures before them have no interval in `docs/08` †.

| Seed | Arm | Top-1 † | MRR † | Must-not-miss recall@3 † | Dangerous FN † | Must-not-miss Δ | MRR Δ |
|---|---|---|---|---|---|---|---|
| 42 | B1-DL′+aug | 0.9487 | 0.9728 | 0.9981 | 0.0004 | +0.0006 [−0.0001, +0.0013] | +0.0018 [+0.0009, +0.0026] |
| 43 | B1-DL′+aug | 0.9493 | 0.9732 | 0.9988 | 0.0002 | +0.0011 [+0.0005, +0.0018] | +0.0025 [+0.0016, +0.0034] |
| 44 | B1-DL′+aug | 0.9476 | 0.9721 | 0.9982 | 0.0004 | +0.0002 [−0.0005, +0.0009] | +0.0005 [−0.0004, +0.0013] |
| 45 | B1-DL′+aug | 0.9506 | 0.9737 | 0.9983 | 0.0004 | +0.0012 [+0.0006, +0.0019] | +0.0033 [+0.0025, +0.0042] |
| 46 | B1-DL′+aug | 0.9485 | 0.9728 | 0.9983 | 0.0003 | +0.0009 [+0.0002, +0.0015] | +0.0014 [+0.0004, +0.0022] |
| 42 | B1-DL+aug | 0.9347 | 0.9649 | 0.9976 | 0.0004 | +0.0018 [+0.0011, +0.0026] | +0.0022 [+0.0016, +0.0029] |
| 43 | B1-DL+aug | 0.9329 | 0.9637 | 0.9969 | 0.0007 | +0.0011 [+0.0003, +0.0018] | +0.0012 [+0.0004, +0.0021] |
| 44 | B1-DL+aug | 0.9355 | 0.9652 | 0.9973 | 0.0004 | +0.0014 [+0.0006, +0.0022] | +0.0022 [+0.0015, +0.0030] |
| 45 | B1-DL+aug | 0.9356 | 0.9652 | 0.9973 | 0.0005 | +0.0014 [+0.0008, +0.0021] | +0.0025 [+0.0016, +0.0033] |
| 46 | B1-DL+aug | 0.9336 | 0.9642 | 0.9969 | 0.0005 | +0.0009 [+0.0002, +0.0017] | +0.0014 [+0.0008, +0.0022] |

7. **D-13 holds at all five seeds, for both arms.** Every network meets the safety targets at 100%
   and 25%, is non-inferior to the logistic regression trained like it at 25%, and passes the
   golden cases, red flags on and off (EXP-021).
8. **Against the logistic regression, the transformer's MRR is higher at nine of ten networks.**
   The exception is seed 44's B1-DL′+aug, whose interval includes 0 (+0.0005 [−0.0004, +0.0013])
   (EXP-021).
9. **Against XGBoost trained like it, the transformer is level, not better** (25%, paired MRR
   difference; EXP-021):
   - B1-DL′+aug against B1-XGB′+aug: higher at seeds 43, 45 and 46 (+0.0013, +0.0018, +0.0010,
     intervals above 0) †, level at 42 and 44. McNemar on top-3 finds no difference at any seed
     (p ≥ 0.17).
   - B1-DL+aug against B1-XGB+aug: level at four seeds. **At seed 46 it is worse:** on top-3 at
     25%, 16 patients only it ranks in the top 3 against 40 only XGBoost does (p = 0.0021); and MRR
     at 50%, −0.0008 [−0.0011, −0.0004].
   - At 50%, top-3 is saturated: zero or one discordant patient at every seed.
10. **What it means.** The transformer earns its place as A0's component over the logistic
    regression it replaced, as D-13 required, but it is no better than XGBoost trained the same way.
    That is the expected result where the headroom is this small (R-19). These are validation
    figures: H6 is judged on the test split with seed 42's model, and every seed's figures are
    reported beside it [pending: EXP-012, test split].

## 6.9 Calibration (EXP-007)

EXP-007 was pre-registered on 2026-10-13 before any calibrator was fitted on project data (§5.10).
Stage 1 calibrated only systems whose validation rankings EXP-019 had already published, so no fused
or deep result was read; the transformer arms followed after D-13, by the same rule. **Stage 2, A0′'s
fused score, on which H5 is judged, waits for EXP-006's α** [pending: EXP-007 stage 2].

**Table 6.20 — Calibration, cross-fitted, calibrator fitted at 100% and applied at each level**
(EXP-007; validation). Brier score over 14 conditions †; top-label ECE, with its interval for the
method the rule chose (**bold**) and † for the others. **B2 is circular.**

| System | Method | Brier 100% | 50% | 25% | ECE 100% | 50% | 25% |
|---|---|---|---|---|---|---|---|
| B1-LR′+aug | **none** | 0.0033 | 0.0218 | 0.0760 | 0.0002 [0.0001, 0.0007] | 0.0015 [0.0011, 0.0029] | 0.0110 [0.0092, 0.0131] |
| | temperature | 0.0033 | 0.0219 | 0.0764 | 0.0002 | 0.0024 | 0.0139 |
| | Platt | 0.0034 | 0.0257 | 0.1130 | 0.0010 | 0.0145 | 0.0981 |
| | isotonic | 0.0033 | 0.0298 | 0.1057 | 0.0000 | 0.0091 | 0.0296 |
| B1-LR+aug | **none** | 0.0034 | 0.0292 | 0.0956 | 0.0009 [0.0004, 0.0013] | 0.0033 [0.0023, 0.0046] | 0.0099 [0.0082, 0.0121] |
| | temperature | 0.0034 | 0.0292 | 0.0951 | 0.0002 | 0.0034 | 0.0059 |
| | Platt | 0.0034 | 0.0368 | 0.1491 | 0.0010 | 0.0138 | 0.0943 |
| | isotonic | 0.0033 | 0.0420 | 0.1650 | 0.0001 | 0.0162 | 0.0617 |
| B2 *(circular)* | none | 0.0447 | 0.1156 | 0.2364 | 0.0379 | 0.0358 | 0.0295 |
| | temperature | 0.0452 | 0.1174 | 0.2360 | 0.0362 | 0.0333 | 0.0154 |
| | **Platt** | 0.0084 | 0.0765 | 0.2382 | 0.0032 [0.0027, 0.0039] | 0.0175 [0.0161, 0.0194] | 0.0510 [0.0486, 0.0537] |
| | isotonic | 0.0075 | 0.0823 | 0.2793 | 0.0013 | 0.0211 | 0.0556 |

**Table 6.21 — The selection rule: mean Brier score over the three levels** (EXP-007) †.

| System | none | temperature | Platt | isotonic | Chosen |
|---|---|---|---|---|---|
| B1-LR′+aug | 0.0337 | 0.0339 | 0.0474 | 0.0463 | none |
| B1-LR+aug | 0.0427 | 0.0426 | 0.0631 | 0.0701 | none (temperature lowest, none within 0.0005) |
| B2 *(circular)* | 0.1323 | 0.1328 | 0.1077 | 0.1230 | Platt |
| B1-DL′+aug | 0.0317 | 0.0317 | 0.0318 | 0.0339 | none (temperature lowest by 0.00005, inside the margin) |
| B1-DL+aug | 0.0396 | 0.0396 | 0.0413 | 0.0481 | none |

**Table 6.22 — The transformer arms, raw scores ("none"), cross-fitted ECE** (EXP-007; validation;
seed 42; after A-10).

| System | 100% | 50% | 25% |
|---|---|---|---|
| B1-DL′+aug | 0.0003 [0.0001, 0.0008] | 0.0024 [0.0016, 0.0036] | 0.0047 [0.0034, 0.0068] |
| B1-DL+aug | 0.0001 [0.0000, 0.0005] | 0.0037 [0.0030, 0.0051] | 0.0039 [0.0028, 0.0062] |

**Findings** (EXP-007).

1. **The ML components need no calibration on DDXPlus.** The rule keeps all four raw: their ECE is
   at most 0.011 at every level, and no method fitted at 100% improves the levels' mean Brier score
   by more than the margin. At 25% both transformers are better calibrated raw (0.0047, 0.0039) than
   the logistic regressions (0.0110, 0.0099). All five systems pass the display rule (ECE < 0.10 at
   all three levels); B2's largest is 0.051 at 25%.
2. **A calibrator fitted on full histories over-sharpens short ones.** At 100% these models are right
   for 99.8% of patients, so a fit there learns that a high score can be trusted. Every method that
   fits more than one number makes 25% worse: for B1-LR′+aug, Platt takes the ECE from 0.011 to 0.098
   and isotonic to 0.030; for the transformers, isotonic takes it to 0.0137 and 0.0289 †. A
   calibrator fitted at the level itself helps (B1-LR′+aug at 25%: 0.011 → 0.002–0.004 †), but it
   cannot be deployed, because a hand-written case has no known level. One calibrator fitted at 100%
   and applied unchanged, as the protocol requires, is the deployable design, and judging methods on
   the levels' mean is what protects short histories.
3. **Isotonic regression is unsafe here.** *(This count was added after seeing its log loss, so it is
   post hoc, and it chooses nothing.)* Fitted at 100%, isotonic regression gives the true condition
   under 0.001 for 2.4% (B1-LR′+aug) and 5.0% (B1-LR+aug) of patients at 25%, and for 8.7% of the
   graph's (10.2% of its must-not-miss patients) †. A displayed "0%" beside a must-not-miss condition
   is the failure the system exists to prevent. The rule chose isotonic for no system.
4. **The graph's posterior is overconfident when it is wrong** (circular): its Brier score at 100% is
   0.0447 raw and 0.0084 after Platt, although its top-label ECE (0.0379) was already within the
   target. For about a quarter of patients at 25%, B2's most probable calibrated condition is not its
   first-ranked one (0.244 under Platt) †.
5. **No method fixes the hand-written cases.** Calibrated on DDXPlus, the logistic regression stays
   near-certain on GC-001 (unstable angina 0.998, MI 0.000 under every method except Platt, 0.014);
   the transformer gives GC-001's unstable angina 0.997 and GC-002's embolism 1.000 †. **The
   dissection rule works:** GC-003's dissection keeps the graph's 0.677 under every method †. *(The
   logistic-regression golden figures were computed before A-10; the DDXPlus figures do not depend
   on it.)*
6. **H5 is not judged here.** It concerns A0's fused score, at 100%, on the test split
   [pending: EXP-007 stage 2] [pending: EXP-012, test split]. How the dashboard shows a probability
   (a number, or a band with a floor such as "< 1%") is decided at stage 2.

## 6.10 Complexity and latency (EXP-022)

Measured on the owner's laptop CPU (Windows 11, 24 logical CPUs, Python 3.11.9, NumPy 1.26.4, one
BLAS thread); batch 1, the dashboard's case; 100 warm-up calls, then 1,000 timed calls, three passes
in shuffled order. Latency percentiles are not ratio metrics, so they carry no bootstrap interval;
the p99 and the maximum are reported instead.

**Table 6.23 — Each model's size and cost per patient** (EXP-022; baseline run of 2026-10-12, the
transformer untrained, since its cost is fixed by the architecture).

| Model | Parameters / trees | Multiply-adds per patient | On disk | Load | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|---|
| B1-LR′+aug | 8,996 | 8,983 | 0.11 MB | 0.003 s | 0.009 ms | 0.010 ms | 0.011 ms | 0.03 ms |
| B1-XGB′+aug | 5,824 trees, 264,396 nodes (depth ≤ 6) | ≤ 6 comparisons per tree | 14.6 MB | 3.16 s | 0.40 ms | 0.43 ms | 0.48 ms | 21.7 ms |
| B1-DL′+aug (untrained) | 225,645 | 9,610,752 | 1.22 MB (JSON) | 0.005 s | 0.75 ms | 0.78 ms | 0.82 ms | 1.4 ms |

**The trained transformer** (re-run of 2026-10-13, after D-13): batch 1, one thread, p50 1.29 ms,
p95 1.38 ms, p99 1.52 ms, max 2.67 ms; 1.23 MB on disk. In the same run B1-LR′+aug's p95 was
0.020 ms and B1-XGB′+aug's 0.59 ms (EXP-022).

**Table 6.24 — The whole ranking through the pipeline** (EXP-022; the four golden cases, 300 timed
runs each; the NetworkX graph, the configured α, the template explainer, no retrieval, no LLM).

| Run | Ranker | p50 | p95 | Max |
|---|---|---|---|---|
| Baseline, 2026-10-12 | B1-LR′+aug | 0.91–1.10 ms | 0.94–1.13 ms | 26–31 ms |
| Baseline, 2026-10-12 | B1-DL′+aug (untrained) | 1.79–1.95 ms | 1.95–2.01 ms | 27–40 ms |
| Re-run, 2026-10-13, with the model's reasons | B1-LR′+aug | — | 2.0–2.8 ms | see note |
| Re-run, 2026-10-13, with the model's reasons | B1-DL′+aug (trained) | — | 10.3–12.6 ms | see note |

*Note:* the re-run records a maximum of 38–62 ms without saying which ranker it belongs to, and no
p50 (EXP-022) †.

**Findings** (EXP-022).

1. **Live re-ranking is feasible by a wide margin.** The whole ranking with the trained transformer
   takes about 0.01 s at p95, against the 0.1 s at which a response conventionally feels instant
   [citation needed: Nielsen's response-time limits]; no clinical standard sets a number. So the
   dashboard re-ranks as each finding is entered (D-14); a re-rank there, which asks for the reasons
   once more for its rows, costs about 20 ms †.
2. **The transformer costs about 80× logistic regression's time and about 1,000× its multiply-adds**
   (0.78 against 0.010 ms at p95 in the baseline), yet stays under 1.5 ms, and its cost is the same
   for every patient.
3. **XGBoost is the slowest to load (3.16 s) and the largest (14.6 MB)**, with the worst single-call
   outlier (21.7 ms).
4. **The model is not the bottleneck.** In the re-run most of the pipeline's time is the explanation
   step, 5.6–8.5 ms on average, which now includes the model's reasons (one batch of the case and each
   case-without-one-finding); the ranker itself takes 1.3–1.5 ms †.
5. **Limits.** Soft real time only: Python on Windows guarantees no deadline. One laptop, one thread,
   the CPU only. The LLM explanation (Phase 3) is outside these figures; NFR-1 allows it 20 s.
   Multithreaded BLAS made one patient cost about 7–18 ms instead of 0.78 ms, so batches of up to 64
   rows now hold BLAS to one thread.

## 6.11 The golden cases

The four hand-written cases (§3.6) have no ground truth beyond the clinical expectation, and their
input passes through the hand-curated case-to-token translation (R-17). They are the evidence that is
not circular.

**Table 6.25 — The golden cases, component by component.**

| Case (expected) | Graph alone, rank (EXP-005) | Model alone, top answer | Fused, red flags off, α = 0.5 (B1-LR and B1-LR′+aug, EXP-006) | Red flags (EXP-008) |
|---|---|---|---|---|
| GC-001 (MI, top 3) | MI 1st | unstable angina, with B1-LR′+aug and B1-DL′+aug after A-10 (EXP-021); the transformer 0.997 † (EXP-007) | MI 2nd (also with both transformer arms, EXP-021) | MI flag |
| GC-002 (PE, top 3) | PE 1st | pulmonary embolism; the transformer 1.000 † (EXP-007) | PE 1st | PE and pneumothorax flags |
| GC-003 (dissection, top 5) | dissection 1st | spontaneous pneumothorax with the transformer, by "tearing pain" (0.557 and 0.757 for the two arms †, EXP-007); no model can score dissection | dissection 1st | dissection flag |
| GC-004 (GERD, top 3) | GERD 1st | GERD; B1-LR′+aug 1.000 † (EXP-007, computed before A-10) | GERD 1st | none |

**Findings.**

1. **Every golden case passes with every model** (B1-LR, B1-LR′+aug and both transformer arms), with
   red flags on and off; GC-001's MI is 2nd with flags off and 1st with them on (EXP-021, after
   A-10). The golden expectations were never changed.
2. **The fusion does what it was designed for on GC-003.** The model cannot score dissection, and the
   transformer alone answers pneumothorax; the pool gives dissection the graph's posterior, so it is
   first even with the red flags off. Under the first design, min-max fusion, it was 3rd, and 4th with
   the "asked"-channel models (EXP-005, EXP-006, EXP-018).
3. **The fusion also overrules a near-certain model on GC-001.** The models give unstable angina
   0.98–1.00, though a history cannot separate it from MI; the floor ε and the graph lift MI to 2nd
   with flags off, and the MI red flag puts it 1st (EXP-006, EXP-021).
4. **On the hand-written cases alone, every α on EXP-006's grid is admissible** for all six
   components (the four `+aug` B1 variants and both transformer arms): GC-001's MI is 2nd up to
   α = 0.85 and 1st from 0.90, under every model (EXP-006, note of 2026-10-13).
5. **A-10 was found here.** Before A-10, the transformer alone answered acute pulmonary edema on
   GC-001 at 0.9997, with its reasons putting about +15 log-odds on the "radiation to jaw or arm"
   finding, because that finding was translated into the jaw answer only pulmonary-edema patients give
   in DDXPlus. After A-10 both models alone answer unstable angina, and the transformer's reasons count
   the finding slightly for MI (+0.5 log-odds) (EXP-021). The fused pipeline had passed GC-001 both
   before and after.

## 6.12 Fusion

[pending: EXP-006 fusion sweep]

EXP-006's rule is fixed (§5.10) and its golden gate recorded (§6.11). The sweep has not been run:
it waits for the team's answer to E-2's first item (§5.9), then the owner's choice of where it runs.
When it runs, this section reports, for each of the six components: the α chosen and the branch of
the rule taken; at every α, Precision@3, how often dissection enters the top 3, and how many
must-not-miss patients dissection alone pushes out of it; and at the chosen α, with and without red
flags, every §3.1–§3.2 metric with its interval, the Brier score over all 14 conditions, ECE and the
reliability table, and McNemar on top-3 and the paired MRR test for fused against its ML component
and against B2. **No fused ranking of a DDXPlus patient has been scored.**

## 6.13 Retrieval and the language model

[pending: retrieval EXP-011] [pending: LLM baselines B3/B4, EXP-009/010]

Retrieval quality (Recall@k, Precision@k, nDCG@10, with two-rater relevance judgements and Cohen's κ),
the supported- and unsupported-claim rates and citation correctness of the grounded explainer, and
the LLM-only (B3) and text-RAG (B4) baselines are Phase 3 work and not yet measured. If B3 and B4 are
not scored at 50% and 25%, the report says so (`docs/05` §3.7).

*2026-10-14.* The retrieval code is built (§4.14.2), and EXP-011's evaluation was pre-registered
before any corpus existed: 26 queries (one per condition, and each golden case's three leading
candidates), a blind 0–2 judgement of each pooled passage by two team members, and for each method
Precision@3, nDCG@3 and nDCG@10, and pooled Recall@10, each with its 95% bootstrap interval; the
hybrid is compared with each method alone on nDCG@10 by the paired bootstrap (EXP-011). Nothing is
measured until the owner's corpus run and the judgements.

## 6.14 Where each component goes wrong, by condition (EXP-023)

**What it is.** A descriptive analysis of the four components alone, at 100%, 50% and 25% of the
history: errors and calibration, by condition. It is not a test.
- Nothing is trained, tuned, fitted or chosen, and no figure feeds a decision. It was not
  pre-registered, has no hypothesis and no verdict, and none of its figures may be quoted as a test
  of H1–H6.
- **Never a fused ranking:** E-2 is unanswered, so no fused validation ranking is read (§6.12).
- The components are B1-DL′+aug (the configured ranker, seed 42), B1-LR′+aug and B1-XGB′+aug
  (seed 42), and B2, the graph alone, **circular on DDXPlus (R-12)**.
- The data are the validation split, 33,963 patients, on EXP-019's masks (both digests
  reproduced). Every overall figure reproduces EXP-019's and EXP-021's (EXP-023).
- 22 figures in `docs/figures/`: a reliability diagram per system and per condition, a confusion
  heatmap per system and level, and top-1 and top-3 recall per condition
  (e.g. `exp023_confusion_b1_dl_prime_aug_25.svg`, `exp023_reliability_b1_dl_prime_aug.svg`).

**Table 6.26 — Top-1, must-not-miss recall@3 and the raw scores' top-label ECE, at 100% / 50% / 25%
of the history** (EXP-023; validation).

| System | Top-1 | Must-not-miss recall@3 | Top-label ECE |
|---|---|---|---|
| B1-DL′+aug | 0.9985 [0.9980, 0.9989] / 0.9865 [0.9854, 0.9877] / 0.9487 [0.9464, 0.9509] | 1.0000 / 1.0000 / 0.9981 [0.9973, 0.9987] | 0.0003 [0.0001, 0.0008] / 0.0024 [0.0016, 0.0036] / 0.0047 [0.0034, 0.0068] |
| B1-LR′+aug | 0.9983 [0.9978, 0.9987] / 0.9856 [0.9844, 0.9868] / 0.9459 [0.9436, 0.9482] | 1.0000 / 1.0000 / 0.9975 [0.9967, 0.9982] | 0.0002 [0.0001, 0.0007] / 0.0015 [0.0011, 0.0029] / 0.0110 [0.0092, 0.0131] |
| B1-XGB′+aug | 0.9985 [0.9980, 0.9989] / 0.9865 [0.9853, 0.9878] / 0.9483 [0.9460, 0.9504] | 1.0000 / 1.0000 / 0.9978 [0.9970, 0.9984] | 0.0002 [0.0000, 0.0006] / 0.0029 [0.0020, 0.0040] / 0.0057 [0.0049, 0.0081] |
| B2 (circular) | 0.9234 [0.9206, 0.9259] / 0.8666 [0.8629, 0.8702] / 0.7687 [0.7640, 0.7733] | 0.9998 [0.9995, 1.0000] / 0.9851 [0.9832, 0.9869] / 0.9350 [0.9309, 0.9386] | 0.0379 [0.0364, 0.0393] / 0.0358 [0.0340, 0.0382] / 0.0295 [0.0274, 0.0324] |

**Findings** (EXP-023).

1. **The three ML components make the same errors, at nearly the same rates.** At every level the
   most frequent top-1 error is unstable angina ranked below stable angina.
   - At 25%, of the 2,748 unstable angina patients: 610 for the transformer (0.222 [0.206, 0.238]),
     620 for the logistic regression (0.226 [0.210, 0.241]) and 608 for XGBoost (0.221 [0.205,
     0.237]).
   - Next come unstable angina ranked below MI (transformer 0.072 [0.062, 0.082]) and MI below
     stable angina (0.055 [0.047, 0.063]).
   - These are the anginas DDXPlus describes with nested evidence (EXP-004's only errors, and the
     graph's blind spot, §6.5). What separates them clinically (rest pain, a crescendo pattern, the
     troponin) is what "what to ask next" asks for (§4.14.1).
2. **Must-not-miss misses at 25% are few, and most are at rank 4.** Of 16,943 must-not-miss
   patients, the transformer misses 33 (0.0019 [0.0014, 0.0027]), 19 of them at rank 4; the
   logistic regression 43 (0.0025 [0.0018, 0.0033]); XGBoost 38 (0.0022 [0.0015, 0.0030]). No ML
   component misses one at 100% or 50%.
3. **Errors appear as the history shortens, and mostly persist.** The masks are nested, so a 25%
   history lies inside its 50% one. The transformer's top-1 errors grow 51 → 457 → 1,742; of its 457
   errors at 50%, 392 persist at 25% and 65 resolve.
4. **The raw scores of the three ML components are calibrated against DDXPlus at every level**
   (top-label ECE ≤ 0.011). For the transformer and the logistic regression this repeats EXP-007's
   finding that they stay raw (§6.9); it is not a new test, and it does not make a hand-written
   case's near-certainty a probability (EXP-018).
5. **B2, the graph alone (circular, R-12), has the expected profile.** 98% of stable angina patients
   go to unstable angina at 100% (0.981 [0.976, 0.986]), the tie rule's doing (EXP-005). At 25% it
   misses 6.5% of must-not-miss patients (0.0650 [0.0612, 0.0686]), myocarditis worst (0.164
   [0.145, 0.183]): all 253 of those patients get MI ranked first. This is why the graph is fused
   rather than used alone; it also does not measure what the graph's knowledge adds, which is aortic
   dissection, and DDXPlus has no dissection patient.

**Limits.** Validation figures, on a synthetic dataset whose generator explains much of the pattern
(R-16); random masks are not a clinician's questions; and the components alone: the fused system's
errors wait for E-2 and EXP-006 [pending: EXP-006 fusion sweep].

## ~~6.14~~ 6.15 The test split: ablations and hypothesis verdicts

[pending: EXP-012, test split]

The test split is opened once, after the freeze (PROGRESS §5: planned 2026-11-06 to 2026-11-07), and
every ablation of Table 5.4 is scored on it at all three levels. Table 6.27 is the frame the verdicts
will fill.

**Table 6.27 — Hypothesis verdicts** (to be filled from EXP-012 only).

| Hypothesis | Verdict (test split) | For how many of the five seeds | What validation already shows |
|---|---|---|---|
| H1 (100%, vs B1-XGB; B1-LR beside) | [pending: EXP-012, test split] | [pending] | Cannot be supported: B1 saturates top-3, and MRR leaves 0.0008 (R-16, §6.2) |
| H1-R (50% and 25%) | [pending: EXP-012, test split] | [pending] | As approved, cannot be met as a whole: the `+aug` comparators are at or near ceiling at 50% (§5.9); E-2 open |
| H2 (100%) | [pending: EXP-012, test split] | [pending] | Cannot be supported: B1's must-not-miss recall@3 is 1.000 (§6.2) |
| H2-R (50% and 25%) | [pending: EXP-012, test split] | [pending] | As H1-R; and the red flags collapse at reduced evidence (§6.6) |
| H3 | [pending: LLM baselines B3/B4, EXP-009/010] | — | Not measured |
| H4 (A1; A1b beside) | [pending: EXP-012, test split] | [pending] | Likely to fail at 100% for the same reason as H1 (R-16) |
| H5 (A0's fused score, 100%) | [pending: EXP-007 stage 2] [pending: EXP-012, test split] | — | The components are calibrated raw on DDXPlus (§6.9); the fused score is not yet calibrated |
| H6 (B1-DL+aug vs B1-LR+aug and B1-XGB+aug, 50% and 25%) | [pending: EXP-012, test split] | [pending] | At 25% the transformer's MRR is above B1-LR+aug's on validation (§6.8); at 50% the comparators are at or near ceiling (§5.9); the B1-XGB+aug comparison is pending |
| Minimum / Target / Stretch, and the reduced-evidence rows | [pending: EXP-012, test split] | — | The Target is out of reach at 100% (R-16); the reduced rows as for H1-R |
