# 6 · Results

> **Which split is which.** Sections 6.2 to 6.14 are **development results**: every figure in them
> comes from the DDXPlus **validation** split (33,963 patients, 16,943 of them with a must-not-miss
> condition) or from the four hand-written golden cases, and they chose the components and the
> fusion weight. **They are not the study's verdicts.** The verdicts are in **§6.15**, from the test
> split (34,657 patients, 16,972 of them with a must-not-miss condition), which was opened **once**,
> on 2026-11-09, after the code was frozen; every figure there is marked `[TEST]`. The two agree
> closely, and no verdict changed between them.
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

## 6.1 What this chapter contains

| Section | Experiment(s) | Split | Status |
|---|---|---|---|
| 6.2 B0 and B1 on full histories; saturation (R-16) | EXP-003, EXP-004 | validation | Done |
| 6.3 B1 on partial histories: the diagnostic (R-18) | EXP-017 | validation | Done |
| 6.4 Masked training copies and the "asked" channel, at three levels | EXP-018, EXP-019 | validation | Done |
| 6.5 The graph alone (circular) | EXP-005, EXP-015, EXP-016, EXP-019 | validation | Done |
| 6.6 The red-flag rules, and their collapse at reduced evidence | EXP-008, EXP-019 | validation | Done |
| 6.7 Seeds 43–46 of the classical models | EXP-020 | validation | Done |
| 6.8 The transformer B1-DL and the safety floor D-13 | EXP-021 | validation | Done, every seed and the full comparison table (2026-10-14) |
| 6.9 Calibration | EXP-007 | validation | Done: stage 1, the transformer arms and stage 2 (2026-11-08); no calibrator ships (decided 2026-11-09) |
| 6.10 Complexity and latency | EXP-022 | validation rows, golden cases | Done |
| 6.11 The golden cases | EXP-005, EXP-006, EXP-018, EXP-021 | hand-written | Done |
| 6.12 Fusion | EXP-006 | validation | Done (2026-11-07): α chosen; fusion adds nothing measurable |
| 6.13 Evidence retrieval and the language-model trial | EXP-011, EXP-009, EXP-010 | none; 50 validation patients | Done as far as it went: the checkpoint failed, and the evidence's relevance was never judged |
| 6.14 Where each component goes wrong, by condition | EXP-023 | validation | Done |
| 6.15 The test split: ablations and hypothesis verdicts | EXP-012 | **test, opened once** | Done (2026-11-09) |

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
   way. It is why amendment 3 added the reduced-evidence levels (§5.4). The test split confirmed
   it: at 100% every comparator misses 0 of 34,657 patients in the top 3 (§6.15).
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
7. **The test split shows the same layer.** Red-flag sensitivity there is 0.8285 [0.8229, 0.8344] at
   100%, 0.3318 [0.3245, 0.3386] at 50% and 0.1369 [0.1317, 0.1420] at 25%, and the flags lower
   top-1 at full evidence because they rank their conditions first even when the flag names the
   wrong one (§6.15, EXP-012 `[TEST]`).

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
   25%. **These are validation figures; only the test read counts** for H6, and there H6 is **not
   supported** (§6.15).
3. **The comparators' figures equal EXP-019's at seed 42.** *(Correction: a check of the cloud bundle
   first quoted B1-LR′+aug's 25% figures as 0.9451, 0.9707, 0.9976 and 0.0005; those were seed 43's,
   EXP-020. Seed 42's, 0.9459, 0.9710, 0.9975 and 0.0004, are used here; EXP-021.)*
4. **The safety targets hold at every seed** on the cloud's own figures: at 25%, must-not-miss
   recall@3 ≥ 0.9969 and the dangerous rate ≤ 0.0007, for both arms; seeds 43–46 stay within 0.003
   of seed 42 on every figure at 25% (EXP-021) †.
5. **The no-findings probe:** B1-DL′+aug answers PSVT at 0.38 and B1-DL+aug atrial fibrillation at
   0.39–0.41 †. That is EXP-018's pattern: no longer certain, not gone (EXP-021).
6. **Disclosed:** these validation figures were read on 2026-10-13, before E-2 was answered.
   E-2's recommendation, written without any deep result, was the answer taken on 2026-11-07
   (amendment 3b, §5.9); the choice of 25% as the only level at which H6 is judged is a choice of
   the level at which these figures already show a small gain (`docs/05` §9, amendment 3b;
   EXP-021).

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
    figures: H6 is judged on the test split with seed 42's model, with every seed's figures beside
    it, and there it is not supported (§6.15).

## 6.9 Calibration (EXP-007)

EXP-007 was pre-registered on 2026-10-13 before any calibrator was fitted on project data (§5.10).
Stage 1 calibrated only systems whose validation rankings EXP-019 had already published, so no fused
or deep result was read; the transformer arms followed after D-13, by the same rule. **Stage 2,
A0′'s fused score, on which H5 is judged, ran on 2026-11-08, after EXP-006 had chosen α** (below).
**No calibrator ships:** the owner decided on 2026-11-09 to keep `calibration.file` null, so the
system shows ranks only and no probability.

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
6. **H5 is not judged here.** It concerns A0's fused score, at 100%, and is judged on the test
   split (§6.15), on the raw fused score, because no calibrator ships (below).

### 6.9.1 Stage 2: the fused score of A0′ (EXP-007, 2026-11-08)

Stage 2 was pre-registered with stage 1 and run on the laptop CPU, where the owner placed the
calibrators (D-7, 2026-10-12): 150 s, both mask digests reproduced, the component B1-DL′+aug at
EXP-006's α = 0.05 and the real graph with BODHI-S. The rule chose **temperature scaling** (the
lowest mean Brier score over the three levels: 0.0338, against isotonic's 0.0342, Platt's 0.0381 and
the raw score's 0.0365).

**Table 6.23 — A0′'s fused score, raw and calibrated** (EXP-007 stage 2; validation; the
calibrator fitted at 100% and applied unchanged at every level; top-label ECE).

| A0′ fused score | Brier 100% / 50% / 25% | ECE 100% | ECE 50% | ECE 25% |
|---|---|---|---|---|
| Raw (this is A5) | 0.0052 / 0.0258 / 0.0786 | 0.0327 [0.0324, 0.0329] | 0.0603 [0.0595, 0.0614] | 0.0773 [0.0757, 0.0793] |
| Temperature scaling (chosen) | 0.0030 / 0.0224 / 0.0759 | 0.0008 [0.0008, 0.0008] | 0.0051 [0.0041, 0.0062] | 0.0176 [0.0156, 0.0195] |

1. **Calibrated against DDXPlus, the raw fused score is already under H5's 0.10 at every level**, so
   the calibrator changes no verdict. It improves the probabilities and never the order.
2. **On the hand-written cases it does what the first stage warned of.** The first listed
   condition's probability goes, for GC-001 (infarction, flagged first while the model ranks
   unstable angina) from 0.012 to 0.001, for GC-002 from 0.936 to 0.998 and for GC-004 from 0.915
   to 0.997; GC-003's dissection, the graph's own, stays 0.677. A short hand-written history is
   near-certain under the model whatever the calibration (R-17).
3. **The decision (the owner, 2026-11-09, taking the recommendation): ranks only, and
   `calibration.file` stays null.** No probability is shown, because of point 2. The calibrator
   above is kept as a result and not deployed, so A5 (no calibration) is A0 exactly, and A0's ECE
   is the raw fused score's.
4. **A figure that differs between two entries.** For the same system on validation, EXP-006
   records the raw fused score's top-label ECE as 0.040 / 0.064 / 0.078 at 100 / 50 / 25% †, and
   stage 2 records 0.0327 / 0.0603 / 0.0773 (intervals in Table 6.23), a little lower. The log does
   not reconcile the two.
   The verdict against H5's 0.10 is the same under both.

## 6.10 Complexity and latency (EXP-022)

Measured on the owner's laptop CPU (Windows 11, 24 logical CPUs, Python 3.11.9, NumPy 1.26.4, one
BLAS thread); batch 1, the dashboard's case; 100 warm-up calls, then 1,000 timed calls, three passes
in shuffled order. Latency percentiles are not ratio metrics, so they carry no bootstrap interval;
the p99 and the maximum are reported instead.

**Table 6.24 — Each model's size and cost per patient** (EXP-022; baseline run of 2026-10-12, the
transformer untrained, since its cost is fixed by the architecture).

| Model | Parameters / trees | Multiply-adds per patient | On disk | Load | p50 | p95 | p99 | Max |
|---|---|---|---|---|---|---|---|---|
| B1-LR′+aug | 8,996 | 8,983 | 0.11 MB | 0.003 s | 0.009 ms | 0.010 ms | 0.011 ms | 0.03 ms |
| B1-XGB′+aug | 5,824 trees, 264,396 nodes (depth ≤ 6) | ≤ 6 comparisons per tree | 14.6 MB | 3.16 s | 0.40 ms | 0.43 ms | 0.48 ms | 21.7 ms |
| B1-DL′+aug (untrained) | 225,645 | 9,610,752 | 1.22 MB (JSON) | 0.005 s | 0.75 ms | 0.78 ms | 0.82 ms | 1.4 ms |

**The trained transformer** (re-run of 2026-10-13, after D-13): batch 1, one thread, p50 1.29 ms,
p95 1.38 ms, p99 1.52 ms, max 2.67 ms; 1.23 MB on disk. In the same run B1-LR′+aug's p95 was
0.020 ms and B1-XGB′+aug's 0.59 ms (EXP-022).

**Table 6.25 — The whole ranking through the pipeline** (EXP-022; the four golden cases, 300 timed
runs each; the NetworkX graph, the configured α, the template explainer, no retrieval, no LLM).

| Run | Ranker | p50 | p95 | Max |
|---|---|---|---|---|
| Baseline, 2026-10-12 | B1-LR′+aug | 0.91–1.10 ms | 0.94–1.13 ms | 26–31 ms |
| Baseline, 2026-10-12 | B1-DL′+aug (untrained) | 1.79–1.95 ms | 1.95–2.01 ms | 27–40 ms |
| Re-run, 2026-10-13, with the model's reasons | B1-LR′+aug | — | 2.0–2.8 ms | see note |
| Re-run, 2026-10-13, with the model's reasons | B1-DL′+aug (trained) | — | 10.3–12.6 ms | see note |

*Note:* the re-run records a maximum of 38–62 ms without saying which ranker it belongs to, and no
p50 (EXP-022) †.

**With the evidence bundle** (re-measure of 2026-11-08, with the trained transformer and the real
corpus, recorded in EXP-011's dated note): retrieval for one candidate takes p50 0.78 ms and p95
1.22 ms; the whole pipeline with evidence takes p95 4.9 ms against 0.9 ms without it, and p95 13.6
to 16.9 ms on the golden cases with the transformer, of which 2.5 to 3.3 ms is retrieval. Start-up
takes 1.8 s.

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
   the CPU only. The language-model explainer is outside these figures, and it does not ship
   (§6.13.2); the template explainer's time is inside them.
   Multithreaded BLAS made one patient cost about 7–18 ms instead of 0.78 ms, so batches of up to 64
   rows now hold BLAS to one thread.

## 6.11 The golden cases

The four hand-written cases (§3.6) have no ground truth beyond the clinical expectation, and their
input passes through the hand-curated case-to-token translation (R-17). They are the evidence that is
not circular.

**Table 6.26 — The golden cases, component by component.**

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
6. **The golden cases still hold at the configured weight.** EXP-006's sweep kept the golden gate at
   every α on its grid for all six components (§6.12). After α = 0.05 was written into the
   configuration on 2026-11-07, the cases were re-run on the real graph and the transformer and all
   held (`PROGRESS.md`, 2026-11-07); `tests/fixtures/golden_cases.yaml` is part of what the freeze
   fixed (§5.11).

## 6.12 Fusion

EXP-006's rule was fixed before any fused ranking of a validation patient was scored (§5.10), and its
golden gate was recorded in §6.11. The sweep ran on 2026-11-07, on validation only, once E-2 had been
answered (§5.9): the owner's Kaggle run of `notebooks/kaggle_exp012_rehearsal.ipynb`, 11 minutes for
the sweep, commit `46f76eb`, both mask digests reproduced. Its output is committed as
`docs/results/exp006_fusion.json`.

**Table 6.27 — The graph's weight α chosen for each ML component** (EXP-006; validation; one α for
all three levels). The rule's constraint is that the fused ranking's must-not-miss recall@3 is at
least the component's own at every level.

| Component | α chosen | Branch of the rule taken |
|---|---|---|
| B1-LR+aug | 0.35 | the highest mean MRR meeting the constraint |
| B1-XGB+aug | 0.45 | the highest mean MRR meeting the constraint |
| B1-LR′+aug | 0.25 | the highest mean MRR meeting the constraint |
| B1-XGB′+aug | 0.10 | the highest mean MRR meeting the constraint |
| B1-DL+aug (A0's component) | 0.30 | no α met the constraint, so the highest mean must-not-miss recall@3 |
| **B1-DL′+aug** (A0′'s component, the configured ranker) | **0.05** | no α met the constraint, so the highest mean must-not-miss recall@3 |

**Findings** (EXP-006).

1. **The golden gate held at every α on the grid for all six components**, so it removed none.
2. **The configured weight is the smallest on the grid.** The configured component is B1-DL′+aug, so
   `configs/config.yaml` reads `ml_weight` 0.95 and `kg_weight` 0.05.
3. **The objective is flat.** For B1-DL+aug the mean MRR over the three levels is 0.9852 from α = 0.05
   to α = 0.45 and falls only past 0.85 (0.9132 at α = 1, the graph alone) †, so the choice among
   small α rests on differences in the fourth decimal.
4. **Fusion adds nothing measurable on DDXPlus validation.** Table 6.28 compares the configured
   component, fused at α = 0.05, with the model alone.
5. **The graph's one contribution that DDXPlus can see is the cost of aortic dissection.** No DDXPlus
   patient has it, and it reaches the top 3 for 0.0% / 0.1% / 1.9% of patients at 100 / 50 / 25%
   evidence † and pushes no must-not-miss patient out of it. What fusion was designed for, a condition
   the model cannot score ranking first when the graph supports it, shows only on the hand-written
   GC-003 (§6.11).
6. **The raw fused score's top-label ECE** is 0.040 / 0.064 / 0.078 at 100 / 50 / 25% † (see also
   §6.9.1, whose stage 2 records slightly lower figures for the same system).
7. **Read this with R-12 and R-16.** DDXPlus draws each patient's evidence from their condition's list,
   so the model already saturates, and the graph's contribution is the one condition the model cannot
   score and the explanation, not accuracy.
8. **The test split gave the same answer** (§6.15): with the flags removed, A4 equals A1, the
   transformer alone, at 25% to the fourth decimal on MRR (0.9639 for both) and within 0.0001 on
   top-3. α was chosen on validation, so the fused validation figures are the optimistic ones; the
   test figures are the honest ones.
9. **What this chapter does not reproduce.** EXP-006 was pre-registered to report, at every α,
   Precision@3, how often dissection enters the top 3 and how many must-not-miss patients it pushes
   out; and, at the chosen α, every ranking and safety metric with and without red flags, McNemar's
   test and the paired MRR test of fused against its component and against B2. `docs/08` records the
   figures of Table 6.28 and point 5 and refers to the output file for the rest; those tables are in
   `docs/results/exp006_fusion.json` and are not copied here.

**Table 6.28 — B1-DL′+aug fused with the graph at α = 0.05, against the model alone** (EXP-006;
validation; the fused ranking alone, red flags off, as the rule scores it). The model-alone figures
at 25% are EXP-021's (Table 6.17).

| Level | MRR, fused | MRR, model alone | Must-not-miss recall@3, fused | Must-not-miss recall@3, model alone |
|---|---|---|---|---|
| 100% | 0.9992 [0.9990, 0.9994] | 0.9992 † | 1.0000 † | 1.0000 † |
| 50% | 0.9932 [0.9926, 0.9938] | 0.9932 † | 1.0000 † | 1.0000 † |
| 25% | 0.9727 [0.9715, 0.9739] | 0.9728 [0.9715, 0.9740] | 0.9978 [0.9971, 0.9985] | 0.9981 [0.9973, 0.9987] |

Paired on top-3, there are no discordant patients at 100% and 50%; at 25% the fused system is right
alone for 1 patient and the model alone for 7 (exact p = 0.070). The MRR difference at 25% is
−0.00007, and its interval includes 0 (EXP-006).

## 6.13 Evidence retrieval and the language-model trial

Two components of Phase 3 were measured here, and **neither measurement is a success**. The
language-model explainer failed its pre-registered checkpoint and does not ship (§6.13.2). The
evidence retrieval was built and costed, and **its relevance was never judged** (§6.13.1). Neither
affects the ranking: the
evidence is attached after ranking, a test checks that it never changes the order, and the template
explainer is what the dashboard shows.

### 6.13.1 Evidence retrieval (EXP-011)

**What was designed** (pre-registered 2026-10-14, before any corpus existed). Hybrid retrieval, BM25
and a dense search fused by reciprocal rank, over PMC Open Access, with 26 queries (one per
condition, and each golden case's three leading candidates). Two team members were to judge each
pooled passage 0 to 2, blind to the method. The metrics were Precision@3 and nDCG, each with its
interval, and the rule was that the hybrid stays unless BM25 alone or dense search alone is better
by a paired interval that excludes 0. A median overlap of the laptop's dense top 10 with the
full-sentence embedding's under 0.5 would count as a poor stand-in (EXP-011).

**What was done.** On 2026-11-09, before any judging, the load was lightened from 489 rows a judge to
159 (each method's top 3), with nDCG@3 deciding the rule. On 2026-11-10 the owner reported that the
team members had no time to judge even that. **No filled sheet exists, and the relevance of the
evidence shown beside a candidate is unjudged.**

**Table 6.29 — EXP-011: what was measured, and what was not** (EXP-011, dated notes of 2026-11-08
and 2026-11-10).

| Item | Result |
|---|---|
| Corpus (the owner's Kaggle run of 2026-11-07, commit `46f76eb`) | 2,067 articles and 26,190 passages from PMC Open Access; 446 articles skipped; 20 searches; embeddings from `S-PubMedBert-MS-MARCO` (768 dimensions, float16) |
| Licences | Every article carries a Creative Commons licence in its XML: 15 licence strings, the largest CC BY 4.0 (979 articles); **844 carry a non-commercial term**, shown with each excerpt. The screen hides 0 passages on the laptop |
| Fidelity of the laptop's query vector, over 56 pairs (4 golden cases × 14 conditions) | Cosine with the full-sentence embedding: median 0.962 (mean 0.963, range 0.950 to 0.984). Overlap of the dense top 10: **median 0.400** (mean 0.404, range 0.00 to 0.80). **Under the pre-registered 0.5, so a poor stand-in**: on average the dense half ranks four of the ten passages that embedding the whole query would, and no remedy keeps the model off the laptop for an arbitrary case |
| Cost on the laptop | Start-up 1.8 s; one candidate's retrieval p50 0.78 ms, p95 1.22 ms; the whole pipeline with evidence p95 4.9 ms against 0.9 ms without (§6.10) |
| Precision@3, nDCG@3, the paired comparison of the hybrid with each method, Cohen's κ | **Not computed: nothing was judged** |
| Treatment language reaching a judge | **Not counted: nothing was judged** |

**What follows.**

1. **The hybrid stays configured by default, not by a result.** The pre-registered rule can move it only
   on evidence, and there is none. The clause under which the team decides whether evidence stays in
   the prototype if Precision@3 is under 0.5 cannot be applied; evidence stays, with the disclosure
   below, and the team may still decide otherwise.
2. **The dashboard says so.** Its evidence caption states that the passages' relevance "has not been
   formally judged (EXP-011)" and asks the reader to take them as leads, not proof (tested). The
   evidence is about diagnosis only and is never used to rank.
3. **Treatment language is excluded by design, not by a measured result.** A sentence with treatment
   language is never stored, embedded or shown, and an article whose title names a treatment is
   dropped (§4.14.2). The screen is lexical, so it can miss a treatment written in words it does not
   list; the count of any that reached a judge was never made.
4. **The evidence is partly non-commercial** (risk R-20, `docs/07`): 844 of the 2,067 articles. The
   project is non-commercial, and each excerpt shows its licence.
5. **A later judgement would be disclosed.** If judges are found and the result favours BM25 or dense
   search alone, the change to `configs/` would come after the freeze and the test opening, and would
   be reported as made after them. It would change no ranking and no test result, since retrieval
   never ranks.

Closed world (R-13): only the 14 conditions have evidence. No result here is circular (R-12), since no
DDXPlus patient is involved, although the query texts do come from the crosswalk's labels.

### 6.13.2 The language-model trial (EXP-009, EXP-010)

**The design** (pre-registered 2026-10-21, before any model was called). One local model, `llama3.1:8b`
through Ollama (D-5), at temperature 0, seed 42, `num_predict` 512 and `num_ctx` 8192, on **50
validation patients** and the four golden cases (listed apart, never pooled). The 50 are the first 50
of the 33,963 in the order of a SHA-256 of their id, not stratified (digest of the ids
`022bdcfa4641cb81`); 29 of them have a must-not-miss condition. Four arms, 416 calls in all:

- **B3**, the model alone: the case and the 14 condition names, at 100%, 50% and 25%;
- **B3r**, B3 told to use only the facts it is given (a control), at 100%;
- **B4**, retrieval and the model: B3's prompt and the best passage for each condition, at 100%, 50% and 25%;
- **A0+LLM**, the grounded explainer: the pipeline's result in the prompt, at 100%.

**An unsupported claim** is a sentence of the explanation that traces to no source the arm was given;
the check is lexical, so a correct paraphrase fails, and the rate measures how closely an explanation
stays within its inputs, not whether it is correct. **H3's test** was fixed as four paired
differences, B3 − B4, B3 − A0+LLM, B3r − B4 and B3r − A0+LLM (the LLM-alone rate minus the grounded
rate, each with a paired 95% interval, judged on A0+LLM's raw answer before the guardrails), and H3 is
supported only if all four intervals lie above 0. **The checkpoint of D-5** (2026-11-03) keeps the
template explainer, and reports B3 and B4 as not carried into the evaluation, if any of four things
holds: A0+LLM's unsupported-claim rate is not under 0.20; more than 10% of its calls failed; the run
did not finish; or anything the guardrails delivered held an unsupported claim or treatment language.

**The run.** The owner's Kaggle run of `notebooks/kaggle_llm_explain.ipynb` on 2026-11-07, 47 minutes
on two T4 GPUs, commit `046f9ac`; Ollama 0.40.15, the model's digest `46e0c10c…`, Q4_K_M. All four
arms ran, and 0 calls failed in any arm.

**Table 6.30 — Unsupported-claim rates** (EXP-009/010; validation; 50 patients; 100% evidence; the
claim check is lexical).

| Arm | Claims | Unsupported | Rate [95% interval] |
|---|---|---|---|
| B3, the model alone | 247 | 247 | 1.000 [1.000, 1.000] |
| B3r, told to use only the facts it is given | 248 | 248 | 1.000 [1.000, 1.000] |
| B4, retrieval and the model | 244 | 243 | 0.996 [0.984, 1.000] |
| A0+LLM, raw answer, before the guardrails | 220 | 163 | **0.741 [0.663, 0.803]** |

**Findings.**

1. **D-5's checkpoint fails**, on its first rule only: A0+LLM's unsupported-claim rate is 0.741
   [0.663, 0.803] (163 of 220 claims), not under 0.20. Rules 2 to 4 held. So **the template explainer
   stays** (`llm.enabled` is false), and **B3 and B4 are reported as not carried into the
   evaluation**, with this entry as the record of what they scored.
2. **H3 is not supported.** Two of its four comparisons pass: B3 − A0+LLM and B3r − A0+LLM are each
   +0.259 [+0.191, +0.333]. B3 − B4 and B3r − B4 are each +0.004 [+0.000, +0.012], whose interval
   touches 0.
3. **The guardrails did their job.** In all 50 cases the grounded explainer's answer failed guardrails
   v1 (1 to 11 unsupported claims each), so the template's text was delivered and `llm` was reported
   as degraded: 50 of 50 fell back, 0 delivered an unsupported claim, 0 delivered treatment language,
   0 delivered an answer that was not grounded. **The model's prose never reached a user.**
4. **Citations did not make the claims supported.** B4's text cites passages (77 citations, none
   invented), but all 72 of its cited sentences, and all 22 of A0+LLM's, were unsupported by the check.
5. **A language model asked to rank the 14 conditions ranks worse than the pipeline, and retrieved
   passages made it worse** (Table 6.31).

**Table 6.31 — The ranking each arm produced** (EXP-009/010; validation; 50 patients; where three
levels are given they are 100 / 50 / 25%; intervals only where the log gives them). A0+LLM's ranking
is the pipeline's through the case-to-token inversion (R-17) and **is not EXP-012's A0**.

| Arm | Evidence | Top-1 | Top-3 | MRR |
|---|---|---|---|---|
| A0+LLM | 100% | 0.80 [0.68, 0.90] | 1.00 [1.00, 1.00] | 0.88 [0.82, 0.94] |
| B3 | 100 / 50 / 25% | 0.44 / 0.38 / 0.28 | 0.78 [0.66, 0.90] / 0.66 / 0.62 | 0.62 [0.52, 0.72] / 0.52 / 0.43 |
| B3r | 100% | 0.50 [0.36, 0.64] | 0.84 [0.74, 0.94] | 0.67 [0.57, 0.76] |
| B4 | 100 / 50 / 25% | 0.20 / 0.22 / 0.24 | 0.58 [0.44, 0.72] / 0.46 / 0.42 | 0.40 [0.31, 0.50] / 0.38 / 0.33 |

**Limits of this trial.**

- Fifty patients give wide intervals, a condition having 2 to 6 of them, and the sample is drawn by hash
  order, not stratified; no per-condition figure is reported from it.
- The claim check is lexical (above). If the rate were judged to come from paraphrase alone, the only
  route the pre-registration allows is a new, pre-registered EXP-009b on fresh patients; none is
  planned, and the first result stays reported.
- One model, one run, one temperature; the run-to-run variance of the model is not measured. The
  pipeline's graph on Kaggle includes BODHI-S's enrichment.
- **The ratings were not made.** The 150-item rating sheet (reasoning validity, evidence relevance,
  clarity, 1 to 5, by two team members) was not rated, by the owner's decision of 2026-11-09, so no
  rated figure and no clinician-reviewed explainability exists.
- Nothing here was run on the test split, and no language-model system was scored there.

## 6.14 Where each component goes wrong, by condition (EXP-023)

**What it is.** A descriptive analysis of the four components alone, at 100%, 50% and 25% of the
history: errors and calibration, by condition. It is not a test.
- Nothing is trained, tuned, fitted or chosen, and no figure feeds a decision. It was not
  pre-registered, has no hypothesis and no verdict, and none of its figures may be quoted as a test
  of H1–H6.
- **Never a fused ranking:** EXP-023 ran on 2026-10-14, before E-2 was answered and before
  EXP-006's sweep, so it reads no fused ranking. The fused system's figures by condition, on the
  test split, are in §6.15.
- The components are B1-DL′+aug (the configured ranker, seed 42), B1-LR′+aug and B1-XGB′+aug
  (seed 42), and B2, the graph alone, **circular on DDXPlus (R-12)**.
- The data are the validation split, 33,963 patients, on EXP-019's masks (both digests
  reproduced). Every overall figure reproduces EXP-019's and EXP-021's (EXP-023).
- 22 figures in `docs/figures/`: a reliability diagram per system and per condition, a confusion
  heatmap per system and level, and top-1 and top-3 recall per condition
  (e.g. `exp023_confusion_b1_dl_prime_aug_25.svg`, `exp023_reliability_b1_dl_prime_aug.svg`).

**Table 6.32 — Top-1, must-not-miss recall@3 and the raw scores' top-label ECE, at 100% / 50% / 25%
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
by-condition figures, on the test split, are in §6.15.

## 6.15 The test split: ablations and hypothesis verdicts

**Every figure in this section is `[TEST]`**: the DDXPlus test split, 34,657 patients in scope, opened
**once**, on 2026-11-09, by the owner's Kaggle run, after the code was frozen (§5.11). Seed 42 unless a
seed is named; each figure carries its evidence level, its model label and its 95% bootstrap interval
(1,000 resamples, seed 42). **Nothing was re-run, re-tuned or looked at twice: the verdicts are what
the pre-registered rules gave** (EXP-012, results recorded 2026-11-10). The patients are synthetic,
the system is closed-world (R-13), and the graph's figures on them are circular (R-12). Of the
seven hypotheses judged on the test split (H1, H2, H1-R, H2-R, H4, H5 and H6), **five are not
supported** (H1, H2, H1-R, H2-R and H6).

### 6.15.1 How the run is known to be valid

- **The code that ran.** `notebooks/kaggle_exp012_test.ipynb`, at commit `f644faa`. The freeze tag
  `v0.9-freeze` (`808bceb`) is an ancestor of it. The files changed since the freeze are
  `PROGRESS.md`, `docs/08`, `docs/11`, `docs/results/exp006_fusion.json`, the notebook and
  `tests/test_notebooks.py`: **nothing under `configs/`, `src/` or `tests/fixtures/`.**
- **The masks.** Graph fingerprint `de349201157a`. **Both mask digests reproduced before any label
  was read**: `0136f67f1ca6542f` and `795acf2689f1415e`, the two recorded in `docs/08` on 2026-11-09
  before anything was scored (§5.4).
- **One opening**, recorded 2026-11-09T08:07:22Z at `f644faa`, with no `reopen_because`. The laptop
  recorded none: it built the parquet and drew the masks, and scored nothing. The result file's
  SHA-256 begins `f4100057ef462b94`, as the run's status file says.
- **Valid as a result.** `valid_as_a_result` is true: 34,657 patients, 16,972 of them with a
  must-not-miss condition; seeds 42 to 46; 102 comparisons; the reading of E-2 in force, `25-only`
  (§5.9). Realised sizes: 21.85 / 11.83 / 6.68 tokens and 15.43 / 8.61 / 5.09 questions a patient at
  100 / 50 / 25%. Scoring took 2,797 s (Python 3.13.18, numpy 2.1.3, pandas 2.3.3, xgboost 2.1.4, the
  same Kaggle environment as the rehearsal, which reproduced 1,944 of 1,944 laptop figures).
- **The weights.** α for each component is EXP-006's: B1-DL+aug 0.30 (A0) and B1-DL′+aug 0.05 (A0′);
  the plain B1-XGB and B1-LR are not EXP-006 components and take 0.5 in A7 only.
- The result files are in `data/interim/exp012_test/` and are never committed.

### 6.15.2 The rehearsal on validation that preceded it

The rehearsal (the owner's Kaggle run of 2026-11-07, 45 minutes, commit `46f76eb`, all five seeds,
the whole validation split, EXP-006's α, both validation digests reproduced) had to reproduce what
earlier experiments recorded **exactly**, or a mismatch would be a bug. `scripts/check_rehearsal.py`
compared 1,944 figures with EXP-019, EXP-021 and EXP-006: **1,944 were exact, none different.**
Its verdicts, on validation, under the same reading, were as follows.

**Table 6.33 — The verdicts on validation** (EXP-012 rehearsal; validation; seed 42; reading `25-only`).

| Claim | A0 (B1-DL+aug) | A0′ (B1-DL′+aug) |
|---|---|---|
| H1, H2 (100%) | not supported | not supported |
| H1-R, H2-R (25%) | not supported | not supported |
| H6 (the deep ranker against the models trained like it) | not supported | not supported |
| H4 (removing the graph hurts must-not-miss recall more than top-1; point estimate) | supported *(see §6.15.4: hollow)* | supported *(see §6.15.4: hollow)* |
| H5 (ECE below 0.10 at 100%) | supported | supported |
| Minimum | met | met |
| Target, 100% | not met (ties at 1.000, R-16; unsupported-claim rate not assessed) | not met (same) |
| Stretch, 100% | not met | not met |
| Target, reduced evidence | not met | met |
| Stretch, reduced evidence | not met | not met |

At 100% the comparators sit at 1.000 (R-16), and at 25% the fused system is level with its
component (§6.12), so no comparison against the models trained like it can be won.

### 6.15.3 The verdicts

**Table 6.34 — Hypothesis verdicts** (EXP-012 `[TEST]`; seed 42; the reading in force, `25-only`,
amendment 3b). *They are the same as the rehearsal's on validation, row for row: no verdict changed
between validation and test.*

| Claim | A0 (B1-DL+aug) | A0′ (B1-DL′+aug) | At how many of seeds 42–46 |
|---|---|---|---|
| H1, H2 (100%) | not supported | not supported | 0 of 5 |
| H1-R, H2-R (25%) | not supported | not supported | 0 of 5 |
| H3 (grounded explanation has the lower unsupported-claim rate) | not supported, by EXP-009's four-pair rule on 50 validation patients; **not carried to the test split** (§6.13.2) | — | — |
| H4 (the graph and flags removed: point estimates) | supported, **but see below** | supported, **but see below** | 5 of 5 |
| H5 (top-label ECE below 0.10 at 100%) | supported | supported | 5 of 5 |
| H6 (the deep ranker against the models trained like it) | not supported | not supported | 0 of 5 |
| Minimum | met | met | 5 of 5 |
| Target, 100% | not met | not met | 0 of 5 |
| Stretch, 100% | not met | not met | 0 of 5 |
| Target, reduced evidence (25%) | **not met** | **met** | 0 of 5 / 5 of 5 |
| Stretch, reduced evidence | not met | not met | 0 of 5 |

Under the older reading (`both`, with the 50% comparisons counted) H1-R, H2-R and H6 fail with more
comparisons, and the reduced Target is not met by either arm; both readings are in the output file.

**Said plainly, in two places the table cannot.**

1. **H4's support is hollow.** The rule is satisfied, and it is measuring something other than the
   graph: removing the graph leaves must-not-miss recall@3 exactly where it was, and the contrast
   the rule reads is the red-flag layer lowering top-1 (§6.15.4). **H4 must not be quoted as
   evidence that the graph helps.**
2. **The reduced-evidence Target is met by one arm, on a handful of patients.** A0′ meets it, missing
   40 of 34,657 patients in the top 3 where the XGBoost trained like it misses 56; A0 does not,
   falling behind its XGBoost comparator by two patients (§6.15.4). It is not a robust win: the fused
   system equals its own component to within a patient.

### 6.15.4 What stands behind each verdict

- **H1, H2 at 100%: the comparators are perfect (R-16).** B1-XGB, B1-LR and both transformers miss
  **0 of 34,657** patients in the top 3 and 0 of 16,972 must-not-miss patients, so no test can pass
  against them. The one H1 check that passes is top-3 against B2, the graph alone (1.0000 against
  0.9992, p = 5.6 × 10⁻⁷), and B2 is circular (R-12). **MRR is lower than B1's**: A0 0.9157 against
  B1-XGB's 0.9994, a difference of −0.0837 [−0.0859, −0.0819]. The cause is the red-flag layer, which
  ranks flagged conditions first even when the flag names the wrong one: A1b, the model plus flags
  and no graph, has the same MRR, 0.9157; A1, the model alone, 0.9994; strict flag precision is 0.509.
- **H1-R, H2-R at 25%.** Both arms beat the original B1-XGB, B1-LR and B2 on every check. They do not
  beat the models trained on masked copies (`+aug`):
  - A0 against B1-XGB+aug is −0.0001 on top-3 (p = 0.883) and −0.0052 on MRR [−0.0062, −0.0044];
    against B1-DL+aug −0.0001 (p = 0.625) and −0.0055 [−0.0060, −0.0049]; against B1-LR+aug it is
    +0.0007 on top-3 (p = 0.0067) but −0.0033 on MRR [−0.0042, −0.0025].
  - A0′ against B1-XGB′+aug and B1-LR′+aug is ahead on top-3 (+0.0005, p = 0.045; +0.0012,
    p = 1.0 × 10⁻⁵) and behind on MRR (−0.0041 [−0.0051, −0.0030]; −0.0024 [−0.0033, −0.0014]); it
    ties B1-DL′+aug on top-3 (0.9988 both) and trails it on MRR (−0.0057 [−0.0063, −0.0051]).
  - Failed checks: A0 5 of 12 (H1-R) and 2 of 5 (H2-R); A0′ 4 of 12 and 2 of 5.
- **H6 at 25%.** The transformer beats the logistic regression trained like it on all three metrics,
  and is level with the XGBoost trained like it.
  - Unprimed: against B1-LR+aug top-3 +0.0008 (p = 0.0024), MRR +0.0022 [+0.0015, +0.0028],
    must-not-miss +0.0018 (p = 4.8 × 10⁻⁶); against B1-XGB+aug top-3 0.0000 (p = 0.883), MRR
    +0.0002 [−0.0005, +0.0009], must-not-miss +0.0003 (p = 0.458).
  - Primed: against B1-LR′+aug all three pass; against B1-XGB′+aug MRR passes (+0.0016 [+0.0007,
    +0.0025]) and top-3 (+0.0004, p = 0.064) and must-not-miss (+0.0005, p = 0.164) do not.
  - At 100% the models tie (a ceiling).
- **H4 is supported by the rule, and the rule is measuring something other than the graph.**
  Removing the graph and the flags (A1) leaves must-not-miss recall@3 exactly where it was (1.0000, a
  degradation of 0.0000), while **top-1 *rises* by 0.1642 [0.1606, 0.1685]** (A0 0.8346 to A1
  0.9987), because the flags no longer push their conditions first. The contrast the rule reads
  (+0.1642) is that rise. Against A1b, which keeps the flags and drops only the graph, both
  degradations are exactly 0.0000 and the contrast is 0: **the graph itself contributes nothing
  measurable.** The same is true of the validation rehearsal, whose H4 also reads "supported" (Table
  6.33 gives that verdict without this caveat). It holds at all five seeds.
- **H5.** A0's top-label ECE at 100% is 0.0099 [0.0094, 0.0103] and A0′'s 0.0398 [0.0394, 0.0402], raw,
  since no calibrator ships (`calibration.file` is null, §6.9.1). At 50% and 25%, descriptive: A0
  0.0355 [0.0342, 0.0368] and 0.0644 [0.0620, 0.0666]; A0′ 0.0648 [0.0640, 0.0659] and 0.0785
  [0.0766, 0.0806]. Hand-written cases remain near-certain (R-17).
- **Minimum.** Top-3 at 100%: A0 1.0000 against B0's 0.3061 [0.3013, 0.3109], also significant.
- **Target and Stretch at 100%.** Not met, because the comparators tie at 1.000 and a tie fails a
  point-estimate comparison (EXP-012, choice 9); ECE and must-not-miss recall pass. **Two criteria are
  not assessed, and are neither met nor passed.** The unsupported-claim rate was measured for the
  language-model explainer (0.741, which does not ship, §6.13.2), not for the template explainer that
  does, and nobody has done the clinician-reviewed explainability the Stretch also lists. (The
  runner's wording "EXP-009/010 not run" is stale: they ran.)
- **Target, reduced evidence: A0′ met, A0 not met, on a few patients.** At 25% A0′ misses 40 of
  34,657 in the top 3 (24 of 16,972 must-not-miss), B1-DL′+aug 41 (24), B1-XGB′+aug 56 (33) and
  B1-LR′+aug 80 (45); A0 misses 73 (32), B1-DL+aug 71 (31), B1-XGB+aug 71 (36) and B1-LR+aug 97 (62).
  The Target compares point estimates, so A0′ passes (against B1-XGB′+aug the must-not-miss
  difference, +0.0005, is not significant, p = 0.164) and A0 fails by **two patients** on top-3
  against B1-XGB+aug (−0.0001, p = 0.883). The Stretch needs every test to pass and neither arm
  does.

### 6.15.5 The systems at the three evidence levels

**Table 6.35 — Ranking and safety at 100%, 50% and 25%** (EXP-012 `[TEST]`; seed 42; 34,657 patients,
16,972 with a must-not-miss condition). B2 is circular (R-12).

| System | Evidence | Top-1 | Top-3 | MRR | Must-not-miss recall@3 |
|---|:--:|---|---|---|---|
| B0 (prevalence only) | any | 0.1062 [0.1030, 0.1094] | 0.3061 [0.3013, 0.3109] | 0.2906 [0.2878, 0.2934] | 0.2168 [0.2105, 0.2231] |
| B1-XGB | 100% | 0.9987 [0.9983, 0.9991] | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9992, 0.9996] | 1.0000 [1.0000, 1.0000] |
|  | 50% | 0.5744 [0.5697, 0.5800] | 0.9899 [0.9887, 0.9909] | 0.7795 [0.7771, 0.7825] | 0.9858 [0.9840, 0.9875] |
|  | 25% | 0.2482 [0.2439, 0.2524] | 0.9035 [0.9003, 0.9068] | 0.5789 [0.5763, 0.5816] | 0.8710 [0.8659, 0.8764] |
| B1-LR | 100% | 0.9986 [0.9982, 0.9990] | 1.0000 [1.0000, 1.0000] | 0.9993 [0.9991, 0.9995] | 1.0000 [1.0000, 1.0000] |
|  | 50% | 0.9436 [0.9412, 0.9460] | 0.9958 [0.9951, 0.9964] | 0.9690 [0.9677, 0.9704] | 0.9930 [0.9917, 0.9942] |
|  | 25% | 0.7652 [0.7604, 0.7697] | 0.9436 [0.9412, 0.9461] | 0.8575 [0.8546, 0.8602] | 0.9198 [0.9156, 0.9238] |
| B1-DL+aug (= A1) | 100% | 0.9987 [0.9983, 0.9991] | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9992, 0.9996] | 1.0000 [1.0000, 1.0000] |
|  | 50% | 0.9841 [0.9827, 0.9854] | 1.0000 [0.9999, 1.0000] | 0.9919 [0.9912, 0.9925] | 1.0000 [1.0000, 1.0000] |
|  | 25% | 0.9328 [0.9298, 0.9353] | 0.9980 [0.9975, 0.9984] | 0.9639 [0.9623, 0.9652] | 0.9982 [0.9975, 0.9988] |
| B1-DL′+aug (= A1′) | 100% | 0.9987 [0.9984, 0.9991] | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9992, 0.9996] | 1.0000 [1.0000, 1.0000] |
|  | 50% | 0.9875 [0.9862, 0.9885] | 1.0000 [0.9999, 1.0000] | 0.9936 [0.9930, 0.9942] | 0.9999 [0.9998, 1.0000] |
|  | 25% | 0.9481 [0.9456, 0.9503] | 0.9988 [0.9985, 0.9991] | 0.9723 [0.9711, 0.9735] | 0.9986 [0.9980, 0.9991] |
| B2 (graph alone; circular, R-12) | 100% | 0.9237 [0.9209, 0.9264] | 0.9992 [0.9989, 0.9995] | 0.9612 [0.9598, 0.9626] | 0.9998 [0.9995, 0.9999] |
|  | 50% | 0.8673 [0.8639, 0.8708] | 0.9854 [0.9841, 0.9867] | 0.9244 [0.9223, 0.9265] | 0.9872 [0.9854, 0.9890] |
|  | 25% | 0.7676 [0.7632, 0.7717] | 0.9389 [0.9365, 0.9415] | 0.8540 [0.8512, 0.8567] | 0.9405 [0.9371, 0.9442] |
| A1b (B1-DL+aug + flags, no graph) | 100% | 0.8346 [0.8303, 0.8382] | 1.0000 [1.0000, 1.0000] | 0.9157 [0.9135, 0.9175] | 1.0000 [1.0000, 1.0000] |
|  | 50% | 0.9291 [0.9262, 0.9318] | 1.0000 [0.9999, 1.0000] | 0.9639 [0.9624, 0.9653] | 1.0000 [1.0000, 1.0000] |
|  | 25% | 0.9223 [0.9192, 0.9249] | 0.9980 [0.9975, 0.9984] | 0.9585 [0.9568, 0.9599] | 0.9982 [0.9975, 0.9988] |
| A4 (A0 without the flags) | 100% | 0.9987 [0.9983, 0.9991] | 1.0000 [1.0000, 1.0000] | 0.9994 [0.9992, 0.9996] | 1.0000 [1.0000, 1.0000] |
|  | 50% | 0.9843 [0.9829, 0.9856] | 0.9999 [0.9999, 1.0000] | 0.9920 [0.9913, 0.9927] | 0.9999 [0.9998, 1.0000] |
|  | 25% | 0.9328 [0.9299, 0.9354] | 0.9979 [0.9974, 0.9983] | 0.9639 [0.9623, 0.9652] | 0.9981 [0.9974, 0.9987] |
| A0 (B1-DL+aug, fused, flagged first) | 100% | 0.8346 [0.8303, 0.8382] | 1.0000 [1.0000, 1.0000] | 0.9157 [0.9135, 0.9175] | 1.0000 [1.0000, 1.0000] |
|  | 50% | 0.9293 [0.9265, 0.9320] | 0.9999 [0.9999, 1.0000] | 0.9640 [0.9626, 0.9654] | 0.9999 [0.9998, 1.0000] |
|  | 25% | 0.9223 [0.9192, 0.9249] | 0.9979 [0.9974, 0.9983] | 0.9584 [0.9568, 0.9599] | 0.9981 [0.9974, 0.9987] |
| A0′ (B1-DL′+aug, fused, flagged first) | 100% | 0.8346 [0.8303, 0.8382] | 1.0000 [1.0000, 1.0000] | 0.9156 [0.9134, 0.9175] | 1.0000 [1.0000, 1.0000] |
|  | 50% | 0.9318 [0.9292, 0.9344] | 1.0000 [0.9999, 1.0000] | 0.9653 [0.9640, 0.9667] | 0.9999 [0.9998, 1.0000] |
|  | 25% | 0.9371 [0.9343, 0.9396] | 0.9988 [0.9985, 0.9992] | 0.9667 [0.9653, 0.9680] | 0.9986 [0.9980, 0.9991] |

**Not in this table.** The plain B1 variants trained on masked copies (B1-XGB+aug, B1-LR+aug and the
primed ones) appear only in the ceilings table (Table 6.37), because `docs/08` records no other
figure for them on the test split. A2 (the graph alone, flagged first) and A7 (the classical model
in the fused system, at either weight) are not reproduced in the record of the test results, so
this report does not quote them. A3, A5 and A6 are identical to A0 by construction (§5.6).

### 6.15.6 Safety, precision and the red-flag layer

**Table 6.36 — Safety, precision and the red-flag layer** (EXP-012 `[TEST]`; seed 42). The safety
targets are must-not-miss recall@3 ≥ 0.95 and a dangerous false-negative rate ≤ 0.02. Red-flag
precision here is the strict version (a flag counts only when it names the true condition); the A-7
version was not part of this output (validation: 0.848).

| System | Evidence | Dangerous false-negative rate | Precision@3 (`D_in`) | Recall@5 (`D_in`) | Recall@5 (full `D`) | Red-flag sensitivity | Red-flag precision (strict) |
|---|:--:|---|---|---|---|---|---|
| A0 | 100% | 0.0000 [0.0000, 0.0000] | 0.7430 [0.7402, 0.7459] | 0.5632 [0.5610, 0.5654] | 0.3303 [0.3284, 0.3322] | 0.8285 [0.8229, 0.8344] | 0.5089 [0.5047, 0.5128] |
|  | 50% | 0.0001 [0.0000, 0.0002] | 0.7470 [0.7444, 0.7498] | 0.5630 [0.5606, 0.5652] | 0.3308 [0.3289, 0.3327] | 0.3318 [0.3245, 0.3386] | 0.6469 [0.6379, 0.6550] |
|  | 25% | 0.0003 [0.0001, 0.0006] | 0.7400 [0.7373, 0.7428] | 0.5625 [0.5600, 0.5647] | 0.3311 [0.3291, 0.3331] | 0.1369 [0.1317, 0.1420] | 0.8220 [0.8084, 0.8352] |
| A0′ | 100% | 0.0000 [0.0000, 0.0000] | 0.7430 [0.7402, 0.7459] | 0.5632 [0.5610, 0.5654] | 0.3303 [0.3284, 0.3322] | 0.8285 [0.8229, 0.8344] | 0.5089 [0.5047, 0.5128] |
|  | 50% | 0.0000 [0.0000, 0.0000] | 0.7470 [0.7445, 0.7499] | 0.5630 [0.5607, 0.5652] | 0.3308 [0.3289, 0.3327] | 0.3318 [0.3245, 0.3386] | 0.6469 [0.6379, 0.6550] |
|  | 25% | 0.0002 [0.0000, 0.0004] | 0.7409 [0.7382, 0.7436] | 0.5630 [0.5605, 0.5652] | 0.3313 [0.3295, 0.3334] | 0.1369 [0.1317, 0.1420] | 0.8220 [0.8084, 0.8352] |
| B1-XGB | 25% | 0.0629 [0.0592, 0.0665] | 0.7048 [0.7018, 0.7079] | 0.5272 [0.5249, 0.5294] | 0.3005 [0.2990, 0.3023] | — | — |
| B1-LR | 25% | 0.0272 [0.0249, 0.0297] | 0.6877 [0.6849, 0.6904] | 0.5298 [0.5275, 0.5319] | 0.3034 [0.3015, 0.3051] | — | — |

**What the two tables say.**

1. **The original B1 fails the safety targets at 25%, and every model trained on masked copies meets
   them.** B1-XGB's must-not-miss recall@3 is 0.8710 [0.8659, 0.8764] with a dangerous
   false-negative rate of 0.0629 [0.0592, 0.0665]; B1-LR's 0.9198 [0.9156, 0.9238] and 0.0272
   [0.0249, 0.0297]. A0 reaches 0.9981 [0.9974, 0.9987] and 0.0003 [0.0001, 0.0006]; A0′ 0.9986
   [0.9980, 0.9991] and 0.0002 [0.0000, 0.0004]. This reproduces EXP-019 on a split nobody tuned on,
   and it is the finding that carries: **the value is in training on masked copies (`+aug`), not in
   the fusion.**
2. **The fusion adds nothing measurable.** A4 (A0 without the flags) equals A1 (the transformer
   alone) at 25% to the fourth decimal on MRR (0.9639 both) and within 0.0001 on top-3; this is
   EXP-006's finding, now on the test split (R-12, R-16). What the flags do is reorder: they lower
   top-1 and MRR at 100% (§6.15.4, H1) and raise nothing in recall.
3. **The red-flag layer is below its target and collapses with less evidence** (EXP-008, EXP-019):
   sensitivity 0.8285 [0.8229, 0.8344] at 100% against a target of 0.95, 0.3318 [0.3245, 0.3386] at
   50% and 0.1369 [0.1317, 0.1420] at 25%.
4. **Precision@3 and Recall@5 are capped by the open-world differentials** (D-8): Recall@5 on the
   full `D` is 0.33.

### 6.15.7 Ceilings

**Table 6.37 — Patients missed in the top 3, of 34,657, and must-not-miss patients missed, of 16,972,
in brackets** (EXP-012 `[TEST]`; seed 42). A comparison against a system that misses five or fewer
cannot reach significance.

| Evidence | B1-XGB | B1-LR | B1-XGB+aug | B1-LR+aug | B1-DL+aug | A0 | B1-XGB′+aug | B1-LR′+aug | B1-DL′+aug | A0′ | B2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100% | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 27 (4) |
| 50% | 351 (241) | 147 (119) | 2 (1) | 3 (2) | 1 (0) | 2 (1) | 1 (1) | 0 (0) | 1 (1) | 1 (1) | 507 (217) |
| 25% | 3,346 (2,190) | 1,954 (1,361) | 71 (36) | 97 (62) | 71 (31) | 73 (32) | 56 (33) | 80 (45) | 41 (24) | 40 (24) | 2,119 (1,009) |

So at 100% and 50% the `+aug` comparators are at a ceiling (R-16, as `docs/05` §1 foresaw from
validation), and **25% is the only level at which a comparison can still be won**. There the fused
system wins against the models not trained on masked copies, and against the ones that were it is
level or marginally ahead on top-3 and behind on MRR.

### 6.15.8 By true condition and by sex

A0′ at 25%; the full tables, for A0, A0′, B1-XGB, B1-LR, B1-DL+aug, B1-DL′+aug and B2 at every
level, are in the output file.

- **By condition.** Top-3 is at least 0.993 for every one of the 13 conditions (lowest: spontaneous
  pneumothorax 0.9933 [0.9888, 0.9970], 1,343 patients). Top-1 is lowest for **unstable angina,
  0.6896 [0.6726, 0.7059]** (2,880 patients), then myocarditis 0.8606 [0.8430, 0.8775] (1,478
  patients), spontaneous pneumothorax 0.8883 [0.8712, 0.9047] and stable angina 0.8927 [0.8793,
  0.9049] (2,386 patients): the unstable-below-stable-angina confusion that EXP-023 found on
  validation (§6.14).
- **By sex.** A0′ top-1 is 0.9375 [0.9338, 0.9410] for women (17,898 patients) and 0.9368 [0.9330,
  0.9408] for men (16,759). The intervals overlap, and **near-equal by-sex results are an artifact of
  DDXPlus's synthetic demographics, not evidence of fairness** (EXP-002).

**By age (validation, not test).** The five age bands were fixed in a dated note in `docs/08` before any
figure by age existed (0-17, 18-39, 40-59, 60-79, 80+; 4,093 / 12,336 / 10,133 / 6,022 / 1,379 validation
patients). The breakdown was run on **validation only**, seed 42, 588 s on the laptop CPU; an age breakdown on
the test split would need a second opening of it, so none was made. Table 6.38 gives the safety figure.

**Table 6.38 — Must-not-miss recall@3 at 25% of the history, by age band** (EXP-012 age breakdown;
validation; seed 42; the safety target is at least 0.95).

| Band | Patients | B1-XGB | B1-LR | B1-DL′+aug | A0 | A0′ |
|---|---:|---|---|---|---|---|
| 0-17 | 4,093 | 0.9198 [0.9069, 0.9332] | 0.9705 [0.9621, 0.9782] | 0.9987 [0.9968, 1.0000] | 0.9955 [0.9923, 0.9981] | 0.9987 [0.9968, 1.0000] |
| 18-39 | 12,336 | 0.8791 [0.8712, 0.8867] | 0.9227 [0.9164, 0.9285] | 0.9984 [0.9973, 0.9992] | 0.9973 [0.9959, 0.9986] | 0.9978 [0.9965, 0.9989] |
| 40-59 | 10,133 | 0.8648 [0.8549, 0.8744] | 0.9205 [0.9124, 0.9277] | 0.9971 [0.9956, 0.9985] | 0.9971 [0.9956, 0.9987] | 0.9971 [0.9956, 0.9985] |
| 60-79 | 6,022 | 0.8747 [0.8622, 0.8860] | 0.9178 [0.9078, 0.9271] | 0.9987 [0.9974, 0.9997] | 0.9984 [0.9968, 0.9997] | 0.9987 [0.9974, 0.9997] |
| 80+ | 1,379 | 0.8620 [0.8355, 0.8869] | 0.9134 [0.8928, 0.9339] | 0.9971 [0.9927, 1.0000] | 0.9971 [0.9927, 1.0000] | 0.9971 [0.9927, 1.0000] |

A0′ keeps must-not-miss recall@3 at 0.9971 or above at 25% in all five bands (80+: 0.9971 [0.9927, 1.0000],
the widest interval), and at 1.0000 in every band at 100% and 50%. The plain models do not meet the target
in most bands: B1-XGB is below 0.95 in all five and B1-LR in four of five. Top-1 is highest in the youngest
band at every level (A0′ at 25%: 0.9807 [0.9763, 0.9849] against 0.9260 to 0.9357 in the other bands); why is
not established here, and DDXPlus's ages and its condition mix by age are generated, so this is a property of
the generator and not a finding about children. **Near-equal results across age or sex are an artifact of
synthetic demographics, not evidence of fairness** (EXP-002). No verdict depends on this breakdown.

### 6.15.9 Across the five seeds

The models take seeds 42 to 46, and the seed-42 verdict is the verdict. At 25%, across the five
seeds (the log gives a standard deviation, not a bootstrap interval) †: A0 top-3 0.9976 (standard deviation 0.0002) and MRR 0.9577 (0.0008); A0′ top-3 0.9986 (0.0001) and MRR
0.9657 (0.0008). Every verdict that reads supported or met holds at 5 of 5 seeds, and every one that reads
not supported or not met holds at 0 of 5, so **no verdict is an accident of seed 42.**

### 6.15.10 What this means, and what stays with it

**In plain terms.**

- The pre-registered comparative claims are mostly **not supported**, as they were on validation. The
  fused system is not better than its own machine-learning component, and at 100% and 50% every
  comparator is already perfect or within a few patients (R-16: DDXPlus draws each patient's evidence
  from their condition's list).
- What the test split does support: the system meets the safety targets at 25% evidence
  (must-not-miss recall@3 0.998, dangerous false-negative rate 0.0003 or 0.0002) where the original
  B1 does not; its probabilities are calibrated at full evidence (ECE below 0.05 raw); training on
  masked copies is what matters; and a deep ranker beats a logistic regression trained the same way
  but only ties the XGBoost.
- **The honest costs:** the red-flag layer lowers top-1 and MRR at full evidence and has a
  sensitivity of 0.83 (0.14 at 25%); H4's support is an artifact of that reordering; the graph's
  contribution is not measurable on DDXPlus (circular, R-12).

**Limits that stay with these results.**

- **Synthetic, closed-world data (R-12, R-13).** Thirteen conditions and one generator, so the test
  split checks overfitting to validation, not clinical validity. No DDXPlus patient has aortic
  dissection, so the one condition only the graph and the rules can rank is shown by the hand-written
  GC-003 alone (§6.11).
- The masks are random, so they measure robustness to missing answers, not history-taking.
- α was chosen on validation.
- The unsupported-claim rate for the shipped explainer and the clinician-reviewed explainability are
  not assessed. No language-model system was run on the test split.
- By-sex parity is an artifact.
- The test split is a second sample from the same generator as validation. **It must not be scored
  again**: a second opening needs a stated reason (`--reopen-because`) and would be disclosed.
