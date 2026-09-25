# 08 — Experiment Log

**Owner:** P2 (maintained by whoever runs the experiment)

> **Log every run that produces a number — including failures.** The ablation table in the final
> report is assembled from this file. An experiment that was not logged did not happen, and
> re-running it at the end costs more than logging it now.

---

## How to use this log

1. Copy the template into a new entry at the **top** of §Entries (newest first).
2. Fill it in **when you run the experiment**, not later.
3. Record the MLflow run ID so parameters and artifacts are recoverable.
4. Write the **Interpretation** honestly. "Worse than baseline, unclear why" is a valid, useful entry.

**Rules**
- Never overwrite an entry; add a new one.
- Development results come from **validation**. Any test-set number must be tagged `[TEST]` and may
  only appear in Phase 4.
- Record the git commit — a result you cannot reproduce is not a result.

---

## Template

```markdown
### EXP-NNN — <short title>

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Author | P_ |
| Config / ablation | A0 / B1 / … |
| Split used | validation \| [TEST] |
| Git commit | `abc1234` |
| MLflow run | `<run_id>` |
| Seed | 42 |

**Question:** What is this run meant to answer?

**Setup:** Model, features, hyperparameters, data version, anything non-default.

**Results:**

| Metric | Value | 95% CI |
|---|---|---|
| Top-1 | | |
| Top-3 | | |
| MRR | | |
| Must-not-miss recall@3 | | |
| ECE | | |

**Interpretation:** What it means. Be honest about negative or confusing results.

**Next action:** What this implies for the next experiment.
```

---

## Entries

*(newest first — add above this line as experiments are run)*

### EXP-019 — Every system at 100%, 50% and 25% evidence: amendment 3's masks (validation split)

| Field | Value |
|---|---|
| Date | 2026-09-25 |
| Author | P4 (run by Claude) |
| Config / ablation | B0; B1-LR, B1-XGB (EXP-004); B1-LR+aug, B1-XGB+aug, B1-LR′+aug, B1-XGB′+aug (EXP-018); B2 (EXP-005); the red-flag layer (EXP-008) |
| Split used | validation, at 100%, 50% and 25% of each patient's evidence (`docs/05` §3.7) |
| Git commit | `0f47d67` recorded the masks, **before** any system was scored on them; the scores are committed with this part of the entry |
| MLflow run | — (scoring only; nothing is trained) |
| Seed | 42 (masks and bootstrap) |

**The masks, recorded before anything was scored on them** (`docs/05` §3.7, amendment 3, approved
2026-09-25). Drawn by `src/ml/evidence_masks.py` under the question rule, with NumPy 1.26.4, by
`scripts/evaluate_reduced_evidence.py --digest-only`:

| | Value |
|---|---|
| Mask digest (§3.7's definition) | **`e99a7a8fbf792675`**, the digest the proposal predicted from the masks alone |
| Asked-set digest (decision A-9: the same lines with the questions asked, in code order, in place of the kept tokens) | **`af3a357f6454838a`** |
| Realised size, 100% | 21.90 tokens, 15.43 questions per patient |
| Realised size, 50% | 11.87 tokens, 8.61 questions; 42.57 of the 84 questions asked |
| Realised size, 25% | 6.71 tokens, 5.10 questions; 21.99 of the 84 questions asked |

The scoring script refuses to run unless both digests reproduce. Nothing had been scored on these
masks when this part of the entry was committed (`0f47d67`).

**Results** (33,963 validate patients; 95% bootstrap intervals; every figure names its level
and model label, `docs/05` §8 rule 8; the amendment-3 disclosure is `docs/05` §9; **B2 is circular
on DDXPlus, R-12**; the system is closed-world, R-13; DDXPlus is synthetic). Nothing was trained;
the run takes 129–137 s on the laptop (`scripts/evaluate_reduced_evidence.py`, which writes
`data/interim/exp019_reduced_evidence.json`; an independent re-run reproduced it byte for byte). The
100% column reproduces EXP-003, EXP-004, EXP-005 and EXP-018 exactly. **Order of events:** git alone
cannot show that nothing was scored before `0f47d67` (the result file is overwritten by every run);
the session's transcript shows the first scoring run starting 9 s after that commit, with only a
digest check and an unmasked B2 run before it.

| top-1 accuracy | 100% | 50% | 25% |
|---|---|---|---|
| B0 | 0.110 [0.106, 0.113] | 0.110 [0.106, 0.113] | 0.110 [0.106, 0.113] |
| B1-LR | 0.998 [0.998, 0.999] | 0.947 [0.944, 0.949] | 0.766 [0.761, 0.770] |
| B1-XGB | 0.998 [0.998, 0.999] | 0.578 [0.573, 0.583] | 0.245 [0.240, 0.250] |
| B1-LR+aug | 0.998 [0.998, 0.999] | 0.982 [0.981, 0.984] | 0.931 [0.928, 0.934] |
| B1-XGB+aug | 0.998 [0.998, 0.999] | 0.983 [0.982, 0.985] | 0.933 [0.931, 0.936] |
| B1-LR′+aug | 0.998 [0.998, 0.999] | 0.986 [0.984, 0.987] | 0.946 [0.944, 0.948] |
| B1-XGB′+aug | 0.998 [0.998, 0.999] | 0.987 [0.985, 0.988] | 0.948 [0.946, 0.950] |
| B2 | 0.923 [0.921, 0.926] | 0.867 [0.863, 0.870] | 0.769 [0.764, 0.773] |

| top-3 accuracy | 100% | 50% | 25% |
|---|---|---|---|
| B0 | 0.306 [0.301, 0.311] | 0.306 [0.301, 0.311] | 0.306 [0.301, 0.311] |
| B1-LR | 1.000 [1.000, 1.000] | 0.997 [0.996, 0.997] | 0.945 [0.943, 0.947] |
| B1-XGB | 1.000 [1.000, 1.000] | 0.991 [0.990, 0.992] | 0.906 [0.903, 0.909] |
| B1-LR+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.996, 0.997] |
| B1-XGB+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.997, 0.998] |
| B1-LR′+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.998 [0.997, 0.998] |
| B1-XGB′+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.998 [0.998, 0.999] |
| B2 | 1.000 [0.999, 1.000] | 0.985 [0.984, 0.986] | 0.937 [0.935, 0.940] |

| MRR | 100% | 50% | 25% |
|---|---|---|---|
| B0 | 0.292 [0.289, 0.295] | 0.292 [0.289, 0.295] | 0.292 [0.289, 0.295] |
| B1-LR | 0.999 [0.999, 0.999] | 0.971 [0.970, 0.972] | 0.858 [0.855, 0.861] |
| B1-XGB | 0.999 [0.999, 0.999] | 0.782 [0.779, 0.784] | 0.578 [0.575, 0.581] |
| B1-LR+aug | 0.999 [0.999, 0.999] | 0.991 [0.990, 0.992] | 0.963 [0.961, 0.964] |
| B1-XGB+aug | 0.999 [0.999, 0.999] | 0.992 [0.991, 0.992] | 0.964 [0.963, 0.966] |
| B1-LR′+aug | 0.999 [0.999, 0.999] | 0.993 [0.992, 0.993] | 0.971 [0.970, 0.972] |
| B1-XGB′+aug | 0.999 [0.999, 0.999] | 0.993 [0.993, 0.994] | 0.973 [0.971, 0.974] |
| B2 | 0.961 [0.960, 0.963] | 0.924 [0.922, 0.926] | 0.854 [0.851, 0.857] |

| must-not-miss recall@3 | 100% | 50% | 25% |
|---|---|---|---|
| B0 | 0.220 [0.214, 0.226] | 0.220 [0.214, 0.226] | 0.220 [0.214, 0.226] |
| B1-LR | 1.000 [1.000, 1.000] | 0.994 [0.993, 0.995] | 0.925 [0.921, 0.929] |
| B1-XGB | 1.000 [1.000, 1.000] | 0.987 [0.985, 0.988] | 0.877 [0.872, 0.882] |
| B1-LR+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.996 [0.995, 0.997] |
| B1-XGB+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.996, 0.998] |
| B1-LR′+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.997, 0.998] |
| B1-XGB′+aug | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.998 [0.997, 0.998] |
| B2 | 1.000 [1.000, 1.000] | 0.985 [0.983, 0.987] | 0.935 [0.931, 0.939] |

| dangerous false-negative rate | 100% | 50% | 25% |
|---|---|---|---|
| B0 | 0.606 [0.599, 0.614] | 0.606 [0.599, 0.614] | 0.606 [0.599, 0.614] |
| B1-LR | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 0.024 [0.021, 0.026] |
| B1-XGB | 0.000 [0.000, 0.000] | 0.004 [0.003, 0.005] | 0.059 [0.056, 0.062] |
| B1-LR+aug | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.001] |
| B1-XGB+aug | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.001] |
| B1-LR′+aug | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] |
| B1-XGB′+aug | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| B2 | 0.000 [0.000, 0.000] | 0.007 [0.006, 0.008] | 0.039 [0.036, 0.042] |

| Precision@3 | 100% | 50% | 25% |
|---|---|---|---|
| B1-LR | 0.729 [0.726, 0.732] | 0.703 [0.700, 0.706] | 0.684 [0.682, 0.687] |
| B1-XGB | 0.764 [0.761, 0.767] | 0.725 [0.722, 0.728] | 0.703 [0.700, 0.706] |
| B1-LR+aug | 0.741 [0.738, 0.744] | 0.724 [0.722, 0.727] | 0.711 [0.708, 0.713] |
| B1-XGB+aug | 0.734 [0.731, 0.737] | 0.730 [0.727, 0.732] | 0.718 [0.715, 0.721] |
| B1-LR′+aug | 0.716 [0.714, 0.719] | 0.715 [0.712, 0.717] | 0.708 [0.705, 0.711] |
| B1-XGB′+aug | 0.744 [0.741, 0.747] | 0.726 [0.723, 0.729] | 0.718 [0.715, 0.721] |
| B2 | 0.756 [0.753, 0.759] | 0.747 [0.744, 0.750] | 0.733 [0.730, 0.736] |

| Recall@5 | 100% | 50% | 25% |
|---|---|---|---|
| B1-LR | 0.553 [0.551, 0.556] | 0.539 [0.536, 0.541] | 0.527 [0.525, 0.530] |
| B1-XGB | 0.569 [0.567, 0.572] | 0.549 [0.547, 0.551] | 0.525 [0.522, 0.527] |
| B1-LR+aug | 0.556 [0.553, 0.558] | 0.546 [0.543, 0.548] | 0.536 [0.533, 0.538] |
| B1-XGB+aug | 0.552 [0.550, 0.554] | 0.551 [0.548, 0.553] | 0.544 [0.542, 0.546] |
| B1-LR′+aug | 0.544 [0.542, 0.547] | 0.537 [0.535, 0.539] | 0.533 [0.531, 0.535] |
| B1-XGB′+aug | 0.568 [0.566, 0.570] | 0.553 [0.551, 0.556] | 0.547 [0.545, 0.549] |
| B2 | 0.570 [0.568, 0.572] | 0.562 [0.560, 0.564] | 0.556 [0.554, 0.559] |

| ECE of raw scores (10 bins) | 100% | 50% | 25% |
|---|---|---|---|
| B1-LR | 0.001 | 0.005 | 0.035 |
| B1-XGB | 0.000 | 0.310 | 0.665 |
| B1-LR+aug | 0.001 | 0.003 | 0.010 |
| B1-XGB+aug | 0.000 | 0.004 | 0.006 |
| B1-LR′+aug | 0.000 | 0.002 | 0.011 |
| B1-XGB′+aug | 0.000 | 0.003 | 0.006 |

**Paired comparisons** (McNemar on top-3, `docs/05` §6 as amended; at 100% every pair is 0 / 0):

| Level | A vs B (the change isolated) | Top-3, A only / B only, p | Must-not-miss patients only |
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

| Red-flag layer | 100% | 50% | 25% |
|---|---|---|---|
| Sensitivity (amendment 2, target ≥ 0.95) | 0.823 [0.818, 0.829] | 0.327 [0.320, 0.334] | 0.135 [0.130, 0.140] |
| Precision (A-7) | 0.848 [0.845, 0.852] | 0.890 [0.884, 0.896] | 0.931 [0.922, 0.941] |
| Precision (strict) | 0.512 [0.508, 0.516] | 0.650 [0.640, 0.659] | 0.822 [0.808, 0.836] |
| Patients without a must-not-miss condition flagged | 0.246 [0.240, 0.253] | 0.050 [0.046, 0.053] | 0.007 [0.006, 0.008] |

**Interpretation:**

1. **Under the approved rule, the original B1 fails the safety targets at 25% of the history.**
   B1-XGB's top-1 falls to 0.245, its must-not-miss recall@3 to 0.877 and its dangerous
   false-negative rate rises to 0.059, against targets of ≥ 0.95 and ≤ 0.02; its raw scores are
   badly calibrated there (ECE 0.665). B1-LR degrades less but still misses both targets
   (0.925, 0.024). **The mask rule matters here.** Under EXP-017's token rule B1-LR met the
   must-not-miss target at 25% (0.974), and B1-XGB reached 0.935; under the approved question rule
   they reach 0.925 and 0.877, although the question rule keeps slightly more evidence (6.71 tokens
   against 6.21). The proposal (§3) predicted only that the figures would differ, and expected the
   question rule to be the gentler one; dropping whole questions hurts more than dropping tokens.
   The rule was fixed and recorded before scoring (`0f47d67`); EXP-017 stays a diagnostic.
2. **Masked training copies repair almost all of it.** Every `+aug` model keeps top-3 ≥ 0.997 and
   must-not-miss recall@3 ≥ 0.996 at 25%, with a dangerous false-negative rate ≤ 0.001 and ECE ≤
   0.011 (B1-LR+aug's 0.9967 and 0.9958 round to those bounds). Against its unaugmented self, at
   25% each `+aug` model gets top-3 right where the plain model fails for 1,811 patients (LR) and
   3,138 (XGB), and the reverse for 47 and 33.
3. **The "asked" channel adds a small, significant gain at 25%** on top of the masked copies:
   top-1 0.931 → 0.946 (LR) and 0.933 → 0.948 (XGB), top-3 better for 79 patients against 42
   (LR, p = 0.001) and 52 against 22 (XGB, p = 0.0007); on must-not-miss patients alone the gain is
   significant for LR (50 / 22, p = 0.0015) but not for XGB (27 / 18, p = 0.23). At 50% it makes no
   significant difference. The ′ models' reduced-evidence input depends on decision A-9, still
   open; the asked-set digest is recorded above, so a change re-runs cleanly.
4. **Once both are trained on masked copies, LR and XGBoost are close.** At 25% the top-3 disagreement
   falls from 845 / 2,162 patients (plain) to 49 / 25 (`+aug`, XGBoost slightly better, p = 0.0075)
   and 44 / 27 (′+aug, p = 0.058). Most of EXP-017's gap came from the training rows, not
   the model family; a small XGBoost edge remains.
5. **R-16 returns at the reduced levels for the augmented B1.** At 25% the `+aug` variants already
   reach top-3 0.997–0.998 and must-not-miss recall@3 0.996–0.998, so a fusion or deep model has
   almost no room above them on those two metrics; top-1 (0.93–0.95), MRR (0.96–0.97), Precision@3
   and Recall@5 still leave room. Against the unaugmented B1, the room at 25% is large. **Which B1
   H1-R and the reduced-evidence Target compare against therefore decides them**; the question is
   in the errata proposal, raised after these results and said so.
6. **The red-flag layer collapses at reduced evidence.** Its sensitivity falls from 0.823 to 0.327
   at 50% and 0.135 at 25%. At 25% the masks keep about 28% of a patient's questions beyond the
   initial one, and most rules need two or three findings together, so the combination rules fall
   furthest (MI 0.907 → 0.025, myocarditis 0.641 → 0.014, pneumothorax 0.325 → 0.009). The one rule
   that can fire on a single finding (acute pulmonary edema, on paroxysmal nocturnal dyspnoea)
   keeps the most (0.958 → 0.319), with PE next (0.922 → 0.282). Its precision rises as it fires
   less (A-7: 0.848 → 0.931). On DDXPlus at
   reduced evidence, must-not-miss safety is carried by the ranker, not by the rules, which bears
   directly on H2-R ("fusion + red flags > ML-only").
7. **EXP-018's interpretation 4, narrowed.** EXP-018 called the retrained models "overconfident
   on short input". On masked validate patients at 25% their raw scores are calibrated (ECE
   0.006–0.011): on DDXPlus a short history often does settle the condition, and the models say so
   correctly. That is calibration against DDXPlus's own generator, so it cannot test EXP-018's point,
   which was about real histories: GC-001's classic presentation gets unstable angina 0.98–1.00
   though only a troponin can separate it from MI. The evidence for that concern is one
   hand-written case, whose input passes through the concept → token inversion (R-17). Both hold:
   calibrated on DDXPlus, and near-certain on hand-written input where certainty is not warranted.
8. **B2 (circular)** degrades between the two: at 25% top-1 0.769 and must-not-miss recall@3
   0.935, below the target. It has the best Precision@3 of any system at 50% and 25% and the best
   Recall@5 at every level (at 100% B1-XGB's Precision@3 is higher, 0.764 against 0.756, and its
   Recall@5 is 0.001 lower).
9. **The configured ranker changes** from B1-LR to **B1-LR′+aug** (`configs/config.yaml`): every
   hand-written case is a partial history, B1-LR misses both safety targets at 25% and B1-LR′+aug
   meets them, and it is the kind of model that sees a case's denials. The four golden cases hold
   with it, with red flags on and off. Two costs, stated: at 100% its Precision@3 is lower than the
   old B1-LR's (0.716 against 0.729); and B1-XGB′+aug is a little better on top-3 at 25% (44 / 27,
   p = 0.058) and on Precision@3 and Recall@5 at every level, so 2c may revisit the family. Its
   training copies used the A-9 rule, so A-9 now governs the configured ranker too.

**Next action:** the team: the errata (E-1), including which B1 the reduced-evidence claims compare
against; A-9. 2c: fuse with the configured ranker, knowing the red flags add little at reduced
evidence. The owner: place the training run for XGBoost's seeds 43–46 (§6).

---

### EXP-018 — B1 retrained for R-18: masked copies, with and without the "asked" channel (validation split, full evidence only)

| Field | Value |
|---|---|
| Date | 2026-09-25 (trained on Colab by the owner; checked on the laptop by Claude) |
| Author | P2 (run by the owner and Claude) |
| Config / ablation | B1-LR and B1-XGB, retrained as **+aug** (the B1 features, fingerprint `3a0d5a5e01d7f427`) and **′+aug** (with the "asked" channel, `ddd14019c66eff10`) |
| Split used | train (with one masked copy of each of its 255,900 patients) → validation, **full evidence only** |
| Git commit | `4583e91` (the job; run.json records no uncommitted changes) |
| MLflow run | — (the bundles are `models/nightingale_b1_asked/b1_aug/` and `.../b1_asked_aug/`, gitignored) |
| Seed | 42 |

**Question:** EXP-017 found B1 XGBoost answering atrial fibrillation to short input, with probability
1.000 to a patient with no findings, because the encoder could not tell an unasked question from a
denial (R-18). Do masked training copies (+aug) remove that, what does the "asked" channel (′) add,
and what does either cost at full evidence? **The decisive comparison, masked validate patients at
50% and 25%, is not in this entry:** it waits for the team to approve a mask rule (amendment 3).

**Setup:** `scripts/train_baselines.py --augment` and `--asked-channel --augment` on a Colab T4
(`notebooks/colab_b1_asked.ipynb`; Python 3.13, NumPy 2.1, scikit-learn 1.6, XGBoost 2.1.4). Each
train patient contributes its full record and one copy keeping a share drawn uniformly from
[0.1, 0.9] of its questions (`src/ml/evidence_masks.py`); the XGBoost holdout is split by patient.
The job took about 3½ minutes per variant after the download. On the laptop, all four models reload
and reproduce every validate metric to within 5 × 10⁻¹². Then two checks that need no mask: a patient
with no findings at all, and the four golden cases (5–13 tokens each after the inversion, R-17).

**Results, full-evidence validate** (95% bootstrap intervals; top-3 and must-not-miss recall@3 are
1.000 for every model, as for B1):

| Model | Top-1 | Precision@3 | Recall@5 |
|---|---|---|---|
| B1-LR (EXP-004) | 0.9983 [0.9978, 0.9987] | 0.7291 [0.7265, 0.7317] | 0.5533 [0.5508, 0.5556] |
| B1-LR+aug | 0.9982 [0.9977, 0.9986] | 0.7411 [0.7383, 0.7439] | 0.5555 [0.5531, 0.5577] |
| B1-LR′+aug | 0.9983 [0.9978, 0.9987] | 0.7163 [0.7138, 0.7191] | 0.5444 [0.5421, 0.5465] |
| B1-XGB (EXP-004) | 0.9985 [0.9980, 0.9989] | 0.7639 [0.7613, 0.7667] | 0.5694 [0.5668, 0.5718] |
| B1-XGB+aug | 0.9984 [0.9980, 0.9988] | 0.7340 [0.7313, 0.7369] | 0.5520 [0.5498, 0.5541] |
| B1-XGB′+aug | 0.9985 [0.9980, 0.9989] | 0.7441 [0.7413, 0.7470] | 0.5679 [0.5657, 0.5701] |

**A patient with no findings** (age 50; either sex gives the same answer):

| Model | Top answer, probability |
|---|---|
| B1-XGB (EXP-017) | atrial fibrillation, **1.000** |
| B1-LR+aug / B1-XGB+aug | PSVT, 0.43 / 0.28 |
| B1-LR′+aug / B1-XGB′+aug | PSVT, 0.23 / pericarditis, 0.22 |

**The golden cases, ML ranker alone** (probability of the top answer; the expected condition's rank):

| Case (expected) | B1-LR | B1-XGB | LR+aug | XGB+aug | LR′+aug | XGB′+aug |
|---|---|---|---|---|---|---|
| GC-001 (MI) | UA 0.89; MI 4th | **AF 1.000**; MI 4th | UA 0.99; 4th | UA 0.98; 4th | UA 1.00; 3rd | UA 0.99; 3rd |
| GC-002 (PE) | PE 0.61 | **AF 0.99**; PE 2nd | PE 1.00 | PE 1.00 | PE 1.00 | PE 1.00 |
| GC-003 (dissection, not in DDXPlus) | pericarditis 0.54 | **AF 1.00** | Boerhaave 0.84 | stable angina 0.70 | UA 0.65 | UA 0.53 |
| GC-004 (GERD) | GERD 0.62 | **AF 0.91**; GERD 2nd | GERD 1.00 | GERD 1.00 | GERD 1.00 | GERD 1.00 |

Fused with the graph, red flags off: every model puts PE and GERD first and MI second; dissection is
3rd with B1 and the +aug models and **4th with both ′+aug models** (the golden case asks for the top
5).

**Interpretation:**

1. **R-18's failure is gone from every retrained model.** Original XGBoost answered atrial
   fibrillation to all four golden cases; no retrained model answers it to any. **The masked copies
   do that on their own**: without the channel, a short row is no longer a rare row, so it no longer
   predicts the condition whose patients answer fewest questions.
2. **At full evidence the retraining costs nothing on top-1**, and moves Precision@3 and Recall@5 by
   up to 0.03 in both directions, with no consistent sign across the two models. DDXPlus's full
   records leave nothing to gain (R-16).
3. **What the channel adds cannot be seen yet.** On the no-findings patient it makes both models
   less sure (0.22–0.23, against 0.28–0.43), which is the intended direction. On the golden cases
   it changes little: MI moves from 4th to 3rd in the ML ranking, and GC-004's two encoded denials
   leave GERD where it already was. The comparison that decides it, masked validate patients with
   and without the channel, waits for amendment 3 and A-9.
4. ~~**A new concern: the retrained models are overconfident on short input.**~~ *Narrowed by EXP-019 (its interpretation 7): on masked validate patients at 25% their raw scores are calibrated against DDXPlus (ECE ≤ 0.011); the concern stands for hand-written input, on GC-001's evidence.* Given GC-001's
   classic history, every retrained model puts unstable angina at 0.98–1.00 and MI near 0, though a
   history cannot separate them (only the troponin can); the original LR said 0.89. On 5–13
   findings these are not probabilities, and they must never be shown as such (docs/04 F-2;
   `calibrated_probability` stays empty until 2b). It is a reason for 2b to calibrate on short
   input, not only at full evidence.
5. **The dissection case slips from 3rd to 4th under the channel models**, because they spread
   their probability differently and a knowledge-graph-only condition can never outrank one the
   ML ranker scores (EXP-005 point 7). It is 2c's question, now with a second example.
6. **The configured ranker stays B1-LR.** Nothing here shows a retrained model is better where it
   matters; masked validate will. The retrained bundles are ready for that comparison.

**Next action:** the team: amendment 3 (the mask rule) and A-9; then score all six models on masked
validate at 50% and 25%, which is when the channel either earns its place or does not. 2b: calibrate
on short input too. 2c: how a knowledge-graph-only condition is fused.

---

### EXP-008 — Red-flag rules on DDXPlus validate: sensitivity, precision and the false-alarm burden (2d) (validation split)

| Field | Value |
|---|---|
| Date | 2026-09-25 |
| Author | P3 (run by Claude) |
| Config / ablation | The red-flag layer alone: `src/reasoning/red_flags.py` on each patient's concepts (`concepts_from_evidences`); no ranker, no graph |
| Split used | validation |
| Git commit | `0c09dd4` (the rules before 2d) and the 2d rules, committed with this entry |
| MLflow run | — (rules only; nothing is trained) |
| Seed | 42 (bootstrap) |

**Question:** EXP-014 found the red-flag rules over-firing (R-15): 59% of validate patients got a
flag, and the aortic-dissection rule flagged 50% of them because back radiation alone fired it.
Three must-not-miss conditions had no rule, so docs/05's red-flag sensitivity (amendment 2) counted
them as missed. Can the rules follow published clinical patterns, cover every must-not-miss
condition, and fire less where they should not?

**Setup:** `scripts/check_red_flags.py`, run on the rules as they stood at `0c09dd4` and after 2d.
The 2d rules (the module docstring gives each one's source):

* **Aortic dissection** after the ADD-RS: chest, back or abdominal pain with a highly specific
  sign (tearing pain, pulse deficit, inter-arm pressure difference), or with findings from two of
  its three categories (predisposing condition, pain features, examination). Back radiation is in
  none of them.
* **Pulmonary embolism**: pleuritic pain or breathlessness *and* a Wells risk a history can record
  (unilateral leg swelling or calf pain, immobilisation, recent surgery, previous DVT). Before, the
  risk factor alone fired it, although its reason named the symptom.
* **New:** unstable angina (a crescendo pattern, or chest pain at rest with an ischaemic character),
  myocarditis (chest pain after a viral illness with breathlessness or palpitations), acute
  pulmonary edema (breathlessness with orthopnoea, paroxysmal nocturnal dyspnoea or known heart
  failure). Three crosswalk concepts were added for them: `SYM:rest_pain` (`E_14`),
  `SYM:crescendo_pattern` (`E_13`), `SYM:paroxysmal_nocturnal_dyspnoea` (`E_67`).
* MI, pneumothorax and Boerhaave are unchanged.

The rules were written from those patterns. Validate was consulted twice before they were fixed,
and it is said here: DDXPlus lists "tearing" (`déchirante`, `E_54_@_V_71`) for 76% of pneumothorax
and 76% of Boerhaave patients, and "chest pain even at rest" (`E_14`) for 56% of pneumothorax
patients, so neither was allowed to define a rule alone for a condition it does not belong to
(rest pain needs an ischaemic character; tearing still fires the dissection rule, as the ADD-RS and
docs/04 §3 say it should). No threshold was fitted to validate.

**Results** (33,963 validate patients, 16,943 of them with a must-not-miss condition; 95% bootstrap
intervals):

| Measure | Before 2d | After 2d |
|---|---|---|
| Rules; must-not-miss conditions without one | 5; 3 | 10; **0** |
| **Red-flag sensitivity** (docs/05 amendment 2, target ≥ 0.95) | 0.449 [0.441, 0.456] | **0.823** [0.818, 0.829] |
| Red-flag precision, strict (A-7 open) | 0.230 [0.226, 0.234] | **0.512** [0.508, 0.516] |
| Any flag, all patients | 59% | 58% |
| **Any flag, patients without a must-not-miss condition** (the false-alarm burden) | 31% | **25%** |
| … stable angina / pericarditis / AF, PSVT, GERD, panic | 98% / 98% / 0% | 95% / 65% / 0% |

| Rule | Own patients flagged, before → after | Everyone else, before → after |
|---|---|---|
| Aortic dissection | no DDXPlus patients | **50% → 8%** (all of it Boerhaave and pneumothorax: "tearing") |
| NSTEMI / STEMI | 0.907 | 24% (unstable angina, stable angina, pulmonary edema) |
| Unstable angina | — → **0.892** [0.880, 0.903] | 0% |
| Pulmonary embolism | 0.785 → **0.922** [0.913, 0.930] | 0% → 2% (pulmonary edema patients reporting calf pain, 502 of 599, or one swollen leg) |
| Spontaneous pneumothorax | 0.325 [0.301, 0.348] | 3% |
| Myocarditis | — → **0.641** [0.617, 0.665] | 6% (pericarditis, 62% of its patients) |
| Acute pulmonary edema | — → **0.958** [0.949, 0.966] | 0% |
| Boerhaave | 0.748 [0.730, 0.767] | 0% |

**A-5, the sudden-onset cut-off** (`E_59` ≥ k), which on DDXPlus changes only the pneumothorax rule:
≥ 8 (the working value) reaches 32% of pneumothorax patients and 3% of everyone else; ≥ 7, 44% and
4%; ≥ 6, 54% and 5%.

**Safety layer v1** (`src/reasoning/safety.py`, called last by the pipeline): through the real
graph and the template explainer, 3,000 validate patients (a random sample) needed no repair — no
treatment language, the disclaimer intact, flags first.

**Interpretation:**

1. **R-15's main cause is gone.** The dissection rule flags 8% of patients instead of 50%, and the
   8% is DDXPlus's vocabulary, not back radiation: "tearing" is how DDXPlus describes pneumothorax
   and oesophageal-rupture pain, and both are must-not-miss emergencies themselves. A rule on
   tearing pain cannot tell them apart from a history alone, and the ADD-RS says it should not try.
2. **The share of patients flagged is the wrong alarm-fatigue measure on DDXPlus.** Half of validate
   patients have a must-not-miss condition, so a perfect rule set flags half of them. The burden on
   everyone else fell from 31% to 25%, and almost all of what remains is the MI rule on stable angina
   (95%): exertional pressure-type pain radiating to the arm is the ischaemic pattern, and at
   presentation nothing in a history separates stable from unstable disease; the rule's advice is an
   ECG and a troponin. Whether that flag is "appropriate" is A-7, the team's.
3. **Sensitivity rose from 0.449 to 0.823 but misses the 0.95 target**, and three conditions are the
   reason: pneumothorax 0.33, myocarditis 0.64, Boerhaave 0.75. Each is capped by DDXPlus's own
   sampling. DDXPlus draws each finding independently, so 20% of Boerhaave patients never report the
   vomiting, 29% of myocarditis patients the viral illness, and the onset speed of pneumothorax is
   drawn uniformly over a range (EXP-014). Chasing the target would mean rules that fire without the
   defining finding, which is fitting the generator, not medicine. The ranker and the graph are the
   other layer for these patients: the target is judged on the whole system (must-not-miss
   recall@3), not on the rules alone.
4. **A-5:** lowering the cut-off to ≥ 6 would lift the pneumothorax rule to 54% for two points of
   false alarms. The recommendation is to keep ≥ 8: "sudden" means within seconds to minutes, and
   the gain comes from how DDXPlus draws its onset values. The team decides.
5. **Circularity (R-12).** DDXPlus generated these patients from the definitions the crosswalk
   reads, so every rate here is an upper bound on what the rules do with real histories, and the
   dissection rule has no patients to be sensitive to. The golden cases remain the clinical check:
   GC-001 raises MI, GC-002 PE and pneumothorax, GC-003 dissection, GC-004 nothing.

**Addendum, 2026-09-25: A-7 settled.** With A-7's mapping (`APPROPRIATE` in `src/reasoning/red_flags.py`,
`docs/04` §3), red-flag precision is **0.848** [0.845, 0.852] (strict 0.512 [0.508, 0.516]). A-7
was settled after this entry, by a three-judge panel told not to read any results; it was not fully
blind to them, since the project notes every session loads state some firing rates (`docs/04` §3).
A-5 was decided the same day: ≥ 8 stays.

**Next action:** the team: A-5 (keep ≥ 8, recommended) and A-7 (which flags are appropriate for which
true conditions; EXP-008's candidates are the MI flag on stable angina, the myocarditis flag on
pericarditis, and the dissection flag on pneumothorax and Boerhaave). Then 2c scores the whole
system's must-not-miss recall with these rules on.

---

### EXP-005 — B2, the graph alone: replacing the overlap score (2a) (validate split)

| Field | Value |
|---|---|
| Date | 2026-09-23 to 2026-09-24 |
| Author | P1 (run by Claude) |
| Config / ablation | B2, knowledge graph only: the full graph (DDXPlus + hand-authored + BODHI-S), no ML ranker, no red flags |
| Split used | validation |
| Git commit | `57275cf` plus the 2a change, committed with this entry |
| MLflow run | — (scoring only; nothing is trained) |
| Seed | 42 (bootstrap) |

**Question:** Which scoring rule should replace the weighted overlap `(matched − 0.5 × denied) /
total`, whose three defects were already measured? (1) It ranks stable angina above must-not-miss
unstable angina even with rest pain present — stable 1.00, unstable 0.875 (`docs/02` §5.1,
limitation 2). (2) By graph score alone GC-001's infarction ranked fifth (EXP-014), and in the
fused pipeline with red flags off it ranked fourth: a strict xfail in `tests/test_ranker.py`. (3)
It divides by everything a condition might show, so BODHI-S's enrichment *lowered* MI's
graph-only top-1 from 0.856 to 0.602 (EXP-016).

**Setup.** One fixed measurement harness judged every candidate. Before any candidate was built it
was run on the overlap score and reproduced every published figure exactly (EXP-015's top-1 0.880
and unstable angina 0.213, EXP-016's 0.856 and MI 0.602, the 1.00 vs 0.875 inversion, GC-001's
infarction fourth with red flags off). It measures, in order of weight: the four golden cases on
graph score alone; the angina case (stable angina's full evidence set plus rest pain, `DDX:E_14`);
GC-004 with and without its three denials; the golden cases through the real pipeline (real graph,
the configured logistic-regression ranker) with red flags off and on; self-retrieval; and, last,
graph-only ranking of the 33,963 validate patients on three graphs (DDXPlus only, + hand-authored,
+ BODHI-S), which is circular (R-12). **Rule: no parameter may be chosen by its validate
performance.** Constants are fixed a priori, candidates are developed against the hand-written
checks only, and each is run on validate once.

*What actually ran.* The plan was five independent designs (naive-Bayes, likelihood ratio,
Personalised PageRank, an IDF-weighted linear score, a free design) judged through clinical,
methodological and engineering lenses. That run was cut off by a usage limit after one design,
PageRank, was complete; its designer's own development runs are not recorded. The comparison was
then finished directly, on the same harness, by crossing two families (PageRank, naive-Bayes) with
two versions of the crosswalk closure described below. An independent review of the chosen
design was cut off by the end of the session, but its probe scripts had already found one real
defect, which is fixed in the shipped score (point 4 below).

**What every design has to solve: the graph speaks two vocabularies.** DDXPlus links its
conditions to *questions* (`DDX:E_55`, "where is your pain?"); the hand-authored dissection and
the BODHI-S facts link to *answers* (`SYM:chest_pain`). Any likelihood-type score reads a missing
edge as "this condition never shows this finding", so every chest-pain patient's `SYM:chest_pain`
would count against the nine un-enriched conditions and hand the case to aortic dissection, the one
condition that names it. The overlap score never had this problem only because it never counted
anything against a condition. The PageRank designer found this and closed the graph under the
crosswalk: answers imply their question (sound, as `expand_case` infers for a case), and a
condition that asks a question but whose answer the graph does not state gets that answer at the
mean of the conditions that do state it. The closure, not the scoring family, turned out to do most
of the work: all four candidates score identically on validate.

**Results — the hand-written checks and the diagnostic validate figures.** Validate columns are
expected top-1 with ties broken at random, the harness's measure (* circular, R-12):

| Score | Graph alone: GC-001 MI / 002 PE / 003 AD / 004 GERD | Unstable vs stable angina + rest pain | Denial lowers MI (GC-004) | Fused, flags off: GC-001 / 003 | Fused, flags on | Validate top-1: ddxplus / +hand / all* | MI top-1 +hand → all* | UA top-1* |
|---|---|---|---|---|---|---|---|---|
| Overlap (before 2a) | 3 / 1 / 1 / 2 | 0.875 vs 1.000 **wrong** | yes | 4 / 3 **fail** | all ok | 0.8796 / 0.8795 / 0.8559 | 0.856 → 0.602 | 0.213 |
| PageRank, max closure | 1 / 1 / 1 / 1 | 0.070 vs 0.065 | yes | 2 / 3 | all ok | 0.9575 / 0.9575 / 0.9559 | 0.992 → 1.000 | 0.987 |
| Naive-Bayes, max closure | 1 / 1 / 2 / 1 | −2.21 vs −6.71 | yes | 2 / 5 | all ok | 0.9575 / 0.9575 / 0.9558 | 0.992 → 1.000 | 0.987 |
| PageRank, noisy-OR closure | 1 / 1 / 1 / 1 | 0.070 vs 0.065 | yes | 2 / 3 | all ok | 0.9575 / 0.9575 / 0.9559 | 0.992 → 1.000 | 0.987 |
| Naive-Bayes, noisy-OR closure | 1 / 1 / 1 / 1 | −2.21 vs −6.71 | yes | 2 / 4 | all ok | 0.9575 / 0.9575 / 0.9558 | 0.992 → 1.000 | 0.987 |
| **The same, each fact counted once (shipped)** | **1 / 1 / 1 / 1** | **−2.21 vs −6.71** | yes | **2 / 3** | all ok | 0.9575 / 0.9575 / 0.9558 | 0.992 → 1.000 | 0.987 |

**B2, the shipped score, through `src/eval/metrics.py`** (the protocol's deterministic tie rule,
registry order; 95% bootstrap intervals, 1,000 resamples; full graph; * circular, R-12):

| Metric | Overlap (before 2a) | Naive-Bayes (2a) |
|---|---|---|
| Top-1 | 0.8583 [0.8549, 0.8620] | **0.9234** [0.9206, 0.9259] |
| Top-3 | 0.9957 [0.9950, 0.9964] | **0.9996** [0.9994, 0.9998] |
| MRR | 0.9246 [0.9227, 0.9266] | **0.9613** [0.9598, 0.9625] |
| Precision@3 (`D_in`) | 0.6224 [0.6199, 0.6253] | **0.7559** [0.7532, 0.7588] |
| Recall@5 (`D_in`) | 0.4959 [0.4933, 0.4983] | **0.5698** [0.5675, 0.5719] |
| Recall@5, full D | 0.2853 [0.2835, 0.2872] | 0.3333 [0.3315, 0.3353] |
| Must-not-miss recall@3 | 0.9926 [0.9914, 0.9940] | **0.9998** [0.9995, 1.0000] |
| Dangerous false-negative rate | 0.0006 [0.0002, 0.0010] | 0.0000 [0.0000, 0.0000] |

McNemar, overlap against naive-Bayes: top-1 2,468 vs 4,678 discordant (p ≈ 10⁻¹⁵⁰); top-3 12 vs
145 (p ≈ 10⁻²⁶). Per condition, the new top-1 is 1.000 for seven conditions, 0.997 unstable angina,
0.997 pneumothorax, 0.979 myocarditis, 0.972 Boerhaave, 0.949 pericarditis — and **0.000 for
stable angina** (below).

*Sensitivity of the two constants, on the hand-written checks only* (never on validate), for the
shipped score: every verdict in the table holds at all twelve settings of LEAK 0.001, 0.003, 0.01,
0.03 and CAP 0.8, 0.9, 0.95. The one movement anywhere is GC-003's dissection going from 3rd to
4th in the fused ranking with flags off at LEAK 0.03 and CAP 0.95, still inside its top 5. (Before
each fact was counted once, LEAK 0.03 had moved GC-003 to 2nd by graph alone and 7th fused: some
of that fragility was the double counting.)

**Interpretation:**

1. **All three defects are fixed.** Unstable angina outranks stable angina when rest pain is
   present, by a likelihood ratio of about 90 (a margin of 4.5 nats, the cost of one unexplained
   finding); and when rest pain is *denied*, stable angina outranks unstable. GC-001's infarction
   is first by graph score alone and second in the fused ranking with red flags off, so the strict
   xfail is now a plain assertion. BODHI-S no longer lowers MI (0.992 → 1.000).
2. **Naive-Bayes over PageRank, although the plan named PageRank.** With the same closure the two
   rank validate identically, and once each fact is counted once they agree on every hand-written
   ranking too. Naive-Bayes holds the angina decision by a factor of about 90 where PageRank holds
   it by 8%, and each finding's contribution is exactly its log-probability, where PageRank's
   includes mass diffused through findings the condition has no edge to. An explanation that
   credits a condition with a finding it cannot explain would be wrong. It is also twice as fast.
3. **The noisy-OR correction was found on a golden case, and is disclosed as such.** With the
   closure's first version, an implied question took the weight of the strongest answer implying
   it: a lower bound, used as an estimate. That under-credited aortic dissection on the generic
   questions every chest-pain patient answers ("pain anywhere?", "where?"), and on GC-003 the
   infarction edged past it by graph score. The standard estimate of giving *some* answer is
   `1 − Π(1 − p)`. It is a structural correction of an identified bias, not a constant tuned on
   validate — it changes no validate figure beyond the fourth decimal — but it was found by
   inspecting GC-003, so GC-003 is not independent evidence for it.
4. **Each fact is counted once — a defect found in review.** A case expanded through the
   crosswalk holds an answer together with the questions it implies: *chest pain* arrives with
   "where is your pain?" and "pain anywhere?". Naive-Bayes counted all three, so chest pain alone
   cost a condition unable to explain it `3 × log(LEAK)`. The answer entails its questions, so
   the likelihood of all three is the answer's alone; the shipped score drops a question whenever
   an answer entailing it is a usable concept, and does the same for a denial that entails a
   narrower yes/no denial (GC-004 denies both exertional pain and `DDX:E_218`). The question is
   kept when the answer is not a graph node: `SYM:diaphoresis` lives on `DDX:E_50`. The review
   agent found it on GC-003, where atrial fibrillation's inflated penalty, as the floor of the
   pipeline's min-max rescaling, compressed every other graph score. It changes no validate top-1
   or top-3 figure; Precision@3 rises from 0.749 to 0.756 and Recall@5 from 0.565 to 0.570, and
   GC-003's dissection moves from 4th to 3rd in the fused ranking with flags off.
5. **The graph alone cannot recognise stable angina, and says so.** Stable angina's evidence set
   sits inside unstable angina's, and DDXPlus records no negatives, so nothing in a stable-angina
   patient's record denies rest pain. The two tie; the protocol's tie rule then puts unstable
   angina first, every time: 0.000 top-1 for stable angina, 0.997 for unstable. That is the safe
   direction, and it is the honest answer — "not mentioned" is not "denied", the rule EXP-017
   forced on the ML model too — but it means that telling the two apart is the ML ranker's job in
   the fusion, never the graph's.
6. **Limitation 1 of the KG card is now explicit rather than hidden.** An answer that only one
   condition states — tearing pain, for dissection — is imputed at that condition's own value for
   every other condition that asks about pain character, so it cannot separate them. The graph
   does not know how rarely pericarditis tears, and the score must not invent it; the red-flag rule
   and answer-level enrichment are where that knowledge belongs. The same mechanism costs
   pericarditis 0.024 of top-1 when BODHI-S is added: where BODHI-S gives it a *below-average*
   likelihood for an answer, that is correctly read as mild evidence against it.
7. **For 2c: with red flags off, a condition the ML ranker cannot score can never rank above a
   condition it can.** DDXPlus has no aortic dissection, so its ML score is 0, and after min-max
   normalisation its fused score is at most the graph weight, 0.5, however strongly the graph
   supports it; conditions with ML support can reach 1.0. On GC-003 the graph ranks dissection
   first by a wide margin, and the fused ranking puts it 3rd. Red flags rank it first today. Min-max
   also lets one extreme score set the floor for everyone (point 4). The fusion design (2c) must
   decide how a knowledge-graph-only condition is fused, and whether log-likelihoods are rescaled
   by min-max at all.
8. **Everything on validate is circular** (R-12), more visibly than before: the graph alone now
   reaches top-1 0.92, must-not-miss recall@3 1.000, and Precision@3 0.756 against B1 XGBoost's
   0.764. That is DDXPlus recognising its own definitions, not skill. The hand-written golden cases
   are the evidence that counts, and on them every expectation now holds by graph score alone.

**Next action:** 2c must settle how a knowledge-graph-only condition is fused (point 7). Answer-level
knowledge for the conditions DDXPlus describes only by questions — starting with the pain
characters — would remove limitation 1 where it matters most. When `docs/05` gains a
reduced-evidence condition (D-10), B2 is scored at every level with the same code.

---

### EXP-017 — B1 under a partial history: where the two baselines stop agreeing (validate split)

| Field | Value |
|---|---|
| Date | 2026-09-23 |
| Author | P2 |
| Config / ablation | B1 (both models), scored at three evidence levels |
| Split used | validation |
| Git commit | `98c746b` (the models are the owner's Colab bundle, trained at `68c14bd`) |
| MLflow run | — (scoring only; no training) |
| Seed | 42 |

**Question:** EXP-004 found B1 near-perfect on full-evidence DDXPlus (R-16). Does that survive an
incomplete history — the ordinary case in a consultation — and do the two B1 models behave the same
way when it does not?

**Setup:** No training, no new model. The two B1 models from the Colab bundle score the same 33,963
validate patients three times. A patient keeps their `initial_evidence` plus a share of the rest;
the mask is drawn from a SHA-256 of `seed:case_id`, so it depends on the patient, never on row
order, and reproduces on any machine. Scoring is `src/eval/metrics.py` throughout, so every figure
carries its 95% bootstrap interval (1,000 resamples, seed 42). The whole run takes 27 s on the
laptop CPU.

**This is a diagnostic, not the reduced-evidence condition of decision D-10.** D-10 is undecided and
`docs/05` is frozen; nothing here amends it. The levels and the masking rule are written down now so
that, if the team adopts option (b), they can be adopted as-is and pre-registered.

**Results:**

| Evidence kept | Mean tokens | Model | Top-1 | Top-3 | Must-not-miss recall@3 | Most-predicted condition |
|---|---|---|---|---|---|---|
| 100% | 21.9 | XGBoost | **0.9985** [0.9980, 0.9989] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | pulmonary embolism, 11% |
| 100% | 21.9 | LogReg | **0.9983** [0.9978, 0.9987] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | pulmonary embolism, 11% |
| 50% | 11.5 | XGBoost | **0.5972** [0.5916, 0.6022] | 0.9972 [0.9967, 0.9978] | 0.9969 [0.9960, 0.9977] | **atrial fibrillation, 46%** |
| 50% | 11.5 | LogReg | **0.9755** [0.9740, 0.9772] | 0.9995 [0.9992, 0.9997] | 0.9990 [0.9985, 0.9995] | pulmonary embolism, 11% |
| 25% | 6.2 | XGBoost | **0.2671** [0.2625, 0.2718] | 0.9463 [0.9438, 0.9488] | **0.9346** [0.9305, 0.9383] | **atrial fibrillation, 76%** |
| 25% | 6.2 | LogReg | **0.8558** [0.8522, 0.8597] | 0.9818 [0.9804, 0.9832] | 0.9736 [0.9712, 0.9760] | pulmonary embolism, 10% |

McNemar on top-1, XGBoost vs logistic regression: at 100% the two differ on 8 cases of 33,963
(p = 0.008, exact); at 50% they differ on 13,279, of which logistic regression is right on **13,064**
(p < 10⁻¹⁵); at 25% on 21,124, of which logistic regression is right on **20,560**.

**A second, sharper form of the same effect.** Scored on nothing at all — age and sex, with no
evidence — XGBoost answers *atrial fibrillation with probability 1.000*. Logistic regression answers
atrial fibrillation at 0.43, which is at least a prior rather than a certainty.

**Interpretation:**

1. **The two models are indistinguishable where the protocol looks, and ~~0.73~~ 0.59 top-1 apart
   where it does not** (at 25% evidence; 0.38 at 50%. *Corrected 2026-09-24: 0.8558 − 0.2671 =
   0.589, from this entry's own table; 0.73 was an arithmetic slip, found by the amendment-3
   drafter.*) A single headline number chose between them on 8 cases out of 33,963. At half a
   history it would have been the wrong choice on 13,064.
2. **XGBoost has learned how much was asked, not only what was answered.** The encoder gives a
   default ("no") answer no column, so a short row and an unfinished interview are the same object
   (`docs/03` §2.2). Atrial fibrillation is the condition whose DDXPlus patients answer the fewest
   questions — a median of 9 positive codes, against 17–19 for infarction and unstable angina — so
   "few answers" is a near-perfect predictor of it *within DDXPlus*. XGBoost reads a sparse matrix's
   absent entries as **missing** and sends them down each split's default branch, which compounds
   it; logistic regression, being linear in the present features, simply weakens its evidence. This
   is risk **R-18**.
3. **It is a safety finding, not only an accuracy one.** At 25% evidence XGBoost's must-not-miss
   recall@3 is 0.9346 [0.9305, 0.9383] — below the 0.95 target in `docs/05` §3.2, with the interval
   entirely below it. Logistic regression stays above it at 0.9736.
4. **It is the concrete answer to the supervisor's question** (2026-09-20, "the selected basic ML
   models may not be helpful in designing a robust model"). He is right that the full-evidence
   figure means little — but the gap it hides is between two *classical* models, and it is visible
   with no deep learning and no GPU. Any deep model must be measured here, not only at 100%.
5. **It does not contradict EXP-004 or R-16.** At 100% evidence every figure reproduces exactly.
   The ceiling is real; it is simply not where the interesting differences live.

**Consequences:** the pipeline's real ranker will be wired to **logistic regression**, not
XGBoost, because a hand-authored case carries 5–13 tokens — exactly the regime where XGBoost
answers "atrial fibrillation" to everything. The approved plan said to wire XGBoost first; this
result overrides that. R-18 opened. It is also the strongest available argument for decision
**D-10** option (b).

**Next action:** settle D-10. Fix the encoding defect behind R-18 by giving the encoder an "asked"
channel (experiment R2 of the deep-ranker plan) and retraining both models on the same rows, which
is the fair comparison. Any deep ranker is scored at all three levels from the start.

---

### EXP-004 — B1, ML-only: logistic regression and XGBoost (validate split)

| Field | Value |
|---|---|
| Date | 2026-09-19 (run) · logged 2026-09-20 |
| Author | P2 (run by the owner on Colab, logged by Claude) |
| Config / ablation | **B1** (docs/05 §4): XGBoost, with multinomial logistic regression as a linear reference |
| Split used | train to fit (XGBoost on 90%, stopped early on the other 10%) · **validate** to score · test **not read** |
| Git commit | `68c14bd` |
| MLflow run | none: `run.json` in the run's bundle records the commit, versions, device and timings |
| Seed | 42 |

**Question:** How good is ML-only, the baseline the full system has to beat? (docs/05 §4: "B1 is
the baseline that matters.")

**Setup:** `scripts/train_baselines.py`, run by the owner through `notebooks/colab_b0_b1.ipynb` on a
Colab T4 (docs/11 §4.1), with Python 3.13, NumPy 2.1.3, scikit-learn 1.6.1 and XGBoost 2.1.4. The
inputs are age, sex and the evidence tokens only, as 607 features (`EvidenceEncoder`, fingerprint
`3a0d5a5e01d7f427`, docs/03 §2.2); no label column is read. Logistic regression: C = 1, max-abs
scaling, converged in 17 iterations. XGBoost: the fixed defaults in `src/ml/baselines.py` (η 0.1,
depth 6, row and column subsampling 0.8), best at round 117 of at most 1,000, with 50 rounds of
patience. No tuning. The job took about a minute: encoding 24 s, logistic regression 4 s, XGBoost
8 s, scoring 20 s. **Checked on the laptop:** the downloaded models re-score validate on the CPU
and reproduce every metric and interval to within 3 × 10⁻¹².

**Results** (validate: 33,963 patients, 16,943 of them with a must-not-miss condition; 95% bootstrap
intervals; "best possible" is the ceiling the labels allow):

| Metric | B0 (EXP-003) | B1 logistic regression | **B1 XGBoost** | Best possible |
|---|---|---|---|---|
| Top-1 accuracy | 0.1097 [0.1065, 0.1132] | 0.9983 [0.9978, 0.9987] | **0.9985 [0.9980, 0.9989]** | 1 |
| Top-3 accuracy | 0.3059 [0.3013, 0.3108] | 1.0000 [1.0000, 1.0000] | **1.0000 [1.0000, 1.0000]** | 1 |
| Top-5 accuracy | 0.4818 [0.4765, 0.4869] | 1.0000 [1.0000, 1.0000] | **1.0000 [1.0000, 1.0000]** | 1 |
| MRR | 0.2924 [0.2895, 0.2954] | 0.9991 [0.9989, 0.9994] | **0.9992 [0.9990, 0.9994]** | 1 |
| Precision@3 | 0.5638 [0.5604, 0.5674] | 0.7291 [0.7265, 0.7317] | **0.7639 [0.7613, 0.7667]** | 0.929 |
| Recall@5 (`D_in`) | 0.4602 [0.4580, 0.4623] | 0.5533 [0.5508, 0.5556] | **0.5694 [0.5668, 0.5718]** | 0.753 |
| Recall@5, full D | 0.2679 [0.2663, 0.2696] | 0.3195 [0.3176, 0.3213] | **0.3338 [0.3318, 0.3360]** | 0.434 |
| Must-not-miss recall@3 | 0.2199 [0.2137, 0.2264] | 1.0000 [1.0000, 1.0000] | **1.0000 [1.0000, 1.0000]** | 1 |
| Dangerous false-negative rate | 0.6064 [0.5989, 0.6136] | 0.0000 [0.0000, 0.0000] | **0.0000 [0.0000, 0.0000]** | 0 |
| Brier score | 0.9183 [0.9179, 0.9187] | 0.0034 [0.0026, 0.0043] | **0.0030 [0.0022, 0.0039]** | 0 |
| ECE, raw scores | 0.0023 | 0.0007 | **0.0001** | 0 |
| Macro F1 | 0.0152 | 0.9982 | **0.9985** | 1 |

Per-condition F1 for XGBoost is 1.000 for 10 of the 13 conditions; the exceptions are stable angina
(0.989), unstable angina (0.991) and MI (0.9998). Logistic regression: stable angina 0.988, unstable
angina 0.990.

**Errors.** XGBoost ranks the wrong condition first for 51 patients: 50 with unstable angina and 1
with MI, each ranked below **stable angina**. Logistic regression's 59: 56 unstable angina and 2 MI
below stable angina, and 1 stable angina below pericarditis. Every one still has its true condition
in the top 3.

**McNemar** (docs/05 §6, on top-3): XGBoost against B0, 23,575 patients only XGBoost gets right and
none the other way (p < 10⁻³⁰⁰). XGBoost against logistic regression: no discordant patients, since
both reach 100%. On top-1, added here: 8 against 0 for XGBoost, exact p = 0.008.

**Interpretation:**

1. **DDXPlus is saturated, and the cause is the data, not skill or leakage.** The inputs exclude
   every label column, the test split was never read, and the laptop reproduces the numbers. The
   data explain them: every validate patient's positive answers lie inside their own condition's
   DDXPlus evidence set, and for **91.7%** inside no other condition's. DDXPlus draws each patient's
   evidence from their own condition's list only, so which questions are answered "yes" nearly
   names the condition. The graph alone reaches 88% through the same structure (EXP-015).
2. **On full-evidence DDXPlus, the full system cannot beat B1 on the headline metrics.** Top-3
   accuracy and must-not-miss recall@3 are already 1.000, and MRR leaves 0.0008. So H1 (fusion >
   ML-only on top-3 and MRR) and H2 (fusion + red flags > ML-only on must-not-miss recall@3) cannot
   be supported there, docs/05 §7's *Target* (A0 > B1 on top-3 and must-not-miss recall) is out of
   reach, and the §6 McNemar test between A0 and B1 on top-3 can show only a tie or a loss. The
   protocol accepts a missed *Target* when the analysis explains why, but a saturated test says
   nothing about fusion either way. → **R-16**, decision **D-10** (PROGRESS §3).
3. **What still separates systems here is agreement with DDXPlus's differential.** XGBoost reaches
   Precision@3 0.764 of a possible 0.929 and Recall@5 0.569 of 0.753, ahead of logistic regression
   (0.729 and 0.553) with intervals that do not overlap. A KG gain on these needs the R-12 caveat
   too: D comes from DDXPlus's own differential generator (EXP-013), and the graph from its
   knowledge base.
4. **The only errors are the angina overlap**, which is the graph's blind spot too (`docs/02` §5.1):
   1.8% of unstable-angina patients get stable angina first. That is the dangerous direction,
   though the true condition stays in the top 3.
5. **XGBoost is B1, as planned.** It beats logistic regression on top-1 (8 to 0, p = 0.008) and on
   the differential metrics. That a linear model comes this close says again how easy the data are.
6. **The raw scores are already calibrated on validate** (ECE 0.0001), so B1 meets H5 (ECE < 0.10
   after calibration) without any. Calibration (2b) matters for fused and reduced-evidence scores.
7. **A percentile bootstrap cannot show uncertainty around a perfect score**, hence [1.0000,
   1.0000]. The exact (Clopper–Pearson) 95% lower bounds are 0.99989 for top-3 and 0.99978 for
   must-not-miss recall@3.
8. The usual caveats: closed-world (R-13), synthetic patients, and aortic dissection absent from
   DDXPlus, so B1 can never rank it; only the KG and its red flag can.

**Next action:** The team decides D-10 before the fusion work (2c). `ConditionRanker` then puts
XGBoost behind the pipeline's ranker interface (on hold until the owner says go). 2a's angina fix
should also be judged against these 50 unstable-angina errors.

---

### EXP-003 — B0 prevalence baseline, and the train split's class balance (validate split)

| Field | Value |
|---|---|
| Date | 2026-09-19 (run) · logged 2026-09-20 |
| Author | P2 (run by the owner on Colab, logged by Claude) |
| Config / ablation | **B0** (docs/05 §4): every patient gets the same ranking, the train split's class frequencies |
| Split used | train to count · **validate** to score · test **not read** |
| Git commit | `68c14bd` |
| MLflow run | none: `run.json` in the run's bundle |
| Seed | 42 (B0 itself is deterministic) |

**Question:** Where is the floor? And does EXP-002's projection of the class balance hold on the
real train split?

**Setup:** The same run as EXP-004. On Colab, `build_ddxplus_chestpain.py --split train` decoded
DDXPlus `train.csv` (Hugging Face revision `2ad986a`) and kept the 13 conditions, and B0 counts
them. Its ranking, the same for every patient: pulmonary embolism, GERD, panic attack, pericarditis,
MI, unstable angina, atrial fibrillation, acute pulmonary edema, PSVT, stable angina, Boerhaave,
myocarditis, pneumothorax.

**Results, B0 on validate** (33,963 patients; 95% bootstrap intervals):

| Metric | Value | 95% CI |
|---|---|---|
| Top-1 accuracy | 0.110 | [0.106, 0.113] |
| Top-3 accuracy | 0.306 | [0.301, 0.311] |
| Top-5 accuracy | 0.482 | [0.476, 0.487] |
| MRR | 0.292 | [0.289, 0.295] |
| Precision@3 | 0.564 | [0.560, 0.567] |
| Recall@5 (`D_in`) | 0.460 | [0.458, 0.462] |
| Recall@5, full D | 0.268 | [0.266, 0.270] |
| Must-not-miss recall@3 (16,943 patients) | 0.220 | [0.214, 0.226] |
| Dangerous false-negative rate (16,943 patients) | 0.606 | [0.599, 0.614] |
| Brier score | 0.918 | [0.918, 0.919] |
| ECE, raw scores | 0.002 | — |
| Macro F1 | 0.015 | — |

**Results, the train split against EXP-002's projection** (validate × 7.743):

| Condition | Train | EXP-002 projected | Difference |
|---|---:|---:|---:|
| Pulmonary embolism | 27,468 | ≈28,844 | −4.8% |
| GERD | 25,979 | ≈26,529 | −2.1% |
| Panic attack | 25,019 | ≈25,065 | −0.2% |
| Pericarditis | 22,785 | ≈23,478 | −3.0% |
| Possible NSTEMI / STEMI | 21,260 | ≈22,789 | −6.7% |
| Unstable angina | 21,244 | ≈21,279 | −0.2% |
| Atrial fibrillation | 21,036 | ≈20,203 | +4.1% |
| Acute pulmonary edema | 19,018 | ≈19,359 | −1.8% |
| PSVT | 18,781 | ≈18,398 | +2.1% |
| Stable angina | 16,995 | ≈18,120 | −6.2% |
| Boerhaave syndrome | 15,080 | ≈16,068 | −6.1% |
| Myocarditis | 11,073 | ≈11,979 | −7.6% |
| Spontaneous pneumothorax | 10,162 | ≈10,880 | −6.6% |
| **Total in scope** | **255,900** (25.0% of 1,025,602) | ≈262,990 | −2.7% |

**Interpretation:**

1. **The floor.** B0 puts pulmonary embolism, GERD and panic attack in everyone's top 3, so its
   must-not-miss recall@3 (0.220) is simply pulmonary embolism's share of the must-not-miss
   patients. Anything scoring near B0 is broken.
2. **EXP-002 holds on the real counts.** The rarest condition is still spontaneous pneumothorax,
   with 10,162 training cases against a projected ≈10,880 (R-03's trigger is 500). The imbalance is
   2.70×, and every condition is within 8% of its projection, so R-03 stays resolved. The label
   audit holds too: the differential's out-of-scope mass (33.3%) and the Recall@5 ceilings (0.432
   for the full D, 0.751 for `D_in`) match validate's (docs/03 §2.1).
3. Sex and age were not re-measured: the train parquet stays on Colab, and the summary it sent back
   holds counts only.

**Next action:** B0 is the floor for every later entry. EXP-002's re-check is done.

---

### EXP-016 — BODHI-S enrichment: coverage, and what it does to the graph (validate split)

| Field | Value |
|---|---|
| Date | 2026-09-19 |
| Author | P1 |
| Config / ablation | graph score only, on three graphs: DDXPlus, plus hand-authored, plus BODHI-S. **Unplanned**; EXP-003–012 keep their planned numbers |
| Split used | **validate** · test **not read** |
| Git commit | task 1b, "enrich MI, pericarditis, PE and GERD from BODHI-S" |
| Seed | n/a (deterministic; ties broken at random, in expectation) |

**Question:** What does BODHI-S add, how much of it fits our vocabulary, and how does it change the
graph's rankings, on DDXPlus patients and on the golden cases?

**Setup:** `src/medical_kg/bodhi_s.py` maps each BODHI-S fact about MI, pericarditis, PE and GERD
to the crosswalk concepts it implies, each at least as broad as the fact. `scripts/build_cardiac_kg.py`
reports coverage and the graph-only ranking, as in EXP-015. `scripts/check_crosswalk.py` re-runs the
golden cases.

**Results:**

| Measure | Value |
|---|---|
| BODHI-S facts about the four conditions | 107: 90 symptom facts, 17 risk-factor facts |
| Mapped | 98, giving 56 edges (MI 20, GERD 15, PE 12, pericarditis 9) |
| Unmappable | 7: belching, hiccups, indigestion, spicy food, lack of exercise, past tuberculosis, a vague "myocardial problem" |
| Zero strength (no edge) | 2: lupus and rheumatoid arthritis, for pericarditis |
| Condition-concept pairs that DDXPlus also states | 21, counted once |
| Graph-only top-1 on validate | 0.879 → **0.856** (top-3 0.998 → 0.996) |
| … MI | 0.856 → **0.602** |
| … PE · GERD · pericarditis | 0.969 → 0.896 · 0.897 → 0.853 · 1.000 → 0.989 |
| … myocarditis | 0.682 → 0.930 |
| Golden cases on the full graph | **all 4 expectations hold** |
| GC-001 (classic MI), by graph score alone | MI 5th → **3rd** (0.358), behind Boerhaave 0.385 and pericarditis 0.366 |
| GC-004 (reflux), by graph score alone | GERD still 2nd (0.248), behind pericarditis (0.273) |

**Interpretation:**

1. **On the golden case, the independent knowledge helps.** MI climbs from fifth to third, because
   BODHI-S knows MI's answers (squeezing pain, radiation to the jaw) where DDXPlus knows only the
   questions.
2. **On DDXPlus patients, it costs: top-1 falls by 2.3 points, and MI's from 86% to 60%.** There
   are two causes. Circularity cuts both ways: the DDXPlus-only graph matches its own generator
   (EXP-015), and knowledge from outside that generator fits its synthetic patients less well. The
   fixable cause is the scoring. Overlap divides the matched weight by all of a condition's
   evidence, so enriching 4 of the 13 conditions penalises those four. MI now carries edges that no
   DDXPlus patient can match (hypotension; DDXPlus records no blood pressure), or that DDXPlus's MI
   patients lack (syncope, palpitations). This is the dilution EXP-014 found, made worse by uneven
   enrichment.
3. **Neither number is the system's accuracy.** The DDXPlus figure is circular, and there are four
   golden cases. The overlap score has to change before the enriched graph ranks anything: for
   example a likelihood-ratio or naive-Bayes score over the likelihood bands, or PPR.
   `only_sources()` can keep BODHI-S out of scoring meanwhile, and its edges still serve the
   explanations, as supporting findings with likelihoods.

**Next action:** 2a replaces the overlap score with one that neither punishes a condition for
knowing more nor mis-ranks the anginas, then re-runs EXP-015 and EXP-016 with
`scripts/build_cardiac_kg.py`.

---

### EXP-015 — The graph alone on validate patients: circularity made visible (validate split)

| Field | Value |
|---|---|
| Date | 2026-09-19 |
| Author | P1 |
| Config / ablation | graph score only (no ML ranker, no red flags), on two graphs: DDXPlus only, and DDXPlus plus the hand-authored aortic dissection. **Unplanned**; EXP-003–012 keep their planned numbers |
| Split used | **validate** · test **not read** |
| Git commit | task 1b, "hand-author aortic dissection into the KG" |
| Seed | n/a (deterministic; ties broken at random, in expectation) |

**Question:** How well does the knowledge graph rank DDXPlus patients on its own? And does the
hand-authored aortic dissection, which no DDXPlus patient has, push its way into their
differentials?

**Setup:** `scripts/build_cardiac_kg.py` (about 7 s on the laptop CPU). Each validate patient
becomes a case through `case_from_ddxplus`, which reads the input columns only, and is ranked by
`score_by_connectivity` alone. A condition tied with t others for the places after a higher-scoring
ones counts as top-k with probability (k − a) / t. Written to `data/interim/cardiac_kg_summary.json`.

**Results:**

| Measure | DDXPlus only | + hand-authored |
|---|---|---|
| Top-1 accuracy | **0.880** | 0.879 |
| Top-3 accuracy | **0.998** | 0.998 |
| Aortic dissection ranked first | 0% | 0% |
| Aortic dissection in the top 3 | 0% | 3.1% |
| Weakest condition, top-1: unstable angina | **0.21** | 0.21 |
| Next weakest: myocarditis | 0.68 | 0.68 |

**Interpretation:**

1. **This is circularity, not skill (R-12).** DDXPlus generated its patients from
   `release_conditions.json`, the file the graph is built from, drawing each patient's evidence
   from their condition's evidence set. Overlap scoring therefore recognises the condition almost
   perfectly. On the hand-written golden cases, the same graph ranks GC-001's classic MI fifth
   (EXP-014). docs/05 §8.7 names the risk; this is its size. A KG number measured on DDXPlus data
   must always be reported with this caveat, and the ablation (EXP-012) cannot credit the KG with
   accuracy on DDXPlus patients. A point for the team, next to D-8.
2. **Unstable angina is the graph's blind spot.** It is ranked first for only 21% of its own
   patients, because stable angina's evidence set sits inside unstable angina's (KG card,
   limitation 2). 2a must fix this.
3. **The hand-authored aortic dissection costs DDXPlus patients almost nothing.** Top-1 moves by
   0.001. Dissection takes a top-3 place for 3.1% of patients, through answer-level markers such as
   tearing pain and sudden onset, and first place for none. On GC-003 it now ranks first on graph
   score alone (0.48), where it used to score 0.

**Next action:** 2a fixes unstable angina and the dilution EXP-014 found, and re-measures here.
Report graph-only numbers on DDXPlus only with the circularity caveat.

---

### EXP-014 — Crosswalk check: concepts, red flags and golden cases on real data (validate split)

| Field | Value |
|---|---|
| Date | 2026-09-18 |
| Author | P1 |
| Config / ablation | — (a check of the 1b crosswalk, **unplanned**; EXP-003–012 keep their planned numbers) |
| Split used | **validate** · test **not read** |
| Git commit | task 1b, "link the hand-authored concepts to DDXPlus evidence (crosswalk)" |
| Seed | n/a (deterministic) |

**Question:** The crosswalk (`src/medical_kg/crosswalk.py`) maps the 33 hand-authored concepts to
DDXPlus answers. Does the mapping behave sensibly on real patients? And now that the red-flag rules
and the golden cases can meet DDXPlus data, how do they behave?

**Setup:** `scripts/check_crosswalk.py` (about 2 s on the laptop CPU). It validates the table
against `release_evidences.json` and derives each validate patient's concepts with
`concepts_from_evidences`. It then runs `evaluate_red_flags` on those concepts, and runs the four
golden cases through the pipeline on the real NetworkX graph, each expanded with `expand_case`
(`ConstantRanker`, no retrieval, template explainer). It writes `data/interim/crosswalk_check.json`.

**Results:**

| Measure | Value |
|---|---|
| Crosswalk entries | 33: 11 exact · 12 close · 3 broader · 3 narrower · 1 related · 3 with no DDXPlus equivalent |
| Validation against the release | no problems |
| Chest pain, share of each condition's patients | 100% MI, PE, pneumothorax · ≥ 99% angina, pericarditis, myocarditis, pulmonary edema · 93% Boerhaave · 82% panic · 73% GERD · 0% AF and PSVT (DDXPlus gives them no chest pain) |
| Tearing pain | Boerhaave 76%, pneumothorax 76%, nobody else |
| Radiation to the back | PE 99% · pericarditis 99% · Boerhaave 99% · MI 74% · unstable angina 71% · stable angina 70% · pulmonary edema 68% |
| **Patients given at least one red flag** | **59%** |
| Aortic-dissection rule | **flags 50% of all patients**, although DDXPlus has no aortic dissection: 99.5% of PE, 98% of pericarditis, 74% of MI patients |
| MI rule | flags 91% of MI patients, and 24% of everyone else: 97% of unstable angina, 96% of pulmonary edema, 95% of stable angina |
| PE rule | 79% of PE patients · 0.5% of everyone else |
| Pneumothorax rule | 32% of pneumothorax patients · 3% of everyone else (mostly PE and pericarditis) |
| Boerhaave rule | 75% of Boerhaave patients · nobody else |
| Golden cases on the real graph | **all 4 expectations hold** |
| GC-001 (classic MI), by graph score alone | MI **5th** (0.348), behind Boerhaave 0.385, pericarditis 0.357, pneumothorax 0.353, stable angina 0.350 |
| GC-004 (reflux), by graph score alone | pericarditis 0.286, above GERD 0.250 |

**Interpretation:**

1. **The mapping behaves as the clinical picture predicts, wherever DDXPlus encodes it.** Heaviness
   appears in the ischaemic conditions, burning in GERD, and pleuritic pain in pneumothorax, PE and
   pericarditis. Unilateral leg swelling appears in PE (44%), bilateral in pulmonary edema (93%).
2. **The red-flag rules over-fire on DDXPlus, and the aortic-dissection rule most of all.** Back
   radiation alone is enough to fire it, and DDXPlus lists back radiation for most PE, pericarditis,
   Boerhaave and ACS patients: it records several radiation sites per patient, likely more than real
   patients report. docs/04 §3 accepts low red-flag precision. But a flag on half of all patients
   carries almost no information. The pipeline also ranks red-flagged candidates first
   (`src/pipeline.py`), so the flags reorder the differential. **Opened R-15.** The MI rule's flags on unstable angina are appropriate,
   because unstable angina is an ischaemic emergency too.
3. **Sudden onset is weak evidence in DDXPlus.** The 0–10 onset speed is drawn uniformly within a
   range for each condition, so the cut-off (≥ 8, open decision A-5 in docs/02 §9) reaches only half
   of pneumothorax patients. Together with the rule's three-way conjunction, that leaves the
   pneumothorax rule catching 32% of them.
4. **The golden cases pass on the real graph only because red flags rank first.** By graph score
   alone, the classic MI case puts MI fifth. Overlap is divided by the size of each condition's
   evidence set, which penalises MI's long risk-factor list. The seven pain questions shared by 12
   conditions then decide the rest. That is limitation 1 of the KG card (questions, not answers) plus
   this dilution. 2a must fix the scoring before the KG score carries weight in fusion.
5. **docs/05's "red-flag sensitivity" measures something else.** It counts cases matching a rule's
   pattern, which is 1.0 by construction for a deterministic rule. The per-condition rates above ask
   how many of a condition's own patients its rule reaches. That is the question 2d needs, and a
   point to raise with the team next to D-8. *Decided 2026-09-19:* docs/05 now defines red-flag
   sensitivity this way (§9, amendment 2).

**Next action:** 2d (EXP-008): tighten the aortic-dissection rule so that back radiation alone cannot
fire it, re-measure all five rules on validate, add the three missing rules and settle A-5. 2a
(EXP-005): fix the dilution, and use GC-001 by graph score alone as a regression case. The pipeline
can now run on the real graph through `expand_case`. CI keeps the stub, because data/ is not
committed.

---

### EXP-013 — Label audit of the chest-pain subset (validate split)

| Field | Value |
|---|---|
| Date | 2026-09-18 |
| Author | P2 |
| Config / ablation | — (data audit, **unplanned**; EXP-003–012 keep their planned numbers) |
| Split used | **validate** · test **not read** (the builder refuses it) |
| Git commit | task 1a, "decode validate.csv into the chest-pain parquet" |
| Seed | n/a (deterministic) |

**Question:** Task 1a decodes the labels for the first time. What does the ground truth look like
for a closed-world system that can only name 13 conditions? And does a listed evidence token always
mean the finding is present?

**Setup:** `scripts/build_ddxplus_chestpain.py` on `validate.csv`, which writes
`data/interim/ddxplus_chestpain_validate.summary.json`. A *ceiling* is the best average score that a
**perfect** system able to list only our 13 conditions could reach, using the docs/05 §3.1
definitions. For Recall@5 that is `min(5, |D ∩ in-scope|) / |D|` per patient; for Precision@3 it is
`min(3, |D ∩ in-scope|) / 3`.

**Results:**

| Measure | Value |
|---|---|
| In-scope patients | 33,963 (matches EXP-002) |
| Ground-truth differential D, size | 12.0 entries on average, 6.7 of them in scope |
| Patients whose D includes out-of-scope conditions | **31,185 (91.8%)** |
| Out-of-scope share of D's probability mass | **33.3%** mean · 34.9% median · 52.4% p90 |
| True pathology inside its own D · ranked first in it | 100% · 71.2% |
| **Recall@5 ceiling, D as docs/05 defines it** | **0.434** |
| Recall@5 ceiling, D restricted to the 13 in-scope conditions | 0.753 |
| Precision@3 ceiling (the same under either D) | 0.929 |
| Tokens that mean "no" (the evidence's default value, or NA) | 45,049, in 31,189 patients |
| … the most common | `E_204_@_V_10`, travelled abroad: N (30,416, **89.6%** of patients) · `E_57_@_V_123`, radiates nowhere (8,857) |

**Interpretation:**

1. **R-13 reaches the labels, not just the inputs.** The ground-truth differentials are open-world:
   a third of their probability mass sits on conditions Nightingale cannot output. The most frequent
   are scombroid food poisoning, anemia, acute dystonic reactions and Guillain-Barré syndrome. These
   come from DDXPlus's differential generator, not from clinical chest-pain reasoning.
2. **As written, Recall@5 cannot exceed 0.434, even for a perfect system.** It mostly measures the
   closed-world gap, plus the fact that |D| > 5, rather than ranking quality. docs/05 is frozen, so
   this goes to the team as decision **D-8** instead of being changed here. Top-k accuracy, MRR and
   must-not-miss recall use the true pathology and are unaffected. Precision@3 barely moves (0.929),
   because out-of-scope entries do not lower it.
3. **A listed token is not a present finding.** Default-valued tokens ("N", "nowhere", 0, NA) are
   listed for 91.8% of patients. Counting "the code is listed" as "the patient has it" would give
   89.6% of patients a travel history. `positive_codes` in `src/ddxplus.py` handles this, and the KG
   matching (1b) and feature encoding (1c) must use it.
4. **EXP-002's 801 "unknown" NA tokens are pain-free patients.** None lists `E_53`, and 771 are PSVT.
   Their pain questions are filled with defaults.

**Next action:** D-8 goes to the team, and must be decided before any Precision@3 or Recall@5 number
is produced. 1b builds KG matching on `positive_codes`, and 1c selects features with `INPUT_COLUMNS`.

*Decided 2026-09-19 (the team):* D-8 restricts D to the in-scope conditions for the headline
Precision@3 and Recall@5, and Recall@5 with the full D is reported alongside (docs/05 §9,
amendment 1).

---

### EXP-002 — Class balance of the 13 conditions (validate split)

| Field | Value |
|---|---|
| Date | 2026-09-18 |
| Author | P2 |
| Config / ablation | — (data audit) |
| Split used | **validate** (132,448 rows) · test **not read** |
| Git commit | step 3/3 of decisions D-1..D-3 |
| Seed | n/a (deterministic) |

**Question:** Are any in-scope conditions too rare to learn (risk R-03 — trigger: < 500 training
cases)? Can every evidence token in real patient rows be decoded?

**Setup:** `scripts/class_balance.py` on DDXPlus `validate.csv` (decision D-1: validate only, 87 MB).
Training counts projected by the exact split ratio train/validate = 1,025,602 / 132,448 = **7.743**
(sizes from the Hugging Face datasets-server API). Only `PATHOLOGY`, `SEX`, `AGE` and `EVIDENCES` are
read; `DIFFERENTIAL_DIAGNOSIS` is untouched.

**Results:**

| Condition | Validate | Share | Train ≈ | % female | Median age | DDXPlus severity |
|---|---:|---:|---:|---:|---:|---:|
| Pulmonary embolism | 3,725 | 11.0% | 28,844 | 52% | 38 | 2 |
| GERD | 3,426 | 10.1% | 26,529 | 52% | 36 | 3 |
| Panic attack | 3,237 | 9.5% | 25,065 | 51% | 39 | 5 |
| Pericarditis | 3,032 | 8.9% | 23,478 | 51% | 40 | 4 |
| Possible NSTEMI / STEMI | 2,943 | 8.7% | 22,789 | 50% | 45 | 1 |
| Unstable angina | 2,748 | 8.1% | 21,279 | 54% | 44 | 2 |
| Atrial fibrillation | 2,609 | 7.7% | 20,203 | 52% | 36 | 3 |
| Acute pulmonary edema | 2,500 | 7.4% | 19,359 | 51% | 47 | 1 |
| PSVT | 2,376 | 7.0% | 18,398 | 49% | 40 | 2 |
| Stable angina | 2,340 | 6.9% | 18,120 | 52% | 46 | 2 |
| Boerhaave syndrome | 2,075 | 6.1% | 16,068 | 51% | 40 | 2 |
| Myocarditis | 1,547 | 4.6% | 11,979 | 54% | 35 | 2 |
| Spontaneous pneumothorax | 1,405 | 4.1% | 10,880 | 52% | 38 | 2 |
| **Total in scope** | **33,963** | 25.6% of validate | **≈262,990** | | | |

Imbalance (largest / smallest): **2.7×**. Evidence tokens in in-scope rows: **87.2%** decode
directly · **12.6%** numeric ordinal (e.g. pain intensity `E_56_@_4`) · **0.1%** unknown — every one
of them `E_54_@_V_11`, the "NA" sentinel that `decode_ddxplus.py` deliberately skips.

**Interpretation:**

1. **R-03 does not trigger.** The rarest condition projects to ≈10,880 training cases — about 22× the
   threshold — and a 2.7× imbalance is mild. `class_weight="balanced"` remains sensible but is not
   load-bearing.
2. **The system is closed-world — a new, undisclosed limitation (→ R-13).** Only 25.6% of DDXPlus
   cases fall inside our 13 conditions. The model will only ever see those, so any presentation is
   forced into one of 13 — including genuine chest-pain causes outside our set, such as pneumonia.
   The other 36 DDXPlus pathologies could later train an out-of-scope / abstention signal.
3. **Synthetic demographics are unrealistic — concrete evidence for `docs/04` §6.** Every condition
   is ~50% female; median MI age is 45 (real-world first MI is typically in the 60s); spontaneous
   pneumothorax is 52% female (in reality strongly male-predominant). The model therefore cannot learn
   real epidemiological priors. **A near-parity by-sex breakdown in Phase 4 will be an artifact of
   generation, not evidence of fairness — it must not be reported as such.**
4. **Feature encoding (1c) must handle three token kinds:** categorical codes (binary /
   multi-choice), **numeric ordinal scales** as ordered numeric features rather than one-hot, and the
   `V_11` "NA" sentinel explicitly.

**Next action:** R-03 resolved on projection — re-confirm when `train.csv` is downloaded. Open R-13.
Carry the three token kinds into 1c.

*Re-checked 2026-09-20 on the real train split (EXP-003): **confirmed.** The rarest condition has
10,162 training cases, the imbalance is 2.70×, and every condition is within 8% of its projection.
Sex and age were not re-measured: the train parquet stays on Colab, and its summary holds counts
only.*

---

### EXP-001 — R-01 crosswalk spike: DDXPlus ↔ BODHI-S

| Field | Value |
|---|---|
| Date | 2026-09-17 |
| Author | P1 |
| Config / ablation | — (feasibility spike) |
| Split used | none — vocabulary files only |
| Git commit | Phase 0c |
| Seed | n/a (deterministic) |

**Question:** Can BODHI-S serve as the cardiac knowledge-graph backbone? Gate: 60% crosswalk
coverage (risk R-01).

**Setup:** `scripts/decode_ddxplus.py` then `scripts/spike_crosswalk.py`. Inputs were
`release_evidences.json` + `release_conditions.json` (141 kB) and BODHI-S `triples.jsonl` +
`nl_facts.jsonl` (4.4 MB). **The 847 MB DDXPlus patient CSVs were not needed** — the gate is a
vocabulary question, not a data-volume one.

**Results:**

| Metric | Value |
|---|---|
| BODHI-S conditions (total) | 555 |
| Condition coverage, strict (exact) | **31%** (4/13) |
| Condition coverage, generous (incl. approximate) | 77% |
| Symptom alignment, content-word | **28%** (35/123) |
| Symptom alignment, naive string | 0% (123 evidences) — *artifact* |
| DDXPlus evidence vocabulary decoded | 919 entries (223 questions → 696 values) |
| Nightingale conditions found in `release_conditions.json` | **13/13** |

**Interpretation:** Below the gate on every measure that matters → **FALLBACK adopted**. Three
findings worth carrying forward:

1. **BODHI-S is broad, not cardiac-deep.** Myocarditis, pneumothorax and Boerhaave are absent;
   stable/unstable angina collapse to one node, as do AF/PSVT.
2. **The 0% naive figure is a measurement artifact, not a result.** DDXPlus evidences are
   patient-facing questions; BODHI-S symptoms are clinical noun phrases. They never match lexically
   even when identical in meaning ("Are you feeling nauseous?" ≡ `Vomit <char> nausea present`).
   The 28% content-word figure is fairer but still generous — some matches hinge on "pain" alone.
3. **DDXPlus's own `release_conditions.json` is a better backbone than the anticipated fallback** —
   curated, 13/13 coverage, with ICD-10 codes and a severity ranking that independently
   corroborates our must-not-miss set.

**Also noted:** DDXPlus English is machine-translated from French and unreliable — `déchirante`
(*tearing*, the aortic-dissection descriptor) renders as **"heartbreaking"**. Prefer codes over
English labels throughout.

**Next action:** Closed R-01; opened **R-12 (circularity)** — the KG and ranker now share a source,
recorded as a reporting obligation in the evaluation protocol §8.7. P1 proceeds to build the KG
loader from `ddxplus_chestpain_conditions.json`.

Full report: [10-spike-r01-crosswalk.md](10-spike-r01-crosswalk.md)

---

### EXP-000 — Log initialised

| Field | Value |
|---|---|
| Date | 2026-09-17 |
| Author | P2 |
| Config | — |
| Git commit | Phase 0 |

**Question:** n/a — placeholder establishing the format.

**Next action:** First real entry will be **EXP-001, the Week-1 feasibility spike**: DDXPlus
evidence decoding and the DDXPlus↔BODHI-S crosswalk coverage measurement. Its result decides
risk **R-01** and therefore the KG backbone (see
[07-risk-register.md](07-risk-register.md)).

---

## Planned experiment sequence

| ID | Experiment | Phase | Decides |
|---|---|---|---|
| EXP-001 | Crosswalk coverage spike | 0 | R-01 — KG backbone |
| EXP-002 | Class balance across 13 conditions | 1 | R-03 — which conditions are learnable |
| EXP-003 | B0 prevalence baseline | 1 | Metric floor. *Run 2026-09-19 on Colab by the owner; logged* |
| EXP-004 | B1 ML-only (LogReg → XGBoost) | 1–2 | The competitor to beat. *Run with EXP-003; logged.* B1 reaches the ceiling of top-3 and must-not-miss recall (R-16 → D-10) |
| EXP-005 | B2 KG-only scoring | 2 | Is the graph useful alone? *Run 2026-09-24 as 2a's design experiment: the overlap score replaced by naive-Bayes over a crosswalk-closed graph; every golden case holds on graph score alone. Validate figures circular (R-12)* |
| EXP-006 | A0 fusion, weight sweep | 2 | Fusion weights (validation only) |
| EXP-007 | Calibration (Platt vs isotonic) | 2 | H5 |
| EXP-008 | Red-flag sensitivity/precision | 2 | Safety layer tuning. *Run 2026-09-25 with 2d: sensitivity 0.449 → 0.823 (target 0.95 not met), the dissection rule's false alarms 50% → 8%; logged* |
| EXP-009 | B3 LLM-only | 3 | H3 |
| EXP-010 | B4 text-RAG + LLM | 3 | H3 |
| EXP-011 | Retrieval quality sweep | 3 | Chunking/embedding choice |
| EXP-012 | **Full ablation A0–A6** `[TEST]` | 4 | H1, H2, H4 |
| EXP-013 | Label audit of the chest-pain subset *(unplanned, run 2026-09-18)* | 1 | D-8: what D means for Precision@3 and Recall@5 |
| EXP-014 | Crosswalk check: concepts, red flags and golden cases on real data *(unplanned, run 2026-09-18)* | 1 | R-15: the red-flag rules over-fire; 2a: graph-only ranking |
| EXP-015 | The graph alone on validate patients *(unplanned, run 2026-09-19)* | 1 | R-12: how big the circularity is; 2a: unstable angina |
| EXP-016 | BODHI-S enrichment: coverage and effect *(unplanned, run 2026-09-19)* | 1 | 2a: a score that does not punish enriched conditions |
| EXP-017 | B1 under a partial history *(unplanned, run 2026-09-23)* | 1 | R-18: the models separate once the history is incomplete; evidence for D-10 (b) |
| EXP-018 | B1 retrained for R-18: masked copies (+aug), with and without the "asked" channel (′) | 1–2 | Whether the channel removes the atrial-fibrillation answer to short input. *Run 2026-09-25 on Colab by the owner; logged.* The masked copies alone remove it; full evidence is unchanged; the retrained models are ~~overconfident on short input~~ near-certain on hand-written input (narrowed by EXP-019). The reduced levels wait for amendment 3 |
| EXP-019 | Every system at 100%, 50% and 25% evidence *(amendment 3, run 2026-09-25)* | 1–2 | The masks' digests recorded before scoring; the original B1 misses the safety targets at 25%, the `+aug` models meet them, the channel adds a small significant gain, the red flags collapse. The configured ranker becomes B1-LR′+aug |
| EXP-020 | XGBoost's seeds 43–46 for B1-XGB, B1-XGB+aug and B1-XGB′+aug *(amendment 3 §6; Colab, the owner's choice)* | 2 | How much each XGBoost figure, at every level, depends on the seed; closes §6's disclosed gap. The seed-42 models stay the ones used |
