# 8 · Conclusion

> **Written 2026-11-12, after the single test read** (`docs/08` EXP-012 `[TEST]`, 34,657 patients, opened
> once on 2026-11-09). Every figure is copied from `docs/08` with its 95% interval, evidence level and
> model label; test-split figures are tagged `[TEST]` and the rest are validation. Failed hypotheses come
> first and are stated as failures (`docs/05` §8 rule 5). The reading in force is `docs/05` v1.5
> (amendment 3b): the reduced-evidence claims are judged at 25% evidence only, with the 50% comparisons
> reported beside them and not counted. The other team members' agreement to that reading is not recorded.

## 8.1 What we built

Nightingale is a decision-support prototype for acute chest pain. It ranks 14 conditions (13 learned from
DDXPlus and aortic dissection, hand-authored into the graph) by pooling a small transformer (B1-DL′+aug,
adopted through a safety floor fixed before training) with a cardiac knowledge graph built from three
sources. Independent rules, written after published clinical patterns, raise the must-not-miss conditions.
The system explains each ranking from the graph and from the model's own reasons, names the next questions
and the investigations to consider, re-ranks live as findings are entered, reports every component that
degraded, and never recommends a treatment. It also holds:

- **Evidence retrieval** over a corpus of 2,067 open-access articles (26,190 passages), diagnosis only,
  each excerpt shown with its licence. **Its relevance was never judged** (EXP-011's judging was dropped
  because the team had no time), so the evidence is shown as unjudged, and the hybrid retriever is the
  default by design, not by a result.
- **A language-model explainer**, built, run on Kaggle and **not shipped**: it failed its pre-registered
  checkpoint (§8.2, RQ3), so the template explainer stays and `llm.enabled` is false.
- **A patient-history panel** for 1,013 synthetic patients, **display only**: it never reaches the ranking,
  and a test enforces that.

It carries the disclaimer on every output path, and its figures carry it too. It is a research prototype
that has had no clinical validation, no regulatory review and no approval by any authority.

## 8.2 Answers to the research questions

Each verdict is the pre-registered rule's, applied once to the test split, **the same as on validation row
for row** (`docs/08` EXP-012). "Seeds" counts how many of seeds 42–46 reach the verdict. A0 is the fused
system with B1-DL+aug as its model, A0′ the same with B1-DL′+aug; the fusion weight α is EXP-006's (0.30 for
A0, 0.05 for A0′), and the fused system is level with its own model (§8.3).

**Failed first.**

- **RQ1 (H1, H1-R): not supported, 0 of 5 seeds.** At 100% every machine-learning comparator ranks the true
  condition in the top 3 for all 34,657 patients `[TEST]`, so nothing can beat it (R-16). The one H1 check
  that passes is top-3 against the graph alone, which misses 27 patients and is circular (R-12). A0's MRR at 100% is **lower** than the model's:
  0.9157 against B1-XGB's 0.9994, a difference of −0.0837 [−0.0859, −0.0819], because the red-flag layer
  ranks flagged conditions first even when the flag names the wrong one. At 25% both arms beat the original
  B1 models and the graph alone but do not beat the models trained on masked copies: A0 against B1-XGB+aug
  is −0.0001 on top-3 (p = 0.883) and −0.0052 on MRR [−0.0062, −0.0044]. A0 fails 5 of the 12 H1-R checks
  and A0′ fails 4.
- **RQ2 (H2, H2-R): not supported, 0 of 5 seeds.** At 100% every machine-learning comparator, A0 and A0′ alike, misses 0 of
  the 16,972 patients with a must-not-miss condition `[TEST]`, a tie. At 25% A0 fails 2 of the 5 H2-R checks
  and A0′ fails 2. The safety result is real but belongs to training on masked copies, not to the fusion:
  the original B1-XGB's must-not-miss recall@3 at 25% is 0.8710 [0.8659, 0.8764] and B1-LR's is 0.9198
  [0.9156, 0.9238], both below the 0.95 target, while A0 reaches 0.9981 [0.9974, 0.9987] and A0′ 0.9986
  [0.9980, 0.9991], and so does every `+aug` model.
- **RQ3 (H3): not supported** (validation, 50 patients and the four golden cases, 100% evidence; two of its
  four comparisons pass). The grounded explainer's raw answers hold an unsupported-claim rate of 0.741
  [0.663, 0.803] (163 of 220 claims), against a bar of under 0.20; the language model alone scores 1.000
  [1.000, 1.000]. The check is lexical, so a correct paraphrase fails, and the rate measures staying within
  the inputs, not correctness. Guardrails v1 stopped all 50 of the grounded answers, so none reached a user.
  Baselines B3 and B4 are reported as not carried into the evaluation, with that entry as their record.
- **RQ6 (H6): not supported, 0 of 5 seeds.** At 25% the transformer beats the logistic regression trained
  like it on top-3, MRR and must-not-miss recall (A0 against B1-LR+aug: top-3 +0.0008, p = 0.0024; MRR
  +0.0022 [+0.0015, +0.0028]), and is level with the XGBoost trained like it (MRR +0.0002 [−0.0005, +0.0009]
  against B1-XGB+aug). H6 needs every comparator, so it fails. At 100% the models tie at a ceiling.

**Supported, with what stands behind the verdict.**

- **RQ4 (H4; A1 and A1b): supported by the rule, 5 of 5 seeds, and the rule is measuring something other than
  the graph.** Removing the graph and the flags (A1) leaves must-not-miss recall@3 exactly where it was (a
  degradation of 0.0000) while **top-1 rises** by 0.1642 [0.1606, 0.1685] (A0 0.8346 to A1 0.9987), because
  the flags no longer push their conditions first. Against A1b, which keeps the flags and drops only the
  graph, both degradations are 0.0000 and the contrast is 0: **the graph itself contributes nothing
  measurable here.** The same holds for H4 on validation.
- **RQ5 (H5): supported, 5 of 5 seeds**, raw: A0's top-label ECE at 100% is 0.0099 [0.0094, 0.0103] and
  A0′'s 0.0398 [0.0394, 0.0402]. No calibrator ships (`calibration.file` is null by the owner's decision of
  2026-11-09, because a short hand-written case is near-certain under any calibration, R-17), so no
  probability is shown, only ranks. At 25%, descriptive and not part of the test, the ECE is 0.0644 [0.0620,
  0.0666] (A0) and 0.0785 [0.0766, 0.0806] (A0′).

**Success criteria** (`docs/05` §7), unprimed and primed judged separately.

- **Minimum: met**, 5 of 5 seeds (top-3 at 100%: A0 1.0000 against B0's 0.3061 [0.3013, 0.3109]).
- **Target at 100% and Stretch at 100%: not met**, because the comparators tie at 1.000 and a tie fails a
  point-estimate comparison.
- **Target at 25%: met by A0′ and not by A0, 5 of 5 and 0 of 5 seeds, on a handful of patients.** A0′ misses
  40 of 34,657 in the top 3, B1-DL′+aug 41, B1-XGB′+aug 56 and B1-LR′+aug 80; A0 misses 73 against
  B1-XGB+aug's 71. The fused system equals its own component to within a patient, so this is not a robust
  win. **The reduced-evidence Stretch is not met by either arm.**
- **Not assessed:** the unsupported-claim rate of the template explainer that ships (the runner does not
  measure it) and the clinician-reviewed explainability the Stretch also lists. Neither is reported as met.

## 8.3 What the project found along the way

1. **On DDXPlus, which questions were answered nearly names the condition**, so full histories cannot
   separate systems: every classifier ranks the true condition in the top 3 for all validation patients
   (EXP-004, R-16), and on the test split for all 34,657 `[TEST]`.
2. **An encoding that cannot tell "denied" from "not asked" made a strong model fail on short histories.**
   Given no evidence at all, XGBoost answered atrial fibrillation with probability 1.000 (EXP-017, R-18).
   Training on masked copies repaired it, and an explicit "asked" channel adds a small gain (EXP-018,
   EXP-019).
3. **On partial histories, must-not-miss safety is carried by the ranker trained on masked copies, not by the
   rules or the graph.** The red-flag layer's sensitivity falls from 0.8285 [0.8229, 0.8344] at 100% to
   0.1369 [0.1317, 0.1420] at 25% `[TEST]`, against a target of 0.95, while the transformer trained on
   masked copies holds must-not-miss recall@3 at 0.9982 [0.9975, 0.9988] (B1-DL+aug) and 0.9986 [0.9980,
   0.9991] (B1-DL′+aug) at 25% `[TEST]`, and the fusion adds nothing measurable over the transformer
   alone (EXP-006, EXP-012; R-12, R-16).
4. **A small transformer is non-inferior to the logistic regression trained like it at 25% on validation**
   (MRR +0.0018 [+0.0009, +0.0026] and must-not-miss recall@3 +0.0006 [−0.0001, +0.0013] for B1-DL′+aug,
   seed 42; EXP-021), at 1.38 ms p95 a patient at batch 1 on the laptop CPU and 13.6 to 16.9 ms p95 through
   the whole pipeline (EXP-022), within the budget for re-ranking as findings are entered.
5. **A clinically sensible translation of a hand-written finding was wrong for the data, and the model's
   reasons found it**: radiation to the jaw maps, in DDXPlus, to acute pulmonary edema only, and the answer
   was moved under the jaw (A-10, R-17). The other 16 representative answers have not been checked the same
   way.
6. **A language model asked for the ranked list of 14 conditions ranked worse than the pipeline at every
   level, and adding retrieved passages made it worse** (EXP-009/010, validation, 50 patients, wide
   intervals).

## 8.4 Limitations

Chapter 7 states them in full: synthetic data with unknown performance on real patients; chest pain only;
a closed world of 14 conditions (R-13); the graph's circularity on DDXPlus (R-12) and the saturation of the
benchmark (R-16); random masks that measure robustness to missing answers, not history-taking; a
case-to-token translation with no ground truth (R-17); red-flag sensitivity below its target; relevance of
the retrieved evidence unjudged and part of it non-commercially licensed (R-20); a lexical claim check; and
no clinical validation of any kind. Near-equal results by sex or by age are an artifact of synthetic
demographics, not evidence of fairness (EXP-002). The test split was read once and is not to be scored
again.

## 8.5 Future work

An independent test set of clinical vignettes, which would escape R-12 but give wide intervals (`docs/07`
R-16, option 3); an out-of-scope or abstention signal trained on DDXPlus's other pathologies (R-13);
checking the other 16 representative answers against DDXPlus (R-17); blind relevance judging of the
retrieved passages (EXP-011's tool and its 159-row pool exist and were not used); a clinically trained
reviewer for the explanations and for A-2's open question; a language-model explainer re-run under a new
pre-registered design (EXP-009b), which none is planned to be; free-text intake, if the project were
extended; and the stretch goals not pursued (a UCI Heart risk sub-model, knowledge-graph embeddings, the
PTB-XL ECG extension).

## 8.6 Closing statement

Nightingale shows that, on synthetic chest-pain data, a small transformer trained on masked copies of the
histories keeps must-not-miss recall@3 above 0.99 when a history is cut to a quarter, and that the graph
and the rules around it add explanations and a safety net, not measurable accuracy. Nightingale provides
decision support only, never a diagnosis, and a qualified clinician makes every decision.
