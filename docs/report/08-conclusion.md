# 8 · Conclusion

> **Placeholder structure, 2026-10-14.** The findings are not final: the fusion sweep, the
> calibration of the fused score, retrieval, the language model and the test read are all still to
> come. This chapter is written last, after EXP-012, and must state every failed hypothesis plainly
> (`docs/05` §8 rule 5). The bullets below are the skeleton and the validation facts it will be built
> from; none of them is a verdict.

## 8.1 What we built

*[One paragraph.]* A decision-support prototype for acute chest pain that ranks 14 conditions by
pooling a small transformer (B1-DL′+aug, adopted through a safety floor fixed before training) with
a cardiac knowledge graph built from three sources, raises must-not-miss conditions through
independent rules after published clinical patterns, explains each ranking from the graph and the
model's own reasons, re-ranks live as findings are entered, reports every degraded component, and
never recommends a treatment. [pending: retrieval EXP-011] [pending: LLM baselines B3/B4,
EXP-009/010] [pending: 1d, the patient knowledge graph]

## 8.2 Answers to the research questions

*[One short paragraph per question, each naming its hypothesis, its verdict on the test split, the
evidence level, the model labels, and for how many of the five seeds the verdict holds. Failed
hypotheses first and plainly.]*

- **RQ1 (H1, H1-R).** [pending: EXP-012, test split] [pending: EXP-006 fusion sweep]
- **RQ2 (H2, H2-R).** [pending: EXP-012, test split]
- **RQ3 (H3).** [pending: LLM baselines B3/B4, EXP-009/010]
- **RQ4 (H4; A1 and A1b).** [pending: EXP-012, test split]
- **RQ5 (H5).** [pending: EXP-007 stage 2] [pending: EXP-012, test split]
- **RQ6 (H6).** [pending: EXP-012, test split]
- **Success criteria** (Minimum, Target, Stretch, and the reduced-evidence rows), unprimed and
  primed, judged separately. [pending: EXP-012, test split]

*Facts already established on validation that the answers must take into account (not verdicts):*
on full histories the ML baselines saturate, so H1, H2 and the Target cannot be supported at 100%
(R-16); as approved, the reduced-evidence claims cannot be met at 50% either, pending the team's
decision E-2 (§5.9); and every graph figure on DDXPlus data is circular (R-12).

## 8.3 What the project found along the way

*[Three to five points, each with its evidence level and model label; candidates, to be confirmed
or dropped after the test read:]*

1. On DDXPlus, which questions were answered nearly names the condition, so full histories cannot
   separate systems (EXP-004, R-16).
2. An encoding that cannot tell "denied" from "not asked" made a strong model fail on short
   histories; masked training copies and an explicit "asked" channel repaired it (EXP-017 to
   EXP-019, R-18).
3. On partial histories, must-not-miss safety on DDXPlus is carried by the ranker, not the rules,
   whose sensitivity falls from 0.823 to 0.135 at 25% (EXP-019).
4. A small transformer is non-inferior to logistic regression at 25% on validation, at about 80
   times its per-patient time and still within live-re-ranking budgets (EXP-021, EXP-022).
5. A clinically sensible translation of a hand-written finding was wrong for the data, and the
   model's reasons found it (A-10, R-17).

## 8.4 Limitations

*[One paragraph pointing to Chapter 7: synthetic data, circularity, closed world, random masks,
the inversion's missing ground truth, no clinical validation.]*

## 8.5 Future work

*[Candidates, from the project's own plans and limitations:]* an independent test set of clinical
vignettes, which would escape R-12 but give wide intervals (`docs/07` R-16, option 3); an
out-of-scope or abstention signal trained on DDXPlus's other 36 pathologies (R-13); checking the
other 16 representative answers against DDXPlus (R-17); a clinically trained reviewer for the
explanations and for A-2's open question; "what to ask next" and free-text intake, if cut
[pending: 3d]; the stretch goals not pursued (UCI Heart risk sub-model, knowledge-graph embeddings,
the PTB-XL ECG extension).

## 8.6 Closing statement

*[Two sentences, to be written last. It must repeat that Nightingale provides decision support
only, never a diagnosis, and that a qualified clinician makes every decision.]*
