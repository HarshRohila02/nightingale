# `docs/05` amendment 3: errata (proposal for the team)

**Drafted:** 2026-09-25, by Claude · **Status:** proposal only; `docs/05` v1.3 is in force as
approved · **Approval:** all four team members, as a §9 entry ("Amendment 3a, errata") · **Decision:**
E-1 in `PROGRESS.md` §3

Amendment 3 was approved on 2026-09-25 and applied to `docs/05` exactly as proposed. Four
independent reviewers then checked the applied text. Two found it faithful to the proposal, and one
checked the facts it states. The fourth read it as a sceptical team member would, and found
contradictions **inside the approved wording itself**. They are listed here rather than fixed,
because `docs/05` changes only through its amendment log. Each correction is proposed as exact
replacement text. None changes a metric, a level, the mask or a threshold. Each says which reading
of the approved text it adopts, so the team can pick a different one.

**Nothing scored so far depends on them.** EXP-019 scores B0, B1 and B2 descriptively at the three
levels. The corrections decide how H1-R, H2-R, H6 and the reduced-evidence Target are *judged*, so
they should be settled before any fusion (2c) or deep-model result exists.

---

## 1. Needed before 2c or B1-DL is judged

| # | Where | The problem | Proposed correction |
|---|---|---|---|
| 1 | §4 labels vs §6, last bullet | **A direct contradiction.** §4 says B1-XGB is "never retrained under the same label". §6 says the missing-seeds gap "is closed when B1-XGB is next retrained". | §4, the B1-XGB/B1-LR row: after "never retrained under the same label", add: *"except that §6's seeds 43–46 are trained on the same rows and encoder and reported beside the seed-42 model, which stays the one used."* |
| 2 | §4 amendment paragraph, §6, §7 | **Two verdict rules at 100%.** §1 and §7 judge H1, H2, Target and Stretch against B1-XGB, with B1-LR "beside". §4 and §6 say a claim of beating B1 must hold against each variant. If A0 beats B1-XGB but not B1-LR, §7 says the Target is met and §4 forbids saying A0 beats B1. | §4 amendment paragraph, replace its third sentence with: *"At 100%, H1, H2, Target and Stretch are judged as first written, against B1-XGB, and that verdict is reported first; a statement that a system beats B1 without naming a variant must also hold against B1-LR, and the B1-LR comparison is reported beside every first-written verdict."* |
| 3 | §6 amendment vs §1, §7 | **Does a claim need significance?** §6: "A claim needs every one of its comparisons to pass at α = 0.05." But the reduced-evidence Target says only ">", and the reduced Stretch adds "with every comparison statistically significant", which would be redundant. | §6, replace that sentence with: *"A hypothesis is supported, and a Stretch met, only when every one of its comparisons passes at α = 0.05. A Target compares point estimates; its tests are reported beside it."* This is the first-written reading: Target "A0 > B1", Stretch "A0 > B1 with statistical significance". |
| 4 | §1 H1-R, H6 vs §6 | **Comparisons with no test.** H1-R and H6 include MRR, but the only test defined is McNemar on top-3. H1-R compares A0 with B1-DL and with KG-only (B2), and §6 lists neither pair. | §6, add a sub-bullet: *"MRR is compared by a paired bootstrap of the per-case difference in reciprocal rank (1,000 resamples, seed 42); it passes when the 95% interval excludes 0. For H1-R, McNemar and the MRR test are also run between A0 and B1-DL and between A0 and B2."* |
| 5 | §6 seeds bullet | "Logistic regression is deterministic and is fitted once", but the same bullet says mask augmentation draws its masks from the model's seed, so **B1-LR+aug changes with the seed** (`scripts/train_baselines.py` fits it on the augmented rows). | Replace that sentence with: *"A logistic regression on fixed rows is deterministic and fitted once; with mask augmentation its rows depend on the seed, so `+aug` logistic regressions take the five seeds too."* |
| 6 | §4 labels, §5, §1 H6 | **"Arm" is never defined, and H6's arm is not fixed.** §4's labels make "B1-DL" the plain model. §5 fixes A0's component as "B1-DL, the `+aug` arm", but H6 and H1-R say only "B1-DL". The proposal's decision 4 said `+aug` for A0 **and** H6. Nor does §7 say what a primed A0′ is compared with. | §4, after the labels table, add: *"An arm is one training variant of a model: plain, `+aug`, ′ or ′+aug. In H1-R and H6, B1-DL means B1-DL+aug, seed 42, the arm A0 uses, and 'trained like it' means B1-LR+aug and B1-XGB+aug. Primed systems are compared with the primed variants trained like them."* |
| 7 | §6 and §9, "fixed before any such model exists" | Untrue for the B1 comparators: B1-LR+aug and B1-XGB+aug, seed 42, were trained before approval (EXP-018), and §9 says so. | §6: *"… is the seed-42 model. It is fixed here, before any B1-DL model exists; the seed-42 `+aug` B1 models of EXP-018 predate this rule and are disclosed in §9."* §9's summary: replace "fixed before any such model exists" with *"fixed before any B1-DL model exists"*. |
| 8 | §3.7 "Drawn once" | **The primed models' input is not covered by the validity check.** The digest covers the kept tokens. The "asked" sets also contain unlisted questions from a second permutation (A-9), so a changed NumPy stream could alter them without changing the digest. EXP-019 records a second digest, but `docs/05` does not require one. | After the digest sentence, add: *"A second digest, of the same lines with the questions asked (in evidence-code order) in place of the kept tokens, covers the input of models with the 'asked' channel; for them, a run that does not reproduce both digests is invalid."* |

## 2. Needed for the final report

| # | Where | The problem | Proposed correction |
|---|---|---|---|
| 9 | Header, frozen note | **Amendment 3 made a false claim explicit.** "Nothing in the frozen text or in amendments 1–2 was chosen with knowledge of results". But amendment 2's own §9 entry discloses that EXP-014 had already measured each red-flag rule's rate on its own patients. | *"…so nothing in the frozen text or in amendment 1 was chosen with knowledge of results. Amendment 2 was made knowing EXP-014's red-flag rates, and amendment 3 with some model results known; their §9 entries say which."* |
| 10 | §3.7 "Tuning" | **Calibration is ambiguous.** It allows "each level has its own" setting, then says reduced-level ECE uses "the same calibrated scores" with no refit. | *"…At 50% and 25%, ECE is reported for the calibrator fitted at 100%, applied unchanged; if a level has its own calibrator, fitted on validation at that level, its ECE is reported beside it."* |
| 11 | §4, "a well-tuned XGBoost" | The bar says "well-tuned", but B1-XGB is pinned to EXP-004's untuned defaults, and tuning parity (the proposal's decision 6) is not in the text. | Depends on decision **T-6**. If no tuning: add to §4's amendment: *"B1-XGB is EXP-004's model with fixed defaults and early stopping, and no ML model is tuned beyond early stopping (decision T-6); the first-written sentence is judged against that model."* |
| 12 | §3.7 "(§2 is unchanged)" | Amendment 3 added a row to §2. | *"(§2's test-access rule is unchanged)"* |
| 13 | §3.7 first sentence; §5 | Blanket "every system … at three levels" and "**all** §3 metrics" sit beside the exceptions: B3 and B4 "when compute allows", and §3.4 and §3.5 at 100% only. | §3.7: *"…is scored at three evidence levels, except as stated below for B3 and B4, on the same patients…"*. §5's amendment: *"…at all three evidence levels, on each §3 metric that §3.7 measures at that level."* |

## 3. Written before amendment 3, found in the same review

| # | Where | The problem | Proposed correction |
|---|---|---|---|
| 14 | §4, the B2 row | B2 still says "symptom-overlap / PageRank". The B2 actually scored, which amendment 3's §9 cites, is naive-Bayes (EXP-005 chose it over PageRank). No §9 entry records the change, and the frozen note requires one for any baseline change. | *"\| **B2** \| KG-only: the graph score (naive-Bayes over the crosswalk-closed graph, EXP-005; first written as symptom-overlap / PageRank) \| Is the graph useful alone? \|"* |
| 15 | §5, A5 | The matrix has no calibration column, so A5's row is identical to A0's. | Add a calibration column, or a note: *"A5 differs from A0 only in using uncalibrated scores."* |
| 16 | §1 H4 vs §5 A1 | H4 is about removing the medical KG, but A1 also switches off the red flags, so A1 cannot isolate the KG. | Add *"A1b — no medical KG, red flags kept"*, or state in §5 that H4 is judged on A1 and the confound is reported. The team chooses. |

---

## 4. A question EXP-019 raises (after its results, and said so)

| # | Where | The question | Options |
|---|---|---|---|
| 17 | §1 H1-R, H2-R; §7 reduced-evidence rows; §4 labels | **Which B1 do the reduced-evidence claims compare against?** §4's labels make B1-XGB and B1-LR the EXP-004 models, trained on full-evidence rows, while A0's ML component is trained on masked copies (`+aug`). EXP-019 shows the choice decides the result: at 25% the plain B1 has lots of room (B1-XGB top-3 0.906, must-not-miss recall@3 0.877), while the `+aug` B1 variants reach 0.997–0.998 on top-3 and 0.996–0.998 on must-not-miss recall@3. By §4's own rule, comparing a `+aug` A0 with a plain B1 "mixes a change of model with a change of … data". | (a) Keep the text: compare with the plain B1, and report the `+aug` comparison beside it, labelled as mixing. (b) **Also require H1-R, H2-R and the reduced-evidence Target to hold against the B1 variants trained like A0's component (`+aug`, and ′ for primed systems)**. This is recommended: it is the harder test and the fair one. Either way, the choice is made after EXP-019 and must be disclosed. |

---

## 5. The §9 row, if approved

```markdown
| 2026-09-__ | **Amendment 3a (errata to amendment 3): §§ header, 1, 3.7, 4, 5, 6 and, by item 17, 7, as listed in docs/proposals/05-amendment-3-errata.md.** Corrections of contradictions and gaps found in amendment 3's approved wording, and the B2 description; no metric, level, mask or threshold changes | Found by an independent review of the applied text on 2026-09-25, the day amendment 3 was approved, before any fusion or deep-model result existed. When the errata were approved, EXP-019 had scored B0, B1 and B2 at the three levels (descriptive, §9 amendment 3) | The team, 2026-09-__ (relayed by the owner) |
```
