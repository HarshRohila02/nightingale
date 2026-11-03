# `docs/05` amendment 3b: the exact wording, ready to apply (draft for the team)

**Drafted:** 2026-10-21, by Claude (a side session of the main session) · **Status:** **a draft
only. Nothing in this file is in force.** `docs/05` v1.4 (amendment 3a) stands as approved until the
team approves an amendment and the main session applies it, with a §9 entry · **Approval:** none yet;
the "approved by" cell of the §9 row (Part B) is left for the owner · **Decision:** E-2 in
`PROGRESS.md` §3 (due 2026-10-23) · **Source:** [05-amendment-3b-errata.md](05-amendment-3b-errata.md)

This file turns the errata proposal (item 1's four options, items 2–11) into the exact words to put
into `docs/05`, so that the main session can apply the team's answer within minutes. It does not
edit `docs/05`, `PROGRESS.md` or any other file, and it invents no result and no approval. The
figures it quotes come from the errata, `docs/05` and `docs/08`; none was recomputed (this worktree
has no `data/` or `models/`).

## 0. How to use it

1. The team answers E-2 item 1 with one of (a)–(d) and approves, amends or rejects each of items
   2–11 (the table in §1 below is the checklist).
2. **If (b) and items 2–11 are approved** (the recommendation): apply Part A as it stands, then the
   §9 row of Part B. **If (a):** apply Part A, but take item 1's edits from Part C instead, and the
   level wording Part C lists for items 2 and 3. **If (c) or (d):** Part C says what changes; the
   wording would need a further draft.
3. Each edit in Part A is a pair, **Old** and **New**, or an **Insert after** with its **New**.
   **Old** and **Insert after** are quoted from `docs/05` v1.4 exactly (line breaks aside; checked
   by a throwaway script, not committed, see §8), so they can be located by search. **New** is the text to put in its place.
4. **Marking.** As amendment 3a did, every changed or added passage carries *(amendment 3b)*;
   short replacements keep the replaced words struck through (~~like this~~), as `docs/05` §3.1 and
   §9 already do. Whole new paragraphs start with *Amendment 3b (…)*.
5. **Choices.** Where the errata gave alternatives or left the wording open, the draft picks one
   and says so under **Choice for the team**. Each is listed again in Part E. The team can change
   any of them; none is settled by approving "items 2–11" as such.
6. The edits are written not to overlap, so each item can be approved, amended or dropped on its
   own. The exception is the edits that depend on item 1 (marked "depends on E-2 item 1").

## 1. The items at a glance

| # | Subject | Passages of `docs/05` it changes | Depends on item 1? | Team |
|---|---|---|---|---|
| **1** | **(b): H1-R, H2-R, H6 and the reduced Target and Stretch are judged at 25%; 50% reported beside, not counted** | header; frozen note; §1 (three hypothesis cells, amendment 3a's paragraph, a new paragraph); §6 (a new bullet); §7 (two rows and amendment 3a's paragraph) | is item 1 | ☐ approve ☐ amend ☐ other option |
| 2 | One table of every claim's comparators | §4 (a new paragraph and two tables) | yes (the "judged at" column) | ☐ ☐ |
| 3 | The tests of each claim | §6 (a new bullet) | yes | ☐ ☐ |
| 4 | What ">" means in a hypothesis | §1 (a new paragraph) | no | ☐ ☐ |
| 5 | A pass has a direction | §6 (a new bullet) | no | ☐ ☐ |
| 6 | Which ECE counts against the target at 50% and 25% | §3.7, "Tuning" | no | ☐ ☐ |
| 7 | The digests' exact form and the scripts that compute them | §3.7, "Drawn once" | no | ☐ ☐ |
| 8 | The seed bullets | §4 (labels table); §6 (last bullet) | no | ☐ ☐ |
| 9 | What A1 removes | §5 | no | ☐ ☐ |
| 10 | EXP-006's rule pinned | §5 | no, but see the flag | ☐ ☐ (two wordings) |
| 11 | The disclosures that accompany a reduced-evidence result | §8, rule 8 | no | ☐ ☐ |

---

## 2. Part A: the recommended answer, item 1 (b) with items 2–11

### Item 1: judge H1-R, H2-R, H6 and the reduced Target and Stretch at 25%, with 50% beside them

#### 1.1 Header: version

**Old**

> **Version:** 1.4 · 2026-10-02 (amendments 1–3 and 3a, §9) · **Owner:** P4

**New**

> **Version:** 1.5 · 2026-__-__ (amendments 1–3, 3a and 3b, §9) · **Owner:** P4

*Version number: 3a moved 1.3 to 1.4, so this follows; the errata says nothing about it (Part E, 16).*

#### 1.2 Header: the list of amendments

**Old**

> Amended on 2026-10-02 (amendment 3a, the errata to amendment 3), with EXP-019's descriptive results known; its §9 entry says so

**New**

> Amended on 2026-10-02 (amendment 3a, the errata to amendment 3), with EXP-019's descriptive results known; its §9 entry says so. Amended on 2026-__-__ (amendment 3b, the follow-ups to amendment 3a), with the validation results of EXP-019 to EXP-023 known; its §9 entry says what else was known *(amendment 3b)*

#### 1.3 The frozen note

**Old**

> Amendment 3a was approved knowing EXP-019's descriptive results, and its §9 entry says so *(amendment 3a)*.

**New**

> Amendment 3a was approved knowing EXP-019's descriptive results, and its §9 entry says so *(amendment 3a)*. Amendment 3b was approved knowing the validation results of EXP-019 to EXP-023, those of the deep ranker included, and its §9 entry says so *(amendment 3b)*.

*The two "known" statements in 1.2 and 1.3 must be true on the day; check them when applying (§8).*

#### 1.4 §1, the hypothesis cells (three edits)

**Old**

> **H1-R** *(amendment 3)*: the same, at **both** 50% and 25% evidence (§3.7), against every ML-only system

**New**

> **H1-R** *(amendment 3)*: the same, at ~~**both** 50% and 25% evidence~~ 25% evidence, the 50% comparison reported beside it and not counted *(amendment 3b)* (§3.7), against every ML-only system

**Old**

> **H2-R** *(amendment 3)*: the same, at **both** 50% and 25% evidence, against every ML-only system

**New**

> **H2-R** *(amendment 3)*: the same, at ~~**both** 50% and 25% evidence~~ 25% evidence, the 50% comparison reported beside it and not counted *(amendment 3b)*, against every ML-only system

**Old**

> at **both** 50% and 25% evidence. At 100% the comparison is reported

**New**

> at ~~**both** 50% and 25% evidence~~ 25% evidence, the 50% comparison reported beside it and not counted *(amendment 3b)*. At 100% the comparison is reported

#### 1.5 §1, amendment 3a's paragraph: a pointer, and its last sentence

**Old**

> *Amendment 3a (errata item 17).*

**New**

> *Amendment 3a (errata item 17).* *(Amendment 3b, below, replaces this paragraph's consequence for H1-R, H2-R, H6 and the reduced-evidence rows: they are judged at 25%. The paragraph stays as the record of what amendment 3a disclosed.)* *(amendment 3b)*

**Old**

> The approval cited only the 25% figures, although EXP-019 had published the 50% ones, so the team is asked again (decision E-2, `docs/proposals/05-amendment-3b-errata.md`); until it answers, the text above stands as approved, and under it §8 rule 5 requires the report to state these failures plainly.

**New**

> The approval cited only the 25% figures, although EXP-019 had published the 50% ones, so the team ~~is asked~~ was asked again (decision E-2, `docs/proposals/05-amendment-3b-errata.md`)~~; until it answers, the text above stands as approved, and under it §8 rule 5 requires the report to state these failures plainly~~. The team's answer is amendment 3b, below *(amendment 3b)*.

#### 1.6 §1, a new paragraph

**Insert after** the last sentence of amendment 3a's paragraph (the one edited in 1.5), before "**Negative results are reportable results.**":

**New**

> *Amendment 3b (decision E-2, item 1, option (b)).* H1-R, H2-R, H6 and the reduced-evidence Target and Stretch (§7) are judged at **25% evidence**. Every one of their comparisons is also run at 50%, with its test and its interval, and the 50% results are reported beside the 25% verdict, labelled *descriptive*: they are not counted in the verdict, and a 50% result that goes against the claim is reported plainly beside it (§8 rule 5). The report states, for each 50% comparison, how many cases its comparator misses on the test split, so that a reader can see where the comparator is at ceiling. H1–H5, the Minimum, the Target and the Stretch keep their level, 100% (amendment 3, above). This choice was made knowing EXP-019's comparator figures at 50% and 25%, and the other results the §9 entry lists *(amendment 3b)*.

*Choice for the team (Part E, 5): "a 50% result that goes against the claim is reported plainly beside it, and does not change the verdict". The errata says "reported beside them, descriptively" and does not say whether an adverse 50% result can defeat a claim. The draft says it cannot.*

#### 1.7 §6, the levels that count

**Insert after** the sentence ending "…between A0 and B1-DL and between A0 and B2 *(amendment 3a)*." (the MRR bullet), as a new bullet at the same indent:

**New**

> - **Levels that count** *(amendment 3b)*. For H1-R, H2-R, H6 and the reduced-evidence Target and Stretch, the comparisons that count are those at 25% evidence. Those at 50% are run and reported as §1 says, and are not counted.

#### 1.8 §7, the two reduced-evidence rows

**Old**

> At **both** 50% and 25% evidence: A0 > **each** B1 variant on Top-3 **and** must-not-miss recall@3

**New**

> At ~~**both** 50% and 25% evidence~~ 25% evidence, the 50% comparison reported beside it *(amendment 3b)*: A0 > **each** B1 variant on Top-3 **and** must-not-miss recall@3

**Old**

> must-not-miss recall@3 ≥ 0.95 at both levels

**New**

> must-not-miss recall@3 ≥ 0.95 at ~~both levels~~ 25% evidence *(amendment 3b)*

#### 1.9 §7, amendment 3a's paragraph

**Old**

> On EXP-019's validation figures those are perfect or within four misses at 50%, and both rows need 50% and 25%, so on those figures neither row can be met, as §1 discloses with what it means on the test split (decision E-2 asks the team whether that stands).

**New**

> On EXP-019's validation figures those are perfect or within four misses at 50%~~, and both rows need 50% and 25%, so on those figures neither row can be met, as §1 discloses with what it means on the test split (decision E-2 asks the team whether that stands)~~. Amendment 3b (decision E-2) judges both rows at 25%, with the 50% comparison reported beside them (§1) *(amendment 3b)*.

**Notes on item 1.** (b) is the errata's recommendation. It was recommended on 2026-10-02, when no deep result existed; the team will answer with EXP-021's validation figures on record (read 2026-10-13), and that bears on H6 (Part E, 3). Even at 25% the claims are testable, not likely: the `+aug` comparators miss 57–111 top-3 and 38–71 must-not-miss cases there (errata §1), so a fused system must repair a significant share of those. If the test split's 25% comparators missed five or fewer cases, (b) has no fallback (Part E, 4).

---

### Item 2: one table of every claim's comparators

**Insert after** the last paragraph of §4, which ends "Any other pairing is reported as mixing a change of model with a change of encoding or data.":

**New**

> *Amendment 3b (errata item 2).* The comparators of every claim are the models in the first table below, and no others; every model is the seed-42 model (§6). The second table gives, for each claim, the level it is judged at, the claimed system, its comparators and its metrics. "A0" stands for A0 and, in the primed column of the first table, A0′: §7 judges the two separately. A comparator's arm is fixed by these tables, not chosen by the reader. A primed system is compared with the plain models and with the primed models trained like it, an unprimed system with the plain models and the unprimed ones; no system is compared with the models of the other prime *(amendment 3b)*.
>
> | Comparator set | For an unprimed system (A0) | For a primed system (A0′) |
> |---|---|---|
> | Plain B1 | B1-XGB, B1-LR | B1-XGB, B1-LR |
> | Trained like it | B1-XGB+aug, B1-LR+aug | B1-XGB′+aug, B1-LR′+aug |
> | Deep ranker | B1-DL+aug | B1-DL′+aug |
> | KG-only | B2 | B2 |
>
> | Claim | Judged at | Claimed system | Compared with | Metrics |
> |---|---|---|---|---|
> | H1 | 100% | A0 | verdict: B1-XGB and B2; reported beside: B1-LR and the deep ranker | Top-3, MRR |
> | H2 | 100% | A0 with its red flags | verdict: B1-XGB; reported beside: B1-LR and the deep ranker | must-not-miss recall@3 |
> | H1-R | 25%; 50% beside | A0 | plain B1, trained like it, the deep ranker, KG-only | Top-3, MRR |
> | H2-R | 25%; 50% beside | A0 with its red flags | plain B1, trained like it, the deep ranker | must-not-miss recall@3 |
> | H6 | 25%; 50% beside | the deep ranker | trained like it | Top-3, MRR, must-not-miss recall@3 |
> | Target | 100% | A0 | B1-XGB; B1-LR beside | Top-3, must-not-miss recall@3 (point estimates); ECE below 0.10; unsupported-claim rate below 20% |
> | Stretch | 100% | A0 | B1-XGB; B1-LR beside | the Target's criteria, with every comparison significant; must-not-miss recall@3 ≥ 0.95; clinician-reviewed explainability |
> | Target, reduced evidence | 25%; 50% beside | A0 | plain B1, trained like it | Top-3, must-not-miss recall@3 (point estimates) |
> | Stretch, reduced evidence | 25%; 50% beside | A0 | plain B1, trained like it | the reduced Target, with every comparison significant; must-not-miss recall@3 ≥ 0.95 |

**Choices for the team** (Part E, 2 and 6):
- *An unprimed A0 is not compared with the ′+aug models.* The errata says it is unclear. §4's last paragraph calls a comparison across encoders "mixing", and 3a's own words say "for primed systems" when it names the ′+aug models, so the draft reads them that way.
- *The reduced Target and Stretch have no deep-ranker or KG-only comparator,* because their wording is "each B1 variant", and §4 lists B1-DL and B2 as separate baselines. H1-R and H2-R do name B1-DL, and H1-R names B2 (3a, §6).
- *A primed A0′ is compared with the plain B1* as 3a's text says, although §4 would call that mixing a change of encoder with a change of model.
- *The 100% Stretch includes the Target's ECE and unsupported-claim criteria* (see item 3).

---

### Item 3: the tests of each claim

**Insert after** the bullet added in 1.7 (or, if item 1 is not approved, after the MRR bullet), as a new bullet at the same indent:

**New**

> - **The tests of each claim** *(amendment 3b)*. For each comparison in §4's tables, Top-3 is compared by McNemar's test on paired top-3 correctness and MRR by the paired bootstrap above; for H1 and H1-R both are run against every comparator the table lists, including B2 and the deep ranker. Must-not-miss recall@3 is compared by McNemar's test on top-3 correctness restricted to must-not-miss cases; for H2 and H2-R it is the only test the verdict rests on, and the test on all patients is reported beside it. H6 tests all three metrics. A Target compares point estimates (above). The Stretch at 100% includes the Target's other criteria (ECE below 0.10, unsupported-claim rate below 20%) as well as its own. H3 and H4 have no test: they are judged on point estimates, each with its 95% bootstrap interval. H3 is supported when the grounded system's unsupported-claim rate is lower than the LLM-only baseline B3's, and H4 when removing the medical KG (A1) lowers must-not-miss recall@3 by more than it lowers Top-1; the report says whether the interval of each difference excludes 0, and neither is called significant.

**Choices for the team** (Part E, 7 and 8):
- *The errata offers two ways to treat H3 and H4:* a list of tests, or "judged on point estimates with intervals". The draft takes the second, which invents no test. Its wording of what H3 and H4 compare is the hypotheses' own words; **naming B3 as H3's comparator is the draft's reading** of "LLM-only" (the alternative is A3, no RAG).
- *H2 and H2-R rest on the restricted test only.* The 3a text says McNemar is run between A0 and each B1 variant and, for H2-R and H6, "also" on the restricted cases, and that a hypothesis needs every comparison to pass. Read literally, H2-R would also need the all-patient top-3 test to pass, which is not H2's metric. The draft reports it beside the verdict.
- *The Stretch at 100% is cumulative* (it includes the Target's ECE and unsupported-claim criteria). The errata asks the question without a recommendation. The alternative, if the team wants the Stretch as the three criteria it lists: replace the last sentence about the Stretch by "The Stretch at 100% has only the criteria §7 lists for it."

---

### Item 4: what ">" means in a hypothesis

**Insert after** the sentence ending "…Top-1 is reported at every level." (the end of the *Amendment 3.* paragraph of §1), as a new paragraph:

**New**

> *Amendment 3b (errata item 4).* In H1, H1-R, H2, H2-R and H6, ">" means that every comparison the hypothesis lists (§4's tables) passes its test at α = 0.05, in the claimed system's favour (§6). H1 and H2 were first written with a bare ">", and amendment 3 says that H1–H5 "keep their first-written meaning"; amendment 3a's item 3 settled that a hypothesis needs every one of its comparisons to pass, and this entry records that this is how H1 and H2 are read. H3 and H4 are judged as §6 says *(amendment 3b)*.

**Choice for the team** (Part E, 7): item 4 as the errata words it ("a hypothesis's '>' means significant") would also reach H3 and H4, which item 3 may leave without a test. The draft limits it to the five hypotheses that have tests.

---

### Item 5: a pass has a direction

**Insert after** the bullet added in item 3 (or, if item 3 is not approved, after the MRR bullet), as a new bullet at the same indent:

**New**

> - **A pass has a direction** *(amendment 3b)*. A comparison passes only when its test is significant at α = 0.05 (McNemar: two-sided p < 0.05; MRR: the 95% interval of the paired difference excludes 0) **and** the system the claim favours is the better one: the one that is right in more of the discordant pairs, or has the higher mean reciprocal rank. A significantly worse system fails the comparison, and the report says it is worse. This defines "passes" in the bullets above.

`paired_bootstrap_difference` already reports `excludes_zero` and `better` separately, so the code loses nothing.

---

### Item 6: which ECE counts against the target at 50% and 25%

**Old**

> if a level has its own calibrator, fitted on validation at that level, its ECE is reported beside it *(amendment 3a)*.

**New**

> if a level has its own calibrator, fitted on validation at that level, its ECE is reported beside it *(amendment 3a)*. The ECE counted against the target (§3.3) at 50% and 25% is that of the calibration setting the pipeline will use, declared in docs/08 before the test split opens (the calibrator fitted at 100% and applied unchanged, which may leave the score as it is, or a level's own); the other is reported beside it *(amendment 3b)*.

**Choice for the team** (Part E, 13): a level's own calibrator cannot be the pipeline's setting for a real case, which has no level (EXP-007 says so). In practice the 100% calibrator counts. If the team prefers it said: replace the parenthesis by "(the calibrator fitted at 100% and applied unchanged, which may leave the score as it is)", and "the other" by "a level's own calibrator, if it has one,".

---

### Item 7: the digests' exact form

Three edits to §3.7, "Drawn once".

**Old**

> Their digest, the NumPy version and the commit are recorded in docs/08 before any fusion or deep-model result is scored on them.

**New**

> Their digests (both, below), the NumPy version and the commit are recorded in docs/08 before any fusion or deep-model result is scored on them *(amendment 3b)*.

**Old**

> The digest is the SHA-256 of one UTF-8 line per patient and reduced level

**New**

> The digest is the first 16 hexadecimal digits *(amendment 3b)* of the SHA-256 of one UTF-8 line per patient and reduced level

**Old**

> `scripts/check_mask_rules.py` computes it this way. A second digest, of the same lines with the questions asked (in evidence-code order) in place of the kept tokens, covers the input of models with the 'asked' channel; for them, a run that does not reproduce both digests is invalid *(amendment 3a)*.

**New**

> `scripts/check_mask_rules.py` computes it this way. A second digest, of the same lines with the questions asked (~~in evidence-code order~~ sorted by the number after `E_`, so that `E_2` comes before `E_10`, and joined by single spaces *(amendment 3b)*) in place of the kept tokens, covers the input of models with the 'asked' channel; for them, a run that does not reproduce both digests is invalid *(amendment 3a)*. `scripts/check_mask_rules.py` computes only the first digest; `digests()` in `scripts/evaluate_reduced_evidence.py` computes both, and `scripts/sweep_fusion.py` calls it *(amendment 3b)*.

*Checked in the code: `digest()` takes `hexdigest()[:16]` of the SHA-256 of the sorted lines joined by `\n`; `digests()` sorts the asked questions with `code_sort_key` (numeric) and joins them with a space; `check_mask_rules.py` prints `digest[:16]` of the token lines only. The errata says the asked digest is "in `scripts/evaluate_reduced_evidence.py` and `scripts/sweep_fusion.py`": precisely, it is computed by the first and called by the second (Part E, 11).*

---

### Item 8: the seed bullets

**Old**

> Always reported, and never retrained under the same label, except that §6's seeds 43–46 are trained on the same rows and encoder and reported beside the seed-42 model, which stays the one used *(amendment 3a)*

**New**

> Always reported, and never retrained under the same label, except that ~~§6's~~ B1-XGB's seeds 43–46 (§6) are trained on the same rows and encoder and reported beside the seed-42 model, which stays the one used *(amendment 3a)*; B1-LR on fixed rows is deterministic and takes no seed, and a refit of seed 42 is a reproducibility check that is never reported under the label *(amendment 3b)*

**Old**

> That gap is disclosed, and it is closed when B1-XGB is next retrained.

**New**

> That gap is disclosed, and it is ~~closed when B1-XGB is next retrained~~ closed by training B1-XGB's seeds 43–46 (EXP-020), reported beside the seed-42 model; EXP-020's refit of seed 42 is a reproducibility check and is never reported as B1-XGB *(amendment 3b)*.

---

### Item 9: what A1 removes

**Insert after** the sentence ending "…without it no rule could fire." (the end of §5's amendment 3a paragraph):

**New**

> A1 removes the same two things and also switches the red flags off; the crosswalk is not part of the medical KG, so neither configuration removes it *(amendment 3b)*.

*Not settled by this sentence, and flagged (Part E, 12): the errata asks whether A1's ML ranker still reads a hand-written case through the crosswalk. On DDXPlus patients, which the evaluation uses, the ranker reads their tokens directly; for hand-written cases it reads `src/ml/case_tokens.py`'s output, which rests on the crosswalk's tables. That is an implementation detail for the ablation runner, not a protocol rule, so the draft does not write it.*

---

### Item 10: EXP-006's rule pinned (two wordings; the team picks one)

**Old**

> Each fused configuration's weight is chosen for its own ML component by the rule EXP-006 records; A7 at A0's weight is reported beside it (errata item 18).

**New, 10-i (the errata's wording: the rule as committed with amendment 3a)**

> Each fused configuration's weight is chosen for its own ML component by the rule EXP-006 records as committed with amendment 3a (`eb3b7d4`); any later change to that rule needs a §9 entry *(amendment 3b)*. A7 at A0's weight is reported beside it (errata item 18).

**New, 10-ii (the rule as committed when amendment 3b is applied, the dated note of 2026-10-13 included; the draft's recommendation)**

> Each fused configuration's weight is chosen for its own ML component by the rule EXP-006 records as committed in `<hash of the commit that applies amendment 3b>`, which includes the dated note of 2026-10-13 (decision D-13: B1-DL′+aug and B1-DL+aug are swept like the other components, and the pipeline takes B1-DL′+aug's weight); any later change to that rule needs a §9 entry *(amendment 3b)*. A7 at A0's weight is reported beside it (errata item 18).

**Why two.** EXP-006 changed after 3a: `c677d22` (2026-10-13, D-13) added a dated note, "before any sweep" and with no §9 entry, that sweeps six components instead of four, makes B1-DL′+aug the component whose α the pipeline adopts, and extends the golden gate to the other models the golden tests run (the script's docstring changed with it). The 3a-era text already said B1-DL+aug "will be swept the same way when it exists", so 10-i can be read as covering the note as an instance of the rule. But if the team reads 10-i as freezing the text at `eb3b7d4`, the sweep as it is now written would depart from the pinned rule and would itself need a §9 entry, which is what item 10 forbids without one. 10-ii pins what will actually run. **The team should say which.** The grid, the objective, the constraint, the golden gate and the tie rule are the same in both (Part E, 1).

---

### Item 11: the disclosures that accompany a reduced-evidence result

**Old**

> and never without the amendment-3 disclosure (§9)

**New**

> and never without the amendment-3 and 3a disclosures (§9) *(amendment 3b)*

*Draft the errata's wording, nothing more. Flag (Part E, 14): amendment 3b's own entry also carries a disclosure (item 1 was chosen knowing the comparator figures). If the team wants it to travel with 25%-only verdicts too, the wording is "the amendment-3, 3a and 3b disclosures (§9)".*

---

## 3. Part B: the new §9 row

To append to `docs/05` §9 after amendment 3a's row, with the date and the bracketed items filled at
approval. The justification states what was known, as 3a's and amendment 3's did. **No approval is
claimed: the last cell is blank for the owner.**

```markdown
| 2026-__-__ | **Amendment 3b (decision E-2): the header and §§1, 3.7, 4, 5, 6, 7 and 8, as listed in `docs/proposals/05-amendment-3b-applied-draft.md` (from `docs/proposals/05-amendment-3b-errata.md`); each changed passage is marked *(amendment 3b)*.** Item 1, option (b): H1-R, H2-R, H6 and the reduced-evidence Target and Stretch are judged at 25% evidence, and their 50% comparisons are run and reported beside them, labelled descriptive and not counted. Items 2–11: one pair of tables naming the comparators of every claim and its metrics (2); the test of each claim, with H3 and H4 judged on point estimates and intervals, and the 100% Stretch including the Target's criteria (3); what ">" means in a hypothesis (4); that a pass needs the claimed system to be the better one (5); which ECE is counted against the target at reduced levels (6); the digests' exact form and the scripts that compute them (7); the seed bullets (8); what A1 removes (9); EXP-006's rule pinned (10, wording [10-i / 10-ii]); the disclosures that accompany a reduced-evidence result (11). No metric, threshold, level, mask, baseline or ablation changes; the level at which six claims are judged, and the comparators of each claim, are fixed. | Items 2–11 were found by the independent verifiers of amendment 3a's applied text on 2026-10-02, the day it was approved and applied. Item 1 comes from EXP-019's 50% figures, published on 2026-09-25 before amendment 3a was approved but not cited in its approval: at 50% the `+aug` comparators miss at most four of 33,963 top-3 cases and three of 16,943 must-not-miss cases, so a comparison there tests nothing, as at 100% (R-16), while at 25% they miss 57–111 and 38–71. **What was known when the team answered** *(to confirm at approval)*: the validation results of EXP-019 (B0, B1, the four `+aug` B1 variants, B2 and the red-flag layer at the three levels), EXP-020 (seeds 43–46 of the same B1 variants; every EXP-019 conclusion holds at every seed), EXP-021 (the small transformer B1-DL, all ten networks scored on validation at the three levels; D-13 passed on 2026-10-13, so A0's ML component is B1-DL+aug and A0′'s B1-DL′+aug; at 50% its top-3 is saturated, at 25% its MRR is higher than the logistic regression's at nine of ten networks and level with XGBoost trained like it), EXP-022 (latency), EXP-023 (the components' errors at the three levels), EXP-007's first stage, and the four hand-written golden cases' fused ranks (EXP-006, red flags off). *[Confirm at approval: no fused ranking of a DDXPlus patient had been scored, EXP-006's sweep had not been run, and the test split had never been opened.]* Option (b) is the one recommended on 2026-10-02, before any deep result existed; the team answered with EXP-021's figures on record, which were read on 2026-10-13. Choosing 25% as the only level at which H6 is judged means choosing the level at which EXP-021 already shows a small MRR gain over the logistic regression (+0.0018 [+0.0009, +0.0026] for B1-DL′+aug), the kind of choice amendment 3 warns about. *(To fill at approval: whether the team weighed this.)* **Choices the errata left open, made by Claude when drafting the wording, and in force only if the team approved the wording:** the comparator tables (an unprimed system is not compared with the ′+aug models; the reduced Target and Stretch have no deep-ranker or KG-only comparator), H2 and H2-R resting on the restricted must-not-miss test alone, H3 and H4 on point estimates and intervals with B3 named as H3's comparator, the cumulative 100% Stretch, two-sided tests, the ECE counted being the pipeline's declared setting, and the version number 1.5. The team may change any of these by a further entry before a result depends on it | [to be filled in by the owner at approval] |
```

---

## 4. Part C: if the team answers item 1 (a); and what (c) and (d) would change

### C1. Option (a): keep the text, report the verdicts as unreachable

Apply Part A **except** the edits of item 1 (1.4 to 1.9), which are replaced by the edits below; items
4–11 are unchanged, and items 2 and 3 change only the words listed under C1.6. Edits 1.1 to 1.3
(header, frozen note) stand, with "Amended on … (amendment 3b …)" unchanged.

**C1.1 §1, the three hypothesis cells.** No edit: H1-R, H2-R and H6 keep "at **both** 50% and 25%
evidence".

**C1.2 §1, amendment 3a's paragraph: the pointer (1.5, first edit).** Not made.

**C1.3 §1, amendment 3a's paragraph: its last sentence.**

**Old**

> The approval cited only the 25% figures, although EXP-019 had published the 50% ones, so the team is asked again (decision E-2, `docs/proposals/05-amendment-3b-errata.md`); until it answers, the text above stands as approved, and under it §8 rule 5 requires the report to state these failures plainly.

**New**

> The approval cited only the 25% figures, although EXP-019 had published the 50% ones, so the team ~~is asked~~ was asked again (decision E-2, `docs/proposals/05-amendment-3b-errata.md`)~~; until it answers, the text above stands as approved, and under it §8 rule 5 requires the report to state these failures plainly~~. The team answered (option (a)): the text above stands as approved, and under it §8 rule 5 requires the report to state these failures plainly, as amendment 3b, below, says *(amendment 3b)*.

**C1.4 §1, the new paragraph (replaces 1.6).**

**Insert after** the last sentence of amendment 3a's paragraph, before "**Negative results are reportable results.**":

**New**

> *Amendment 3b (decision E-2, item 1, option (a)).* The team kept the text of H1-R, H2-R, H6 and the reduced-evidence Target and Stretch (§7) as approved in amendment 3a: each needs both 50% and 25% evidence. For each of them the report states how many cases each comparator misses at 50% on the test split. Where a McNemar comparison cannot reach α = 0.05 (the comparator misses five cases or fewer), the report says that the claim could not be supported at 50%, and why (the comparator's ceiling, R-16); such a claim is reported as not supported, with its 25% comparisons beside it. RQ1, RQ2 and RQ6 are then answered descriptively wherever a claim could not be supported *(amendment 3b)*.

**C1.5 §6, "Levels that count" (replaces 1.7).**

> - **Levels that count** *(amendment 3b)*. For H1-R, H2-R, H6 and the reduced-evidence Target and Stretch, the comparisons that count are those at 50% and at 25% evidence (§1).

**§7 rows (1.8): no edit.** **§7, amendment 3a's paragraph (1.9):**

**Old**

> (decision E-2 asks the team whether that stands).

**New**

> (decision E-2 asked the team whether that stands; it kept the text, amendment 3b) *(amendment 3b)*.

**C1.6 Items 2 and 3.** In item 2's claims table, replace "25%; 50% beside" by "50% and 25%" in the rows H1-R, H2-R, H6, Target (reduced evidence) and Stretch (reduced evidence). In item 3's bullet no change. In item 6 no change.

**C1.7 §9 row.** Replace the item 1 sentence by: "Item 1, option (a): the text of H1-R, H2-R, H6 and the reduced-evidence Target and Stretch is kept; they stay judged at both 50% and 25%, and a claim with a comparison that cannot be tested is reported as not supported, with the reason." Delete the sentence in the justification about "Choosing 25% as the only level for H6". Replace "the level at which six claims are judged… are fixed" by "the comparators of each claim are fixed".

### C2. What option (c) would change

(c) returns to item 17's option (a) at 50%: H1-R, H2-R and the reduced Target would compare a `+aug` A0 with the plain B1 at 50%.
- §1: H1-R, H2-R cells and amendment 3a's paragraph, to say that the `+aug` B1 variants are comparators at 25% only (3a's reading applies at 25%, the original item 17 (a) at 50%).
- §4: the comparator table gets a level-dependent row ("Trained like it": 25% only), and the "mixing a change of model with a change of encoding or data" sentence needs an exception for 50%, which contradicts its reason.
- §7: the reduced Target's wording, likewise.
- **H6 stays unreachable at 50%**, because its comparators are the `+aug` models by definition (3a, item 6): (c) needs (a), (b) or (d) beside it for H6. The plain B1 variants are not at ceiling at 50% on validation (top-3 0.991 for B1-XGB and 0.997 for B1-LR, so about 0.9% and 0.3% of patients missed), so a comparison with them is testable; nobody has computed whether A0 would win it.
- The runner's verdict logic gets a comparator set that depends on the level.

### C3. What option (d) would change

(d) treats a comparison as untestable, not failed, when its comparator misses five or fewer cases on the test split; a claim is judged on its testable levels.
- §6 gets a bullet defining "untestable": which cases count (top-3 misses for the Top-3 test; must-not-miss misses for the restricted test; MRR has no such threshold and the errata does not say), and that a claim with no testable level is reported as untestable, not failed.
- §1 and §7 say that H1-R, H2-R, H6 and the reduced rows are judged on their testable levels.
- The rule is applied to counts on the test split, which nobody has seen, so its §9 entry must say it was decided before and applies after (the errata's own point).
- The verdict becomes three-valued (supported, not supported, untestable), and the runner must compute the comparators' miss counts.
- (b) and (d) are not exclusive in principle; the errata lists them as alternatives (Part E, 4).

---

## 5. Part D: everything outside `docs/05` that depends on the answer

None of these is edited by this file. "Now" is the current wording, quoted or described; (c) and (d) differ from (a) only where noted.

### D1. Scripts and tests

| Where | Now | Under (b) | Under (a) |
|---|---|---|---|
| `scripts/sweep_fusion.py`, module docstring, lines 5–6 | "**This is tuning on project data, so it runs only where the owner says (D-7), and only after the team answers E-2's first item** (which level the reduced-evidence claims are judged at)." **There is no gate in the code**: no flag, no check; the gate is this sentence, `docs/08` and the owner (D-7). `tests/test_sweep_fusion.py` has no E-2 text | Replace "and only after the team answers E-2's first item (…)" by "E-2 was answered on 2026-__-__ (option (b)): the reduced-evidence claims are judged at 25% (docs/05 amendment 3b)". **The sweep's objective is unchanged** (the mean over the three levels of MRR, one α for every level); whether it should become 25%-only is not asked by the errata and would need a §9 entry under item 10 (Part E, 9) | Same, with "option (a): judged at both levels, the text of amendment 3a kept". (c), (d): same as (a) |
| `scripts/sweep_fusion.py`, what it reports | McNemar on top-3 (all patients, and must-not-miss patients) for fused against the ML component alone, fused against B2, and fused plus red flags against the ML component alone, each with the paired bootstrap of the per-case MRR difference | Unchanged, but item 5 makes every comparison report which system is better; the comparisons against the plain and `+aug` B1 variants, B1-DL and the claims' pass or fail belong to EXP-012, not the sweep | Same |
| `scripts/error_analysis.py` lines 11 and 285; `tests/test_error_analysis.py` line 261 (docstring); `scripts/calibrate.py` line 218 | Strings and docstrings saying "E-2 is unanswered" / "waits for E-2's first item". The test checks by source search that the script scores no fused ranking; that stays true | Stale wording only, no code change: "E-2 was answered; the fused ranking waits for EXP-006" | Same |

### D2. `docs/08`

| Where | Now | Under (b) / (a) |
|---|---|---|
| EXP-006, "Status: the sweep waits for two things", item (1) | Waits for "the team's answer to E-2's first item" | Strike (1) the way E-1 is struck, and add "settled 2026-__-__: option (b) [or (a)] (`docs/05` amendment 3b)"; item (2), the owner's choice of where (D-7), stays |
| EXP-006, "What this cannot decide" | "No H1-R or H2-R verdict: A0 does not exist yet…" | (b): add that these are judged at 25% (amendment 3b). (a): no change. The rule's objective is unchanged unless the team says (Part E, 9) |
| `docs/08` line 138 (EXP-023), lines 303–306 (EXP-007), index row at 2462 | "the fused system's errors wait for E-2 and EXP-006"; "before the team answers E-2" (EXP-007's `--fused` paragraph); "waits for E-2's first item, then D-7" (EXP-006's index row) | "waits for EXP-006" / "waits for D-7"; a dated note, not an edit of the history |
| New entry | EXP-012 and EXP-009/010 are not yet logged | (b): EXP-012's pre-registration lists the claims, comparators and levels of item 2's tables (see D6). (a): the same, with both levels |

### D3. `CLAUDE.md`

| Where | Now | Under (b) | Under (a) |
|---|---|---|---|
| The "Amendment 3a" paragraph, "Hard rules" | "…the reduced-evidence claims (H1-R, H2-R, the reduced Target) must also hold against the `+aug` B1 variants; MRR is compared by a paired bootstrap …; a model with the "asked" channel needs both mask digests to reproduce." | Add: "**Amendment 3b (v1.5, 2026-__-__, E-2)**: H1-R, H2-R, H6 and the reduced Target and Stretch are judged at 25% only, the 50% comparison reported beside them and not counted; every claim's comparators are the tables in `docs/05` §4; a comparison passes only if the claimed system is the better one." | Add: "**Amendment 3b (v1.5, …)**: the reduced-evidence claims stay judged at both 50% and 25%, as 3a wrote; a claim whose comparator misses five or fewer cases at 50% on the test split is reported as not supported, with the reason." |
| The calibration/fusion gotcha | "…EXP-006's pre-registered sweep (it waits for E-2's first item, then D-7; until then an untuned 0.5)" | "(it waits for D-7; until then an untuned 0.5)" | Same |

### D4. `docs/18-project-report.md` (and `docs/18-project-report.pdf`, regenerated after it)

| Lines | Now | Under (b) | Under (a) |
|---|---|---|---|
| 118, 119, 123 (the RQ table) | "**H1-R:** the same at 50% and 25% evidence"; "**H2-R:** at 50% and 25% too"; "**H6:** … at 50% and 25%" | "at 25% (50% reported beside, not counted)" | unchanged |
| 129–130 | "As written, the reduced-level claims need both 50% and 25%, and at 50% the comparators are … (decision **E-2**, still open; §12.3)" | State the outcome: judged at 25% by amendment 3b | State the outcome: kept; reported as not supported where untestable |
| 42, 108, 521, 800, 819, 878, 930, 939–940 | "waits on decision E-2" (the experiments table, O4, the sweep, §12.2, §12.3) | "E-2 answered (b); the sweep waits for the owner's choice of where it runs"; drop E-2 from §12.3's blocking list | Same, with (a) |

### D5. `docs/report/` (the report draft, chapters 0–8)

| Where | Now | Under (b) | Under (a) |
|---|---|---|---|
| `05-evaluation.md` changelog table (line 21–22) | Rows to version 1.4 | Add a row 1.5, "Amendment 3b (E-2)", with what was known (Part B) | Same, with (a) |
| `05-evaluation.md` §5.9 (226–253) | "The open decision E-2", with the four options and "the decision is the team's, and it had not been made on 2026-10-14" | Retitle "Decision E-2 and its outcome"; keep the facts; replace the status bullet with the answer | Same, with (a) |
| `05-evaluation.md` Table 5.5, lines 202–203 | "At **both** 50% and 25% evidence …"; "… ≥ 0.95 at both levels" | "25% evidence, the 50% comparison reported beside it"; "at 25%" | unchanged |
| `05-evaluation.md` §5.7 (statistics) | The McNemar and bootstrap bullets; "passing when the 95% interval excludes 0" | Add the levels that count and the direction of a pass (items 1, 3, 5) | Add the direction of a pass |
| `05-evaluation.md` §5.4 (digests) | The digest table and "the questions asked" | Add the exact form (first 16 hex digits; sorted by the number after `E_`) (item 7) | Same |
| `01-introduction.md` lines 101–106 and 115–118 | The hypothesis table with "at **both** 50% and 25%"; "The team has been asked again … had not answered on 2026-10-13" | The table with 25%; the bullet states the outcome | table unchanged; the bullet states the outcome |
| `06-results.md` Table 6.27 (lines 833–840) and lines 513–514 | "As approved, cannot be met as a whole … E-2 open"; "…read on 2026-10-13, before the team answered E-2" | "Judged at 25%; the 50% comparison descriptive"; keep the disclosure and add that the answer followed it | "As approved …; the team kept the text" |
| `07-limitations-ethics.md` lines 105–109, the disclosure table (rows 2026-10-02, 2026-10-13, "[pending] E-2's answer") | "the team has been asked again (E-2)"; "[pending] E-2's answer" | Fill the pending row; add the disclosure in Part B | Same, with (a) |
| `08-conclusion.md` lines 25–37 | "pending the team's decision E-2 (§5.9)" | State the outcome | Same |

### D6. The verdict logic a future ablation runner must implement (`scripts/evaluate_ablations.py`, EXP-012; `PROGRESS.md` 5.1)

The runner reads item 2's two tables as data. For each claim and each prime (A0, A0′):
1. **Levels counted.** Under (b): H1, H2, Target, Stretch at 100%; H1-R, H2-R, H6, reduced Target, reduced Stretch at 25%. Under (a): the last five at both 50% and 25%.
2. **Comparators and metrics** from the tables: Top-3 by McNemar, MRR by the paired bootstrap, must-not-miss recall@3 by McNemar on the restricted cases (with the all-patient test beside it for H2 and H2-R); H3 and H4 by point estimate and interval.
3. **A pass** = significant at α = 0.05 (two-sided) **and** the claimed system better (item 5); a significantly worse system fails.
4. **A hypothesis is supported, a Stretch met,** only if every counted comparison passes; a Target compares point estimates and reports its tests beside it.
5. **Under (b), the 50% comparisons** of the five reduced claims are run into a separate *descriptive* table, with each comparator's miss count on the test split.
6. **Seeds.** The verdict is for the seed-42 models; it is recomputed for seeds 43–46 with each seed's own models, and the report says for how many of the five seeds it also holds (§6).
7. **Validity checks before any scoring:** both mask digests reproduce (§3.7 as amended by item 7); the test split opens once, within a single opening, at all three levels; the model files carry their fingerprints.
8. **Unprimed and primed verdicts are separate** (§7).
- Under (c) the comparator set depends on the level. Under (d) a verdict has a third value, *untestable*, and the runner needs each comparator's miss count before it tests.

### D7. Records that mention E-2 and are not edited by the amendment

`PROGRESS.md` (§1 Next action and Blocked on, §3 the E-2 row, §5, §6, §8: the main session's); `docs/README.md` line 33 (the index entry for the errata: add this file); `docs/07-risk-register.md` R-19 ("H6 may well not be supported": under (b) add that H6 is judged at 25%); `docs/16-cloud-handoff.md`, `docs/17-cloud-session-tasks.md`, `docs/19-work-partition.md` (hand-over notes that say no fused ranking may be read until E-2 is answered: stale after the answer, historical otherwise); `docs/12-project-overview.md` and `docs/13-simplified-project-guide.md` (superseded and withdrawn: leave as records); `docs/presentation/midterm-prep.md` line 286 (O4 "waits for E-2").

---

## 6. Part E: ambiguities and inconsistencies (flagged, not resolved here)

1. **Item 10 against D-13.** The errata pins "the rule as committed with amendment 3a". EXP-006's text and `scripts/sweep_fusion.py` changed on 2026-10-13 (`c677d22`) in a way the dated note says is "before any sweep", without a §9 entry. Which text is "the rule" decides whether the planned sweep complies. Two wordings in item 10.
2. **The deep ranker at 50% is not in the errata's table.** H1-R and H2-R name B1-DL (A0's own component, so a fused-against-component comparison) as a comparator, and the errata's miss counts cover only the four `+aug` B1 models. EXP-021 says only that top-3 at 50% is saturated, "zero or one discordant patient at every seed" against XGBoost; no miss count for B1-DL or B1-DL′+aug at 50% is in `docs/08`. Under (b) this does not matter; under (c) and (d) it would.
3. **(b) and what is already known about H6.** (b) was recommended on 2026-10-02 without any deep result; EXP-021 was read on 2026-10-13, so the team answers with it in view. It restricts H6 to the level where the transformer's small MRR gain over the logistic regression is already known, and at that level it is level with XGBoost trained like it and worse at seed 46. `docs/05` §1 (amendment 3) warns against choosing "the metric where a gap is already known to exist". The Part B row says this; the team should decide whether that is enough.
4. **(b) has no fallback if 25% is also at ceiling on test.** Validation shows the comparators missing 57–111 top-3 and 38–71 must-not-miss cases at 25%, so a significant win is possible; the test split is unread. (b) and (d) are listed as alternatives and the errata does not say whether they can be combined.
5. **What "reported beside, descriptively" allows.** Whether an adverse 50% result can defeat a claim is not said. The draft says no.
6. **The comparator readings of item 2.** The errata says it is unclear whether an unprimed A0 must also beat the ′+aug models; the draft says it need not, because 3a names them "for primed systems" and §4 calls a comparison across encoders "mixing". It also takes 3a's wording that a primed A0′ is compared with the plain B1 (a different encoder, so "mixing" as well), and gives the reduced Target and Stretch no deep-ranker or KG-only comparator, because their wording is "each B1 variant" while H1-R and H2-R name B1-DL.
7. **Item 4 and item 3 pull apart for H3 and H4.** Item 4 says a hypothesis's ">" means significant; item 3 offers that H3 and H4 be judged on point estimates. The draft limits item 4 to the five hypotheses with tests. Which systems H3 compares (B3 against the grounded system, or A3 against A0) is not defined anywhere.
8. **"Also run" for H2-R and H6.** §6 says McNemar on all cases is run for each comparison and "also" on the restricted cases for H2-R and H6, and that a hypothesis needs every comparison to pass; read literally H2-R would need the all-patient top-3 test too, which is not H2's metric. The draft counts the restricted test only.
9. **EXP-006's objective under (b).** The sweep picks one α by the mean over the three levels of MRR. With the claims judged at 25%, the errata does not say whether the objective should follow. Left unchanged by the draft; changing it would need a §9 entry (item 10).
10. **McNemar's variant and sidedness are not in `docs/05`.** The §1 sentences that "the best two-sided p is 0.125 at four, 0.0625 at five" assume the exact binomial test. `src/eval/metrics.py` uses the exact test below 25 discordant pairs and a continuity-corrected chi-square above. Item 5's "two-sided" is the draft's wording.
11. **The errata's description of where the asked digest is computed** ("in `scripts/evaluate_reduced_evidence.py` and `scripts/sweep_fusion.py`") is loose: it is `digests()` in the first, called by the second. `docs/05` §3.7 also names one digest where 3a made two a validity condition (item 7 fixes both).
12. **A1's ML input.** See item 9's note: the question belongs to the runner, not the protocol.
13. **The ECE that counts (item 6).** A level's own calibrator cannot be the pipeline's setting for a case with no level (EXP-007); the clause "or a level's own" is in the draft because the errata lists it.
14. **Item 11 stops at 3a.** Amendment 3b's disclosure may also need to accompany reduced-evidence results.
15. **The Target's ">" on a tie.** At 100% B1 saturates, so A0 and B1-XGB may both score 1.000 on top-3. The protocol does not say a tie fails; ">" suggests it does. Neither the errata nor this draft says.
16. **Version and date.** The draft numbers the protocol 1.5 and leaves the date blank; nothing in the errata fixes either.
17. **Item 8's text describes a gap that is already closed.** EXP-020 trained seeds 43–46 on 2026-10-01 and scored them on 2026-10-12, so "closed by training B1-XGB's seeds 43–46" is true now. The draft words it as a rule, not as a date.

## 7. What the owner must decide that the errata did not make explicit

1. **Which of 10-i and 10-ii** (Part A, item 10; Part E, 1). Without a choice the sweep's compliance is unclear.
2. **Whether to disclose, and how, that the team answers E-2 with EXP-021's figures on record** (Part E, 3; the Part B row has a placeholder).
3. **Whether (b) stands alone or with (d) as a fallback** (Part E, 4).
4. **Whether an adverse 50% result can defeat a claim** (Part E, 5).
5. **Whether EXP-006's objective should follow the 25% level** (Part E, 9).
6. **The drafter's choices for items 2 and 3** (the comparator reading, H2's restricted test, H3's comparator B3, the cumulative Stretch): approving "items 2–11" does not settle them.
7. **The date, the version number (1.5) and who files the §9 row** (the blank "approved by" cell is the owner's).

## 8. Checks done, and checks to run when applying

**Done, on 2026-10-21, in this worktree** (nothing is gitignored-data dependent):
- Every **Old** and **Insert after** block of Part A and Part C was checked by a throwaway script (not committed) against `docs/05` v1.4 (as of `2613868`; `docs/05` last changed in `eb3b7d4`): each is a substring of the file once whitespace is collapsed.
- The digest facts of item 7 were read from `scripts/evaluate_reduced_evidence.py` and `scripts/check_mask_rules.py`.
- The "now" wordings of Part D were read from the files named (line numbers as of `2613868`).
- The comparator miss counts are the errata's, cross-checked against EXP-019's paired comparisons (for example B1-XGB+aug against B1-LR+aug at 50%: 3 / 0 on top-3, as four and one misses imply). They were not recomputed.

**To run when applying** (the main session):
1. `git log` since `2613868` shows no fused ranking of a DDXPlus patient scored (no `data/interim/exp006_fusion.json`), no test-split read, and `docs/05` unchanged since `eb3b7d4`; if any of that has changed, rewrite 1.2, 1.3 and the Part B justification.
2. Re-run the Old-text check against the then-current `docs/05`.
3. Fill the date (1.1, 1.2, the §9 row), the hash in 10-ii if chosen, and the options chosen.
4. Apply Part C instead of 1.4–1.9 if the answer is (a).
5. Update the files of Part D and `PROGRESS.md` (§3 E-2 row to §6, §1, §8) in the same commit; regenerate `docs/18-project-report.pdf`.
6. Leave the "approved by" cell as the owner words it.

*Nothing in this file is in force until the team approves it and the main session applies it to `docs/05` with a §9 entry.*
