# `docs/05` amendment 3a: follow-up questions and errata (proposal for the team)

**Drafted:** 2026-10-02, by Claude · **Status:** proposal only; `docs/05` v1.4 (amendment 3a) is in
force as approved · **Approval:** all four team members, as a §9 entry ("Amendment 3b") ·
**Decision:** E-2 in `PROGRESS.md` §3

The team approved the errata to amendment 3 on 2026-10-02 ("go with recommendations"), and they
were applied as amendment 3a. Independent verifiers then checked the result, in two rounds. They
found it faithful to what was approved, but found one consequence nobody had stated (item 1, which
needs an answer) and gaps that the amended text still leaves (items 2–11). None of these is fixed
in `docs/05`, because it changes only through its amendment log. **Nothing scored so far depends
on them.** **EXP-006's sweep waits for item 1**, so that it is decided with no fused result on
DDXPlus known; items 2–11 must be settled before any A0 or B1-DL verdict.

---

## 1. Needs an answer: the `+aug` comparators are at ceiling at 50%

Item 17 (b), which the team approved, makes H1-R, H2-R and the reduced-evidence Target and Stretch
also answer to the B1 variants trained like A0's ML component, and item 6 makes the same models
H6's comparators (as amendment 3's decision 4 intended). **On EXP-019's validation patients those
models are perfect, or within four misses, at 50%:**

| Model at 50% | Top-3 misses (of 33,963) | Must-not-miss misses (of 16,943) |
|---|---|---|
| B1-XGB+aug | 1 | 0 |
| B1-LR+aug | 4 | 3 |
| B1-LR′+aug | 0 | 0 |
| B1-XGB′+aug | 0 | 0 |

A point estimate cannot exceed 1.000, and McNemar's test cannot reach α = 0.05 until the comparator
misses at least six cases (the best two-sided p is 0.125 at four misses, 0.0625 at five). Each of
H1-R, H2-R, H6 and the reduced Target and Stretch must hold at **both** 50% and 25%, so **on these
figures none of them can be met as a whole**; their 25% comparisons can still be won (the
comparators miss 57–111 top-3 and 38–71 must-not-miss cases there), but each claim also needs 50%.
H1, H2 and the Target were already out of reach at 100% (R-16). **So the text as approved leaves no
comparative claim against B1 that can be supported at any level, H6 (RQ6's only hypothesis)
included.** These are validation figures; the verdicts are taken once on the test split, where
the comparators would have to miss at least six cases at 50% for a significant win, which these
figures make very unlikely.

Claude recommended item 17 (b) citing only the 25% figures; EXP-019 had published the 50% ones
before the approval. Every option below is therefore chosen with the comparators' figures known.
No fused result on DDXPlus and no deep model exists yet, so none is chosen knowing how A0 or B1-DL
does: this is the same kind of knowledge amendment 3 used when R-16 led it to add the reduced
levels.

| Option | What it means | H6 |
|---|---|---|
| (a) Keep the text as approved, and report these verdicts as unreachable | Nothing changes; RQ1, RQ2 and RQ6 can then only be answered descriptively | unreachable |
| **(b) Judge H1-R, H2-R, H6 and the reduced Target and Stretch at 25% only**; report the 50% comparisons beside them, descriptively | Keeps the `+aug` comparators (the fair, harder test) where they leave room; the 50% level stays in every table | judged at 25% |
| (c) Return to item 17's option (a) at 50% (plain B1 there) | Compares a `+aug` A0 with a plain B1 at 50%, which §4 calls mixing a change of model with a change of data; and it does not help H6, whose comparators are the `+aug` models by definition | unreachable |
| (d) Treat a comparison as untestable, not failed, when its comparator misses five or fewer cases on the test split | Decided now, applied to the test data; a claim is then judged on its testable levels | decided on test |

**Recommended: (b).** At 50% the comparators are at ceiling, so a comparison there tests nothing,
as at 100% (R-16); at 25% they leave room, and the `+aug` comparators stay. It must be decided
before EXP-006's sweep and before any deep model exists, and its §9 entry must say it was chosen
knowing EXP-019's comparator figures. (a) is the most conservative but answers none of the
comparative questions; (d) is principled but makes the verdict depend on how many cases the
comparators miss on test, which is harder to explain.

---

## 2. Gaps the amended text leaves

| # | Where | The gap | Proposed correction |
|---|---|---|---|
| 2 | §1, §4, §6, §7 | **The comparator set of each claim is still spread over four places.** Item 6 fixes B1-DL as B1-DL+aug only "in H1-R and H6", but H2-R also runs "against every ML-only system". "B1 variant" means B1-XGB and B1-LR in §4, includes the `+aug` models in §7's reduced Target, and is used in both senses within one sentence of §6 ("between A0 and each B1 variant, and between B1-DL and each B1 variant trained like it"). It is unclear whether an unprimed A0 must also beat the ′+aug models. | One table in §4 naming, for H1, H2, H1-R, H2-R, H6, the Target, the Stretch and the two reduced rows, the exact comparators (label, arm, seed 42), separately for unprimed and primed systems; §6's tests run against every comparator in it |
| 3 | §6, §1 | **Comparisons with no defined test**, now that a hypothesis needs every comparison to pass: H1-R and H2-R against the `+aug` B1 comparators of item 17 (b), the very comparisons item 1 is about (§6 lists only "each B1 variant", B1-DL and B2); H1 against KG-only at 100% (the A0–B2 tests are listed only for H1-R); H2 at 100% (the must-not-miss McNemar is listed only for H2-R and H6); H2-R against B1-DL; H3 and H4 have no test at all; the first-written Stretch names no metrics | List each claim's tests at each level, or state that H3 and H4 are judged on point estimates with intervals; say whether the 100% Stretch includes the Target's ECE and unsupported-claim criteria |
| 4 | §1 vs §6 (item 3) | Item 3 reads the Target's bare ">" as a point estimate and the Stretch's "with statistical significance" as a test. H1 and H2 were also written with a bare ">", yet the new rule requires significance for every hypothesis, while §1 says H1–H5 "keep their first-written meaning" | State in §1 that a hypothesis's ">" means significant at α = 0.05, and record that item 3 settles this for H1 and H2 |
| 5 | §6 (item 4) | **A pass has no direction.** The MRR test "passes when the 95% interval excludes 0", so a system significantly *worse* passes; McNemar's "passes at α = 0.05" is two-sided too. (`paired_bootstrap_difference` reports `excludes_zero` and `better` separately, so nothing is lost.) | "A comparison passes when p < 0.05 (or the interval excludes 0) **and** the claimed system is the better one" |
| 6 | §3.7 (item 10), §3.3 | **Which ECE counts against the target (< 0.10) at 50% and 25%**: the calibrator fitted at 100%, or a level's own when it has one | Name the one judged: the calibration setting the pipeline will use, declared before the test split opens, with the other beside it |
| 7 | §3.7 (item 8) | The digest rule is under-specified: "in evidence-code order" is numeric in the code (`code_sort_key`: E_2 before E_10), not string order; and the recorded digests are the first 16 hex digits of the SHA-256, not the whole hash. A reimplementation from the text could fail to reproduce `af3a357f6454838a` and declare a valid run invalid | "sorted by the number after `E_`", and "the first 16 hex digits of the SHA-256". Also: §3.7 says `scripts/check_mask_rules.py` computes the digest, but that script computes only the token digest (the asked digest is in `scripts/evaluate_reduced_evidence.py` and `scripts/sweep_fusion.py`), and "their digest … recorded in docs/08" names one digest where 3a makes two a validity condition: name the second script, and require both digests in docs/08 |
| 8 | §6, last bullet; §4 labels | §6 still says the seed gap "is closed when B1-XGB is next retrained", while §4 now says it is never retrained except for seeds 43–46. EXP-020 also refits seed 42 of B1-XGB as a reproducibility check, which neither covers. §4's exception sits in the "B1-XGB, B1-LR" row though plain B1-LR takes no seeds | "closed when its seeds 43–46 are trained (EXP-020)"; the seed-42 refit is a check, never reported as B1-XGB; scope the exception to B1-XGB |
| 9 | §5, A1 | A1b now says it keeps the crosswalk, which the red-flag rules need; A1 (red flags off) does not say whether its ML ranker's hand-written input still goes through the crosswalk | Say what "no medical KG" removes in A1 too: the graph's scoring and the fusion, not the crosswalk |
| 10 | §5 (item 18) | Item 18 binds "the rule EXP-006 records", but `docs/08` is not frozen, and EXP-006 has been revised three times (all before its sweep) | Pin it: the rule as committed with amendment 3a, any later change needing a §9 entry |
| 11 | §8, rule 8 | Every reduced-evidence result must carry the amendment-3 disclosure, but not 3a's, although item 17 was chosen after EXP-019 | "the amendment-3 and 3a disclosures (§9)" |

---

## 3. The §9 row, if approved

```markdown
| 2026-__-__ | **Amendment 3b (decision E-2): §§1, 3.3, 3.7, 4, 5, 6, 7 and 8, as listed in docs/proposals/05-amendment-3b-errata.md.** Item 1: option (_); items 2–11: the comparator table, the tests of each claim, the direction of a pass, the ECE judged at reduced levels, the digest's exact form, the seed bullets, A1's crosswalk, EXP-006's rule pinned, and §8's disclosure | Found by the independent verifiers of amendment 3a on 2026-10-02, the day it was approved and applied; item 1 from EXP-019's 50% figures, published before amendment 3a was approved but not cited in it. *(To fill at approval: whether any fused ranking of a DDXPlus patient had been scored, or any deep model existed.)* | The team, 2026-__-__ (relayed by the owner) |
```
