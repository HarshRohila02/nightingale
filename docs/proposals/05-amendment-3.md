# `docs/05` amendment 3: reduced evidence and the deep ranker (proposal for the team)

**Drafted:** 2026-09-23, by Claude for the owner to send · **Revised:** 2026-09-24, after the
numbers were verified (§8) · **Status:** proposal only. `docs/05` has not been changed. **Approval:** all four team members (`docs/05` §9) · **Settles:** D-10, option (b).
**Records:** D-11 · **Sources:** EXP-003, EXP-004 and EXP-017 (`docs/08`), R-16 and R-18
(`docs/07`), `PROGRESS.md` §3, and Gate 1 of the deep-ranker plan

---

## 1. Summary

1. **A reduced-evidence condition (D-10 (b)).** Every system and every ablation is also scored at
   **50%** and **25%** of each patient's evidence, alongside 100%. Each patient keeps their initial
   evidence. The mask is a fixed function of `case_id` and seed 42, drawn once, with its digest
   recorded. `initial_evidence` decides what is kept but is never a model input.
2. **The mask works by question, not by token as it did in EXP-017** (recommended, §3). At 25%,
   EXP-017's token rule left 53% of patients with a pain descriptor but no "yes, I have pain". That
   never happens at full evidence. The proposed rule keeps or drops whole questions, keeps a kept
   follow-up's parent, and makes the levels nested.
3. **B1 is not redefined.** It stays "the baseline that matters" and is now reported as two
   variants at every level, **B1-XGB** and **B1-LR**. A claim of beating B1 must hold against
   **both**, so nobody can pick the flattering variant later. EXP-017 found them 0.0002 apart in
   top-1 at 100% and 0.59 apart at 25%.
4. **D-11 in protocol terms.** A new ML-only baseline, **B1-DL** (the deep ranker), becomes A0's ML
   component. A new ablation, **A7**, swaps B1 back in, so the deep model's contribution is measured
   rather than assumed. A new hypothesis, **H6**, fixes before training what "the deep model is more
   robust" has to mean.
5. **Seeds.** §6's five seeds are **42–46**. The model that is used is **seed 42**, fixed now and
   never the best-scoring seed. A prime (′) marks a model retrained with the encoder's planned
   "asked" channel (R-18), and the original models stay reported.
6. **The first-written tests stay.** H1, H2 and the §7 *Target* are judged at 100% exactly as first
   written and reported first. The reduced-evidence versions (H1-R, H2-R, a reduced-evidence Target)
   are reported beside them, never in their place, and they require **both** levels.
7. **This amendment was written after models were trained and evaluated.** §6 below says so, in
   text for the final report.

---

## 2. The new §9 row (exact text)

```markdown
| 2026-09-__ | **Amendment 3 (decisions D-10 and D-11): §§1, 2, 3.7 (new), 4, 5, 6, 7, 8.** Every system in §4 and every configuration in §5 is also scored at 50% and 25% of each patient's evidence. The mask is defined in §3.7: whole questions, the initial evidence always kept, a kept follow-up's parent kept, one permutation per patient keyed by `case_id` and seed 42, drawn once, digest recorded. `initial_evidence` defines the mask and is never an input. H1-R, H2-R, H6 (new, RQ6) and a reduced-evidence Target and Stretch are added **beside** the first-written H1, H2, Target and Stretch. Those are still judged at 100% against B1-XGB and are reported first. B1 is not redefined: it is reported as two variants (B1-XGB, as first written; B1-LR, EXP-004's linear reference) at every level, and a claim of beating B1 must hold against each. B1-DL, a deep ranker, is added as an ML-only baseline and as A0's ML component, and A7 swaps B1 back in. §6's five seeds are 42–46 for every stochastic model. The model used is the seed-42 model, fixed before any such model exists. A prime (′) marks models retrained with the encoder's "asked" channel (R-18); the originals stay reported | **Unlike amendments 1 and 2, this amendment was written after models had been trained and evaluated.** When it was drafted (2026-09-23), B0 and both B1 models had been scored on the validation split (EXP-003, EXP-004). B1's saturation of full-evidence DDXPlus was known (R-16), so H1, H2 and the Target cannot be met at 100%. EXP-017 had scored both B1 models at these exact levels, under a token-level version of this mask: at 25%, XGBoost's top-1 was 0.27 and its must-not-miss recall@3 0.935, while logistic regression held 0.86 and 0.974. The KG-only graph had been scored at 100% (EXP-015, EXP-016, and with the 2a score EXP-005). The 50% and 25% levels themselves had been proposed in decision D-10 on 2026-09-20 (commit `98c746b`), before any model was scored at them; EXP-017 used them three days later. The causes are R-16 and the project supervisor's email of 2026-09-20, which suggested deep learning for a more robust model. The mask works by question rather than by token because, at 25%, EXP-017's token mask left 53% of validation patients with a positive follow-up answer whose parent "yes" had been dropped, which never happens at full evidence. This was measured on the masks alone; no model was scored under the new rule before it was fixed. What keeps the amendment honest: the levels, the mask and the new hypotheses are fixed before any fusion result on DDXPlus data or any deep-model result exists; the test split has never been read; the first-written hypotheses and Target are kept and reported first; and every claim against B1 must hold against both variants, so knowing that XGBoost degrades at reduced evidence cannot flatter any system. The reduced-evidence validation figures of B0, B1 and B2 are descriptive, not pre-registered | The team, 2026-09-__ (relayed by the owner) |
```

---

## 3. The mask: by token (EXP-017) or by question? Recommendation: by question

EXP-017's rule masks **tokens**. It keeps the tokens equal to the initial evidence or starting with
`<initial>_@_`, calls the rest `R`, and keeps `n = round(len(R) × share)` of them, chosen by
`default_rng(int.from_bytes(sha256(f"42:{case_id}").digest()[:8], "big")).choice(len(R), n,
replace=False)`, in their original order. The proposal keeps the key, the seed and the RNG, and
changes three things. The unit is the **question**. A kept follow-up brings its **parent** with it.
One **permutation** per patient replaces an independent draw per level, so the levels nest.

`scripts/check_mask_rules.py` reproduces every figure in this table from the validate parquet and
the release file: mask properties only, nothing trained and no model scored. It implements both
rules exactly as stated here and in §3.7, and prints each rule's digest.

| Validate, 33,963 patients (full evidence: 21.9 tokens, 15.4 questions) | Token rule (EXP-017) | Question rule (proposed) |
|---|---|---|
| Mean tokens kept, 50% / 25% | 11.45 / 6.21 (matches EXP-017's 11.5 / 6.2) | 11.87 / 6.71 (8.6 / 5.1 questions) |
| Patients keeping a **positive** follow-up answer (e.g. `E_55`, where the pain is) whose parent "yes" (`E_53`, pain) was dropped: at full evidence | 0 | 0 |
| The same, at 50% / 25% | **12,830 (37.8%) / 18,041 (53.1%)** | 0 / 0, by construction |
| Patients keeping only part of a multi-choice answer (e.g. one of several pain locations), 50% / 25% | 29,040 (85.5%) / 27,516 (81.0%) | 0 / 0, by construction |
| Patients whose 25% mask lies inside their 50% mask | 1,398 (4.1%) | 33,963 (100%), by construction |

**Why the question rule:**

1. **The token rule builds patients that no interview produces.** At full evidence, no patient has
   a positive follow-up answer without its parent. At 25%, half of them do. A model scored on those
   patients sees an input it never saw in training and could never see in use, so any drop in
   accuracy mixes "less evidence" with "impossible evidence".
2. **It matches the planned "asked" channel.** The R-18 fix adds one "asked" column per question.
   Under the question rule, a question is either asked, with all its answers, or not asked. Under
   the token rule, a half-kept multi-choice answer is neither.
3. **Nested levels make the curve paired.** With one permutation per patient, the 25% patient is the
   50% patient with less history, so a difference between levels measures the amount of evidence
   alone. EXP-017's draws were independent per level, and only 4.1% were nested.
4. **The parent is added, not the follow-up dropped.** Pain that radiates is pain, the same rule as
   the crosswalk's `PARENT_QUESTION`. Dropping orphaned follow-ups instead would strip most pain
   descriptors from chest-pain patients at 25%.

**What it costs:**

- Choosing a rule after EXP-017 is itself a post-hoc choice. The reasons above are properties of the
  masks, measured without scoring any model. **Recommendation: nobody scores any model under either
  rule until the team has approved one**, so the choice cannot follow the results.
- EXP-017's figures cannot serve as B1's reduced-evidence baseline. B1 must be re-scored under the
  pinned rule, which takes about 30 s of scoring and no training. EXP-017 stays in the log as a
  diagnostic, with its numbers.
- The question rule keeps slightly more evidence than the token rule (6.71 against 6.21 tokens at
  25%), so its figures will not match EXP-017's. The rule applies to every system alike, so it
  favours no system.

If the team prefers the token rule, §7 decision 3 gives the replacement paragraph.

---

## 4. Wording changes, section by section

### 4.1 Header and the frozen note

**Current:**

```markdown
**Version:** 1.2 · 2026-09-19 (amendments 1–2, §9) · **Owner:** P4
**Status:** 🔒 **FROZEN** as of 2026-09-17, end of Phase 0 — before any model was trained.
Amended on 2026-09-19 through §9, still before any model was trained or evaluated

> Frozen means: no metric, baseline, ablation, or success threshold in this document may change
> without a dated entry in the amendment log (§9) **and** disclosure in the final report. No model
> had been trained at the time of freezing, so nothing here was chosen with knowledge of results.
```

**Proposed:**

```markdown
**Version:** 1.3 · 2026-09-__ (amendments 1–3, §9) · **Owner:** P4
**Status:** 🔒 **FROZEN** as of 2026-09-17, end of Phase 0 — before any model was trained.
Amended on 2026-09-19 through §9, still before any model was trained or evaluated.
Amended again on 2026-09-__ (amendment 3), **after** B0 and B1 had been trained and scored on the
validation split; its §9 entry lists what was known when it was written

> Frozen means: no metric, baseline, ablation, or success threshold in this document may change
> without a dated entry in the amendment log (§9) **and** disclosure in the final report. No model
> had been trained at the time of freezing, so nothing in the frozen text or in amendments 1–2 was
> chosen with knowledge of results. Amendment 3 was written with some results known, and its §9
> entry says which.
```

The last sentence of the note has to change. As it stands ("nothing here was chosen with knowledge
of results"), it would become false once amendment 3 is in the document.

### 4.2 §1 Research questions and hypotheses

**Current** (the RQ1 and RQ2 rows; there is no RQ6):

```markdown
| **RQ1** | Does fusing KG reasoning with an ML ranker improve differential accuracy over either alone? | **H1:** Fusion > ML-only and > KG-only on Top-3 accuracy and MRR |
| **RQ2** | Does the system improve **safety** over an ML-only baseline? | **H2:** Fusion + red flags > ML-only on must-not-miss recall@3 |
```

**Proposed** (RQ3–RQ5 unchanged, and RQ6 added after RQ5):

```markdown
| **RQ1** | Does fusing KG reasoning with an ML ranker improve differential accuracy over either alone? | **H1:** Fusion > ML-only and > KG-only on Top-3 accuracy and MRR. **H1-R** *(amendment 3)*: the same, at **both** 50% and 25% evidence (§3.7), against every ML-only system (B1-XGB, B1-LR and B1-DL, §4) |
| **RQ2** | Does the system improve **safety** over an ML-only baseline? | **H2:** Fusion + red flags > ML-only on must-not-miss recall@3. **H2-R** *(amendment 3)*: the same, at **both** 50% and 25% evidence, against every ML-only system |
| **RQ6** *(amendment 3)* | Does a deep ranker hold up better than classical ML when the history is incomplete? | **H6:** B1-DL > **each** B1 variant trained like it (§4) on Top-3 accuracy, MRR **and** must-not-miss recall@3, at **both** 50% and 25% evidence. At 100% the comparison is reported, and a tie is the expected result (R-16) |
```

**Added below the table:**

```markdown
*Amendment 3.* H1–H5 keep their first-written meaning. They are judged at 100% evidence, H1 and H2
against B1-XGB, and they are reported first. The same comparisons against B1-LR and B1-DL are
reported beside them. H1-R, H2-R and H6 are separate hypotheses with separate verdicts. A claim
names the hypothesis it rests on, and "H1 is supported" always means H1 as first written. H6 uses
H1's and H2's metrics, not top-1, although EXP-017 found the widest gap between the B1 variants in
top-1: choosing the metric where a gap is already known to exist would be choosing with knowledge
of results. Top-1 is reported at every level.
```

### 4.3 §2 Data and splits (one row added, nothing changed)

**Current** (the row the new one follows):

```markdown
| Inputs | `AGE`, `SEX`, `EVIDENCES` (decoded) |
```

**Proposed** (the Inputs row unchanged, one row inserted after it):

```markdown
| Inputs | `AGE`, `SEX`, `EVIDENCES` (decoded) |
| Evidence levels *(amendment 3)* | Every system is scored at 100%, 50% and 25% of each patient's evidence (§3.7). `INITIAL_EVIDENCE` decides what is kept at 50% and 25%, and is **never an input** |
```

The *Test access* row is not touched (§5 below).

### 4.4 New §3.7 Evidence levels (inserted after §3.6)

**Current:** there is no §3.7.

**Proposed:**

````markdown
### 3.7 Evidence levels *(amendment 3)*

Every system in §4 and every configuration in §5 is scored at three evidence levels, on the same
patients: **100%**, the patient as DDXPlus lists them (every result as first defined), **50%** and
**25%**. Every §3.1–§3.3 metric, with its interval, is reported at every level, and the §3.2–§3.3
targets apply at every level unchanged; a level where a target is missed is reported as missed.
§3.4 and §3.5 are measured at 100% only, and §3.6 does not depend on the level.

**The mask.** Age and sex are always kept. The patient's initial evidence (the `initial_evidence`
column, docs/03 §2.1) is always kept, with all its answers. The distinct question codes of the
patient's other tokens (the part of a token before `_@_`), in the order they first appear, form the
list `Q`. Each patient gets one permutation of `Q`:

```python
rng   = numpy.random.default_rng(
            int.from_bytes(hashlib.sha256(f"42:{case_id}".encode("utf-8")).digest()[:8], "big"))
order = rng.permutation(len(Q))
kept  = [Q[i] for i in order[: round(len(Q) * share)]]   # Python's round: halves go to even
```

A kept question keeps every one of its tokens, and a dropped question loses every one. If a kept
question follows up another question that the patient has (`code_question` in
release_evidences.json: `E_54`–`E_59` follow `E_53`, and `E_152` follows `E_151`), the parent is
kept too, even when it was not drawn. The kept tokens stay in their original order. Every level is a
prefix of the same permutation, so a patient's 25% history is part of their 50% history.

**The whole system sees only the masked patient.** The ML ranker, the knowledge graph and the
red-flag rules all read the kept tokens. No component reads a dropped one. **`initial_evidence`
defines the mask and is never a model input** (§2). The mask's output includes the set of questions
kept, so the encoder's "asked" channel (R-18) can be set from it. How that channel treats questions
DDXPlus does not list is decided with the encoder, not here. *(2026-09-24: the encoder's working
answer is open decision A-9 in `docs/02` §9, for the team with this amendment. It does not change
the kept tokens or the digest.)*

**Drawn once.** `src/ml/evidence_masks.py` draws the masks for the validation split once. Their
digest, the NumPy version and the commit are recorded in docs/08 before any fusion or deep-model
result is scored on them. The digest is the SHA-256 of one UTF-8 line per patient and reduced
level, `case_id<TAB>level<TAB>kept tokens joined by single spaces`, with the level written `0.5` or
`0.25`; the lines are sorted as strings (so by `case_id`, then level) and joined by `\n`, with no
trailing newline. `scripts/check_mask_rules.py` computes it this way.
NumPy does not promise the same random stream across versions, so the recorded masks are the
reference, not the code: a run whose masks do not reproduce the digest is invalid, and a test pins
worked examples of the rule. The same code, at the same commit, draws the test split's masks when
that split is opened in Phase 4, and the test split is scored at all three levels within that single
opening (§2 is unchanged).

**Tuning.** §2's "validation only" covers the reduced levels. Whatever is tuned, such as fusion
weights and calibration, is frozen before the test split opens. The report says whether one setting
serves every level or each level has its own. H5 is judged at 100%. At 50% and 25%, ECE is
reported for the same calibrated scores, and no calibrator is refitted for the purpose.

**Training on masked patients** (mask augmentation) is allowed on the train split only, with masks
drawn from the training seed, never from these evaluation masks. Such a model carries the suffix
`+aug` (§4).

**Realised size.** The share counts questions, and parents are added, so the share of tokens kept
differs from the nominal share. The mean tokens and questions kept at each level are reported with
every result.

**LLM baselines.** B3 and B4 are scored at 50% and 25% on the same case sample as at 100% when
compute allows. If they are not, the report says so.
````

### 4.5 §4 Baselines

**Current:**

```markdown
| **B1** | ML-only (XGBoost on decoded evidences) | The serious competitor |

**B1 is the baseline that matters.** If the full system cannot beat a well-tuned XGBoost, the
architecture is not justified and the report must say so.
```

**Proposed** (B0, B2, B3 and B4 unchanged; B1 extended; one row added after it):

```markdown
| **B1** | ML-only on decoded evidences, in two variants *(amendment 3)*: **B1-XGB** (XGBoost, as first written) and **B1-LR** (multinomial logistic regression, EXP-004's linear reference) | The serious competitor |
| **B1-DL** *(amendment 3, decision D-11)* | ML-only deep ranker on the same features | A0's ML component (§5), measured against B1 at every evidence level; it never replaces B1 as the comparator |

**B1 is the baseline that matters.** If the full system cannot beat a well-tuned XGBoost, the
architecture is not justified and the report must say so.

*Amendment 3.* Both B1 variants are reported at every evidence level, in every table where B1
appears, and neither may be dropped. A claim that a system beats B1 must hold against **each**
variant at the level claimed. At 100% the first-written verdict, against B1-XGB, is reported, with
the B1-LR comparison beside it. The reason is EXP-017: the two variants are 0.0002 apart in top-1 at
full evidence and 0.59 apart at 25%, so choosing one after the results would decide the outcome.

**Model labels.** A label names the model, its encoder and its training rows:

| Label | Meaning |
|---|---|
| B1-XGB, B1-LR | As trained in EXP-004: commit `68c14bd`, feature fingerprint `3a0d5a5e01d7f427`, full-evidence training rows. Always reported, and never retrained under the same label |
| B1-DL | The deep ranker, on the same encoder and the same training rows |
| `+aug` suffix | Trained on training rows with mask augmentation (§3.7), e.g. B1-XGB+aug |
| ′ (prime) | Retrained on the encoder with the "asked" channel (R-18), which has a new fingerprint: B1-XGB′, B1-LR′, B1-DL′, and A0′ when the pipeline uses them. The unprimed models stay reported, and a primed model never replaces its original in any table |

Two models are compared as models only when they share an encoder and training rows (the same prime
and the same `+aug`). Any other pairing is reported as mixing a change of model with a change of
encoding or data.
```

### 4.6 §5 Ablation matrix

**Current** (the last row, and the sentence after the table):

```markdown
| A6 — no patient KG (flat features) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Each configuration is evaluated on **all** §3 metrics. The expected headline finding is that
**A1 and A4 degrade safety metrics disproportionately** — i.e. the knowledge graph earns its place
on safety rather than on raw accuracy.
```

**Proposed** (A0–A6 unchanged; A7 added; the paragraph kept, with one added after it):

```markdown
| A6 — no patient KG (flat features) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| A7 — classical ML in place of the deep ranker *(amendment 3)* | ✅ | ✅ | B1 | ✅ | ✅ | ✅ | ✅ |

Each configuration is evaluated on **all** §3 metrics. The expected headline finding is that
**A1 and A4 degrade safety metrics disproportionately** — i.e. the knowledge graph earns its place
on safety rather than on raw accuracy.

*Amendment 3.* Every configuration is evaluated at all three evidence levels (§3.7). A0's ML
component is **B1-DL, the `+aug` arm, seed 42** (decision D-11; §6). A7 is A0 with that component
replaced by a B1 variant trained like it, and it is reported twice (A7-LR, A7-XGB), so the deep
ranker's contribution inside the full system is measured, not assumed. A0's ML component may be
changed only before the test split opens, on validation results, by a dated entry in §9. A7 is
reported either way.
```

### 4.7 §6 Statistical testing

**Current:**

```markdown
- **Bootstrap** 1,000 resamples of the test set → 95% confidence intervals on every headline metric.
- **McNemar's test** for paired Top-3 correctness between A0 and B1.
- Significance level α = 0.05; **report effect sizes and CIs, not just p-values.**
- Fixed seed (42); report mean ± std over 5 seeds for any stochastic component.
- **No metric is reported without a confidence interval.**
```

**Proposed** (the five bullets unchanged; one bullet added):

```markdown
- **Bootstrap** 1,000 resamples of the test set → 95% confidence intervals on every headline metric.
- **McNemar's test** for paired Top-3 correctness between A0 and B1.
- Significance level α = 0.05; **report effect sizes and CIs, not just p-values.**
- Fixed seed (42); report mean ± std over 5 seeds for any stochastic component.
- **No metric is reported without a confidence interval.**
- *Amendment 3.* Each item above applies separately at each evidence level (§3.7).
  - McNemar's test is run between A0 and **each** B1 variant, and between B1-DL and each B1
    variant trained like it (§4). For H2-R and H6 it is also run on top-3 correctness restricted to
    must-not-miss cases. A claim needs every one of its comparisons to pass at α = 0.05. Because a
    single failure defeats the claim (an intersection–union test), no multiplicity correction is
    needed. Every comparison is reported, whether it passes or not.
  - The five seeds are **42, 43, 44, 45 and 46**. They bind B1-DL in every arm, and every retrained
    XGBoost, whose row and column subsampling make it stochastic. Logistic regression is
    deterministic and is fitted once. Mask augmentation draws its masks from the model's seed.
  - **The model used, in A0, in A7, in every paired test and in the pipeline, is the seed-42
    model.** It is fixed here, before any of these models exists, and never replaced by the
    best-scoring seed. Every seed's figures are kept. Each hypothesis verdict is for the seed-42
    model, and the report says for how many of the five seeds the verdict would also hold.
  - This rule already applied to XGBoost, but EXP-004 trained it with seed 42 only. That gap is
    disclosed, and it is closed when B1-XGB is next retrained.
```

### 4.8 §7 Success criteria

**Current:**

```markdown
| **Minimum** | Pipeline runs end-to-end; A0 beats B0; all metrics computed with CIs; ablations complete |
| **Target** | A0 > B1 on Top-3 **and** must-not-miss recall; ECE < 0.10; unsupported-claim rate < 20% |
| **Stretch** | A0 > B1 with statistical significance; must-not-miss recall ≥ 0.95; clinician-reviewed explainability |
```

**Proposed** (the three rows unchanged; two rows and a paragraph added):

```markdown
| **Minimum** | Pipeline runs end-to-end; A0 beats B0; all metrics computed with CIs; ablations complete |
| **Target** | A0 > B1 on Top-3 **and** must-not-miss recall; ECE < 0.10; unsupported-claim rate < 20% |
| **Stretch** | A0 > B1 with statistical significance; must-not-miss recall ≥ 0.95; clinician-reviewed explainability |
| **Target, reduced evidence** *(amendment 3)* | At **both** 50% and 25% evidence: A0 > **each** B1 variant on Top-3 **and** must-not-miss recall@3 |
| **Stretch, reduced evidence** *(amendment 3)* | The reduced-evidence Target with every comparison statistically significant (§6); must-not-miss recall@3 ≥ 0.95 at both levels |

*Amendment 3.* Minimum, Target and Stretch are judged at 100% evidence exactly as first written,
with B1 = B1-XGB, and the same comparisons against B1-LR are reported beside them. The two
reduced-evidence rows are reported **beside** them and never in their place: meeting the
reduced-evidence Target does not meet the Target. R-16 makes the first-written Target unreachable on
full-evidence DDXPlus, and the report says so rather than presenting the reduced-evidence row as the
Target. Each criterion is judged separately for the unprimed and the primed systems (§4), and both
verdicts are reported.
```

### 4.9 §8 Reporting rules (one rule added)

**Current:** rules 1–7 (rule 7 is the circularity caveat). There is no rule 8.

**Proposed** (rules 1–7 unchanged):

```markdown
8. *(Amendment 3)* **Every figure names its evidence level and its model label (§4).** A
   reduced-evidence result is never reported without the same system's full-evidence result beside
   it, and never without the amendment-3 disclosure (§9). Reduced evidence removes answers at
   random, and a clinician does not ask at random, so these results measure robustness to missing
   answers. They must not be presented as a simulation of history-taking. The mask cannot move a
   patient's kept answers outside their own condition's DDXPlus evidence list, so R-16's cause is
   weakened at reduced evidence, not removed.
```

---

## 5. What is explicitly not changed

- **§2, every existing row.** The DDXPlus splits, case-level splitting, tuning on validation only,
  inputs (`AGE`, `SEX`, `EVIDENCES`), the forbidden `DIFFERENTIAL_DIAGNOSIS` input, and ground truth.
  **The test split is still opened once, in Phase 4.** It is scored at all three levels within that
  one opening.
- **§3.1–§3.6.** Every metric definition, amendments 1 (D-8, `D_in`) and 2 (red-flag sensitivity
  against the true condition), and every target in §3.2–§3.3.
- **H1–H5 as first written.** They are judged at 100%, H1 and H2 against B1-XGB, and reported first.
  H3, H4 and H5 get no reduced-evidence version. The ablation rows at 50% and 25% are reported, but
  they are not a hypothesis test.
- **B0, B2, B3 and B4.** B1 is still XGBoost as first written, still "the baseline that matters",
  and the bold sentence about beating a well-tuned XGBoost stays word for word.
- **A0–A6**, their columns, and the expected headline finding.
- **§6's five existing bullets.** 1,000 bootstrap resamples, α = 0.05, seed 42, and no metric
  without an interval.
- **§7's Minimum, Target and Stretch**, word for word, judged at 100%.
- **§8 rules 1–7**, including the synthetic-data statement and the R-12 circularity caveat.
- **The freeze date and the recorded history.** The header still says the document was frozen on
  2026-09-17 before any model was trained, and that amendments 1–2 came before any model was trained
  or evaluated. Both statements are true. The new line does not reuse that phrase for amendment 3.
- **No logged result changes.** EXP-003, EXP-004 and EXP-017 keep their numbers and their labels.
  EXP-017 stays a diagnostic under its token rule.
- **`src/contracts.py`, the golden cases and the disclaimer** are untouched.

---

## 6. Disclosure paragraph (for the final report)

> **Disclosure: amendment 3 to the evaluation protocol was made after models had been evaluated.**
> The protocol was frozen on 2026-09-17, before any model was trained, and its first two amendments
> were also made before any model was trained or evaluated. The third was not. When it was written,
> in late September 2026, the prevalence baseline (B0) and both ML-only baselines (B1: XGBoost and
> logistic regression) had been trained and scored on the validation split. We already knew that B1
> saturates full-evidence DDXPlus (top-3 accuracy and must-not-miss recall@3 both 1.000), so
> hypotheses H1 and H2 and the Target could not be met there. The two reduced levels had been proposed on 2026-09-20,
> before any model was scored at them, but a diagnostic had then scored both B1 models at the
> evidence levels the amendment adopts (100%, 50% and 25%), under a token-level version of its
> mask: at 25%, XGBoost's top-1 accuracy fell to 0.27 and its must-not-miss recall@3
> to 0.935, while logistic regression held 0.86 and 0.974. The amendment was prompted by that
> ceiling and by the project supervisor's suggestion that deep learning might give a more robust
> model. The reduced-evidence condition was therefore chosen knowing that differences between
> systems appear there. It is a stress test in which answers are removed at random, not a
> simulation of how a clinician takes a history. Four things limit the damage. The levels, the
> masking rule and the new hypotheses were fixed before any fusion result on DDXPlus data or any
> deep-model result existed. The hypotheses and Target as first written are reported first and
> unchanged. Every claim against B1 had to hold against both of its variants, so knowing that
> XGBoost degrades at reduced evidence could not favour any system. And the test split was not read
> until Phase 4. The knowledge-graph-only baseline had been scored at full evidence, after its
> score was replaced (EXP-005). The reduced-evidence validation figures for B0, B1 and the KG-only
> baseline are descriptive. The confirmatory results are those on the test split.

---

## 7. Decisions the team still has to make

| # | Decision | Recommended | If the team chooses otherwise |
|---|---|---|---|
| 1 | **Approve amendment 3** (all four members; `docs/05` is frozen) | Approve | Approving it settles D-10 (b). The reduced-evidence parts stand on their own |
| 2 | **D-11**: ratify the deep ranker as A0's ML component | (a), ratify | If (b), B1-DL stays a baseline only: delete "A0's ML component" in §4, and in §5 A0's ML component becomes a B1 variant, fixed now, with A7 swapping B1-DL in instead. H6 is unaffected |
| 3 | **The mask rule** (§3 above) | By question, parent kept, nested | EXP-017's token rule: in §3.7 replace the paragraph from "The distinct question codes" to "part of their 50% history" with: *"Of the patient's other tokens, in their original order, call the list `R`; keep `n = round(len(R) × share)` of them, chosen by `rng.choice(len(R), n, replace=False)` with the same `rng`, in their original order."* The levels are then not nested, and B1's validation figures already exist (EXP-017) |
| 4 | **Which B1-DL arm is used** in A0 and in H6 (fixed before training) | `+aug`, since the deployed input is short: golden cases carry 5–13 tokens | The plain arm: change "the `+aug` arm" in §5. Both arms are reported either way |
| 5 | **H6's metrics** | Top-3, MRR and must-not-miss recall@3, the metrics of H1 and H2 | Any other set is fine if it is fixed now. Choosing top-1 would be choosing the metric where EXP-017 already showed the gap |
| 6 | **Tuning parity** between B1-DL and B1 | No tuning for any ML model beyond early stopping on a 10% training holdout, which is what EXP-004 did and what the plan's MLP does | An equal budget of validation tuning, declared before training. An untuned B1 against a tuned B1-DL would not be a fair H6 |
| 7 | **XGBoost's missing seeds** (§6 already required five; EXP-004 used one) | Train seeds 43–46 in the same session as the next B1 retrain. That is a training run, so the owner chooses where (D-7) | Disclose the gap and leave it |
| 8 | **D-10 (c): independent vignettes written by the team** | Optional; worth it if time allows, since it is the only evaluation that escapes both R-12 and R-16 | If adopted, add to §3.7: *"Clinical vignettes written by the team, separate from the golden cases, are frozen (hash recorded) before any system is scored on them, scored as written by every system, reported with their (wide) intervals, never used for tuning, and carry no Target."* Also decide who writes them and by when |
| 9 | **B3 and B4 at reduced evidence** | Best effort, as drafted | Required: delete "when compute allows" in §3.7 |

---

## 8. Notes for the owner before sending

- **A figure that contradicted earlier reporting — corrected.** EXP-017's interpretation 1 said
  the two B1 variants were "0.73 top-1 apart" at 25%, and `PROGRESS.md`, the `ml:` config comment
  and the ranker's docstring said 0.7. EXP-017's own table gives 0.8558 − 0.2671 = **0.589** (0.378
  at 50%), as does its McNemar count ((20,560 − 564) / 33,963). The repo was corrected on
  2026-09-24 (commit `868f837`), with the wrong figures struck through.
- **The levels did come before EXP-017 — verified.** `git log -S "say 25% and 50%" -- PROGRESS.md`
  shows the wording in `98c746b` (2026-09-20); EXP-017 first appears in `cb8ddc6` (2026-09-23). The
  §9 row and §6 now say so.
- **The mask figures in §3 are reproducible** with `scripts/check_mask_rules.py` (seconds of CPU,
  no model, no training). They belong in `docs/08` when `src/ml/evidence_masks.py` lands, and that
  module's tests will pin them.
- **The adversarial critique of this draft did not run** (the session hit a usage limit and then
  ended). Instead Claude checked every number against `docs/08` and `docs/07`, re-derived the
  mask table with a script written to the rule as stated, and closed one ambiguity: how a level and
  the line breaks are written in the digest (§3.7). The team's own review is the remaining check.
- **Labels clash with the plan.** The deep-ranker plan used B1′ for "XGBoost retrained with mask
  augmentation" and D1′ for the deep equivalent. This amendment keeps ′ for the "asked" channel and
  uses `+aug` for augmentation, so the plan's labels should follow once the amendment is approved.
- **Until the team decides, nobody scores any model under either mask rule**, so that the rule
  cannot be chosen by its results. Nothing deep is trained before Gate 1 either way.
