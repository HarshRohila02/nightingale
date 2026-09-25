# 05 — Evaluation Protocol (Pre-Registered)

**Version:** 1.3 · 2026-09-25 (amendments 1–3, §9) · **Owner:** P4
**Status:** 🔒 **FROZEN** as of 2026-09-17, end of Phase 0 — before any model was trained.
Amended on 2026-09-19 through §9, still before any model was trained or evaluated.
Amended again on 2026-09-25 (amendment 3), **after** B0 and B1 had been trained and scored on the
validation split; its §9 entry lists what was known when it was written

> Frozen means: no metric, baseline, ablation, or success threshold in this document may change
> without a dated entry in the amendment log (§9) **and** disclosure in the final report. No model
> had been trained at the time of freezing, so nothing in the frozen text or in amendments 1–2 was
> chosen with knowledge of results. Amendment 3 was written with some results known, and its §9
> entry says which.

> **Why this document is written first.** Choosing metrics after seeing results is how a project
> talks itself into a conclusion. Everything measurable is fixed here, in advance. After the freeze
> date, changes are permitted only via the amendment log in §9, with a dated justification.
>
> **The test set is opened once, in Phase 4.** All development uses train/validation only.

---

## 1. Research questions and hypotheses

| # | Research question | Hypothesis |
|---|---|---|
| **RQ1** | Does fusing KG reasoning with an ML ranker improve differential accuracy over either alone? | **H1:** Fusion > ML-only and > KG-only on Top-3 accuracy and MRR. **H1-R** *(amendment 3)*: the same, at **both** 50% and 25% evidence (§3.7), against every ML-only system (B1-XGB, B1-LR and B1-DL, §4) |
| **RQ2** | Does the system improve **safety** over an ML-only baseline? | **H2:** Fusion + red flags > ML-only on must-not-miss recall@3. **H2-R** *(amendment 3)*: the same, at **both** 50% and 25% evidence, against every ML-only system |
| **RQ3** | Does evidence grounding reduce unsupported claims versus an unconstrained LLM? | **H3:** RAG-grounded explanation has a lower unsupported-claim rate than LLM-only |
| **RQ4** | Which components actually contribute? | **H4:** Removing the medical KG degrades must-not-miss recall more than it degrades Top-1 |
| **RQ5** | Are the confidence scores calibrated? | **H5:** Post-calibration ECE < 0.10 |
| **RQ6** *(amendment 3)* | Does a deep ranker hold up better than classical ML when the history is incomplete? | **H6:** B1-DL > **each** B1 variant trained like it (§4) on Top-3 accuracy, MRR **and** must-not-miss recall@3, at **both** 50% and 25% evidence. At 100% the comparison is reported, and a tie is the expected result (R-16) |

*Amendment 3.* H1–H5 keep their first-written meaning. They are judged at 100% evidence, H1 and H2
against B1-XGB, and they are reported first. The same comparisons against B1-LR and B1-DL are
reported beside them. H1-R, H2-R and H6 are separate hypotheses with separate verdicts. A claim
names the hypothesis it rests on, and "H1 is supported" always means H1 as first written. H6 uses
H1's and H2's metrics, not top-1, although EXP-017 found the widest gap between the B1 variants in
top-1: choosing the metric where a gap is already known to exist would be choosing with knowledge
of results. Top-1 is reported at every level.

**Negative results are reportable results.** If H1 fails, that is a finding about hybrid
architectures, not a project failure. It must be reported honestly.

---

## 2. Data and splits

| Aspect | Decision |
|---|---|
| Dataset | DDXPlus, filtered to the 13 in-scope chest-pain conditions |
| Splits | DDXPlus-provided train / validate / test |
| Split level | Case level (each synthetic patient appears in exactly one split) |
| Tuning | **Validation only** — including fusion weights and calibration |
| Test access | **Once**, Phase 4, after all development is frozen |
| Inputs | `AGE`, `SEX`, `EVIDENCES` (decoded) |
| Evidence levels *(amendment 3)* | Every system is scored at 100%, 50% and 25% of each patient's evidence (§3.7). `INITIAL_EVIDENCE` decides what is kept at 50% and 25%, and is **never an input** |
| **Forbidden input** | `DIFFERENTIAL_DIAGNOSIS` — this is the label. Using it is leakage. |
| Ground truth (ranking) | `DIFFERENTIAL_DIAGNOSIS` (ranked list) |
| Ground truth (top-1) | `PATHOLOGY` |

Report per-split case counts and per-condition class balance before any modelling.

---

## 3. Metrics

### 3.1 Ranking (primary)

Let the system output ranked candidates `ĉ₁…ĉₙ`; `y` = true pathology; `D` = ground-truth
differential set.

*Amendment 1 (2026-09-19, §9):* `D_in` is the part of `D` inside the 13 in-scope conditions.
The headline Precision@3 and Recall@5 use `D_in`. Recall@5 with the full `D`, as first written,
is reported alongside it, for comparison with published DDXPlus results.

| Metric | Definition | Reported as |
|---|---|---|
| **Top-1 accuracy** | fraction where `ĉ₁ = y` | % |
| **Top-3 accuracy** | fraction where `y ∈ {ĉ₁,ĉ₂,ĉ₃}` | % ← *primary ranking metric* |
| **Top-5 accuracy** | fraction where `y ∈ top 5` | % |
| **MRR** | mean of `1 / rank(y)` | 0–1 |
| **Precision@3** | \|top-3 ∩ D_in\| / 3, identical to \|top-3 ∩ D\| / 3 because the system outputs only in-scope conditions *(amendment 1)* | 0–1 |
| **Recall@5** | ~~\|top-5 ∩ D\| / \|D\|~~ → \|top-5 ∩ D_in\| / \|D_in\| *(amendment 1)*; the full-`D` figure is reported alongside | 0–1 |
| **Per-condition F1** | macro and micro across 13 conditions | 0–1 |

### 3.2 Safety (headline)

| Metric | Definition | Target |
|---|---|---|
| **Must-not-miss recall@3** | Of cases whose true condition is must-not-miss, fraction where it appears in the top 3 | **≥ 0.95** |
| **Dangerous false-negative rate** | Fraction of must-not-miss cases ranked **outside** the top 5 | ≤ 0.02 |
| **Red-flag sensitivity** | ~~Of cases matching a red-flag pattern, fraction where the flag fired~~ → Of cases whose true condition is must-not-miss, the fraction where a flag fired **for that condition**, overall and per condition. A must-not-miss condition without a rule counts as missed *(amendment 2)* | ≥ 0.95 |
| **Red-flag precision** | Of fired flags, fraction clinically appropriate | reported, not targeted |

> Red-flag precision is deliberately **not** targeted. Over-flagging is an accepted cost
> ([04-safety-ethics.md](04-safety-ethics.md) §3).

### 3.3 Calibration

ECE (10 bins), Brier score, reliability diagrams — per condition and overall. **Target ECE < 0.10.**
No score is reported as a probability until it passes calibration.

### 3.4 Evidence retrieval

Recall@k, Precision@k, nDCG@10, plus **supported-claim rate** (fraction of generated claims
traceable to a retrieved passage), **unsupported-claim rate**, and **citation correctness** (does
the cited passage actually support the claim?).

Relevance judgements: 100 sampled candidate–evidence pairs, labelled independently by two team
members; report Cohen's κ. Disagreements resolved by discussion.

### 3.5 Explainability (qualitative)

50 sampled cases scored 1–5 on reasoning validity, evidence relevance, and clarity by two team
members. If a clinically-trained reviewer is available, their judgement supersedes; state clearly in
the report which applies. **Do not describe team members as clinical experts.**

### 3.6 System

Latency (p50/p95) with and without the LLM; graceful-degradation coverage (each failure in
[02-architecture.md](02-architecture.md) §7 exercised).

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

---

## 4. Baselines

| # | Baseline | Purpose |
|---|---|---|
| **B0** | Majority class / prevalence prior | Floor — anything below this is broken |
| **B1** | ML-only on decoded evidences, in two variants *(amendment 3)*: **B1-XGB** (XGBoost, as first written) and **B1-LR** (multinomial logistic regression, EXP-004's linear reference) | The serious competitor |
| **B1-DL** *(amendment 3, decision D-11)* | ML-only deep ranker on the same features | A0's ML component (§5), measured against B1 at every evidence level; it never replaces B1 as the comparator |
| **B2** | KG-only (symptom-overlap / PageRank) | Is the graph useful alone? |
| **B3** | LLM-only (local model, case in prompt, no retrieval, no KG) | Tests the "just use an LLM" position |
| **B4** | Text-RAG + LLM (retrieval, no KG, no ML) | Tests whether retrieval alone suffices |

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

---

## 5. Ablation matrix

| Config | Patient KG | Medical KG | ML | Fusion | Red flags | RAG | LLM |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **A0 — Full Nightingale** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| A1 — no medical KG | ✅ | ❌ | ✅ | — | ❌ | ✅ | ✅ |
| A2 — no ML | ✅ | ✅ | ❌ | — | ✅ | ✅ | ✅ |
| A3 — no RAG | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| A4 — no red flags | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| A5 — no calibration | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
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

---

## 6. Statistical testing

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

---

## 7. Success criteria

| Level | Criteria |
|---|---|
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

Failing "Target" while meeting "Minimum" is an acceptable, publishable outcome provided the analysis
explains *why*.

---

## 8. Reporting rules

1. Every metric reported with a 95% CI.
2. Every ablation row reported, including ones that contradict our hypotheses.
3. Per-condition results reported — aggregate numbers hide rare-condition failure.
4. The **synthetic-data limitation** stated alongside every accuracy claim.
5. Failed hypotheses stated plainly in the abstract/conclusion, not buried.
6. Fusion weights, hyperparameters, seeds, and model versions disclosed.
7. **The circularity caveat must accompany every RQ1/RQ4 result** (risk R-12). The cardiac KG is
   derived largely from DDXPlus `release_conditions.json`, the same source the ML ranker trains on,
   so the KG is not a fully independent knowledge source. State this wherever the value of the KG
   is claimed. The independent contributions that remain — hand-authored red-flag rules, aortic
   dissection (absent from the training data entirely), and BODHI-S enrichment for four conditions
   — should be reported separately from the shared-source component. See
   [10-spike-r01-crosswalk.md](10-spike-r01-crosswalk.md).
8. *(Amendment 3)* **Every figure names its evidence level and its model label (§4).** A
   reduced-evidence result is never reported without the same system's full-evidence result beside
   it, and never without the amendment-3 disclosure (§9). Reduced evidence removes answers at
   random, and a clinician does not ask at random, so these results measure robustness to missing
   answers. They must not be presented as a simulation of history-taking. The mask cannot move a
   patient's kept answers outside their own condition's DDXPlus evidence list, so R-16's cause is
   weakened at reduced evidence, not removed.

---

## 9. Amendment log

| Date | Change | Justification | Approved by |
|---|---|---|---|
| 2026-09-17 | Initial version | — | — |
| 2026-09-17 | Added reporting rule §8.7 (circularity caveat) | R-01 spike resolved to a fallback that derives the KG from DDXPlus, the ranker's own training source. Added **before** freezing and before any model was trained. | — |
| 2026-09-17 | **Document FROZEN** | End of Phase 0. No model trained to date. | — |
| 2026-09-19 | **Amendment 1 (decision D-8), §3.1.** Precision@3 and Recall@5 use `D_in`, the part of the ground-truth differential inside the 13 in-scope conditions. Recall@5 with the full `D`, as first written, is reported alongside it | EXP-013 audited the validate labels before any model existed. 91.8% of in-scope patients' differentials include conditions the system cannot output, a third of their probability mass. As first written, Recall@5 could not exceed 0.434 even for a perfect system, so it measured the closed-world gap (R-13) more than the ranking. With `D_in` a perfect system reaches 0.753 (`D_in` holds 6.7 conditions on average, often more than 5). Precision@3 does not change (a perfect system reaches 0.929 under either `D`). Decided before any model was trained or evaluated. The full-`D` figure stays in every report, so the change cannot flatter a result | The team, 2026-09-19 (relayed by the owner) |
| 2026-09-19 | **Amendment 2, §3.2.** Red-flag sensitivity is measured against the true condition: of cases whose true condition is must-not-miss, the fraction where a flag fired for that condition, overall and per condition. A must-not-miss condition without a rule counts as missed. The target is unchanged (≥ 0.95) | As first written (of cases matching a red-flag pattern, the fraction where the flag fired), the metric is 1.0 by construction for fixed rules on structured input, so it could never fail. Counting any flag instead of the condition's own would reward the over-firing EXP-014 found (R-15). Disclosure: EXP-014 had already measured each rule on its own patients on validate (MI 91%, PE 79%, Boerhaave 75%, pneumothorax 32%; unstable angina, myocarditis and acute pulmonary edema have no rule yet). The amended metric is therefore harder to meet than the original, and the target was not changed | The team, 2026-09-19 (relayed by the owner) |
| 2026-09-25 | **Amendment 3 (decisions D-10 and D-11): §§1, 2, 3.7 (new), 4, 5, 6, 7, 8.** Every system in §4 and every configuration in §5 is also scored at 50% and 25% of each patient's evidence. The mask is defined in §3.7: whole questions, the initial evidence always kept, a kept follow-up's parent kept, one permutation per patient keyed by `case_id` and seed 42, drawn once, digest recorded. `initial_evidence` defines the mask and is never an input. H1-R, H2-R, H6 (new, RQ6) and a reduced-evidence Target and Stretch are added **beside** the first-written H1, H2, Target and Stretch. Those are still judged at 100% against B1-XGB and are reported first. B1 is not redefined: it is reported as two variants (B1-XGB, as first written; B1-LR, EXP-004's linear reference) at every level, and a claim of beating B1 must hold against each. B1-DL, a deep ranker, is added as an ML-only baseline and as A0's ML component, and A7 swaps B1 back in. §6's five seeds are 42–46 for every stochastic model. The model used is the seed-42 model, fixed before any such model exists. A prime (′) marks models retrained with the encoder's "asked" channel (R-18); the originals stay reported | **Unlike amendments 1 and 2, this amendment was written after models had been trained and evaluated.** When it was drafted (2026-09-23), B0 and both B1 models had been scored on the validation split (EXP-003, EXP-004). B1's saturation of full-evidence DDXPlus was known (R-16), so H1, H2 and the Target cannot be met at 100%. EXP-017 had scored both B1 models at these exact levels, under a token-level version of this mask: at 25%, XGBoost's top-1 was 0.27 and its must-not-miss recall@3 0.935, while logistic regression held 0.86 and 0.974. The KG-only graph had been scored at 100% (EXP-015, EXP-016, and with the 2a score EXP-005). The 50% and 25% levels themselves had been proposed in decision D-10 on 2026-09-20 (commit `98c746b`), before any model was scored at them; EXP-017 used them three days later. The causes are R-16 and the project supervisor's email of 2026-09-20, which suggested deep learning for a more robust model. The mask works by question rather than by token because, at 25%, EXP-017's token mask left 53% of validation patients with a positive follow-up answer whose parent "yes" had been dropped, which never happens at full evidence. This was measured on the masks alone; no model was scored under the new rule before it was fixed. **Between the proposal and its approval** (2026-09-24 to 2026-09-25): the red-flag rules were revised and scored on validation at 100% (EXP-008); and B1-LR+aug, B1-XGB+aug, B1-LR′+aug and B1-XGB′+aug were trained, with seed 42 only (EXP-018; B1-LR and B1-XGB themselves are unchanged). Those four were scored on validation at 100% only, and on two short inputs that use no mask: a patient with no findings, and the four golden cases. The short inputs showed that the `+aug` models no longer answer atrial fibrillation to short input, and that they are overconfident on it; this bears on H6 and on the `+aug` arm chosen for A0. The XGBoost `+aug` models chose their stopping round on held-out training patients, half of them masked copies (§3.7 allows masks on the train split). Before approval, models had been scored on masked **validation** patients only in EXP-017, under the token rule; EXP-017 also probed a patient with no findings. What keeps the amendment honest: the levels, the mask and the new hypotheses are fixed before any fusion result on DDXPlus data or any deep-model result exists; the test split has never been read; the first-written hypotheses and Target are kept and reported first; and every claim against B1 must hold against both variants, so knowing that XGBoost degrades at reduced evidence cannot flatter any system. The reduced-evidence validation figures of B0, B1 and B2 are descriptive, not pre-registered | The team, 2026-09-25 (relayed by the owner) |

*Amendments after the freeze date require an entry here and must be disclosed in the final report.*
