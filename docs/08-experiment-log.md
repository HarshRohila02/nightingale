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
| EXP-003 | B0 prevalence baseline | 1 | Metric floor |
| EXP-004 | B1 ML-only (LogReg → XGBoost) | 1–2 | The competitor to beat |
| EXP-005 | B2 KG-only scoring | 2 | Is the graph useful alone? |
| EXP-006 | A0 fusion, weight sweep | 2 | Fusion weights (validation only) |
| EXP-007 | Calibration (Platt vs isotonic) | 2 | H5 |
| EXP-008 | Red-flag sensitivity/precision | 2 | Safety layer tuning |
| EXP-009 | B3 LLM-only | 3 | H3 |
| EXP-010 | B4 text-RAG + LLM | 3 | H3 |
| EXP-011 | Retrieval quality sweep | 3 | Chunking/embedding choice |
| EXP-012 | **Full ablation A0–A6** `[TEST]` | 4 | H1, H2, H4 |
| EXP-013 | Label audit of the chest-pain subset *(unplanned, run 2026-09-18)* | 1 | D-8: what D means for Precision@3 and Recall@5 |
| EXP-014 | Crosswalk check: concepts, red flags and golden cases on real data *(unplanned, run 2026-09-18)* | 1 | R-15: the red-flag rules over-fire; 2a: graph-only ranking |
| EXP-015 | The graph alone on validate patients *(unplanned, run 2026-09-19)* | 1 | R-12: how big the circularity is; 2a: unstable angina |
| EXP-016 | BODHI-S enrichment: coverage and effect *(unplanned, run 2026-09-19)* | 1 | 2a: a score that does not punish enriched conditions |
