# 1 · Introduction

> **Draft 2, 2026-11-11.** Brought up to date from draft 1 (2026-10-13) to the repository after the
> single opening of the test split (2026-11-09) and the compliance pass (2026-11-10). Figures tagged
> `[TEST]` are test-split figures; the others are validation figures and say so. Every figure in
> this report names its evidence level and its model label (`docs/05` §8 rule 8).

**Nightingale — Clinical Intelligence & Reasoning System**

*Authors:* [Author 1, P1: cardiac knowledge graph] · [Author 2, P2: data and ML] ·
[Author 3, P3: retrieval, LLM and safety] · [Author 4, P4: platform and evaluation]
*Supervisor:* [Prof. Ashish Kumar, as named in `docs/14`; confirm title and affiliation]
*Institution and programme:* [to be added]

> ⚠️ **Clinical decision support — not a diagnosis.** Nightingale is a research prototype. Its
> output is generated from synthetic training data and must not be used for real patient care.
> Rankings are not certainties and confidence values are estimates. A qualified clinician is
> responsible for all diagnostic and treatment decisions.

*(The disclaimer above is the exact `DISCLAIMER` text of `src/contracts.py` and `docs/04` §2. It
appears on every result the system returns.)*

## 1.1 The problem

Acute chest pain is a frequent cause of emergency visits (Gulati et al. 2021), and it is a
high-stakes one, because the dangerous causes look like the benign ones at first. The clinician has
to separate the causes that kill if missed from the benign ones, often from the history and
examination alone, before confirmatory tests return. The dangerous causes include myocardial
infarction, pulmonary embolism, pneumothorax, aortic dissection and oesophageal rupture; the
benign ones include reflux and panic attack.

We call a condition **must-not-miss** when a missed or low ranking risks death (`docs/01` §1.3).
The cost of an error is not symmetric. A false alarm costs a test; a missed myocardial infarction
can cost a life (`docs/04` §3). A useful aid must therefore be judged first on whether it keeps
the dangerous conditions in view, and only then on how often its first answer is right.

Two kinds of computational aid fail clinicians in different ways (`docs/00` §1):

1. **Black-box classifiers** output a label or a probability with no reasoning a clinician can
   inspect, so a clinician cannot safely trust or override them.
2. **Large language models** write fluent clinical text, but can state false facts (Ji et al.
   2023), cannot reliably cite their sources, and are poorly calibrated [citation needed: LLM
   source attribution and calibration in clinical text].

Neither shows the two things a clinician reasons with: what evidence supports each candidate, and
what information is still missing to tell the candidates apart.

## 1.2 Aim and scope

We built a clinician-facing decision-support prototype that ranks the causes of acute chest pain
and explains the ranking. As shipped, it combines:

1. a machine-learning ranker, a small transformer (B1-DL′+aug) trained on the 255,900 synthetic
   DDXPlus chest-pain training patients and one masked copy of each, with logistic regression
   (B1-LR′+aug) as the backup (§4.9);
2. a cardiac knowledge graph of conditions and their findings, built from three sources, with the
   source of every edge recorded;
3. a fusion step that pools the two opinions into one ranking, with the graph's weight α = 0.05
   chosen on validation (EXP-006);
4. red-flag rules for the must-not-miss conditions, which run independently of the ranker;
5. a template explanation written from the graph and the model's own reasons, checked sentence by
   sentence against the result (guardrails v1), with the investigations to consider and the
   questions to ask next;
6. evidence excerpts from PubMed Central Open Access articles, each with its citation and its own
   licence, shown beside the first three candidates and never used to rank. Their relevance was
   never judged (EXP-011).

Three further components were built and do **not** ship in the ranking or the explanation. A
grounded explainer written by a local language model (`llama3.1:8b`) was run on Kaggle and failed
its pre-registered checkpoint, so the template explainer stays and `llm.enabled` is false
(EXP-009/010, §4.14). A calibrator for the fused score was fitted but is not deployed: the owner
decided on 2026-11-09 that the system shows ranks only, no probability (EXP-007, §4.12). A history
of synthetic Synthea patients is drawn beside a case as context and never feeds the ranking
(§4.15).

The scope is deliberately narrow (`docs/00` §4). The system ranks **14 conditions**: the 13
chest-pain conditions present in DDXPlus, and aortic dissection, which DDXPlus does not contain and
which the knowledge graph and a red-flag rule cover (Table 1.1). It is closed-world: a
presentation caused by anything else is still ranked as one of these 14 (risk R-13, Chapter 7).

**Table 1.1 — The conditions in scope** (`README.md`, `src/conditions.py`).

| Condition | Category | Must-not-miss | In DDXPlus |
|---|---|:---:|:---:|
| Possible NSTEMI / STEMI (acute MI) | Cardiac, ischaemic | yes | yes |
| Unstable angina | Cardiac, ischaemic | yes | yes |
| Stable angina | Cardiac, ischaemic | | yes |
| Pericarditis | Cardiac, inflammatory | | yes |
| Myocarditis | Cardiac, inflammatory | yes | yes |
| Acute pulmonary edema | Cardiac, failure | yes | yes |
| Atrial fibrillation | Cardiac, arrhythmia | | yes |
| PSVT | Cardiac, arrhythmia | | yes |
| Pulmonary embolism | Non-cardiac mimic | yes | yes |
| Spontaneous pneumothorax | Non-cardiac mimic | yes | yes |
| Boerhaave (oesophageal rupture) | Non-cardiac mimic | yes | yes |
| GERD | Non-cardiac mimic | | yes |
| Panic attack | Non-cardiac mimic | | yes |
| Aortic dissection | Vascular | yes | **no**: knowledge graph and red-flag rule only |

Out of scope (`docs/00` §4): autonomous diagnosis; any treatment or drug recommendation; real or
identifiable patient data; clinical deployment; presentations other than chest pain; training a
medical language model; any claim of clinical efficacy. No clinician user study was done.

## 1.3 Research questions and hypotheses

The questions, hypotheses and success criteria were fixed in the evaluation protocol before any
model was trained (`docs/05`, frozen 2026-09-17). They change only through its dated amendment log,
and every amendment is disclosed in this report (Chapter 7, §7.10). The table below reproduces
`docs/05` §1 as it stands at version 1.5 (amendment 3b, 2026-11-07), word for word, with the
protocol's own strike-throughs; the last column, which is ours, gives each hypothesis's amendment
status.

**Table 1.2 — Research questions and hypotheses** (`docs/05` §1, v1.5).

| # | Research question | Hypothesis | Status |
|---|---|---|---|
| **RQ1** | Does fusing KG reasoning with an ML ranker improve differential accuracy over either alone? | **H1:** Fusion > ML-only and > KG-only on Top-3 accuracy and MRR. **H1-R** *(amendment 3)*: the same, at ~~**both** 50% and 25% evidence~~ 25% evidence, the 50% comparison reported beside it and not counted *(amendment 3b)* (§3.7), against every ML-only system (B1-XGB, B1-LR and B1-DL, §4, and the B1 variants trained like A0's ML component *(amendment 3a)*) | H1 as first written (2026-09-17). H1-R added by amendment 3 (2026-09-25); its comparators widened by amendment 3a (2026-10-02); its level narrowed to 25% by amendment 3b (2026-11-07) |
| **RQ2** | Does the system improve **safety** over an ML-only baseline? | **H2:** Fusion + red flags > ML-only on must-not-miss recall@3. **H2-R** *(amendment 3)*: the same, at ~~**both** 50% and 25% evidence~~ 25% evidence, the 50% comparison reported beside it and not counted *(amendment 3b)*, against every ML-only system, the B1 variants trained like A0's ML component included *(amendment 3a)* | H2 as first written. H2-R added by amendment 3; widened by amendment 3a; its level narrowed to 25% by amendment 3b |
| **RQ3** | Does evidence grounding reduce unsupported claims versus an unconstrained LLM? | **H3:** RAG-grounded explanation has a lower unsupported-claim rate than LLM-only | As first written; judged as a point estimate with its interval (amendment 3b) |
| **RQ4** | Which components actually contribute? | **H4:** Removing the medical KG degrades must-not-miss recall more than it degrades Top-1 | As first written; ablation A1b reported beside A1 (amendment 3a); judged as a point estimate with its interval (amendment 3b) |
| **RQ5** | Are the confidence scores calibrated? | **H5:** Post-calibration ECE < 0.10 | As first written; reduced-level reporting fixed by amendment 3a |
| **RQ6** *(amendment 3)* | Does a deep ranker hold up better than classical ML when the history is incomplete? | **H6:** B1-DL > **each** B1 variant trained like it (§4) on Top-3 accuracy, MRR **and** must-not-miss recall@3, at ~~**both** 50% and 25% evidence~~ 25% evidence, the 50% comparison reported beside it and not counted *(amendment 3b)*. At 100% the comparison is reported, and a tie is the expected result (R-16) | Added by amendment 3 (2026-09-25); its level narrowed to 25% by amendment 3b |

Three facts about these hypotheses must be stated at the start, because they shape every result
that follows.

- **H1 and H2 cannot be supported on full-evidence DDXPlus.** The plain B1-XGB, B1-LR and both
  transformers miss 0 of the 34,657 test patients in the top 3 and 0 of the 16,972 who have a
  must-not-miss condition at 100% evidence `[TEST]` (EXP-012; validation gave the same verdict, EXP-004
  and EXP-012's rehearsal, risk R-16). Nothing is left to improve on those two metrics. The protocol's response, amendment 3, adds scoring at 50%
  and 25% of each patient's evidence.
- **As first approved, H1-R, H2-R, H6 and the reduced-evidence Target and Stretch could not be
  met.** Each needed both 50% and 25%, and at 50% the comparators trained on masked copies are
  perfect or within a few patients (`docs/05` §1, amendment 3a disclosure). The team was asked how
  these claims should be judged (decision E-2, §5.9). **The owner answered on 2026-11-07 ("go with the
  recommendation")**, and amendment 3b (v1.5) now judges them at 25% evidence, with the 50%
  comparisons run and reported beside them as descriptive. The other team members' agreement is
  not recorded, and the report says so. Even at 25% the claims were not supported (Table 1.3).
- **Negative results are reportable results** (`docs/05` §1). If a hypothesis fails, we say so
  in the abstract and the conclusion (`docs/05` §8 rule 5).

The test split was opened once, on 2026-11-09, after the code and configuration were frozen
(tag `v0.9-freeze`) and both mask digests were recorded (§3.2.1, EXP-012). **Table 1.3** gives each
verdict in a sentence or two, with the figures copied from `docs/08` EXP-012. Intervals are 95%
bootstrap intervals; seed 42 unless stated; "A0" is the system with B1-DL+aug and "A0′" the one with
B1-DL′+aug (the configured ranker).

**Table 1.3 — Where each question stands** (`[TEST]` unless marked; `docs/08` EXP-012, EXP-009/010).

| Question | Verdict | What stands behind it |
|---|---|---|
| **RQ1**: H1 (100%), H1-R (25%) | **Not supported**, A0 and A0′, at 0 of 5 seeds | At 100% the comparators are perfect (R-16), and A0's MRR is lower than B1-XGB's by −0.0837 [−0.0859, −0.0819], because the red-flag layer ranks flagged conditions first even when the flag names the wrong one. At 25% both arms beat the original B1-XGB, B1-LR and B2 but not the models trained on masked copies: A0 against B1-XGB+aug is −0.0001 on top-3 (p = 0.883) and −0.0052 on MRR [−0.0062, −0.0044]. |
| **RQ2**: H2 (100%), H2-R (25%) | **Not supported**, A0 and A0′ | At 25%, must-not-miss recall@3 is 0.9986 [0.9980, 0.9991] for A0′ and the same, 0.9986 [0.9980, 0.9991], for the transformer alone (B1-DL′+aug). The original B1 misses the safety target of 0.95 at 25% (B1-XGB 0.8710 [0.8659, 0.8764]; B1-LR 0.9198 [0.9156, 0.9238]); the systems trained on masked copies meet it. |
| **RQ3**: H3 | **Not supported** (validation only) | On 50 validation patients the grounded language-model explainer's unsupported-claim rate was 0.741 [0.663, 0.803] against a bar of under 0.20; two of H3's four comparisons pass. It failed D-5's checkpoint, so the template explainer ships (EXP-009/010). Not carried to the test split. |
| **RQ4**: H4 | Supported by the pre-registered rule, **but hollow** | At 100%, removing the graph changes must-not-miss recall@3 by exactly 0.0000. The point estimate passes only because removing the flags as well raises top-1 by 0.1642 [0.1606, 0.1685] (A0 0.8346, A1 0.9987). It is not evidence that the graph helps. |
| **RQ5**: H5 | Supported | Raw top-label ECE at 100%: A0 0.0099 [0.0094, 0.0103], A0′ 0.0398 [0.0394, 0.0402], under 0.10. No calibrator ships and no probability is shown. |
| **RQ6**: H6 (25%) | **Not supported** | The transformer beats the logistic regression trained like it on all three metrics and is level with the XGBoost trained like it: B1-DL+aug against B1-XGB+aug is 0.0000 on top-3 (p = 0.883), +0.0002 on MRR [−0.0005, +0.0009] and +0.0003 on must-not-miss recall@3 (p = 0.458). |

The Minimum is met: top-3 at 100% is 1.0000 [1.0000, 1.0000] for A0 against 0.3061 [0.3013, 0.3109]
for B0.
The reduced-evidence Target is met by A0′ and not by A0, **on a handful of patients**: at 25%
A0′ misses 40 of 34,657 patients in the top 3 and B1-XGB′+aug 56, while A0 misses 73 and
B1-XGB+aug 71. The Stretch is not met at either level. The unsupported-claim rate of the shipped
template explainer and a clinician's review of the explanations were not assessed. No verdict
changed between validation and test.

What the results do support is narrower than the hypotheses. Trained on masked copies and shown
which questions were asked, the transformer-based systems meet the safety targets at 25% of the
history: must-not-miss recall@3 0.9981 [0.9974, 0.9987] for A0 and 0.9986 [0.9980, 0.9991] for A0′,
against 0.8710 [0.8659, 0.8764] and 0.9198 [0.9156, 0.9238] for the original B1-XGB and B1-LR. The
value is in the training on masked
copies, not in the fusion, which adds nothing measurable on DDXPlus (R-12, R-16). All of this is
on synthetic, closed-world data, and none of it is a claim about real patients.

## 1.4 Decision support only

Nightingale never says "the patient has a myocardial infarction". It ranks candidates, says why,
shows what is missing, and raises a red flag when a must-not-miss pattern is present; the
clinician decides (`README.md`, `docs/04` §1). This framing is built into the system, not only
written in the report:

- the disclaimer is attached to every result and cannot be removed or hidden (`docs/04` §2;
  enforced by the safety layer, §4.11);
- no output recommends a treatment, names a drug or gives a dose (FR-6.5; §4.11);
- no probability is shown at all: the system gives ranks only, and `calibration.file` is null
  (`docs/05` §3.3; §4.12);
- a failed or missing component is always reported (`degraded_components`), because a silently
  degraded medical tool is a safety problem (`docs/02` §7).

It is an academic research prototype, not a medical device. It has had no clinical validation and
no regulatory review (`docs/04` §1), and it was built and evaluated on synthetic and open data
only.

## 1.5 Contributions

We do not claim that combining a knowledge graph with machine learning or a language model is new;
that combination exists in the literature (`docs/00` §5) [citation needed: knowledge graphs
combined with ML or LLMs for clinical decision support]. What this project contributes is:

1. **A red-flag-aware differential ranker for one high-stakes presentation, evaluated on
   must-not-miss recall** rather than top-1 accuracy alone (`docs/00` §5). The ranker pools an ML
   model and a knowledge graph in a logarithmic opinion pool, and independent rules after
   published clinical patterns (ADD-RS, Wells, Braunwald) raise the must-not-miss conditions.
2. **A knowledge graph that covers a diagnosis absent from the training data.** Aortic dissection
   is not in DDXPlus; 20 hand-authored edges from the ADD-RS and the IRAD registry let the graph
   rank it, and a rule flags it. On the hand-written dissection case (GC-003) the fused ranking
   puts it first even with the red flags switched off, with the configured transformer as well as
   with the logistic regressions (EXP-006's golden gate; §6.11).
3. **A pre-registered evaluation with three evidence levels, read once.** The protocol was frozen
   before any model existed. Amendment 3 added scoring at 100%, 50% and 25% of each history, with
   the masks drawn once and their digests recorded before any system was scored on them. The test
   split was opened once, by a run that refused to start unless the code was frozen and both
   digests reproduced (EXP-012).
4. **A measured finding about the data and the models.** On full DDXPlus histories every ML model
   is near-perfect, because of how the data were generated (R-16). With part of the history
   removed, two models that agreed to 0.0002 in top-1 diverge sharply. We traced this to the
   feature encoding, which could not tell a denied question from one never asked (R-18), and
   repaired it with masked training copies and an explicit "asked" channel (EXP-017 to EXP-019).
5. **A small deep ranker adopted only through a safety floor fixed before training.** A small
   FT-Transformer-style model (B1-DL) replaced logistic regression as the configured ranker only
   after it met the safety targets and was shown non-inferior to logistic regression at 25% of the
   history on validation, at all five seeds (decision D-13, EXP-021). It runs on a laptop CPU as a
   NumPy forward pass, and its size and latency are measured (EXP-022). On the test split it beats
   a logistic regression trained the same way and ties an XGBoost trained the same way (H6 is not
   supported).
6. **The ablation study**, run once on the test split (EXP-012), which isolates what each component
   adds and reports the negative results. The graph adds nothing measurable on DDXPlus (R-12,
   R-16); the red-flag layer lowers top-1 and MRR at full evidence; and the system's safety at
   reduced evidence comes from training on masked copies, not from the fusion.
7. **A pre-registered checkpoint that was allowed to stop a component.** The grounded
   language-model explainer was built, run and stopped by D-5's checkpoint (an unsupported-claim
   rate of 0.741 [0.663, 0.803] against a bar of under 0.20), so the template explainer ships
   (EXP-009/010).

## 1.6 Structure of the report

Chapter 2 gives the clinical background and related work. Chapter 3 describes the data and their
licences. Chapter 4 describes the system as built. Chapter 5 sets out the pre-registered
evaluation protocol. Chapter 6 reports the results, validation and the single opening of the test
split. Chapter 7 states the limitations, the safety design and every disclosure. Chapter 8
concludes.
