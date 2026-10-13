# 1 · Introduction

> **Draft 1, 2026-10-13.** Validation results only; the test split is opened once, in Phase 4
> (`docs/05` §2). Every figure in this report names its evidence level and its model label
> (`docs/05` §8 rule 8).

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

Acute chest pain is one of the most common presentations in emergency and primary care, and one
of the highest-stakes [citation needed: incidence of chest pain presentations]. The clinician has
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
2. **Large language models** write fluent clinical text, but can state false facts, cannot
   reliably cite their sources, and are poorly calibrated [citation needed: LLM hallucination and
   calibration in clinical text].

Neither shows the two things a clinician reasons with: what evidence supports each candidate, and
what information is still missing to tell the candidates apart.

## 1.2 Aim and scope

We built a clinician-facing decision-support prototype that ranks the causes of acute chest pain
and explains the ranking. It combines four parts:

1. a machine-learning ranker trained on 255,900 synthetic DDXPlus patients;
2. a cardiac knowledge graph of conditions and their findings, built from three sources, with the
   source of every edge recorded;
3. a fusion step that pools the two opinions into one ranking;
4. red-flag rules for the must-not-miss conditions, which run independently of the ranker.

Evidence retrieval from the literature and an explanation written by a local language model are
planned as Phase 3 [pending: retrieval EXP-011] [pending: LLM baselines B3/B4, EXP-009/010].

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
medical language model; any claim of clinical efficacy.

## 1.3 Research questions and hypotheses

The questions, hypotheses and success criteria were fixed in the evaluation protocol before any
model was trained (`docs/05`, frozen 2026-09-17). They change only through its dated amendment log,
and every amendment is disclosed in this report (Chapter 7, §7.7). The table below reproduces
`docs/05` §1 as it stands at version 1.4, word for word; the last column, which is ours, gives
each hypothesis's amendment status.

**Table 1.2 — Research questions and hypotheses** (`docs/05` §1, v1.4).

| # | Research question | Hypothesis | Status |
|---|---|---|---|
| **RQ1** | Does fusing KG reasoning with an ML ranker improve differential accuracy over either alone? | **H1:** Fusion > ML-only and > KG-only on Top-3 accuracy and MRR. **H1-R** *(amendment 3)*: the same, at **both** 50% and 25% evidence (§3.7), against every ML-only system (B1-XGB, B1-LR and B1-DL, §4, and the B1 variants trained like A0's ML component *(amendment 3a)*) | H1 as first written (2026-09-17). H1-R added by amendment 3 (2026-09-25); its comparators widened by amendment 3a (2026-10-02) |
| **RQ2** | Does the system improve **safety** over an ML-only baseline? | **H2:** Fusion + red flags > ML-only on must-not-miss recall@3. **H2-R** *(amendment 3)*: the same, at **both** 50% and 25% evidence, against every ML-only system, the B1 variants trained like A0's ML component included *(amendment 3a)* | H2 as first written. H2-R added by amendment 3; widened by amendment 3a |
| **RQ3** | Does evidence grounding reduce unsupported claims versus an unconstrained LLM? | **H3:** RAG-grounded explanation has a lower unsupported-claim rate than LLM-only | As first written |
| **RQ4** | Which components actually contribute? | **H4:** Removing the medical KG degrades must-not-miss recall more than it degrades Top-1 | As first written; ablation A1b reported beside A1 (amendment 3a) |
| **RQ5** | Are the confidence scores calibrated? | **H5:** Post-calibration ECE < 0.10 | As first written; reduced-level reporting fixed by amendment 3a |
| **RQ6** *(amendment 3)* | Does a deep ranker hold up better than classical ML when the history is incomplete? | **H6:** B1-DL > **each** B1 variant trained like it (§4) on Top-3 accuracy, MRR **and** must-not-miss recall@3, at **both** 50% and 25% evidence. At 100% the comparison is reported, and a tie is the expected result (R-16) | Added by amendment 3 (2026-09-25) |

Three facts about these hypotheses must be stated at the start, because they shape every result
that follows.

- **H1 and H2 cannot be supported on full-evidence DDXPlus.** The ML-only baseline already ranks
  the true condition in the top 3 for every validation patient, and keeps every must-not-miss
  condition in the top 3 (EXP-004, risk R-16). Nothing is left to improve on those two metrics.
  The protocol's response, amendment 3, adds scoring at 50% and 25% of each patient's evidence.
- **As approved, H1-R, H2-R, H6 and the reduced-evidence Target and Stretch cannot be met on the
  validation figures.** Each needs both 50% and 25%, and at 50% the comparators trained on masked
  copies are perfect or within four misses (`docs/05` §1, amendment 3a disclosure). The team has
  been asked again how these claims are judged (decision E-2, §5.9), and had not answered on
  2026-10-13.
- **Negative results are reportable results** (`docs/05` §1). If a hypothesis fails, we say so
  in the abstract and the conclusion (`docs/05` §8 rule 5).

## 1.4 Decision support only

Nightingale never says "the patient has a myocardial infarction". It ranks candidates, says why,
shows what is missing, and raises a red flag when a must-not-miss pattern is present; the
clinician decides (`README.md`, `docs/04` §1). This framing is built into the system, not only
written in the report:

- the disclaimer is attached to every result and cannot be removed or hidden (`docs/04` §2;
  enforced by the safety layer, §4.11);
- no output recommends a treatment, names a drug or gives a dose (FR-6.5; §4.11);
- no raw score is shown as a probability until it passes calibration (`docs/05` §3.3; §4.12);
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
   rank it, and a rule flags it. On the hand-written dissection case the fused ranking puts it
   first even with the red flags switched off (with the logistic-regression components; §6.11).
3. **A pre-registered evaluation with three evidence levels.** The protocol was frozen before any
   model existed. Amendment 3 added scoring at 100%, 50% and 25% of each history, with the masks
   drawn once and their digests recorded before any system was scored on them.
4. **A measured finding about the data and the models.** On full DDXPlus histories every ML model
   is near-perfect, because of how the data were generated (R-16). With part of the history
   removed, two models that agreed to 0.0002 in top-1 diverge sharply. We traced this to the
   feature encoding, which could not tell a denied question from one never asked (R-18), and
   repaired it with masked training copies and an explicit "asked" channel (EXP-017 to EXP-019).
5. **A small deep ranker adopted only through a safety floor fixed before training.** A small
   FT-Transformer-style model (B1-DL) replaced logistic regression as the configured ranker only
   after it met the safety targets and was shown non-inferior to logistic regression at 25% of the
   history on validation (decision D-13, EXP-021). It runs on a laptop CPU as a NumPy forward pass,
   and its size and latency are measured (EXP-022).
6. **The ablation study** that isolates what each component adds, including negative results
   [pending: EXP-012, test split].

## 1.6 Structure of the report

Chapter 2 gives the clinical background and related work. Chapter 3 describes the data and their
licences. Chapter 4 describes the system as built. Chapter 5 sets out the pre-registered
evaluation protocol. Chapter 6 reports the validation results so far. Chapter 7 states the
limitations, the safety design and every disclosure. Chapter 8 concludes.
