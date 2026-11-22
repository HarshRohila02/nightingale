# Introduction

## 1.1 Overview

Nightingale (Clinical Intelligence and Reasoning System) is a decision-support prototype for **acute chest pain**. A qualified clinician enters a patient's age, sex and findings (present, denied or not asked), and the system returns a ranked list of 14 possible causes (a **differential diagnosis**) with reasons and the questions worth asking next. It **ranks and flags; it never diagnoses**, and it has had no clinical validation. Every result carries this disclaimer: "Clinical decision support — not a diagnosis. Nightingale is a research prototype. Its output is generated from synthetic training data and must not be used for real patient care. Rankings are not certainties and confidence values are estimates. A qualified clinician is responsible for all diagnostic and treatment decisions."

The 14 conditions (Table 1.1) are the 13 chest-pain conditions in DDXPlus, a public synthetic dataset [@tchango2022], and aortic dissection, which DDXPlus lacks and the graph and a red-flag rule cover. Eight are **must-not-miss**: a missed case risks death. The system is **closed-world**: a presentation with any other cause, pneumonia for example, is still ranked as one of the 14.

**Table 1.1.** The 14 conditions in scope; must-not-miss conditions in bold.
| Group | Conditions |
|---|---|
| Cardiac, ischaemic | **Myocardial infarction (NSTEMI or STEMI)**; **unstable angina**; stable angina |
| Cardiac, other | Pericarditis; **myocarditis**; **acute pulmonary edema**; atrial fibrillation; paroxysmal supraventricular tachycardia (PSVT) |
| Non-cardiac mimics | **Pulmonary embolism**; **spontaneous pneumothorax**; **Boerhaave syndrome** (rupture of the oesophagus); gastro-oesophageal reflux disease (GERD); panic attack |
| Vascular | **Aortic dissection** (graph and red-flag rule only; not in DDXPlus) |

A **crosswalk**, a hand-built table, links our finding names to DDXPlus's coded questions. Two independent routes then score the conditions: a small transformer trained on synthetic DDXPlus patients, and a **knowledge graph** (129 nodes, 321 edges) built from DDXPlus's condition definitions, the BODHI-S dataset [@bodhis_card] and 20 edges we wrote for aortic dissection [@hagan2000; @evangelista2016]. A **logarithmic opinion pool** [@genest1986; @hinton2002] fuses the two, weighting the graph 0.05, a value chosen on validation data. Red-flag rules, written after published clinical patterns, run independently and list flagged conditions first. Evidence passages from 2,067 open-access articles are shown beside the top candidates, never used to rank; their relevance was not judged.

**Labels.** B0 is the prevalence prior and B2 the graph alone. B1 is machine learning (ML) only: B1-XGB (XGBoost), B1-LR (logistic regression) and B1-DL (the transformer). A suffix +aug marks training with masked copies of the histories added; a prime (′) marks an encoder that also records which questions were asked. A0 is the fused system with B1-DL+aug and A0′ the same with B1-DL′+aug, the configured system; A1 is the model alone and A4 the fused system without red flags. An **evidence level** of 100%, 50% or 25% is the share of each patient's questions whose answers the system sees (random masks, drawn once before scoring).

**Headline.** Figures marked `[TEST]` come from the test split (34,657 synthetic patients, opened once, 2026-11-09), with 95% bootstrap intervals (the range over 1,000 resamples of the patients). **Must-not-miss recall@3** is the share of must-not-miss patients whose true condition is in the top three. At 25% evidence, every model trained on masked copies keeps it above the 0.95 target (A0′: 0.9986 [0.9980, 0.9991], `[TEST]`); the original XGBoost and logistic regression do not (Section 5.3). Six hypotheses are not supported (Table 1.2). The graph alone ranks the true condition first for 0.9237 [0.9209, 0.9264] of patients (B2, 100%, `[TEST]`), which is circularity, not skill: DDXPlus generated its patients from the definitions the graph is built from.

## 1.2 Problem Statement

Acute chest pain is a frequent cause of emergency visits [@gulati2021], and the causes that kill if missed must be told from benign ones, often from the history alone. A false alarm costs a test; a missed infarction can cost a life.

**The problem.** Rank, from a structured presentation, a fixed set of conditions so that (a) a must-not-miss condition is rarely ranked low (recall@3 of at least 0.95); (b) the clinician sees which findings support, contradict or are missing; (c) every statement of evidence traces to a source the system holds; and (d) a failed component is reported.

**Gaps.** G1: safety is rarely the measured goal, and top-1 accuracy can hide the loss of a rare, dangerous condition. G2: a learned classifier's reasoning cannot be inspected. G3: language models write fluent text that may be false [@ji2023; @pal2023]. G4: the works we cite do not state how a fixed classifier behaves on incomplete histories. G5: a graph plus a learned ranker is untested against its parts, and aortic dissection, absent from the training data, cannot be learned.

The protocol (frozen 2026-09-17 before any model was trained; now v1.5) counts a hypothesis as supported only when every comparison it lists passes; the suffix R marks the 25% version.

**Table 1.2.** Research questions and verdicts (`[TEST]`, seed 42, 100% unless marked; H3 on validation). "At none of 5 seeds": none of seeds 42 to 46.
| Research question (RQ) | Hypothesis (H) | Verdict |
|---|---|---|
| RQ1. Does fusing graph and ML ranker improve accuracy? | H1: fusion beats ML-only and graph-only on top-3 accuracy and mean reciprocal rank (MRR) at 100%. H1-R: the same at 25%, against every ML-only system, including those trained like its model | **Not supported** (both), at none of 5 seeds |
| RQ2. Does the system improve safety over ML-only? | H2: fusion plus red flags beats ML-only on must-not-miss recall@3 at 100%. H2-R: the same at 25% | **Not supported** (both), at none of 5 seeds; at 100% every ML comparator misses 0 of 16,972 patients with a must-not-miss condition |
| RQ3. Does grounding reduce unsupported claims? | H3: a retrieval-grounded explanation has a lower unsupported-claim rate than a language model alone | **Not supported** (50 validation patients; two of four comparisons pass): 0.741 [0.663, 0.803] against a bar of under 0.20 (A0+LLM, 100%) |
| RQ4. Which components contribute? | H4: removing the graph lowers must-not-miss recall more than top-1 | **Supported by the rule, but hollow**, at 5 of 5 seeds: removing the graph changes must-not-miss recall@3 by exactly 0.0000; the contrast comes from the red flags lowering top-1 |
| RQ5. Are confidence scores calibrated? | H5: expected calibration error (ECE, the gap between stated confidence and observed accuracy) after calibration below 0.10 | **Supported** at 5 of 5 seeds, on raw scores: 0.0099 [0.0094, 0.0103] for A0 and 0.0398 [0.0394, 0.0402] for A0′ (100%); no calibrator ships |
| RQ6. Does a deep ranker hold up better on incomplete histories? | H6: the transformer beats each classical model trained like it on top-3, MRR and must-not-miss recall@3 at 25% | **Not supported**, at none of 5 seeds: it beats the logistic regression and is level with the XGBoost |

H1 and H2 cannot be supported on full histories, where every ML comparator ranks every test patient's true condition in the top 3 (risk R-16). The project owner took the recommendation to judge the reduced-evidence claims at 25% only (2026-11-07); the other members' agreement is not recorded.

## 1.3 Existing System

Clinicians rely on knowledge, prediction rules and guidelines. Prediction rules (the aortic dissection detection risk score, ADD-RS [@hiratzka2010]; Braunwald [@braunwald1989]; Wells) each score one condition, and none ranks the whole differential; guidelines [@gulati2021] are read by a clinician, not applied to a case. Symptom checkers and learned classifiers, such as those on DDXPlus [@tchango2022; @alam2023], give a label or probability whose reasoning cannot be inspected, and the works we cite do not state their behaviour on incomplete histories [[citation needed: published evaluations of symptom checkers]]. Knowledge-graph systems [@rotmensch2017; @jiang2024] were built for other tasks. Language models write fluent text but can state false facts, and grounding in retrieved passages [@lewis2020] must be checked. We kept something from each: the rules became the red-flag layer, the graph the source of explanations, the classifier the ranker and retrieval the evidence panel. Gaps G1 to G4 remain open.

## 1.4 User Requirement Analysis

The software requirements specification (SRS) of 2026-09-17 (after IEEE 830) names three user classes: the clinician (primary; needs the dangerous causes in view, reasons for each rank, what is missing, speed and clear limits), the researcher (the team) and the evaluator or supervisor. No clinician tested the system, so its fit to that user is not measured.

The SRS lists 45 functional requirements (FR), each with a priority (M, must have; S, should have; C, could have), and 10 non-functional requirements (NFR). Table 1.3 gives each family's status as built, judged from the code and records.

**Table 1.3.** Requirement families and their status as built.
| Family | Count | Priority | Status as built |
|---|:---:|:---:|---|
| Intake (FR-1 to 3) | 12 | M, S, C | **Partly met.** Present, denied and not-asked findings, and rejection of malformed cases, are met; vitals are accepted but unread; free text is ignored by design and tested; the patient graph is a display-only card for 1,013 synthetic Synthea patients; free-text language processing (FR-2.1 to 2.4) is **not built** |
| Ranking and reasoning (FR-4, 5) | 10 | M, S, C | **Met.** The transformer scores 13 conditions and the graph all 14; the pool adds nothing measurable; supporting, missing and contradicting findings, graph paths, next questions and investigations are shown. Ranks only, no probability (no calibrator ships; short hand-written cases get near-certain scores, R-17) |
| Safety (FR-6) | 6 | M, S | **Met, except the audit log (not built).** One rule per must-not-miss condition, flagged first; aortic dissection from 20 hand-authored edges; the disclaimer on every result; no treatment output. The rules' own sensitivity is below its target |
| Evidence (FR-7) | 4 | S, M | **Met in form.** 2,067 articles, 26,190 passages with citation and licence; keyword and vector search without a reranking model; relevance not judged; degrades gracefully |
| Explanation (FR-8) | 5 | S | **Met by a template explainer.** The language-model explainer failed its checkpoint and does not ship; guardrails check each sentence by words only |
| Interface and API, the application programming interface (FR-9) | 5 | M, S | **Met, except confidence.** A Streamlit dashboard that re-ranks as each finding is entered and a REST API; no confidence value is shown |
| Evaluation (FR-10) | 3 | M | **Partly met.** Metrics with intervals are computed; retrieval relevance, explanation ratings and the template's unsupported-claim rate were not assessed; A3, A5 and A6 equal A0 by construction, and A2 and A7 are not reproduced in the test record |
| Non-functional (NFR-1 to 10) | 10 | none | **Mostly met.** Whole pipeline 13.6 to 16.9 ms at the 95th percentile (p95) against 5 s; red-flag sensitivity 0.8285 [0.8229, 0.8344] (A0, 100%, `[TEST]`) is below its 0.95 target; 1,494 tests pass (15 skipped), coverage not in the records [[team: add the coverage figure from the continuous-integration log]]; synthetic data only; continuous integration (CI) on Linux, macOS untried; clinician interpretability **not assessed** |

Every must-have requirement is met or partly met. Three things were not done: free-text processing, the audit log and a clinician user study (NFR-6).

## 1.5 Feasibility Study

**Table 1.4.** Feasibility verdict by dimension.
| Dimension | Finding | Verdict |
|---|---|---|
| Technical | Runs on a laptop processor; whole pipeline 13.6 to 16.9 ms p95 against 5 s required; training needed a cloud GPU | Feasible |
| Economic | Free tiers, open data and software; no spending recorded | Feasible |
| Operational | One command starts it; failures are named; no clinician has tried it | Feasible as a prototype; usability not assessed |
| Legal and ethical | Synthetic and open data; non-commercial; not a medical device | Feasible with conditions |
| Schedule | Core delivered; language-model explainer does not ship; relevance not judged | Met with a reduced scope |

**Technical.** A week-1 spike (Section 2.1) moved the graph's backbone from BODHI-S to DDXPlus's definitions, which causes the circularity of risk R-12. The transformer runs as a NumPy forward pass of 1.23 MB and takes 1.38 ms at p95 for one patient (experiment EXP-022); training ran on a cloud graphics processing unit (GPU).

**Legal and ethical.** DDXPlus is CC-BY-4.0 and Synthea Apache-2.0. BODHI-S (CC-BY-NC-4.0), the embedding model (CC-BY-NC-2.0) and 844 of the 2,067 articles are non-commercial, so the project is non-commercial, with Eka Care credited (Appendix A). Only synthetic and open data are used, so we judged that no ethics approval is needed [[team: confirm with the university's project guidelines]]. Nightingale is not a medical device.

**Schedule.** The plan had five phases in an 8 to 10 week budget from 2026-09-17; on 2026-10-12 the presentation date moved to 2026-11-15 and the buffer week was lost. The cut line fixed in advance was applied: the language-model explainer failed its checkpoint and stays out, and the judging of retrieval relevance was dropped for lack of time. Realised risks, among them R-04 (hallucination), R-12 (circularity) and R-16 (ceiling), are in the risk register. The `v1.0` tag had not been made at the time of writing [[team: update if made]].

# Literature Review

## 2.1 Review of Related Work

We review the work in the order the system uses it. References were checked in Crossref or arXiv on 2026-11-11 and the dataset card read on 2026-11-10; unverified sources are marked.

**Rules and guidelines.** A clinical prediction rule scores one condition from a short list of findings; three stand behind our red flags. The ADD-RS [@hiratzka2010] scores high-risk conditions, pain features and examination features. The International Registry of Acute Aortic Dissection (IRAD) gives how often findings occur in dissection [@hagan2000], and we used these frequencies to weight our hand-authored edges. The Wells criteria for pulmonary embolism [[citation needed: Wells criteria, original publication]] and Braunwald's classification of unstable angina [@braunwald1989] supply the other patterns. The 2021 chest pain guideline [@gulati2021] structures the evaluation, and the investigations our system lists come from it and from specialty guidelines [[citation needed: specialty guidelines named in src/reasoning/investigations.py]]; none is a treatment.

**Synthetic data and learned classifiers.** DDXPlus [@tchango2022] is a public synthetic dataset of about 1.3 million generated patients across 49 pathologies. Its English text is machine-translated from French and unreliable, so we work from evidence codes. Each patient's evidence is drawn only from their own condition's list, so on full histories a classifier can be near-perfect, and our graph is built from the same knowledge base, so any graph figure on DDXPlus is circular. DDxT, a generative transformer for differential diagnosis [@alam2023], is the nearest published model; whether it or others are evaluated on incomplete histories is not stated in what we reviewed [[citation needed: published deep models on DDXPlus on incomplete histories; the team's 2026-10-11 literature review is not in the repository]].

**Knowledge graphs.** A knowledge graph states medical relations explicitly, so a system can show a path. Rotmensch et al. learned a disease-symptom graph from electronic medical records [@rotmensch2017]; GraphCare builds patient-specific graphs, from language models and biomedical graphs, to train a neural network for healthcare predictions [@jiang2024]. BODHI-S [@bodhis_card], a condition-symptom graph from Eka Care, was our planned backbone, but a week-1 spike found that only 4 of our 13 conditions matched it exactly (31% against a gate of 60%; EXP-001), because it covers 555 conditions and merges distinctions the differential needs. We built the graph from DDXPlus's definitions and used BODHI-S to enrich four conditions.

**Pooling, deep learning on tables and missing answers.** A logarithmic opinion pool combines probabilities as a renormalised weighted geometric mean [@genest1986]. Because it multiplies, one near-zero estimate vetoes a condition, so we floor the model's probabilities at 0.01, and a condition the model cannot score (aortic dissection) takes the graph's own probability. Our deep ranker is a small FT-Transformer [@gorishniy2021], which turns each input into a token; XGBoost and logistic regression are our comparators. When answers are missing, the best predictor depends on which are missing [@lemorvan2020]. In DDXPlus an unlisted question means "no", so "not asked" and "denied" differ; we keep them apart with an "asked" channel and masked training copies (Chapter 4).

**Retrieval, language models and alert fatigue.** Retrieval-augmented generation grounds each statement in a retrieved, citable passage [@lewis2020], but language models can state false facts [@ji2023], including in medicine [@pal2023]; Section 5.5 reports that our grounded explainer failed its checkpoint. Flags raised for many patients risk alert fatigue [@ancker2017], so we measure the alarm burden on patients without a must-not-miss condition.

## 2.2 Comparison

**Table 2.1.** Prominent related works compared with this project ("n/s": the record we hold does not state it).
| Work | Data | Approach and output | Gap against this project |
|---|---|---|---|
| DDXPlus [@tchango2022] | Synthetic: about 1.3 million patients, 49 pathologies | Benchmark generated from a curated knowledge base: coded evidence, pathology, differential with probabilities | A dataset, not a system: no ranker or explanation; also the source of our circularity |
| DDxT [@alam2023] | n/s | Generative transformer; a differential diagnosis | Safety focus, incomplete histories and explanation: n/s |
| Learned health graph [@rotmensch2017] | Electronic medical records | Disease-symptom graph learned from records, for decision support and symptom checking | Learned, not curated for one presentation; no red flags; n/s |
| GraphCare [@jiang2024] | n/s | Patient-specific graphs from language models and biomedical graphs, with a neural network; healthcare predictions | Predicts outcomes, not a chest-pain differential; no red flags; n/s |
| ADD-RS, Braunwald classification [@hiratzka2010; @rogers2011; @braunwald1989] | Guidelines and registry data | Hand-built scores; a risk category for one condition | One condition each; our red flags follow them and add the other conditions |
| Hallucination studies [@ji2023; @pal2023] | A survey and a medical benchmark | Measure false statements by language models | The risk we tested; our grounded explainer failed its checkpoint |
| **This project** | Synthetic DDXPlus patients; four hand-written cases | Transformer and graph pooled; independent red-flag rules; retrieval beside the ranking; a ranked 14-condition differential with red flags, reasons, next questions and evidence | Closed-world; synthetic; no clinical validation |

Among the works we cite, none reports, for one presentation, a differential judged first on must-not-miss recall, a fixed ranker scored on histories with answers removed, and a grounded explainer checked against a stated unsupported-claim bar; that is a statement about Table 2.1, not the whole literature. Fusion adds nothing measurable here; the project adds the evaluation: three evidence levels fixed before scoring, one opening of the test split, every failure reported.

## 2.3 Objectives of Project

The charter set eight objectives on 2026-09-17, before any model was built; each answers one or more of the gaps G1 to G5 (Table 2.2).

**Table 2.2.** Objectives, the gaps they answer and their final status (`[TEST]` unless marked).
| ID | Objective and measurable outcome | Gap | Final status and what stands behind it |
|---|---|---|---|
| O1 | A patient knowledge graph from case data; built for at least 95% of test cases | G2 | **Met in a reduced form.** Built from Synthea records (1,013 patients alive on 2026-01-01, of 1,359 generated); display only, never an input (the project owner's decision of 2026-10-20). The outcome as worded was not assessed, because a DDXPlus patient has no history |
| O2 | A cardiac knowledge graph; all 13 conditions with qualifiers | G2, G5 | **Met.** 129 nodes and 321 edges (245 from DDXPlus, 56 from BODHI-S, 20 hand-authored) over 14 conditions; every edge records its source |
| O3 | A machine-learning ranker; top-3 accuracy well above the prevalence baseline | G1, G4 | **Met.** Top-3 at 100%: 1.0000 [1.0000, 1.0000] for B1-DL+aug against 0.3061 [0.3013, 0.3109] for B0, at a ceiling (R-16) |
| O4 | Fuse graph and ranker into one explainable ranking; fusion beats both alone | G5 | **Not met as worded.** Fusion is built (graph weight 0.05), but H1 and H1-R are not supported and the fused system is level with its own model (A4 and A1, 25%: MRR 0.9639 [0.9623, 0.9652] for both) |
| O5 | Detect and surface must-not-miss conditions; recall@3 of at least 0.95 | G1 | **Met.** At 25%, A0′ 0.9986 [0.9980, 0.9991] and every model trained on masked copies. The rules alone reach sensitivity 0.8285 [0.8229, 0.8344] at 100% (A0), 0.3318 [0.3245, 0.3386] at 50% and 0.1369 [0.1317, 0.1420] at 25%, below their own 0.95 target |
| O6 | Retrieve and cite real evidence; at least 80% of generated claims traceable | G3 | **Not met as worded (built, not assessed).** The corpus is wired and each excerpt shows citation and licence; relevance was never judged, and the claim generator failed (0.741 [0.663, 0.803] unsupported, A0+LLM, 100%, validation) |
| O7 | Quantify each component's contribution; an ablation study at three evidence levels | G4, G5 | **Met.** A0 to A7 and A1b scored at three levels and run once on the test split; A3, A5 and A6 equal A0 by construction; A2 and A7 are not reproduced in the test record |
| O8 | A usable clinician-facing interface; a working dashboard | G2 | **Met.** A dashboard that re-ranks live, a FastAPI service and a command-line demonstration; not tested with clinicians |

Plainly: O2, O3, O5, O7 and O8 were met; O1 in a reduced form; O4 and O6 not as worded. O4 is the central research question, and its answer is negative on this benchmark.

**Success criteria.** The Minimum (a working, explained, ranked prototype with red flags, documentation and baselines) is met at 5 of 5 seeds. The Target and Stretch were written for full histories, where every ML comparator is already perfect (R-16), so they are not met at 100%. The reduced-evidence Target is met by A0′ and not by A0, by a handful of patients (A0′ misses 40 of 34,657 in the top 3, B1-XGB′+aug 56; A0 misses 73, B1-XGB+aug 71), so it is not a robust win; the Stretch is met at neither level, and its explanation criteria were not assessed.

```abbreviations
A = Ablation configuration (A0 is the full system; A0′ is the configured one)
ADD-RS = Aortic Dissection Detection Risk Score
API = Application Programming Interface
B = Baseline (B0 prevalence prior, B1 machine learning only, B2 graph alone)
CC-BY = Creative Commons Attribution (licence)
CC-BY-NC = Creative Commons Attribution-NonCommercial (licence)
CI = Continuous Integration
DL = Deep Learning
ECE = Expected Calibration Error
EXP = Experiment (an entry in the project's experiment log)
FR = Functional Requirement
FT-Transformer = Feature Tokenizer Transformer
G = Gap (G1 to G5, Section 1.2)
GERD = Gastro-oesophageal Reflux Disease
GPU = Graphics Processing Unit
H = Hypothesis (H1 to H6; the suffix R marks the reduced-evidence version)
IEEE = Institute of Electrical and Electronics Engineers
IRAD = International Registry of Acute Aortic Dissection
LR = Logistic Regression
ML = Machine Learning
MRR = Mean Reciprocal Rank
NFR = Non-Functional Requirement
NSTEMI = Non-ST-Elevation Myocardial Infarction
O = Objective (O1 to O8, Section 2.3)
p95 = 95th Percentile
PSVT = Paroxysmal Supraventricular Tachycardia
R = Risk (an entry in the project's risk register, for example R-12)
REST = Representational State Transfer
RQ = Research Question
SRS = Software Requirements Specification
STEMI = ST-Elevation Myocardial Infarction
XGB = XGBoost (eXtreme Gradient Boosting)
aug = augmented with masked copies of the training histories
```
