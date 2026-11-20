# Introduction

## 1.1 Overview

Nightingale (Clinical Intelligence and Reasoning System) is a decision-support prototype for **acute chest pain**. Its intended user is a qualified clinician who is assessing a patient with chest pain. The clinician enters the patient's age, sex and findings, each marked present, denied or not asked. The system returns a ranked list of 14 possible causes, called a **differential diagnosis**, with the reasons for each, the findings still missing, the investigations to consider and the questions worth asking next. Nightingale **ranks and flags; it never diagnoses**. We built and tested it on synthetic data only, and it has had no clinical validation. Every result carries this disclaimer: "Clinical decision support — not a diagnosis. Nightingale is a research prototype. Its output is generated from synthetic training data and must not be used for real patient care. Rankings are not certainties and confidence values are estimates. A qualified clinician is responsible for all diagnostic and treatment decisions."

The system ranks the 14 conditions of Table 1.1: the 13 chest-pain conditions present in DDXPlus, a public synthetic dataset of patients [@tchango2022], and aortic dissection, which DDXPlus lacks and which the knowledge graph and a red-flag rule cover. Eight conditions are **must-not-miss**, meaning that a missed or low-ranked case risks death. The system is **closed-world**: a presentation caused by anything else, pneumonia for example, is still ranked as one of the 14, and the dashboard says so.

**Table 1.1.** The 14 conditions in scope (`src/conditions.py`).
| Condition | Category | Must-not-miss | In DDXPlus |
|---|---|:---:|:---:|
| Possible NSTEMI / STEMI (acute myocardial infarction) | Cardiac, ischaemic | yes | yes |
| Unstable angina | Cardiac, ischaemic | yes | yes |
| Stable angina | Cardiac, ischaemic | no | yes |
| Pericarditis | Cardiac, inflammatory | no | yes |
| Myocarditis | Cardiac, inflammatory | yes | yes |
| Acute pulmonary edema | Cardiac, failure | yes | yes |
| Atrial fibrillation | Cardiac, arrhythmia | no | yes |
| PSVT (paroxysmal supraventricular tachycardia) | Cardiac, arrhythmia | no | yes |
| Pulmonary embolism | Non-cardiac mimic | yes | yes |
| Spontaneous pneumothorax | Non-cardiac mimic | yes | yes |
| Boerhaave syndrome (rupture of the oesophagus) | Non-cardiac mimic | yes | yes |
| GERD (gastro-oesophageal reflux disease) | Non-cardiac mimic | no | yes |
| Panic attack | Non-cardiac mimic | no | yes |
| Aortic dissection | Vascular | yes | no: graph and rule only |

**The architecture, in two paragraphs.** A case is first mapped through a **crosswalk**, a hand-built table that links our own finding names to DDXPlus's coded questions and answers. Two independent routes then score the 14 conditions. The first is a machine-learning ranker: a small transformer (B1-DL′+aug, defined below) trained on 255,900 synthetic DDXPlus chest-pain patients and one masked copy of each, with logistic regression (B1-LR′+aug) kept as the backup. The second is a cardiac **knowledge graph**, a network of conditions and findings with 129 nodes and 321 edges, built from three sources: DDXPlus's own condition definitions, the BODHI-S dataset of Eka Care [@bodhis_card] for four conditions, and edges we wrote by hand for aortic dissection from published sources [@hagan2000; @evangelista2016]. Every edge records its source. The graph scores a case by a naive-Bayes log-likelihood. A **logarithmic opinion pool** [@genest1986; @hinton2002] fuses the two scores, giving the graph a weight of 0.05 that we chose on validation data. Red-flag rules, written after published clinical patterns, run independently of the ranker and place any flagged condition first.

The ranking is explained from the graph's paths and from the transformer's own reasons, found by removing one finding at a time and rescoring. The system also lists investigations to consider and the findings worth asking next. Passages from 2,067 open-access articles are shown beside the first three candidates, each with its citation and licence; they never influence the ranking, and their relevance was not judged. A safety layer attaches the disclaimer and removes any sentence that recommends a treatment, and the result names every component that failed. A Streamlit dashboard that re-ranks as each finding is entered, a FastAPI service and a command-line demonstration run on a laptop processor. Three further parts were built but take no part in the ranking or the explanation that ships: an explainer written by a language model failed its pre-registered checkpoint; a calibrator for the fused score was fitted but is not deployed, so the system shows ranks and no probabilities; and a history of synthetic Synthea patients [@walonoski2018] is drawn beside a case for context but never feeds the ranking.

**Labels and evidence levels.** B0 is the prevalence prior, the same ranking for every patient, and B2 is the graph alone. B1 is the machine-learning-only baseline, in two variants: B1-XGB (XGBoost, a tree-ensemble method) and B1-LR (multinomial logistic regression). B1-DL is the transformer. A label with +aug names a model trained with masked copies of the training histories added; a prime (′) names a model whose encoder also records which questions were asked. A0 is the full fused system with B1-DL+aug as its model, and A0′ is the same with B1-DL′+aug; A0′ is the configured system. A1 is A0 without the graph and the red flags, which leaves the model alone, A1b keeps the red flags and drops only the graph, and A4 removes the red flags. An **evidence level** of 100%, 50% or 25% is the share of each patient's questions whose answers the system sees. The masks were drawn once, at random, before any system was scored on them.

**Headline results.** The test split holds 34,657 synthetic patients in scope and was opened once, on 2026-11-09. Figures marked `[TEST]` come from it, each with its 95% bootstrap interval in square brackets (the range that holds the middle 95% of a figure when the patients are resampled 1,000 times). **Must-not-miss recall@3** is the share of patients with a must-not-miss condition whose true condition is among the top three. At 25% of the history it is 0.8710 [0.8659, 0.8764] for B1-XGB and 0.9198 [0.9156, 0.9238] for B1-LR, both below the 0.95 target, and 0.9986 [0.9980, 0.9991] for A0′ (all `[TEST]`); every model trained on masked copies meets the target. Six pre-registered hypotheses are not supported (H1, H1-R, H2, H2-R, H3 and H6; Table 1.2). The fused system is level with the transformer alone: at 25%, A4 and A1 both have a mean reciprocal rank (MRR, the average of 1 divided by the rank of the true condition) of 0.9639 [0.9623, 0.9652] (`[TEST]`). The graph alone ranks the true condition first for 0.9237 [0.9209, 0.9264] of patients (B2, 100%, `[TEST]`), and that is circularity, not skill: DDXPlus generated its patients from the definitions the graph is built from. The red-flag layer lowers top-1 accuracy at full evidence, from 0.9987 [0.9983, 0.9991] for B1-DL+aug alone to 0.8346 [0.8303, 0.8382] for A0 (100%, `[TEST]`), because it lists flagged conditions first even when the flag names another condition. The grounded language-model explainer had an unsupported-claim rate of 0.741 [0.663, 0.803] (A0+LLM, 100%, 50 validation patients) against a bar of under 0.20, so it does not ship. What the results support is narrower than the hypotheses: training on masked copies, not fusion, keeps the safety target when the history is short, on synthetic, closed-world data.

**Contributions.** We do not claim that combining a knowledge graph with machine learning or a language model is new; that combination exists [@jiang2024; @rotmensch2017]. The project contributes:

1. a differential ranker for one high-stakes presentation that is judged first on must-not-miss recall, with independent red-flag rules after published patterns;
2. a graph that covers a diagnosis absent from the training data: aortic dissection, from 20 hand-authored edges, ranks first on the hand-written case GC-003 even with the red flags switched off;
3. a pre-registered evaluation at three evidence levels, in which the test split was opened once, through a gate that refused to run unless the code was frozen;
4. a measured finding about incomplete histories: an encoding that could not tell a denied question from one never asked made the original XGBoost model answer atrial fibrillation with probability 1.000 when given no findings, and masked training copies repaired it;
5. a small transformer adopted only through a safety floor fixed before training, which runs on a laptop processor;
6. an ablation study with its negative results, and a pre-registered checkpoint that stopped the language-model explainer.

Chapter 2 reviews related work and sets out the objectives. Chapter 3 describes the data, Chapter 4 the methodology and the system as built, Chapter 5 the results on validation and on the test split, and Chapter 6 the conclusions and future scope.

## 1.2 Problem Statement

Acute chest pain is a frequent cause of emergency department visits [@gulati2021]. The clinician must separate the causes that kill if missed, among them myocardial infarction, pulmonary embolism, pneumothorax, aortic dissection and rupture of the oesophagus, from benign ones such as reflux and panic attack. This often has to be done from the history and examination alone, before confirmatory tests return. The two kinds of error are not equal: a false alarm costs a test, and a missed myocardial infarction can cost a life.

**The problem.** Given a structured chest-pain presentation (age, sex and findings, each marked present, denied or not asked), produce a ranked differential over a fixed set of conditions so that (a) a must-not-miss condition is rarely ranked low, measured as must-not-miss recall@3 with a target of at least 0.95; (b) the clinician can see which findings support, contradict or are missing for each candidate; (c) every statement of evidence traces to a source the system holds; and (d) the system reports plainly when a component has failed, instead of degrading silently.

**The gaps.** Five gaps in existing aids motivate the project.

- **G1, safety is not the measured goal.** A ranking judged by top-1 accuracy can hide the loss of a rare, dangerous condition. We therefore made must-not-miss recall the headline metric.
- **G2, no inspectable reasoning.** A learned classifier returns a label or a probability, and a clinician cannot audit why a condition ranks high. A useful aid also shows what information is still missing.
- **G3, fluent text without support.** Language models write convincing clinical text but can state false facts [@ji2023; @pal2023], so a statement must be tied to a source.
- **G4, incomplete histories.** A history is often short. The works we cite do not state how a fixed classifier behaves when part of a history is removed (Section 2.2).
- **G5, an untested combination.** Combining a graph with a learned ranker is not new, but whether it beats its parts for one presentation is a question to test. A condition absent from the training data, aortic dissection, cannot be learned at all.

**Research questions.** The questions, hypotheses and success criteria were written in an evaluation protocol that we froze on 2026-09-17, before any model was trained. Table 1.2 gives them as they stand at version 1.5 of the protocol, with each verdict on the test split. A hypothesis counts as supported only when every comparison it lists passes. The suffix R marks the version judged at reduced evidence. H1-R, H2-R and H6 are judged at 25% evidence since amendment 3b; their 50% comparisons are reported beside them as descriptive.

**Table 1.2.** Research questions, hypotheses and verdicts (`docs/05` v1.5; `[TEST]`, seed 42, except H3, which is validation only). The seed count says how many of the five seeds, 42 to 46, agree.
| RQ | Research question | Hypothesis | Verdict |
|---|---|---|---|
| RQ1 | Does fusing graph reasoning with a machine-learning ranker improve differential accuracy over either alone? | H1: fusion beats ML-only and graph-only on top-3 accuracy and MRR at 100% evidence. H1-R: the same at 25%, against every ML-only system, including those trained like the fused system's model | **Not supported** (both), at none of 5 seeds |
| RQ2 | Does the system improve safety over an ML-only baseline? | H2: fusion plus red flags beats ML-only on must-not-miss recall@3 at 100%. H2-R: the same at 25% | **Not supported** (both), at none of 5 seeds. At 100% every machine-learning comparator misses 0 of 16,972 patients with a must-not-miss condition |
| RQ3 | Does evidence grounding reduce unsupported claims compared with an unconstrained language model? | H3: a retrieval-grounded explanation has a lower unsupported-claim rate than a language model alone | **Not supported** (validation, 50 patients; two of its four comparisons pass). The grounded explainer scored 0.741 [0.663, 0.803] against a bar of under 0.20 (A0+LLM, 100%) |
| RQ4 | Which components contribute? | H4: removing the graph lowers must-not-miss recall more than it lowers top-1 | **Supported by the rule, but hollow**, at 5 of 5 seeds: removing the graph changes must-not-miss recall@3 by exactly 0.0000, and the contrast comes from the red flags lowering top-1 |
| RQ5 | Are the confidence scores calibrated? | H5: expected calibration error (ECE, the gap between stated confidence and observed accuracy) after calibration is below 0.10 | **Supported** at 5 of 5 seeds, on raw scores: 0.0099 [0.0094, 0.0103] for A0 and 0.0398 [0.0394, 0.0402] for A0′ (100%). No calibrator ships |
| RQ6 | Does a deep ranker hold up better than classical machine learning when the history is incomplete? | H6: the transformer beats each classical model trained like it on top-3, MRR and must-not-miss recall@3 at 25% | **Not supported**, at none of 5 seeds: it beats the logistic regression and is level with the XGBoost |

Three facts shape these hypotheses. First, H1 and H2 cannot be supported on full DDXPlus histories: every machine-learning comparator ranks the true condition in the top 3 for all 34,657 patients at 100% (`[TEST]`), because DDXPlus draws each patient's evidence from their own condition's list (risk R-16). Scoring at 50% and 25% evidence (amendment 3) was the protocol's response. Second, as first approved, the reduced-evidence claims needed both 50% and 25%, and at 50% the comparators trained on masked copies were already perfect or nearly so. The project owner, the team member who holds the repository, took the recommendation to judge them at 25% only (2026-11-07); the other team members' agreement is not recorded in the repository. Third, a negative result is a reportable result, and this report states every failed hypothesis as a failure.

## 1.3 Existing System

Clinicians today rely on clinical knowledge, validated prediction rules and guidelines. Software aids sit beside them: symptom checkers, learned classifiers, knowledge-graph systems and, more recently, language models. Table 1.3 lists what each does and what it leaves open for our problem.

**Table 1.3.** Existing aids for chest-pain assessment and what they leave open.
| Existing aid | What it does | What it leaves open | Sources |
|---|---|---|---|
| Clinical prediction rules: the aortic dissection detection risk score (ADD-RS), the Wells criteria, the Braunwald classification | Score one condition from a short list of findings | Each covers one condition; some need findings that a short history may lack; none ranks the whole differential | [@hiratzka2010; @rogers2011; @braunwald1989]; Wells: [[citation needed: Wells criteria, original publication]] |
| Guidelines | Structure the evaluation of chest pain and the tests to consider | Written for a clinician to read and apply; they are not a tool that scores a given case | [@gulati2021] |
| Symptom checkers | Put reported symptoms to a disease-symptom model | We evaluated none: [[citation needed: published evaluations of symptom checkers]] | [@rotmensch2017] |
| Learned classifiers, for example on DDXPlus | Output a label or a probability for each condition | The reasoning is not inspectable; the cited works do not state their behaviour with incomplete histories or on must-not-miss conditions | [@tchango2022; @alam2023] |
| Knowledge-graph systems | State medical relations explicitly, curated or learned from records, and use them for prediction | Those we cite were built for other tasks: drug repurposing, precision medicine, general prediction | [@himmelstein2017; @chandak2023; @rotmensch2017; @jiang2024] |
| Language models, with or without retrieval | Write fluent clinical text | Can state false facts; grounding in retrieved passages is meant to reduce this but must be checked | [@ji2023; @pal2023; @lewis2020] |

We kept something from each aid. The prediction rules became the red-flag layer, each rule following a published pattern and running independently of the ranker. The knowledge graph became the source of explanations. The learned classifier became the ranker. Retrieval became the evidence panel, shown beside the ranking and never used to rank, and the language model became a tested option that did not pass its checkpoint. Four points remain open, and the project is built around them: no cited aid ranks the whole chest-pain differential with must-not-miss recall as its measured goal (G1); learned rankers give no reasoning a clinician can check (G2); generated text needs a source for every claim (G3); and the behaviour of a fixed ranker on a short history is not stated in the works we cite (G4).

## 1.4 User Requirement Analysis

**Users.** The software requirements specification (SRS), written on 2026-09-17 and adapted from the IEEE 830 standard, names three user classes (Table 1.4). The clinician is the primary user. We did not test the system with a clinician, so how well it serves that user is not measured.

**Table 1.4.** User classes, their needs and how the build serves them (`docs/01` §2.2).
| User | Who | Needs | How the build serves them |
|---|---|---|---|
| Clinician (primary) | Qualified medical staff evaluating a chest-pain case; low technical skill | Dangerous causes in view; the reasons for each rank; what is missing; a fast answer; clear limits | Red flags listed first; supporting, missing and contradicting findings; what to ask next; ranks, not probabilities; live re-ranking; the disclaimer on every screen |
| Researcher (team) | Runs experiments, evaluates and ablates; high technical skill | Reproducible runs; one record of every number; a guard on the test split | The ablation runner with its gate; the experiment log; fixed seeds and recorded commits |
| Evaluator or supervisor | Assesses the project; medium technical skill | Honest reporting; a working demonstration | This report; the demo guide; 1,494 passing tests; the pre-registered protocol |

**Requirements.** The SRS lists 45 functional requirements (FR), each with a priority (M, must have; S, should have; C, could have), and 10 non-functional requirements (NFR). Tables 1.5 to 1.9 give each one's status as the system now stands, judged from the code and the project's records. "Met" means built and checked; "partly met" says what is missing.

**Table 1.5.** Functional requirements for case intake, clinical language processing and the patient graph.
| ID | Requirement | Priority | Status as built |
|---|---|:---:|---|
| FR-1.1 | Accept a structured case: age, sex, findings with qualifiers, risk factors, vitals | M | **Partly met.** Age, sex, findings and risk factors are accepted; some qualifiers are separate concepts, such as radiation to the jaw or arm. The contract accepts vitals, but no component reads them |
| FR-1.2 | Accept free text as a supplement | S | **Not used, by design.** The contract accepts free text and every component ignores it; a test checks that a hostile text changes no ranking |
| FR-1.3 | Mark a finding present, absent or unknown | M | **Met.** The dashboard and the API keep denied and not asked apart |
| FR-1.4 | Reject malformed cases with a clear error | M | **Met.** A malformed case is refused with HTTP 422 and field errors |
| FR-1.5 | Load a synthetic case from Synthea or DDXPlus | M | **Partly met.** The four golden cases (hand-written vignettes) load as examples, and a Synthea patient is shown as a display-only card; the dashboard has no loader for DDXPlus patients |
| FR-2.1 to 2.4 | Extract entities, negation and temporality from free text and map them to concepts | S, C | **Not built.** The language-processing module is empty; cases are entered as structured findings |
| FR-3.1 | Construct a patient graph of findings, qualifiers and assertion status | M | **Partly met.** Built for 1,013 synthetic Synthea patients as display-only cards (of 1,359 generated); a typed case is scored as a set of concepts, not as its own graph |
| FR-3.2 | Preserve provenance | S | **Met.** Each finding records its source |
| FR-3.3 | Discard the patient graph after the session | M | **Met.** A case is held in memory only |

**Table 1.6.** Functional requirements for ranking, reasoning and explanation.
| ID | Requirement | Priority | Status as built |
|---|---|:---:|---|
| FR-4.1 | A machine-learning score for each condition | M | **Met.** The transformer scores 13 conditions; aortic dissection, absent from the training data, is scored by the graph |
| FR-4.2 | A graph score from the patient-to-graph match | M | **Met.** A naive-Bayes log-likelihood over the graph |
| FR-4.3 | Fuse both into one ranking | M | **Met.** A logarithmic opinion pool with graph weight 0.05; the fusion adds nothing measurable (Chapter 5) |
| FR-4.4 | Order by fused score and keep the per-signal scores | M | **Met.** Each candidate keeps its model, graph and fused scores; the dashboard shows their ranks |
| FR-4.5 | Calibrate before showing any value as a probability | S | **Met by not showing one.** A calibrator (temperature scaling) was fitted and not deployed; the system shows ranks only |
| FR-5.1 to 5.3 | List supporting, missing and contradicting findings | M | **Met.** Each candidate shows the three lists |
| FR-5.4 | Expose the graph's reasoning path | M | **Met** |
| FR-5.5 | Suggest the next test or question | C | **Met.** The findings that would narrow the differential most, ranked by the graph, and the investigations to consider, each with its guideline |

**Table 1.7.** Functional requirements for safety, evidence and the language-model explainer.
| ID | Requirement | Priority | Status as built |
|---|---|:---:|---|
| FR-6.1 | Evaluate red-flag rules for every case, independently of the ranker | M | **Met.** One rule per must-not-miss condition; the rules' own sensitivity is below its target (Section 5.4) |
| FR-6.2 | Show a red flag prominently even for a low-ranked condition | M | **Met.** Flagged conditions are listed first |
| FR-6.3 | Support aortic dissection by a graph rule | M | **Met.** 20 hand-authored edges and a rule |
| FR-6.4 | Show the disclaimer on every result | M | **Met.** Attached by the safety layer to every result and screen, and printed on the error-analysis figures |
| FR-6.5 | Never output a treatment or drug recommendation | M | **Met.** The safety layer removes such sentences; evidence passages with treatment language are never stored or shown |
| FR-6.6 | Log inputs, outputs and rule firings for audit | S | **Not built.** The safety layer logs its repairs; there is no audit log |
| FR-7.1 | Retrieve passages for each candidate | S | **Met.** 2,067 articles, 26,190 passages; their relevance was not judged |
| FR-7.2 | Rank passages by relevance | S | **Partly met.** BM25 keyword search and dense vector search, fused by reciprocal rank; no reranking model |
| FR-7.3 | Return a citation with every passage | S | **Met.** Title, identifier, link and licence |
| FR-7.4 | Degrade gracefully if retrieval fails | M | **Met.** No evidence is shown and `rag` is reported as degraded |
| FR-8.1 | Generate a readable explanation | S | **Met by a template explainer.** The language-model explainer was built and failed its checkpoint, so it does not ship |
| FR-8.2 to 8.4 | Constrain generation to retrieved evidence and graph facts; flag unsupported claims; never introduce a condition outside the set | S | **Met for the template explainer**, which states only what the result holds; guardrails check each sentence, by words only. The language-model explainer did not meet them |
| FR-8.5 | Treat free text as untrusted | S | **Met.** Ignored by every component, and tested |

**Table 1.8.** Functional requirements for the interface and the evaluation.
| ID | Requirement | Priority | Status as built |
|---|---|:---:|---|
| FR-9.1 | A dashboard for case entry and result display | M | **Met.** A Streamlit dashboard that re-ranks as each finding is entered |
| FR-9.2 | Show the ranked differential with confidence and red flags | M | **Partly met.** Ranks and red flags are shown; no confidence value, because no calibrator ships and short hand-written cases get near-certain scores |
| FR-9.3 | Show supporting, missing and contradicting findings for each candidate | M | **Met** |
| FR-9.4 | Show evidence with citations | S | **Met**, under a caption saying the relevance is unjudged |
| FR-9.5 | A REST API for diagnosis | M | **Met.** Endpoints for diagnosis, next questions, conditions, concepts and health |
| FR-10.1 | Compute all metrics in the protocol | M | **Partly met.** Ranking, safety and calibration metrics are computed with intervals; retrieval relevance, explanation ratings and the template explainer's unsupported-claim rate were not assessed |
| FR-10.2 | Run the ablations A0 to A7 and A1b at three evidence levels | M | **Partly met.** Run once on the test split; A3, A5 and A6 equal A0 by construction; A2 and A7 are not reproduced in the test record |
| FR-10.3 | Log every run | M | **Met.** The experiment log and each run's record |

**Table 1.9.** Non-functional requirements.
| ID | Category | Status as built |
|---|---|---|
| NFR-1 | Response under 5 s without the language model | **Met.** The whole pipeline takes 13.6 to 16.9 ms at the 95th percentile (p95) on a laptop processor, with the transformer and retrieval (EXP-022); no language model runs |
| NFR-2 | Every ranking traceable to findings or graph paths | **Met.** Findings, graph paths and the transformer's reasons; no clinician has rated an explanation |
| NFR-3 | Must-not-miss recall@3 of at least 0.95; red flags independent of the ranker | **Met by the ranker** at every level for models trained on masked copies; the rules are independent, and their own sensitivity, 0.8285 [0.8229, 0.8344] (A0, 100%, `[TEST]`), is below its 0.95 target |
| NFR-4 | A failed component degrades without crashing | **Met.** The failure is named in the result |
| NFR-5 | Reproducibility | **Met.** Fixed seeds; each run records its commit, library versions and encoder fingerprint; a rehearsal on validation reproduced 1,944 of 1,944 earlier figures |
| NFR-6 | A clinician can interpret a result without training | **Not assessed.** No clinician user study |
| NFR-7 | Typed interfaces, testable modules, at least 70% coverage of core logic | **Partly shown.** Typed contracts and 1,494 passing tests (15 skipped); the coverage figure is not in the project's records [[team: add the coverage figure from the continuous-integration log]] |
| NFR-8 | Synthetic data only; no personal data; untrusted free text | **Met.** Servers listen on 127.0.0.1 only |
| NFR-9 | Runs on Linux, Windows and macOS | **Partly shown.** Continuous integration (CI) runs on Linux; development and the demonstration ran on Windows; macOS was not tried |
| NFR-10 | Non-commercial use; attributions kept | **Met.** The dashboard credits every source, Eka Care included |

Every must-have requirement is met or partly met; none is unmet. Three things were not done: free-text language processing (FR-2.1 to 2.4), the audit log (FR-6.6) and a clinician user study (NFR-6). Three things changed in kind: the dashboard shows ranks instead of confidence values (FR-9.2, FR-4.5), the explanation comes from a template instead of a language model (FR-8.1 to 8.4), and the patient graph is display only (FR-3.1).

## 1.5 Feasibility Study

Table 1.10 gives the verdict for each dimension; the paragraphs that follow give the reasons.

**Table 1.10.** Feasibility verdict by dimension.
| Dimension | Question | Finding | Verdict |
|---|---|---|---|
| Technical | Can the ranking run on a laptop in the time required? | Yes: processor only, 13.6 to 16.9 ms p95 against 5 s required; training needed a cloud GPU | Feasible |
| Economic | Can it be built without paid services? | Yes: free tiers, open data, no licence fee; no spending is recorded | Feasible |
| Operational | Can a clinician operate it, and does it fail safely? | One command starts it; failures are reported; no clinician has tried it | Feasible as a prototype; usability not assessed |
| Legal and ethical | Can the data and models be used and shared? | Non-commercial use only, with attribution; synthetic data only; not a medical device | Feasible with conditions |
| Schedule | Can the plan be met? | The core was delivered; the language-model explainer does not ship and evidence relevance was not judged | Met with a reduced scope |

**Technical.** We asked two technical questions early and answered both by building. The first was whether the project's finding names could be aligned with DDXPlus and with BODHI-S. A feasibility spike in week 1 found that only 4 of 13 conditions matched BODHI-S exactly (31% against a gate of 60%), so we built the graph from DDXPlus's definitions instead, which was the fallback written in advance, and accepted the circularity that follows (risk R-12). The second question was whether the whole ranking can run on a laptop. It can. The transformer runs as a NumPy forward pass, occupies 1.23 MB, and takes 1.38 ms at p95 for one patient (one thread), and the whole pipeline takes 13.6 to 16.9 ms at p95 on the golden cases (EXP-022), against a requirement of under 5 s. Training is heavier: the transformers were trained on a cloud GPU (Kaggle, two T4 GPUs, 2 hours 10 minutes for ten networks), because the project's compute rule sends every training run to the cloud unless the owner chooses otherwise. The graph runs on the free tier of AuraDB and falls back to a local copy when that service is paused, which the system reports.

**Economic.** The design rule was no paid service: no paid language-model service, free cloud tiers (Kaggle notebooks, AuraDB Free), open data and open software. The hardware was the team's laptops. DDXPlus is licensed CC-BY-4.0 and Synthea Apache-2.0, and each open-access article carries its own licence. The project's files record no spending.

**Operational.** The dashboard starts with one command, listens on 127.0.0.1 only and re-ranks as each finding is entered. A case is held in memory and discarded. A failed component never crashes a result; `degraded_components` names it. Two limits apply: no clinician has tested the dashboard (NFR-6), and the system is a research prototype, not a tool for use in a clinic.

**Legal and ethical.** Table 1.11 lists the licences, which we re-verified on 2026-11-10 from the sources' public metadata. The project is non-commercial in three independent ways: BODHI-S, the embedding model that produced the passage vectors, and 844 of the 2,067 articles. Attribution to Eka Care is mandatory and is on the dashboard. Only synthetic and open data are used, so the project judged that it needs no real patient record, no data-use agreement and no ethics-board approval (`docs/04` §5) [[team: confirm with the university's project guidelines that no ethics approval is required]]. Nightingale is not a medical device and has had no regulatory review. The report makes no claim of clinical efficacy.

**Table 1.11.** Data and model licences used (`docs/03` §1.5, checked 2026-11-10).
| Source | Used for | Licence as found | Consequence |
|---|---|---|---|
| DDXPlus [@ddxplus_card] | Training and test data; ground truth | CC-BY-4.0 | Attribution |
| BODHI-S [@bodhis_card] | Enrichment of four conditions in the graph | CC-BY-NC-4.0 | Non-commercial; credit Eka Care; no BODHI-S text is committed |
| Synthea | Display-only patient histories | Apache-2.0 | Permissive licence |
| PubMed Central open-access articles | Evidence passages | Each article's own licence; 844 of 2,067 non-commercial | Licence shown with every excerpt |
| Passage-embedding model | Vectors computed once on Kaggle | CC-BY-NC-2.0 | Retrieval bundle is non-commercial |
| Llama 3.1 8B | The language-model trial only; not shipped | Meta's community licence; the page is gated and was not re-verified | Nothing derived from it is distributed |

**Schedule.** The plan had five phases in an 8 to 10 week budget, with Week 1 starting on 2026-09-17 (Table 1.12). On 2026-10-12 the date for presenting the prototype and the report moved earlier, from 2026-11-18 to 2026-11-15, which removed the buffer week. The plan also had a cut line with dated checkpoints, fixed in advance. It named the prototype, the red-flag layer and the ablation study as never to be cut, and it said that a language-model explainer that was not grounded would fall back to the template explanations. That is what happened: the explainer failed its checkpoint and stays out of the product, and the judging of retrieval relevance was dropped because the team had no time. The `v1.0` tag had not been made at the time of writing [[team: update if the tag has been made]].

**Table 1.12.** The plan against what happened (`PROGRESS.md`; dates are 2026).
| Phase | Planned end | What happened |
|---|---|---|
| 0, preparation and documentation | 2026-09-23 | Done 2026-09-17: documents, scaffold, feasibility spike, protocol frozen |
| 1, data and knowledge foundations | 2026-10-07, moved to 2026-10-21 | Walking skeleton reached 2026-09-24; last piece, the patient-history panel, done 2026-11-10 |
| 2, reasoning, fusion and prototype | 2026-10-21, moved to 2026-10-25 | First prototype tagged 2026-10-14; every part done; the demo video is left to the team |
| 3, evidence and explanation | 2026-11-04, moved to 2026-11-03 | Second prototype tagged 2026-11-08; closed 2026-11-10 with retrieval relevance not judged |
| 4, evaluation, ablations and write-up | 2026-11-18, moved to 2026-11-15 | Test split opened once on 2026-11-09; report text final on 2026-11-12 |

**Risks that were realised.** The risk register opened 20 risks. Table 1.13 lists the ones that happened, with what each did to the results.

**Table 1.13.** Realised risks (`docs/07`).
| Risk | What happened | Effect |
|---|---|---|
| R-02, timeline too short | The date moved earlier and no buffer week remained | The cut line was applied (above) |
| R-04, language-model hallucination | The grounded explainer's unsupported-claim rate was 0.741 [0.663, 0.803] (A0+LLM, 100%, validation) | The explainer does not ship; the template does |
| R-09, fusion no better than ML alone | The fused system is level with the transformer alone | Reported as a finding (R-12, R-16) |
| R-12, circularity | The graph and the learned ranker share DDXPlus as their source; the graph alone ranks the true condition first for 0.9237 [0.9209, 0.9264] of patients (B2, 100%, `[TEST]`) | Every graph figure on DDXPlus carries the caveat; the graph's own value cannot be measured there |
| R-16, ceiling on full histories | At 100% every machine-learning comparator misses 0 of 34,657 patients in the top 3 (`[TEST]`) | H1, H2 and the first-written Target cannot be supported at 100%; 50% and 25% levels were added |
| R-17, case-to-token inversion has no ground truth | Radiation to the jaw was mapped to an answer that only pulmonary-edema patients give in DDXPlus (decision A-10) | One of 17 answers was moved; the other 16 are not yet checked against the data |
| R-19, deep ranker no better | It beats the logistic regression and is level with the XGBoost trained like it (H6) | Reported; the logistic regression stays as the backup |
| R-20, evidence of unmeasured relevance, partly non-commercial | The relevance judging was dropped; 844 of 2,067 articles are non-commercial | The dashboard says so; the project stays non-commercial |

# Literature Review

## 2.1 Review of Related Work

We review the work in the order in which the system uses it: clinical rules and guidelines, the dataset, knowledge graphs, the pooling and calibration of scores, deep learning on tables with missing answers, retrieval and language models, and alert fatigue. Every journal and preprint reference was looked up in Crossref or arXiv on 2026-11-11, and the two dataset cards were read on 2026-11-10. Where we could not verify a source, the text carries a marker for the team to resolve.

### 2.1.1 Computer-aided diagnosis and clinical prediction rules

Computer programs have supported diagnosis for decades, from probabilistic models to expert systems [[citation needed: early computer-aided diagnosis, for example probabilistic diagnosis and naive Bayes for acute abdominal pain]]. One form is the clinical prediction rule: a short list of findings that scores one condition. Three rules stand behind our red flags.

The aortic dissection detection risk score (ADD-RS) comes from the 2010 guideline on thoracic aortic disease [@hiratzka2010]. It groups its markers into three categories: high-risk conditions (for example connective tissue disease), high-risk pain features (abrupt onset, severe intensity, a ripping or tearing character) and high-risk examination features (for example a pulse deficit or a blood-pressure difference between the arms). Rogers et al. reported its sensitivity at initial presentation [@rogers2011]. The International Registry of Acute Aortic Dissection (IRAD) describes how often findings occur in patients with dissection [@hagan2000; @evangelista2016]; we took these frequencies as weights for the graph's hand-authored edges. The Wells criteria for pulmonary embolism score signs of deep-vein thrombosis, immobilisation, recent surgery and previous thrombosis [[citation needed: Wells criteria, original publication]]. Braunwald's classification of unstable angina [@braunwald1989] supplies the history features, such as a crescendo pattern or pain at rest, that our unstable-angina rule uses.

Each rule covers one condition. A clinician facing undifferentiated chest pain must hold several in mind, and none ranks the whole differential. Our red-flag layer and our ranker fill that gap together.

### 2.1.2 Chest pain guidelines and the must-not-miss conditions

The 2021 multi-society guideline of the American Heart Association and the American College of Cardiology structures the evaluation and diagnosis of chest pain [@gulati2021]. For the investigations the system suggests, we cite the guideline of the relevant specialty: the 2021 chest pain guideline [@gulati2021], the British Thoracic Society guideline on pleural disease [@roberts2023], the World Society of Emergency Surgery guidelines on oesophageal emergencies [@chirica2019] and several European Society of Cardiology guidelines [[citation needed: the European Society of Cardiology guidelines named in src/reasoning/investigations.py, with full references]]. These tables list investigations to consider. They never recommend a treatment.

Eight of our 14 conditions are must-not-miss: myocardial infarction, unstable angina, pulmonary embolism, spontaneous pneumothorax, Boerhaave syndrome, myocarditis, acute pulmonary edema and aortic dissection. A history rarely settles the question alone. In the summary our team used as its domain primer (a teaching summary, not a clinical reference), the confirmatory tests are the troponin (a blood marker of heart-muscle damage), the electrocardiogram (ECG), the D-dimer (a blood test used to rule pulmonary embolism out) and the chest X-ray. This is why the system ranks and flags but never concludes.

### 2.1.3 Synthetic patients and automatic diagnosis

DDXPlus [@tchango2022] is a public synthetic dataset for automatic medical diagnosis: about 1.3 million generated patients across 49 pathologies, each with age, sex, a list of coded evidence (symptoms and antecedents), the presenting complaint, the true pathology and a ranked differential with probabilities. Its licence is CC-BY-4.0 [@ddxplus_card]. The patients are generated from a curated knowledge base of conditions and their evidence. We found the English text of the release to be machine-translated from French and unreliable: the French descriptor for tearing pain reads "heartbreaking" in English, so we work from the evidence codes. Two properties of the generation matter later. Each patient's evidence is drawn only from their own condition's list, so on full histories a classifier can be near-perfect. And our knowledge graph is built from the same knowledge base, so any graph figure on DDXPlus is circular (Chapter 5).

DDxT is a generative transformer for differential diagnosis [@alam2023]. We did not review how it or other published deep models treat incomplete histories [[citation needed: published deep models on DDXPlus, and whether any is evaluated on incomplete histories]].

Two other sources were considered. Synthea generates synthetic patient records [@walonoski2018]; we used it for display-only patient histories (1,359 patients generated, 1,013 alive on 2026-01-01). PTB-XL is a public electrocardiography dataset [@wagner2020]; an ECG extension was a stretch goal that we did not pursue.

### 2.1.4 Knowledge graphs for clinical reasoning

A knowledge graph (KG) states medical relations explicitly, for example that chest pain radiating to the jaw is linked to myocardial infarction, so a system can show a path instead of a score [[citation needed: knowledge graphs in clinical decision support]]. Open biomedical graphs include Hetionet, built to prioritise drugs for repurposing [@himmelstein2017], and PrimeKG, built for precision medicine [@chandak2023]; the Human Phenotype Ontology is a standard vocabulary of phenotypes [@kohler2021]. Rotmensch et al. learned a disease-symptom graph from electronic medical records [@rotmensch2017], and GraphCare builds patient-specific graphs, from language models and biomedical graphs, to train a neural network for healthcare predictions [@jiang2024]. These graphs are broad or built for other tasks, and we chose none as our backbone.

BODHI-S [@bodhis_card] is a condition-symptom graph from Eka Care whose facts carry likelihood bands; it is licensed CC-BY-NC-4.0. It was our planned backbone. A feasibility spike in week 1 found that only 4 of our 13 conditions matched it exactly (31%, against a gate of 60%), because it covers 555 conditions across medicine and merges distinctions the differential needs, such as one node for both anginas (EXP-001). We therefore built the graph from DDXPlus's own condition definitions and used BODHI-S to enrich four conditions. A crosswalk, whose entries use the SKOS relations (exact, close, broader, narrower, related) [[citation needed: the W3C SKOS Primer]], links our finding names to DDXPlus's answers so that each link allows only sound inferences. We compared two scores for the graph, Personalised PageRank [[citation needed: Personalised PageRank]] and a naive-Bayes log-likelihood [[citation needed: naive Bayes for diagnosis]], and chose naive Bayes (EXP-005).

### 2.1.5 Fusing opinions and calibrating scores

When two systems each give a probability for every condition, a **logarithmic opinion pool** combines them as a weighted geometric mean that is renormalised [@genest1986]. It is also known as a product of experts [@hinton2002]. A pool multiplies, so one expert's near-zero probability vetoes a condition whatever the other says. We guard against this by flooring the model's probabilities at 0.01, and a condition the model cannot score, aortic dissection, takes the graph's own probability.

A model that says 70% should be right about 70% of the time. Calibration is measured by the expected calibration error (ECE) [[citation needed: the expected calibration error]], the Brier score [@brier1950] and reliability diagrams. We compared Platt scaling [[citation needed: Platt, probabilistic outputs for support vector machines]], fitted for each condition against the rest [@zadrozny2002], isotonic regression [@zadrozny2002] and temperature scaling [@guo2017]. We explain the transformer by occlusion, removing each recorded finding and rescoring [[citation needed: occlusion-based attribution]]. SHAP [@lundberg2017] was planned and is not used.

### 2.1.6 Deep learning on tables, and missing answers

Our deep ranker is an FT-Transformer [@gorishniy2021] at a smaller size. Each input feature becomes a token, and attention lets every finding see every other, which suits DDXPlus, where exertional pain means one thing beside rest pain and another without it. Its feed-forward layers use ReGLU [@shazeer2020] and it is trained with AdamW [@loshchilov2019]. Tree ensembles remain strong competitors on tabular data [@gorishniy2021], so we kept XGBoost and logistic regression as comparators.

When inputs are missing, the best predictor depends on which values are missing as well as on those observed [@lemorvan2020]. In DDXPlus an unlisted question means "no", so "not asked" and "denied" are two different states that an encoding must keep apart. We do so with an explicit "asked" channel and with masked training copies (Chapter 4). A denied finding counts against a condition; a finding that was not asked counts for nothing.

### 2.1.7 Retrieval and language models

Retrieval-augmented generation (RAG) grounds each statement in a retrieved, citable passage [@lewis2020]. Language models can state false facts [@ji2023], including in the medical domain [@pal2023]; whether they cite their sources reliably and whether their confidence is calibrated needs its own source [[citation needed: language-model source attribution and calibration in clinical text]]. We tested the grounded approach in Phase 3, with an 8-billion-parameter model run on a cloud notebook and a corpus of 2,067 open-access articles; Section 5.5 reports that it failed its checkpoint.

### 2.1.8 Alert fatigue

A flag raised for half of all patients risks being ignored. This is alert fatigue, and Ancker et al. studied how workload, work complexity and repeated alerts affect it in a clinical decision support system [@ancker2017]. We therefore judge the red-flag rules both by whom they reach and by whom they flag without reason, and we measure the alarm burden on patients without a must-not-miss condition, never on all patients, because about half of our test patients have one (16,972 of 34,657).

## 2.2 Comparison

Table 2.1 sets the works we cite beside this project, and Table 2.2 compares them on the dimensions that matter here. "n/s" means the record we hold does not state it; it does not mean the work lacks the property.

**Table 2.1.** Prominent related works compared with this project.
| Work | Data | Approach | Output | Gap against this project |
|---|---|---|---|---|
| DDXPlus [@tchango2022] | Synthetic: about 1.3 million patients, 49 pathologies | Benchmark generated from a curated knowledge base | Coded evidence, pathology and a differential with probabilities | A dataset, not a system: no ranker and no explanation; also the source of our circularity |
| DDxT [@alam2023] | n/s | Generative transformer | A differential diagnosis | Safety focus, incomplete histories and explanation: n/s |
| Learned health knowledge graph [@rotmensch2017] | Electronic medical records | Disease-symptom graph learned from the records | A graph for decision support and symptom checking | Learned from records, not curated for one presentation; no red flags; n/s |
| Hetionet and PrimeKG [@himmelstein2017; @chandak2023] | Integrated biomedical knowledge | Large graphs for drug repurposing and precision medicine | A graph | Broad: no differential for chest pain |
| GraphCare [@jiang2024] | n/s | Patient-specific graphs from language models and biomedical graphs, with a neural network | Healthcare predictions | Predicts outcomes, not a chest-pain differential; no red flags; n/s |
| ADD-RS, Braunwald classification [@hiratzka2010; @rogers2011; @braunwald1989] | Guidelines and registry data | Hand-built scores | A risk category for one condition | One condition each; our red flags follow them and add the other conditions |
| Hallucination studies [@ji2023; @pal2023] | A survey and a medical benchmark | Measure false statements by language models | Evidence of the risk | The risk we tested; our grounded explainer failed |
| **This project** | Synthetic DDXPlus patients; four hand-written cases | Transformer and graph pooled; independent red-flag rules; retrieval beside the ranking | A ranked 14-condition differential, red flags, reasons, next questions, evidence | Closed-world; synthetic; no clinical validation |

**Table 2.2.** Dimension-by-dimension comparison (figures for this project: `[TEST]` unless marked).
| Dimension | What the cited works state | This project |
|---|---|---|
| Data | DDXPlus is synthetic [@tchango2022]; Rotmensch et al. learn from electronic medical records [@rotmensch2017] | Synthetic DDXPlus patients and four hand-written cases; no real patient data; Synthea histories are display only |
| Approach | A generative transformer [@alam2023]; a learned graph [@rotmensch2017]; patient graphs with a neural network [@jiang2024]; one-condition scores [@rogers2011; @braunwald1989] | A transformer and a graph pooled; independent rules; retrieval beside the ranking, never inside it |
| Explanation | Graph-based work offers a graph, and rules offer a score's items; for DDxT, n/s | Supporting, missing and contradicting findings with graph paths; the transformer's reasons by occlusion; template text checked sentence by sentence |
| Safety focus | Rules target one dangerous condition each [@hiratzka2010; @braunwald1989]; the other works, n/s | Eight must-not-miss conditions, a rule each, and must-not-miss recall@3 as the headline metric; the rules' sensitivity is 0.8285 [0.8229, 0.8344] (A0, 100%), below its target |
| Incomplete histories | n/s in the works we cite [[citation needed: the team's literature review of 2026-10-11, which is not in the repository; list the studies it covered]] | Scored at 100%, 50% and 25%; at 25% B1-XGB reaches must-not-miss recall@3 of 0.8710 [0.8659, 0.8764] and A0′ 0.9986 [0.9980, 0.9991] |
| Uncertainty | DDXPlus's differential carries probabilities [@tchango2022]; calibration methods are standard [@guo2017]; others n/s | Raw ECE 0.0398 [0.0394, 0.0402] (A0′, 100%); no probability is shown; hand-written cases get near-certain scores (R-17) |
| Evidence used | Retrieval grounds statements in passages [@lewis2020]; language models may state false facts [@ji2023; @pal2023] | 2,067 articles, 26,190 passages, shown beside the ranking, relevance unjudged; the language-model explainer's unsupported-claim rate was 0.741 [0.663, 0.803] (A0+LLM, 100%, validation) |

The comparison shows where this project sits and where it does not. Among the works we cite, none reports, for one presentation, a differential judged first on must-not-miss recall, a fixed ranker scored on histories with answers removed, and a grounded explainer checked against a stated unsupported-claim bar. That is a statement about the works in Table 2.1, not about the whole literature. Our results do not show that combining a graph with a learned ranker helps: on DDXPlus the fusion adds nothing measurable. What the project adds is the evaluation: three evidence levels fixed before scoring, a test split opened once, and every failure reported.

## 2.3 Objectives of Project

The charter set eight objectives on 2026-09-17, before any model was built. Each answers one or more of the gaps G1 to G5 of Section 1.2. Table 2.3 defines them and their measurable outcomes; Table 2.4 gives each one's final status and what stands behind it.

**Table 2.3.** Objectives, the gaps they answer, and their measurable outcomes (`docs/00` §3).
| ID | Objective | Gap | Measurable outcome |
|---|---|---|---|
| O1 | Build a patient knowledge graph from case data | G2 | Built for at least 95% of test cases |
| O2 | Build a cardiac knowledge graph of condition-finding links | G2, G5 | All 13 conditions, with qualifiers |
| O3 | Build a machine-learning ranker that produces a ranked differential | G1, G4 | Top-3 accuracy well above the prevalence baseline |
| O4 | Fuse the graph and the ranker into one explainable ranking | G5 | Fusion beats both alone |
| O5 | Detect and surface must-not-miss conditions as red flags | G1 | Must-not-miss recall@3 of at least 0.95 |
| O6 | Retrieve and cite real evidence for each candidate | G3 | At least 80% of generated claims traceable to a retrieved passage |
| O7 | Quantify each component's contribution | G4, G5 | An ablation study, at three evidence levels |
| O8 | Deliver a usable clinician-facing interface | G2 | A working dashboard over the full pipeline |

**Table 2.4.** Final status of each objective (`[TEST]` unless marked).
| ID | Final status | What stands behind it |
|---|---|---|
| O1 | **Met in a reduced form** | Built from Synthea's synthetic records, not from the chest-pain cases: 1,013 patients alive on 2026-01-01, of 1,359 generated. It is display only and never an input (the project owner's decision of 2026-10-20). The 95% outcome, worded for test cases, was not assessed as worded, because a DDXPlus patient has no history |
| O2 | **Met** | 129 nodes and 321 edges from three sources (245 from DDXPlus, 56 from BODHI-S, 20 hand-authored) over 14 conditions; the source of every edge is recorded |
| O3 | **Met** | Logistic regression, XGBoost and a small transformer, which is the configured ranker. Top-3 at 100%: 1.0000 [1.0000, 1.0000] for B1-DL+aug against 0.3061 [0.3013, 0.3109] for B0, at a ceiling (R-16) |
| O4 | **Not met as worded** | The fusion is built, with graph weight 0.05, but H1 and H1-R are not supported and the fused system is level with its own model (A4 and A1, 25%: MRR 0.9639 [0.9623, 0.9652] for both) |
| O5 | **Met** | One rule per must-not-miss condition. Must-not-miss recall@3 meets the 0.95 target at every level for models trained on masked copies (A0′, 25%: 0.9986 [0.9980, 0.9991]). The rules alone reach a sensitivity of 0.8285 [0.8229, 0.8344] (A0, 100%), below their own 0.95 target |
| O6 | **Not met as worded** (built, not assessed) | The corpus (2,067 articles, 26,190 passages) is wired, and each excerpt shows its citation and licence. The 80% outcome was not assessed: relevance was never judged, and the component that generates claims failed (0.741 [0.663, 0.803] unsupported, A0+LLM, 100%, validation) |
| O7 | **Met** | A0 to A7 and A1b scored at three levels and run once on the test split; A3, A5 and A6 equal A0 by construction, and A2 and A7 are not reproduced in the test record |
| O8 | **Met** | A Streamlit dashboard that re-ranks live, a FastAPI service and a command-line demonstration; not tested with clinicians |

Plainly: O2, O3, O5, O7 and O8 were met; O1 was met in a reduced form; O4 and O6 were not met as worded. O4 is the project's central research question, and its answer is negative on this benchmark.

**Success criteria.** The charter set three levels (Table 2.5). The Minimum is met. The Target and the Stretch were written for full histories, where every machine-learning comparator is already perfect (R-16), so by the protocol's own reading they are not met at 100%. The reduced-evidence Target is met by A0′ and not by A0, by a handful of patients, and the fused system equals its own component to within a patient, so this is not a robust win.

**Table 2.5.** Success criteria and their results (`docs/00` §6 and `docs/05` §7, as amended; `[TEST]`).
| Level | Criteria | Result |
|---|---|---|
| Minimum | A working prototype that returns a ranked, explained differential with red flags; complete documentation; evaluation against baselines | **Met** at 5 of 5 seeds. Top-3 at 100%: 1.0000 [1.0000, 1.0000] for A0 against 0.3061 [0.3013, 0.3109] for B0 |
| Target | The above, plus retrieval-grounded evidence with citations, calibrated confidence and a complete ablation study | Evidence with citations is built and its relevance unjudged; a calibrator was fitted and not deployed; the ablation study is done. By the protocol's reading, **not met at 100%** (the comparators tie at 1.000) and **met at 25% by A0′ only**: A0′ misses 40 of 34,657 patients in the top 3, B1-XGB′+aug 56; A0 misses 73 against B1-XGB+aug's 71 |
| Stretch | "What to ask next"; an ECG extension; a publishable write-up | What to ask next is built; the ECG extension was not pursued; this report is the write-up. By the protocol's reading, **not met at 100% or at 25%** by either arm; the template explainer's unsupported-claim rate and clinician-reviewed explainability were not assessed |

Table 2.6 sets the project's numeric targets beside the results.

**Table 2.6.** Numeric targets and where each stands (`[TEST]` unless marked).
| Target | Result |
|---|---|
| Must-not-miss recall@3 at least 0.95 | **Met** at 25% by every model trained on masked copies (A0 0.9981 [0.9974, 0.9987]; A0′ 0.9986 [0.9980, 0.9991]); missed by B1-XGB 0.8710 [0.8659, 0.8764] and B1-LR 0.9198 [0.9156, 0.9238] |
| Dangerous false-negative rate at most 0.02 | **Met** at 25% by A0 0.0003 [0.0001, 0.0006] and A0′ 0.0002 [0.0000, 0.0004]; missed by B1-XGB 0.0629 [0.0592, 0.0665] and B1-LR 0.0272 [0.0249, 0.0297] |
| Red-flag sensitivity at least 0.95 | **Not met** (A0): 0.8285 [0.8229, 0.8344] at 100%, 0.3318 [0.3245, 0.3386] at 50%, 0.1369 [0.1317, 0.1420] at 25% |
| ECE below 0.10 | **Met** on raw scores at 100%: A0 0.0099 [0.0094, 0.0103]; A0′ 0.0398 [0.0394, 0.0402] |
| Unsupported claims below 20% | **Not met** by the language-model explainer (0.741 [0.663, 0.803], A0+LLM, 100%, validation), which does not ship; **not assessed** for the template explainer |
| Response under 5 s without the language model | **Met**: 13.6 to 16.9 ms at p95 for the whole pipeline on a laptop processor |

```abbreviations
A = Ablation configuration (A0 is the full system; A0′ is the configured one)
ADD-RS = Aortic Dissection Detection Risk Score
API = Application Programming Interface
B = Baseline (B0 prevalence prior, B1 machine learning only, B2 graph alone)
BM25 = Best Matching 25 (a keyword-ranking function)
CC-BY = Creative Commons Attribution (licence)
CC-BY-NC = Creative Commons Attribution-NonCommercial (licence)
CI = Continuous Integration
DL = Deep Learning
ECE = Expected Calibration Error
ECG = Electrocardiogram
EXP = Experiment (an entry in the project's experiment log)
FR = Functional Requirement
FT-Transformer = Feature Tokenizer Transformer
G = Gap (G1 to G5, Section 1.2)
GC = Golden Case (a hand-written clinical vignette enforced by tests)
GERD = Gastro-oesophageal Reflux Disease
GPU = Graphics Processing Unit
H = Hypothesis (H1 to H6; the suffix R marks the reduced-evidence version)
HTTP = Hypertext Transfer Protocol
ID = Identifier
IEEE = Institute of Electrical and Electronics Engineers
IRAD = International Registry of Acute Aortic Dissection
KG = Knowledge Graph
LLM = Large Language Model
LR = Logistic Regression
MB = Megabyte
ML = Machine Learning
MRR = Mean Reciprocal Rank
NFR = Non-Functional Requirement
NSTEMI = Non-ST-Elevation Myocardial Infarction
O = Objective (O1 to O8, Section 2.3)
p95 = 95th percentile
PSVT = Paroxysmal Supraventricular Tachycardia
R = Risk (an entry in the project's risk register, for example R-12)
RAG = Retrieval-Augmented Generation
ReGLU = Rectified-linear Gated Linear Unit
REST = Representational State Transfer
RQ = Research Question
SHAP = SHapley Additive exPlanations
SKOS = Simple Knowledge Organization System
SRS = Software Requirements Specification
STEMI = ST-Elevation Myocardial Infarction
T4 = a model of NVIDIA graphics processor
XGB = XGBoost (eXtreme Gradient Boosting)
aug = augmented with masked copies of the training histories
```
