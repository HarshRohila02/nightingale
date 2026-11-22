# Exploratory Data Analysis

## 3.1 Dataset

Every dataset in this project is synthetic, openly licensed, or both. No real patient record enters the system, the training, the evaluation or this report. Table 3.1 lists each source with its licence, size and role. We first read the licences on 2026-09-17 and re-verified them on 2026-11-10, which added the embedding model and the language model to the register and struck out two sources that were never used (UCI Heart Disease and PTB-XL). Identifiers such as EXP-004 (an entry in the experiment log), R-12 (a risk in the risk register), D-13 (a project decision) and A-8 (an architecture decision) point to the project's records, where every number in this report can be found.

**Table 3.1.** Data and knowledge sources, with licence, size and role (licences as read on 2026-11-10).

| Source | Licence | Size | Role | Status |
|---|---|---|---|---|
| DDXPlus | CC-BY-4.0 | 49 pathologies; 1,025,602 training, 132,448 validation and 134,529 test patients | Patients and their true conditions; the backbone of the knowledge graph | In use; test split opened once |
| BODHI-S | CC-BY-NC-4.0 (non-commercial) | 555 conditions; two files of 13,204 rows | Enriches four conditions of the knowledge graph | In use |
| Hand-authored aortic dissection | Our own work, from published sources | 20 graph edges | The only knowledge of a condition DDXPlus lacks | In use |
| Golden clinical cases | Our own work, synthetic | 4 cases | Executable clinical expectations | In use |
| PubMed Central Open Access articles | Per article: 15 Creative Commons licence strings; 844 of 2,067 articles non-commercial | 2,067 articles, 26,190 passages | Evidence excerpts shown beside candidates | In use; relevance never judged |
| Synthea | Apache-2.0 | 1,359 patients, 1,013 alive on 2026-01-01 | Patient-history card | In use, display only |
| S-PubMedBert-MS-MARCO | CC-BY-NC-2.0 (non-commercial) | 768-dimensional passage vectors | Passage vectors, computed once on Kaggle | In use, offline |
| Llama 3.1 8B | Meta's community licence (gated page not re-verified) | One model | Language-model trial only | Tried, not shipped |

### 3.1.1 DDXPlus

DDXPlus [@tchango2022] is the main source. A simulator drew its patients from a curated knowledge base of 49 pathologies, each with its symptoms, risk factors, an ICD-10 code and a severity ranking. A patient record holds an age, a sex, a list of coded evidence tokens (each token answers a question put to the simulated patient), the presenting complaint, the true pathology, and a ranked differential diagnosis with probabilities. The dataset is released under CC-BY-4.0 [@ddxplus_card]. We pin the Hugging Face revision `2ad986ac…`, the card's current one, so that every run reads the same files.

We keep the patients whose true pathology is one of 13 chest-pain conditions (Table 3.2). Seven of the 13 are must-not-miss conditions, which means that a missed one risks death. A fourteenth condition, aortic dissection, is also must-not-miss; DDXPlus does not contain it, and Section 3.1.3 explains how the system learns about it.

**Table 3.2.** DDXPlus splits and the chest-pain subset (numbers of patients; EXP-002, EXP-003 and EXP-012).

| Split | Source rows | In scope | Share in scope | Role |
|---|---:|---:|---:|---|
| Training | 1,025,602 | 255,900 | 25.0% | Fitting models, on a cloud GPU only |
| Validation | 132,448 | 33,963 | 25.6% | All development; every figure not tagged `[TEST]` |
| Test | 134,529 | 34,657 | not recorded | Opened once, on 2026-11-09 |

The test split stayed closed during development. It opened once, on 2026-11-09, after the tag `v0.9-freeze` had fixed the code and the configuration and after the digests of the evaluation masks had been written into the experiment log (EXP-012). A gate in `src/eval/freeze.py` enforces this (Section 3.3). The test split must not be scored again, and every figure drawn from it carries the tag `[TEST]` in this report.

### 3.1.2 BODHI-S

BODHI-S [@bodhis_card], from Eka Care, is a knowledge graph of 555 conditions across all of medicine. It ships as two line-aligned files of 13,204 rows: structured edges that link a symptom to a condition (conditions are identified by SNOMED CT codes, where DDXPlus uses ICD-10), and the same facts written as sentences. Each fact carries a likelihood band such as rare, medium or high, which DDXPlus has no equivalent for.

BODHI-S enriches only four conditions because it matches only four. In the feasibility study (EXP-001) it matched 4 of our 13 conditions exactly, a coverage of 31% against our gate of 60%. It is broad rather than deep in cardiology: it has no myocarditis, pneumothorax or Boerhaave syndrome, and it merges stable and unstable angina into one node, as it does atrial fibrillation and PSVT. DDXPlus's own condition file therefore became the backbone of the graph, and BODHI-S enriches myocardial infarction, pericarditis, pulmonary embolism and GERD. Of its 107 facts about them, 98 map to the project's concepts and give 56 graph edges (EXP-016). The choice of backbone created the circularity risk R-12 (Section 3.2.6).

BODHI-S is non-commercial: academic use is allowed, commercial use is not, and Eka Care must be credited wherever the graph is used. The built graph inherits the restriction. The repository holds no BODHI-S text, and the team's private AuraDB copy holds identifiers and likelihood bands but no text.

### 3.1.3 The hand-authored aortic dissection and the golden cases

Aortic dissection is a tear in the wall of the body's main artery, rare and deadly if missed. DDXPlus has no such patients, so no model trained on it can rank the condition. We wrote 20 graph edges by hand from the markers of the Aortic Dissection Detection Risk Score (ADD-RS) [@rogers2011; @hiratzka2010], adding the pain's location, sharp pain, syncope, hypertension and cocaine use. Each edge's weight is the frequency of the finding among dissection patients, taken from the International Registry of Acute Aortic Dissection (IRAD) where it reports one [@hagan2000; @evangelista2016], banded as in Table 4.14. Each edge records its evidence and reference.

Four hand-written, synthetic cases encode what a clinician would expect from the system (Table 3.3). They run in the test suite, and when one fails we fix the component and never the expectation. They have no ground truth beyond that expectation, and each reaches the model through a hand-curated translation (assumption A-8, Section 4.9). A fifth case, a benign-looking ischaemic presentation that raises no red flag, was not added, because the freeze fixes the folder of golden cases. No golden case therefore tests that presentation.

**Table 3.3.** The golden clinical cases (`tests/fixtures/golden_cases.yaml`).

| Case | Presentation | Expectation |
|---|---|---|
| GC-001 | Classic acute coronary syndrome: a 58-year-old man with exertional crushing central chest pain radiating to the jaw and left arm, sweating; diabetic smoker | Myocardial infarction in the top 3; an infarction red flag |
| GC-002 | Pulmonary embolism: sudden pleuritic chest pain and breathlessness, one swollen leg after recent immobilisation | Embolism in the top 3; an embolism red flag |
| GC-003 | Aortic dissection: tearing pain radiating to the back with an inter-arm blood-pressure difference | Dissection in the top 5; a dissection red flag |
| GC-004 | Benign reflux: burning pain after meals, worse lying flat, ischaemic features explicitly denied | GERD in the top 3; no cardiac red flag |

### 3.1.4 The retrieval corpus

The evidence shown beside the first three candidates comes from excerpts of PubMed Central Open Access articles, fetched once by the owner's Kaggle run of 2026-11-07: 2,067 articles and 26,190 passages from 20 searches, with 446 articles skipped. An article is kept only if its own XML names a Creative Commons licence, which we record. There are 15 licence strings, the largest CC BY 4.0 (979 articles), and 844 articles carry a non-commercial term. The corpus is for diagnosis only: a sentence with treatment language is never stored, embedded or shown. It sits in a folder that git ignores, and the limit of 3 requests a second set by NCBI (the National Center for Biotechnology Information) is respected by a 0.35 s wait between requests. The passages' relevance was never judged, because the two team members who were to judge them had no time, so no precision figure exists.

### 3.1.5 Synthea

Synthea [@walonoski2018] is a generator, not a fixed dataset. We ran version 4.0.0, pinned by its release tag, with seed 42 on Kaggle, and obtained 1,359 synthetic patients, 1,013 of them alive on 2026-01-01. A table, not a model, maps Synthea's SNOMED codes to six risk-factor concepts: diabetes, heart failure, hypertension, obesity, pregnancy and smoking. The history is for display only (the owner's decision of 2026-10-20): it is drawn beside a case and never feeds the ranking, the case's findings or its risk factors, and no model is trained or scored on it. The ablation that removes the patient graph (A6) therefore equals the full system by construction.

## 3.2 Exploratory Data Analysis and Visualisations

We audited DDXPlus before building anything on it. The audit found how the classes are balanced, how artificial the demographics are, four forms of evidence token, three traps in the coding and a saturation effect that decided how the system is evaluated. Counts come from the validation split unless the text says otherwise.

### 3.2.1 Class balance

Figure 3.1 shows the number of validation patients per condition, and Table 3.4 gives the numbers beside the training counts. Pulmonary embolism is the largest class and spontaneous pneumothorax the smallest. On the training split the largest class is 2.70 times the smallest, and the smallest has 10,162 patients (EXP-003). The imbalance is mild, so no condition is too rare to learn, and the risk we had registered for rare classes (R-03) did not materialise. The audit also exposed a limit that reaches the whole project: only 25.6% of validation patients fall inside our 13 conditions, so the system is closed-world (R-13). It knows 14 causes of chest pain and nothing else, and a patient with pneumonia is forced into one of the 14. Every result in this report carries that caveat.

![Figure 3.1. Patients per condition in the validation split (33,963 patients); must-not-miss conditions in red.](img/eda_class_balance.png)

**Table 3.4.** The 13 DDXPlus conditions in scope: patients in the validation and training splits, share female and median age (EXP-002 and EXP-003; sex and age were measured on validation only).

| Condition | Validation | Train | % female | Median age | Must-not-miss |
|---|---:|---:|---:|---:|---|
| Pulmonary embolism | 3,725 | 27,468 | 52% | 38 | yes |
| GERD | 3,426 | 25,979 | 52% | 36 | no |
| Panic attack | 3,237 | 25,019 | 51% | 39 | no |
| Pericarditis | 3,032 | 22,785 | 51% | 40 | no |
| Possible NSTEMI / STEMI | 2,943 | 21,260 | 50% | 45 | yes |
| Unstable angina | 2,748 | 21,244 | 54% | 44 | yes |
| Atrial fibrillation | 2,609 | 21,036 | 52% | 36 | no |
| Acute pulmonary edema | 2,500 | 19,018 | 51% | 47 | yes |
| PSVT | 2,376 | 18,781 | 49% | 40 | no |
| Stable angina | 2,340 | 16,995 | 52% | 46 | no |
| Boerhaave syndrome | 2,075 | 15,080 | 51% | 40 | yes |
| Myocarditis | 1,547 | 11,073 | 54% | 35 | yes |
| Spontaneous pneumothorax | 1,405 | 10,162 | 52% | 38 | yes |
| **Total in scope** | **33,963** | **255,900** | | | 7 of 13 |

Of the 33,963 validation patients, 16,943 have a must-not-miss condition, the denominator of every must-not-miss figure on validation. On the test split it is 16,972 of 34,657 patients.

### 3.2.2 Generated demographics

The demographics are a simulator's, not a population's. Figure 3.2 shows the median and interquartile range of age for each condition. Every condition is between 49% and 54% female, and overall 17,500 validation patients (51.5%) are female and 16,463 (48.5%) male. Median ages run from 35 for myocarditis to 47 for acute pulmonary edema. Myocardial infarction has a median age of 45, where a first infarction in the real world typically occurs in the patient's 60s [[citation needed: age at first myocardial infarction]]. Spontaneous pneumothorax is 52% female in DDXPlus, although in reality it is strongly male-predominant [[citation needed: sex distribution of spontaneous pneumothorax]].

![Figure 3.2. Median and interquartile range of age by condition in the validation split (generated ages).](img/eda_age_by_condition.png)

A model trained on this data cannot learn real epidemiology, and a near-equal result by sex or by age is an artefact of how the data were generated. We never present it as evidence of fairness.

### 3.2.3 Findings per patient

A patient lists 21.9 evidence tokens on average (EXP-017). Many are default answers that mean "no" (Section 3.2.5), so we count the questions answered with something other than "no", which we call positive findings. Figure 3.3 shows their distribution: a mean of 14.1 and a median of 14 per patient, with a minimum of 2 and a maximum of 25. A history is often incomplete when a clinician wants a first ranking, so the system must also work with far fewer findings, and we score every system on partial histories too (Section 3.2.7).

![Figure 3.3. Histogram of positive findings per patient in the validation split (full history).](img/eda_findings_per_patient.png)

### 3.2.4 The four forms of evidence token

Evidence is coded, not written: the token `E_54_@_V_161` is the answer `V_161` to the pain-location question `E_54`. The release's vocabulary has 919 entries, made of 223 questions and 696 answer values, and our 13 conditions use 84 of the questions. Tokens come in four forms (Table 3.5), and the encoder of Section 3.2.7 handles each.

**Table 3.5.** Evidence tokens in the chest-pain subset, by form (numbers of tokens; docs/03 §2.1).

| Token form | Example | Validation | Train |
|---|---|---:|---:|
| Binary | `E_91` (fever) | 295,726 | 2,237,663 |
| Categorical value | `E_54_@_V_161` | 353,233 | 2,634,321 |
| Numeric ordinal, 0 to 10 | `E_56_@_4` (pain intensity) | 94,062 | 704,592 |
| The "NA" sentinel | `E_54_@_V_11` | 801 | 6,734 |
| **All tokens** | | **743,822** | **5,583,310** |

The numeric ordinals are 12.6% of tokens, and they must be encoded as ordered numbers and not as one-hot categories, because pain of intensity 8 is closer to 9 than to 2. The NA sentinel belongs to patients with no pain at all: none of the 801 lists the question that links pain to the consultation, and 771 of them have PSVT.

### 3.2.5 Three traps in the data

**A listed token can mean "no".** Every categorical, multi-choice and ordinal question has a default answer that means "no". The token `E_204_@_V_10`, "did not travel abroad", is listed for 89.6% of validation patients, and `E_57_@_V_123` means "the pain radiates nowhere". The validation split holds 45,049 such tokens in 31,189 patients (EXP-013). Reading "the code is listed" as "the patient has it" would give 89.6% of patients a travel history, so every component uses the function `positive_codes`, which returns only the questions answered with something other than "no".

**The English is machine-translated from French and unreliable.** The French word for tearing pain, `déchirante`, the classic descriptor of aortic dissection, is rendered "heartbreaking", and `lancinante` (shooting) is rendered "haunting" (EXP-001 and docs/10). Every component works on evidence codes and never on the English labels, and our hand-curated tables quote the French.

**The ground-truth differential is open-world.** Each patient's ranked differential, D, holds 12.0 conditions on average, of which 6.7 are in scope. For 91.8% of in-scope validation patients (31,185) it includes conditions the system cannot output, and these hold 33.3% of its probability mass on average (median 34.9%, 90th percentile 52.4%). As first defined, Recall@5 could not exceed 0.434 even for a perfect 13-condition system, so it measured the closed-world gap more than the ranking. Decision D-8, taken by the team on 2026-09-19 before any model was trained, restricts the headline Precision@3 and Recall@5 to the in-scope part of D. A perfect system then reaches 0.753 on Recall@5 and 0.929 on Precision@3, which is the same under either definition. Recall@5 on the full D is always reported beside it, so the change cannot flatter a result (EXP-013).

The label itself is never an input. `DIFFERENTIAL_DIAGNOSIS` is stored as `label_differential`, every label column starts with `label_`, and feature code selects the input columns by name (age, sex and evidences). A test shuffles the label columns and checks that the feature matrix does not change.

### 3.2.6 Saturation: why full histories cannot separate the systems

DDXPlus draws each patient's evidence only from the evidence list of that patient's own condition. For every validation patient the positive answers lie inside their own condition's list, and for 91.7% of patients they lie inside no other condition's list (EXP-004). The questions answered "yes" therefore nearly name the condition, and the result is a ceiling effect (risk R-16). On complete validation histories the baseline that uses machine learning alone (B1) is near-perfect: a plain XGBoost model (B1-XGB) ranks the true condition first for 0.9985 [0.9980, 0.9989] of patients and a plain logistic regression (B1-LR) for 0.9983 [0.9978, 0.9987]. Both place the true condition in the top 3 for 1.0000 [1.0000, 1.0000] of patients and reach a must-not-miss recall@3 of 1.0000 [1.0000, 1.0000] (100% evidence, validation, EXP-004). The cause is the data, not skill or leakage: the inputs exclude every label column, and the test split was never read.

Two consequences follow. On complete histories nothing can beat these baselines on top-3 accuracy or must-not-miss recall, so the hypotheses that the full system beats them cannot be supported there. The test split confirms it: at 100% evidence every machine-learning model (B1-XGB, B1-LR, their `+aug` forms, B1-DL+aug and B1-DL′+aug) misses none of the 34,657 patients in the top 3, and so do the full systems A0 and A0′ (`[TEST]`, EXP-012). The graph alone (B2) is the exception, with 27 patients missed, 4 of them must-not-miss. And because the knowledge graph is built from the knowledge base that generated the patients, the graph alone recognises DDXPlus patients almost perfectly. That is circularity (risk R-12), not evidence that the graph knows medicine, and we state it wherever a graph result appears.

The saturation disappears when the history is incomplete, and the two baselines then part company (Table 3.6). The diagnostic masks of EXP-017 keep each patient's initial evidence plus a share of the other tokens; they were a first probe, and Chapter 5 uses the approved question-level masks of Section 3.2.7. With a quarter of the history, XGBoost ranks the true condition first for 0.2671 [0.2625, 0.2718] of patients against 0.8558 [0.8522, 0.8597] for logistic regression, and its must-not-miss recall@3 is 0.9346 [0.9305, 0.9383], below the 0.95 target (risk R-18). Given no evidence at all, XGBoost answers atrial fibrillation with probability 1.000. The encoder gave a default "no" answer no column, so a short row and an unfinished interview looked the same, and atrial fibrillation is the condition whose DDXPlus patients answer the fewest questions. This finding drove the design of the encoder and of the training described next.

**Table 3.6.** Plain B1-XGB and B1-LR on partial histories (validation, seed 42, EXP-017's diagnostic masks, top-1 and top-3 accuracy with 95% intervals).

| Evidence kept | Model | Top-1 | Top-3 |
|---|---|---|---|
| 100% | B1-XGB | 0.9985 [0.9980, 0.9989] | 1.0000 [1.0000, 1.0000] |
| 100% | B1-LR | 0.9983 [0.9978, 0.9987] | 1.0000 [1.0000, 1.0000] |
| 50% | B1-XGB | 0.5972 [0.5916, 0.6022] | 0.9972 [0.9967, 0.9978] |
| 50% | B1-LR | 0.9755 [0.9740, 0.9772] | 0.9995 [0.9992, 0.9997] |
| 25% | B1-XGB | 0.2671 [0.2625, 0.2718] | 0.9463 [0.9438, 0.9488] |
| 25% | B1-LR | 0.8558 [0.8522, 0.8597] | 0.9818 [0.9804, 0.9832] |

### 3.2.7 Feature engineering: the encoder and the evidence masks

**The encoder.** A model needs numbers, so `EvidenceEncoder` (`src/ml/features.py`) turns age, sex and the evidence tokens into a fixed row. It reads nothing else, so no label can leak in, and its columns come from the release files and never from patients. A binary evidence is a 0 or 1. A categorical question has one column for "answered with something other than the default" and one per non-default answer. An ordinal keeps its value, ordered, plus an "answered" flag, because 0 can be a real answer. A default answer gets no column, so it reads exactly like an unlisted question, and a model records the encoder's fingerprint and refuses rows encoded with another.

The defect behind R-18 led to a second version. The "asked" channel adds one column per question, `E_nn=unasked`, which is 1 when the question was not asked, so the model can tell a denial from silence. At full evidence every question counts as asked, so only a model trained on masked copies learns anything from the channel. We never derive "asked" from the default answers DDXPlus lists, because DDXPlus lists them only for conditions whose definition includes the question, and that would leak the label.

**Table 3.7.** The two versions of the encoder (docs/03 §2.2).

| Version | Columns | Fingerprint | Used by |
|---|---:|---|---|
| Without the "asked" channel | 607 | `3a0d5a5e01d7f427` | The original B1 models and the unprimed `+aug` models |
| With the "asked" channel | 691 | `ddd14019c66eff10` | The primed (′) models, including the configured ranker B1-DL′+aug |

**The evidence masks.** Every system is scored at three evidence levels: 100%, 50% and 25% of each patient's history. Age, sex and the initial evidence (the presenting complaint) are always kept. The other questions are put in a random order drawn from a hash of the case identifier, and the first half or quarter are kept. A kept question keeps all its answers, a dropped question loses all of them, and a follow-up question keeps its parent. A patient's 25% history is part of their 50% history. The masks are drawn once per split, and their digests, short fingerprints of the whole set, are written into the experiment log before any system is scored on them: `e99a7a8fbf792675` and `af3a357f6454838a` for validation, `0136f67f1ca6542f` and `795acf2689f1415e` for the test split. A run that does not reproduce both digests is invalid. These random masks measure robustness to missing answers; they do not measure how well the system takes a history.

Models are also trained on masked copies. The training set holds every training patient plus one masked copy of each, which keeps a share of the questions drawn uniformly between 10% and 90%, from the training seed and never from the evaluation masks. Such models carry the suffix `+aug`, and the suffix ′ marks the use of the "asked" channel.

## 3.3 Related Sections

**Data management and privacy.** Nothing under `data/` or `models/` is ever committed, and neither is the `.env` file that holds the Neo4j credentials; the compliance check of 2026-11-10 found no path under `data/` in the git history. Raw downloads are never edited and every transformation is a script, so the chain from download to result can be repeated. Seeds are fixed (42, and 42 to 46 where the protocol requires five), and each run records its commit and library versions. The project holds no personal data, because every source is synthetic, a knowledge base, or published literature. Free text in a case is untrusted and ignored by every component, and a test shows that a hostile text changes no ranking, flag or explanation.

**Licences.** The project as built is non-commercial in three independent ways: BODHI-S, the embedding model and the 844 corpus articles whose licences allow non-commercial use only. The dashboard, the README and the project documents say so, and Appendix A carries the attribution block.
**Protecting the test split.** The evaluation protocol was frozen on 2026-09-17, before any model was trained, and has changed since only through its amendment log. The single opening of the test split followed these rules.

- The tag `v0.9-freeze` must exist and be an ancestor of the current commit, nothing under `configs/`, `src/` (outside `src/eval/`) or `tests/fixtures/` may have changed since, and the working tree must be clean.
- The opening is recorded, and a second opening needs a stated reason and is disclosed.
- Both mask digests were recorded before anything was scored, and the scoring notebook stops unless both reproduce. A test checks that no other notebook may fetch the test file.
- Development used the validation split only, including the fusion weight, the calibrators and the choice of ranker. A rehearsal of the whole ablation on validation reproduced 1,944 of 1,944 earlier figures exactly before the freeze. Because the fusion weight and the calibrators were chosen on validation patients, fused and calibrated validation figures are optimistic, and the test split checks that nothing was fitted to them.

**Ethics and bias.** Nightingale is an academic research prototype and not a medical device. It has had no clinical validation and no regulatory review, and it is not for use on real patients. Because the demographics are synthetic, the breakdowns by sex and by age (Section 5.6) can show only that the safety result does not depend on them in this generator; they say nothing about fairness for real patients. The test split is a second sample from the same generator as validation, so it does not test the system on real patients.

# Methodology

## 4.1 Introduction to Languages (Front End and Back End)

We wrote the whole system in Python 3.11.9, the team standard (decision D-2) and the version that continuous integration uses. The back end, the front end, the training scripts, the tests and the notebooks all use it. Table 4.1 shows which technology serves which layer.

**Table 4.1.** Languages and technologies by layer.

| Layer | Technology | Where |
|---|---|---|
| Language | Python 3.11.9 | Everything |
| Contracts | Pydantic v2 models | `src/contracts.py` |
| Back end, reasoning | Plain Python modules behind four interfaces | `src/pipeline.py` and the packages under `src/` |
| Back end, service | FastAPI served by uvicorn on 127.0.0.1:8000 | `src/api/app.py`, `scripts/serve_api.py` |
| Front end | Streamlit on 127.0.0.1:8501 | `app/streamlit_app.py`, `app/view.py` |
| Graph database | Neo4j AuraDB Free (Neo4j 5.27) through the Python driver; NetworkX as fallback | `src/medical_kg/` |
| Model runtime | A NumPy forward pass for the transformer; scikit-learn and XGBoost for the baselines | `src/ml/` |
| Files | Parquet through PyArrow, JSON, YAML configuration | `data/`, `models/`, `configs/` |
| Training | PyTorch in Kaggle notebooks | `scripts/train_deep.py`, `notebooks/` |

The reasoning pipeline, `DiagnosisPipeline` in `src/pipeline.py`, is a chain of ordinary Python modules, each behind an interface (a protocol) defined in `src/contracts.py`, so the model, the graph, the retriever and the explainer can each be replaced by a stub without anything downstream noticing. A web service written with FastAPI (a Python framework for web interfaces) and served by uvicorn (the server that runs it) wraps the pipeline and answers requests as JSON. The service and the dashboard listen on 127.0.0.1 only, so they are reachable from the same machine and no other.

The front end is a Streamlit dashboard. Streamlit turns a Python script into a web page: each change to the form re-runs the script, which calls the pipeline and redraws the result. A ranking takes about 10 to 25 ms on the laptop CPU, so the dashboard re-ranks live as each finding is ticked (decision D-14). The whole pipeline's response time is p95 13.6 to 16.9 ms on the golden cases with the corpus loaded, where p95 is the time within which 95% of requests finish (EXP-022 and EXP-011).

The knowledge graph lives in Neo4j, a graph database queried in the language Cypher, on AuraDB Free, a free cloud instance. The store reads the graph once at start-up and scores it in memory, so a paused instance cannot stall a consultation: when Aura is paused or offline, the system falls back to NetworkX over the locally built graph and reports `graph_backend` as degraded. The transformer trains with PyTorch on a cloud GPU, but the laptop runs it as a NumPy forward pass over weights stored as JSON, so inference needs neither PyTorch nor a GPU.

## 4.2 Any Other Supporting Languages and Packages

Table 4.2 lists the packages for data and machine learning, and Table 4.3 those for the service, the interface and engineering. A version without a note is the installed version recorded in the project; "series" means the compatible release that the requirements files pin, where no exact installed version is recorded.

**Table 4.2.** Packages for data, graph and machine learning.

| Package | Version | Role |
|---|---|---|
| NumPy | 1.26 series (1.26.4 drew the evaluation masks) | Arrays; the transformer's forward pass; retrieval |
| pandas | 2.2 series | Tables and the evaluation |
| PyArrow | 17 series | Parquet files |
| scikit-learn | 1.9.1 | Logistic regression |
| XGBoost | 2.1.4 | Gradient-boosted trees (baseline) |
| PyTorch | Training only, on Kaggle; 2.11.0 in the laptop's separate test environment | Training the transformer |
| Pydantic | 2.9 series | Typed contracts |
| PyYAML | 6.0 series | Configuration |
| neo4j driver | 5.28 | Connection to AuraDB |
| NetworkX | 3.6 | In-memory graph; the fallback store |
| sentence-transformers | 3.2 series, on Kaggle only | Embedding the corpus passages |

**Table 4.3.** Packages and services for the service, the interface and engineering.

| Package or service | Version | Role |
|---|---|---|
| FastAPI | 0.143 | The API |
| uvicorn | 0.54 | The server for the API |
| Streamlit | 1.65 | The dashboard |
| httpx | 0.28 series | The API's test client |
| pytest | 8.3 series | Tests: 1,494 passed and 15 skipped on 2026-11-19 |
| black and ruff | 26.5 and 0.16 series | Formatting and linting, pinned in one file shared with CI |
| GitHub Actions | Python 3.11, with a throwaway Neo4j container | Continuous integration |
| Kaggle and Colab notebooks | Kaggle GPU T4 x2; Colab T4 first | Training, the corpus, Synthea, the language-model trial and the test scoring |
| Synthea | 4.0.0 (Java), on Kaggle only | Synthetic patient histories |
| Ollama | 0.34 on the laptop; 0.40.15 on Kaggle for the trial | The language-model trial only (Llama 3.1 8B) |

## 4.3 User Characteristics

Nightingale has one primary user and two secondary groups, and it is explicitly not for two others (Table 4.4). These descriptions come from the requirements specification and the safety document. The repository records no clinician's review and no user study.

**Table 4.4.** User characteristics.

| User | What they know | What they need | Context of use |
|---|---|---|---|
| Clinician (primary): qualified medical staff evaluating an acute chest-pain presentation | Clinical reasoning; little or no machine learning | A ranked differential with red flags first, why each candidate is there, what to ask or test next, and a plain statement of limits | At a workstation, with an incomplete history, wanting an answer that updates as findings are entered (under 5 s required; about 10 to 25 ms measured) |
| Researcher (the team) | The code, the data and the protocol | To run experiments, ablations and the evaluation, and to trace every number to a record | Laptop for short jobs, Kaggle for training and long runs |
| Evaluator and supervisor | The subject at a general level | To judge the work: its claims, its evidence and its honesty | Reading the report and watching the demonstration |
| Not for: patients and the public | — | — | Excluded: the output is not advice to a patient |
| Not for: autonomous use | — | — | Excluded: a qualified clinician must stay in the loop |

The clinician shapes the design choices that show. The specification assumes low technical skill, so the interface has to explain itself. The output is a ranking and never a diagnosis, red flags come first, and the disclaimer appears on every screen. The system is also not intended for triage without clinician review, for treatment or dosing advice, for any presentation other than acute chest pain, or for real clinical care in any setting. Clinicians evaluate chest pain by guidelines such as the 2021 AHA/ACC guideline [@gulati2021]; the prototype does not replace them.

## 4.4 Constraints

Table 4.5 lists the constraints that shaped the project and what each one forced.

**Table 4.5.** Constraints and their consequences.

| Area | Constraint | Consequence |
|---|---|---|
| Compute | The owner's laptop has a CPU for inference and a small GPU for short tests only. Packages are installed per task (D-4), training runs where the owner decides (D-7), and a laptop GPU test needs the owner's yes each time (D-9) | Every training job runs on Kaggle; the laptop runs the model as a NumPy forward pass |
| Data | Synthetic data only; no real patient record under any circumstances; closed world of 14 conditions (R-13); the test split opened once | No clinical claim; every result carries the synthetic and closed-world caveats |
| Licences | BODHI-S, the embedding model and 844 corpus articles are non-commercial | The project stays academic and non-commercial; every source is credited |
| Safety | No treatment, drug or dose advice; the disclaimer on every result; red flags independent of the model; degraded components always shown | A safety layer repairs every result last, and a failed component never crashes the answer |
| Regulation | An academic prototype, not a medical device; no clinical validation or regulatory review | No claim of clinical efficacy anywhere |
| Time | Four people, 8 to 10 weeks; a working prototype and the report were to be presented on 2026-11-15, which left no buffer | Components that failed their checkpoints were cut and reported (the language-model explainer, calibrated probabilities) |
| Infrastructure | Free tiers only: AuraDB Free pauses after 72 hours unused, Kaggle and Colab are shared, and NCBI allows 3 requests a second without a key | A local fallback for the graph; rate-limited corpus fetching; no paid service |

## 4.5 Use Case Model, Flow Chart and DFDs

Figure 4.1 shows the use cases: the clinician has seven and the evaluator two, and Table 4.6 describes them. The clinician has no use case called "diagnose". The system returns a ranked differential, evidence and questions, and the decision stays with the clinician.

![Figure 4.1. Use-case model: the clinician's and the evaluator's use cases.](img/usecase.png)

**Table 4.6.** The use cases of Figure 4.1.

| Use case | Actor | What happens |
|---|---|---|
| Enter a chest-pain case | Clinician | Age, sex, findings marked present, denied or not asked, and risk factors are entered; an invalid case is refused with field errors |
| View the ranked differential | Clinician | The 14 conditions are ranked, red-flagged ones first |
| See red flags | Clinician | Each must-not-miss pattern is raised with the published rule that fired, whatever the rank |
| See why each candidate is there | Clinician | Supporting, contradicting and missing findings, the graph's paths and the model's strongest reasons |
| See what to ask next | Clinician | Cited tests to consider, and the unrecorded findings whose answer would most narrow the differential |
| Read cited evidence | Clinician | Excerpts from open-access articles, each with its citation, link and licence |
| Load an example case | Clinician | One of the four golden cases fills the form |
| Run the ablation study | Evaluator | Every configuration is scored at three evidence levels with intervals |
| Score the test split once | Evaluator | The gate of Section 3.3 opens the split a single time |

Figure 4.2 is the flow chart of one request. A case goes three ways: to the crosswalk, which translates its concepts into DDXPlus's questions for the knowledge graph; to a second translation that chooses one DDXPlus answer per concept for the encoder and the ranker; and to the red-flag rules, which run on their own path so that a rule can raise a condition the model scored near zero. The ranker and the graph each give a distribution over the conditions, and the fusion pools them. The candidates carry the supporting, contradicting and missing findings, and the model's reasons, the questions and tests to ask next, the retrieved evidence and the template explanation hang from them. Guardrails check the explanation, the safety check runs last, and the result goes to the API and the dashboard. The dashed boxes are built but not part of the ranking: a calibrator that was fitted and not deployed, a language-model explainer that failed its checkpoint and is switched off, and the Synthea history, which is displayed and never an input.

![Figure 4.2. The request path of the system, as built (solid: shipped; dashed: built but off, not deployed, or display only).](img/architecture.png)

Figure 4.3 is the data-flow diagram. The clinician is the only external entity. Six processes handle the data: capturing the case, translating concepts, ranking, flagging, explaining and checking, and retrieving evidence. Four stores feed them: the knowledge graph, the model bundle, the retrieval bundle and the Synthea panel. The first three feed processes, but the fourth sends data only to the clinician, drawn beside the case. Its arrow never enters a process, which is the display-only rule drawn as a diagram.

![Figure 4.3. Data-flow diagram: the clinician, six processes and four data stores.](img/dfd.png)

## 4.6 Database Design

The system has no relational database. Its data lives in five stores (Table 4.7), and the knowledge graph is the one that behaves like a database.

**Table 4.7.** The system's data stores.

| Store | Format and place | Integrity check | Read by |
|---|---|---|---|
| Knowledge graph | Neo4j AuraDB, label `CardiacKG`; a local NetworkX copy built from files | A fingerprint over every node and edge | The graph score, the questions, the evidence |
| Chest-pain parquet | One Parquet file per split, in `data/interim/` | Counts checked against the experiment log | Training and evaluation only |
| Model bundles | JSON weights in `models/`; XGBoost as `.ubj` | SHA-256 of every array; the encoder's fingerprint | The ranker |
| Retrieval bundle | Passages, vectors and a manifest in `data/raw/retrieval_corpus/` | SHA-256 of every file in the manifest | The retriever |
| Patient-display file | `patients.json`, 1.7 MB, built from Synthea | None needed: it feeds no ranking | The dashboard's card |

**The knowledge graph** has 129 nodes and 321 edges (Table 4.8). Its nodes are 14 conditions, 66 symptoms and 49 risk factors: the 84 DDXPlus questions, two more that BODHI-S uses, and 29 answer-level concept nodes (18 from the hand-authored facts and 11 from BODHI-S). Every edge records its source, so the report can separate the part of the graph that comes from DDXPlus, the source it shares with the patients, from the independent parts (risk R-12): 76 of the 321 edges do not come from DDXPlus. The graph is a multigraph keyed by source, so a fact that two sources both state is kept twice and counted once.

**Table 4.8.** Size of the knowledge graph (docs/02 §5.1).

| Measure | Count |
|---|---:|
| Condition nodes | 14 |
| Symptom nodes | 66 |
| Risk-factor nodes | 49 |
| Edges from DDXPlus (weight 1.0) | 245 |
| Edges from BODHI-S (weighted by likelihood band) | 56 |
| Edges hand-authored for aortic dissection (weighted by likelihood band) | 20 |

In Neo4j every node carries the namespace label `CardiacKG` and its type label, and a metadata node records the graph's fingerprint. The loader, `scripts/load_neo4j.py`, writes the graph to Aura only when the fingerprint of the local build differs from the stored one, and every other script opens Aura read-only. A hand edit made in Neo4j changes the fingerprint and is caught. Figure 4.4 in Section 4.8 draws the graph as an entity-relationship diagram.

The other stores are plain files. The chest-pain parquet keeps inputs and labels in separate, prefixed columns (Table 4.12). The model bundles are JSON and never pickles, because a pickle can run code when it is loaded. The retrieval bundle holds 26,190 passages with 768-dimensional vectors in float16, and a bundle that fails its checks gives no evidence, so the result reports `rag` as degraded.

## 4.7 Table Structure

This section lists the real structures. They are Python data classes and files, not SQL tables, and we copied the field names from the code.

**Table 4.9.** The graph records (`src/medical_kg/loader.py`).

| Record | Field | Type | Meaning |
|---|---|---|---|
| KGNode | `id` | string | `COND:*` for a condition, `DDX:E_nn` for a DDXPlus question, `SYM:*` or `RF:*` for a hand-authored concept |
| KGNode | `type` | enumeration | Condition, Symptom or RiskFactor |
| KGNode | `label` | string | The label shown to the user |
| KGNode | `properties` | mapping | Extra attributes of the node |
| KGEdge | `condition_id` | string | The condition the edge starts from |
| KGEdge | `concept_id` | string | The symptom or risk-factor node it points to |
| KGEdge | `relation` | enumeration | `HAS_SYMPTOM` or `HAS_RISK_FACTOR` |
| KGEdge | `source` | enumeration | `ddxplus`, `bodhi_s` or `hand_authored` |
| KGEdge | `weight` | float | P(finding given condition) from a likelihood band; 1.0 for DDXPlus |
| KGEdge | `properties` | mapping | The band, the evidence and the reference, for hand-authored edges |

The condition nodes come from a registry in `src/conditions.py`, which holds each condition's `id`, `label`, `category` (cardiac ischaemic, inflammatory, failure or arrhythmia; non-cardiac mimic; vascular), `is_must_not_miss`, `in_training_data` (false for aortic dissection), `ddxplus_label`, `red_flag_hint` and `aliases`.

**Table 4.10.** A crosswalk entry (`src/medical_kg/crosswalk.py`). The crosswalk links the 67 hand-authored concepts to DDXPlus answers.

| Record | Field | Type | Meaning |
|---|---|---|---|
| CrosswalkEntry | `concept_id` | string | The hand-authored concept, for example `SYM:chest_pain` |
| CrosswalkEntry | `label` | string | Its label |
| CrosswalkEntry | `match` | enumeration | exact, close, broader, narrower, related or none |
| CrosswalkEntry | `pattern` | Pattern or none | None exactly when the match is none |
| CrosswalkEntry | `note` | string | The reason, with the DDXPlus French |
| Pattern | `code` | string | The DDXPlus question, for example `E_55` |
| Pattern | `values` | set of strings | The answers that count; never the "no" default or NA |
| Pattern | `min_ordinal` | integer or none | For 0 to 10 scales, the lowest answer that counts |
| Pattern | `side` | enumeration or none | For paired body locations, one side or both |

The match type is a relation from the Simple Knowledge Organization System (SKOS) and says how the DDXPlus answer relates to the concept: it is the same as it (exact), nearly the same (close), covers more (broader), covers less (narrower), or is only linked to it (related). Each type allows only the inferences that are sound (Table 4.11). A denial carries over only to a yes/no question: denying radiation to the jaw or arm says nothing about radiation to the back.

**Table 4.11.** What each match type allows (docs/02 §5.2).

| Match | Concept present implies answer | Answer implies concept | Concept denied implies question denied |
|---|---|---|---|
| exact, close | yes | yes | yes, for a yes/no question |
| broader | yes | no | no |
| narrower | no | yes | yes, for a yes/no question |
| related, none | no | no | no |

**Table 4.12.** The chest-pain parquet, one row per in-scope patient (docs/03 §2.1).

| Column | Role | Type | Meaning |
|---|---|---|---|
| `case_id` | metadata | string | `ddxplus-<split>-<row>`, traceable to the raw file |
| `split` | metadata | string | train, validate or test |
| `source_row` | metadata | integer | Row number in the raw CSV |
| `initial_evidence` | metadata | string | The presenting complaint; defines the masks and is never an input |
| `age` | input | integer | Years, 0 to 109 |
| `sex` | input | string | M or F; synthetic, about 50% female |
| `evidences` | input | list of strings | The raw tokens, nothing dropped |
| `positive_codes` | input | list of strings | Questions answered with something other than "no" |
| `label_condition_id` | label | string | The true condition's identifier |
| `label_pathology` | label | string | The raw DDXPlus pathology |
| `label_differential` | label | list of structures | The differential with probabilities; out-of-scope entries kept |

**Table 4.13.** The contract types (`src/contracts.py`). They change only with the agreement of all four team members.

| Type | Fields | Role |
|---|---|---|
| Finding | `concept_id`: string; `label`: string; `assertion`: present, absent or unknown; `qualifiers`: mapping of strings; `source`: string; `evidence_span`: string or none | One clinical fact; absent and unknown are different facts |
| PatientCase | `case_id`: string; `age`: integer 0 to 120; `sex`: M, F or O; `findings`: list of Finding; `risk_factors`: list of Finding; `vitals`: mapping to numbers; `free_text`: string or none, ignored by design | The request |
| ReasoningPath | `finding_id`: string; `condition_id`: string; `path`: list of strings; `weight`: float | A graph path that explains a finding |
| FindingAssessment | `finding_id`: string; `label`: string; `role`: supporting, contradicting or missing; `rationale`: string or none | How one finding bears on one candidate |
| Evidence | `source`: string; `title`: string; `passage`: string; `citation`: string; `url`: string or none; `score`: float | A retrieved passage |
| Candidate | `condition_id`: string; `label`: string; `ml_score`, `kg_score`, `fused_score`: float; `calibrated_probability`: float or none; `is_must_not_miss`, `red_flag`: boolean; `red_flag_reason`: string or none; `assessments`: list of FindingAssessment; `paths`: list of ReasoningPath; `evidence`: list of Evidence | One condition in the differential |
| Explanation | `text`: string; `grounded`: boolean; `unsupported_claims`: list of strings; `citations`: list of strings | The explanation and the result of the claim check |
| DiagnosisResult | `case_id`: string; `candidates`: list of Candidate; `red_flags`: list of strings; `suggested_next_tests`: list of strings; `explanation`: Explanation or none; `disclaimer`: string; `degraded_components`: list of strings; `generated_at`: date and time | The response |

**Table 4.14.** The likelihood bands and their edge weights (`src/medical_kg/loader.py`, assumption A-6).

| Band | P(finding given condition) | Weight |
|---|---|---:|
| zero | none | 0.0 |
| rare | under 5% | 0.03 |
| low | 5% to 19% | 0.12 |
| medium | 20% to 49% | 0.35 |
| high | 50% to 79% | 0.65 |
| very high | 80% or more | 0.9 |

The weight is the middle of the band. BODHI-S publishes no numeric bands, so these weights are an assumption. DDXPlus edges keep weight 1.0, because DDXPlus lists a finding without saying how often it occurs, and the graph score caps that weight at 0.9.

## 4.8 ER Diagrams

Figure 4.4 shows the knowledge graph and its crosswalk. A `CONDITION` has zero or more `KG_EDGE` records and each edge belongs to exactly one condition: a one-to-many relation. A `CONCEPT_NODE`, a symptom or risk-factor node, is linked by zero or more edges, and each edge points to exactly one concept: also one-to-many. Together the two relations make the graph a many-to-many link between conditions and concepts, held in the edge table. An edge is unique for its triple of condition, concept and source, which is why two sources that state the same fact leave two edges. A `CONCEPT_NODE` maps through zero or more `CROSSWALK_ENTRY` records; in the shipped table each hand-authored concept has one entry, whose pattern names one DDXPlus question and the answers that count. The edge's `properties` hold its likelihood band and its references, and the node's `properties` hold whatever the source adds.

![Figure 4.4. Entity-relationship diagram of the knowledge graph and the crosswalk.](img/erd_kg.png)

Figure 4.5 shows the request and result types. One `PATIENT_CASE` contains zero or more `FINDING` records, which hold both its findings and its risk factors, and it is answered by exactly one `DIAGNOSIS_RESULT`. A result ranks one or more `CANDIDATE` records, in practice 14, and carries at most one `EXPLANATION`. Each candidate is explained by zero or more `FINDING_ASSESSMENT` records and cites zero or more `EVIDENCE` records. The `Evidence` type carries a source, a title, a passage, a citation, a link and a score but no licence field (Table 4.13); a change to the contracts needs all four team members, so the dashboard reads each excerpt's licence from the retriever's record of the cited article instead.

![Figure 4.5. Entity-relationship diagram of the request and result types (the contracts).](img/erd_contracts.png)

## 4.9 Assumptions and Dependencies

The system rests on a few judgement calls that the data could not settle. Table 4.15 lists them with the setting in use and the record that holds each. The team accepted or the owner decided each one, and none was tuned on the test split.

**Table 4.15.** Assumptions, their settings and records.

| Assumption | Setting | Record |
|---|---|---|
| A-5: what counts as sudden onset on DDXPlus's 0 to 10 onset-speed scale | 8 or more. A lower cut-off would reach more pneumothorax patients only because of how DDXPlus draws onset values | docs/02 §9; EXP-008 |
| A-6: the weight of each likelihood band | The middle of the band (Table 4.14) | docs/02 §9 |
| A-8: a hand-written case may use narrower crosswalk matches, and one DDXPlus answer stands for each of 17 concepts | Admitted; the owner approved the 17 answers on 2026-10-12, and who reviewed them is not recorded | docs/02 §9; risk R-17 |
| A-10: the answer for radiation to the jaw or arm | Changed on 2026-10-13 from the jaw, which only acute pulmonary edema patients give in DDXPlus, to under the jaw, which infarction and angina patients give | docs/02 §9; EXP-021 |
| A-9: how a mask treats the questions a patient did not list | They count as asked at the same share as the listed ones, so "asked" does not leak the label | docs/02 §9; EXP-019 |
| The fusion's floor on the model's probabilities | 0.01, fixed in advance and never tuned (decision A-2) | docs/02 §6 and §9 |
| The graph score's constants | A leak of 0.01 and a cap of 0.9, fixed before any measurement | EXP-005 |
| The patients are synthetic | No claim about real patients; parity by sex or age is an artefact | EXP-002; docs/04 §6 |

The hand-written cases reach the model through an inversion that has no ground truth (risk R-17): the tests round-trip each concept, but nothing shows that a real patient with that concept gives the chosen answer. The models are also near-certain on hand-written cases where a history cannot decide, so no raw score is shown as a probability.

The system depends on outside resources (Table 4.16), and each has a fallback or a stated consequence.

**Table 4.16.** Dependencies and what happens without them.

| Dependency | What depends on it | If it is missing |
|---|---|---|
| DDXPlus, Hugging Face revision `2ad986ac…` | Training, evaluation and the graph backbone | The tests that need data skip; the pinned revision keeps the files identical |
| BODHI-S | Enrichment of four conditions | The graph is built without it and is not the graph that was evaluated |
| Kaggle | Every training and long job; the corpus; Synthea; the test scoring | Jobs wait for the owner's choice of another runtime (D-7) |
| Neo4j AuraDB Free | The shared copy of the graph | Fallback to NetworkX, reported as `graph_backend` |
| PubMed Central and NCBI E-utilities | The evidence corpus, fetched once | The pinned bundle is used; without it the result reports `rag` |
| The model bundle, with its SHA-256 and encoder fingerprint | The configured ranker | A graph-only ranking, reported as `ml` |
| Ollama and Llama 3.1 8B | The language-model trial only | Nothing: the explainer is switched off |
| GitHub Actions | Continuous integration | Tests run locally; CI has no data or models, so data-gated tests skip there |

## 4.10 Algorithm Discussion

This section explains each algorithm in plain words and then precisely. Sections 4.10.1 and 4.10.2 cover the graph score and the translation of a case, 4.10.3 and 4.10.4 the transformer and its training, 4.10.5 the fusion, 4.10.6 the red flags and the safety layer, 4.10.7 calibration, 4.10.8 retrieval and 4.10.9 to 4.10.11 the explanations.

### 4.10.1 The graph score

In plain words, the graph asks of each condition: if this patient had this condition, how likely would these findings be? It adds up the log-probabilities, which is the naive-Bayes method, and treats the findings as independent once the condition is known. A finding the condition explains counts for it, a finding it cannot explain counts against it, a denied finding counts against the conditions that expect it, and a finding the case does not mention counts for nothing, because not mentioned is not denied. With a uniform prior over the 14 conditions, the score of condition c is:

```
score(c) = sum over present findings k of log P(k | c)
         + sum over denied findings k of log(1 - P(k | c))
```

P(k | c) is the weight of the edge from c to k. A condition with no edge to a present finding gets a leak of 0.01, and DDXPlus's weight of 1.0 ("listed, frequency unknown") is capped at 0.9; we fixed both constants before measuring anything. The fusion reads the scores as a posterior, a softmax over the conditions.

The graph is first closed under the crosswalk. DDXPlus links conditions to questions while the hand-authored edges point to answers, and without a closure every patient's chest pain would count against the nine conditions that DDXPlus describes only by questions and hand the case to aortic dissection, the one condition that names the answer. Two rules close it. Upward, an answer implies its question: a condition with answer-level edges of weights w1, w2 and so on, implying a question, lists that question with probability 1 − (1 − w1)(1 − w2)…, the chance of giving at least one of those answers. Downward, a condition that asks a question whose answer the graph does not state gets that answer at the mean of the conditions that do state it, so ignorance counts as average. Each fact is counted once, and the score splits exactly into each finding's contribution, which is what the reasoning paths cite.

Two limitations are built in. An answer that only one condition states, such as tearing pain for dissection, cannot discriminate between the others. And stable and unstable angina tie when nothing is denied, because stable angina's 20 questions all lie among unstable angina's 24; the tie rule puts the must-not-miss one first, and separating them is the ranker's job. We measured Personalised PageRank against naive Bayes and chose naive Bayes: the two rank validation identically, but naive Bayes holds the angina decision by a factor of about 90 where PageRank holds it by 8% (EXP-005).

### 4.10.2 The encoder and the case-to-token step

The encoder of Section 3.2.7 turns a patient into 607 or 691 numbers. A hand-written case, however, is written in concepts such as `SYM:chest_pain`, and a model needs DDXPlus tokens such as `E_55_@_V_101`. A separate module, `src/ml/case_tokens.py`, chooses the answer that a patient with that concept would have given, from two hand-curated tables of 17 entries that quote the DDXPlus French. Taking the lowest answer identifier would be wrong: it gives forearm for radiation to the jaw or arm, which no validation patient gives. Narrower matches are admitted (assumption A-8), because without them sudden onset, exertional pain and relief by rest produce nothing, and those findings separate embolism, pneumothorax, dissection and the anginas. A denial is never a token. It marks its question as asked only when it answers a whole yes/no question, and only a model with the "asked" channel can see that.

### 4.10.3 The transformer

A transformer is a neural network in which every input token can look at every other through attention. Ours is a small FT-Transformer [@gorishniy2021], a design for tables, and the label B1-DL′+aug names it: the deep-learning form of the machine-learning baseline, trained on masked copies and given the "asked" channel. There is one token for each of the 84 DDXPlus questions, plus one for age, one for sex and one classification token, which makes 87 tokens for every patient. A question's token is a learned bias plus the sum of its columns' scaled values times their embeddings, so an unasked question adds its `unasked` embedding. Table 4.17 gives the settings, which we fixed before training. The feed-forward layers use a gated activation, ReGLU [@shazeer2020], and the optimiser is AdamW [@loshchilov2019].

**Table 4.17.** The configured transformer, B1-DL′+aug (`src/ml/deep.py`, `scripts/train_deep.py`).

| Setting | Value |
|---|---|
| Tokens | 84 questions, age, sex and a classification token: 87 |
| Width, blocks, heads | 96; 2 blocks; 4 heads of 24 |
| Feed-forward | ReGLU of width 128; pre-normalisation |
| Dropout | 0.2 on attention; 0.1 in the feed-forward layers |
| Size | 225,645 parameters (217,581 without the "asked" channel); 9,610,752 multiply-adds per patient |
| Optimiser | AdamW, learning rate 1e-4, weight decay 1e-5 |
| Batches | 1,024 in 32-bit floats; at most 50 epochs; patience 5; best epoch kept |
| Training rows | 511,800: 460,620 fitted and 51,180 held out by patient for early stopping |
| Seeds | 42 to 46 for both arms; the configured model is seed 42 |

The model has two implementations. PyTorch trains it on a Kaggle GPU, and the laptop, the pipeline and the tests run a NumPy forward pass over weights stored as JSON, each array base64-encoded in float32 with its SHA-256. On 2,000 validation rows the two implementations differ by at most 4.3 × 10⁻⁷ in probability with no top-1 disagreement, and the laptop reproduces seed 42's validation metrics to within 7 × 10⁻¹¹ (EXP-021). One patient takes p95 1.38 ms at batch size 1 (EXP-022).

### 4.10.4 Training on masked copies, and the safety floor

The comparison models were trained on complete histories and fail on short ones (Section 3.2.6), so the `+aug` models, the transformer among them, train on masked copies: each training patient appears once in full and once with a share of its questions kept. A model with the "asked" channel learns from the copies which questions were never asked. The transformer trains on 90% of the rows and the logistic regression on all of them, an asymmetry that can only disadvantage the transformer, and no model is tuned beyond early stopping (decision T-6).

Decision D-13, fixed before training as code in `src/eval/safety_floor.py`, put safety first. The transformer could replace logistic regression as the ranker only if, on validation, it met the safety targets at 100% and 25% of the history (must-not-miss recall@3 of at least 0.95 and a dangerous false-negative rate of at most 0.02), was non-inferior to the logistic regression trained like it at 25% on must-not-miss recall@3 and on mean reciprocal rank (MRR, the average of one over the rank of the true condition), with a paired 95% lower bound above −0.005, and passed every golden case with red flags on and off. It passed on 2026-10-13. At 25% of the history on validation, B1-DL′+aug has an MRR of 0.9728 [0.9715, 0.9740] and a must-not-miss recall@3 of 0.9981 [0.9973, 0.9987], against 0.9710 [0.9697, 0.9722] and 0.9975 [0.9967, 0.9982] for B1-LR′+aug, and the paired difference in MRR is +0.0018 [+0.0009, +0.0026] (EXP-021). The rule held at all five seeds for both arms, so B1-DL′+aug at seed 42 is the configured ranker and B1-LR′+aug is the backup, switched back by two keys in the configuration file.

### 4.10.5 The logarithmic opinion pool

The fusion combines the model's opinion and the graph's by a logarithmic opinion pool [@genest1986]: a weighted geometric mean of the two probability distributions, renormalised. It belongs to the family of products of experts [@hinton2002], in which each expert can veto what the other supports. Over the 13 conditions the model knows,

```
log q(c) = (1 - α) · log max(p_model(c), ε) + α · log P_graph(c),  normalised over the 13
```

where `P_graph` is the graph's posterior, α = 0.05 is the graph's weight and ε = 0.01 is a floor. Three choices were fixed on 2026-09-25, before any fused ranking of a validation patient was scored. First, a condition the model cannot score takes its probability from the graph alone: aortic dissection gets its graph posterior, and the 13 share the remainder in the pool's proportions. The first design, a weighted sum of rescaled scores, capped dissection at the graph's weight, so it could never rank first. Second, no model probability below 0.01 is taken at face value, so that one near-certain model cannot veto the graph. Third, an absent signal abstains: a degraded model leaves the graph's order exactly, a failed graph leaves the model's, and a model whose scores are not finite non-negative probabilities counts as failed.

The weight α is the one tuned quantity. A pre-registered sweep on validation chose 0.05 for the configured component B1-DL′+aug and 0.30 for B1-DL+aug (EXP-006), and the weight is never retuned by hand. The fusion adds nothing measurable on DDXPlus. At 25% of the history, the fused ranking of B1-DL′+aug and the graph at α = 0.05 has an MRR of 0.9727 [0.9715, 0.9739] against 0.9728 [0.9715, 0.9740] for the model alone (validation, EXP-006 and EXP-021), and a must-not-miss recall@3 of 0.9978 [0.9971, 0.9985] against 0.9981 [0.9973, 0.9987]. Read this with risks R-12 and R-16: DDXPlus draws each patient's evidence from their condition's list, so the model already saturates, and the graph's contribution is the one condition the model cannot score, together with the explanation.

### 4.10.6 Red-flag rules and the safety layer

The red-flag rules run on every case, independently of the ranker, so a rule can raise a condition the model scored near zero. Each rule follows a published pattern and not a single suggestive finding, because a flag on half of all patients teaches users to ignore flags [@ancker2017], and every must-not-miss condition has a rule (Table 4.18). The dissection rule follows the ADD-RS [@rogers2011; @hiratzka2010] and the unstable-angina rule follows Braunwald's classification [@braunwald1989]. The pulmonary-embolism rule follows the Wells criteria [[citation needed: Wells criteria for pulmonary embolism]]. The pipeline lists every flagged candidate before every unflagged one, whatever its score.

**Table 4.18.** The red-flag rules (docs/04 §3).

| Condition | Pattern |
|---|---|
| Possible NSTEMI / STEMI | Chest pain with two of: exertional, pressure character, radiation to the jaw or arm, sweating |
| Unstable angina | A crescendo pattern, or chest pain at rest with an ischaemic character |
| Pulmonary embolism | Pleuritic pain or breathlessness and a Wells risk: one swollen leg or calf pain, immobilisation, recent surgery or previous deep-vein thrombosis |
| Spontaneous pneumothorax | Sudden pleuritic pain with breathlessness (sudden: onset speed 8 or more of 10, A-5) |
| Myocarditis | Chest pain after a viral illness, with breathlessness or palpitations |
| Acute pulmonary edema | Breathlessness with orthopnoea, nocturnal breathlessness or known heart failure |
| Boerhaave syndrome | Chest pain after forceful vomiting |
| Aortic dissection | Pain with tearing character, a pulse deficit or an inter-arm blood-pressure difference, or findings from two of the ADD-RS's three categories; back radiation alone is in none of them |

Red-flag sensitivity is the share of a must-not-miss condition's own patients whose flag reaches them. It is 0.823 [0.818, 0.829] at 100% evidence on validation, below the 0.95 target, because DDXPlus often omits the defining finding, and we do not loosen a rule to chase it (EXP-008). Judged by a three-judge panel's majority over all 104 flag and condition pairs (decision A-7), red-flag precision is 0.848 [0.845, 0.852] at 100% on validation. The repository records no clinician's review of the rules, and DDXPlus generated its patients from the definitions the crosswalk reads (risk R-12), so none of these rates is evidence about real histories.

The safety layer runs last and repairs rather than fails. It makes sure the disclaimer is exactly the required text, that every flagged candidate comes before every unflagged one and is named in the red flags, and that no sentence recommends a treatment, names a drug or gives a dose. Such a sentence is removed and listed among the unsupported claims, while investigations such as an ECG are kept. The layer is deterministic and lexical, and on 3,000 validation patients it found nothing to repair.

### 4.10.7 Calibration

A model's raw score is not necessarily the probability that it is right. Calibration adjusts the scores so that a score of 0.8 is right about 80% of the time. We implemented four methods: none, temperature scaling [@guo2017], which divides the model's logits by one learned number, and one-versus-rest Platt scaling and isotonic regression [@zadrozny2002]. A condition with no patient in the fitting data keeps its raw probability, and calibration never changes the ranking. The measure is the expected calibration error (ECE), the average gap between confidence and accuracy.

Stage 1 of EXP-007 found the logistic regressions and both transformer arms already calibrated, so they stay raw. Stage 2 fitted the fused score of A0′, the full system built on the primed transformer, and the rule chose temperature scaling. Its top-label ECE went from 0.0327 [0.0324, 0.0329], 0.0603 [0.0595, 0.0614] and 0.0773 [0.0757, 0.0793] raw to 0.0008 [0.0008, 0.0008], 0.0051 [0.0041, 0.0062] and 0.0176 [0.0156, 0.0195] at 100%, 50% and 25% evidence (A0′, validation, EXP-007). The owner decided on 2026-11-09 that no calibrator is deployed, because a short hand-written case is near-certain under any calibration (risk R-17). The system therefore shows ranks and never a probability, and the calibrator is a result, not part of the product.

### 4.10.8 Evidence retrieval

Retrieval-augmented generation [@lewis2020] gives a language model passages to quote. Nightingale uses the retrieval half only: it shows passages beside a candidate, and evidence never changes the ranking, which a test checks. The retriever is a hybrid of two searches. BM25 is a classic word-matching score, with the standard settings 1.2 for term saturation and 0.75 for length normalisation, fixed and not tuned. The dense search compares vectors: the passage vectors (768 dimensions) were computed once on Kaggle with the S-PubMedBert-MS-MARCO model, and on the laptop a query vector is a mix of vectors precomputed for a fixed vocabulary of queries, so the laptop runs no embedding model. The two rankings are merged by reciprocal rank fusion, in which each passage scores the sum, over the rankings that contain it, of one over (60 + its rank). Everything runs in NumPy, and an exact product over tens of thousands of passages takes milliseconds, so there is no FAISS index.

Before an excerpt is shown, the system screens it again for treatment language, and the safety layer drops any evidence that recommends a treatment. Retrieving one candidate takes p95 1.22 ms. Two limits are stated on the dashboard and in this report. The passages' relevance was never judged. And the pre-registered check of the query-vector mix found that its dense top 10 overlaps the full-sentence embedding's top 10 by a median of only 0.400, below the 0.5 threshold we set, so the laptop's query vector is a poor stand-in for embedding the whole query.

### 4.10.9 The model's reasons

The graph explains itself through its paths, but a neural network does not. We ask the model directly, by leave-one-finding-out occlusion (`src/ml/reasons.py`): remove each recorded finding in turn, rescore, and measure how each condition's log-odds move. A finding whose removal lowers a condition's odds is a reason for that condition, and a denial is a finding too. The case and each case-without-one-finding are scored in one batch, which takes 0.4 to 6.5 ms. The explanation names the reasons, strongest first, and shows no number. The limits are stated with every use: each finding is judged alone, so two findings that matter only together can each look weak; a finding that cannot be turned into a token is invisible to the model; and aortic dissection, which the model never saw, has no model reasons. The reasons found the A-10 problem: on GC-001 the transformer alone answered acute pulmonary edema, because the jaw answer the case used is one that only pulmonary-edema patients give in DDXPlus.

### 4.10.10 What to ask next

The question "what should the clinician ask or test next?" has two parts. The tests to consider come from a hand-authored table with one cited line for each of the 14 conditions, in the order its guideline uses them; the pipeline shows them for the first three candidates and every red-flagged one. Every line says "consider" and lists investigations only, never a treatment, and a test scans the table for treatment language. Two of its references were checked on the publishers' pages [@roberts2023; @chirica2019].

The questions to ask come from expected information gain. Of the findings that the case has neither recorded nor denied, the system ranks those whose answer would most reduce the graph's uncertainty over the 14 conditions. Precisely, for a question q with answers a, the gain is:

```
gain(q) = H(posterior) - sum over answers a of P(a) · H(posterior given a)
```

H is entropy, a measure of uncertainty, and the gain is the mutual information between the condition and the answer: never negative, and 0 for a question every condition answers alike. The probabilities come from the same naive-Bayes likelihoods that rank the case, through the crosswalk. The five best questions are returned, each with the conditions a yes and a no would favour. A finding the record already settles is not asked, nor is one that cannot apply to the recorded sex. The ranking is myopic, one question at a time, and it uses the graph alone, so the order matters and the number does not. On GC-001 the first questions are rest pain and a crescendo pattern, which bear on unstable angina against infarction.

### 4.10.11 Explanations, guardrails and the language-model trial

The template explainer writes the explanation from the graph and the model, so it cannot state a fact the system does not hold. For each candidate it keeps every supporting question the graph matched, with the finding that answered it; it names five of the expected findings that were not recorded and counts the rest; and it names the top candidate's model reasons. It is the explainer that ships.

Guardrails v1 check the explanation after it is written, whoever wrote it. Each sentence must trace to something the result holds: a candidate, the red flags, the model's reasons, a suggested test or a cited passage. A sentence that traces to none is listed in `unsupported_claims`, and `grounded` becomes false. An explanation may cite only the retrieved passages, and a link goes only to PubMed Central. The check is lexical: it catches a word that the result does not hold and not a wrong relation between words that it does.

We also built a grounded language-model explainer, which asks a local model (`llama3.1:8b` through Ollama) to phrase the explanation from a prompt that holds only what the result holds. Decision D-5 fixed its checkpoint in advance: the template stays unless the unsupported-claim rate is under 0.20. In the trial (EXP-009 and EXP-010: 50 validation patients and the four golden cases, run once on Kaggle) the grounded explainer's raw answers had an unsupported-claim rate of 0.741 [0.663, 0.803]. The checkpoint failed, so the template explainer stays and `llm.enabled` is false. Guardrails v1 stopped all 50 answers, so none reached a user. The claim check is lexical and punishes paraphrase, so the rate measures how far the answers stay within the inputs and not whether they are correct, and one model, one run and a sample of 50 limit the result further.

## 4.11 Implementation of Algorithm with Screen Shots and Figures

This section shows the implementation in use. The dashboard runs the pipeline of Figure 4.2 live, the console demo prints the same result as text, and the API serves it as JSON. Nothing here is clinical advice: the system is a research prototype built on synthetic patients and knows 14 causes of acute chest pain.

### 4.11.1 The dashboard

The dashboard (`app/streamlit_app.py`) starts with `streamlit run` and serves 127.0.0.1:8501. Each finding is marked present, denied or not asked, the four golden cases load as examples, and the result updates on every change to the form. The three figures below show golden case GC-001, the acute coronary syndrome case of Table 3.3.

Figure 4.6 shows the ranking view. The ischaemic red flag fires, and possible NSTEMI / STEMI is listed first. The fused order shows ranks and no raw scores, with the model's and the graph's own ranks beside it, and the header shows the time taken, about 10 to 25 ms. The model's own first answer for this case is unstable angina, which is listed second only because the pipeline lists flagged candidates first. A "Running without" line names any component that is degraded, and the disclaimer is on every screen.

![Figure 4.6. Dashboard, ranking view, golden case GC-001: red flag first, the fused order with the model's and the graph's ranks, the response time and the disclaimer.](img/dashboard_ranking.png)

Figure 4.7 shows why each candidate is there. A candidate's row opens to the findings that support it (✓), contradict it (✗) or are still missing (?), the graph's reasoning paths, and the model's strongest reasons for and against. For GC-001 the model's reasons against infarction include exertional onset and pressure-type pain, because in DDXPlus those findings belong to the anginas. Below it, "What to ask next" lists rest pain and a crescendo pattern, and the tests to consider: a 12-lead ECG and serial high-sensitivity troponin, each with its guideline.

![Figure 4.7. Dashboard, the explanation view for golden case GC-001: supporting, contradicting and missing findings, the model's reasons, the questions to ask next and the tests to consider.](img/dashboard_why.png)

Figure 4.8 shows the evidence panel. Each of the first three candidates has excerpts from open-access articles with a citation, a link to the PubMed Central page, the article's own licence and an attribution. A caption under the excerpts says that their relevance has not been formally judged and asks the reader to take them as leads, not proof. Elsewhere on the page, an expander offers a card with a synthetic patient's history, which never changes the ranking, and the page ends with a section that names every data source and licence.

![Figure 4.8. Dashboard, the evidence panel for golden case GC-001: excerpts with citation, link, licence and the caption that their relevance is unjudged.](img/dashboard_evidence.png)

### 4.11.2 The console demo

`scripts/demo.py` runs a golden case through the same pipeline and prints the result. Listing 4.1 is the real output for GC-003, aortic dissection, trimmed to the setup lines, the red flag, the differential and the disclaimer; the warning-sign glyph is replaced by an exclamation mark, and long lines are wrapped or shortened with [...]. On the day of this run the AuraDB instance was unavailable (it may be paused), so the local graph answered and the demo says so under "Degraded components". The model has never seen a dissection patient, and its column shows a dash for it, yet dissection is ranked first, with a red flag, from the graph and the rule.

**Listing 4.1.** The console demo's output for golden case GC-003 (`python scripts/demo.py --case GC-003`, laptop CPU, a few seconds). The score is a log-probability, not a calibrated probability; "ml" is the model's raw score, "kg" the graph's log-likelihood.

```
ranker: the transformer B1-DL′ (deep) over 13 conditions
Graph: networkx, 129 nodes, 321 edges (bodhi_s 56, ddxplus 245, hand_authored 20); [...]

  Case: GC-003
  RED FLAGS
    * Aortic dissection: Pain with a highly specific sign of aortic dissection
      (tearing character, pulse deficit or inter-arm blood-pressure difference)
      [...] absent from the training data and is detected by rule only.
  DIFFERENTIAL (ranked)
   1. ! Aortic dissection                  score -0.390  (ml   —  / kg -4.57)
   2.   Spontaneous pneumothorax           score -1.895  (ml 0.56 / kg -11.34)
   3.   Acute pulmonary edema              score -2.068  (ml 0.37 / kg -6.84)
   4.   Boerhaave syndrome                 score -4.756  (ml 0.03 / kg -11.34)
   5.   Unstable angina                    score -4.938  (ml 0.02 / kg -6.84)
  Degraded components: graph_backend

    Clinical decision support — not a diagnosis. Nightingale is a
    research prototype. Its output is generated from synthetic training
    data and must not be used for real patient care. Rankings are not
    certainties and confidence values are estimates. A qualified
    clinician is responsible for all diagnostic and treatment decisions.
```

### 4.11.3 The API

`scripts/serve_api.py` serves the pipeline with uvicorn on 127.0.0.1:8000, and FastAPI documents it at `/docs`. Table 4.19 lists the endpoints. A malformed case is refused with HTTP 422 and its field errors. Every result carries the disclaimer, and the response header `X-Elapsed-Ms` gives the time taken.

**Table 4.19.** The API's endpoints (`src/api/app.py`).

| Method and path | Input | Output |
|---|---|---|
| POST `/diagnose` | A patient case (Table 4.13) | The full diagnosis result as JSON, with `X-Elapsed-Ms` in the header |
| POST `/next-questions` | A patient case | The five findings whose answer would most narrow the differential |
| GET `/conditions` | None | The 14 conditions the system can rank |
| GET `/concepts` | None | The findings the system understands, for building a form |
| GET `/health` | None | Which ranker and graph answer, what is degraded, and the disclaimer |

```abbreviations
A0 = The full system: the ranker fused with the graph, plus the red-flag layer (A0′ uses the primed transformer)
AHA/ACC = American Heart Association / American College of Cardiology
ADD-RS = Aortic Dissection Detection Risk Score
API = Application Programming Interface
B2 = The baseline that uses the knowledge graph alone
B1 = The baseline that uses machine learning alone (B1-XGB, B1-LR, B1-DL are its XGBoost, logistic-regression and deep-learning forms)
BM25 = Best Matching 25 (a word-matching ranking function)
CC = Creative Commons
CI = Continuous Integration
CPU = Central Processing Unit
CSV = Comma-Separated Values
DFD = Data-Flow Diagram
ECE = Expected Calibration Error
ECG = Electrocardiogram
ER = Entity-Relationship
EXP = Experiment (an entry in the experiment log)
FT-Transformer = Feature-Tokenizer Transformer
GERD = Gastro-Oesophageal Reflux Disease
GPU = Graphics Processing Unit
HTTP = Hypertext Transfer Protocol
ICD-10 = International Classification of Diseases, 10th revision
IRAD = International Registry of Acute Aortic Dissection
JSON = JavaScript Object Notation
KG = Knowledge Graph
LR = Logistic Regression
MRR = Mean Reciprocal Rank
NA = Not Applicable (the sentinel answer)
NCBI = National Center for Biotechnology Information
NSTEMI = Non-ST-Elevation Myocardial Infarction
PSVT = Paroxysmal Supraventricular Tachycardia
ReGLU = Rectified Gated Linear Unit
SHA-256 = Secure Hash Algorithm, 256-bit
SKOS = Simple Knowledge Organization System
SNOMED CT = Systematized Nomenclature of Medicine, Clinical Terms
SQL = Structured Query Language
STEMI = ST-Elevation Myocardial Infarction
XGBoost = eXtreme Gradient Boosting
XML = Extensible Markup Language
```
