# Prompt: generate the mid-term evaluation slides

**How to use it:**
1. Do `midterm-prep.md` §9 first: fill in the names, check every reference, and delete any that
   could not be confirmed from the reference slides below.
2. Paste everything from the line **"PROMPT STARTS HERE"** to the end into a slide generator:
   Claude, Gamma, Microsoft Copilot in PowerPoint, or Google Slides with Gemini.
3. Afterwards, check every number on the slides against `docs/08-experiment-log.md`.

The prompt is self-contained: the generator needs no other file.

---

**PROMPT STARTS HERE**

You are preparing a **16:9 slide deck** for a university mid-term project evaluation (course PRJ1).
The project is **Nightingale — Clinical Intelligence & Reasoning System**: a four-person team's
research prototype that ranks the possible causes of **acute chest pain** and explains the ranking.
The audience is faculty examiners in computer science and engineering. The talk lasts about **15
minutes plus questions**, and is shared by four presenters (P1–P4).

### The marking scheme, which decides the deck's weight
- **Literature review:**
  - comprehensiveness: **10 marks**;
  - research gap: **5 marks**.
- **Methodology:**
  - objective / problem definition: **5 marks**;
  - proposed methodology (tools, techniques, methods, dataset): **5 marks**.

Give the literature review the most slides. Progress and results are not marked: show them briefly
(two slides), only as proof that the methodology works.

### Presenters by slide
Put the presenter's label in each slide's speaker notes:
- **P1:** slides 4, 7, 14;
- **P2:** slides 6, 13, 15, 19;
- **P3:** slides 5, 8, 10, 16;
- **P4:** slides 1–3, 9, 11, 12, 17, 18, 20, 21.

### Rules you must follow
1. **Use only the facts, numbers and references given below.** Do not add statistics, citations,
   authors, years or claims of your own. If a slide needs something that is not given, write
   `[TO ADD]` instead.
2. **Copy every number exactly, with its interval** in square brackets and its label (e.g.
   "validation, 25% of the history"). Never round away an interval.
3. **Safety wording:**
   - The system **ranks and flags; it never diagnoses.**
   - Never present any treatment or drug as advice.
   - Put the disclaimer on slide 2 and in the footer of every results slide: *"Decision support
     only — not a diagnosis. Synthetic data; validation results."*
4. **Be honest about limits.** Keep every caveat given with a number (synthetic data, circularity,
   closed world).
5. **Design:**
   - Clean and academic.
   - White or very light background, one dark text colour, one accent colour (deep blue or teal).
   - **Red only for red flags and safety.**
   - At most 6 bullets per slide and about 12 words per bullet. Fonts readable from the back of a
     room: body ≥ 18 pt, tables ≥ 14 pt.
   - Number every slide.
   - Add **speaker notes** to every slide from the "Notes" given.
   - Diagrams as simple boxes and arrows; no clip-art, no stock photos of doctors.
6. **Citations on slides:** short form "(Author Year)". The full list goes on the reference slides
   at the end, numbered.

---

### Slide 1 · Title
- **Title:** Nightingale — Clinical Intelligence & Reasoning System
- **Subtitle:** A safety-first, explainable differential-diagnosis assistant for acute chest pain
- **Footer fields:** PRJ1 Mid-term Evaluation · [Team member names and roll numbers] · Supervisor:
  [Name, title] · [Institution] · [Date]
- **Notes:** Introduce the team and the four segments:
  - P1, the medical knowledge graph and graph reasoning;
  - P2, data and the machine-learning rankers;
  - P3, clinical safety, fusion, evidence and explanation;
  - P4, the platform, evaluation and integration.

### Slide 2 · The problem
**Bullets:**
- Acute chest pain is common and high-stakes in emergency care (Gulati 2021).
- Clinicians must separate killers (heart attack, pulmonary embolism, aortic dissection, oesophageal
  rupture) from benign causes (reflux, panic), often before any test returns.
- Missed acute cardiac ischaemia still happens in emergency departments (Pope 2000).
- Existing aids fail in two ways:
  - **ML classifiers** give a label with no inspectable reasoning;
  - **large language models** give fluent text that can be false and uncited (Ji 2023).
- Neither shows **what supports each cause** or **what is still missing**.

**Callout box:** *Decision support only — never a diagnosis. Research prototype on synthetic data.*

**Notes:** The cost of an error is not symmetric. A false alarm costs a test; a missed heart attack
can cost a life. So the system is judged first on keeping the dangerous causes in view.

### Slide 3 · Scope
**Two-column table, the 14 conditions:**

| Must-not-miss (8) | Others (6) |
|---|---|
| Possible NSTEMI / STEMI | Stable angina |
| Unstable angina | Pericarditis |
| Myocarditis | Atrial fibrillation |
| Acute pulmonary edema | PSVT |
| Pulmonary embolism | GERD |
| Spontaneous pneumothorax | Panic attack |
| Boerhaave syndrome | |
| Aortic dissection* | |

- *\*Not in the training data: covered by the knowledge graph and a rule.*
- **In scope:** one presentation, 14 conditions, synthetic and open data, ranked differential with
  red flags and evidence.
- **Out of scope:** diagnosis, treatment or drug advice, real patient data, clinical deployment.

**Notes:** The system is closed-world: anything else is still forced into one of these 14. We say
so wherever we report results.

### Slide 4 · Literature review (1/5): computer-aided diagnosis, 1959 → today
**Timeline graphic (left to right):**
- **1959:** diagnosis as probabilistic reasoning (Ledley 1959)
- **1972:** naive Bayes for acute abdominal pain (de Dombal 1972)
- **1976–1991:** expert systems MYCIN (Shortliffe 1976), INTERNIST-I (Miller 1982), DXplain
  (Barnett 1987), QMR-DT (Shwe 1991)
- **2015:** online symptom checkers list the right diagnosis first only about a third of the time
  (Semigran 2015)
- **2020:** causal ML improves diagnostic accuracy on vignettes (Richens 2020)

**Takeaway bar:** Knowledge-based systems explain but do not learn; learned systems learn but do
not explain. → We combine them.

**Notes:** Our graph score is a naive Bayes, the same family as de Dombal's, now over a knowledge
graph.

### Slide 5 · Literature review (2/5): chest pain and the clinical rules we follow
**Bullets:**
- **Evaluating chest pain:** the 2021 AHA/ACC guideline (Gulati 2021); risk scores such as HEART
  (Six 2008).
- **Aortic dissection:**
  - the ADD-RS risk score (Hiratzka 2010), validated by Rogers 2011 and by ADvISED with D-dimer
    (Nazerian 2018);
  - the frequencies of its findings from the IRAD registry (Hagan 2000; Evangelista 2016).
- **Pulmonary embolism:** the Wells criteria (Wells 2000). **Unstable angina:** the Braunwald
  classification (Braunwald 1989).
- **Which tests to consider:** ESC 2023 acute coronary syndromes (Byrne 2023); ESC 2019 pulmonary
  embolism (Konstantinides 2020).

**Callout, our measurement:**
- Rule-based red flags reach must-not-miss patients with sensitivity **0.823 [0.818, 0.829]** on
  full histories, but only **0.135 [0.130, 0.140]** at 25% of the history (validation).
- Rules alone are not enough.

**Notes:** Each red-flag rule follows one of these published patterns, never a single suggestive
finding. The rules need two or three findings together, which a short history rarely holds.

### Slide 6 · Literature review (3/5): automatic diagnosis, and deep learning on tabular data
**Left column, automatic diagnosis:**
- **DDXPlus** (Fansi Tchango 2022): ~1.3 M synthetic patients, 49 pathologies, coded evidence
  and a differential. **Our dataset.**
- Interactive agents that choose which questions to ask:
  - AARLC (Yuan 2021);
  - CASANDE (Fansi Tchango 2022b), which also emphasises ruling out severe conditions.
- A transformer that generates the differential (DDxT, Alam 2023).
- Earlier dialogue-based work (Wei 2018; Kao 2018). Real records (MIMIC-III, Johnson 2016) need
  credentialed access.

**Right column, tabular deep learning and missing data:**
- FT-Transformer (Gorishniy 2021), from attention (Vaswani 2017), ReGLU (Shazeer 2020) and AdamW
  (Loshchilov 2019).
- Trees still often beat deep models on tables (Grinsztajn 2022; XGBoost, Chen 2016).
- Missing data: theory (Rubin 1976); the best predictor depends on which inputs are missing
  (NeuMiss, Le Morvan 2020); training on masked inputs (Douglas 2017).

**Notes:** This is why XGBoost and logistic regression stay as comparators, and why our deep model
had to pass a safety floor before it could be used.

### Slide 7 · Literature review (4/5): knowledge graphs, combining opinions, calibration, explanation
**Bullets:**
- **Vocabularies and large biomedical graphs:** UMLS (Bodenreider 2004), HPO (Köhler 2021),
  Hetionet (Himmelstein 2017), PrimeKG (Chandak 2023). Broad, not built for one presentation's
  differential.
- **Graphs for diagnosis and prediction:** a health graph learned from records (Rotmensch 2017);
  GraphCare (Jiang 2024); BODHI-S (Eka Care); SKOS mapping relations (Miles 2009).
- **Combining two models:** the logarithmic opinion pool (Genest 1986), a product of experts
  (Hinton 2002).
- **Calibration:** Platt (1999), isotonic (Zadrozny 2002), temperature (Guo 2017); ECE (Naeini
  2015), Brier (1950).
- **Explanation:** SHAP (Lundberg 2017), LIME (Ribeiro 2016), occlusion (Zeiler 2014); prefer
  interpretable reasoning in high-stakes decisions (Rudin 2019).

**Notes:** BODHI-S was our planned backbone, but only 31% of our conditions matched it, so it
enriches four conditions instead.

### Slide 8 · Literature review (5/5): retrieval, LLMs and safety
**Bullets:**
- **Retrieval:** BM25 (Robertson 2009); dense retrieval (Karpukhin 2020); biomedical encoders
  (PubMedBERT, Gu 2021); reciprocal rank fusion (Cormack 2009).
- **Retrieval-augmented generation** (Lewis 2020); medical RAG benchmark MedRAG (Xiong 2024).
- **LLMs in medicine:** Med-PaLM (Singhal 2023), GPT-4 on exams (Nori 2023): strong knowledge, but
  they hallucinate (Ji 2023; Med-HALT, Pal 2023).
- **Safety in decision support:** alert fatigue (Ancker 2017); benefits and risks (Sutton 2020).

**Notes:** We let a language model only *phrase* reasoning that the graph derived, and check every
sentence against a source. That is planned work.

### Slide 9 · Literature review: summary comparison
**Table (keep ✓ / ✗ / text exactly as given):**

| Approach | Explains | Learns from data | Safety-first evaluation | Incomplete histories | Calibrated | Cites evidence | Ranks a condition absent from training |
|---|---|---|---|---|---|---|---|
| Expert systems (INTERNIST-I, QMR, DXplain) | ✓ | ✗ | ✗ | partly | ✗ | ✗ | ✓ |
| Classical ML / XGBoost | ✗ | ✓ | rarely | weak | sometimes | ✗ | ✗ |
| Interactive agents on DDXPlus (AARLC, CASANDE) | partly | ✓ | partly | by design | rarely | ✗ | ✗ |
| LLMs (Med-PaLM, GPT-4) | prose only | ✓ | ✗ | ✓ | ✗ | unreliable | unverified |
| Medical RAG (MedRAG) | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | partly |
| **Nightingale (ours)** | **✓** | **✓** | **✓** | **✓ tested at 100/50/25%** | **measured** | **✓** | **✓** |

**Small caption:** *Our reading of the reviewed literature.*

### Slide 10 · Research gap
**Statement (large text):** No reviewed system ranks acute chest pain in a way that is all of
these:
- explainable from a knowledge graph;
- safety-first, judged on must-not-miss recall;
- robust to an incomplete history, with "not asked" kept apart from "denied";
- calibrated;
- able to rank a dangerous condition missing from its training data.

**Five gaps, each with our response:**

| Gap | Our response |
|---|---|
| **G1** Complete histories hide differences: on full histories XGBoost scores top-1 0.9985 [0.9980, 0.9989] and logistic regression 0.9983 [0.9978, 0.9987] (validation), because of how DDXPlus is generated | Evaluate at **100%, 50% and 25%** of each history, with masks fixed in advance |
| **G2** "Not asked" is treated as "no": plain XGBoost falls to 0.245 [0.240, 0.250] top-1 at 25% | An explicit **"asked" channel** plus training on masked copies |
| **G3** Accuracy is reported, not safety | Headline metric **must-not-miss recall@3 ≥ 0.95**; independent red-flag rules |
| **G4** Black boxes and ungrounded text | **Knowledge-graph reasoning paths**, the model's reasons, **cited evidence** |
| **G5** A model cannot rank what it never saw | **Fusion** with the graph keeps aortic dissection rankable |

**Notes:** We do not claim "knowledge graph plus ML" is new. The novelty is the combination of
safety-first evaluation, incomplete histories, and coverage beyond the training data.

### Slide 11 · Problem definition and objectives
**Formal definition (boxed):**
- **Input:** age, sex, findings each marked *present / denied / not asked*, risk factors.
- **Output:** a ranked list over 14 conditions, with:
  - red flags;
  - supporting (✓), missing (?) and contradicting (✗) findings;
  - tests to consider and questions to ask;
  - cited evidence;
  - the components that failed;
  - the disclaimer.
- **Objective:** first **must-not-miss recall@3 ≥ 0.95** with dangerous false negatives ≤ 0.02, at
  every evidence level; then top-3 and MRR; calibrated (ECE < 0.10); explainable; fast enough to
  re-rank live.

**Objectives table:**

| # | Objective | Status |
|---|---|---|
| O1 | Patient knowledge graph | Planned |
| O2 | Cardiac knowledge graph | **Done** |
| O3 | ML ranker | **Done** |
| O4 | Fuse graph and ML | Built; weight sweep pending |
| O5 | Must-not-miss red flags | **Done** |
| O6 | Cited evidence | Code done; corpus pending |
| O7 | Ablation of every component | Phase 4 |
| O8 | Clinician-facing dashboard | **Done** |

**Research questions (one line each):**
- RQ1, fusion vs each alone;
- RQ2, safety;
- RQ3, grounding vs hallucination;
- RQ4, which components matter;
- RQ5, calibration;
- RQ6, deep vs classical on incomplete histories.

### Slide 12 · Proposed methodology: architecture
**Draw this flow as a left-to-right box diagram.** Solid boxes are built; dashed boxes are planned.

```
Patient case (age · sex · findings: present / denied / not asked)
  ├─► Crosswalk (67 concepts ↔ DDXPlus questions) ─► Cardiac knowledge graph (naive-Bayes score)
  ├─► Case → tokens ─► Encoder (691 columns + "asked" channel) ─► ML ranker (small transformer)
  │        graph posterior + model probabilities ─► Fusion (logarithmic opinion pool)
  └─► Red-flag rules (ADD-RS · Wells · Braunwald)  ─────────────┐
Fusion + red flags ─► Candidates (✓ supporting · ? missing · ✗ contradicting, graph paths)
  ─► Explanation (+ model's reasons) ─► What to ask next ─► Evidence (BM25 + dense, RRF)
  ─► Safety check (disclaimer · flags first · no treatment advice) ─► API + Dashboard
Planned (dashed): calibration · grounded local-LLM explanation with claim check · patient graph (Synthea)
```

**Notes:** Five principles:
1. every module is replaceable;
2. degrade, never crash, and show what failed;
3. safety runs independently of the model;
4. explanations are derived from the graph, not invented;
5. typed interfaces between modules.

### Slide 13 · Dataset
**Table:**

| Source | Used for | Size | Licence |
|---|---|---|---|
| **DDXPlus** (synthetic) | Training and validation | **255,900** train and **33,963** validation patients in our 13 conditions; 16,943 validation patients have a must-not-miss condition | CC-BY-4.0 |
| DDXPlus knowledge base | Graph backbone | 84 evidence questions | CC-BY-4.0 |
| BODHI-S | Graph enrichment (4 conditions) | 107 facts → 56 edges | CC-BY-NC-4.0 |
| Hand-authored (ADD-RS, IRAD) | Aortic dissection | 20 edges | Ours |
| PMC Open Access | Cited evidence | Built on Kaggle (pending) | Creative Commons per article |
| Golden cases | Clinical expectations that must always pass | 4 hand-written vignettes | Ours |

**Bullets:**
- The test split stays closed until Phase 4, opened once.
- Evidence is coded; the English is machine-translated from French, so we work on codes.
- A listed answer can mean "no" (e.g. "did not travel"), which is handled explicitly.

**Notes:** The class imbalance is at most 2.70×. Demographics are synthetic, so equal results by
sex mean nothing.

### Slide 14 · Methods (1/3): the knowledge graph
**Bullets:**
- **129 nodes, 321 edges** from three sources:
  - 245 DDXPlus edges;
  - 56 BODHI-S edges;
  - 20 hand-authored edges.

  Every edge records its source.
- **Stored on** Neo4j AuraDB; NetworkX takes over automatically if the cloud database is paused.
- **The crosswalk:** 67 concepts mapped to DDXPlus answers with SKOS match types (exact, close,
  broader, narrower), so only sound inferences are drawn.
- **The graph score:** a naive-Bayes log-likelihood → a posterior over the 14 conditions. Denied
  findings count against a condition; findings not mentioned count for nothing.
- **Explanation:** supporting, missing and contradicting findings, with reasoning paths.

**Caveat line:** *On DDXPlus patients the graph is circular (built from the same knowledge base),
so a graph-alone accuracy there is not evidence of skill.*

### Slide 15 · Methods (2/3): the ML rankers and fusion
**Bullets:**
- **Baselines:** B0 (prevalence), B1 (logistic regression, XGBoost), B2 (the graph alone).
- **The "asked" channel:** 691 input columns that keep *not asked* apart from *denied*. Models are
  trained on full histories plus **masked copies**.
- **The deep ranker:** a small **FT-Transformer**:
  - 84 question tokens plus age, sex and [CLS];
  - width 96, 2 blocks, 4 heads; **225,645 parameters**;
  - trained on a Kaggle GPU over 5 seeds;
  - runs on a laptop CPU in **1.38 ms** (p95) per patient.
- **Safety floor fixed before training (D-13):** use it only if it meets the safety targets and is
  no worse than logistic regression at 25% of the history. **It passed**, at all 5 seeds.
- **Fusion:** a logarithmic opinion pool of the model's probabilities and the graph's posterior; a
  condition the model never saw takes the graph's view.

**Notes:** The model's probabilities are floored at 0.01, so an overconfident model cannot veto
the graph. The pool's one weight is set by a pre-registered sweep.

### Slide 16 · Methods (3/3): safety, explanation, what to ask next, evidence
**Bullets:**
- **Red-flag rules,** one per must-not-miss condition, after ADD-RS, Wells and Braunwald;
  independent of the model; flagged causes listed first.
- **Safety check:** the disclaimer always; no treatment or drug language; any failed component
  shown.
- **Explanation:** the graph's reasons, plus **the model's reasons** (remove each finding, see what
  moves).
- **What to ask next:**
  - the tests to consider, cited to guidelines;
  - the questions with the most expected information gain over the graph.
- **Evidence:**
  - BM25 + dense search fused by reciprocal rank, over PMC Open Access;
  - every sentence with treatment language screened out;
  - each excerpt with its link and licence.
- **Planned:** calibration (Platt / isotonic / temperature); a grounded local LLM with a claim check;
  the patient graph.

### Slide 17 · Evaluation methodology and tools
**Left, evaluation:**
- A **pre-registered** protocol, frozen before any model was trained; changes by dated amendment
  only.
- **Three evidence levels:** 100%, 50% and 25% of each history; masks fixed and recorded before
  scoring.
- **Metrics:**
  - **must-not-miss recall@3** (headline) and the dangerous false-negative rate;
  - top-1, top-3, MRR;
  - red-flag sensitivity;
  - ECE and Brier;
  - retrieval nDCG;
  - latency.
- **Statistics:** 95% bootstrap intervals, McNemar's test, paired bootstraps, 5 seeds.
- **Baselines** B0–B4; **ablation** A0–A7 (remove one component at a time).

**Right, tools (icons or a compact table):**
- Python 3.11 · NumPy · pandas · scikit-learn · XGBoost
- PyTorch (training on a Kaggle GPU) · NetworkX · Neo4j AuraDB
- FastAPI · Streamlit · sentence-transformers (Kaggle only)
- GitHub Actions CI · 1,100+ automated tests

### Slide 18 · Plan and progress
**Timeline graphic:**

| Phase | Weeks | Status |
|---|---|---|
| 0, preparation and protocol | 1 | ✓ done |
| 1, data and knowledge graph | 2–3 | ✓ done; the patient graph remains |
| 2, reasoning, fusion, prototype | 4–5 | ✓ prototype `v0.1` (14 Oct) |
| 3, evidence and explanation | 6–7 | in progress |
| 4, test split, ablations, report | 8–9 | planned; presentation 15 Nov |

**Done so far:**
- the knowledge graph;
- three ML rankers and the transformer;
- fusion and the red flags;
- the dashboard with live re-ranking (~10 ms);
- the API;
- what to ask next;
- the retrieval code;
- 1,100+ tests.

### Slide 19 · Early validation results (not the final evaluation)
**Footer:** *Decision support only — not a diagnosis. Synthetic data; validation results.*

**Table, at 25% of each patient's history:**

| Model | Must-not-miss recall@3 | Dangerous false-negative rate | Top-1 |
|---|---|---|---|
| Logistic regression, standard training | 0.925 [0.921, 0.929] | 0.024 [0.021, 0.026] | 0.766 [0.761, 0.770] |
| XGBoost, standard training | 0.877 [0.872, 0.882] | 0.059 [0.056, 0.062] | 0.245 [0.240, 0.250] |
| **Transformer, masked training + "asked" channel** | **0.998 [0.997, 0.999]** | 0.0004 [0.0001, 0.0007] | 0.949 [0.946, 0.951] |

**Bullets:**
- **Targets:** must-not-miss recall@3 ≥ 0.95 and dangerous false negatives ≤ 0.02. The standard
  models miss them; the transformer meets them.
- The transformer is level with XGBoost trained the same way. We report that.
- **The commonest error for every model:** unstable angina ranked below stable angina (0.222 [0.206,
  0.238] of unstable-angina patients at 25%). Only a troponin separates them, which is why "what to
  ask next" exists.

**Optional:** a 2-minute live demo of the dashboard.

### Slide 20 · Limitations and next steps
**Limitations:**
- synthetic patients;
- a closed world of 14 conditions;
- the knowledge graph is circular on DDXPlus;
- random masks are not real history-taking;
- no clinical validation.

**Next:**
- the fusion weight sweep;
- calibration;
- the retrieval corpus;
- a grounded LLM explanation;
- the patient graph;
- **the test split, opened once, with the full ablation study**;
- the final report on 15 Nov.

### Slide 21 · Thank you / questions
- Title: "Thank you — questions?"
- Repeat the disclaimer.
- The repository link: `github.com/HarshRohila02/nightingale`

### Slides 22–24 · References (appendix)
Split the numbered list over three slides, small font (12–14 pt), keeping the numbers.

**22 · Diagnosis, chest pain and clinical rules**
1. Ledley RS, Lusted LB. Reasoning foundations of medical diagnosis. *Science*, 1959.
2. de Dombal FT et al. Computer-aided diagnosis of acute abdominal pain. *BMJ*, 1972.
3. Shortliffe EH. *Computer-Based Medical Consultations: MYCIN*. Elsevier, 1976.
4. Miller RA, Pople HE, Myers JD. INTERNIST-I. *N Engl J Med*, 1982.
5. Shwe MA et al. Probabilistic diagnosis using a reformulation of the INTERNIST-1/QMR knowledge base. *Methods Inf Med*, 1991.
6. Barnett GO et al. DXplain. *JAMA*, 1987.
7. Semigran HL et al. Evaluation of symptom checkers for self diagnosis and triage. *BMJ*, 2015.
8. Richens JG, Lee CM, Johri S. Improving the accuracy of medical diagnosis with causal machine learning. *Nat Commun*, 2020.
9. Gulati M et al. 2021 AHA/ACC guideline for the evaluation and diagnosis of chest pain. *Circulation*, 2021.
10. Pope JH et al. Missed diagnoses of acute cardiac ischemia in the emergency department. *N Engl J Med*, 2000.
11. Six AJ, Backus BE, Kelder JC. Chest pain in the emergency room: value of the HEART score. *Neth Heart J*, 2008.
12. Hiratzka LF et al. 2010 ACCF/AHA guidelines for thoracic aortic disease. *Circulation*, 2010.
13. Rogers AM et al. Sensitivity of the aortic dissection detection risk score. *Circulation*, 2011.
14. Nazerian P et al. ADD-RS plus D-dimer for acute aortic syndromes: the ADvISED study. *Circulation*, 2018.
15. Hagan PG et al. The International Registry of Acute Aortic Dissection (IRAD). *JAMA*, 2000.
16. Evangelista A et al. Insights from the International Registry of Acute Aortic Dissection. *Glob Cardiol Sci Pract*, 2016.
17. Wells PS et al. Derivation of a simple clinical model to categorize patients' probability of pulmonary embolism. *Thromb Haemost*, 2000.
18. Braunwald E. Unstable angina: a classification. *Circulation*, 1989.
19. Byrne RA et al. 2023 ESC Guidelines for the management of acute coronary syndromes. *Eur Heart J*, 2023.
20. Konstantinides SV et al. 2019 ESC Guidelines on acute pulmonary embolism. *Eur Heart J*, 2020.

**23 · Data, models and knowledge graphs**
21. Fansi Tchango A et al. DDXPlus: a new dataset for automatic medical diagnosis. *NeurIPS Datasets & Benchmarks*, 2022.
22. Fansi Tchango A et al. Towards trustworthy automatic diagnosis systems by emulating doctors' reasoning with deep RL. *NeurIPS*, 2022.
23. Yuan H, Yu PS. Efficient symptom inquiring and diagnosis via adaptive alignment of RL and classification. arXiv, 2021.
24. Alam MM et al. DDxT: deep generative transformer models for differential diagnosis. 2023.
25. Wei Z et al. Task-oriented dialogue system for automatic diagnosis. *ACL*, 2018.
26. Kao HC, Tang KF, Chang EY. Context-aware symptom checking using hierarchical RL. *AAAI*, 2018.
27. Johnson AEW et al. MIMIC-III. *Sci Data*, 2016.
28. Walonoski J et al. Synthea. *JAMIA*, 2018.
29. Gorishniy Y et al. Revisiting deep learning models for tabular data. *NeurIPS*, 2021.
30. Vaswani A et al. Attention is all you need. *NeurIPS*, 2017.
31. Shazeer N. GLU variants improve transformer. arXiv, 2020.
32. Loshchilov I, Hutter F. Decoupled weight decay regularization. *ICLR*, 2019.
33. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on tabular data? *NeurIPS*, 2022.
34. Chen T, Guestrin C. XGBoost. *KDD*, 2016.
35. Rubin DB. Inference and missing data. *Biometrika*, 1976.
36. Le Morvan M et al. NeuMiss networks. *NeurIPS*, 2020.
37. Douglas L et al. A universal marginalizer for amortized inference in generative models. arXiv, 2017.
38. Bodenreider O. The Unified Medical Language System (UMLS). *Nucleic Acids Res*, 2004.
39. Köhler S et al. The Human Phenotype Ontology in 2021. *Nucleic Acids Res*, 2021.
40. Himmelstein DS et al. Hetionet. *eLife*, 2017.
41. Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable precision medicine (PrimeKG). *Sci Data*, 2023.
42. Rotmensch M et al. Learning a health knowledge graph from electronic medical records. *Sci Rep*, 2017.
43. Jiang P et al. GraphCare. *ICLR*, 2024.
44. Eka Care. BODHI-S condition–symptom knowledge graph. Hugging Face.
45. Miles A, Bechhofer S. SKOS Reference. W3C Recommendation, 2009.

**24 · Fusion, calibration, explanation, retrieval, LLMs, safety and statistics**
46. Genest C, Zidek JV. Combining probability distributions. *Stat Sci*, 1986.
47. Hinton GE. Training products of experts. *Neural Comput*, 2002.
48. Platt J. Probabilistic outputs for support vector machines. *Adv. Large Margin Classifiers*, 1999.
49. Zadrozny B, Elkan C. Transforming classifier scores into accurate multiclass probability estimates. *KDD*, 2002.
50. Guo C et al. On calibration of modern neural networks. *ICML*, 2017.
51. Naeini MP, Cooper GF, Hauskrecht M. Well calibrated probabilities using Bayesian binning. *AAAI*, 2015.
52. Brier GW. Verification of forecasts expressed in terms of probability. *Mon Weather Rev*, 1950.
53. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS*, 2017.
54. Ribeiro MT, Singh S, Guestrin C. "Why should I trust you?" *KDD*, 2016.
55. Zeiler MD, Fergus R. Visualizing and understanding convolutional networks. *ECCV*, 2014.
56. Rudin C. Stop explaining black box models for high stakes decisions. *Nat Mach Intell*, 2019.
57. Robertson S, Zaragoza H. The probabilistic relevance framework: BM25 and beyond. *Found Trends Inf Retr*, 2009.
58. Karpukhin V et al. Dense passage retrieval. *EMNLP*, 2020.
59. Gu Y et al. Domain-specific language model pretraining for biomedical NLP (PubMedBERT). *ACM Trans Comput Healthc*, 2021.
60. Cormack GV, Clarke CLA, Büttcher S. Reciprocal rank fusion. *SIGIR*, 2009.
61. Lewis P et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*, 2020.
62. Singhal K et al. Large language models encode clinical knowledge. *Nature*, 2023.
63. Nori H et al. Capabilities of GPT-4 on medical challenge problems. arXiv, 2023.
64. Ji Z et al. Survey of hallucination in natural language generation. *ACM Comput Surv*, 2023.
65. Pal A et al. Med-HALT. *CoNLL*, 2023.
66. Xiong G et al. Benchmarking retrieval-augmented generation for medicine. *Findings of ACL*, 2024.
67. Ancker JS et al. Alert fatigue in a clinical decision support system. *BMC Med Inform Decis Mak*, 2017.
68. Sutton RT et al. An overview of clinical decision support systems. *NPJ Digit Med*, 2020.
69. McNemar Q. Note on the sampling error of the difference between correlated proportions. *Psychometrika*, 1947.
70. Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
71. Nielsen J. *Usability Engineering*. Academic Press, 1993.

**Final check before you finish:**
- every number matches the prompt exactly, with its interval;
- no citation appears that is not in the list;
- the disclaimer is on slides 2, 19 and 21;
- every slide has speaker notes.

**PROMPT ENDS HERE**
