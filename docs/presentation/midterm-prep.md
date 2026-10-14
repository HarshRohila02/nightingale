# Mid-term evaluation (PRJ1): what we present, and how we prepare

**Version 1.0 · 2026-10-14 · for all four team members.** The slide generator's prompt is in
[`midterm-ppt-prompt.md`](midterm-ppt-prompt.md); this page is what we say, and the facts behind it.

> ⚠️ Nightingale is **decision support, never a diagnosis**, built on **synthetic** patients. Say
> so on the first content slide and again before any figure.

---

## 0. What we are marked on

The course announcement lists four criteria, 25 marks in all. Its **"See details"** text was not in
the screenshot. Paste it into this section, then check the plan below against it (time limit,
slide limit, format, who must speak).

| Part | Criterion | Marks | Slides (prompt file) | Suggested presenter | Time |
|---|---|---:|---|---|---:|
| Literature review | Comprehensiveness of the literature review | 10 | 4–9 | P1 (graph) and P2 (data, ML) | 5 min |
| Literature review | Research gap | 5 | 10 | P2 | 1.5 min |
| Methodology | Objective / problem definition | 5 | 2–3, 11 | P4 (platform, evaluation) | 2 min |
| Methodology | Proposed methodology: tools, techniques, methods, dataset | 5 | 12–17 | P1, P2, P3, P4, each their part | 5 min |
| — | Progress so far, and an optional 2-minute live demo | — (supports the methodology) | 18–19 | P4 | 1.5–3 min |
| — | Questions | — | 20–21 | All | — |

**Weight the time the way the marks are weighted.** The literature review carries 15 of the 25
marks, so it gets the most time. Progress and results are not marked here. Show them briefly, as
proof that the methodology works, not as the talk.

**Fill in before the day:** the four names, the roll numbers, the supervisor's name and title
(the repository records "Prof. Ashish Kumar": confirm), the institution, and the date.

## 1. Three rules for every slide

1. **Every number carries its evidence level and its interval.** The source is `docs/08`, and every
   figure here is validation only. Example: "must-not-miss recall@3 0.998 [0.997, 0.999] at 25% of
   the history, validation". Never round away the interval, and never say "accuracy" without saying
   which.
2. **Every reference is checked before it goes on a slide** (§9).
   - **[R]**: already cited in the repository (some details still to check).
   - **[V]**: added on 2026-10-14 from general knowledge, to fill the literature review. Each [V]
     must be checked against its DOI or Google Scholar page before use. Drop any that cannot be
     confirmed; never fix one up from memory.
3. **No overclaiming.** We rank and flag; we never diagnose. Full-history accuracy is high because
   of how DDXPlus was generated, not because of skill. A graph figure on DDXPlus patients is
   circular. The test split is not opened until Phase 4. (§8 lists the sentences to avoid.)

---

## 2. Part A: the literature review (10 marks)

**What "comprehensive" should look like to an examiner:**
- the field's history, from the classics to current work;
- every component of our system tied to its prior work;
- the clinical sources our rules come from;
- the dataset's own literature;
- the limits of each line of work, which leads straight into the gap.

Present it as **eight strands**, one row each in a summary table (prompt slide 9), with a slide per
group of strands. For each work, give one line on what it did and one line on what it leaves open
for us.

### A1 · Computer-aided diagnosis: from probability to expert systems to modern ML
- **Ledley & Lusted (1959)** [V]: diagnosis framed as probabilistic (Bayesian) reasoning over
  symptoms. This is the root of our naive-Bayes graph score.
- **de Dombal et al. (1972)** [V]: computer-aided diagnosis of acute abdominal pain with naive
  Bayes, in a real emergency setting. A narrow presentation and a small condition set, like ours.
- **MYCIN (Shortliffe 1976)** [V], **INTERNIST-I (Miller, Pople & Myers 1982)** [V],
  **QMR-DT (Shwe et al. 1991)** [V] and **DXplain (Barnett et al. 1987)** [V]: rule-based and
  probabilistic knowledge bases.
  - Strength: explainable.
  - Limit: hand-built, hard to keep current, and they learn nothing from data.
- **Symptom checkers (Semigran et al. 2015)** [V]: online checkers listed the right diagnosis first
  only about a third of the time, a weak safety record.
- **Richens, Lee & Johri (2020)** [V]: causal (counterfactual) diagnosis outperformed associative
  ranking on clinical vignettes. Reasoning structure matters, not only fit to data.
- **What it leaves open:** knowledge-based systems explain but do not learn, and learned systems
  learn but do not explain. **We combine them.**

### A2 · Acute chest pain, and the clinical rules our red flags follow
- **The 2021 AHA/ACC chest-pain guideline (Gulati et al. 2021)** [V]: how chest pain is evaluated,
  and how common it is in emergency departments (check the exact visit figure in the guideline
  before quoting it).
- **Missed diagnoses (Pope et al. 2000)** [V]: a small share of acute cardiac ischaemia patients
  (about 2%: check) were sent home from the emergency department. **This is why our headline
  metric is must-not-miss recall, not top-1.**
- **The HEART score (Six et al. 2008)** [V]: an example of a published chest-pain risk score.
- **Aortic dissection:**
  - the ADD-RS, from the 2010 thoracic aortic guideline (Hiratzka et al. 2010) [R];
  - its validation by Rogers et al. (2011) [R], and ADD-RS with D-dimer, the ADvISED study
    (Nazerian et al. 2018) [V];
  - the IRAD registry (Hagan et al. 2000; Evangelista et al. 2016) [R], which gives the
    frequencies of the findings.

  Our 20 hand-authored dissection edges and the dissection red flag come from these.
- **Pulmonary embolism, the Wells criteria (Wells et al. 2000)** [V]; **unstable angina, the
  Braunwald classification (Braunwald 1989)** [V]. Our PE and unstable-angina rules follow them.
- **The investigation guidelines behind "what to ask next":** ESC 2023 ACS (Byrne et al. 2023)
  [V] and ESC 2019 PE (Konstantinides et al. 2020) [V]. The other sources are listed in
  `src/reasoning/investigations.py`.
- **What it leaves open:** the rules are validated one condition at a time, and need the defining
  finding to be present. Our measurements show they collapse on short histories: red-flag
  sensitivity is 0.823 [0.818, 0.829] on full histories, 0.327 [0.320, 0.334] at 50% and 0.135
  [0.130, 0.140] at 25% (EXP-008, EXP-019, validation). **Rules alone are not enough. A ranker
  must keep the dangerous conditions in view too.**

### A3 · Automatic diagnosis: datasets and models
- **DDXPlus (Fansi Tchango et al. 2022, NeurIPS Datasets & Benchmarks)** [R]:
  - about 1.3 million synthetic patients across 49 pathologies, each with coded evidence, the
    true pathology and a ranked differential;
  - generated from a knowledge base of conditions and their evidence;
  - our dataset.
- **Its benchmark models are interactive agents** that choose which questions to ask:
  - AARLC (Yuan & Yu 2021) [V] and BASD, as reported in the DDXPlus paper [V];
  - CASANDE (Fansi Tchango et al. 2022, NeurIPS) [V], which also emphasises ruling out severe
    conditions.
- **DDxT (Alam et al. 2023)** [V]: a transformer that generates the differential.
- **Earlier symptom-inquiry work on dialogue data:** Wei et al. (2018, ACL) [V] and Kao, Tang &
  Chang (2018, AAAI) [V], reinforcement learning on small real datasets.
- **Why not real records:** MIMIC-III (Johnson et al. 2016) [V] needs credentialed access, and the
  project uses open or synthetic data only (a locked decision).
- **Synthea (Walonoski et al. 2018)** [R]: synthetic patient records, planned for the patient
  graph (1d).
- **What it leaves open:**
  - the agents are judged on the histories they collect themselves;
  - in the studies we reviewed we found no evaluation of a fixed classifier on histories with a
    given share of answers removed, keeping "denied" apart from "never asked", with a safety
    metric at each level, or of what such a model costs to run.

### A4 · Deep learning on tabular data, and missing answers
- **FT-Transformer (Gorishniy et al. 2021, NeurIPS)** [R]: each feature becomes a token, and
  attention runs across them. Our deep ranker follows it.
- **Its building blocks:** attention (Vaswani et al. 2017) [V], ReGLU feed-forward layers
  (Shazeer 2020) [V], and the AdamW optimiser (Loshchilov & Hutter 2019) [V].
- **Grinsztajn, Oyallon & Varoquaux (2022)** [V]: tree models such as XGBoost (Chen & Guestrin
  2016) [V] still often beat deep models on tabular data. That is why XGBoost and logistic
  regression stay as our comparators, and why "deep" must earn its place (D-13).
- **Missing data:**
  - Rubin (1976) [V] set out the theory of missing data;
  - NeuMiss (Le Morvan et al. 2020, NeurIPS) [R, details to check] showed that the best predictor
    with missing inputs is non-linear in both the values and the missingness pattern;
  - the Universal Marginaliser (Douglas et al. 2017) [V] trains on randomly masked inputs.

  Our "asked" channel and masked training copies follow this line.
- **What it leaves open:** applying it to diagnosis, where "not asked" and "denied" must stay two
  different states. Our feature encoding first confused them (R-18).

### A5 · Medical knowledge graphs and vocabularies
- **Vocabularies:** UMLS (Bodenreider 2004) [V], HPO (Köhler et al. 2021) [V], SNOMED CT.
- **Large biomedical graphs:** Hetionet (Himmelstein et al. 2017) [V] and PrimeKG (Chandak,
  Huang & Zitnik 2023) [V]. They are broad, and not built for one presentation's differential.
- **A graph learned from records:** Rotmensch et al. (2017) [V], with a noisy-OR model, close to
  our naive-Bayes graph score.
- **Graphs for prediction:** GraphCare (Jiang et al. 2024, ICLR) [V] adds personalised graphs to
  prediction models.
- **BODHI-S (Eka Care)** [R]: a condition–symptom graph with likelihood bands. It was our planned
  backbone; a week-1 spike found only 31% of our conditions in it (EXP-001), so it enriches four.
- **SKOS (Miles & Bechhofer 2009, W3C)** [R]: the mapping relations (exact, close, broader,
  narrower) our crosswalk uses.
- **What it leaves open:** a graph that keeps a condition the training data never saw in the
  ranking. Aortic dissection is not in DDXPlus, yet the graph and a rule keep it rankable.

### A6 · Combining opinions, calibration and explaining a model
- **Combining two distributions:** the logarithmic opinion pool (Genest & Zidek 1986) [R, details
  to check], related to the product of experts (Hinton 2002) [V].
- **Calibration:**
  - Platt scaling (Platt 1999) [R];
  - isotonic and one-vs-rest calibration (Zadrozny & Elkan 2002) [R];
  - temperature scaling (Guo et al. 2017) [R], which also showed that modern networks are often
    overconfident;
  - ECE (Naeini, Cooper & Hauskrecht 2015) [V] and the Brier score (Brier 1950) [V].
- **Explanation:**
  - SHAP (Lundberg & Lee 2017) [V] and LIME (Ribeiro, Singh & Guestrin 2016) [V];
  - occlusion (Zeiler & Fergus 2014) [V], which our "model's reasons" adapts to findings;
  - Rudin (2019) [V], who argues for interpretable reasoning in high-stakes decisions.
- **What it leaves open:** calibration and explanation are rarely reported for diagnosis models at
  several evidence levels. We found that a calibrator fitted on full histories over-sharpens short
  ones (EXP-007).

### A7 · Retrieval and large language models in medicine
- **Retrieval:**
  - BM25 (Robertson & Zaragoza 2009) [V];
  - dense passage retrieval (Karpukhin et al. 2020) [V];
  - biomedical encoders such as PubMedBERT (Gu et al. 2021) [V], on which our embedding model
    builds;
  - reciprocal rank fusion (Cormack, Clarke & Büttcher 2009) [V], our hybrid of the two.
- **Retrieval-augmented generation (Lewis et al. 2020)** [V].
- **LLMs in medicine:**
  - Med-PaLM (Singhal et al. 2023, Nature) [V] and GPT-4 on medical exams (Nori et al. 2023) [V]:
    strong knowledge;
  - hallucination surveys (Ji et al. 2023) [V] and Med-HALT (Pal et al. 2023) [V]: false
    statements;
  - MedRAG (Xiong et al. 2024) [V]: retrieval helps medical question answering.
- **What it leaves open:** checking every sentence against a citable source, and keeping treatment
  advice out of the retrieved text. Our retrieval screens treatment language out; 3c adds the
  claim check.

### A8 · Safety and the human side of decision support
- **Alert fatigue (Ancker et al. 2017)** [V]: repeated alerts are ignored. So we measure how often
  a flag fires for patients who have no must-not-miss condition (25%, EXP-008), not only how many
  it catches.
- **Benefits and risks of clinical decision support (Sutton et al. 2020)** [V].

### The summary table (prompt slide 9)
| Approach | Explains its ranking | Learns from data | Safety-first evaluation | Works on incomplete histories | Calibrated | Cites evidence | Ranks a condition absent from training |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Expert systems (INTERNIST-I, QMR, DXplain) | ✓ | ✗ | ✗ | partly | ✗ | ✗ | ✓ |
| Classical ML / XGBoost classifiers | ✗ | ✓ | rarely | weak (R-18) | sometimes | ✗ | ✗ |
| Interactive agents on DDXPlus (AARLC, CASANDE) | partly | ✓ | partly | by design | rarely | ✗ | ✗ |
| LLMs (Med-PaLM, GPT-4) | prose only | ✓ | ✗ | ✓ | ✗ | unreliable | ✓, but unverified |
| RAG for medicine (MedRAG) | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | partly |
| **Nightingale (ours)** | ✓ graph paths and the model's reasons | ✓ | ✓ must-not-miss recall first | ✓ tested at 100/50/25% | measured (EXP-007) | ✓ (3a, corpus pending) | ✓ aortic dissection |

**Our own reading of the literature:** present the ✓/✗ marks as such, and be ready to defend each
one.

---

## 3. Part B: the research gap (5 marks)

**One sentence for the slide:** *No reviewed system ranks the causes of acute chest pain in a way
that is all of these: explainable from a knowledge graph; safety-first, judged on must-not-miss
recall; robust to an incomplete history, with "not asked" kept apart from "denied"; calibrated; and
able to rank a dangerous condition its training data never contained.*

**The five gaps.** For each: what is missing, how we know, and how we address it.

| # | Gap | Evidence | What we do |
|---|---|---|---|
| G1 | **Evaluation on complete histories hides the differences between models.** On DDXPlus full histories, ML models are near-perfect | Our measurement, not a claim about others: XGBoost top-1 0.9985 [0.9980, 0.9989] and logistic regression 0.9983 [0.9978, 0.9987], 100%, validation (EXP-004, R-16). Each patient's evidence comes only from their condition's list | Score every system at **100%, 50% and 25%** of each history, with fixed masks recorded before scoring (amendment 3) |
| G2 | **"Not asked" is treated as "no".** A standard encoding gives an unasked question no column, so a short history looks like a patient who denied everything | At 25%, plain XGBoost: top-1 0.245 [0.240, 0.250], must-not-miss recall@3 0.877 [0.872, 0.882] (EXP-019) | An explicit **"asked" channel** plus masked training copies. At 25%, the transformer reaches must-not-miss recall@3 0.998 [0.997, 0.999] (EXP-021) |
| G3 | **Accuracy is reported, not safety.** Top-1 dominates; a missed MI costs more than a false alarm | The literature reports top-1 or differential accuracy; the clinical literature on missed ischaemia (Pope et al. 2000) | Headline metric **must-not-miss recall@3 ≥ 0.95**; the dangerous false-negative rate ≤ 0.02; red-flag rules independent of the model |
| G4 | **Black boxes and ungrounded text.** ML gives no reasoning; LLMs produce text without verified sources | A1, A7 | A **knowledge graph** gives supporting, missing and contradicting findings with reasoning paths; the **model's reasons** come from occlusion; retrieval **cites** passages, screened of treatment advice |
| G5 | **Training data limits what can be ranked.** A pure ML model cannot rank a condition it never saw | DDXPlus has no aortic dissection | **Fusion** (logarithmic opinion pool) keeps the graph's view, so dissection ranks first on GC-003 even with red flags off |

**Also novel in method:**
- a **pre-registered** evaluation protocol, frozen before any model was trained, with every change
  dated and disclosed;
- a deep model adopted only through a **safety floor fixed before training** (D-13);
- **cost and latency** reported (1.38 ms p95 a patient; the whole pipeline 10–13 ms p95, EXP-022).

**What we do not claim:**
- that "knowledge graph plus ML" is new (it is not, `docs/00` §5);
- clinical efficacy;
- that our deep model beats XGBoost trained the same way. It is level (EXP-021).

---

## 4. Part C: objective and problem definition (5 marks)

**The problem in two sentences.** Acute chest pain is common and high-stakes: the clinician must
separate the causes that kill if missed from benign ones, often from the history alone, before any
test returns. Existing aids either give an unexplained label (ML) or fluent but unverified text
(LLMs), and neither shows what supports each candidate or what is still missing.

**Formal definition (prompt slide 11).**
- **Input:** a case `x` = age, sex, and findings, each marked **present**, **denied** or **not
  asked** (FR-1.3), plus risk factors. In practice the history is often incomplete.
- **Output:** a ranking of the 14 conditions `C` (13 from DDXPlus plus aortic dissection), with:
  - red flags for the 8 must-not-miss conditions;
  - for each candidate, the supporting (✓), missing (?) and contradicting (✗) findings, with the
    graph's reasoning paths and the model's reasons;
  - the tests to consider and the questions to ask next;
  - cited evidence;
  - the components that failed, if any;
  - the disclaimer.
- **Objective:** first, keep the must-not-miss conditions in view (must-not-miss recall@3 ≥ 0.95
  and dangerous false negatives ≤ 0.02, at every evidence level); then rank well (top-3, MRR) and
  be calibrated (ECE < 0.10); always explainably, and fast enough to re-rank live.

**Objectives** (`docs/00` §3; O7 as amended):

| # | Objective | Measured by | Status (2026-10-14) |
|---|---|---|---|
| O1 | Patient knowledge graph from case data | Built for ≥ 95% of cases | Planned (1d) |
| O2 | Cardiac knowledge graph of condition↔finding links | All 13 conditions, with qualifiers | **Done**: 129 nodes, 321 edges, three sources |
| O3 | ML ranker | Top-3 well above the prevalence baseline | **Done**: logistic regression, XGBoost, transformer |
| O4 | Fuse graph and ML into one explainable ranking | Fusion beats both alone | Built; its weight sweep waits for E-2 |
| O5 | Must-not-miss red flags | Must-not-miss recall@3 ≥ 0.95 | **Done**: a rule per must-not-miss condition |
| O6 | Cited evidence per candidate | ≥ 80% of claims traceable | Code done; the corpus waits for the Kaggle run |
| O7 | Each component's contribution | Ablation A0–A7 and A1b, at three levels | Phase 4 (test split) |
| O8 | Usable clinician-facing interface | Working dashboard | **Done**: Streamlit, live re-ranking, `v0.1-prototype` |

**Research questions** (frozen protocol, `docs/05` §1): show the six in short form.
- RQ1: does fusion beat ML-only and graph-only?
- RQ2: does it improve safety?
- RQ3: does grounding reduce unsupported claims?
- RQ4: which components contribute?
- RQ5: is it calibrated?
- RQ6: does a deep ranker hold up better on incomplete histories?

**Scope:**
- **In:** one presentation; 14 conditions; synthetic and open data; decision support only.
- **Out:** diagnosis, any treatment or drug advice, real patient data, clinical deployment.

---

## 5. Part D: the proposed methodology (5 marks)

### D1 · Dataset
| Source | What | Size used | Licence | Role |
|---|---|---|---|---|
| **DDXPlus** | Synthetic patients, coded evidence, true pathology, differential | **255,900** train and **33,963** validation patients in our 13 conditions (25% of each split); the test split stays closed until Phase 4 | CC-BY-4.0 | Training, validation, graph backbone |
| DDXPlus knowledge base | Conditions and their evidence questions | 84 questions for our 13 conditions; the vocabulary holds 919 entries | CC-BY-4.0 | Graph backbone |
| BODHI-S | Condition–symptom graph with likelihood bands | 107 facts for MI, pericarditis, PE, GERD; 56 edges | CC-BY-NC-4.0 (non-commercial) | Enrichment |
| Hand-authored | Aortic dissection from the ADD-RS and IRAD | 20 edges, each with its reference | Our own | Covers a condition DDXPlus lacks |
| PMC Open Access | Article passages, Creative Commons licences only | Built on Kaggle (pending) | CC per article | Cited evidence |
| Synthea | Synthetic patient records | Planned | Open | Patient graph (1d) |
| Golden cases | 4 hand-written clinical vignettes (ACS, PE, dissection, GERD) | 4 | Our own | Executable clinical expectations; they must always pass |

**Data facts to know:**
- 16,943 validation patients have a must-not-miss condition.
- The class imbalance is 2.70× at most; the rarest condition is pneumothorax, with 10,162 training
  patients.
- Evidence comes in four token forms; the "listed but means no" trap is handled (EXP-013).
- DDXPlus's English is machine-translated, so we work on codes.
- Demographics are synthetic (~50% female everywhere).

### D2 · Methods: the pipeline, step by step
1. **Crosswalk:** 67 hand-authored concepts mapped to DDXPlus answers, by SKOS match type.
2. **Knowledge graph:** 129 nodes and 321 edges from three sources, each edge keeping its source.
   It is stored on Neo4j AuraDB, with a NetworkX fallback.
3. **Graph score:** a naive-Bayes log-likelihood over the graph, closed under the crosswalk. It
   gives a posterior over the 14 conditions (EXP-005).
4. **Feature encoding:** 691 columns, with the **"asked" channel** that keeps "not asked" apart
   from "denied". Models are trained on full histories plus **masked copies**.
5. **ML rankers:**
   - B0, a prevalence baseline;
   - B1, logistic regression and XGBoost;
   - **B1-DL, a small FT-Transformer:**
     - 84 question tokens plus age, sex and [CLS];
     - width 96, 2 blocks, 4 heads; 225,645 parameters;
     - trained on a Kaggle GPU over 5 seeds;
     - runs on the laptop as a NumPy forward pass.
6. **Fusion:** a logarithmic opinion pool of the model's probabilities, floored at 0.01, and the
   graph's posterior. Its weight α is to be chosen by the pre-registered sweep EXP-006.
7. **Red-flag rules:** independent of the model, after ADD-RS, Wells and Braunwald, one per
   must-not-miss condition.
8. **Safety layer:** the disclaimer always attached, flags first, no treatment or drug language,
   failures reported (`degraded_components`).
9. **Explanation:** the graph's supporting, missing and contradicting findings and reasoning paths,
   plus the **model's reasons** (leave-one-finding-out occlusion).
10. **What to ask next:**
    - cited diagnostic tests per condition;
    - the unrecorded findings ranked by **expected information gain** over the graph.
11. **Evidence retrieval:**
    - BM25 and a dense search with `S-PubMedBert-MS-MARCO` vectors precomputed on Kaggle, fused by
      reciprocal rank (k = 60);
    - any sentence with treatment language is screened out.
12. **Planned:**
    - calibration (Platt, isotonic, temperature; EXP-007 stage 2);
    - a grounded local-LLM explanation with a claim check (3b, 3c);
    - the patient graph from Synthea (1d).

### D3 · Evaluation methodology
- **Pre-registered:** `docs/05` was frozen before any model was trained, and changes only through
  dated amendments.
- **The test split is opened once, in Phase 4.** Everything so far is validation.
- **Three evidence levels**, 100%, 50% and 25% of each history, with masks drawn once and their
  digests recorded before scoring (`e99a7a8fbf792675`).
- **Metrics:**
  - top-1, top-3 and MRR;
  - **must-not-miss recall@3** (headline) and the dangerous false-negative rate;
  - red-flag sensitivity and precision;
  - ECE and Brier;
  - retrieval Precision@k and nDCG;
  - latency.
- **Statistics:** a 95% bootstrap interval on every figure (1,000 resamples, seed 42); McNemar's
  test (McNemar 1947) [V]; paired bootstraps (Efron & Tibshirani 1993) [V]; five seeds.
- **Baselines:** B0, B1-LR, B1-XGB, B1-DL, B2 (the graph alone), and B3/B4 (an LLM alone,
  retrieval plus LLM; planned).
- **Ablation:** A0 is the full system, then A1 (no medical graph), A1b, A2 (no ML), A3 (no
  retrieval), A4 (no red flags), A5 (no calibration), A6 (no patient graph) and A7 (classical ML
  in place of the transformer).

### D4 · Tools
| Purpose | Tool |
|---|---|
| Language and data | Python 3.11, NumPy, pandas, PyArrow |
| Classical ML | scikit-learn (logistic regression), XGBoost |
| Deep learning | PyTorch on a Kaggle T4 GPU for training only; a NumPy forward pass on the laptop CPU for inference |
| Knowledge graph | Neo4j AuraDB Free; NetworkX as the fallback |
| Retrieval | BM25 and a NumPy dense search; `sentence-transformers` with `pritamdeka/S-PubMedBert-MS-MARCO` on Kaggle; NCBI E-utilities for PMC Open Access |
| Service and interface | FastAPI, with `/diagnose`, `/next-questions` and `/health`; Streamlit dashboard |
| LLM (planned) | Ollama, a local model; which one is decision D-5 |
| Compute | The laptop CPU; Kaggle (GPU T4 x2) and Colab for cloud jobs |
| Engineering | Git and GitHub, pull requests, GitHub Actions CI; 1,100+ automated tests including the golden cases; black and ruff |

### D5 · Plan and timeline
- Phases 0–4 over Weeks 1–9.
- `v0.1-prototype` on 2026-10-14.
- Feature-complete on 2026-11-03.
- The test split opened once on 2026-11-06/07.
- **The prototype and the report presented on 2026-11-15** (`PROGRESS.md` §5).

---

## 6. Progress so far (not marked; one slide, plus an optional demo)

Validation only, synthetic DDXPlus patients. The figures are from `docs/08`.

| At 25% of the history | Must-not-miss recall@3 | Dangerous FN rate | Top-1 |
|---|---|---|---|
| Plain logistic regression (B1-LR) | 0.925 [0.921, 0.929] | 0.024 [0.021, 0.026] | 0.766 [0.761, 0.770] |
| Plain XGBoost (B1-XGB) | 0.877 [0.872, 0.882] | 0.059 [0.056, 0.062] | 0.245 [0.240, 0.250] |
| **Transformer, masked + "asked" (B1-DL′+aug, seed 42)** | **0.998 [0.997, 0.999]** | 0.0004 [0.0001, 0.0007] | 0.949 [0.946, 0.951] |

- **Targets:** must-not-miss recall@3 ≥ 0.95 and dangerous FN ≤ 0.02. The plain models miss them;
  the transformer meets them.
- **The transformer against logistic regression trained the same way:** non-inferior at all five
  seeds; MRR +0.0018 [+0.0009, +0.0026] at seed 42 (EXP-021). It is **level with XGBoost trained
  the same way**: say so.
- **The commonest error, for every model:** unstable angina ranked below stable angina, 0.222
  [0.206, 0.238] of unstable angina patients at 25% for the transformer (EXP-023). Only a troponin
  separates them, which is why "what to ask next" exists.
- **Built:** the dashboard with live re-ranking (~10 ms), the API, the red flags, the model's
  reasons, what to ask next, and the retrieval code. 1,100+ tests; CI green.

**Optional live demo, 2 minutes,** following `docs/15-prototype-demo.md` §4:
- GC-003: dissection first, never seen in training;
- GC-001: live re-ranking.

Have screenshots ready in case the laptop fails.

---

## 7. Likely questions, and answers

1. **"Isn't 99.8% accuracy suspicious?"** Yes, and we say so. DDXPlus draws each patient's
   evidence only from their condition's list, so full histories are easy (R-16). That is why we
   test at 50% and 25%, where the models separate.
2. **"Why synthetic data?"** Real records (MIMIC) need credentialed access and ethics approval; the
   project uses open or synthetic data only. We state the limit everywhere.
3. **"Why a knowledge graph if ML is so accurate?"**
   - It explains (paths, supporting and missing findings).
   - It covers a condition absent from training (dissection).
   - It is independent of the ML's errors.
   - Its value is tested by ablation A1/A1b in Phase 4.
4. **"Isn't the graph circular?"** On DDXPlus patients, yes: the graph is built from the knowledge
   base that generated them (R-12). We never report a graph-alone DDXPlus figure as skill (its
   top-1 0.9234 [0.9206, 0.9259] is circular), and we disclose it. The golden cases are the
   evidence that counts.
5. **"Why a transformer? Did deep learning win?"**
   - The supervisor asked for the best-suited deep model and its cost.
   - Findings interact, and attention models that.
   - It had to pass a safety floor fixed before training.
   - It is not better than XGBoost trained the same way, and we report that.
6. **"How do you handle missing information?"**
   - An explicit "asked" channel, so "not asked" ≠ "denied".
   - Masked training copies.
   - Evaluation at 3 levels with masks fixed in advance.
7. **"How is it safe?"**
   - Must-not-miss recall is the headline.
   - Red flags run independently and sort first.
   - No treatment advice, ever.
   - The disclaimer on every result.
   - Failures are shown, not hidden.
8. **"How do you prevent LLM hallucination?"**
   - The explanation is derived from the graph, not generated.
   - Any LLM may only phrase it, every sentence checked against a source (3c), with a fallback to
     the template.
   - Retrieved text is screened of treatment language.
9. **"Is it real time?"** Soft real time: 1.38 ms p95 per patient for the model and 10–13 ms p95
   for the whole pipeline on a laptop CPU (EXP-022), so it re-ranks as findings are ticked.
10. **"How do you know you didn't tune on the test set?"**
    - The test split was never downloaded; it is opened once, in Phase 4.
    - The protocol was frozen first.
    - The masks' digests were recorded before scoring.
11. **"What are the limitations?"** Synthetic data; a closed world of 14 conditions (R-13); random
    masks are not real history-taking; circularity (R-12); no clinical validation.
12. **"What is left?"**
    - The fusion sweep (waiting on a team decision, E-2);
    - calibration stage 2;
    - the retrieval corpus;
    - the LLM explainer and claim check;
    - the patient graph;
    - the test-split evaluation and ablations;
    - the report.

## 8. Sentences to avoid, and what to say instead

| Avoid | Say instead |
|---|---|
| "The system diagnoses…" | "The system ranks the possible causes and flags the dangerous ones" |
| "99.8% accurate" | "near-perfect on complete synthetic histories, which is the dataset's construction (R-16); we test at 50% and 25%" |
| "The graph alone is 92% accurate" | "on DDXPlus the graph-alone figure is circular, so we don't count it" |
| "Deep learning outperforms" | "the transformer is non-inferior to logistic regression and level with XGBoost trained the same way" |
| "Clinically validated" | "validated on synthetic data; no clinical validation" |
| "Results on the test set" | "validation results; the test split is opened once, in Phase 4" |
| Any drug or treatment name as advice | (never) |

## 9. Before the presentation

**One week before:**
- [ ] Paste the announcement's **"See details"** into §0 and adjust the plan.
- [ ] Fill in the names, roll numbers, supervisor and institution.
- [ ] **Check every [V] reference** against its DOI or Google Scholar page (title, authors, year,
      venue). Delete any that cannot be confirmed. Check the [R] references marked ⚑ in
      `docs/report/02-background.md` §2.8.
- [ ] Generate the slides from `midterm-ppt-prompt.md`, then check every number against `docs/08`.

**The day before:**
- [ ] Rehearse to time: literature review 6–7 minutes, methodology 7 minutes.
- [ ] Assign the questions in §7, one owner each.
- [ ] If demoing, follow `docs/15-prototype-demo.md` §2: pull, run the tests, resume Aura, warm
      the dashboard.

**On the day:**
- [ ] Laptop on the charger.
- [ ] Screenshots as a backup.
- [ ] The disclaimer on the first content slide.

---

## 10. References

Numbered in the order the prompt file uses. **[R]** is cited in the repository; ⚑ means its full
details are still to check. **[V]** was added 2026-10-14 from general knowledge and must be checked
before use.

**A1 · Computer-aided diagnosis**
1. Ledley RS, Lusted LB. Reasoning foundations of medical diagnosis. *Science.* 1959;130(3366):9–21. [V]
2. de Dombal FT, Leaper DJ, Staniland JR, McCann AP, Horrocks JC. Computer-aided diagnosis of acute abdominal pain. *BMJ.* 1972;2(5804):9–13. [V]
3. Shortliffe EH. *Computer-Based Medical Consultations: MYCIN.* Elsevier; 1976. [V]
4. Miller RA, Pople HE Jr, Myers JD. INTERNIST-I, an experimental computer-based diagnostic consultant for general internal medicine. *N Engl J Med.* 1982;307(8):468–476. [V]
5. Shwe MA, Middleton B, Heckerman DE, et al. Probabilistic diagnosis using a reformulation of the INTERNIST-1/QMR knowledge base. *Methods Inf Med.* 1991;30(4):241–255. [V]
6. Barnett GO, Cimino JJ, Hupp JA, Hoffer EP. DXplain: an evolving diagnostic decision-support system. *JAMA.* 1987;258(1):67–74. [V]
7. Semigran HL, Linder JA, Gidengil C, Mehrotra A. Evaluation of symptom checkers for self diagnosis and triage: audit study. *BMJ.* 2015;351:h3480. [V]
8. Richens JG, Lee CM, Johri S. Improving the accuracy of medical diagnosis with causal machine learning. *Nat Commun.* 2020;11:3923. [V]

**A2 · Chest pain and clinical rules**
9. Gulati M, Levy PD, Mukherjee D, et al. 2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR guideline for the evaluation and diagnosis of chest pain. *Circulation.* 2021;144(22):e368–e454. [V]
10. Pope JH, Aufderheide TP, Ruthazer R, et al. Missed diagnoses of acute cardiac ischemia in the emergency department. *N Engl J Med.* 2000;342(16):1163–1170. [V]
11. Six AJ, Backus BE, Kelder JC. Chest pain in the emergency room: value of the HEART score. *Neth Heart J.* 2008;16(6):191–196. [V]
12. Hiratzka LF, Bakris GL, Beckman JA, et al. 2010 ACCF/AHA/AATS/ACR/ASA/SCA/SCAI/SIR/STS/SVM guidelines for the diagnosis and management of patients with thoracic aortic disease. *Circulation.* 2010;121(13):e266–e369. [R]
13. Rogers AM, Hermann LK, Booher AM, et al. Sensitivity of the aortic dissection detection risk score. *Circulation.* 2011;123(20):2213–2218. [R]
14. Nazerian P, Mueller C, Soeiro AM, et al. Diagnostic accuracy of the aortic dissection detection risk score plus D-dimer for acute aortic syndromes: the ADvISED prospective multicenter study. *Circulation.* 2018;137(3):250–258. [V]
15. Hagan PG, Nienaber CA, Isselbacher EM, et al. The International Registry of Acute Aortic Dissection (IRAD): new insights into an old disease. *JAMA.* 2000;283(7):897–903. [R]
16. Evangelista A, Maldonado G, Gruosso D, et al. Insights from the International Registry of Acute Aortic Dissection. *Glob Cardiol Sci Pract.* 2016;2016(1):e201608. [R]
17. Wells PS, Anderson DR, Rodger M, et al. Derivation of a simple clinical model to categorize patients probability of pulmonary embolism. *Thromb Haemost.* 2000;83(3):416–420. [V]
18. Braunwald E. Unstable angina: a classification. *Circulation.* 1989;80(2):410–414. [V]
19. Byrne RA, Rossello X, Coughlan JJ, et al. 2023 ESC Guidelines for the management of acute coronary syndromes. *Eur Heart J.* 2023;44(38):3720–3826. [V]
20. Konstantinides SV, Meyer G, Becattini C, et al. 2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism. *Eur Heart J.* 2020;41(4):543–603. [V]

**A3 · Automatic diagnosis**
21. Fansi Tchango A, Goel R, Wen Z, Martel J, Ghosn J. DDXPlus: a new dataset for automatic medical diagnosis. *NeurIPS Datasets and Benchmarks.* 2022. arXiv:2205.09148. [R ⚑]
22. Fansi Tchango A, Goel R, Martel J, Wen Z, Marceau Caron G, Ghosn J. Towards trustworthy automatic diagnosis systems by emulating doctors' reasoning with deep reinforcement learning (CASANDE). *NeurIPS.* 2022. [V]
23. Yuan H, Yu PS. Efficient symptom inquiring and diagnosis via adaptive alignment of reinforcement learning and classification (AARLC). arXiv preprint, 2021. [V: confirm venue]
24. Alam MM, Raff E, Biderman S, Oates T, Holt J. DDxT: deep generative transformer models for differential diagnosis. 2023. [V: confirm authors and venue]
25. Wei Z, Liu Q, Peng B, et al. Task-oriented dialogue system for automatic diagnosis. *ACL.* 2018. [V]
26. Kao HC, Tang KF, Chang EY. Context-aware symptom checking for disease diagnosis using hierarchical reinforcement learning. *AAAI.* 2018. [V]
27. Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. *Sci Data.* 2016;3:160035. [V]
28. Walonoski J, Kramer M, Nichols J, et al. Synthea: an approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record. *J Am Med Inform Assoc.* 2018;25(3):230–238. [R ⚑ (the repository holds no year or venue)]

**A4 · Tabular deep learning and missing data**
29. Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. Revisiting deep learning models for tabular data. *NeurIPS.* 2021. [R ⚑]
30. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. *NeurIPS.* 2017. [V]
31. Shazeer N. GLU variants improve transformer. arXiv:2002.05202, 2020. [V]
32. Loshchilov I, Hutter F. Decoupled weight decay regularization. *ICLR.* 2019. [V]
33. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS Datasets and Benchmarks.* 2022. [V]
34. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. *KDD.* 2016. [V]
35. Rubin DB. Inference and missing data. *Biometrika.* 1976;63(3):581–592. [V]
36. Le Morvan M, Josse J, Moreau T, Scornet E, Varoquaux G. NeuMiss networks: differentiable programming for supervised learning with missing values. *NeurIPS.* 2020. [R ⚑]
37. Douglas L, Zarov I, Gourgoulias K, et al. A universal marginalizer for amortized inference in generative models. arXiv:1711.00695, 2017. [V]

**A5 · Knowledge graphs and vocabularies**
38. Bodenreider O. The Unified Medical Language System (UMLS): integrating biomedical terminology. *Nucleic Acids Res.* 2004;32(Database issue):D267–D270. [V]
39. Köhler S, Gargano M, Matentzoglu N, et al. The Human Phenotype Ontology in 2021. *Nucleic Acids Res.* 2021;49(D1):D1207–D1217. [V]
40. Himmelstein DS, Lizee A, Hessler C, et al. Systematic integration of biomedical knowledge prioritizes drugs for repurposing (Hetionet). *eLife.* 2017;6:e26726. [V]
41. Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable precision medicine (PrimeKG). *Sci Data.* 2023;10:67. [V]
42. Rotmensch M, Halpern Y, Tlimat A, Horng S, Sontag D. Learning a health knowledge graph from electronic medical records. *Sci Rep.* 2017;7:5994. [V]
43. Jiang P, Xiao C, Cross A, Sun J. GraphCare: enhancing healthcare predictions with personalized knowledge graphs. *ICLR.* 2024. [V]
44. Eka Care. BODHI-S condition–symptom knowledge graph. Hugging Face, `ekacare/BODHI-S`, CC-BY-NC-4.0. [R ⚑]
45. Miles A, Bechhofer S, eds. SKOS Simple Knowledge Organization System Reference. W3C Recommendation; 2009. [R ⚑]

**A6 · Fusion, calibration and explanation**
46. Genest C, Zidek JV. Combining probability distributions: a critique and an annotated bibliography. *Stat Sci.* 1986;1(1):114–135. [R ⚑]
47. Hinton GE. Training products of experts by minimizing contrastive divergence. *Neural Comput.* 2002;14(8):1771–1800. [V]
48. Platt J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: *Advances in Large Margin Classifiers.* MIT Press; 1999. [R ⚑]
49. Zadrozny B, Elkan C. Transforming classifier scores into accurate multiclass probability estimates. *KDD.* 2002. [R ⚑]
50. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML.* 2017. [R ⚑]
51. Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated probabilities using Bayesian binning. *AAAI.* 2015. [V]
52. Brier GW. Verification of forecasts expressed in terms of probability. *Mon Weather Rev.* 1950;78(1):1–3. [V]
53. Lundberg SM, Lee SI. A unified approach to interpreting model predictions (SHAP). *NeurIPS.* 2017. [V]
54. Ribeiro MT, Singh S, Guestrin C. "Why should I trust you?": explaining the predictions of any classifier (LIME). *KDD.* 2016. [V]
55. Zeiler MD, Fergus R. Visualizing and understanding convolutional networks. *ECCV.* 2014. [V]
56. Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nat Mach Intell.* 2019;1:206–215. [V]

**A7 · Retrieval and LLMs**
57. Robertson S, Zaragoza H. The probabilistic relevance framework: BM25 and beyond. *Found Trends Inf Retr.* 2009;3(4):333–389. [V]
58. Karpukhin V, Oğuz B, Min S, et al. Dense passage retrieval for open-domain question answering. *EMNLP.* 2020. [V]
59. Gu Y, Tinn R, Cheng H, et al. Domain-specific language model pretraining for biomedical natural language processing (PubMedBERT). *ACM Trans Comput Healthc.* 2021;3(1). [V]
60. Cormack GV, Clarke CLA, Büttcher S. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR.* 2009. [V]
61. Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS.* 2020. [V]
62. Singhal K, Azizi S, Tu T, et al. Large language models encode clinical knowledge (Med-PaLM). *Nature.* 2023;620:172–180. [V]
63. Nori H, King N, McKinney SM, Carignan D, Horvitz E. Capabilities of GPT-4 on medical challenge problems. arXiv:2303.13375, 2023. [V]
64. Ji Z, Lee N, Frieske R, et al. Survey of hallucination in natural language generation. *ACM Comput Surv.* 2023;55(12). [V]
65. Pal A, Umapathi LK, Sankarasubbu M. Med-HALT: medical domain hallucination test for large language models. *CoNLL.* 2023. [V]
66. Xiong G, Jin Q, Lu Z, Zhang A. Benchmarking retrieval-augmented generation for medicine (MedRAG). *Findings of ACL.* 2024. [V]

**A8 · Safety and evaluation**
67. Ancker JS, Edwards A, Nosal S, et al. Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. *BMC Med Inform Decis Mak.* 2017;17:36. [V]
68. Sutton RT, Pincock D, Baumgart DC, et al. An overview of clinical decision support systems: benefits, risks, and strategies for success. *NPJ Digit Med.* 2020;3:17. [V]
69. McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika.* 1947;12(2):153–157. [V]
70. Efron B, Tibshirani RJ. *An Introduction to the Bootstrap.* Chapman & Hall; 1993. [V]
71. Nielsen J. *Usability Engineering.* Academic Press; 1993 (response-time limits). [R ⚑]
