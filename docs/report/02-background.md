# 2 · Background and related work

> **Draft 1, 2026-10-13.** Every reference here is one the project's documents or code already
> cite. Where the report needs a source the repository does not hold, the text says
> `[citation needed: …]`. No reference, author or year has been added from outside the repository;
> the full bibliographic details of several references must still be checked (§2.8).

## 2.1 Acute chest pain as a differential-diagnosis problem

A **differential diagnosis** is a ranked list of candidate conditions, not a single answer
(`docs/01` §1.3). Chest pain is the textbook case. The clinician weighs the character of the pain
(pressure, sharp, burning, tearing), where it radiates (jaw, arm, back), how fast it began, what
makes it worse or better (exertion, breathing, posture, lying flat), and the risk factors, and
then decides which tests to order (`docs/09` §1.2) [citation needed: a standard
emergency-medicine account of the chest-pain history].

Table 2.1 summarises the hallmark features the team used as its domain primer (`docs/09` §1.2).
They are a teaching summary, not a clinical reference.

**Table 2.1 — Hallmarks of the conditions in scope** (`docs/09` §1.2).

| Condition | Hallmarks |
|---|---|
| Acute MI (NSTEMI / STEMI) | Crushing central pain radiating to the jaw or arm, sweating, exertional onset, risk factors |
| Unstable / stable angina | Exertional pain; unstable when at rest or on a worsening pattern |
| Pericarditis | Pleuritic, positional pain (better sitting forward), often after a viral illness |
| Myocarditis | After a viral illness; chest pain with breathlessness; can mimic MI |
| Acute pulmonary edema | Severe breathlessness, orthopnoea, frothy sputum, crackles |
| Atrial fibrillation / PSVT | Palpitations, irregular (AF) or sudden regular (PSVT) tachycardia |
| Pulmonary embolism | Pleuritic pain, breathlessness, one swollen leg, immobilisation or surgery |
| Spontaneous pneumothorax | Sudden pleuritic pain and breathlessness |
| Boerhaave syndrome | Severe pain after forceful vomiting; rare |
| GERD | Burning, after meals, worse lying flat |
| Panic attack | Palpitations, hyperventilation, tingling, situational |
| Aortic dissection | Tearing pain radiating to the back; blood-pressure difference between the arms |

A history rarely settles the question alone. The confirmatory tests the system refers to are the
troponin (a marker of heart-muscle damage), the ECG, the D-dimer (sensitive but not specific, used
to rule pulmonary embolism out) and the chest X-ray (`docs/09` §1.2). For example, no history can
separate unstable angina from a myocardial infarction; only the troponin can (EXP-019,
interpretation 7). This is why the system ranks and flags but never concludes.

**Must-not-miss conditions.** Eight of the 14 conditions in scope are must-not-miss: MI, unstable
angina, pulmonary embolism, spontaneous pneumothorax, Boerhaave, myocarditis, acute pulmonary
edema and aortic dissection (Table 1.1). DDXPlus's own severity ranking of its conditions
independently agrees with this set (`docs/10`).

**Alarm fatigue.** Raising a flag for danger costs little once; raising it for half of all
patients teaches users to ignore flags (R-15) [citation needed: alarm fatigue in clinical
decision support]. A rule set must therefore be judged both on whom it reaches and on whom it
flags without reason (§5.3).

## 2.2 Clinical prediction rules behind the red flags

Each red-flag rule follows a published clinical pattern rather than any single suggestive finding
(`docs/04` §3).

- **The ADD-RS** (aortic dissection detection risk score), from the 2010 ACCF/AHA thoracic aortic
  disease guideline (Hiratzka et al. 2010), groups its markers into three categories: high-risk
  conditions (connective tissue disease, a family history of aortic disease, known aortic valve
  disease, recent aortic manipulation, a known thoracic aortic aneurysm), high-risk pain features
  (abrupt onset, severe intensity, ripping or tearing character), and high-risk examination
  features (a pulse deficit, an inter-arm blood-pressure difference, a focal neurological deficit,
  a new aortic regurgitation murmur, hypotension or shock). Rogers et al. (2011) validated it on
  patients from the International Registry of Acute Aortic Dissection (IRAD). The IRAD reports
  (Hagan et al. 2000; Evangelista et al. 2016) give the frequency of many of these findings, which
  the graph uses as edge weights (§3.3).
- **The Wells criteria** for pulmonary embolism [citation needed: Wells criteria, original
  publication]. The rule uses the risks a history can record: signs of deep-vein thrombosis,
  immobilisation, recent surgery and previous thrombosis (`src/reasoning/red_flags.py`).
- **The Braunwald classification** of unstable angina [citation needed: Braunwald
  classification, original publication]. The rule uses its history features: a crescendo pattern,
  or pain at rest with an ischaemic character.

## 2.3 Synthetic patients and the DDXPlus benchmark

**DDXPlus** (Tchango et al., NeurIPS 2022, arXiv:2205.09148) is a large synthetic dataset for
automatic medical diagnosis: about 1.3 million generated patients across 49 pathologies, each
with age, sex, a list of coded evidence (symptoms and antecedents), the presenting complaint, the
true pathology, and a ranked differential diagnosis with probabilities (`docs/03` §1.1). The
patients are generated from a curated knowledge base of conditions and their evidence
(`release_conditions.json`), and the English text is machine-translated from French (EXP-001).
Chapter 3 describes what this generation process implies for evaluation; two facts matter here.
First, each patient's evidence is drawn only from their own condition's list, so on full
histories a classifier can be near-perfect (EXP-004, R-16). Second, the knowledge graph in this
project is built from that same knowledge base, so any graph figure measured on DDXPlus patients
is circular (R-12).

The published deep models on DDXPlus are multilayer perceptrons and transformers (`docs/14`)
[citation needed: published deep models on DDXPlus]. The project's literature review found no
study of a fixed deep classifier on randomly masked DDXPlus histories, or of its cost (`docs/14`)
[citation needed: the literature review of 2026-10-11 is not in the repository].

**Other synthetic and open sources.** Synthea (Walonoski et al., MITRE) [citation needed: year and
venue] generates synthetic patient records, including cardiac modules; it is planned for the
patient knowledge graph and pipeline testing (Phase 1d, not yet built). The UCI Heart Disease
dataset (Janosi, Steinbrunn, Pfisterer & Detrano, UCI ML Repository) [citation needed: year] holds
real de-identified cardiac features, but it is a binary disease classifier and cannot rank a
differential, so it is at most a stretch-goal risk sub-model (`README.md`). PTB-XL (Wagner et al.)
[citation needed: year and venue] is a public ECG dataset, a stretch goal not pursued.

## 2.4 Knowledge graphs for clinical reasoning

A knowledge graph encodes medical relations explicitly, for example that chest pain radiating to
the jaw is associated with myocardial infarction, so a system can show a reasoning path rather
than a score (`docs/00` §2) [citation needed: knowledge graphs in clinical decision support].
Several open biomedical graphs exist: Hetionet (CC0) and PrimeKG, and terminologies such as HPO,
UMLS and SNOMED CT (`README.md`) [citation needed: Hetionet, PrimeKG, HPO]. They are broad, and
none was chosen as the backbone here.

**BODHI-S** (Eka Care, `ekacare/BODHI-S` on Hugging Face; CC-BY-NC-4.0) is a condition–symptom
knowledge graph whose facts carry qualifiers (severity, onset, location, radiation) and
likelihood bands, with conditions identified by SNOMED CT (`docs/03` §1.2). It was the planned
backbone. A feasibility spike in week 1 found that only 4 of our 13 conditions matched it exactly
(31%, against a 60% gate), because it covers 555 conditions across all of medicine and merges
distinctions the differential needs (one node for both anginas, one for AF and PSVT; EXP-001,
`docs/10`). It is used to enrich four conditions instead (§3.3).

**Vocabulary mapping.** The project's own concepts and DDXPlus's coded answers are linked by a
crosswalk whose entries use the SKOS mapping relations (exact, close, broader, narrower, related),
read as in the W3C SKOS Primer §4.5 (`docs/09` §2), so that each relation allows only the
inferences that are logically sound (§4.4).

**Scoring a graph.** Two families of graph score were compared for this project: Personalised
PageRank [citation needed: Personalised PageRank] and a naive-Bayes log-likelihood
[citation needed: naive Bayes for diagnosis]. Naive Bayes was chosen (EXP-005, §4.5).

## 2.5 Combining probabilistic opinions

When two systems each give a probability distribution over the same conditions, a
**logarithmic opinion pool** combines them as a weighted geometric mean, renormalised (Genest and
Zidek 1986, as cited in `src/fusion/pool.py`) [citation needed: full bibliographic details]. It is
also known as a product of experts [citation needed: product of experts]. A pool multiplies, so
one expert's near-zero probability vetoes a condition whatever the other says; Chapter 4 explains
how the system guards against that (§4.10).

## 2.6 Calibration

A model saying "70%" should be right about 70% of the time (`docs/09` §1.3). Calibration is
measured by the expected calibration error (ECE), the Brier score and reliability diagrams
(`docs/05` §3.3) [citation needed: ECE, Brier score]. The methods compared here (EXP-007) are:

- **Platt scaling** (Platt 1999), a logistic fit to a model's scores, here one-vs-rest per
  condition (Zadrozny and Elkan 2002);
- **isotonic regression** (Zadrozny and Elkan 2002), a monotone, non-parametric fit;
- **temperature scaling** (Guo et al. 2017), Platt scaling's one-parameter multi-class form,
  which keeps the order of the conditions.

The full bibliographic details of these three references are not in the repository
[citation needed: titles and venues of Platt 1999, Zadrozny and Elkan 2002, Guo et al. 2017].

## 2.7 Deep learning on tabular data with missing answers

The deep ranker is an **FT-Transformer** (Gorishniy et al. 2021, "Revisiting deep learning models
for tabular data", as cited in `src/ml/deep.py`), at a smaller size and with the reference
implementation's (rtdl) defaults [citation needed: venue of Gorishniy et al. 2021; the rtdl
library]. Each input feature becomes a token, and attention lets every finding see every other.
That suits DDXPlus, where findings interact: exertional pain means one thing beside rest pain and
another without it (`src/ml/deep.py`). Its feed-forward layers use ReGLU [citation needed:
gated linear unit variants], and it is trained with AdamW [citation needed: AdamW].

When inputs are missing, the best predictor is in general non-linear in both what was observed
and what was asked (NeuMiss, 2020, as cited in `docs/14`) [citation needed: authors, title and
venue of NeuMiss]. In DDXPlus an unlisted question means "no", so "not asked" and "denied" are
two different states that a feature encoding must keep apart (`docs/01` FR-1.3; R-18). This project
does so with an explicit "asked" channel and masked training copies (§4.6).

**Explaining a model's ranking.** The system explains the model by leave-one-finding-out
occlusion: remove each recorded finding, rescore, and see how each condition's odds move
(`src/ml/reasons.py`) [citation needed: occlusion-based attribution]. SHAP was planned
(`docs/02`, `README.md`) [citation needed: SHAP] but is not used.

**Language models and retrieval.** Retrieval-augmented generation grounds each explanatory
statement in a retrieved, citable passage (`docs/00` §2) [citation needed: retrieval-augmented
generation]. This is Phase 3 work [pending: retrieval EXP-011] [pending: LLM baselines B3/B4,
EXP-009/010].

**Response time.** About 0.1 s is the conventional limit at which a response feels instant
(Nielsen's limits, as cited in EXP-022) [citation needed: Nielsen's response-time limits]. No
clinical standard sets a number (EXP-022).

## 2.8 References held in the repository

The list below is everything this chapter cites that the repository already records, with the
detail it records. Entries marked ⚑ need their full details checked before submission.

- Evangelista A, Maldonado G, Gruosso D, et al. Insights from the International Registry of Acute
  Aortic Dissection. *Glob Cardiol Sci Pract.* 2016;2016(1):e201608. doi:10.21542/gcsp.2016.8.
  (`src/medical_kg/hand_authored.py`)
- Genest and Zidek, 1986 (cited for the logarithmic opinion pool; title not recorded). ⚑
  (`src/fusion/pool.py`)
- Gorishniy et al. 2021. "Revisiting deep learning models for tabular data". ⚑
  (`src/ml/deep.py`)
- Guo et al. 2017 (cited for temperature scaling; title not recorded). ⚑
  (`src/fusion/calibration.py`)
- Hagan PG, Nienaber CA, Isselbacher EM, et al. The International Registry of Acute Aortic
  Dissection (IRAD): new insights into an old disease. *JAMA.* 2000;283(7):897-903.
  (`src/medical_kg/hand_authored.py`)
- Hiratzka LF, Bakris GL, Beckman JA, et al. 2010 ACCF/AHA/AATS/ACR/ASA/SCA/SCAI/SIR/STS/SVM
  guidelines for the diagnosis and management of patients with thoracic aortic disease.
  *Circulation.* 2010;121(13):e266-e369. (`src/medical_kg/hand_authored.py`)
- Janosi, Steinbrunn, Pfisterer & Detrano. UCI Heart Disease, UCI ML Repository. ⚑ (`README.md`)
- NeuMiss, 2020 (authors and title not recorded). ⚑ (`docs/14`)
- Nielsen's response-time limits (no source recorded). ⚑ (EXP-022)
- Platt 1999 (cited for Platt scaling; title not recorded). ⚑ (`src/fusion/calibration.py`)
- Rogers AM, Hermann LK, Booher AM, et al. Sensitivity of the aortic dissection detection risk
  score. *Circulation.* 2011;123(20):2213-2218. (`src/medical_kg/hand_authored.py`)
- Eka Care. BODHI-S, condition–symptom knowledge graph, `ekacare/BODHI-S`, Hugging Face,
  CC-BY-NC-4.0. ⚑ (`README.md`, `docs/03`)
- Tchango et al. DDXPlus: A New Dataset for Automatic Medical Diagnosis. NeurIPS 2022,
  arXiv:2205.09148. ⚑ (`README.md`, `docs/03`)
- Wagner et al. PTB-XL, a large publicly available ECG dataset. PhysioNet. ⚑ (`README.md`)
- Walonoski et al. Synthea: synthetic patient generator. MITRE. ⚑ (`README.md`)
- W3C SKOS Primer, §4.5 (mapping properties). ⚑ (`docs/09` §2)
- Zadrozny and Elkan 2002 (cited for one-vs-rest Platt scaling and isotonic regression; title not
  recorded). ⚑ (`src/fusion/calibration.py`)
- Also named without a citation: Hetionet, PrimeKG, PyKEEN, scispaCy, medspaCy (`README.md`); the
  Wells criteria and the Braunwald classification (`docs/04` §3); McNemar's test, the bootstrap and
  the Clopper–Pearson interval (`docs/05` §6, EXP-004).
