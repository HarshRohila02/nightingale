# 3 · Data

> **Draft 1, 2026-10-14.** All data are synthetic or open-licensed; no real patient record enters
> the project (`docs/03`, rule 1). Nothing under `data/` is committed (rule 2). Every count below
> comes from the experiment log (`docs/08`) or the dataset cards in `docs/03` §2.

## 3.1 Sources and licences

**Table 3.1 — Data and knowledge sources** (`docs/03` §1, `README.md`).

| Source | Identifier | Licence | Role here | Status |
|---|---|---|---|---|
| DDXPlus | `aai530-group6/ddxplus` (Hugging Face) | CC-BY-4.0 | Training and validation patients; the ground-truth differential; the knowledge-graph backbone (`release_conditions.json`) | In use |
| BODHI-S | `ekacare/BODHI-S` (Hugging Face) | **CC-BY-NC-4.0, non-commercial** | Enrichment of four conditions in the knowledge graph | In use |
| Hand-authored aortic dissection | ADD-RS (Hiratzka et al. 2010; Rogers et al. 2011) and IRAD (Hagan et al. 2000; Evangelista et al. 2016) | Our own work, from published sources | 20 knowledge-graph edges and a red-flag rule | In use |
| Golden clinical cases | `tests/fixtures/golden_cases.yaml` | Our own work, hand-written, synthetic | Executable clinical expectations (§3.6) | In use |
| Synthea | `synthetichealth/synthea` (generator) | Free of restrictions | Patient knowledge graph and pipeline testing | Planned (Phase 1d) [pending: 1d] |
| PubMed / PMC | NCBI E-utilities; PMC Open Access subset | Open; only the PMC OA subset is redistributable | Evidence corpus for retrieval | Planned (Phase 3a) [pending: retrieval EXP-011] |
| UCI Heart Disease | `MLLab-TS/heart_disease_uci` | Open (mirror to verify) | Cardiac-risk sub-model | Stretch goal, not started |
| PTB-XL | PhysioNet `ptb-xl` | CC-BY (to verify) | ECG extension | Stretch goal, not started |

The licences were verified on Hugging Face on 2026-09-17 and must be re-verified before
submission (`docs/03` §1, §7) [pending: licence re-verification, Phase 4e]. The attribution block
that the final report and every public artefact must carry is reproduced in Appendix A (§3.8).

## 3.2 DDXPlus

### 3.2.1 What the dataset is

DDXPlus (Tchango et al. 2022) is a synthetic dataset of generated patients across 49 pathologies.
Its splits hold 1,025,602 training patients and 132,448 validation patients (EXP-002, EXP-003),
and about 135,000 test patients (`README.md`) [to verify: the exact test count]. Each patient has:

- **inputs:** age, sex and a list of coded evidence tokens (`EVIDENCES`);
- **the presenting complaint:** `INITIAL_EVIDENCE`, which is always also in the evidence list;
- **labels:** the true pathology (`PATHOLOGY`) and a ranked differential diagnosis with
  probabilities (`DIFFERENTIAL_DIAGNOSIS`).

The patients are generated from a curated knowledge base of conditions and their evidence,
`release_conditions.json`, with explicit symptoms and antecedents, an ICD-10 code and a severity
ranking for every condition; all 13 of our conditions are in it (EXP-001, `docs/10`).

**The test split stays closed until Phase 4.** It is opened once, after development is frozen
(`docs/05` §2). The parquet builder refuses `--split test` unless `evaluation.allow_test_split` in
`configs/config.yaml` is true, and it is false; there is one parquet per split, so the test file
cannot exist before then (`docs/03` §2, §2.1). `test.csv` has never been downloaded to the laptop
(PROGRESS §7), and a test checks that no cloud notebook fetches it (`tests/test_notebooks.py`).

### 3.2.2 The chest-pain subset

We keep the patients whose true pathology is one of the 13 in-scope conditions: **33,963 of
132,448 validation patients (25.6%)** and **255,900 of 1,025,602 training patients (25.0%)**
(EXP-002, EXP-003). Table 3.2 gives the class balance. The rarest condition, spontaneous
pneumothorax, has 10,162 training patients, and the largest class is 2.70 times the smallest, so
no condition is too rare to learn (risk R-03, resolved).

**Table 3.2 — The 13 conditions in DDXPlus** (validation counts, share, % female and median age
from EXP-002; training counts from EXP-003).

| Condition | Validation | Share | Train | % female | Median age |
|---|---:|---:|---:|---:|---:|
| Pulmonary embolism | 3,725 | 11.0% | 27,468 | 52% | 38 |
| GERD | 3,426 | 10.1% | 25,979 | 52% | 36 |
| Panic attack | 3,237 | 9.5% | 25,019 | 51% | 39 |
| Pericarditis | 3,032 | 8.9% | 22,785 | 51% | 40 |
| Possible NSTEMI / STEMI | 2,943 | 8.7% | 21,260 | 50% | 45 |
| Unstable angina | 2,748 | 8.1% | 21,244 | 54% | 44 |
| Atrial fibrillation | 2,609 | 7.7% | 21,036 | 52% | 36 |
| Acute pulmonary edema | 2,500 | 7.4% | 19,018 | 51% | 47 |
| PSVT | 2,376 | 7.0% | 18,781 | 49% | 40 |
| Stable angina | 2,340 | 6.9% | 16,995 | 52% | 46 |
| Boerhaave syndrome | 2,075 | 6.1% | 15,080 | 51% | 40 |
| Myocarditis | 1,547 | 4.6% | 11,073 | 54% | 35 |
| Spontaneous pneumothorax | 1,405 | 4.1% | 10,162 | 52% | 38 |
| **Total in scope** | **33,963** | 25.6% of validate | **255,900** | | |

Of the 33,963 validation patients, **16,943 have a must-not-miss condition** (EXP-004), which is
the denominator of every must-not-miss figure in Chapter 6. Sex and age were measured on the
validation split only; the training parquet stayed on the cloud runtime, and its summary holds
counts only (EXP-003).

### 3.2.3 The evidence tokens

Evidence is coded, not written: `E_54_@_V_161` is a pain-location answer. The release's evidence
vocabulary has 919 entries, 223 questions and 696 answer values (EXP-001), of which the 13
conditions use 84 questions (`docs/03` §2.2). Tokens come in four forms (EXP-002, `docs/03` §2.1):

**Table 3.3 — Evidence tokens in the chest-pain subset** (`docs/03` §2.1).

| Token form | Example | Validation | Train |
|---|---|---:|---:|
| Binary | `E_91` (fever) | 295,726 | 2,237,663 |
| Categorical value | `E_54_@_V_161` | 353,233 | 2,634,321 |
| Numeric ordinal (0–10) | `E_56_@_4` (pain intensity) | 94,062 | 704,592 (12.6%) |
| The "NA" sentinel | `E_54_@_V_11` | 801 | 6,734 |
| **All tokens** | | **743,822** | **5,583,310** |

Three properties of these tokens shaped the design.

1. **A listed token can mean "no".** Every categorical, multi-choice and ordinal evidence has a
   default value that means "no". `E_204_@_V_10`, "did not travel abroad", is listed for 89.6% of
   validation patients, and `E_57_@_V_123` means "the pain radiates nowhere". The validation split
   holds 45,049 such tokens in 31,189 patients (EXP-013). Reading "the code is listed" as "the
   patient has it" would give 89.6% of patients a travel history, so every component uses
   `positive_codes`, the questions answered with something other than "no" (14.1 per patient on
   average, `docs/03` §2.1).
2. **The 801 NA tokens belong to patients with no pain at all.** None of them lists `E_53` (pain
   related to the consultation), and 771 are PSVT patients (EXP-013).
3. **The English is machine-translated from French, and unreliable.** `déchirante` (tearing, the
   classic dissection descriptor) is rendered "heartbreaking", and `lancinante` (shooting)
   "haunting" (EXP-001, `docs/10`). Every component therefore works on evidence codes, and the
   hand-curated tables quote the French.

### 3.2.4 How DDXPlus was generated, and what that implies

**Each patient's evidence is drawn only from their own condition's evidence list.** Every
validation patient's positive answers lie inside their own condition's evidence set, and for
**91.7%** inside no other condition's (EXP-004). Which questions are answered "yes" therefore
nearly names the condition. Two consequences run through the whole report:

- **Saturation (R-16).** On full histories an ML model can be near-perfect, which leaves the
  headline comparisons no room (§6.2).
- **Circularity (R-12).** The knowledge graph is built from the same knowledge base the patients
  were generated from, so the graph alone recognises DDXPlus patients almost perfectly, and that
  is not skill (§6.5).

Some DDXPlus conventions differ from clinical usage, and we found three of them the hard way:

- "Tearing" pain is listed for 76% of pneumothorax and 76% of Boerhaave patients, and for nobody
  else (EXP-014, EXP-008); a tearing-pain rule therefore fires on two other must-not-miss
  emergencies.
- "Chest pain even at rest" is listed for 56% of pneumothorax patients (EXP-008).
- Radiation to the jaw (`E_57_@_V_121`) is given only by acute pulmonary edema patients, 1,316 of
  them (53%). DDXPlus's MI and angina patients radiate to the biceps, the shoulders, the thyroid
  cartilage and the thoracic spine (0.57–0.75 each) and under the jaw (`V_163`, 0.37–0.40), never
  to `V_121` or the chin (EXP-021). This is why decision A-10 changed the answer our jaw/arm
  concept stands for (§4.8, §7.4).

DDXPlus also draws some findings independently, so a patient can lack the defining one: 20% of
Boerhaave patients never report the vomiting and 29% of myocarditis patients the viral illness,
and the onset speed of pneumothorax is drawn uniformly over a range (EXP-008). DDXPlus also
records several radiation sites per patient, likely more than real patients report (EXP-014).

### 3.2.5 The label audit (EXP-013) and decision D-8

The ground-truth differential `D` is open-world. It holds 12.0 conditions on average, of which 6.7
are in scope. For **91.8%** of in-scope validation patients (31,185) it includes conditions the
system cannot output, which hold **33.3%** of its probability mass on average (median 34.9%, 90th
percentile 52.4%). The true pathology is always inside its own `D`, and ranked first in it for
71.2% of patients (EXP-013). The most frequent out-of-scope entries come from DDXPlus's
differential generator, not from chest-pain reasoning.

As first written, Recall@5 could not exceed **0.434** even for a perfect 13-condition system, so it
measured the closed-world gap more than the ranking. **Decision D-8** (the team, 2026-09-19,
`docs/05` amendment 1) restricts the headline Precision@3 and Recall@5 to `D_in`, the part of `D`
inside the 13 in-scope conditions. A perfect system then reaches 0.753 on Recall@5, and 0.929 on
Precision@3, which is the same under either definition. Recall@5 on the full `D` is always reported
beside it, so the change cannot flatter a result. The decision was made before any model was
trained. The training split agrees: out-of-scope mass 33.3%, ceilings 0.432 (full `D`) and 0.751
(`D_in`) for Recall@5, and 0.931 for Precision@3 (`docs/03` §2.1).

**`DIFFERENTIAL_DIAGNOSIS` is the label and is never an input** (`docs/03` §4). In the parquet it
is `label_differential`, every label column starts with `label_`, and feature code selects
`INPUT_COLUMNS` (age, sex, evidences) rather than "every column except the label". A test shuffles
the label columns and checks that the feature matrix does not change (`docs/03` §2.2).

### 3.2.6 Synthetic demographics (EXP-002)

The demographics are a simulator's, not a population's. Every one of the 13 conditions is about
50% female; the median age of MI patients is 45, where a first MI in the real world typically
occurs in the patient's 60s [citation needed: age at first myocardial infarction]; and spontaneous
pneumothorax is 52% female, although in reality it is strongly male-predominant [citation needed:
sex distribution of spontaneous pneumothorax] (EXP-002, `docs/04` §6). The training data carry no
real sex or age priors. **A near-equal result by sex is therefore an artefact of generation and
will not be presented as evidence of fairness** (§7.3).

## 3.3 The knowledge graph's sources

The cardiac knowledge graph has three sources, and every edge records which one it came from
(`ddxplus`, `bodhi_s` or `hand_authored`), so the report can separate the graph's shared-source
contribution from its independent ones (R-12; `docs/05` §8 rule 7). Its structure is in §4.3.

### 3.3.1 DDXPlus's knowledge base (the backbone)

`release_conditions.json` links each condition to the evidence **questions** its patients may
answer, with no frequencies: 245 edges (160 symptom, 85 risk-factor), each with weight 1.0, over
84 questions (`docs/02` §5.1). It became the backbone after the feasibility spike (EXP-001):
BODHI-S matched only 4 of our 13 conditions exactly (31%, against a 60% gate), symptom alignment was
28% by content words, and naive string matching gave 0%, which is an artefact, because DDXPlus
evidences are patient-facing questions and BODHI-S symptoms are clinical noun phrases (`docs/10`).

### 3.3.2 BODHI-S (enrichment for four conditions)

BODHI-S (Eka Care) is a condition–symptom knowledge graph of 555 conditions, shipped as two
line-aligned files of 13,204 rows, a structured file of edges keyed by identifiers and a file of
the same facts as sentences, with conditions identified by SNOMED CT (`docs/03` §1.2). Its facts
carry likelihood bands, which DDXPlus has no equivalent for.

It enriches the four conditions it matches exactly: MI, pericarditis, pulmonary embolism and GERD.
Of its 107 facts about them, 98 map to the project's concepts, giving 56 edges (MI 20, GERD 15,
PE 12, pericarditis 9); 7 cannot be mapped and 2 have zero strength; 21 condition–concept pairs
restate a DDXPlus fact and are counted once (EXP-016, `docs/02` §5.1).

**Licence.** BODHI-S is CC-BY-NC-4.0: academic use is permitted, commercial use is not, and
attribution to Eka Care is mandatory wherever the graph is used or described (`docs/03` §1.2). The
built graph inherits the restriction. **The repository holds no BODHI-S text:** the enrichment code
keys facts by BODHI-S's own identifiers, with paraphrased notes, and the graph's card reports counts
only. The team's AuraDB copy holds BODHI-S identifiers and likelihood bands, no text, and stays
private (`docs/03` §1.2). This report quotes no BODHI-S fact.

### 3.3.3 Hand-authored aortic dissection

DDXPlus has no aortic dissection, so neither the DDXPlus graph nor the ML ranker can know it. We
wrote **20 edges** by hand (`src/medical_kg/hand_authored.py`): the 13 markers of the ADD-RS
(Hiratzka et al. 2010; Rogers et al. 2011), plus the pain's location, sharp pain, syncope,
hypertension and cocaine use. Each weight is P(finding | dissection) in the likelihood bands of
decision A-6, taken from the IRAD registry where it reports a frequency (Hagan et al. 2000;
Evangelista et al. 2016) (Table 3.4). Every edge records its evidence and reference.

**Table 3.4 — Likelihood bands of the hand-authored dissection edges** (`docs/02` §5.1; weights
from A-6: rare 0.03, low 0.12, medium 0.35, high 0.65, very high 0.9).

| Band | Findings |
|---|---|
| very high (≥ 80%) | abrupt onset 85% |
| high (50–79%) | chest pain 73%, hypertension 72%, sharp pain about 60%, back pain 53%, tearing pain about 50%, severe pain (no frequency transcribed) |
| medium (20–49%) | aortic regurgitation murmur 32%, hypotension over 25% of type A, pain radiating to the back (no separate figure) |
| low (5–19%) | known aortic aneurysm 16%, pulse deficit 15%, syncope 13%, Marfan syndrome 5%; ADD-RS markers with no published frequency (inter-arm blood-pressure difference, focal neurological deficit, aortic valve disease, family history) |
| rare (< 5%) | iatrogenic 4%, cocaine 1.8% |

In all, 76 of the graph's 321 edges do not come from DDXPlus: 20 hand-authored and 56 from BODHI-S
(R-12).

## 3.4 Storage and reproducibility

- `data/` is gitignored in full: `raw/` holds the downloads unedited, and every transformation is
  a script, so the chain from download to result is reproducible (`docs/03` §2).
- Trained models live in `models/`, also gitignored, saved as JSON (XGBoost as `.ubj`), never as
  pickles, each with the feature fingerprint of its encoder (§4.6).
- Seeds are fixed (42, and 42–46 where `docs/05` §6 requires five), and each run records its commit
  and library versions (`run.json`).
- DVC is deliberately not used: the Hugging Face datasets are versioned, and deterministic download
  scripts suffice at this scale (`docs/03` §3).

## 3.5 The validation split's role

All development, tuning and model selection use the validation split only: the fusion weight
(EXP-006), the calibrators (EXP-007) and D-13's choice of ranker (EXP-021) (`docs/03` §4, `docs/05`
§2). Every figure in Chapter 6 is therefore a validation figure. Because the fusion weight and the
calibrators are chosen on these patients, fused and calibrated validation figures will be
optimistic until the test split is read (EXP-006).

## 3.6 The golden clinical cases

Four hand-written, synthetic cases encode what a clinician would expect the system to do
(`tests/fixtures/golden_cases.yaml`). They run in CI, and if one fails the component is fixed, never
the expectation (`docs/06` §6).

**Table 3.5 — The golden cases.**

| Case | Presentation | Expectation |
|---|---|---|
| GC-001 | Classic acute coronary syndrome: a 58-year-old man with exertional crushing central chest pain radiating to the jaw and left arm, sweating; diabetic smoker | MI in the top 3; an MI red flag |
| GC-002 | Pulmonary embolism: sudden pleuritic chest pain and breathlessness, one swollen leg after recent immobilisation | PE in the top 3; a PE red flag |
| GC-003 | Aortic dissection: tearing pain radiating to the back with an inter-arm blood-pressure difference | Dissection in the top 5; a dissection red flag |
| GC-004 | Benign reflux: burning pain after meals, worse lying flat, ischaemic features explicitly denied | GERD in the top 3; no cardiac red flag |

They have no ground truth beyond the clinical expectation, and every hand-written case reaches the
model through a hand-curated translation (R-17, §4.8). A fifth case, a benign-looking ischaemic
presentation that raises no red flag, is with the team (GC-005) [pending: GC-005, the team].

## 3.7 Compliance checklist

The data checklist of `docs/03` §7 is ticked before submission [pending: Phase 4e]: no real patient
data at any stage; `data/` absent from the git history; all six licences re-verified and dated;
Eka Care's attribution in the report, the README and the UI credits; the project stated as
non-commercial wherever BODHI-S is referenced; NCBI rate limits respected and the PMC corpus not
redistributed; the test set opened only once, in Phase 4; `DIFFERENTIAL_DIAGNOSIS` confirmed absent
from the features.

## 3.8 Appendix A: attribution block

Reproduced from `docs/03` §6, as the final report must carry it:

> This project uses **DDXPlus** (Tchango et al., NeurIPS 2022, CC-BY-4.0); **BODHI-S** by
> **Eka Care** (CC-BY-NC-4.0, used for non-commercial academic research); **Synthea** (MITRE); the
> **UCI Heart Disease** dataset (Janosi, Steinbrunn, Pfisterer & Detrano, UCI ML Repository); and
> literature retrieved from **PubMed / PubMed Central** (NCBI).

*(If Synthea, UCI Heart Disease or PubMed are not used by the time of submission, the block is
trimmed to what was used.)*
