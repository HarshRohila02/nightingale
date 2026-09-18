# 03 — Data Management & Licensing Plan

**Version:** 1.0 · 2026-09-17 · **Owner:** P2

> **Two rules that override everything else in this document:**
> 1. **No real, identifiable patient data enters this project. Ever.** All data is synthetic or
>    open-licensed.
> 2. **Nothing in `data/` is ever committed.** It is gitignored. Data is reproduced by running
>    `scripts/download_data.py`, not by cloning.

---

## 1. Dataset register

Verified on Hugging Face 2026-09-17. **Re-verify licences before the final report** — mirrors and
terms change.

| # | Dataset | Identifier | Licence | Role |
|---|---|---|---|---|
| 1 | **DDXPlus** | `aai530-group6/ddxplus` | **CC-BY-4.0** | Primary ML training + differential ground truth |
| 2 | **BODHI-S** | `ekacare/BODHI-S` | **CC-BY-NC-4.0** ⚠️ | Cardiac medical KG seed |
| 3 | **UCI Heart Disease** | `MLLab-TS/heart_disease_uci` | Open (verify mirror) | Cardiac-risk sub-model / real-data sanity check |
| 4 | **Synthea** | `synthetichealth/synthea` | Free of restrictions | Synthetic patient records, Patient KG testing |
| 5 | **PubMed / PMC** | NCBI E-utilities | Open (PMC OA subset redistributable) | RAG evidence corpus |
| 6 | **PTB-XL** *(stretch)* | PhysioNet `ptb-xl` | CC-BY (verify) | ECG multimodal extension |

### 1.1 DDXPlus — the backbone
- 1.03M train / 135k validate / 132k test synthetic patients; 49 pathologies.
- Columns: `AGE`, `SEX`, `PATHOLOGY` (ground truth), `DIFFERENTIAL_DIAGNOSIS` (**ranked list with
  probabilities**), `EVIDENCES`, `INITIAL_EVIDENCE`.
- **Evidences are coded** (`E_54_@_V_161`). Decoding via the release evidence-mapping file is a
  prerequisite for all modelling — see `scripts/decode_ddxplus.py` (the vocabulary) and
  `scripts/build_ddxplus_chestpain.py` (the patient rows, §2.1).
- We filter to the **13 in-scope chest-pain conditions**; all other pathologies are dropped.
- *Attribution:* Tchango et al., DDXPlus (NeurIPS 2022), arXiv:2205.09148.

### 1.2 BODHI-S — licence constraint ⚠️
- Ships **two line-aligned files of 13,204 rows each** (verified 2026-09-17):
  - `data/triples.jsonl` — structured edges:
    `{head: <symptom UUID>, head_type: "Symptom", relation: "PRESENT_IN", tail: "<SNOMED CT id>",
    tail_type: "Condition", properties: {likelihood_condition_given_symptom,
    likelihood_symptom_given_condition}}`. The likelihood qualifiers (`rare`/`medium`/`high`) are
    directly usable as KG edge weights and DDXPlus has no equivalent.
  - `data/nl_facts.jsonl` — the same facts as English sentences
    (`Chest pain <radiate> to jaw (Symptom) is a symptom present in Acute myocardial infarction
    (Condition).`). Because the files are line-aligned, `nl_facts[i]` decodes the UUID/SNOMED ids
    in `triples[i]`.
- Conditions are identified by **SNOMED CT** codes; DDXPlus uses **ICD-10**. Joining the two is a
  manual, 13-row mapping — not a general terminology problem at this scale.
- **Coverage is limited for our scope**: BODHI-S spans 555 conditions across all of medicine and
  matches only 4 of our 13 exactly. It is therefore an *enrichment* source, not the KG backbone —
  see [10-spike-r01-crosswalk.md](10-spike-r01-crosswalk.md).
- **CC-BY-NC-4.0 — NON-COMMERCIAL.** Consequences, which are binding:
  - Academic/research use: permitted.
  - Any commercial use, product, or paid service built on it: **prohibited**.
  - **Attribution to Eka Care is mandatory** wherever the KG is used or described.
  - Derivative artifacts (the built KG) inherit the restriction — the Neo4j database we build from
    it may not be commercialised or redistributed without checking the licence terms.
- If the project is ever taken commercial, BODHI-S must be replaced (Hetionet, CC0, is the
  candidate substitute).

### 1.3 Synthea
Generator, not a fixed dataset. Generated output is explicitly free of cost, privacy, and security
restrictions. Use the cardiac modules; fix the seed and record it for reproducibility.

### 1.4 PubMed / PMC
- Access via NCBI E-utilities; **respect rate limits** (3 req/s without an API key, 10 with).
- Only the **PMC Open Access subset** may be redistributed. Abstracts retrieved via E-utilities are
  for our local index only — **do not commit the corpus to the repository.**
- Fetching is slow: start the download in Week 2 even though it is used in Week 6.

---

## 2. Storage layout

```
data/                        # gitignored in its entirety
├── raw/                     # exactly as downloaded, never edited
│   ├── ddxplus/
│   ├── bodhi_s/
│   ├── uci_heart/
│   └── pubmed/
├── interim/                 # decoded / parsed, pre-feature
│   ├── ddxplus_evidences.json                  # decoded evidence vocabulary (decode_ddxplus.py)
│   ├── ddxplus_chestpain_conditions.json       # the 13 conditions' evidence sets: the KG source
│   ├── ddxplus_chestpain_<split>.parquet       # the 13-condition filter, one row per patient (§2.1)
│   ├── ddxplus_chestpain_<split>.summary.json  # its counts and label audit (EXP-013)
│   ├── cardiac_kg_summary.json                 # the KG card's numbers (build_cardiac_kg.py)
│   └── bodhi_cardiac_triples.jsonl
├── processed/               # model-ready
│   ├── train.parquet  val.parquet  test.parquet
│   └── symptom_crosswalk.csv         # DDXPlus evidence ↔ KG concept
└── external/                # KG dumps, FAISS indices
```

**`raw/` is immutable.** Every transformation is a script in `scripts/` or `src/`, so the chain
from download to result is reproducible.

Only the in-scope subset of DDXPlus is materialised; the plan's `ddxplus_decoded.parquet` (all 49
pathologies) is not built. It would be needed only for the R-13 stretch goal, an out-of-scope
signal trained on the other 36 pathologies. There is one parquet per split, so the test file cannot
exist until Phase 4 opens that split.

### 2.1 Dataset card — `ddxplus_chestpain_<split>.parquet` · *added 2026-09-18*

Built by `python scripts/build_ddxplus_chestpain.py [--split train|validate]`. The decoding logic
is in `src/ddxplus.py`, which is covered by `tests/test_ddxplus.py`. One row per patient whose
`PATHOLOGY` is one of the 13 conditions. The builder refuses `--split test` unless
`evaluation.allow_test_split` in `configs/config.yaml` is `true`. That switch is flipped once, in
Phase 4.

| Column | Role | Type | Meaning |
|---|---|---|---|
| `case_id` | metadata | str | `ddxplus-<split>-<row>`: stable, and traceable to the raw CSV |
| `split` | metadata | str | `train` or `validate` (`test` only in Phase 4) |
| `source_row` | metadata | int | 0-based row number in the raw CSV |
| `initial_evidence` | metadata | str | The presenting complaint. It is always also in `evidences`. **Not an input**, because docs/05 §2 lists only AGE, SEX and EVIDENCES |
| `age` | **input** | int | 0–109 |
| `sex` | **input** | str | `M` or `F`. Synthetic: about 50% female for every condition (EXP-002) |
| `evidences` | **input** | list[str] | The raw tokens, in DDXPlus order, with nothing dropped |
| `positive_codes` | **input** | list[str] | Evidence *questions* answered with something other than "no". Derived from `evidences` alone |
| `label_condition_id` | **label** | str | `COND:*` id of `PATHOLOGY`: the top-1 ground truth |
| `label_pathology` | **label** | str | The raw DDXPlus `PATHOLOGY` string |
| `label_differential` | **label** | list[struct] | `DIFFERENTIAL_DIAGNOSIS` as `{pathology, condition_id, probability}`. Out-of-scope entries are kept, with `condition_id = null` |

**Never train on a `label_*` column.** The prefix makes the leakage rule in §4 mechanical: feature
code selects `INPUT_COLUMNS` from `src/ddxplus.py`, and never "every column except the label".

**Tokens that mean "no".** A token being listed does not mean the finding is present. Every
categorical, multi-choice and ordinal evidence has a default value that means "no". `E_204_@_V_10`
("travelled abroad: N") is listed for 89.6% of patients, and `E_57_@_V_123` means "the pain radiates
nowhere". Use `positive_codes`, never "the code appears in `evidences`". An ordinal at its default of
0 is not counted as positive, although for `E_59` ("how fast did the pain appear?") 0 can be a real
answer. The raw value stays in `evidences` for the ML features either way.

**Validate split, built 2026-09-18:**

| Measure | Value |
|---|---|
| Rows | **33,963** of 132,448 (25.6%). Per-condition counts match EXP-002 exactly |
| Tokens | 743,822: 295,726 binary · 353,233 categorical value · 94,062 numeric ordinal · 801 NA |
| Tokens that mean "no" | 45,049, in 31,189 patients |
| Positive codes per patient | 14.1 on average |
| File size | 4.6 MB |

The 801 NA tokens (`E_54_@_V_11`, EXP-002's "unknown" tokens) belong to patients with **no pain at
all**. None of them lists `E_53` ("pain related to the consultation"), and 771 are PSVT. Their pain
questions are simply filled with defaults. The label audit is **EXP-013** in
[08-experiment-log.md](08-experiment-log.md).

---

## 3. Reproducibility

- `scripts/download_data.py` fetches everything; re-runnable and idempotent.
- Record dataset **revision/commit hash** for each HF dataset in `data/raw/MANIFEST.json`.
- Fix seeds (`RANDOM_SEED = 42`) for splits, Synthea generation, and model training.
- Pin dependency versions in `requirements.txt`.
- **DVC is deliberately not used** — HF datasets are already versioned and deterministic download
  scripts are sufficient at this scale. Revisit only if data volume becomes unmanageable.

---

## 4. Splits and leakage policy

| Rule | Reason |
|---|---|
| Use DDXPlus's **provided** train/validate/test splits | Comparable to published results |
| Split at **patient/case level**, never at row level | Prevents the same case appearing in train and test |
| **The test set is opened once**, at Phase 4 | Repeated peeking invalidates the study |
| All tuning uses **validation only** | Includes fusion weights and calibration |
| Class balance reported per condition, per split | Rare conditions will otherwise be silently invisible |

> **Leakage hazard specific to this project:** DDXPlus `DIFFERENTIAL_DIAGNOSIS` is the ground-truth
> label for ranking. It must **never** be used as an input feature. Only `EVIDENCES`, `AGE`, and
> `SEX` are inputs. Anyone who accidentally trains on the differential column will get excellent
> metrics and a worthless model.

---

## 5. What may and may not be committed

| Artifact | Committed? |
|---|---|
| Download/transform scripts | ✅ Yes |
| `symptom_crosswalk.csv` (small, our own work) | ✅ Yes |
| Schema and config files | ✅ Yes |
| Small golden-case fixtures (hand-written, synthetic) | ✅ Yes |
| Raw or processed datasets | ❌ Never |
| PubMed corpus / FAISS indices | ❌ Never |
| Trained model binaries | ❌ No — regenerate from scripts |
| Anything containing real patient data | ❌ Never (must not exist) |

Enforced by `.gitignore`. If a dataset is ever committed by accident, purge it from history rather
than deleting it in a later commit.

---

## 6. Attribution block

Reproduce in the final report and any public artifact:

> This project uses **DDXPlus** (Tchango et al., NeurIPS 2022, CC-BY-4.0); **BODHI-S** by
> **Eka Care** (CC-BY-NC-4.0, used for non-commercial academic research); **Synthea** (MITRE); the
> **UCI Heart Disease** dataset (Janosi, Steinbrunn, Pfisterer & Detrano, UCI ML Repository); and
> literature retrieved from **PubMed / PubMed Central** (NCBI).

---

## 7. Compliance checklist (tick before submission)

- [ ] No real patient data used at any stage
- [ ] `data/` absent from git history
- [ ] All six licences re-verified and recorded with dates
- [ ] Eka Care attribution present in report, README, and UI credits
- [ ] Project stated as non-commercial wherever BODHI-S is referenced
- [ ] NCBI rate limits respected; PMC corpus not redistributed
- [ ] Test set opened only once, at Phase 4
- [ ] `DIFFERENTIAL_DIAGNOSIS` confirmed absent from the feature set
