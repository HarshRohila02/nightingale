# 01 — Software Requirements Specification

**Project:** Nightingale — Clinical Intelligence & Reasoning System
**Version:** 1.0 · 2026-09-17
**Standard:** adapted from IEEE 830

---

## 1. Introduction

### 1.1 Purpose
Specifies functional and non-functional requirements for Nightingale, a clinical decision-support
prototype that produces an explained, ranked differential diagnosis for **acute chest pain**.
Audience: development team, project supervisor, evaluators.

### 1.2 Scope
Nightingale accepts a chest-pain patient case, constructs a patient knowledge graph, ranks the 13
in-scope conditions using a machine-learning model fused with cardiac knowledge-graph reasoning,
flags must-not-miss conditions, retrieves supporting medical evidence, and presents an explained
result to a clinician. **Nightingale does not diagnose.** The clinician is the decision-maker.

### 1.3 Definitions

| Term | Meaning |
|---|---|
| **Differential diagnosis** | A ranked list of candidate conditions, not a single answer |
| **Patient KG** | Graph of one patient's findings, built per case |
| **Medical KG** | Static graph of cardiac condition↔symptom knowledge |
| **Must-not-miss** | A condition where a missed/low ranking risks death |
| **Red flag** | A rule-triggered alert for a must-not-miss pattern |
| **Finding** | One clinical fact with assertion status (present/absent/unknown) |
| **Grounded claim** | A statement traceable to a retrieved source passage |
| **Calibration** | Agreement between stated confidence and observed frequency |

### 1.4 References
`README.md`; [00-project-charter.md](00-project-charter.md);
[02-architecture.md](02-architecture.md); [05-evaluation-protocol.md](05-evaluation-protocol.md).

---

## 2. Overall Description

### 2.1 Product perspective
A self-contained research prototype. No integration with hospital systems, EHRs, or external
clinical services. All data is synthetic or open-licensed.

### 2.2 User classes

| User | Description | Technical skill |
|---|---|---|
| **Clinician (primary)** | Qualified medical staff evaluating a chest-pain case | Low — UI must be self-explanatory |
| **Researcher (team)** | Runs experiments, evaluates, ablates | High |
| **Evaluator / supervisor** | Assesses the project | Medium |

### 2.3 Operating environment
Python 3.11 (team standard; the code requires ≥ 3.10); Neo4j on AuraDB Free (cloud; a local Docker
instance is optional); a self-hosted Ollama LLM; Streamlit UI in a modern browser;
Linux/Windows/macOS. Development and the demo run on laptops. Heavy jobs (training, tuning, and
batch LLM or embedding runs) run on free cloud tiers or a university GPU, and the project owner
places each one (decision D-7, 2026-09-18; [11-compute-runbook.md](11-compute-runbook.md)).

### 2.4 In-scope conditions (13)

**Cardiac:** Possible NSTEMI/STEMI · Unstable angina · Stable angina · Pericarditis · Myocarditis ·
Acute pulmonary edema · Atrial fibrillation · PSVT
**Non-cardiac mimics:** Pulmonary embolism · Spontaneous pneumothorax · Boerhaave · GERD ·
Panic attack
**KG-only (not in training data):** Aortic dissection — red-flag rule only.

### 2.5 Constraints
- No paid API; LLM runs locally.
- **BODHI-S is CC-BY-NC-4.0** — the project must remain non-commercial.
- No real patient data under any circumstances.
- 8–10 week delivery window.

### 2.6 Assumptions and dependencies
- DDXPlus coded evidences can be mapped to medical-KG symptom concepts *(under verification — see
  [07-risk-register.md](07-risk-register.md) R-01)*.
- PubMed/PMC remain publicly accessible within rate limits.

---

## 3. Functional Requirements

Priority: **M** = must have (MVP) · **S** = should have · **C** = could have (stretch).

### 3.1 Case intake

| ID | Requirement | Priority |
|---|---|---|
| FR-1.1 | Accept a structured case: age, sex, symptoms with qualifiers, risk factors, vitals | M |
| FR-1.2 | Accept free-text clinical narrative as an alternative/supplement | S |
| FR-1.3 | Allow a finding to be marked **present, absent, or unknown** | M |
| FR-1.4 | Validate and reject malformed cases with a clear error | M |
| FR-1.5 | Load a synthetic case from Synthea or DDXPlus for demo/testing | M |

> FR-1.3 is not optional. "Patient denies leg swelling" is diagnostically different from "leg
> swelling not asked" — conflating them will mis-rank pulmonary embolism.

### 3.2 Clinical NLP

| ID | Requirement | Priority |
|---|---|---|
| FR-2.1 | Extract clinical entities (symptoms, risk factors) from free text | S |
| FR-2.2 | Detect negation and assign assertion status | S |
| FR-2.3 | Detect temporality (current vs historical) | C |
| FR-2.4 | Normalise extracted terms to medical-KG concept identifiers | S |

### 3.3 Patient knowledge graph

| ID | Requirement | Priority |
|---|---|---|
| FR-3.1 | Construct a patient KG of findings, qualifiers and assertion status | M |
| FR-3.2 | Preserve provenance (which input produced each node) | S |
| FR-3.3 | Discard/isolate the patient KG after the session | M |

### 3.4 Diagnostic ranking

| ID | Requirement | Priority |
|---|---|---|
| FR-4.1 | Produce an ML score for each of the 13 conditions | M |
| FR-4.2 | Produce a KG score from patient↔medical graph connectivity | M |
| FR-4.3 | Fuse both into a single ranked differential | M |
| FR-4.4 | Return conditions ordered by fused score with per-signal scores retained | M |
| FR-4.5 | Calibrate scores before presenting any value as a probability | S |

### 3.5 Reasoning & explanation

| ID | Requirement | Priority |
|---|---|---|
| FR-5.1 | For each candidate, list **supporting** findings (present and expected) | M |
| FR-5.2 | For each candidate, list **missing** findings (expected, not recorded) | M |
| FR-5.3 | For each candidate, list **contradicting** findings (absent but expected, or present and inconsistent) | M |
| FR-5.4 | Expose the KG reasoning path for each supporting finding | M |
| FR-5.5 | Suggest the next test/question that best discriminates the top candidates | C |

### 3.6 Safety

| ID | Requirement | Priority |
|---|---|---|
| FR-6.1 | Evaluate red-flag rules for every case, independent of ML ranking | M |
| FR-6.2 | Surface a red flag prominently even when the condition ranks low | M |
| FR-6.3 | Support aortic dissection via KG rule despite absence from training data | M |
| FR-6.4 | Display a decision-support disclaimer on every result | M |
| FR-6.5 | Never output a treatment or drug recommendation | M |
| FR-6.6 | Log inputs, outputs, scores and rule firings for audit | S |

### 3.7 Evidence retrieval (RAG)

| ID | Requirement | Priority |
|---|---|---|
| FR-7.1 | Retrieve relevant passages from the PubMed/PMC corpus per candidate | S |
| FR-7.2 | Rank/rerank retrieved passages by relevance | S |
| FR-7.3 | Return citations (title, identifier, link) with every passage | S |
| FR-7.4 | Degrade gracefully to KG-only explanations if retrieval fails | M |

### 3.8 LLM explanation

| ID | Requirement | Priority |
|---|---|---|
| FR-8.1 | Generate a readable explanation for the ranked differential | S |
| FR-8.2 | Constrain generation strictly to retrieved evidence and KG facts | S |
| FR-8.3 | Flag or suppress any claim not supported by a retrieved passage | S |
| FR-8.4 | Never introduce a condition outside the 13 in-scope set | S |
| FR-8.5 | Treat all free-text input as untrusted (prompt-injection resistant) | S |

### 3.9 Interface

| ID | Requirement | Priority |
|---|---|---|
| FR-9.1 | Dashboard for case entry and result display | M |
| FR-9.2 | Display ranked differential with confidence and red flags | M |
| FR-9.3 | Display ✓ supporting / ? missing / ✗ contradicting per candidate | M |
| FR-9.4 | Display evidence with citations | S |
| FR-9.5 | Expose a REST API (`POST /diagnose`) | M |

### 3.10 Evaluation

| ID | Requirement | Priority |
|---|---|---|
| FR-10.1 | Compute all metrics defined in the evaluation protocol | M |
| FR-10.2 | Run all five ablation configurations reproducibly | M |
| FR-10.3 | Log every run with parameters and metrics | M |

---

## 4. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | **Performance** | End-to-end response < 5 s without LLM; < 20 s with local LLM |
| NFR-2 | **Explainability** | Every ranking must be traceable to findings and/or KG paths — no unexplained output |
| NFR-3 | **Safety** | Must-not-miss recall@3 ≥ 0.95 target; red-flag rules evaluated independently of ML |
| NFR-4 | **Reliability** | Component failure (Neo4j, LLM, retrieval) degrades gracefully, never crashes the result |
| NFR-5 | **Reproducibility** | Fixed seeds, pinned dependencies, scripted data download; any run re-creatable from logs |
| NFR-6 | **Usability** | A clinician unfamiliar with the system can interpret a result without training |
| NFR-7 | **Maintainability** | Typed interfaces; modules independently testable; ≥70% coverage on core logic |
| NFR-8 | **Security/Privacy** | Synthetic data only; no PII; free text treated as untrusted |
| NFR-9 | **Portability** | Runs on Linux/Windows/macOS via documented setup |
| NFR-10 | **Licensing** | Non-commercial use only while BODHI-S is a dependency; attributions retained |

---

## 5. Use Cases

### UC-1 — Rank a chest-pain case (primary)
**Actor:** Clinician · **Pre:** system running, KG loaded, model trained
1. Clinician enters demographics, symptoms with qualifiers, risk factors, vitals
2. System builds the patient KG
3. System computes ML and KG scores and fuses them
4. System evaluates red-flag rules
5. System retrieves evidence for top candidates
6. System generates a grounded explanation
7. System displays ranked differential, red flags, ✓/?/✗ findings, evidence, disclaimer

**Post:** clinician has an explained differential. **Alt:** retrieval/LLM unavailable → KG-only
explanation (FR-7.4). **Exception:** invalid case → validation error (FR-1.4).

### UC-2 — Investigate a ranking
Clinician selects a candidate → system shows KG reasoning paths, supporting/missing/contradicting
findings, per-signal scores and cited evidence.

### UC-3 — Red flag on a low-ranked condition
Patient reports tearing pain radiating to the back. Aortic dissection is absent from training data,
so the ML score is zero; the KG rule fires and the UI surfaces the red flag regardless of rank.

### UC-4 — Run the evaluation suite
Researcher runs the harness → all five configurations execute → metrics logged → ablation table
produced.

---

## 6. External Interfaces

**API:** `POST /diagnose` (PatientCase → DiagnosisResult), `GET /conditions`, `GET /health`.
**Data:** DDXPlus, BODHI-S, Synthea, UCI Heart, PubMed/PMC — see
[03-data-management.md](03-data-management.md).
**Schemas:** defined in [02-architecture.md](02-architecture.md) §4 and implemented in
`src/contracts.py`.

---

## 7. Requirements Traceability

| Objective (Charter) | Requirements |
|---|---|
| O1 Patient KG | FR-3.1–3.3, FR-2.* |
| O2 Medical KG | FR-4.2, FR-5.4 |
| O3 ML ranker | FR-4.1, FR-4.4 |
| O4 Fusion | FR-4.3, FR-4.5 |
| O5 Red flags | FR-6.1–6.3, NFR-3 |
| O6 Evidence | FR-7.*, FR-8.2–8.3 |
| O7 Ablations | FR-10.* |
| O8 Interface | FR-9.* |
