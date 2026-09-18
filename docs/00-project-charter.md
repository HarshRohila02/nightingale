# 00 — Project Charter / Synopsis

**Project:** Nightingale — Clinical Intelligence & Reasoning System
**Type:** University major project (academic research prototype)
**Team size:** 4
**Duration:** 8–10 weeks
**Repository:** `github.com/HarshRohila02/nightingale`
**Status:** Phase 0 — Preparation
**Version:** 1.0 · Created 2026-09-17

---

## 1. Problem Statement

Acute chest pain is one of the most common and highest-stakes presentations in emergency and
primary care. The clinician must rapidly separate life-threatening causes (myocardial infarction,
pulmonary embolism, tension pneumothorax, aortic dissection, oesophageal rupture) from benign ones
(GERD, musculoskeletal pain, panic attack) — often from history and examination alone, before
confirmatory tests return.

Existing computational aids fail clinicians in two distinct ways:

1. **Black-box machine-learning classifiers** output a label or probability with no inspectable
   reasoning. A clinician cannot audit *why* a condition was ranked highly, so cannot safely
   override or trust it.
2. **Large language models** produce fluent clinical-sounding text but hallucinate facts, cannot
   cite their sources reliably, and are poorly calibrated — unacceptable where a wrong ranking can
   kill.

Neither surfaces the two things a clinician actually reasons with: **what evidence supports each
candidate**, and **what information is still missing** to discriminate between them.

## 2. Motivation

A knowledge graph encodes medical relationships *explicitly* — that chest pain radiating to the jaw
is associated with myocardial infarction — so the system can show a reasoning path rather than a
score. Retrieval-augmented generation grounds every explanatory statement in a real, citable
source. Combining these with a machine-learning ranker gives a system that is simultaneously
**accurate** (ML), **explainable** (KG), and **factual** (RAG).

Whether that combination genuinely outperforms its parts is an open, testable question. That
question is this project's research contribution.

## 3. Objectives

| # | Objective | Measurable outcome |
|---|---|---|
| O1 | Build a patient knowledge graph from structured and free-text chest-pain case data | Patient KG constructed for ≥95% of test cases without error |
| O2 | Build a cardiac medical knowledge graph of condition↔symptom relationships | KG covering all 13 in-scope conditions with symptom qualifiers |
| O3 | Train a machine-learning ranker producing a ranked differential | Top-3 accuracy materially above a majority-class baseline |
| O4 | Fuse KG and ML signals into a single explainable ranking | Fusion outperforms both single-method baselines |
| O5 | Detect and surface must-not-miss conditions as red flags | Must-not-miss recall@3 ≥ 0.95 (target) |
| O6 | Retrieve and cite real medical evidence for each candidate | ≥80% of generated claims traceable to a retrieved passage |
| O7 | Quantify the contribution of each component | Complete ablation study across 5 configurations |
| O8 | Deliver a usable clinician-facing interface | Working dashboard demonstrating the full pipeline |

## 4. Scope

### In scope
- **One clinical presentation:** acute chest pain (± breathlessness, palpitations)
- **13 conditions:** 8 cardiac + 5 non-cardiac must-not-miss mimics (see
  [01-srs.md](01-srs.md) §2.4 and the root `README.md`)
- **Decision support only** — ranked differential, evidence, reasoning, red flags
- **Synthetic and open data only** — DDXPlus, BODHI-S, Synthea, UCI Heart, PubMed/PMC

### Explicitly out of scope
- Autonomous diagnosis or any treatment recommendation
- Real, identifiable patient data; hospital or clinical deployment
- Presentations other than chest pain
- Training a medical language model from scratch
- Any claim of clinical efficacy without clinical validation

## 5. Research Contribution

The novelty is **not** "a knowledge graph plus an LLM" — that combination exists in the literature.
The contributions are:

1. A **contradiction- and red-flag-aware** differential ranker for a single high-stakes
   presentation, evaluated on **must-not-miss recall** rather than top-1 accuracy alone.
2. A rigorous **ablation study** isolating what each component (patient KG, medical KG, ML ranker,
   RAG, guardrails) actually contributes — including negative results.
3. A demonstration that a **knowledge graph can cover a diagnosis absent from the training data**
   (aortic dissection is not in DDXPlus but is encoded in the KG as a red-flag rule).

## 6. Success Criteria

**Minimum (project passes):** working prototype that accepts a chest-pain case and returns a ranked,
explained differential with red flags; complete documentation; evaluation against baselines.

**Target:** the above plus RAG-grounded evidence with citations, calibrated confidence, and a
complete ablation study showing which components help.

**Stretch:** "what to ask next" test suggestion; ECG multimodal extension; publishable write-up.

## 7. Deliverables

| Deliverable | Phase | Audience |
|---|---|---|
| Phase 0 documentation set | Week 1 | Supervisor, team |
| Walking skeleton (end-to-end run) | Week 3 | Team |
| Working prototype + demo video | Week 5 | Review presentation |
| Feature-complete system | Week 7 | Team |
| Evaluation + ablation results | Week 9 | Report |
| Final report / thesis + presentation | Week 9 | University |

## 8. Timeline Summary

| Phase | Weeks | Outcome |
|---|---|---|
| 0 — Preparation & documentation | 1 | Docs, scaffold, feasibility spike resolved |
| 1 — Data & knowledge foundations | 2–3 | **Walking skeleton** |
| 2 — Reasoning, fusion & prototype | 4–5 | **Working prototype (demo)** |
| 3 — Evidence & explanation | 6–7 | Feature-complete |
| 4 — Evaluation & write-up | 8–9 | Results + report |
| Buffer | 10 | Contingency, stretch goals |

## 9. Team & Responsibilities

| Role | Owns |
|---|---|
| **P1 — Cardiac Medical KG** | KG schema, KG built from DDXPlus's condition KB with BODHI-S enrichment, graph scoring, reasoning paths, red-flag rules |
| **P2 — Data & ML** | DDXPlus decoding, features, rankers, calibration, error analysis |
| **P3 — RAG + LLM + Safety** | Patient KG, evidence retrieval, grounded explanation, safety layer |
| **P4 — Platform & Evaluation** | Contracts, API, dashboard, evaluation harness, CI, documentation |

## 10. Constraints & Assumptions

- **Compute:** team laptops for development, plus free-tier cloud and (if granted) a university GPU
  for heavy jobs. No paid API budget; the language model is self-hosted via Ollama. *(Updated
  2026-09-18, decision D-7.)* Every training or tuning run, and any job over ~5 minutes, runs where
  the project owner decides: normally Google Colab, Kaggle, Lightning AI / Studio Lab or the
  university GPU. The owner's laptop GPU (RTX 5060, 8 GB) is **only for short tests, each run only
  after the owner says yes** (D-9). Neo4j runs on AuraDB Free. See [11-compute-runbook.md](11-compute-runbook.md) and risk
  R-14.
- **Time:** 8–10 weeks with daily work.
- **Licensing:** BODHI-S is CC-BY-NC-4.0 — the project must remain **non-commercial**.
- **Assumption:** DDXPlus coded evidences can be aligned with BODHI-S symptom names. *This is
  unverified and is the subject of the Week-1 feasibility spike;* a fallback is defined in
  [07-risk-register.md](07-risk-register.md). **Tested 2026-09-17: it does not hold.** Only 31% of
  conditions matched, against a 60% gate, so the fallback was adopted: the KG is built from
  DDXPlus's own condition knowledge base, and BODHI-S enriches four conditions
  ([10-spike-r01-crosswalk.md](10-spike-r01-crosswalk.md)).

---

## Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Project guide / supervisor | | | |
| Team lead | | | |
