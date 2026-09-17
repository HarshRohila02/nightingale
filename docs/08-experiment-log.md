# 08 — Experiment Log

**Owner:** P2 (maintained by whoever runs the experiment)

> **Log every run that produces a number — including failures.** The ablation table in the final
> report is assembled from this file. An experiment that was not logged did not happen, and
> re-running it at the end costs more than logging it now.

---

## How to use this log

1. Copy the template into a new entry at the **top** of §Entries (newest first).
2. Fill it in **when you run the experiment**, not later.
3. Record the MLflow run ID so parameters and artifacts are recoverable.
4. Write the **Interpretation** honestly. "Worse than baseline, unclear why" is a valid, useful entry.

**Rules**
- Never overwrite an entry; add a new one.
- Development results come from **validation**. Any test-set number must be tagged `[TEST]` and may
  only appear in Phase 4.
- Record the git commit — a result you cannot reproduce is not a result.

---

## Template

```markdown
### EXP-NNN — <short title>

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Author | P_ |
| Config / ablation | A0 / B1 / … |
| Split used | validation \| [TEST] |
| Git commit | `abc1234` |
| MLflow run | `<run_id>` |
| Seed | 42 |

**Question:** What is this run meant to answer?

**Setup:** Model, features, hyperparameters, data version, anything non-default.

**Results:**

| Metric | Value | 95% CI |
|---|---|---|
| Top-1 | | |
| Top-3 | | |
| MRR | | |
| Must-not-miss recall@3 | | |
| ECE | | |

**Interpretation:** What it means. Be honest about negative or confusing results.

**Next action:** What this implies for the next experiment.
```

---

## Entries

*(newest first — add above this line as experiments are run)*

### EXP-001 — R-01 crosswalk spike: DDXPlus ↔ BODHI-S

| Field | Value |
|---|---|
| Date | 2026-09-17 |
| Author | P1 |
| Config / ablation | — (feasibility spike) |
| Split used | none — vocabulary files only |
| Git commit | Phase 0c |
| Seed | n/a (deterministic) |

**Question:** Can BODHI-S serve as the cardiac knowledge-graph backbone? Gate: 60% crosswalk
coverage (risk R-01).

**Setup:** `scripts/decode_ddxplus.py` then `scripts/spike_crosswalk.py`. Inputs were
`release_evidences.json` + `release_conditions.json` (141 kB) and BODHI-S `triples.jsonl` +
`nl_facts.jsonl` (4.4 MB). **The 847 MB DDXPlus patient CSVs were not needed** — the gate is a
vocabulary question, not a data-volume one.

**Results:**

| Metric | Value |
|---|---|
| BODHI-S conditions (total) | 555 |
| Condition coverage, strict (exact) | **31%** (4/13) |
| Condition coverage, generous (incl. approximate) | 77% |
| Symptom alignment, content-word | **28%** (35/123) |
| Symptom alignment, naive string | 0% (123 evidences) — *artifact* |
| DDXPlus evidence vocabulary decoded | 919 entries (223 questions → 696 values) |
| Nightingale conditions found in `release_conditions.json` | **13/13** |

**Interpretation:** Below the gate on every measure that matters → **FALLBACK adopted**. Three
findings worth carrying forward:

1. **BODHI-S is broad, not cardiac-deep.** Myocarditis, pneumothorax and Boerhaave are absent;
   stable/unstable angina collapse to one node, as do AF/PSVT.
2. **The 0% naive figure is a measurement artifact, not a result.** DDXPlus evidences are
   patient-facing questions; BODHI-S symptoms are clinical noun phrases. They never match lexically
   even when identical in meaning ("Are you feeling nauseous?" ≡ `Vomit <char> nausea present`).
   The 28% content-word figure is fairer but still generous — some matches hinge on "pain" alone.
3. **DDXPlus's own `release_conditions.json` is a better backbone than the anticipated fallback** —
   curated, 13/13 coverage, with ICD-10 codes and a severity ranking that independently
   corroborates our must-not-miss set.

**Also noted:** DDXPlus English is machine-translated from French and unreliable — `déchirante`
(*tearing*, the aortic-dissection descriptor) renders as **"heartbreaking"**. Prefer codes over
English labels throughout.

**Next action:** Closed R-01; opened **R-12 (circularity)** — the KG and ranker now share a source,
recorded as a reporting obligation in the evaluation protocol §8.7. P1 proceeds to build the KG
loader from `ddxplus_chestpain_conditions.json`.

Full report: [10-spike-r01-crosswalk.md](10-spike-r01-crosswalk.md)

---

### EXP-000 — Log initialised

| Field | Value |
|---|---|
| Date | 2026-09-17 |
| Author | P2 |
| Config | — |
| Git commit | Phase 0 |

**Question:** n/a — placeholder establishing the format.

**Next action:** First real entry will be **EXP-001, the Week-1 feasibility spike**: DDXPlus
evidence decoding and the DDXPlus↔BODHI-S crosswalk coverage measurement. Its result decides
risk **R-01** and therefore the KG backbone (see
[07-risk-register.md](07-risk-register.md)).

---

## Planned experiment sequence

| ID | Experiment | Phase | Decides |
|---|---|---|---|
| EXP-001 | Crosswalk coverage spike | 0 | R-01 — KG backbone |
| EXP-002 | Class balance across 13 conditions | 1 | R-03 — which conditions are learnable |
| EXP-003 | B0 prevalence baseline | 1 | Metric floor |
| EXP-004 | B1 ML-only (LogReg → XGBoost) | 1–2 | The competitor to beat |
| EXP-005 | B2 KG-only scoring | 2 | Is the graph useful alone? |
| EXP-006 | A0 fusion, weight sweep | 2 | Fusion weights (validation only) |
| EXP-007 | Calibration (Platt vs isotonic) | 2 | H5 |
| EXP-008 | Red-flag sensitivity/precision | 2 | Safety layer tuning |
| EXP-009 | B3 LLM-only | 3 | H3 |
| EXP-010 | B4 text-RAG + LLM | 3 | H3 |
| EXP-011 | Retrieval quality sweep | 3 | Chunking/embedding choice |
| EXP-012 | **Full ablation A0–A6** `[TEST]` | 4 | H1, H2, H4 |
