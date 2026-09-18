# 04 — Ethics, Safety & Clinical Risk

**Version:** 1.0 · 2026-09-17 · **Owner:** P3

> This is a system that ranks potentially fatal diagnoses. This document defines what it must never
> do, how it fails, and what we owe the person reading its output.

---

## 1. Intended use

**Nightingale is a research prototype that provides decision support to qualified medical staff
evaluating an acute chest-pain presentation.** It produces a ranked differential with supporting
evidence, reasoning, and red flags, so that a clinician can reason *with* it.

### Explicitly NOT intended for
- Autonomous diagnosis or any use without a qualified clinician in the loop
- Direct use by patients or the public
- Treatment, drug, or dosing recommendations
- Triage decisions made without clinician review
- Real clinical care, in any setting
- Any presentation other than acute chest pain

### Regulatory position
This is an **academic research prototype, not a medical device.** It has undergone no clinical
validation, no regulatory review, and no approval by any authority (CDSCO, FDA, EU MDR, or other).
Deploying it in a care setting would require a regulatory pathway this project does not attempt.
No claim of clinical efficacy may appear in the report, the UI, or any presentation.

---

## 2. The disclaimer

This exact text appears on **every** result (FR-6.4), as the `DISCLAIMER` constant in
`src/contracts.py`:

> ⚠️ **Clinical decision support — not a diagnosis.** Nightingale is a research prototype. Its
> output is generated from synthetic training data and must not be used for real patient care.
> Rankings are not certainties and confidence values are estimates. A qualified clinician is
> responsible for all diagnostic and treatment decisions.

It must not be collapsible, dismissible, or rendered less prominently than the ranking.

---

## 3. Must-not-miss policy

The central safety principle:

> **The cost of ranking a fatal condition too low is not symmetric with the cost of ranking a benign
> condition too high.** A false alarm costs a test. A missed myocardial infarction costs a life.

Consequences for the design:

1. **Must-not-miss recall is the headline metric**, not top-1 accuracy
   ([05-evaluation-protocol.md](05-evaluation-protocol.md)).
2. **Red-flag rules run independently of the ML ranker** (FR-6.1). A rule may fire on a condition
   the model scored near zero.
3. **The system is deliberately biased toward flagging danger.** We accept lower precision on red
   flags to protect recall.
4. **Absent ≠ unknown.** "Denies leg swelling" lowers PE; "leg swelling not asked" must not.

### Must-not-miss conditions

| Condition | Red-flag pattern | In training data |
|---|---|---|
| Possible NSTEMI/STEMI | Crushing/pressure pain, exertional, radiates jaw/arm, diaphoresis, risk factors | ✅ |
| Unstable angina | Ischaemic pattern at rest or worsening | ✅ |
| Pulmonary embolism | Pleuritic pain + breathlessness + unilateral leg swelling / immobilisation | ✅ |
| Spontaneous pneumothorax | Sudden pleuritic pain + breathlessness | ✅ |
| Myocarditis | Post-viral chest pain + breathlessness | ✅ |
| Acute pulmonary edema | Severe breathlessness, orthopnoea | ✅ |
| Boerhaave | Severe pain after forceful vomiting | ✅ |
| **Aortic dissection** | **Tearing pain radiating to back; inter-arm BP difference** | ❌ **KG rule only** |

Aortic dissection is the clearest demonstration of why the knowledge graph exists: the ML model
cannot represent it at all, and the rule layer covers the gap.

---

## 4. Failure modes and mitigations

| # | Failure mode | Consequence | Mitigation |
|---|---|---|---|
| F-1 | **LLM hallucinates a clinical fact** | Clinician misled by fluent, false text | Generation constrained to retrieved evidence; claim-support check; unsupported claims flagged/suppressed (FR-8.2–8.3) |
| F-2 | **Miscalibrated confidence** | "85%" read as clinical certainty | No raw score displayed as probability; calibration measured and reported; qualitative bands preferred |
| F-3 | **Must-not-miss ranked low** | Potentially fatal miss | Independent red-flag rules; must-not-miss recall as headline metric |
| F-4 | **Automation bias** | Clinician defers to the ranking | UI shows reasoning and *missing* information; contradicting evidence displayed; disclaimer always visible |
| F-5 | **Wrong symptom normalisation** | Findings map to wrong concept; silent corruption | Crosswalk unit-tested; unmapped terms surfaced, never silently dropped |
| F-6 | **Distribution shift** | DDXPlus is synthetic; real presentations differ | Documented as a primary limitation; no clinical-performance claim |
| F-7 | **Prompt injection via free text** | Malicious note manipulates the explanation | Free text treated as untrusted data; never concatenated as instructions; output validated against the candidate set (FR-8.4–8.5) |
| F-8 | **Silent component degradation** | User trusts a KG-only result believing evidence was checked | `degraded_components` always rendered in the UI |
| F-9 | **Class imbalance hides rare conditions** | Rare condition never ranked | Per-condition metrics reported; class weighting |
| F-10 | **Stale medical knowledge** | KG/literature out of date | Source dates recorded; scope limited to well-established relationships |

---

## 5. Data ethics and privacy

- **Synthetic and open data only.** No real patient records, therefore no IRB approval, no data-use
  agreement, and no personal data processing. This is a deliberate design choice, not an oversight
  — see [03-data-management.md](03-data-management.md).
- No user accounts or personal data collected by the prototype.
- Patient cases entered during a session are held in memory and discarded (FR-3.3).
- Audit logs (FR-6.6) record inputs, scores, and rule firings for research reproducibility. Since
  all inputs are synthetic, these carry no privacy risk — a property that would **not** hold in a
  real deployment, and which the report must state.

## 6. Bias considerations

DDXPlus is synthetic and generated from a rule-based simulator; it does not reflect real-world
demographic variation in presentation. Known concerns to report rather than claim to have solved:

- **Sex differences in ACS presentation** are clinically real (women more often present atypically).
  A model trained on synthetic data may under-represent this — a genuine safety concern that must
  be stated as a limitation.
- Age distribution and comorbidity patterns are simulator artifacts.
- BODHI-S is India/SNOMED-tagged; its symptom vocabulary may carry regional framing.

Report per-condition and by-sex breakdowns where sample size permits; do not claim fairness that
has not been measured.

**Measured evidence (EXP-002, 2026-09-18):** in DDXPlus every one of the 13 conditions is roughly
50% female, median age for acute MI is 45 (real-world first MI typically occurs in the 60s), and
spontaneous pneumothorax is 52% female (in reality strongly male-predominant). The training data
carries **no real sex or age priors.** Consequently, **near-parity by-sex results in Phase 4 are an
artifact of how the data was generated and must not be presented as evidence of fairness.**

---

## 7. Limitations to state explicitly

Every report, presentation, and demo must state:

1. Trained and evaluated on **synthetic** data; performance on real patients is **unknown**.
2. Scope is **acute chest pain only**; behaviour on any other presentation is undefined.
3. Aortic dissection is handled by rule, not learned.
4. **Closed-world:** only 13 conditions are ranked. A presentation caused by anything else (e.g.
   pneumonia) is still forced into one of the 13 (risk R-13). Red flags fire independently.
5. No clinical validation; no prospective study; no clinician user study at scale.
6. Confidence values are estimates from a research model, not clinical probabilities.

---

## 8. Ethical review checklist

- [ ] No real patient data at any stage
- [ ] Disclaimer present on every output path (UI, API, report figures)
- [ ] No treatment or drug recommendation anywhere in the system
- [ ] Red-flag layer functions independently of the ML ranker
- [ ] `degraded_components` surfaced in the UI
- [ ] Limitations section present in the report
- [ ] No claim of clinical efficacy or accuracy in real care
- [ ] Free-text input handled as untrusted
- [ ] Sex/age breakdowns reported where sample size allows
