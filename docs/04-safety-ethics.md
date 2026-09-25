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

| Condition | Red-flag pattern (as built in 2d, `src/reasoning/red_flags.py`) | In training data |
|---|---|---|
| Possible NSTEMI/STEMI | Chest pain with two of: exertional, pressure character, radiation to jaw/arm, diaphoresis | ✅ |
| Unstable angina | A crescendo pattern, or chest pain at rest with an ischaemic character | ✅ |
| Pulmonary embolism | Pleuritic pain or breathlessness **and** a Wells risk: unilateral leg swelling or calf pain, immobilisation, recent surgery, previous DVT | ✅ |
| Spontaneous pneumothorax | Sudden pleuritic pain + breathlessness | ✅ |
| Myocarditis | Chest pain after a viral illness, with breathlessness or palpitations | ✅ |
| Acute pulmonary edema | Breathlessness with orthopnoea, paroxysmal nocturnal dyspnoea or known heart failure | ✅ |
| Boerhaave | Chest pain after forceful vomiting | ✅ |
| **Aortic dissection** | **Pain with tearing character, a pulse deficit or an inter-arm BP difference; or findings from two ADD-RS categories** | ❌ **KG rule only** |

Each pattern follows a published one (ADD-RS for dissection, Wells for PE, Braunwald for unstable
angina) rather than any single suggestive finding: a flag on half of all patients teaches users to
ignore flags (R-15). EXP-008 measures them on DDXPlus validate.

**Which flags count as appropriate (decision A-7, 2026-09-25).** Red-flag precision (`docs/05`
§3.2, reported, not targeted) needs to know when a flag on another condition is still right. The
definition was fixed first: a flag for F on a patient whose final diagnosis is T is appropriate when
T lies on F's urgent diagnostic pathway (the work-up that excludes F also establishes or excludes
T), or F and T overlap as one disease process, and not merely because T is serious, typical of the
findings, or visible on a broad test. Three independent judges (an emergency physician's pathway, a
cardiologist's guidance, an alarm-fatigue sceptic) judged all 104 flag/condition pairs (8 flags × 13
other conditions); two votes of three make a pair appropriate. Each judge worked separately, was told
not to read any results, and read only `src/reasoning/red_flags.py` and `src/conditions.py`. **The
panel was not fully blind to how often flags fire**, though: the project notes loaded into every
session (`CLAUDE.md`, `PROGRESS.md`) and that module's docstring state several rates (the dissection
rule's 50% → 8%, the MI rule on 24% of patients without MI), and `PROGRESS.md` already called the MI
flag on unstable angina appropriate. The owner delegated the decision to
Claude, who adopted the panel's majority unchanged (`APPROPRIATE` in `src/reasoning/red_flags.py`):

| Flag | Also appropriate for (votes of 3) | Single yes, not adopted |
|---|---|---|
| nstemi stemi | unstable angina (3), stable angina (3), pericarditis (3), myocarditis (3), acute pulmonary edema (3), atrial fibrillation (2), psvt (2) | — |
| unstable angina | nstemi stemi (3), stable angina (3), pericarditis (3), myocarditis (3), acute pulmonary edema (2), atrial fibrillation (2), psvt (2) | — |
| myocarditis | nstemi stemi (3), unstable angina (3), pericarditis (3), acute pulmonary edema (2), atrial fibrillation (2), psvt (2) | — |
| acute pulmonary edema | nstemi stemi (3), unstable angina (2), myocarditis (2), atrial fibrillation (2) | psvt (1) |
| pulmonary embolism | its own condition only | — |
| spontaneous pneumothorax | its own condition only | boerhaave (1) |
| boerhaave | its own condition only | spontaneous pneumothorax (1) |
| aortic dissection | its own condition only | — |

The ischaemic flags share the ECG-and-troponin pathway with the cardiac conditions; the flags for
dissection, embolism, pneumothorax and Boerhaave are appropriate for their own condition only, so
DDXPlus's "tearing" pain on pneumothorax and Boerhaave still counts against the dissection rule.

**The safety layer v1** (`src/reasoning/safety.py`) checks every result last: the disclaimer is the
text in §2, every flagged candidate is listed before every unflagged one and named in the red flags
(FR-6.2), and no sentence recommends a treatment, names a drug or gives a dose (FR-6.5). A sentence
that does is removed and recorded in the explanation's unsupported claims.

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
