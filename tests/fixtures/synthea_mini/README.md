# A hand-made Synthea output, for the patient-KG tests

**Invented, not generated.** Five patients written by hand in the CSV layout of Synthea v4.0.0
(`CSVConstants.java`: the same columns, in the same order), to test
`src/patient_kg/` without running Synthea. The codes are those of Synthea's own modules
(`src/patient_kg/mapping.py`); everything else is made up and refers to no one.

| Patient | What it tests (as of 2026-01-01) |
|---|---|
| `p-0001` | hypertension and diabetes active; prediabetes ended; hypertriglyceridaemia active but not mapped; smoked daily, now an ex-smoker (the latest observation wins); a condition that starts after the day is left out |
| `p-0002` | pregnant now (an earlier pregnancy ended); obese; never smoked |
| `p-0003` | died before the day, so no findings |
| `p-0004` | heart failure; a "smokes daily" condition still active beside a later "ex-smoker" observation (present wins); atrial fibrillation, a condition Nightingale ranks, not mapped |
| `p-0005` | nothing that maps; an acute bronchitis outside the table |
