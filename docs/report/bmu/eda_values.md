# Values behind the exploratory figures

Computed from `data/interim/ddxplus_chestpain_validate.parquet` (33,963 validation patients) by `scripts/make_report_eda_figures.py`.
These are descriptive counts of the generated data, not experimental results; no interval applies.

## Patients per condition (`eda_class_balance.png`)

- Pulmonary embolism: 3,725 patients (must-not-miss)
- GERD: 3,426 patients
- Panic attack: 3,237 patients
- Pericarditis: 3,032 patients
- Possible NSTEMI / STEMI: 2,943 patients (must-not-miss)
- Unstable angina: 2,748 patients (must-not-miss)
- Atrial fibrillation: 2,609 patients
- Acute pulmonary edema: 2,500 patients (must-not-miss)
- PSVT: 2,376 patients
- Stable angina: 2,340 patients
- Boerhaave syndrome: 2,075 patients (must-not-miss)
- Myocarditis: 1,547 patients (must-not-miss)
- Spontaneous pneumothorax: 1,405 patients (must-not-miss)

## Age by condition (`eda_age_by_condition.png`)

- Myocarditis: median 35, interquartile range 20 to 54
- Atrial fibrillation: median 36, interquartile range 19 to 55
- GERD: median 36, interquartile range 19 to 55
- Pulmonary embolism: median 38, interquartile range 21 to 56
- Spontaneous pneumothorax: median 38, interquartile range 22 to 56
- Panic attack: median 39, interquartile range 24 to 56
- Boerhaave syndrome: median 40, interquartile range 26 to 57
- Pericarditis: median 40, interquartile range 26 to 58
- PSVT: median 40, interquartile range 23 to 58
- Unstable angina: median 44, interquartile range 30 to 58
- Possible NSTEMI / STEMI: median 45, interquartile range 31 to 60
- Stable angina: median 46, interquartile range 31 to 60
- Acute pulmonary edema: median 47, interquartile range 31 to 61

## Findings per patient (`eda_findings_per_patient.png`)

- Positive findings per patient: mean 14.1, median 14, minimum 2, maximum 25

## Sex

- F: 17,500 patients (51.5%)
- M: 16,463 patients (48.5%)
