# The BMU-format report: conventions for everyone who writes or builds it

> This folder holds the Markdown source of the project report in the format of BML Munjal University's
> *Body Format* and *Project File Front* templates (the two Word files in the repository root). A build script
> turns the Markdown into a Word document with the template's layout. The facts and figures come from the
> repository's records; this folder reorganises them into the template's chapters. **Decision support only,
> never a diagnosis**: nothing in the report may suggest otherwise.

## 1. Files and owners

| File | Contents | Written by |
|---|---|---|
| `abstract.md` | The abstract (at most 300 words) | writer A |
| `part1.md` | Chapter 1 (Introduction) and Chapter 2 (Literature Review) | writer A |
| `part2.md` | Chapter 3 (Exploratory Data Analysis) and Chapter 4 (Methodology) | writer B |
| `part3.md` | Chapter 5 (Results), Chapter 6 (Conclusion and Future Scope) and Appendices A to D | writer C |
| `refs.md` | The reference database (finished; do not edit) | lead |
| `img/` | Figures already rendered (list in section 5; do not edit) | lead |
| `diagrams/` | Mermaid sources of the diagrams (do not edit) | lead |

Edit only your own files. Do not commit, push, tag, delete or move anything. The lead assembles and builds.

## 2. The fixed structure (headings and numbers are the template's; use them exactly)

Chapters are numbered by the order of the `#` headings across `part1.md`, `part2.md`, `part3.md`: do **not** write
"Chapter N" in a heading, the builder prints it. Section numbers are written by hand and must match the list.

```
# Introduction                                       (Chapter 1)
## 1.1 Overview
## 1.2 Problem Statement
## 1.3 Existing System
## 1.4 User Requirement Analysis
## 1.5 Feasibility Study
# Literature Review                                  (Chapter 2)
## 2.1 Review of Related Work          (### 2.1.1 ... by topic, in a flow of the topic, every claim cited)
## 2.2 Comparison                      (a table comparing prominent works with this project)
## 2.3 Objectives of Project           (clearly and precisely defined; based on the gaps; each with its status)
# Exploratory Data Analysis                          (Chapter 3)
## 3.1 Dataset
## 3.2 Exploratory Data Analysis and Visualisations
## 3.3 Related Sections
# Methodology                                        (Chapter 4)
## 4.1 Introduction to Languages (Front End and Back End)
## 4.2 Any Other Supporting Languages and Packages
## 4.3 User Characteristics
## 4.4 Constraints
## 4.5 Use Case Model, Flow Chart and DFDs
## 4.6 Database Design
## 4.7 Table Structure
## 4.8 ER Diagrams
## 4.9 Assumptions and Dependencies
## 4.10 Algorithm Discussion
## 4.11 Implementation of Algorithm with Screen Shots and Figures
# Results                                            (Chapter 5)
## 5.1 Experimental Set-up
## 5.2 Development Results on the Validation Split
## 5.3 Results on the Test Split
## 5.4 Safety, Red Flags and Calibration
## 5.5 Evidence Retrieval and the Language-Model Trial
## 5.6 Breakdowns by Condition, Sex and Age
## 5.7 Discussion
# Conclusion and Future Scope                        (Chapter 6)
## 6.1 Conclusion
## 6.2 Future Scope
## Appendix A. Data Sources, Licences and Attribution       (use "## Appendix X. title", appended after Chapter 6)
## Appendix B. Golden Clinical Cases
## Appendix C. Reproducibility
## Appendix D. How to Run the System
```

The abstract, acknowledgement, declarations, lists, bibliography and the plagiarism-check page are produced by
the lead. Writer A writes `abstract.md`; nobody else writes front matter.

## 3. The Markdown subset the builder understands (anything else is printed as plain text)

- `# Title` = chapter; `## 1.1 Title` = section; `### 1.1.1 Title` = subsection (numbers by hand). No deeper levels.
- Paragraphs are separated by a blank line. Do not hard-wrap inside a sentence you want to keep together (a newline
  inside a paragraph is read as a space).
- `- item` bullets and `1. item` numbered lists (two spaces of indent for a nested level). A list must be preceded by a blank line.
- Inline: `**bold**`, `*italic*`, `` `code` ``. Superscripts and subscripts: `^text^` and `~text~`. Use real Unicode for
  symbols (α, ′, ≥, ≤, ×, →, ², ·); the template font (Times New Roman) has them. Never use `*` or `_` inside a word.
- **Tables** are pipe tables with a header row and a separator row. The caption is a line immediately before the table,
  in exactly this form (the number is `chapter.sequence`, counted by hand, in order of appearance):
  `**Table 5.3.** Caption text, ending with the split and the evidence level.`
  Keep a table to at most 7 columns and about 14 rows; split bigger ones. Put the unit, split, evidence level and model
  label in the caption or the header. Left-align text columns; write numbers as the record does (value with its interval).
- **Figures**: `![Figure 4.2. Caption text.](img/name.png)` (path relative to this folder). Number as for tables.
  A missing image file prints a highlighted placeholder, so you may reference the three dashboard screenshots listed in section 5.
- **Code or console listings**: a fenced block (three backticks) is printed in a small monospace font. Use it for
  short outputs only (at most 25 lines of at most 90 characters).
- **Citations**: `[@gulati2021]` or `[@gulati2021; @hagan2000]`; the builder numbers them in order of first use
  (IEEE style) and builds the bibliography. **Use only the keys in `refs.md`.**
- **Placeholders** the team must fill or resolve are written `[[like this]]`; the builder highlights them in yellow
  (as the template highlights its own fields). Use them for a source that is not in `refs.md`
  (`[[citation needed: what the claim is]]`) and for anything only the team knows. Never invent a reference.
- Cross-references are plain text ("Table 5.3", "Figure 4.2", "Section 4.10", "Chapter 3").
- No HTML, no footnotes, no equations in LaTeX: write formulas in words or in plain Unicode on their own line, in a fenced block.

## 4. Content rules (they decide whether the report is acceptable)

1. **Every number is copied from the repository's records**, chiefly `docs/08-experiment-log.md` (the single record of
   every result), `docs/15-prototype-demo.md` §5, `docs/18-project-report.md` and the already-checked chapters in
   `docs/report/0[1-8]-*.md`. **Never compute a new number; never quote one you cannot find there.** If something you
   need is not recorded, write that it was not assessed.
2. **Every figure from an experiment carries its 95% interval in square brackets, its evidence level (100%, 50%, 25%),
   its model label (A0, A0′, B1-XGB, B1-LR, B1-DL′+aug and so on) and its split** (validation, or `[TEST]` for the single
   opening of the test split on 2026-11-09). Counts and settings (patients, passages, versions) need no interval.
3. **State failures as failures.** H1, H2, H1-R, H2-R, H3 and H6 are *not supported*; H4, H5 and the Minimum hold, and
   H4's support is hollow (the graph contributes nothing measurable; the contrast comes from the red-flag reorder);
   the reduced-evidence Target is met by A0′ only, by a handful of patients; the language-model explainer failed its
   checkpoint and does not ship; the evidence's relevance was never judged. Do not soften any of it.
4. **Keep every caveat that travels with a figure**: synthetic data (no claim about real patients); the closed world of
   14 conditions (R-13); the knowledge graph's circularity on DDXPlus (R-12); the ceiling effect on full histories
   (R-16); near-equal results by sex or age are an artifact of synthetic demographics, not evidence of fairness; random
   masks measure robustness to missing answers, not history-taking; the hand-written cases' near-certainty (R-17).
5. **No treatment, drug or dose advice anywhere. Never weaken the disclaimer.** The disclaimer (exact text) is
   "Clinical decision support — not a diagnosis. Nightingale is a research prototype. Its output is generated from
   synthetic training data and must not be used for real patient care. Rankings are not certainties and confidence
   values are estimates. A qualified clinician is responsible for all diagnostic and treatment decisions." (without
   the warning-sign emoji). If you quote it, copy it from `src/contracts.py` (`DISCLAIMER`).
6. **Do not write about AI assistance** and do not name tools as authors: the team has not decided how it is
   acknowledged. Technology names (Python, NumPy, Streamlit and so on) are fine. Do not claim clinical validity.
7. Write in the first person plural for the team ("we built", "we measured"), short active sentences, no padding.
   British or American spelling: use one (the existing chapters use British; keep it).
8. Explain each technical term at first use in plain words, because the examiners may not be machine-learning
   specialists. Define abbreviations at first use, and keep a list of every abbreviation you use (section 6).
9. Do not invent anything about the team, the university, the supervisor or the dates. Use `[[...]]` placeholders.
10. Edit only your own files. If two records contradict each other, do not fix either: report it to the lead.

## 5. Figures already rendered in `img/` (use these; the lead cannot make others in time)

| File | What it shows | Use in |
|---|---|---|
| `architecture.png` | The full request path of the system (a flow chart) | 4.5, 4.11 |
| `dfd.png` | Data-flow diagram: the clinician, six processes, four data stores | 4.5 |
| `usecase.png` | Use-case model: the clinician's and the evaluator's use cases | 4.5 |
| `erd_kg.png` | ER diagram of the knowledge graph and the crosswalk | 4.6, 4.8 |
| `erd_contracts.png` | ER diagram of the request and result data types (contracts) | 4.7, 4.8 |
| `eda_class_balance.png` | Patients per condition in the validation split (must-not-miss conditions in red) | 3.2 |
| `eda_age_by_condition.png` | Median and interquartile range of age by condition (generated ages) | 3.2 |
| `eda_findings_per_patient.png` | Histogram of positive findings per patient (full history) | 3.2 |
| `exp023_recall_top1_by_level.png` | Per-condition top-1 recall at the three evidence levels (transformer, validation) | 3.2 or 5.6 |
| `exp023_recall_top3_by_level.png` | The same for top-3 recall | 5.6 |
| `exp023_reliability_b1_dl_prime_aug.png` | Reliability (calibration) diagram of the configured ranker | 5.4 |
| `exp023_reliability_by_condition_b1_dl_prime_aug.png` | Reliability diagram by condition | 5.4 |
| `exp023_confusion_b1_dl_prime_aug_100.png`, `exp023_confusion_b1_dl_prime_aug_25.png` | Confusion matrices of the configured ranker at 100% and 25% evidence (validation) | 5.6 |
| `dashboard_ranking.png`, `dashboard_why.png`, `dashboard_evidence.png` | Dashboard screenshots (the lead is trying to capture them; if absent the builder prints a placeholder) | 4.11 |

The EXP-023 figures come from `docs/08` EXP-023 (validation, seed 42, the components alone); caption them so.

## 6. List of abbreviations

Each writer ends the file with a fenced block marked `abbreviations`, one `ABBR = Full form` per line, for every
abbreviation used in the writer's chapters. The lead merges them into the list of abbreviations. Example:

```abbreviations
KG = Knowledge Graph
MRR = Mean Reciprocal Rank
```

## 7. Report back (plain text, concise)

Files written and their word counts; any figure you could not find in the records; any contradiction between files;
every `[[placeholder]]` you left and why; anything you want the lead to check.
