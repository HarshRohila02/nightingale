# The project report in the BMU format

`docs/report/Nightingale-Project-Report-BMU.docx` is the project report in the layout of BML Munjal
University's *Body Format* and *Project File Front* templates (the two Word files in the repository root).
This folder holds its source and how to rebuild it.

## What is here

| Path | What |
|---|---|
| `00-conventions.md` | The structure, the Markdown subset, the content rules (read this before editing) |
| `abstract.md`, `part1.md`, `part2.md`, `part3.md` | The text: abstract; chapters 1-2; chapters 3-4; chapters 5-6 and appendices A-D |
| `refs.md` | The reference database (31 verified entries and 2 dataset cards, IEEE style) |
| `img/` | Figures: rendered diagrams, EDA charts, EXP-023 figures, the university logo |
| `diagrams/` | Mermaid sources of the diagrams |
| `eda_values.md` | The counts and quartiles behind the three EDA charts |

## How the format maps to the templates

- **Front file:** the title page (with the university logo), then the order of the items it lists: declarations,
  acknowledgement, table of contents, abstract. The lists of figures, tables and abbreviations (from the body
  template) follow, in roman page numbers.
- **Body file:** the six numbered chapters (Introduction; Literature Review; Exploratory Data Analysis;
  Methodology; Results; Conclusion and Future Scope) with the template's section titles, the bibliography in IEEE
  style, and the appendices. The *Problem Statement* the front file lists is Section 1.2.
- The template's chapter numbers are inconsistent (it skips Chapter 4 and starts its section numbers one
  chapter behind); the report numbers chapters 1 to 6 and sections to match.
- Page numbers and the table of contents are Word fields, updated by `scripts/finish_bmu_report.ps1`.

## Rebuild

```bash
./.venv/Scripts/python.exe scripts/make_report_eda_figures.py        # only if the EDA charts change
./.venv/Scripts/python.exe scripts/build_bmu_report.py                # writes the .docx
```

```powershell
# opens the .docx in Word, updates the table of contents and every page reference, and saves it
scripts\finish_bmu_report.ps1 -Docx docs\report\Nightingale-Project-Report-BMU.docx -Pdf docs\report\Nightingale-Project-Report-BMU.pdf
```

The Mermaid diagrams are rendered with headless Chrome and `mermaid@10.9.1` from jsDelivr (not scripted in
the repository: render `diagrams/*.mmd` and save the PNG into `img/`).

## What the team still has to fill in (highlighted yellow in the document)

- the student name(s), enrollment number(s), University Roll No., the dates of the work, the supervisor's name and
  designation (title page, declarations, acknowledgement);
- how any AI assistance is acknowledged (the team's open decision);
- the plagiarism-check report (the last page; similarity must be under 10%);
- the three dashboard screenshots (Figures 4.6 to 4.8): start the dashboard, load golden case GC-001, save
  the screenshots as `img/dashboard_ranking.png`, `img/dashboard_why.png` and `img/dashboard_evidence.png`, and rebuild;
- every `[citation needed: ...]` (a source the repository does not hold and that was not verified), and the other
  `[[...]]` placeholders that name a fact only the team knows.
