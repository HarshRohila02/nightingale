# Archive — superseded reference documents

These are the documents the project started from. They are kept for **provenance only** — none of
them describes the current project, and none should be used as a specification. The current state
of the project is in [`../../PROGRESS.md`](../../PROGRESS.md); the current specification is the
root `README.md` and `docs/00`–`10`.

Archived 2026-09-18 by user decision **D-3**. Files were renamed on archiving so their names describe
what they actually are; the original filenames are recorded below.

| File | Original filename | What it is | Why it is not current |
|---|---|---|---|
| `original-research-prompt.md` | `medical_differential_diagnosis_research_prompt(1) (1).md` | The founding deep-research **prompt**: a 36-section brief for an AI differential-diagnosis CDSS (patient KG + medical KG + ML + RAG + LLM) | A prompt, not a report. Its scope was all of differential diagnosis; the project has since narrowed to **acute chest pain** and switched to open/synthetic data only. |
| `chatgpt-generic-project-plan.pdf` | `Project Plan for an AI_ML System.pdf` | A ChatGPT-generated, **domain-agnostic** AI/ML project plan ("since the project domain is unspecified") | Written before the domain was chosen. Recommends production/industry practices (microservices, Kubernetes, continuous retraining, $10k–$100k budgets) that conflict with this project's scope. Its `【n†Lx-Ly】` citations are dead ChatGPT browsing markers. |
| `chatgpt-generic-research-guide.md` | `deep-research-report.md` | A generic **"Guide to Conducting a Research Task"** — clarifying questions, the CRAAP test, example briefs on remote-work productivity and EV charging | **Misnamed at source.** Despite the original filename it is *not* a research report on this project and never mentions differential diagnosis, knowledge graphs or clinical NLP. Renamed so nobody mistakes it for one. |

**Deleted, not archived:** `medical_differential_diagnosis_research_prompt(1) (2).md` — a
byte-identical copy of the research prompt (SHA-256
`fc0212c3aa2f8fb9419e662756606f4b25f858c84d64e0bbe91db348b7ec4626`, verified immediately before
deletion). Its content is preserved in full as `original-research-prompt.md`.

**What is still worth borrowing from these files:** only generic engineering hygiene from the PDF —
baselines, cross-validation with bootstrap confidence intervals, SHAP, experiment tracking,
reproducibility, a risk register. All of that is already captured in `docs/05`–`docs/07`.
