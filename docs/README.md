# Nightingale — Documentation Index

> **Where is the project right now?** See [`../PROGRESS.md`](../PROGRESS.md) — current phase,
> in-flight work, pending decisions and history. It is updated with every task.

Phase 0 documentation set. Read in this order if you are new to the project.

| # | Document | What it answers | Owner |
|---|---|---|---|
| — | [Prerequisites & Learning Guide](09-prerequisites.md) | *"What do I need to know before I can contribute?"* | All |
| 00 | [Project Charter / Synopsis](00-project-charter.md) | Why this project exists, what success looks like | P4 |
| 01 | [Software Requirements Specification](01-srs.md) | What the system must do (functional + non-functional) | P4 |
| 02 | [Architecture & Interface Contracts](02-architecture.md) | How the pieces fit; **the data schemas everyone must honour** | P1 + P2 |
| 03 | [Data Management & Licensing](03-data-management.md) | Which datasets, under what licence, stored how; the chest-pain parquet's dataset card (§2.1) | P2 |
| 04 | [Ethics, Safety & Clinical Risk](04-safety-ethics.md) | What the system must never do; failure modes | P3 |
| 05 | [Evaluation Protocol](05-evaluation-protocol.md) | How we measure success — **locked before modelling**; v1.3 adds the reduced-evidence levels (amendment 3, 2026-09-25); v1.4 applies its errata (amendment 3a, 2026-10-02) | P4 |
| 06 | [Engineering Conventions](06-engineering-conventions.md) | Branching, style, testing, definition of done | P4 |
| 07 | [Risk Register](07-risk-register.md) | What could go wrong and who is watching it | P1 |
| 08 | [Experiment Log](08-experiment-log.md) | Running record of every run | P2 |
| 10 | [Spike R-01: Crosswalk](10-spike-r01-crosswalk.md) | Measured result that set the knowledge-graph strategy | P1 |
| 11 | [Compute Runbook](11-compute-runbook.md) | Where each job runs; the owner's laptop-GPU steps; cloud jobs; AuraDB setup | Owner + P4 |
| 12 | [Project Overview](12-project-overview.md) | A readable summary as of 2026-10-02: what is done, the techniques and the subjects they come from, a flowchart to the final result, and the time left. `PROGRESS.md` stays the source of truth | All |
| 13 | [Simplified Project Guide](13-simplified-project-guide.md) | A guide to the 2026-10-02 scope cut: **withdrawn** on 2026-10-11 and kept as a record | — |
| 14 | [Supervisor Brief](14-supervisor-brief.md) | The one-page brief for the owner's meeting with the supervisor on 2026-10-12 | Owner |
| 16 | [Cloud-Session Handoff](16-cloud-handoff.md) | **Read first in a new session (2026-10-14):** where each workstream stopped, the `wip/` branches, what a cloud machine cannot do (no `data/`, `models/` or `.env`), the owner's standing rules, the suggested order | Next session |
| 17 | [Cloud-Session Tasks](17-cloud-session-tasks.md) | **The cloud session's work list:** the tasks it can do with no risk and no input from the owner (T1–T8), how to work (a branch and a PR per task), and what it must leave to the owner, the team or the laptop | Next session |
| — | [Mid-term evaluation: preparation](presentation/midterm-prep.md) · [slide prompt](presentation/midterm-ppt-prompt.md) | The PRJ1 mid-term presentation (2026-10-14): what to say against each marked criterion (literature review, research gap, objective, methodology), the figures with their intervals, likely questions, references to check; and a self-contained prompt for a slide generator | All four |
| 15 | [The Working Prototype](15-prototype-demo.md) | How to start the dashboard and the API, the checklist for the day before a presentation, a five-minute demo script over the golden cases, the figures to quote with their caveats, and likely questions (`v0.1-prototype`, 2026-10-14) | Presenter |
| — | [Proposal: `docs/05` amendment 3](proposals/05-amendment-3.md) | Reduced evidence (D-10) and the deep ranker (D-11): **approved 2026-09-25** and applied to `docs/05` v1.3; kept as the record of the proposal | All four |
| — | [Proposal: amendment 3 errata](proposals/05-amendment-3-errata.md) | Contradictions an independent review found inside amendment 3's approved wording, with exact corrections: **approved 2026-10-02** and applied to `docs/05` v1.4 as amendment 3a; kept as the record | All four |
| — | [Proposal: amendment 3a follow-ups](proposals/05-amendment-3b-errata.md) | What the verifiers of amendment 3a found: the `+aug` comparators are perfect or within four misses at 50%, so, as approved, the reduced-evidence claims cannot be met (needs an answer before the fusion sweep), and ten gaps left in the text: **for the team** (E-2) | All four |

| — | [The report, first draft](report/00-outline.md) | Chapters 0–8 of the final report (2026-10-14): validation results so far, with placeholders for what is pending | All four |

## Reading paths

- **New team member:** Prerequisites → 00 → 02 → 06
- **Supervisor / reviewer:** 00 → 01 → 05 → 04
- **Starting to code:** 02 (contracts) → 06 (conventions) → 03 (data) → 11 (where jobs run)

## Document status

| Document | Status | Must be frozen by |
|---|---|---|
| 00–04, 06–11 | Draft, living | — |
| **05 Evaluation Protocol** | **🔒 Frozen 2026-09-17**, changed only via its amendment log (amendments 1–2 on 2026-09-19: D-8 and red-flag sensitivity; 3 on 2026-09-25: reduced evidence and the deep ranker; 3a on 2026-10-02: its errata) | Before any model is trained ✅ |

> ⚠️ Document 05 is the one that must not change after modelling begins. Changing metrics after
> seeing results is how a project loses its scientific validity. Amendments must be dated and
> justified in the document's amendment log.

---

*Nightingale is a clinical **decision-support** research prototype. It is not a medical device and
must not be used for real patient care. See [04-safety-ethics.md](04-safety-ethics.md).*
