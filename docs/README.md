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
| 05 | [Evaluation Protocol](05-evaluation-protocol.md) | How we measure success — **locked before modelling** | P4 |
| 06 | [Engineering Conventions](06-engineering-conventions.md) | Branching, style, testing, definition of done | P4 |
| 07 | [Risk Register](07-risk-register.md) | What could go wrong and who is watching it | P1 |
| 08 | [Experiment Log](08-experiment-log.md) | Running record of every run | P2 |
| 10 | [Spike R-01: Crosswalk](10-spike-r01-crosswalk.md) | Measured result that set the knowledge-graph strategy | P1 |
| 11 | [Compute Runbook](11-compute-runbook.md) | Where each job runs; the owner's laptop-GPU steps; cloud jobs; AuraDB setup | Owner + P4 |

## Reading paths

- **New team member:** Prerequisites → 00 → 02 → 06
- **Supervisor / reviewer:** 00 → 01 → 05 → 04
- **Starting to code:** 02 (contracts) → 06 (conventions) → 03 (data) → 11 (where jobs run)

## Document status

| Document | Status | Must be frozen by |
|---|---|---|
| 00–04, 06–11 | Draft, living | — |
| **05 Evaluation Protocol** | **🔒 Frozen 2026-09-17**, changed only via its amendment log (proposed amendment: D-8 in `PROGRESS.md`) | Before any model is trained ✅ |

> ⚠️ Document 05 is the one that must not change after modelling begins. Changing metrics after
> seeing results is how a project loses its scientific validity. Amendments must be dated and
> justified in the document's amendment log.

---

*Nightingale is a clinical **decision-support** research prototype. It is not a medical device and
must not be used for real patient care. See [04-safety-ethics.md](04-safety-ethics.md).*
