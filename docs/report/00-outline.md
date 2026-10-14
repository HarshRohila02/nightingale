# 00 · Outline of the final report

> **Draft 1, 2026-10-14.** The report is presented with the working prototype on **2026-11-15**
> (PROGRESS §5). It is drafted alongside the work and completed after the test split is read
> (planned 2026-11-06 to 2026-11-07). Sources: `README.md`, `PROGRESS.md`, `docs/00`–`docs/14`, and
> the experiment log `docs/08`, from which **every figure is copied exactly**. Nothing in this folder
> is a verdict until EXP-012.

## Title and authors

**Nightingale — Clinical Intelligence & Reasoning System: a red-flag-aware decision-support
prototype for acute chest pain**

*Authors:* [four team members, P1–P4, names to be added] · *Supervisor:* [Prof. Ashish Kumar, as
named in `docs/14`; confirm] · *Institution, programme, date:* [to be added]

## Structure and word budget

Budgets count prose only: tables, figures, code, references and appendices are extra. The total is
about 20,000 words [to confirm against the university's limit with the supervisor]. "Draft" is the
prose count of this first draft.

| # | File | Chapter | Contents | Budget | Draft |
|---|---|---|---|---:|---:|
| — | — | Front matter | Title page; abstract (≤ 300 words, failed hypotheses stated plainly, `docs/05` §8 rule 5); the disclaimer; acknowledgements, including the project's AI assistance | 400 | not yet written |
| 1 | `01-introduction.md` | Introduction | The problem; aim and scope; RQs and hypotheses as `docs/05` §1 states them, with their amendment status; decision support only; contributions | 1,500 | 1,451 |
| 2 | `02-background.md` | Background and related work | Chest pain; clinical prediction rules; DDXPlus; knowledge graphs; opinion pools; calibration; tabular deep learning; references held in the repository | 2,000 | 1,946 |
| 3 | `03-data.md` | Data | Sources and licences; DDXPlus and its generator; the label audit and D-8; synthetic demographics; BODHI-S; the hand-authored dissection; golden cases; the closed test split; attribution block | 2,300 | 2,237 |
| 4 | `04-system.md` | The system as built | Contracts; the graph and crosswalk; the naive-Bayes score; the encoder and "asked" channel; the rankers; case → tokens (A-8, A-10); the transformer; fusion; red flags and safety; calibration; reasons; API and dashboard; degradation; what is not built | 4,000 | ~4,000 (trim in revision) |
| 5 | `05-evaluation.md` | Evaluation protocol | Pre-registration and amendments; metrics; evidence levels and masks with digests; baselines; ablations; statistics; success criteria; E-2; the rules fixed outside the protocol (EXP-006, EXP-007, D-13) | 2,200 | 2,188 |
| 6 | `06-results.md` | Results | Validation results so far, every figure with level, label and interval; placeholders for the fusion sweep, retrieval, the LLM baselines and the test read | 5,000 | 4,998 |
| 7 | `07-limitations-ethics.md` | Limitations, safety and ethics | Synthetic data; R-12; bias; R-17 and A-10; R-16; random masks; R-13 and what a probability means; red-flag limits; the safety design; every disclosure, dated | 2,500 | 2,440 |
| 8 | `08-conclusion.md` | Conclusion | Placeholder structure; written last | 1,000 | 648 (skeleton) |
| — | — | References | The list of §2.8, completed | — | partial |
| — | — | Appendices | A: attribution block (§3.8); B: the amendment log (`docs/05` §9, verbatim); C: the golden cases (`tests/fixtures/golden_cases.yaml`); D: reproducibility (commits, fingerprints, digests, versions); E: the demo and how to run it | — | A only |

## Checklist: what is still pending

Each item names the experiment, phase or decision it waits for.

### Results that do not exist yet

- [ ] **EXP-006, the fusion weight sweep** (2c): §6.12, A0's and every fused figure, and α for the
  configured component. Waits for **E-2 item 1** (the team), then the owner's choice of where it
  runs (D-7).
- [ ] **EXP-007 stage 2, calibration of A0′'s fused score** (2b): H5's input; how the dashboard shows
  a probability. Waits for EXP-006's α.
- [ ] **EXP-021's remaining reports** (2f): non-inferiority at seeds 43–46, and the full comparison
  table (McNemar on top-3 and the paired MRR test against B1-LR(′)+aug and B1-XGB(′)+aug at every
  level and seed). To run on Kaggle, the owner's choice (`notebooks/kaggle_exp021_reports.ipynb`,
  PROGRESS §2); waits for the owner's run.
- [ ] **EXP-011, retrieval** (3a): corpus, embeddings, BM25, metrics; §6.13 and FR-9.4.
- [ ] **EXP-009 / EXP-010, the LLM baselines B3 and B4** (3b), and the grounded explainer (3b, 3c):
  H3; the unsupported-claim rate in the Target. Waits for D-5 (which model) and D-7 (where).
- [ ] **"What to ask next"** (3d) and the reliability diagrams and per-condition error analysis
  (3e); 3d is first on the cut line if the prototype slips past 2026-10-28.
- [ ] **The patient knowledge graph and Synthea patients** (1d): needed for A0 and A6 as the
  ablation matrix defines them.
- [ ] **Explainability scoring** (`docs/05` §3.5): 50 cases, two raters, and whether a clinically
  trained reviewer is available (which decides the Stretch line).
- [ ] **EXP-012, the test split, opened once** (Phase 4, planned 2026-11-06 to 11-07): every
  ablation A0–A7 and A1b at three levels, every hypothesis verdict, the seeds count for each, the
  per-condition and by-sex breakdowns (§6.14, Chapter 8).
- [ ] **Latency with the LLM** (`docs/05` §3.6, NFR-1): Phase 3.

### Decisions that are open

- [ ] **E-2** (the team): how H1-R, H2-R, H6 and the reduced-evidence Target and Stretch are judged
  (§5.9). Blocks EXP-006.
- [ ] **A-2's clinical question** (a clinician): should the fusion rank dissection 2nd–3rd for
  ADD-RS-0 presentations with back pain or hypertension?
- [ ] **GC-005** (the team): a benign-looking ischaemic case that raises no red flag.
- [ ] **D-5** (the owner): which local LLM (Phase 3).
- [ ] **The other 16 representative answers** checked against DDXPlus, as A-10 checked one (R-17).

### Writing and checks before submission

- [ ] Every figure marked **†** in Chapter 6 gets its 95% interval from the run's saved output
  (`docs/05` §8 rule 1), or an explicit statement of why none applies (latency percentiles,
  counts).
- [ ] Every `[citation needed: …]` is resolved from a source the team has read, or the sentence is
  cut (§2.8 lists the references already held; most need full details checked).
- [ ] The DDXPlus test-split count is verified (§3.2.1).
- [ ] Licences re-verified and dated (`docs/03` §7, Phase 4e).
- [ ] The compliance checklists of `docs/03` §7 and `docs/04` §8 ticked (Phase 4e).
- [ ] The disclaimer on every figure that shows system output (`docs/04` §8).
- [ ] The abstract and Chapter 8 state every failed hypothesis plainly (`docs/05` §8 rule 5).
- [ ] The R-12 caveat beside every graph result; the synthetic-data caveat beside every accuracy
  claim; the level and label beside every figure (`docs/05` §8 rules 4, 7, 8).
- [ ] Acknowledgement of the project's AI assistance, worded with the team.
- [ ] Test counts (§4.17) and the configured ranker re-checked at the freeze (`v0.9-freeze`).
- [ ] The amendment log reproduced verbatim as Appendix B, with any amendment after 2026-10-02.
- [ ] Figures: the architecture diagram (Figure 4.1), and from 3e the reliability diagrams.

## Conventions used in every chapter

- **We** is the team. Short sentences, active voice, no claims the docs do not support.
- Every figure: its value, its 95% interval, its evidence level, its model label, and its source
  (`EXP-nnn`). Validation figures are labelled as such; test figures appear only after EXP-012 and
  are tagged `[TEST]` (`docs/08` rules).
- Placeholders: `[pending: …]` for results and work not yet done; `[citation needed: …]` for a
  source the repository does not hold; `[to verify: …]` for a fact to check.
- Never: BODHI-S text, DDXPlus patient rows, `.env` content, or any treatment or drug
  recommendation.
