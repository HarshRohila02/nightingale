# 00 · Outline of the final report

> **Brought to the final state on 2026-11-12.** Chapters 1 to 8 now describe the project after the single
> test read (2026-11-09), the freeze tag `v0.9-freeze`, the compliance pass (2026-11-10) and the age
> breakdown (2026-11-11). Sources: `README.md`, `PROGRESS.md`, `docs/00`–`docs/15`, and the experiment log
> `docs/08`, from which **every figure is copied exactly** (a script checked that all 700 interval figures
> and every other three- or four-decimal number in chapters 1 to 8 occur in the repository's records).
> What stays open is listed below; none of it is a result still to be run.

## Title and authors

**Nightingale — Clinical Intelligence & Reasoning System: a red-flag-aware decision-support
prototype for acute chest pain**

*Authors:* [four team members, P1–P4, names to be added] · *Supervisor:* [Prof. Ashish Kumar, as
named in `docs/14`; confirm] · *Institution, programme, date:* [to be added]

## Structure and word budget

Counts are prose only: tables, figures, code, headings, quotation banners, references and appendices are
excluded. The original plan was about 20,000 words [to confirm against the university's limit with the
supervisor]. **The chapters now hold about 32,500 words of prose**, so if the limit is near 20,000 the report
needs trimming, chiefly chapters 4 and 6, whose tables carry the figures and whose prose can be cut first.

| # | File | Chapter | Contents | Budget | Now |
|---|---|---|---|---:|---:|
| — | — | Front matter | Title page; abstract (≤ 300 words, failed hypotheses stated plainly, `docs/05` §8 rule 5); the disclaimer; acknowledgements | 400 | not written (team) |
| 1 | `01-introduction.md` | Introduction | The problem; aim and scope; RQs and hypotheses as `docs/05` v1.5 states them, with each verdict; decision support only; contributions | 1,500 | 2,003 |
| 2 | `02-background.md` | Background and related work | Chest pain; clinical prediction rules; DDXPlus; knowledge graphs; opinion pools; calibration; tabular deep learning; the verified references (`references.md`) | 2,000 | 1,721 |
| 3 | `03-data.md` | Data | Sources and licences; DDXPlus and its generator; the label audit and D-8; synthetic demographics; BODHI-S; the corpus; Synthea; the hand-authored dissection; golden cases; the splits and the single opening of the test split; attribution block | 2,300 | 3,198 |
| 4 | `04-system.md` | The system as built | Contracts; the graph and crosswalk; the naive-Bayes score; the encoder and "asked" channel; the rankers; case → tokens (A-8, A-10); the transformer; fusion; red flags and safety; calibration; reasons; evidence; the explainer that ships and the one that does not; API and dashboard; degradation; what is not built | 4,000 | 6,901 |
| 5 | `05-evaluation.md` | Evaluation protocol | Pre-registration and amendments (to 3b); metrics; evidence levels and masks with digests; baselines; ablations; statistics; success criteria; E-2 and its answer; the rules fixed outside the protocol; the freeze, the rehearsal and the single opening; what was not assessed | 2,200 | 4,085 |
| 6 | `06-results.md` | Results | Development results on validation, then §6.15, the test split `[TEST]` with every verdict, every figure with level, label and interval; the age breakdown on validation | 5,000 | 10,314 |
| 7 | `07-limitations-ethics.md` | Limitations, safety and ethics | Synthetic data; R-12; bias and the age and sex breakdowns; R-17 and A-10; R-16; random masks; R-13 and what a probability means; red-flag limits; the safety design; every disclosure, dated, to 2026-11-11 | 2,500 | 2,683 |
| 8 | `08-conclusion.md` | Conclusion | What was built; the answer to each research question, failures first; what the project found; limitations; future work; the closing statement | 1,000 | 1,657 |
| — | `references.md` | References | 27 verified entries cited, 4 verified for other chapters, and the items not verified with the reason for each | — | the lead deletes its "Working notes" section before submission |
| — | — | Appendices | A: attribution block (§3.8); B: the amendment log (`docs/05` §9, verbatim); C: the golden cases (`tests/fixtures/golden_cases.yaml`); D: reproducibility (commits, fingerprints, digests, versions); E: the demo and how to run it (`docs/15`) | — | A only |

## Checklist: what is done, and what is still open

### Results (all recorded in `docs/08`)

- [x] EXP-006, the fusion weight sweep: α = 0.05 for the configured component (A0′), 0.30 for A0; the fusion
  adds nothing measurable (2026-11-07).
- [x] EXP-007 stage 2, calibration of A0′'s fused score: temperature scaling chosen, **not deployed**
  (`calibration.file` null, ranks only, 2026-11-09).
- [x] EXP-021's remaining reports (2026-10-14) and EXP-022's re-measures (2026-11-08).
- [x] EXP-011, retrieval: the corpus is built and wired; **the relevance judging was not done** (no time) and
  is disclosed as a gap; the query-vector mix's median top-10 overlap is 0.400.
- [x] EXP-009 / EXP-010, the language-model baselines: ran on Kaggle 2026-11-07; D-5's checkpoint failed
  (unsupported-claim rate 0.741 [0.663, 0.803]); the template explainer ships; B3 and B4 are not carried into
  the evaluation; H3 is not supported.
- [x] "What to ask next" (3d), reliability diagrams and per-condition errors (3e).
- [x] The patient knowledge graph and Synthea patients (1d): built, display only (A6 equals A0 by construction).
- [x] EXP-012, the test split, opened once (2026-11-09): every ablation at three levels, every verdict, the
  seeds, the by-condition and by-sex breakdowns (§6.15).
- [x] The age breakdown (five bands, validation only; a test-split run would need a second opening).
- [ ] **Not assessed, and stated as such:** the unsupported-claim rate of the template explainer that ships;
  the 1–5 explanation ratings (the owner decided on 2026-11-09 not to rate the sheet); the
  clinician-reviewed explainability line of the Stretch criterion; A2 and A7 on the test split in this
  report's tables (they are in the runner's output file).

### Decisions

- [x] **E-2**: answered by the owner on 2026-11-07, amendment 3b (`docs/05` v1.5); the other members'
  agreement is not recorded, and the report says so (§5.9, §7.10).
- [ ] **A-2's clinical question** (a clinician): should the fusion rank dissection 2nd–3rd for ADD-RS-0
  presentations with back pain or hypertension? Still open, and stated as open.
- [x] **GC-005**: not added (the freeze fixes `tests/fixtures/`); the report says so.
- [x] **D-5**: `llama3.1:8b`, kept with a checkpoint; the checkpoint failed.
- [ ] **The other 16 representative answers** checked against DDXPlus, as A-10 checked one (R-17): not done;
  stated as a limitation.

### Writing and checks before submission (the team's)

- [ ] **The university's word limit and format** (the chapters are about 32,500 words of prose, §above).
- [ ] **How AI assistance is acknowledged** (the team's open decision), and the acknowledgement text.
- [ ] **Names, institution, programme, date and the supervisor's title** (placeholders kept as they were).
- [ ] **The abstract and the front matter** (the conclusion states every failed hypothesis; the abstract
  should repeat them, `docs/05` §8 rule 5).
- [ ] **The 19 `[citation needed]` markers that remain** (§2.8 and `references.md` Part C list why each could
  not be verified: the Wells criteria's first-author mismatch in Crossref, Nielsen's response-time limits, the
  UCI Heart Disease year, real-world epidemiology of MI and pneumothorax, and similar). Either supply a
  source the team has read or cut the sentence; never fill one from memory.
- [ ] **The 56 reference candidates marked [V]** in `docs/presentation/midterm-prep.md` §10 were not checked
  here; three errors were found in it (DDxT's authors, the Wells entry, Med-HALT's venue) and are listed in
  `references.md`'s working notes. Check them before any slide reuses them.
- [ ] **Appendices B to E** (the amendment log verbatim, the golden cases, reproducibility, the demo).
- [x] The licences re-verified and dated (2026-11-10); the compliance checklists of `docs/03` §7 and
  `docs/04` §8 ticked with evidence; the disclaimer on every figure (the 22 figures carry it).
- [x] The R-12 caveat beside every graph result; the synthetic-data caveat beside every accuracy claim; the
  level, label and split beside every figure (`docs/05` §8 rules 4, 7, 8).
- [x] Every figure marked **†** is one the log gives without an interval (a standard deviation, a latency
  percentile or a count); each says so.

## Conventions used in every chapter

- **We** is the team. Short sentences, active voice, no claims the docs do not support.
- Every figure: its value, its 95% interval, its evidence level, its model label, and its source
  (`EXP-nnn`). Validation figures are labelled as such; test figures are tagged `[TEST]` (`docs/08` rules).
- Placeholders: `[citation needed: …]` for a source the repository does not hold and that was not verified;
  `[to verify: …]` for a fact to check; `[team to confirm]` style for names and wording the team owns.
- Never: BODHI-S text, DDXPlus patient rows, `.env` content, or any treatment or drug recommendation.
