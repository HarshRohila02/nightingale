# 14 — Brief for the supervisor meeting, 2026-10-12

**For:** Prof. Ashish Kumar · **From:** the owner, for the team · **Prepared:** 2026-10-11
(also published as a private page for the owner to show). **Decision support only, never a
diagnosis.**

This answers the supervisor's email of 2026-10-04: *"check which deep learning model is best suited
to your approach and gives accurate results. Also, check the model complexity too, as real-time
models are preferred nowadays."* Everything under "proposed" still needs the team's approval
(`docs/05` amendment 4, `PROGRESS.md` §3).

## What is built (Weeks 1–3)

- **Data:** DDXPlus, 255,900 training and 33,963 validation patients across 13 conditions. Aortic
  dissection, which the data does not contain, comes from the knowledge graph.
- **Knowledge graph:** 129 nodes and 321 edges from three sources, on Neo4j AuraDB with a NetworkX
  fallback.
- **Models:** a prevalence floor (B0), logistic regression and XGBoost (B1), plus models retrained
  on deliberately incomplete histories.
- **Reasoning:**
  - a naive-Bayes graph score;
  - a logarithmic-opinion-pool fusion;
  - red-flag rules following ADD-RS, Wells and Braunwald;
  - a safety check.
- **Checks:** an end-to-end demo and 734 tests, including four hand-written golden clinical cases.

## The finding that shapes the deep-learning study

With a complete history, every ML model reaches about 99.8% top-1 (R-16). Models differ only when
part of the history is missing (EXP-019, validation split, seed 42; intervals in `docs/08`):

| Model | Top-1, full | Top-1, ¼ history | Must-not-miss recall@3, ¼ |
|---|---|---|---|
| XGBoost, normal training (B1-XGB) | 0.998 | 0.245 | 0.877 |
| Logistic regression, normal training (B1-LR) | 0.998 | 0.766 | 0.925 |
| Logistic regression, masked copies + "asked" channel (B1-LR′+aug) | 0.998 | 0.946 | 0.997 |
| XGBoost, masked copies + "asked" channel (B1-XGB′+aug) | 0.998 | 0.948 | 0.998 |

## The deep-learning study (proposed, EXP-021)

There are three candidates. Each uses a configuration fixed before training, because nothing is
tuned beyond early stopping (T-6). Each is trained like B1-LR′+aug and with seeds 42–46, so 15
networks in all. **One, the winner, runs the system.** Logistic regression stays as the comparison
and the backup.

| Candidate | Architecture | Parameters | Multiply-adds per patient |
|---|---|---|---|
| Plain MLP | 691 → 3 × 256 → 13 | 312,077 | 0.31 M |
| Residual MLP | Gorishniy et al. 2021 ResNet, 2 blocks, d = 192 | 432,589 | 0.43 M |
| Small transformer | 84 question tokens (asked / not asked) + age, sex, CLS; d = 96, 2 layers, 4 heads | ≈0.28 M | ≈10–16 M (the same for every patient) |

**Why a deep model might win here:** when inputs are missing, the best predictor is non-linear in
the observed values and the mask (NeuMiss, 2020). We found no published study of a fixed deep
classifier on randomly masked DDXPlus histories, or of its cost; the published DDXPlus agents ask
questions interactively instead.

**Selection rule (proposed):**
1. **Safe enough:**
   - must-not-miss recall@3 ≥ 0.95 and dangerous false negatives ≤ 2%, at full and ¼ history;
   - non-inferior to B1-LR′+aug (margin 0.005);
   - all four golden cases pass.
2. **Most accurate:** the highest MRR at ¼ history, as the five-seed mean.
3. **Prefer the cheaper model** when it is within 0.005 MRR of the best.
4. **Fallback:** if no deep model passes step 1, B1-LR′+aug stays the ranker.

**Complexity reported for every model (EXP-022):**
- parameters, multiply-adds and size on disk;
- training time;
- laptop-CPU latency p50, p95 and p99;
- the pipeline's latency, stage by stage.

## Real time

- **Feasible as soft real time.** Each case takes milliseconds on a laptop CPU, with predictable
  cost.
- **Not hard real time.** No deadline can be guaranteed on Python and Windows, and decision support
  does not need one.
- **Useful for live re-ranking** as findings are ticked. The literature-grounded LLM explanation
  follows seconds later.

## The finished project (the original scope) and its dates

**The finished project contains:**
- the Streamlit dashboard and the FastAPI service;
- the patient knowledge graph (Synthea);
- the chosen deep model fused with the medical knowledge graph;
- calibration;
- the red-flag rules;
- "what to ask next";
- literature retrieval and a grounded local-LLM explanation with citation checks;
- the full ablation study.

**Dates:**

| Milestone | Date |
|---|---|
| Deep-model results | 20–26 Oct |
| Prototype and demo video | 24 Oct |
| Feature-complete | 4 Nov |
| Test split opened once | ~7 Nov |
| Final report | 18 Nov |

## Questions for the supervisor

1. Is comparing an MLP, a residual MLP and a small transformer what "which deep learning model is
   best suited" means, or should a specific family be included (a GNN, a larger transformer)?
2. Is "safe first, then most accurate on incomplete histories, preferring the cheaper model on a
   tie" the right meaning of "accurate results"?
3. Real time: is a measured laptop-CPU response time (p95/p99, with a stated bound) what is
   expected? Or something else, such as streaming input or another device?
4. Is CPU-only deployment acceptable?
5. If time runs short, the agreed cut order is "what to ask next", then hybrid retrieval, then the
   LLM explainer. Is that acceptable?

## Limits stated with every result

- DDXPlus patients are synthetic.
- The knowledge graph is built from the same source the model learns from (R-12).
- The system knows only 14 conditions (R-13).
- Random masks are not real history-taking.

*The supervisor's answers are recorded in `PROGRESS.md` (§3 and §8) after the meeting.*
