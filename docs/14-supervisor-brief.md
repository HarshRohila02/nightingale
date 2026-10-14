# 14 — Brief for the supervisor meeting, 2026-10-12

**For:** Prof. Ashish Kumar · **From:** the owner, for the team · **Prepared:** 2026-10-11,
**updated 2026-10-12** with the owner's decisions (also published as a private page for the owner
to show). **Decision support only, never a
diagnosis.**

This answers the supervisor's email of 2026-10-04: *"check which deep learning model is best suited
to your approach and gives accurate results. Also, check the model complexity too, as real-time
models are preferred nowadays."* **Updated 2026-10-12:** the owner has chosen the deep model, a
small transformer (D-12), so the three-way comparison of the first version is withdrawn, and with
it the selection rule that needed the team's approval.

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

## The deep-learning model: a small transformer (EXP-021)

**The owner's final choice (2026-10-12):** one deep model, a small transformer, runs the system.
Logistic regression (B1-LR′+aug) stays as the comparison and the backup. The configuration is
fixed before training, because nothing is tuned beyond early stopping (T-6).

| | |
|---|---|
| Input | One token per DDXPlus question (84), each the sum of its answers' embeddings plus an "asked / not asked" embedding; plus age, sex and a [CLS] token |
| Model | FT-Transformer style (Gorishniy et al. 2021): 2 blocks, width 96, 4 attention heads, ReGLU feed-forward, the paper's default dropouts and optimiser |
| Size | 225,645 parameters (0.9 MB as float32; 1.23 MB as its JSON file, EXP-022); about 9.6 M multiply-adds per patient, the same for every patient |
| Training | Like B1-LR′+aug: the full histories plus one masked copy of each patient; seeds 42–46, on a ~~Colab~~ Kaggle GPU (2026-10-13). The seed-42 model is the one used |
| Laptop | A NumPy forward pass on the CPU, with nothing to install |

**Why a transformer:** findings interact (exertional pain means something different with and
without rest pain), and attention lets each finding see the others. When inputs are missing, the
best predictor is non-linear in what was observed and what was asked (NeuMiss, 2020). The
published deep models on DDXPlus are MLPs and transformers; we found no study of a fixed deep
classifier on randomly masked DDXPlus histories, or of its cost.

**Safe first, then accurate (D-13).** The transformer replaces logistic regression only if, on the
validation split:
1. must-not-miss recall@3 ≥ 0.95 and dangerous false negatives ≤ 2%, at full and ¼ history;
2. it is no worse than logistic regression at ¼ history on must-not-miss recall@3 and MRR (a
   margin of 0.005, by a paired bootstrap);
3. all four golden cases pass.

Otherwise logistic regression stays, and the transformer is reported as the comparison. Either way
the report gives both models' accuracy and cost side by side.

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
- **The owner's decision (D-14):** "real time" means the ranking updates as the findings are
  entered. It is measured first, and built into the dashboard if it is fast enough and helps.

## The finished project (the original scope) and its dates

**The finished project contains:**
- the Streamlit dashboard and the FastAPI service;
- the patient knowledge graph (Synthea);
- the small transformer fused with the medical knowledge graph;
- calibration;
- the red-flag rules;
- "what to ask next";
- literature retrieval and a grounded local-LLM explanation with citation checks;
- the full ablation study.

**Dates:**

| Milestone | Date |
|---|---|
| Transformer trained and scored | 14–20 Oct |
| Prototype (`v0.1`) | 25 Oct |
| Feature-complete | 3 Nov |
| Test split opened once | 6–7 Nov |
| **Working prototype and report presented** | **15 Nov** |

## Questions for the supervisor

1. We chose a small transformer as the deep model, reported beside logistic regression and
   XGBoost with its size and cost. Is one well-justified deep model acceptable, or do you expect a
   comparison of several deep families?
2. Is "safe first, then accurate on incomplete histories" the right meaning of "accurate results"?
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
