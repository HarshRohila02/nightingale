# 09 — Prerequisites & Learning Guide

**Purpose:** everything the team needs to know *before* and *during* the build, what is already
assumed, and what each role must pick up. Work through the shared section in Week 1 alongside
Phase 0 documentation; role sections can be learned just-in-time at the start of your phase.

> **Baseline assumed for everyone:** comfortable with Python, basic machine learning
> (train/test split, classification, overfitting), and able to read a Jupyter notebook. If that is
> not true for you, say so now — not in Week 4.

---

## 1. Shared prerequisites — everyone, Week 1

### 1.1 Tooling (half a day)

| Skill | Why it matters here | Get to this level |
|---|---|---|
| **Git branching & PRs** | Four people on one repo; merge conflicts will happen | Create a branch, rebase, open a PR, resolve a conflict |
| **Virtual environments** | Reproducible results are a grading criterion | `venv` + pinned `requirements.txt` |
| **Docker basics** *(optional since 2026-09-18)* | Only needed for a local Neo4j. The team uses AuraDB Free ([11](11-compute-runbook.md) §5) | `docker compose up -d`, read logs, stop/remove |
| **Type hints + dataclasses** | All inter-module contracts are typed | Read and write `@dataclass` with `list[str]`, `Optional[...]` |
| **pytest** | Golden clinical cases are enforced in CI | Write a test, use fixtures, run one test file |

### 1.2 The medical domain primer (1 day — *do not skip this*)

You are building a cardiac triage tool. You cannot sanity-check your own output if you do not know
what the words mean. Every team member should be able to explain:

**Anatomy of the complaint**
- **Chest pain descriptors** and why each matters diagnostically:
  - *character* — crushing/pressure (ischaemic) vs sharp/stabbing vs burning (reflux)
  - *radiation* — to jaw / left arm / between shoulder blades (the last suggests dissection)
  - *onset* — sudden (PE, pneumothorax, dissection) vs gradual
  - *aggravating/relieving* — worse on exertion (ischaemic), worse lying flat / better sitting
    forward (pericarditis), worse on inspiration = **pleuritic** (PE, pneumothorax, pericarditis)

**The 13 in-scope conditions** — for each, know the 3–4 hallmark features. At minimum:

| Condition | Hallmarks you must recognise |
|---|---|
| **Acute MI (NSTEMI/STEMI)** | Crushing central pain, radiates jaw/arm, diaphoresis, exertional, risk factors |
| **Unstable / stable angina** | Exertional; unstable = at rest or worsening pattern |
| **Pericarditis** | Pleuritic, positional (better sitting forward), often post-viral |
| **Myocarditis** | Post-viral, chest pain + breathlessness, can mimic MI |
| **Acute pulmonary edema** | Severe breathlessness, orthopnoea, frothy sputum, crackles |
| **Atrial fibrillation / PSVT** | Palpitations, irregular (AF) or sudden-onset regular (PSVT) tachycardia |
| **Pulmonary embolism** | Pleuritic pain, breathlessness, **unilateral leg swelling**, immobilisation/surgery |
| **Spontaneous pneumothorax** | Sudden pleuritic pain + breathlessness, tall thin young male |
| **Boerhaave** | Severe pain after forceful vomiting — rare, catastrophic |
| **GERD** | Burning, post-prandial, worse lying flat, relieved by antacids |
| **Panic attack** | Palpitations, hyperventilation, paraesthesia, situational |
| **Aortic dissection** *(KG-only)* | **Tearing** pain radiating to back, BP difference between arms |

**The confirmatory tests** we reference in "what to ask next":
- **Troponin** — cardiac muscle damage marker; rises in MI
- **ECG** — ST elevation/depression; the key ACS discriminator
- **D-dimer** — sensitive but non-specific; used to rule *out* PE
- **Chest X-ray** — pneumothorax, pulmonary edema

**Why "must-not-miss" drives our design:** ranking GERD above MI is not merely a wrong answer — it
is a potentially fatal one. This asymmetry is why our headline metric is must-not-miss recall, not
accuracy. Read [04-safety-ethics.md](04-safety-ethics.md) once you understand the above.

*Resources:* any standard emergency-medicine chest-pain chapter; Life in the Fast Lane (LITFL) and
Geeky Medics have free, readable summaries. **One hour of reading here saves days of debugging
nonsense output.**

### 1.3 Project-specific concepts (half a day)

- **Differential diagnosis** = a *ranked list* of candidates, not one answer. Our ground truth
  (DDXPlus `DIFFERENTIAL_DIAGNOSIS`) is itself a ranked list with probabilities.
- **Decision support vs diagnosis** — the legal and ethical line this project must never cross.
- **Knowledge graph** — nodes + typed edges; here `Condition —HAS_SYMPTOM→ Symptom`.
- **Calibration** — a model saying "70%" should be right 70% of the time. Raw scores are *not*
  probabilities until proven.

### 1.4 Compute: where jobs run (30 minutes) · *updated 2026-09-18*

The rules are in [11-compute-runbook.md](11-compute-runbook.md) §1 (decision D-7). In short:

- Tests, lint and scripts that take **under ~5 minutes** run on your own machine.
- **Every training or tuning run, and any job over ~5 minutes, runs where the project owner
  decides.** That is usually the cloud or the university GPU. Code that could use a GPU reads
  `compute.device` from `configs/config.yaml`, which defaults to `cpu`.
- **The owner's laptop GPU is only for short tests, each needing the owner's yes first** (D-9). Three things use
  a GPU without saying so:
  - torch code that picks CUDA automatically, once the CUDA build is installed;
  - XGBoost with `device="cuda"`;
  - **any model run through Ollama**, which loads onto the GPU by default.

Cloud skills, for whoever runs a heavy job:

| Skill | Why it matters here | Get to this level |
|---|---|---|
| **Colab / Kaggle / Lightning AI notebooks** | Heavy jobs run there | Pick a GPU runtime; clone the repo at a given commit; install the requirements; fetch the data from Hugging Face ([11](11-compute-runbook.md) §4) |
| **Saving work before a session ends** | Free sessions end without warning | Write outputs as you go (Google Drive in Colab, `/kaggle/working` in Kaggle) and resume from the last checkpoint |
| **Keeping the data private** | BODHI-S is non-commercial and DDXPlus needs attribution | Private notebooks only; never publish the data, or a model trained on it |

**If the university GPU is granted** (risk R-14), whoever sets it up should learn the following
before Phase 3:

| Skill | Why it matters here | Get to this level |
|---|---|---|
| **SSH** keys, `scp` / `rsync` | All access is remote | Log in with a key; sync the repo. `data/` is never committed, so re-create it there with `scripts/download_data.py` |
| **The cluster's job scheduler** (often SLURM) | GPU jobs are usually queued, not run interactively | Submit a one-GPU job (`sbatch` with `--gres=gpu:1`), watch it (`squeue`), cancel it (`scancel`), read its log |
| **Matching torch to the driver** | The cluster's CUDA driver decides which torch build works | Read `nvidia-smi`, choose the matching PyTorch CUDA index, build a Python 3.11 env |
| **Serving Ollama remotely** | The app calls the model over HTTP | Set `OLLAMA_HOST`, or tunnel it: `ssh -L 11434:localhost:11434 <host>` |

---

## 2. Role-specific prerequisites

### P1 — Cardiac Medical KG

| Topic | Depth needed | Resource |
|---|---|---|
| **Graph data modelling** | Nodes, relationships, properties, when to use a property vs a node | Neo4j "Graph Data Modeling" free course |
| **Cypher query language** | `MATCH`, `WHERE`, variable-length paths `-[:HAS_SYMPTOM*1..2]-`, `WITH`, aggregation | Neo4j GraphAcademy (free, ~4 h) |
| **Neo4j Python driver** | Sessions, parameterised queries, batch writes | Official driver docs |
| **Graph algorithms** | Personalised PageRank, degree/weight scoring — conceptually | NetworkX docs; Neo4j GDS overview |
| **Text parsing** | Regex/rule parsing of BODHI-S triples into edges | Python `re` |
| **Neo4j AuraDB** *(added 2026-09-18)*: the team's Neo4j runs in Neo4j's cloud (D-6) | Connect the Python driver to the `neo4j+s://` URI read from `.env`; never hard-code or print the password | Aura docs; Neo4j Python driver docs |
| **NetworkX** *(added 2026-09-18)*: the KG's working backend until Neo4j runs (D-6) | `MultiDiGraph` with keyed parallel edges, node and edge attributes, `out_edges(data=True)`, `nx.freeze`; `pagerank(personalization=…)` for 2a | NetworkX tutorial |
| **Vocabulary mapping** *(added 2026-09-19)*: how the crosswalk links hand-authored concepts to DDXPlus answers | The SKOS mapping relations (exact, close, broader, narrower, related) and **which way each lets a finding travel**: a broader answer is implied by the concept, a narrower one implies it, and a denial carries over only to a yes/no question. Why UMLS and OMOP record mappings the same way | W3C SKOS Primer §4.5 (mapping properties); `src/medical_kg/crosswalk.py` and [02-architecture.md](02-architecture.md) §5.2 |

*Where to start (updated 2026-09-19):* the KG is built from three sources (DDXPlus, the
hand-authored aortic dissection, BODHI-S), with the crosswalk linking the vocabularies. Read
`src/medical_kg/`, the KG card and the crosswalk card in [02-architecture.md](02-architecture.md)
§5.1–§5.2, above all their **known limitations**, and EXP-014 to EXP-016 in
[08](08-experiment-log.md). Done so far:
1. ~~**The crosswalk** from `DDX:E_nn` to the `SYM:*` / `RF:*` ids used by the red-flag rules and the
   golden cases.~~ Done 2026-09-19 (§5.2).
2. ~~**BODHI-S enrichment.**~~ Done 2026-09-19: each fact maps to the concepts it implies,
   keeping its qualifiers where the vocabulary has a concept for them (`src/medical_kg/bodhi_s.py`).
3. ~~**Aortic dissection**, hand-authored.~~ Done 2026-09-19 from the ADD-RS and IRAD
   (`src/medical_kg/hand_authored.py`).

Next: the Neo4j store on AuraDB (once the owner's instance exists), then 2a, **a score that does
not punish a condition for having more evidence** (EXP-016) and separates the anginas
(EXP-015). Read up on naive-Bayes and likelihood-ratio scoring as well as Personalised PageRank.

Every edge must carry its `source`, because the R-12 disclosure reports the KG by source.

### P2 — Data & ML

| Topic | Depth needed | Resource |
|---|---|---|
| **Multi-class ranking** | Predicting a ranked list, not one label; `predict_proba` | scikit-learn user guide |
| **Class imbalance** | Some of our 13 conditions will be rare | `class_weight`, stratified splits |
| **XGBoost** | Basic training + tuning | XGBoost docs |
| **Calibration** | Platt scaling, isotonic regression, reliability curves, ECE, Brier | `sklearn.calibration` |
| **Ranking metrics** | Top-k accuracy, MRR, Precision@k, Recall@k, nDCG | Read [05-evaluation-protocol.md](05-evaluation-protocol.md) |
| **SHAP** | Feature attribution for explanations | SHAP docs, `TreeExplainer` |
| **Leakage discipline** | Patient-level splits; never inspect the test set | — |
| **Parquet with list columns**: the chest-pain data is one parquet per split, with list and struct columns | `pd.read_parquet`, `Series.explode`, reading a `pyarrow` schema | pandas / pyarrow docs |

*First task you will hit:* ~~decoding the coded evidences~~, which is done (task 1a, 2026-09-18).
Read `src/ddxplus.py` and [03-data-management.md](03-data-management.md) §2.1 before writing
features. Three things matter:
- **four token forms:** binary (`E_91`), categorical value (`E_54_@_V_161`), numeric ordinal
  (`E_56_@_4`, to be encoded as ordered) and the NA sentinel (`E_54_@_V_11`);
- **tokens that mean "no":** a listed `E_204_@_V_10` means "did *not* travel". Use
  `positive_codes`, never "the code is listed";
- **select features with `INPUT_COLUMNS`:** every label column starts with `label_`, and one of them
  is the differential, which must never be an input.

### P3 — RAG, LLM & Safety

| Topic | Depth needed | Resource |
|---|---|---|
| **Embeddings & vector search** | Cosine similarity, indexing, top-k retrieval | sentence-transformers docs |
| **FAISS** | Build an index, search, persist | FAISS wiki |
| **Chunking strategy** | Why chunk size and overlap change retrieval quality | Any RAG primer |
| **BM25 / hybrid retrieval** | Lexical vs semantic; why combining beats either | `rank_bm25` |
| **Running a local LLM** | Ollama: pull, run, call from Python. Keep it on the CPU (`num_gpu: 0`) unless GPU use is approved (§1.4) | Ollama docs |
| **Prompt grounding & guardrails** | Constraining output to retrieved context; refusing when unsupported | — |
| **Prompt injection** | Free-text intake is untrusted input | OWASP LLM Top 10 |
| **PubMed E-utilities** | Programmatic literature fetch; rate limits | NCBI E-utilities docs |

*First task you will hit:* PubMed/PMC fetching is slow and rate-limited — start the corpus download
in Week 2 even though you will not use it until Week 6.

### P4 — Platform & Evaluation

| Topic | Depth needed | Resource |
|---|---|---|
| **FastAPI** | Routes, Pydantic models, async basics | FastAPI tutorial |
| **Pydantic** | Validation, nested models — these *are* our contracts | Pydantic docs |
| **Streamlit** | Widgets, layout, state, caching | Streamlit docs (~3 h) |
| **pytest + fixtures** | Golden-case regression tests | pytest docs |
| **GitHub Actions** | Run tests on push | Actions quickstart |
| **MLflow** | Log params/metrics, compare runs for the ablation table | MLflow tracking docs |
| **Experiment design** | Baselines, ablations, why pre-registration matters | [05-evaluation-protocol.md](05-evaluation-protocol.md) |

---

## 3. Suggested Week-1 schedule

| Day | Everyone | In parallel |
|---|---|---|
| 1 | Tooling setup; repo clone; env working | P4 drafts charter + SRS |
| 2 | **Medical domain primer** | P2 begins DDXPlus download |
| 3 | Project concepts; read architecture contracts | **Feasibility spike begins** |
| 4 | Role-specific learning | Spike continues |
| 5 | Role-specific learning | **Spike decision gate**; docs finalised |

## 4. Honest notes

- **Do not try to learn everything up front.** The shared section and your role's first two rows
  are enough to start. Learn the rest when you reach it.
- **The medical primer is the one thing people skip and regret.** If your model ranks GERD above MI
  for a crushing exertional pain with jaw radiation, you need to *notice* that it is wrong.
- **Cypher is easier than it looks** (roughly SQL for graphs) — budget 3–4 hours, not days.
- **Calibration is the most commonly misunderstood topic** on this list and is directly assessed in
  our evaluation. P2 should not defer it.

---

## 5. Knowledge checkpoints

Before Phase 1 you should be able to answer:

1. Why is *must-not-miss recall* a better headline metric than top-1 accuracy here?
2. What distinguishes pericarditis from myocardial infarction on history alone?
3. Why can a raw model score not be reported as a probability?
4. What is the difference between the patient KG and the medical KG?
5. Which of our 13 conditions is **not** in the training data, and how is it handled?

*(Answers: see §1.2, §1.3, [04-safety-ethics.md](04-safety-ethics.md), and
[02-architecture.md](02-architecture.md).)*
