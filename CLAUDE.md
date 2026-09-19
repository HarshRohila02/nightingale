# CLAUDE.md — Nightingale

**Nightingale — Clinical Intelligence & Reasoning System.** A university major project: a
clinician-facing decision-support prototype that ranks the causes of **acute chest pain**, using an ML
ranker fused with a cardiac knowledge graph, independent red-flag rules, evidence retrieval (RAG) and a
guardrailed local LLM. Four people, 8–10 weeks. **Decision support only — never a diagnosis.**

## Where we are — read this first

`PROGRESS.md` is the single source of truth for the current phase and sub-phase, in-flight work,
pending user decisions and history. It is imported below so it loads in every session. **Do not infer
progress from the folder contents, and follow its update protocol (§4) on every task** — including
the write-ahead In-flight marker, so an interrupted session can always be recovered.

@PROGRESS.md

> If the contents of `PROGRESS.md` do not appear above, read it with the Read tool before doing
> anything else.

## Working agreements (set by the user)

- **Ask before any lengthy process** — large downloads, model pulls, long training runs, heavy installs.
- **Keep the owner's laptop light. Where each job runs is the owner's call** (`docs/11` §1, D-7):
  - Run locally without asking: tests, lint, formatting, and short scripts expected to finish in
    **under ~5 minutes**.
  - **Ask first, and ask where**, before **every model training or tuning run on project data**
    (however short or small the sample; a unit test fitting a toy model on synthetic rows is a
    test) and before **any job expected to take over ~5 minutes, CPU jobs included**. The options are the
    university GPU, Colab, Kaggle, Lightning AI / Studio Lab, or the laptop. Suggest the cloud, and
    give the expected runtime.
  - **The laptop GPU is for short tests only, and each test needs the owner's yes in chat (D-9).**
    Before any GPU use on the laptop (a CUDA check, a short model load, any Ollama model), say
    what the test does and how long it takes, then ask. Run it only after a yes. A yes covers that
    one test, never later ones. Keep tests to minutes; anything longer is a cloud job. Afterwards,
    unload the model or release the GPU, and report the result. The owner can also run tests by
    hand (`docs/11` §2–§3).
  - Install packages **per task, as needed** (D-4), and ask before each download. Code stays on the
    CPU unless `compute.device` in `configs/config.yaml` says otherwise.
- **Commit and push at every sub-phase boundary**, with `PROGRESS.md` updated in the same commit.
- **Do not delete or move files** without explicit approval.
- If something contradicts what was previously reported to the user, say so plainly.

## Hard rules

- Never commit anything under `data/`.
- Never use DDXPlus `DIFFERENTIAL_DIAGNOSIS` as a model input — it is the label. In the parquet it
  is `label_differential`; select features with `INPUT_COLUMNS` from `src/ddxplus.py`.
- Never touch the test split before Phase 4 (`docs/05` §2).
- Never remove or weaken the disclaimer; never add treatment or drug recommendations.
- `src/contracts.py` changes need all four team members — flag them rather than just making them.
- `docs/05-evaluation-protocol.md` is **frozen** — change it only through its amendment log.
- Golden clinical cases (`tests/fixtures/golden_cases.yaml`) must stay green. If one fails, fix the
  component, not the expectation.

## Gotchas already learned the hard way

- **DDXPlus English is machine-translated from French** and unreliable: `déchirante` (*tearing*, the
  aortic-dissection descriptor) becomes "heartbreaking". Use evidence **codes**, not English labels.
- DDXPlus evidences are **questions**; BODHI-S symptoms are **noun phrases**. Naive string matching
  between them reports a false 0%.
- BODHI-S `triples.jsonl` and `nl_facts.jsonl` are **line-aligned** (13,204 rows) — zip by index to
  decode its UUIDs. BODHI-S identifies conditions by SNOMED, DDXPlus by ICD-10.
- The KG backbone is DDXPlus `release_conditions.json`, not BODHI-S (R-01). That creates
  **circularity risk R-12**, which must be disclosed wherever results claim value for the KG.
  Every KG edge carries a `source` (`ddxplus` / `bodhi_s` / `hand_authored`); keep it that way.
- **KG concept ids come in two vocabularies, linked by the crosswalk** (`src/medical_kg/crosswalk.py`,
  `docs/02` §5.2). DDXPlus questions are `DDX:E_nn`; the red-flag rules and golden cases use
  hand-authored `SYM:*` / `RF:*` ids. Pass a hand-authored case through `expand_case()` before the
  real graph scores it, and get a DDXPlus patient's concepts with `concepts_from_evidences()`. Every
  new `SYM:*` / `RF:*` id needs a crosswalk entry, and a test enforces it;
  `canonical_concept_id()` decides which graph node it lives on.
- **The red-flag rules over-fire on DDXPlus** (EXP-014, R-15). 59% of validate patients get a flag,
  and the aortic-dissection rule flags 50%, because back radiation alone fires it. The golden cases
  pass on the real graph only because flagged candidates rank first: by graph score alone, GC-001's
  MI ranks fifth. Neither result shows the system working.
- **The KG knows questions, not answers** (`docs/02` §5.1). Stable angina's evidence set also sits
  entirely inside unstable angina's. So with rest pain present, overlap scoring still ranks stable
  angina (1.00) above must-not-miss unstable angina (0.875). 2a must fix this; until then, don't
  trust a KG-only ranking of the anginas. (DDXPlus's edges are question-level; the hand-authored
  edges, and BODHI-S's, are answer-level and weighted by likelihood bands, open decision A-6.)
- **On DDXPlus patients the graph alone scores 88% top-1, and that is circularity** (EXP-015,
  R-12): DDXPlus generated them from the definitions the graph is built from. Never report a KG
  number on DDXPlus data without saying so. Unstable angina is first for only 21% of its own.
  BODHI-S lowers the figure to 86%, and MI's to 60% (EXP-016): the overlap score punishes the four
  conditions BODHI-S enriches. 2a must replace the score before the enriched graph ranks anything.
- **BODHI-S is CC-BY-NC: never commit its text.** `src/medical_kg/bodhi_s.py` keys facts by
  BODHI-S's ids, with paraphrased notes, and the KG card prints counts only.
- The system is **closed-world** (R-13): it only knows 13 conditions, so e.g. pneumonia gets forced
  into one of them. Say so wherever results are reported.
- DDXPlus demographics are synthetic: every condition is ~50% female, and MI has a median age of 45.
  **Near-equal by-sex results are an artifact, not evidence of fairness** (EXP-002).
- Patient evidence tokens come in four forms: binary, categorical value, **numeric ordinal**
  (`E_56_@_4`, 12.6% of tokens; encode as ordered, not one-hot) and the `V_11` "NA" sentinel.
  `EvidenceEncoder` (`src/ml/features.py`, `docs/03` §2.2) encodes all four into 607 columns
  fixed by the release files, not by patients. A trained model must record its `fingerprint`
  and refuse rows encoded with another.
- **XGBoost reads a sparse matrix's absent entries as missing, but a dense array's zeros as
  values**, and the same trees then answer differently. `XGBoostBaseline` (`src/ml/baselines.py`)
  turns every input into CSR; never call its booster directly. Models are saved as JSON (XGBoost:
  `.ubj`), never pickles, and trained models live in `models/`, which is gitignored.
- **A listed DDXPlus token can mean "no".** `E_204_@_V_10` means "did not travel" and is listed for
  89.6% of patients; `E_57_@_V_123` means "radiates nowhere". To find what a patient actually has,
  use `positive_codes()` in `src/ddxplus.py`, never "the code appears in EVIDENCES" (EXP-013).
- **The ground-truth differentials are open-world** (EXP-013): 33% of their probability mass is on
  conditions we can't output. **D-8, decided 2026-09-19** (`docs/05` amendment 1): the headline
  Precision@3 and Recall@5 count only the in-scope part of D (Recall@5 can reach 0.753), and
  Recall@5 with the full D (ceiling 0.434) is always reported beside it.
- **Score every system with `src/eval/metrics.py`**, which implements `docs/05` §3.1–§3.3 and
  §6 as amended, with a 95% bootstrap interval for every ratio metric (1,000 resamples,
  seed 42). Never report a figure without its interval. Red-flag precision counts a flag as
  appropriate only when it names the true condition, until the team settles A-7.
- **Red-flag sensitivity counts a must-not-miss condition's own patients** that its flag reaches
  (`docs/05` amendment 2). Report the rule's rate on everyone else beside it (R-15).
- The Windows console is **cp1252** — printing ⚠ or ✓ crashes unless stdout is reconfigured
  (see `scripts/demo.py`).
- The team standard is **Python 3.11** (`.venv` and CI). The machine's default `python` is still
  **3.14**, so plain `python` silently bypasses the venv — always use `./.venv/Scripts/python.exe`.
- Tool versions live in **`requirements-dev.txt`**, the single source shared with CI. Bump black or
  ruff there, never in one place only.
- The laptop GPU is an **RTX 5060 (Blackwell, `sm_120`)**. GPU tests run from a separate
  **`.venv-gpu`** (torch ≥ 2.7 built for CUDA 12.8), set up on 2026-09-18 with torch 2.11.0+cu128
  and verified with `scripts/check_gpu.py --device cuda` (`docs/11` §2). The
  main `.venv` stays CPU-only, so everyday work cannot touch the GPU. A cloud or university GPU
  needs the CUDA build that matches its own driver.
- **Ollama puts a model on the GPU automatically**, so running any local model is laptop-GPU use
  and needs the owner's yes first (D-9, `docs/11` §3). Forcing the CPU (`num_gpu: 0`) just turns it
  into a heavy CPU job, which also needs asking. `ollama stop <model>` frees the GPU at once.
- **Neo4j runs on AuraDB Free** (D-6). Its credentials live only in `.env`: never open, print or
  commit `.env`. The graph sits under the label `CardiacKG`, and `scripts/load_neo4j.py` writes it
  when the local build changes. **Aura's home database is named after the instance id, not
  `neo4j`**: never hard-code a database name. The Neo4j store reads the graph once at start-up
  and scores it in memory, so it answers exactly as NetworkX does; when Aura is paused or
  offline, `open_graph_store()` falls back to NetworkX and reports `graph_backend`. The live
  tests (marker `neo4j`) run in CI against a throwaway container, and locally only with
  `NEO4J_TEST_DOTENV=1`; they use the `NightingaleTest` namespace and delete it afterwards.

## Commands (Windows venv)

```bash
./.venv/Scripts/python.exe -m pytest                        # tests, incl. golden clinical cases
./.venv/Scripts/python.exe -m black src tests scripts       # format
./.venv/Scripts/python.exe -m ruff check src tests scripts  # lint
./.venv/Scripts/python.exe scripts/demo.py --case GC-003    # walking skeleton
./.venv/Scripts/python.exe scripts/build_ddxplus_chestpain.py   # 1a: validate.csv -> parquet
./.venv/Scripts/python.exe scripts/build_cardiac_kg.py          # 1b: build the KG, print its card
./.venv/Scripts/python.exe scripts/check_crosswalk.py           # 1b: crosswalk, red flags, golden cases on real data
./.venv/Scripts/python.exe scripts/load_neo4j.py                # 1b: write the KG to AuraDB if it changed; check it
# scripts/train_baselines.py trains B0/B1: a training run, so only where the owner says (D-7);
# B0/B1 run on Colab via notebooks/colab_b0_b1.ipynb (docs/11 §4.1)
```

## Where things are

`docs/README.md` document index · `docs/02` architecture and contracts · `docs/03` §2.1 the
chest-pain parquet, §2.2 the model features (`src/ml/features.py`) · `src/eval/metrics.py` the
`docs/05` metrics · `docs/05` evaluation protocol (frozen) · `docs/07` risk register · `docs/08`
experiment log · `docs/09` learning guide · `docs/10` R-01 spike report · `src/ddxplus.py` DDXPlus
decoding · `src/medical_kg/` the KG from its three sources (DDXPlus, `hand_authored.py`,
`bodhi_s.py`, merged by `cardiac_kg.py`), its NetworkX and Neo4j stores and the crosswalk (cards: `docs/02`
§5.1–§5.2) · `docs/11`
compute runbook: where jobs run, the owner's GPU steps, cloud jobs, AuraDB
