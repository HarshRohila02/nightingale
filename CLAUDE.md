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
- **Never use the laptop GPU without asking first.** The team is trying to get a university GPU; the
  owner's RTX 5060 is used only with explicit permission, asked before each new use — a CUDA torch
  install, a GPU training run, running a model through Ollama. A yes covers that use only. Default to
  CPU (`compute.device: cpu` in `configs/config.yaml`). Where GPU work runs is decision **D-7**.
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
- **KG concept ids come in two vocabularies.** DDXPlus questions are `DDX:E_nn`; the red-flag
  rules and golden cases use hand-authored `SYM:*` / `RF:*` ids. Until the crosswalk links them,
  the pipeline and golden cases stay on the stub store.
- **The KG knows questions, not answers** (`docs/02` §5.1). Stable angina's evidence set also sits
  entirely inside unstable angina's. So with rest pain present, overlap scoring still ranks stable
  angina (1.00) above must-not-miss unstable angina (0.875). 2a must fix this; until then, don't
  trust a KG-only ranking of the anginas.
- The system is **closed-world** (R-13): it only knows 13 conditions, so e.g. pneumonia gets forced
  into one of them. Say so wherever results are reported.
- DDXPlus demographics are synthetic: every condition is ~50% female, and MI has a median age of 45.
  **Near-equal by-sex results are an artifact, not evidence of fairness** (EXP-002).
- Patient evidence tokens come in four forms: binary, categorical value, **numeric ordinal**
  (`E_56_@_4`, 12.6% of tokens; encode as ordered, not one-hot) and the `V_11` "NA" sentinel.
- **A listed DDXPlus token can mean "no".** `E_204_@_V_10` means "did not travel" and is listed for
  89.6% of patients; `E_57_@_V_123` means "radiates nowhere". To find what a patient actually has,
  use `positive_codes()` in `src/ddxplus.py`, never "the code appears in EVIDENCES" (EXP-013).
- **The ground-truth differentials are open-world** (EXP-013): 33% of their probability mass is on
  conditions we can't output, so `Recall@5` as `docs/05` defines it tops out at 0.434. Decision
  D-8 is open. Don't report that metric until it is settled.
- The Windows console is **cp1252** — printing ⚠ or ✓ crashes unless stdout is reconfigured
  (see `scripts/demo.py`).
- The team standard is **Python 3.11** (`.venv` and CI). The machine's default `python` is still
  **3.14**, so plain `python` silently bypasses the venv — always use `./.venv/Scripts/python.exe`.
- Tool versions live in **`requirements-dev.txt`**, the single source shared with CI. Bump black or
  ruff there, never in one place only.
- The laptop GPU is an **RTX 5060 (Blackwell)**. If the owner ever approves using it, torch must be
  ≥ 2.7 with CUDA 12.8 (the cu128 index). Until then, install the **CPU** torch build. A
  university GPU would need the CUDA build that matches its own driver.
- **Ollama puts a model on the GPU automatically.** Even a smoke test of a local model counts as
  using the laptop GPU, so ask first, or force the CPU with the request option `num_gpu: 0`.

## Commands (Windows venv)

```bash
./.venv/Scripts/python.exe -m pytest                        # tests, incl. golden clinical cases
./.venv/Scripts/python.exe -m black src tests scripts       # format
./.venv/Scripts/python.exe -m ruff check src tests scripts  # lint
./.venv/Scripts/python.exe scripts/demo.py --case GC-003    # walking skeleton
./.venv/Scripts/python.exe scripts/build_ddxplus_chestpain.py   # 1a: validate.csv -> parquet
./.venv/Scripts/python.exe scripts/build_cardiac_kg.py          # 1b: build the KG, print its card
```

## Where things are

`docs/README.md` document index · `docs/02` architecture and contracts · `docs/03` §2.1 the
chest-pain parquet · `docs/05` evaluation protocol (frozen) · `docs/07` risk register · `docs/08`
experiment log · `docs/09` learning guide · `docs/10` R-01 spike report · `src/ddxplus.py` DDXPlus
decoding · `src/medical_kg/` the KG and its NetworkX store (card: `docs/02` §5.1)
