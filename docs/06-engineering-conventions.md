# 06 — Engineering Conventions

**Version:** 1.0 · 2026-09-17 · **Owner:** P4

Four people, one repository, ten weeks. These conventions exist to prevent the two failures that
kill student projects: **merge chaos** and **nothing works on anyone else's machine.**

---

## 1. Environment

```bash
py -3.11 -m venv .venv        # macOS/Linux: python3.11 -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install "torch~=2.7" --index-url https://download.pytorch.org/whl/cpu    # CPU build first
pip install -r requirements.txt
cp .env.example .env          # Neo4j runs on AuraDB Free; add its details (docs/11 §5)
```

- **Where jobs run** (decision D-7, 2026-09-18; full rules in
  [11-compute-runbook.md](11-compute-runbook.md)). Tests, lint and scripts under ~5 minutes run
  locally. **Every training or tuning run, and any job over ~5 minutes (CPU jobs too), runs where
  the project owner decides**: the cloud (Colab, Kaggle, Lightning AI / Studio Lab), the
  university GPU, or the laptop. The owner's **laptop GPU is only for short tests the owner runs by
  hand**, from a separate `.venv-gpu`. Code that can use a GPU reads `compute.device` from
  `configs/config.yaml` (default `cpu`) and never picks CUDA on its own. That rules out
  `torch.cuda.is_available()` auto-selection, XGBoost `device="cuda"` by default, and Ollama
  calls, since Ollama puts a model on the GPU unless told `num_gpu: 0`.
- **Write long jobs to survive a disconnect.** Cloud sessions end without warning, so a job script
  must save its outputs as it goes and be able to resume.

- **Python 3.11** — the team standard (decision D-2), matching CI. The code only *needs* ≥ 3.10 (for
  `X | None` in the contracts), but everyone develops on 3.11 so behaviour and tool output are
  identical. Where the machine's default `python` is another version, create the venv explicitly
  with `py -3.11` (Windows) or `python3.11` — plain `python -m venv` will silently use the wrong one.
- **Tool versions** (pytest, black, ruff) live only in `requirements-dev.txt`, which CI installs too.
- **All dependencies pinned** (`package==1.2.3`). An unpinned dependency that silently upgrades
  mid-project and changes results is an avoidable disaster.
- Adding a dependency: justify it in the PR. Prefer the standard library.
- `RANDOM_SEED = 42` in `configs/` and used everywhere stochastic.

---

## 2. Branching

Trunk-based with short-lived branches. `master` is always runnable.

```
master                    ← always working; protected
  ├── feat/p1-kg-parser
  ├── feat/p2-ddxplus-decode
  ├── fix/negation-handling
  └── docs/evaluation-protocol
```

| Prefix | Use |
|---|---|
| `feat/` | New capability |
| `fix/` | Bug fix |
| `docs/` | Documentation only |
| `exp/` | Experiment/spike — may be messy, may be deleted |
| `refactor/` | Behaviour-preserving change |

**Rules**
- Branch names carry the owner: `feat/p3-faiss-index`.
- **Rebase on master before opening a PR.** Do not merge master into your branch repeatedly.
- Branches live **≤ 3 days**. Longer means the task was too big.
- Never commit directly to `master` except documentation typos.

## 3. Commits

Conventional-commits style:

```
<type>(<scope>): <imperative summary, ≤72 chars>

Why this change was needed. What approach was taken and why.
Anything a reviewer would otherwise have to ask.
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `exp`.
Scopes: `kg`, `ml`, `nlp`, `rag`, `llm`, `fusion`, `eval`, `api`, `ui`, `data`.

```
feat(kg): parse BODHI-S qualifiers into edge properties

BODHI-S encodes qualifiers inline ("Chest pain <radiate> to jaw").
Parse them into HAS_SYMPTOM properties rather than separate nodes,
so path queries stay single-hop and reasoning paths stay readable.
```

Commit **working increments**, not end-of-day dumps. Never commit secrets, data, or model binaries.

## 4. Pull requests

Every change to `master` goes through a PR with **at least one review**. P4 reviews infrastructure;
otherwise review across workstreams — it is the cheapest way to spread knowledge.

**PR must state:** what changed, why, how it was tested, and any contract impact.

**Reviewer checks:** contracts respected · tests present and passing · no data/secrets committed ·
no silent behaviour change in another module · docstrings on public functions.

A PR that changes `src/contracts.py` requires **all four** members to approve.

## 5. Code style

| Concern | Tool / rule |
|---|---|
| Formatting | **black** (line length 100) |
| Linting | **ruff** |
| Typing | Type hints on every public function; `mypy` advisory, not blocking |
| Docstrings | Google style on public functions; explain *why*, not *what* |
| Naming | `snake_case`; clinical terms spelled out (`myocardial_infarction`, not `mi`) |
| Imports | stdlib → third-party → local, separated |

Run before pushing: `black src tests && ruff check src tests && pytest -q`

**Clinical naming matters.** `pe` could be pulmonary embolism, physical examination, or a typo.
Write it out.

## 6. Testing

| Type | Scope | Requirement |
|---|---|---|
| **Unit** | Single function | Every non-trivial function in `src/` |
| **Contract** | Schema validation | Every model in `contracts.py` |
| **Golden clinical cases** | End-to-end vignettes | `tests/fixtures/golden_cases.yaml` — **must pass in CI** |
| **Integration** | Module boundaries | Each Protocol implementation |

**Golden cases are non-negotiable.** They encode clinical expectations as executable assertions —
e.g. *58 y/o male, exertional crushing chest pain radiating to jaw, diaphoresis, diabetic smoker* →
**ACS in top-3 and red flag fired.** If a change breaks one, the change is wrong until proven
otherwise.

Coverage target: **≥70% on core logic** (`ml`, `medical_kg`, `fusion`, `reasoning`). UI and scripts
exempt.

## 7. CI

GitHub Actions on every push and PR: install → `black --check` → `ruff` → `pytest`.
**A red build blocks merge.** Fix it or revert; never merge through a failure.

## 8. Definition of Done

A task is done when **all** hold:

- [ ] Code merged to `master` via reviewed PR
- [ ] Honours `src/contracts.py`; no ad-hoc schemas
- [ ] Unit tests written and passing; golden cases still green
- [ ] CI green
- [ ] Public functions typed and docstringed
- [ ] Documentation updated if behaviour or interface changed
- [ ] Experiment logged in [08-experiment-log.md](08-experiment-log.md) if it produced a metric
- [ ] Degrades gracefully if it depends on an optional component

## 9. Working agreements

- **Daily 15-minute standup:** done / doing / blocked. Blocked > 24 h escalates to the whole team.
- **Contracts are sacred.** Need a change? Raise it before writing code around it.
- **Stub, don't wait.** If your input isn't ready, use the stub and keep moving.
- **Log every experiment**, including failures — the ablation table is assembled from these.
- **Never tune on test.** See [05-evaluation-protocol.md](05-evaluation-protocol.md).
- **Ask on day 1, not day 5.** A blocked day costs 1% of the project.

## 10. Repository layout

See [02-architecture.md](02-architecture.md) §8. Rules: `src/` is importable library code;
`scripts/` is runnable entry points; `notebooks/` is exploration only — **nothing in a notebook is a
deliverable**; `data/` is gitignored.
