# Contributing to Nightingale

Nightingale is a university major project built by a team of four over 8–10 weeks. These notes get
you productive quickly; the detail lives in [`docs/`](docs/README.md).

---

## Before you write code

0. Read [`PROGRESS.md`](PROGRESS.md) — where the project is right now, what is in flight, and what is
   waiting on a decision. **Updating it is part of every task** (its §4 protocol): mark your task
   In-flight before you start, and tick it off in the same commit as the work. This is how four
   people and any number of Claude sessions stay in sync without asking "where are we?"
1. Read [`docs/09-prerequisites.md`](docs/09-prerequisites.md) — including the **cardiac domain
   primer**. You cannot sanity-check clinical output you do not understand.
2. Read [`docs/02-architecture.md`](docs/02-architecture.md) §4 — the interface contracts.
3. Read [`docs/06-engineering-conventions.md`](docs/06-engineering-conventions.md) — branching,
   style, definition of done.

## Setup

```bash
git clone https://github.com/HarshRohila02/nightingale.git && cd nightingale
py -3.11 -m venv .venv                # team standard is Python 3.11 (D-2); macOS/Linux: python3.11 -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128   # GPU build, BEFORE requirements
pip install -r requirements.txt
docker compose up -d                  # Neo4j
python scripts/download_data.py       # fetches DDXPlus, BODHI-S, UCI Heart
```

## Workflow

```bash
git checkout -b feat/p2-your-thing
# ... work ...
black src tests && ruff check src tests && pytest -q
git commit -m "feat(ml): add isotonic calibration to the ranker"
git push -u origin feat/p2-your-thing   # then open a PR
```

- Branch names carry your role: `feat/p1-…`, `fix/p3-…`
- Rebase on `master` before opening a PR; keep branches under 3 days
- Every PR needs one review. **Changes to `src/contracts.py` need all four.**

## Non-negotiables

| Rule | Why |
|---|---|
| **Never commit anything from `data/`** | Licensing and repo size; data is reproducible from scripts |
| **Never commit secrets or API keys** | — |
| **Never tune on the test set** | Invalidates the study — see `docs/05-evaluation-protocol.md` |
| **Never use `DIFFERENTIAL_DIAGNOSIS` as a feature** | It is the label. This is leakage. |
| **Never remove the disclaimer** from any output path | Safety requirement FR-6.4 |
| **Never add a treatment or drug recommendation** | Out of scope and unsafe — FR-6.5 |
| **Golden clinical cases must stay green** | They encode clinical expectations |

## Clinical correctness

This project ranks potentially fatal conditions. If you see output that is clinically wrong — a
benign condition ranked above an emergency for a dangerous presentation — **raise it immediately**,
even if every test passes. Tests encode what we thought to check; clinical judgement catches what we
did not.

Write clinical terms out in full: `pulmonary_embolism`, not `pe`.

## Logging experiments

Any run that produces a metric gets an entry in
[`docs/08-experiment-log.md`](docs/08-experiment-log.md) — including runs that failed or
underperformed. Negative results are results.

## Questions

Ask at standup, or open a draft PR early and ask in it. A day blocked is 1% of this project.
