# 11 — Compute Runbook: where jobs run, laptop GPU tests, cloud jobs

**Version:** 1.4 · 2026-09-20 (§4.1: the B0/B1 job has run; v1.3, §4.1: the B0/B1 Colab job; v1.2, §5: the AuraDB instance exists and holds the graph) · **Set by:** the project owner (decisions D-4, D-6, D-7 and D-9 in
`PROGRESS.md` §6)

> **The owner's laptop is a development machine, not a compute server.** Heavy work goes to the
> cloud or the university GPU. The laptop GPU is only for short tests. **Claude runs each test only
> after the owner says yes in chat** (D-9), and each yes covers one test. The owner can also run
> them by hand from §2–§3.

---

## 1. Where each job runs

| Job | Runs on | Ask first? |
|---|---|---|
| Tests, lint, formatting, editing | Laptop CPU | No |
| Short scripts expected to finish in **under ~5 minutes** (data builds, the KG card, quick checks) | Laptop CPU | No |
| **Every model training or tuning run**, however short | Cloud or university GPU; the laptop only if the owner picks it | **Yes: where it runs** |
| **Any other job expected to take over ~5 minutes**, CPU jobs included | Cloud or university GPU | **Yes: where it runs** |
| Batch LLM or embedding runs (B3/B4 baselines, embedding the corpus) | Cloud or university GPU | **Yes: where it runs** |
| **GPU tests on the laptop**: does CUDA work, a short model load, an Ollama smoke test | Laptop GPU (§2–§3): Claude runs the test after the owner's yes, or the owner runs it by hand | **Yes: every test** (D-9) |
| The Neo4j graph database | Neo4j AuraDB Free, in Neo4j's cloud (§5) | — |
| Installing packages | Laptop, one task at a time, only what that task needs | Yes, before each download |

**What counts as training.** Training a model on project data, even a small sample, is a training
run and needs asking. A unit test that fits a toy model on a few synthetic rows is a test.

**How Claude asks.** Before any job in a "Yes" row, Claude says what the job does, how long it
should take, what it produces, and asks where to run it: the university GPU (once available),
Google Colab, Kaggle, Lightning AI / Studio Lab, or the laptop. The cloud is the default suggestion.
For a **laptop-GPU test**, Claude says what the test does and how long it takes, runs it only after
a yes in chat, then frees the GPU and reports the result. A yes covers that one test, never later
ones.

---

## 2. Laptop GPU: one-time setup (done 2026-09-18; ~20 minutes plus a ~3 GB download to redo)

GPU tests run from a **separate environment, `.venv-gpu`**, which has the CUDA build of PyTorch. The
main `.venv` keeps the CPU build, so everyday work *cannot* use the GPU. The GPU is used only when
you deliberately run something with `.venv-gpu`.

Run these in PowerShell, from the repository folder (`D:\PRJ-1\medical`).

**Step 1. Check the driver.**

```powershell
nvidia-smi
```

The table must name the **NVIDIA GeForce RTX 5060 Laptop GPU**, and the top-right corner must show
**CUDA Version: 12.8 or higher**. That figure is the newest CUDA your driver supports. If it is lower,
update the driver with the NVIDIA App or from your laptop maker's support page, restart, and check
again.

**Step 2. Create the GPU environment.** `.venv-gpu/` is gitignored.

```powershell
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gpu\Scripts\python.exe -m pip install --no-cache-dir "torch~=2.7" --index-url https://download.pytorch.org/whl/cu128
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-dev.txt
```

The torch download is about 3 GB. The RTX 5060 is a Blackwell GPU, which needs torch 2.7 or newer
**built for CUDA 12.8** (`cu128`). Older builds and CPU builds cannot use it. `torch~=2.7` installs
the newest 2.x release on the cu128 index (2.11.0 on 2026-09-18). `--no-cache-dir` stops pip from
keeping a second 3 GB copy of the download on the C: drive.

**Step 3. Check that it works.** This takes a few seconds.

```powershell
.\.venv-gpu\Scripts\python.exe scripts\check_gpu.py --device cuda
```

A good result has these lines: **CUDA available** is `True`, **device** names the RTX 5060,
**compute capability** is `12.0`, **sm_120 in build** is `yes`, and **matmul test** is `OK`. If you
ran it yourself, paste the whole output to Claude.

| If you see | It means | Fix |
|---|---|---|
| **torch build CUDA** is `None` | The CPU build of torch was installed | Repeat the torch line of step 2 |
| **sm_120 in build** is `NO` | The torch build is too old for Blackwell | Reinstall torch from the `cu128` index |
| **CUDA available** is `False` | The driver is too old, or Windows is not exposing the GPU | Redo step 1; check the GPU in Device Manager |

**Status on the owner's laptop: set up on 2026-09-18.** Claude ran steps 1–3, because the owner
asked it to. Driver 591.91 (CUDA Version 13.1) · Python 3.11.9 · torch 2.11.0+cu128, built for
`sm_75` to `sm_120` · compute capability 12.0 · `sm_120` in build: yes · matmul test OK at
6.6 TFLOP/s (float32, on the charger). The test suite also passes inside `.venv-gpu`: 89 tests pass,
and the 2 checks for a missing torch skip there. The environment takes 4.3 GB on D:. Re-run step 3
after a driver update.

---

## 3. Laptop GPU: running a test session

Claude runs these after the owner says yes in chat (D-9). The owner can also run them by hand.

**Before you start:** plug in the charger (laptop GPUs slow down on battery) and keep the vents clear.
Claude checks the power source before a test. A test should finish within minutes. Anything longer
is a cloud job (§4).

**To watch the GPU (optional, in a second window):** `nvidia-smi -l 2` refreshes every 2 seconds
(Ctrl+C stops it). Task Manager → Performance → GPU (NVIDIA) shows the same.

**PyTorch tests** always run with the GPU environment:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\<the test script>.py --device cuda
```

The GPU is released when the script ends. Ctrl+C stops it early.

**Ollama (local LLM) tests.** Nothing to download: `qwen2.5-coder:7b` is already installed.

1. Start Ollama from the Start menu, or run `ollama serve` in a terminal.
2. Run `ollama run qwen2.5-coder:7b "Reply with one word: ready"`.
3. In a second window, run `ollama ps`. The PROCESSOR column should say **100% GPU**. If it says
   CPU, update Ollama and the driver.

Measured on the owner's laptop on 2026-09-18 (Ollama 0.34.2, `qwen2.5-coder:7b`): 100% GPU with a
4,096-token context; the model took about 4.7 GB of the 8 GB; generation ran at about 71 tokens/s.
The first prompt after loading took 13.6 s, most likely GPU warm-up. A repeat took 0.3 s in total.

**To stop Ollama:** `ollama stop qwen2.5-coder:7b` frees the GPU memory at once; otherwise the model
stays loaded for about 5 minutes. Quit Ollama from its tray icon to stop the server itself. If Ollama
launches when you sign in to Windows, you can switch that off in Settings → Apps → Startup.

---

## 4. Cloud jobs: Google Colab (free), Kaggle, Lightning AI / Studio Lab

Every heavy job follows the same pattern, whatever the platform.

1. **Claude prepares the job** as a committed script in `scripts/`. It tells you the commit to use,
   the expected runtime, whether a GPU is needed, and the files the job will produce.
2. **Open a notebook** on the platform you chose, and set it up. The repository is public, so no
   token is needed:

   ```python
   !git clone https://github.com/HarshRohila02/nightingale.git
   %cd nightingale
   !git checkout <the commit Claude names>
   !pip install -q -r requirements-dev.txt huggingface_hub   # plus any extras Claude lists
   ```

   Our pins can be older than the platform's defaults (numpy 1.26, for example). If the platform
   asks you to **restart the session** after the install, restart, then carry on from
   `%cd nightingale`. Cloud runtimes may also run a newer Python than the team's 3.11; the code
   needs 3.10 or newer, and step 6 records the versions actually used.

3. **Fetch the data from Hugging Face**, never from GitHub or from your laptop:

   ```python
   from huggingface_hub import hf_hub_download
   for name in ["release_evidences.json", "release_conditions.json", "validate.csv"]:
       hf_hub_download("aai530-group6/ddxplus", name, repo_type="dataset",
                       revision="2ad986acc1ec62fb4a94171acc43f4fdd5bfde53",
                       local_dir="data/raw/ddxplus")
   ```

   The pinned `revision` is the snapshot already on the laptop, so the cloud and the laptop work from
   identical data. Training jobs add `train.csv` to the list. **`test.csv` is never fetched before
   Phase 4.**
4. **Run the commands Claude gives.** Job scripts save their outputs as they go, so a disconnect
   loses little.
5. **Bring back only small outputs**, such as a metrics JSON or a trained model file. Data and
   models are never committed.
6. **Log the run** in `docs/08`: the platform, the GPU type, the commit and the runtime. Save
   `pip freeze` next to the outputs.

| | Google Colab (free) | Kaggle Notebooks | Lightning AI / Studio Lab |
|---|---|---|---|
| **Choose a GPU** | Runtime → Change runtime type → T4 GPU | Settings → Accelerator → GPU. Needs a phone-verified account | Chosen when you start the studio or runtime |
| **Internet** | On by default | Settings → Internet → On. Needs a phone-verified account | On |
| **Limits** | A session ends after some hours, or sooner when idle. Free GPU access is not guaranteed | A weekly GPU-hour quota, shown in your account. **Save Version → Save & Run All** keeps running with the browser closed | Free monthly hours or credits. Storage persists between sessions |
| **Where outputs go** | Google Drive: `from google.colab import drive; drive.mount('/content/drive')` | `/kaggle/working`, which becomes the notebook's Output tab | The studio's persistent storage |

**Rules on every platform:**

- **Keep notebooks and outputs private.** Never publish DDXPlus or BODHI-S data, or a model trained
  on them, as a public notebook, dataset or model. BODHI-S is CC-BY-NC-4.0 (`docs/03` §1.2).
- **Never put a secret in a cell.** Nothing here needs a token. If something ever does, use the
  platform's secrets store: the key icon in Colab, Add-ons → Secrets in Kaggle.
- **The test split stays closed until Phase 4.** The scripts refuse it unless `configs/config.yaml`
  allows it, and that file comes with the pinned commit.

### 4.1 The B0/B1 job (EXP-003, EXP-004) · *ready 2026-09-19 · run 2026-09-19*

The owner chose Colab for it (D-7). Everything runs from one notebook in the repository, which
asks for a T4 GPU and falls back to the CPU when none is free:

1. Open <https://colab.research.google.com/github/HarshRohila02/nightingale/blob/master/notebooks/colab_b0_b1.ipynb>
   and sign in with your Google account.
2. In the first code cell, set `COMMIT` to the commit Claude names ("master" also works: the run
   records the exact commit).
3. Choose **Runtime → Run all**. If Colab warns that the notebook is not from Google, choose
   **Run anyway**: it is this repository's notebook.
4. Wait about 20–30 minutes. Most of it is downloading and decoding the 670 MB `train.csv`.
5. Your browser downloads **`nightingale_b0_b1.zip`**. Unzip it into `models/b0_b1/` in the
   project folder on the laptop; `models/` is gitignored, so it is never committed. Then tell
   Claude.

The zip holds the three models as JSON (`b0_prevalence.json`, `b1_logreg.json`, and XGBoost's
`b1_xgboost.ubj` with `b1_xgboost.meta.json`), `metrics.json` (every `docs/05` metric with its
95% interval), `run.json` (the commit, versions, GPU and timings), `features.json`,
`pip-freeze.txt` and the two splits' summaries. It holds no patient rows. The job is
`scripts/train_baselines.py`, and `tests/test_baselines.py` runs it end to end on a synthetic
mini-release in CI.

**Done 2026-09-19.** The owner ran the notebook at commit `68c14bd` on a Colab T4. The job itself
took about a minute: encoding 24 s, logistic regression 4 s, XGBoost 8 s on the GPU, scoring 20 s.
Colab ran Python 3.13, NumPy 2.1 and scikit-learn 1.6; the laptop has Python 3.11, NumPy 1.26 and
scikit-learn 1.9. The models are plain JSON, and the laptop reloads them and reproduces every
validate score to within 3 × 10⁻¹². Results: EXP-003 and EXP-004 in `docs/08`.

### 4.2 The R-18 retrain (EXP-018) · *ready 2026-09-24 · Colab, the owner's choice of 2026-09-25*

The owner chose Colab (D-7), as for 4.1, with `COMMIT = "4583e91"`. The notebook is
`notebooks/colab_b1_asked.ipynb`, and the steps are 4.1's with that notebook: open
<https://colab.research.google.com/github/HarshRohila02/nightingale/blob/master/notebooks/colab_b1_asked.ipynb>,
set `COMMIT`, **Runtime → Run all**. It takes about 25–35 minutes, most of it the same download and
decoding as 4.1; the two training runs take a few minutes each. It trains B1 twice on the train
split plus one masked copy of every patient: **B1 +aug** (the original features) and **B1′+aug**
(with the "asked" channel). Your browser downloads **`nightingale_b1_asked.zip`**; unzip it into
`models/` on the laptop, which gives `models/b1_aug/` and `models/b1_asked_aug/`, then tell
Claude. Validate is scored at full evidence only, and each bundle records what each model answers
to a patient with no findings. No patient rows are in the zip.

**Done 2026-09-25.** The owner ran the notebook at `4583e91` on a Colab T4; each variant took about
3½ minutes after the download. Unzipped, the bundles sit in `models/nightingale_b1_asked/b1_aug/` and
`.../b1_asked_aug/` (a folder named after the zip, as for 4.1). The laptop reloads all four models
and reproduces every validate score to within 5 × 10⁻¹². Results: EXP-018 in `docs/08`.

---

## 5. Neo4j AuraDB Free: one-time setup (the owner, ~10 minutes)

Neo4j runs in Neo4j's own cloud, so nothing extra runs on the laptop. The knowledge graph holds no
patient data, and ours (~100 nodes) is far below the free tier's limits.

1. Sign in to the Aura console at <https://console.neo4j.io> and create a **Free** instance. Only you
   can create the account.
2. When the instance is ready, **download the credentials file**. The password is shown only once.
3. Copy `.env.example` to `.env` if you haven't already. Then copy the URI, the username and the
   password from the credentials file into the `NEO4J_*` lines of `.env`. The URI looks like
   `neo4j+s://<instance-id>.databases.neo4j.io`. `.env` is gitignored. **Never paste the password
   into a chat or a commit.**
4. Tell Claude it is done. The Neo4j store task (1b) will connect using `.env`. Claude does not open
   or print `.env`.

A free instance pauses after 72 hours without use; resume it with **Play** on its card in the
console. After 30 days paused, Aura deletes it: then create a new one, update `.env` and run
`scripts/load_neo4j.py`. The Neo4j store falls back to the in-memory NetworkX graph whenever Aura
cannot be reached (`docs/02` §7). On 2026-09-24 the instance's host name did
not resolve at all, most likely because it had been paused; that is handled as unreachable too. Only `scripts/load_neo4j.py` writes to
Aura: `scripts/demo.py` reads it, and falls back rather than write when Aura's copy is stale. `docker-compose.yml` remains for anyone who prefers a local
Neo4j.

**Done 2026-09-19.** The owner created the instance, and the graph is loaded under the label
`CardiacKG`: 129 nodes and 321 edges. To refresh it after the graph changes, or to check it:

```bash
./.venv/Scripts/python.exe scripts/load_neo4j.py          # writes only if Aura's copy differs
./.venv/Scripts/python.exe scripts/load_neo4j.py --check  # compares, never writes
```

To look at the graph, open the instance's query tool in the console and run
`MATCH (c:CardiacKG:Condition)-[r]->(x) RETURN c, r, x LIMIT 100`. On AuraDB the home database
is named after the instance id, not `neo4j`, so leave `NEO4J_DATABASE` empty in `.env`.

---

## 6. University GPU: when access is granted

Access has not been granted yet (risk R-14). Whoever sets it up should first learn the skills in
[09-prerequisites.md](09-prerequisites.md) §1.4 (SSH, the job scheduler, matching torch to the
driver, serving Ollama remotely), and then add the connection steps to this section.
