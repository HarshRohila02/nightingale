# 11 — Compute Runbook: where jobs run, laptop GPU tests, cloud jobs

**Version:** 1.5 · 2026-10-02 (§4.3: the seeds job, and what amendment 3a's item 5 makes of its logistic regressions; v1.4, §4.1: the B0/B1 job has run; v1.3, §4.1: the B0/B1 Colab job; v1.2, §5: the AuraDB instance exists and holds the graph) · **Set by:** the project owner (decisions D-4, D-6, D-7 and D-9 in
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

### 4.3 XGBoost's seeds 43–46 (EXP-020) · *ready 2026-09-25 · Colab, the owner's choice of 2026-09-25*

`docs/05` §6 (amendment 3) binds every retrained XGBoost to seeds 42–46; only seed 42 exists for
each XGBoost arm. The notebook is `notebooks/colab_b1_seeds.ipynb`, and the steps are 4.1's with
that notebook: open
<https://colab.research.google.com/github/HarshRohila02/nightingale/blob/master/notebooks/colab_b1_seeds.ipynb>,
set `COMMIT` to the commit Claude names, **Runtime → Run all**, and keep the tab open. It takes
about **50–60 minutes**: some 20 for the download and decoding, then about 8 per seed (from 4.1's and
4.2's timings on a T4). It runs `scripts/train_baselines.py` 13 times: seeds 43–46 of **B1-XGB**,
**B1-XGB+aug** and **B1-XGB′+aug**, and seed 42 of B1-XGB again as a check that must reproduce
EXP-004. Each run also refits B0 and logistic regression (the `+aug` logistic regressions vary with
the seed; they are kept for the errata's item 5, approved on 2026-10-02 as part of `docs/05`
amendment 3a, so they are B1-LR+aug's and B1-LR′+aug's seeds 43–46). A dropped session can
be re-run: finished runs are skipped. Your browser downloads **`nightingale_b1_seeds.zip`**; unzip
it into `models/nightingale_b1_seeds/`, then tell Claude. The seed-42 models stay the ones used;
the laptop scores every seed at 100%, 50% and 25% on the recorded masks (scoring only). No patient
rows are in the zip.

**Done 2026-10-01** (recorded 2026-10-11, EXP-020). The owner ran it at `b676f99`: all 13 runs, in
33 min 25 s of training (19:51–20:24 UTC). The bundle was unzipped into
`models/nightingale_b1_seeds/` on 2026-10-02. Seed 42 of B1 reproduces EXP-004 byte for byte. The
50%/25% scoring is still to do; it needs a script change.

---

### 4.4 The deep ranker B1-DL, a small transformer (EXP-021) · *ready 2026-10-12 · ~~a Colab GPU, the owner's choice of 2026-10-12~~ **Kaggle**, the owner's choice of 2026-10-13*

**Why Kaggle** (the owner, 2026-10-13): a Kaggle *commit* (**Save & Run All**) runs the whole
notebook in the background, so the browser can be closed once it has started. Colab needs the tab
open for the 2–4 hours. The notebook is `notebooks/kaggle_b1_dl.ipynb`; it runs the same
`scripts/train_deep.py` as the Colab one, and when Kaggle gives two GPUs (T4 x2) it trains two
seeds at once. ~~**Wait for one thing first:** the outcome of A-8's clinical review~~ (A-8 approved
2026-10-12, so nothing waits).

1. **Once:** a Kaggle account with a verified phone number (*Settings → Phone verification*).
   Kaggle allows GPUs and internet only on verified accounts.
2. **Create → New Notebook**, then **File → Import Notebook**: upload
   `notebooks/kaggle_b1_dl.ipynb` (or paste its GitHub link).
3. In the right-hand panel, under **Session options**: **Accelerator → GPU T4 x2**, and **Internet →
   on**. Not P100: recent PyTorch builds no longer support it.
4. Leave `COMMIT = "master"`, or set the commit Claude names. Every `run.json` records the exact
   commit either way.
5. **Save Version → Save & Run All (Commit) → Save.** The run starts in the background; **close the
   browser**.
   - The **smoke cell** trains one epoch on 1% of the patients, then the real run starts. If the
     smoke test fails, the version stops there, before it spends GPU time.
   - The **real run** trains both arms at seeds 42–46: ten networks, roughly **2–4 hours** on two
     T4s, plus about 20 minutes to download and decode the data. Then it scores every network at
     50% and 25% on the recorded masks, on the CPU (about 30–60 minutes). Kaggle's limits are
     12 hours a session and about 30 GPU hours a week.
6. Come back when Kaggle emails that the version has finished. Open the notebook, then **the latest
   version → Output**, and download **`nightingale_b1_dl.zip`**. `STATUS.txt` inside it says which
   arms finished. Unzip it into `models/nightingale_b1_dl/` and tell Claude.

**If the version stops early** (a limit, an error): open the notebook, choose **Add Input → Your
Work → Notebook Output Files** and pick its last version, then **Save & Run All** again. Finished
arms are copied from that output and skipped; the per-seed logs are in the output's `logs/`.

**Colab instead:** `notebooks/colab_b1_dl.ipynb` does the same with one GPU, saving to Google
Drive, but the browser tab must stay open: open
<https://colab.research.google.com/github/HarshRohila02/nightingale/blob/master/notebooks/colab_b1_dl.ipynb>,
**Runtime → Change runtime type → T4 GPU**, then **Run all**. Hugging Face's GPU hardware is paid.

The laptop then:
- re-scores seed 42 at 100%, 50% and 25% with the NumPy forward pass;
- applies D-13 (`src/eval/safety_floor.py`);
- runs the golden cases with the transformer;
- and, only if it passes, switches `configs/config.yaml` to `backend: deep`.

No patient rows are in the zip: the data is downloaded to `/tmp`, outside Kaggle's saved output.
**Keep the notebook private** (Kaggle's default; models trained on DDXPlus are never published).

---

### 4.5 The transformer's full report (EXP-021) · *ready 2026-10-13 · Kaggle, the owner's choice of 2026-10-13*

**From 2026-10-13, every cloud job runs on Kaggle** (the owner: "let's do all our work on kaggle as we
can do with broswer close as well"). The Colab sections above stay as records.

D-13 has already decided the ranker on seed 42. This job produces the rest of EXP-021's report:
- every seed of both transformer arms, against B1-LR(′)+aug and B1-XGB(′)+aug, at 100%, 50% and
  25%;
- McNemar's test and the paired bootstraps;
- D-13's first two checks at every seed.

It runs on the CPU, because the laptop's NumPy forward pass is what gets scored. The golden cases
were checked on the laptop for all ten networks (`scripts/report_deep.py --golden-only`). The
comparator models are not in the repository, so they travel as a private Kaggle dataset.

1. **Once: the models as a private dataset.**
   - On kaggle.com, **Create → New Dataset**, upload `models/kaggle_upload/nightingale_models.zip`
     (49 MB; written by Claude), name it `nightingale-models`, **Private**.
   - Models trained on DDXPlus are never published.
2. **Create → New Notebook → File → Import Notebook**: `notebooks/kaggle_exp021_reports.ipynb`.
3. **Add Input → Your Datasets → nightingale-models.**
4. **Session options:** Accelerator **None**, Internet **on**.
5. **Save Version → Save & Run All (Commit) → Save**, then close the browser.
   - The smoke cell (300 patients) stops a broken setup early.
   - The report should take 30–60 minutes. This is an estimate: about 25 minutes on the laptop's
     CPU, and Kaggle's CPU is slower.
6. From the latest version's **Output**, download `exp021_reports.zip`, unzip it into
   `data/interim/`, and tell Claude.

The notebook downloads DDXPlus's validation split only: not `train.csv`, never `test.csv`.

---

### 4.6 The evidence corpus for retrieval (task 3a, EXP-011) · *ready 2026-10-14 · Kaggle, the owner's choice of 2026-10-13*

This job builds the **retrieval bundle**: passages from **PMC Open Access** articles about the 14
conditions, and their embeddings. The dashboard then shows a few of them beside each top candidate,
each with its PMCID as a link and its licence. **Evidence never changes the ranking.** The laptop
runs no model for it, only NumPy (`src/rag/`).

The notebook is `notebooks/kaggle_retrieval_corpus.ipynb`. It runs `scripts/build_retrieval_corpus.py`:
- **PMC searches** for each condition, restricted to the Open Access subset, diagnosis-focused
  articles first; every search is recorded in the bundle.
- **NCBI's E-utilities with no API key**, at most 3 requests a second.
- **An article is kept only with a Creative Commons licence in its XML**, and the licence is
  recorded with it.
- **Diagnosis only.** No sentence with treatment language is kept, and no article whose title
  names a drug, a dose or a procedure (`src/rag/screening.py`; FR-6.5: the system never recommends
  a treatment).
- **The embedding** of the passages and of a fixed query vocabulary, with
  `pritamdeka/S-PubMedBert-MS-MARCO`.

**It never touches DDXPlus or any patient data**: nothing to keep out of the output but the corpus
itself.

1. **Create → New Notebook → File → Import Notebook**: `notebooks/kaggle_retrieval_corpus.ipynb`.
2. **Session options:** Accelerator **GPU T4 x2** (one GPU is used, for the embedding), Internet
   **on**. Without a GPU it still works, but the embedding takes about an hour instead of minutes.
3. In the first code cell:
   - Leave `COMMIT = "master"`, or set the commit Claude names. The bundle's manifest records the
     exact commit either way.
   - `PER_CONDITION = 150` articles per condition.
   - `EMAIL` is optional. NCBI asks tools for a contact address, so that it can write before
     blocking one. Leave it empty to send none.
4. **Save Version → Save & Run All (Commit) → Save**, then close the browser.
   - **When.** NCBI asks that large jobs run at weekends, or between 9 pm and 5 am US Eastern time
     (6:30 am to 2:30 pm in India). This one is about 200 requests, small by NCBI's standards,
     but starting it in that window is the polite choice.
   - The **smoke cell** builds a bundle of 2 conditions × 3 articles and loads it with the laptop's
     own code. It takes a minute or two, including the model's download (about 0.4 GB). If it
     fails, the version stops there.
   - The **real build** should take **30–60 minutes**. This is an estimate, not a measurement: the
     first run is the measurement.
     - About 3,000 articles are planned (150 × 1.5 per condition, fewer after overlaps). They are
       fetched in about 150 requests of 20 full-text articles; NCBI's response time, not the rate
       limit, should dominate.
     - Then chunking, a minute or two, and embedding 15,000–25,000 passages on one T4 in half
       precision, a few minutes.
     - Kaggle allows 12 hours a session.
5. From the latest version's **Output**, download **`retrieval_corpus.zip`**. Expect tens of MB:
   the passages' text, and float16 vectors at about 1.5 kB a passage. `STATUS.txt` inside it should
   start with `DONE`.
6. **On the laptop:**
   - **Stop the dashboard and the API first.**
   - Unzip the file into `data/raw/retrieval_corpus/`, so that
     `data/raw/retrieval_corpus/manifest.json` exists (the zip has no top folder). `data/` is
     gitignored: the corpus is never committed (`docs/03` §1.4).
   - In `configs/config.yaml`, set **`retrieval.bundle: data/raw/retrieval_corpus`**.
   - Start the dashboard again. Its caption should say "retrieval: BM25 + dense (RRF) over N
     passages from M PMC Open Access articles". Start-up takes a few seconds longer: about 4 s per
     20,000 passages, measured on a synthetic bundle.
   - **If the bundle fails its checks** (a file changed or truncated: every file's SHA-256 is in
     its manifest), the dashboard says "Running without: rag" and its log says why. Nothing else
     breaks.
   - Tell Claude: EXP-011's relevance judgements come next (`docs/08`).

**If the version stops early** (a limit, an NCBI outage): open the notebook, choose **Add Input →
Your Work → Notebook Output Files** and pick its last version, then **Save & Run All** again. The
articles already fetched are copied from that output's `retrieval_cache/` and not fetched again.

**Keep the notebook private** (Kaggle's default). The articles are Open Access, but the corpus is
never committed or republished. A CC BY-NC article is used only for this non-commercial coursework,
and the dashboard shows each excerpt's attribution and licence.

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
