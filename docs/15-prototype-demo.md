# 15 · The working prototype: how to start it and how to demo it

**Version:** ~~1.0 · 2026-10-14 (`v0.1-prototype`)~~ 1.1 · 2026-10-14, for `v0.2-prototype`: "what
to ask next" (3d) and evidence with citations (3a) are in (`docs/17` T1). `v0.2-prototype` is tagged
on the laptop, after the checks on the real graph and transformer (`docs/17` §6) · for whoever
presents Nightingale.

> ⚠️ **Decision support, never a diagnosis.** Nightingale is a research prototype built on
> **synthetic** patients (DDXPlus). It knows 14 causes of acute chest pain and nothing else (R-13).
> It must not be used for real patient care, and the presenter should say so first.

---

## 1. What the prototype is

A clinician enters a chest-pain presentation (age, sex, findings present or denied, risk factors)
and gets the 14 possible causes ranked, live, as each finding is ticked:

- **The ranking** fuses a small **transformer** (B1-DL′+aug, the configured ranker since D-13) with
  a **cardiac knowledge graph** (DDXPlus, BODHI-S and a hand-authored aortic dissection, on Neo4j
  AuraDB or the local copy). It uses a logarithmic opinion pool (`docs/02` §6).
- **Red flags** come from independent rules after published patterns (ADD-RS, Wells, Braunwald).
  They always come first.
- **Why each candidate is there:**
  - the graph's supporting (✓), contradicting (✗) and missing (?) findings, with its reasoning
    paths;
  - the model's own strongest reasons, found by removing each finding in turn (2b).
- **What to ask next (3d):**
  - the findings not yet recorded whose answer would most change the knowledge graph's ranking
    (expected information gain), each with the conditions a yes or a no would favour
    (`docs/02` §5.3);
  - the tests to consider for the top three candidates and every red flag: investigations only,
    each from a cited guideline, phrased "consider" (`src/reasoning/investigations.py`).
- **Evidence with citations (3a):** for the first three candidates, excerpts from PubMed Central
  Open Access articles, found by BM25 and a dense search fused by reciprocal rank (EXP-011), each
  with its citation, a link to its PMC page and its licence. Diagnosis only: a sentence with any
  treatment language is never shown (`src/rag/screening.py`). **It stays empty until the corpus is
  built** by the owner's Kaggle run (`docs/11` §4.6); until then each row says so.
- **Robustness:**
  - every component that fails is named on the screen (`degraded_components`);
  - the disclaimer is on every screen;
  - ranks are shown, not raw scores, because the scores are not calibrated probabilities (EXP-007).

**Not in this version yet** (say so if asked):
- ~~"what to ask next" (3d) and evidence with citations (3a), both being built;~~ in since
  `v0.2-prototype` (above); the evidence needs the corpus bundle (§2);
- the LLM explanation (3b), where the template explanation stands in;
- calibrated probabilities (EXP-007 stage 2);
- the test-split results (Phase 4).

---

## 2. Before the presentation (the day before)

1. `git pull` on the presenting laptop.
2. Run the tests, which must be green: `./.venv/Scripts/python.exe -m pytest`.
3. **Resume AuraDB** in the Neo4j console (**Play**): a Free instance pauses after 72 h unused.
   - If you forget, nothing breaks: the dashboard falls back to the local graph and shows
     `graph_backend` in "Running without".
   - To use the local graph on purpose: `NIGHTINGALE_GRAPH=networkx`.
4. Check that `data/` and `models/` are on the laptop. They are not in git.
   - `models/nightingale_b1_dl/b1_dl_asked_aug/seed42/b1_dl.json` must exist.
   - Without it, the dashboard says `ml` is degraded and ranks with the graph alone.
5. **For the evidence:** the corpus bundle in `data/raw/retrieval_corpus/` and
   `retrieval.bundle: data/raw/retrieval_corpus` in `configs/config.yaml` (`docs/11` §4.6).
   - Without it nothing breaks: each candidate's Evidence says no literature is shown, and
     "Running without" lists `rag`.
6. Plug the laptop in, close heavy applications, and start the dashboard once to warm it.

## 3. Starting it

```bash
./.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

Then open <http://127.0.0.1:8501>. It serves on this machine only. Optionally, show the API too:

```bash
./.venv/Scripts/python.exe scripts/serve_api.py
```

It serves <http://127.0.0.1:8000/docs>, where `POST /diagnose` returns the same result as JSON.

**If the browser fails,** the console shows the same result, the evidence, the questions and the
tests included:

```bash
./.venv/Scripts/python.exe scripts/demo.py --case GC-001
```

## 4. A five-minute demo script

Use **Start from an example** in the sidebar: it loads a golden case, one of the hand-written
clinical expectations the tests enforce.

1. **GC-003, aortic dissection: the architecture's whole argument** (1 min).
   - Dissection ranks **first**, with a red flag, although the model has **never seen** a dissection
     patient. DDXPlus has none. The row says "Model: not in the training data: the graph and the
     rules only".
   - Open its row: the graph's reasoning names tearing pain, radiation to the back, sudden onset and
     the inter-arm blood-pressure difference.
   - The point: a model alone cannot rank what it was never trained on; knowledge and rules can.
2. **GC-001, acute coronary syndrome: live re-ranking and the reasons** (~~1½~~ 2 min).
   - The ischaemic red flag fires and MI is first. The header shows the time taken: about 10 ms.
   - Read the explanation: it names the model's strongest reasons for and against the top
     candidate.
   - **Remove "Exertional onset"**, then add it back: the list re-ranks at once (D-14).
   - **Move "Diaphoresis" to Findings denied:** a denial is evidence, while a finding not chosen is
     "not asked". The ranking changes differently for each.
   - **Scroll to "What to ask next".** The graph's first questions are **rest pain** and the
     **crescendo pattern**: the findings that separate unstable angina from stable angina, which
     the graph ties when neither is recorded (EXP-005). The tests to consider are a **12-lead
     ECG** and **serial high-sensitivity troponin** (ESC 2023 ACS; AHA/ACC 2021).
     - Say: questions are asked, never answered for the patient; tests are investigations only,
       each with its guideline; the clinician decides.
   - **Open MI's row: its Evidence.** With the corpus bundle, excerpts from PMC Open Access
     articles, each with its citation and a link to its PMC page. Without it, the row says that no
     literature is shown (§2, step 5).
3. **Building a history from nothing** (1 min). Clear the findings.
   - With none, the model sees age and sex only, and says so.
   - Add chest pain, then pressure-type pain, then radiation to the jaw/arm, one at a time, and
     watch the differential narrow.
   - This is what the system is for, an incomplete history, and why it was trained on masked
     histories (EXP-018/019).
4. **GC-002 (pulmonary embolism) and GC-004 (GERD), quickly** (30 s): a different red flag, and a
   benign answer with no flag.
5. **Resilience, if time allows** (30 s): restart with `NIGHTINGALE_GRAPH=networkx`, or pause Aura.
   The ranking is identical, and "Running without: graph_backend" says what changed.

## 5. Numbers you may quote

All are **validation** figures on synthetic DDXPlus patients. Each comes from `docs/08`, with its
level and model label (`docs/05` §8).

- **At 25% of the history** (EXP-019, EXP-021):

  | System | Must-not-miss recall@3 | Dangerous false-negative rate | Top-1 |
  |---|---|---|---|
  | Plain logistic regression (B1-LR) | 0.925 [0.921, 0.929] | 0.024 [0.021, 0.026] | 0.766 [0.761, 0.770] |
  | Plain XGBoost | 0.877 [0.872, 0.882] | 0.059 [0.056, 0.062] | 0.245 [0.240, 0.250] |
  | The configured transformer, B1-DL′+aug, seed 42 | **0.998 [0.997, 0.999]** | 0.0004 [0.0001, 0.0007] | 0.949 [0.946, 0.951] |

  - The targets are must-not-miss recall@3 ≥ 0.95 and a dangerous false-negative rate ≤ 0.02.
  - Both plain models miss them; the transformer meets them.
  - The plain models were trained on complete histories. The transformer was trained on masked
    copies and sees which questions were asked.
- **The transformer against the logistic regression trained the same way:** non-inferior at every
  seed, with a higher MRR at nine of ten networks (+0.0018 [+0.0009, +0.0026] at seed 42).
  - It is **level with XGBoost trained the same way**: say so if asked whether deep learning won.
  - Its cost: 225,645 parameters and about 1.4 ms per patient on the laptop CPU.
- **Red-flag sensitivity:** 0.823 [0.818, 0.829] on complete histories (EXP-008).
  - It is below its 0.95 target because DDXPlus often leaves out the defining finding.
  - **It falls to 0.135 at 25%**, because most rules need two or three findings together. This is
    why the fused ranking matters as well as the rules.

**Caveats to state with any number:**
- On complete histories every model reaches about 99.8% top-1, because DDXPlus is synthetic and
  each patient's evidence comes only from their condition (R-16). That is why the protocol scores
  50% and 25% of the history.
- Any graph-only figure on DDXPlus is circular (R-12): the graph was built from DDXPlus's own
  definitions.
- The system is closed-world (R-13): a patient with pneumonia is forced into one of the 14
  conditions.

## 6. Questions the audience may ask

- **"Is it accurate?"** On synthetic data, yes, but synthetic data overstates it. The honest test
  is the held-out test split, opened once in Phase 4 (`docs/05` §2), and a clinician's review of the
  golden cases.
- **"Why a transformer?"**
  - The supervisor asked which deep model suits the approach, and at what cost.
  - The owner chose a small FT-Transformer (D-12) and required it to be at least as safe as
    logistic regression before it could replace it (D-13). It passed at all five seeds.
  - It is not better than XGBoost trained the same way, and we report that.
- **"Why does the model count exertional pain *against* a heart attack?"**
  - Because DDXPlus does: in its synthetic patients, exertional pain belongs to the anginas.
  - The model's reasons show what it learned from the data; they are not medical advice.
  - The graph and the red-flag rule still put MI first for GC-001, which is why the system fuses
    several sources rather than trusting one.
  - Showing that disagreement is the point of the reasons.
- **"Can it be used on patients?"** No. It is decision support, built on synthetic data, with no
  clinical validation.
- **"What if a component fails?"** The ranking continues with what is left, and the screen names
  what is missing (`docs/02` §7).
- **"Where do the questions come from?"** From the knowledge graph alone, by expected information
  gain (`docs/02` §5.3). The model is not consulted, every condition counts the same, and one
  question is weighed at a time. The order matters, not the number.
- **"Are the tests advice?"** No. They are diagnostic investigations only, each from a published
  guideline, phrased "consider". There is never a treatment or a drug, and the safety layer
  removes any line that reads as one.
- **"Can the evidence recommend a treatment?"** No. A sentence with any treatment language is
  screened out when the corpus is built, and every excerpt is screened again just before it is
  shown (`src/rag/screening.py`). An article whose title names a treatment is dropped, and the
  safety layer removes any evidence that recommends one.
