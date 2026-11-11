# 15 · The working prototype: how to start it and how to demo it

**Version:** ~~1.0 · 2026-10-14 (`v0.1-prototype`)~~ ~~1.1 · 2026-10-14, for `v0.2-prototype`~~ 1.2 · 2026-11-10,
for `v1.0`: the final prototype, with the test-split results, the corpus, the credits and what was and
was not done. Every claim below was checked on 2026-11-10 against the running system (the console demo
on the golden cases and the dashboard) or against `docs/08` · for whoever presents Nightingale.

> ⚠️ **Decision support, never a diagnosis.** Nightingale is a research prototype built on
> **synthetic** patients (DDXPlus). It knows 14 causes of acute chest pain and nothing else (R-13).
> It must not be used for real patient care, and the presenter should say so first. It is for
> **non-commercial academic use only** (BODHI-S and the embedding model are non-commercial licences).

---

## 1. What the prototype is

A clinician enters a chest-pain presentation (age, sex, findings present or denied, risk factors)
and gets the 14 possible causes ranked, live, as each finding is ticked:

- **The ranking** fuses a small **transformer** (B1-DL′+aug, the configured ranker since D-13) with
  a **cardiac knowledge graph** (DDXPlus, BODHI-S and a hand-authored aortic dissection, on Neo4j
  AuraDB or the local copy). It uses a logarithmic opinion pool (`docs/02` §6) with the graph's
  weight α = 0.05, chosen by EXP-006's pre-registered sweep. **The fusion adds nothing measurable
  over the transformer alone** (EXP-006, and again on the test split, EXP-012): say so if asked.
- **Red flags** come from independent rules after published patterns (ADD-RS, Wells, Braunwald).
  They always come first. They are why the safety figures hold at low evidence, and also why the
  top-1 is lower than the transformer's alone (§5).
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
  Open Access articles (2,067 articles, 26,190 passages, built on Kaggle on 2026-11-07), found by
  BM25 and a dense search fused by reciprocal rank, each with its citation, a link to its PMC page,
  its attribution and **its own Creative Commons licence** (844 of the 2,067 articles are
  non-commercial). Diagnosis only: a sentence with any treatment language is never shown
  (`src/rag/screening.py`). **Two honest limits, stated on the screen and to be stated aloud:**
  the passages' relevance was **never judged** (EXP-011: the team had no time), and the laptop's
  query vector is a poor stand-in for embedding the whole query (median top-10 overlap 0.40).
  Read the passages as leads, not proof.
- **Patient history (1d), display only:** a synthetic Synthea patient's history, chosen from a list of 1,013 patients
  (alive on 2026-01-01) and drawn as a card beside the case: the active conditions (disorders first, then the
  social history and findings Synthea lists as active), the risk factors the history supports, a
  "Medication history (not advice)" and counts of procedures and observations. **It never feeds the
  ranking, the case's findings or its risk factors** (the owner's decision of 2026-10-20): the ranking is
  the same whether a card is chosen or not (tested on the page, and checked in the browser). A Synthea
  patient is not a DDXPlus patient, so the card's age and sex are not tied to the form (a 59-year-old
  woman's card can sit beside a 55-year-old man's case), and the page says it is context only.
- **Credits and licences:** a "Data sources, credits and licences" section at the bottom of the
  page names every source, its licence (including Eka Care for BODHI-S) and that the project is
  non-commercial.
- **Robustness:**
  - every component that fails is named on the screen (`degraded_components`);
  - the disclaimer is on every screen, every API result and every figure;
  - ranks are shown, not raw scores, because the scores are not calibrated probabilities and a
    short hand-written case is near-certain under any calibration (EXP-007, R-17).

**Not in this version** (say so if asked):
- **The LLM explanation (3b)** was built and tried on 50 patients, and it failed its own
  checkpoint: 74% of its claims were unsupported (the bar was under 20%), so the plain template
  explanation stands and `llm.enabled` is false (D-5, EXP-009/010).
- **Calibrated probabilities:** decided against; ranks only (above).
- **Judged evidence:** see above (EXP-011).
- **Any clinical validation,** a user study, or real patient data.
- **An age breakdown** of the results (there is a by-sex one, and by-sex parity is an artifact of
  synthetic demographics, not evidence of fairness).

---

## 2. Before the presentation (the day before)

1. `git pull` on the presenting laptop.
2. Run the tests, which must be green: `./.venv/Scripts/python.exe -m pytest`.
3. **Resume AuraDB** in the Neo4j console (**Play**): a Free instance pauses after 72 h unused.
   - If you forget, nothing breaks: the dashboard falls back to the local graph and shows
     `graph_backend` in "Running without" (it will, in the console demo too).
   - To use the local graph on purpose: `NIGHTINGALE_GRAPH=networkx`.
4. Check that `data/` and `models/` are on the laptop. They are not in git.
   - `models/nightingale_b1_dl/b1_dl_asked_aug/seed42/b1_dl.json` must exist.
   - Without it, the dashboard says `ml` is degraded and ranks with the graph alone.
   - **Presenting from another laptop?** The safest course is to present from the laptop where it
     already runs. Otherwise the demo needs a clone, the `.venv` (Python 3.11, `requirements-dev.txt`)
     and about 75 MB of files that are not in git, read off this laptop on 2026-11-10: the model
     `models/nightingale_b1_dl/b1_dl_asked_aug/seed42/` (1.3 MB); `data/raw/ddxplus/release_conditions.json`
     and `release_evidences.json`; `data/raw/bodhi_s/` (4.4 MB); `data/interim/ddxplus_chestpain_conditions.json`
     and `ddxplus_evidences.json`; `data/raw/retrieval_corpus/` (63 MB); and
     `data/interim/patient_display/patients.json` (1.7 MB). **Checked on 2026-11-11** by copying exactly
     this list into a fresh clone: `scripts/demo.py --case GC-003` gave the same ranking (dissection
     first, identical scores) with only `graph_backend` degraded. Still run it on the presenting
     laptop, and check that "Running without" names nothing else, before relying on it.
5. **For the evidence:** the corpus bundle in `data/raw/retrieval_corpus/` (with its
   `manifest.json`); `retrieval.bundle` is already set in `configs/config.yaml`.
   - Without it nothing breaks: each candidate's Evidence says no literature is shown, and
     "Running without" lists `rag`.
6. **For the patient-history panel:** `./.venv/Scripts/python.exe scripts/build_patient_display.py` writes
   `data/interim/patient_display/patients.json` from the Synthea output in `data/raw/synthea_patient_kg/`
   (about 12 s, 1.7 MB).
   - Without it nothing breaks: the page shows one line saying how to build it.
7. Plug the laptop in, close heavy applications, and start the dashboard once to warm it.
8. **Do not touch the test split.** It was opened once, on 2026-11-09 (`docs/08` EXP-012). Nothing
   in the demo reads it, and nothing should score it again.

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
2. **GC-001, acute coronary syndrome: live re-ranking and the reasons** (2 min).
   - The ischaemic red flag fires and NSTEMI / STEMI is listed first. The header shows the time
     taken: about 10 to 25 ms.
   - Read the explanation: it names the model's strongest reasons for and against the top
     candidate. Here the model's reasons *against* include exertional onset and pressure-type pain,
     because in DDXPlus those belong to the anginas; unstable angina is the model's own first
     answer, listed second only because red flags come first.
   - **Remove "Exertional onset"**, then add it back: the list re-ranks at once (D-14).
   - **Move "Diaphoresis" to Findings denied:** a denial is evidence, while a finding not chosen is
     "not asked". The ranking changes differently for each.
   - **Scroll to "What to ask next".** The graph's first questions are **rest pain** and the
     **crescendo pattern**, which bear on unstable angina against MI here. The tests to consider
     are a **12-lead ECG** and **serial high-sensitivity troponin** (ESC 2023 ACS; AHA/ACC 2021).
     The anginas are also where every model errs most (EXP-023).
     - Say: questions are asked, never answered for the patient; tests are investigations only,
       each with its guideline; the clinician decides.
   - **Open a candidate's Evidence.** Excerpts from PMC Open Access articles, each with its
     citation, a link to its PMC page and its licence. **Say it before anyone asks: the relevance of
     these passages was never judged**, and the page says so under them.
3. **Building a history from nothing** (1 min). Clear the findings.
   - With none, the model sees age and sex only, and says so.
   - Add chest pain, then pressure-type pain, then radiation to the jaw/arm, one at a time, and
     watch the differential narrow.
   - This is what the system is for, an incomplete history, and why it was trained on masked
     histories (EXP-018/019).
4. **GC-002 (pulmonary embolism) and GC-004 (GERD), quickly** (30 s): a different red flag
   (embolism, and pneumothorax beside it), and a benign answer, GERD first, with no flag.
5. **The patient-history panel** (30 s): open "Patient history (Synthea, display only)", choose a patient and
   show the card. Then show that **the form and the ranking do not change**: the history is context, never
   an input. Say that Synthea lists social history (employment, education) among its "active
   conditions", so the card keeps those apart as "findings on record", and that medications are
   history, never advice.
6. **Resilience, if time allows** (30 s): restart with `NIGHTINGALE_GRAPH=networkx`, or pause Aura.
   The ranking is identical, and "Running without: graph_backend" says what changed.
7. **The credits** (10 s): open "Data sources, credits and licences" at the bottom and say the
   project is for non-commercial academic use.

## 5. Numbers you may quote

The final results are on the **test split**, opened **once** (2026-11-09), after the code was frozen
(`v0.9-freeze`) and the two mask digests were written down. They are `[TEST]` figures on **synthetic**
DDXPlus patients, 34,657 of them, each with its 95% interval and level (`docs/08` EXP-012; `docs/05`
§8). The validation figures reproduce them closely, and no verdict changed between the two.

- **At 25% of the history** (the level where the comparison is still open):

  | System | Must-not-miss recall@3 | Dangerous false-negative rate | Top-1 | Top-3 |
  |---|---|---|---|---|
  | Plain logistic regression (B1-LR) | 0.9198 [0.9156, 0.9238] | 0.0272 [0.0249, 0.0297] | 0.7652 [0.7604, 0.7697] | 0.9436 [0.9412, 0.9461] |
  | Plain XGBoost (B1-XGB) | 0.8710 [0.8659, 0.8764] | 0.0629 [0.0592, 0.0665] | 0.2482 [0.2439, 0.2524] | 0.9035 [0.9003, 0.9068] |
  | The transformer alone (B1-DL′+aug) | 0.9986 [0.9980, 0.9991] | 0.0001 [0.0000, 0.0002] | 0.9481 [0.9456, 0.9503] | 0.9988 [0.9985, 0.9991] |
  | **The full system** (A0′: transformer + graph + red flags) | **0.9986 [0.9980, 0.9991]** | 0.0002 [0.0000, 0.0004] | 0.9371 [0.9343, 0.9396] | 0.9988 [0.9985, 0.9992] |

  - The targets are must-not-miss recall@3 ≥ 0.95 and a dangerous false-negative rate ≤ 0.02.
  - **Both plain models miss them; the transformer-based systems meet them.** The plain models were
    trained on complete histories. The transformer was trained on masked copies and sees which
    questions were asked. That is the finding that carries.
  - **The full system is not better than the transformer alone.** Top-3 is level; top-1 is lower
    (0.9371 against 0.9481) because the red flags rank their conditions first even when the flag
    names the wrong one (strict flag precision 0.509 [0.505, 0.513] at full evidence). The graph
    adds nothing measurable on DDXPlus, and any graph-only figure there is circular (R-12).
- **The pre-registered claims, on the test split:** H1, H2 (full evidence), H1-R, H2-R (25%) and H6
  are **not supported**; H4, H5 and the Minimum hold; the reduced-evidence Target is met by the
  primed system only, by a handful of patients (it misses 40 of 34,657 in the top 3 at 25%, against
  56 for XGBoost trained the same way and 41 for the transformer alone). Say them as they stand.
  - **At full evidence every comparator misses 0 of 34,657 patients** (R-16: DDXPlus draws each
    patient's evidence only from their condition's list), so nothing can beat them there.
  - **H4 is supported, but hollow:** removing the graph changes must-not-miss recall by exactly
    zero; the point estimate passes because the red-flag layer lowers top-1. Do not quote H4 as
    evidence that the graph helps.
- **The transformer against the logistic regression trained the same way:** it wins; against
  XGBoost trained the same way it is **level** (H6). Say so if asked whether deep learning won.
  Its cost: 225,645 parameters and about 1.4 ms per patient on the laptop CPU; the whole pipeline
  takes about 14 to 17 ms at p95 (EXP-022).
- **Red-flag sensitivity** (test): 0.8285 [0.8229, 0.8344] at full evidence, against a 0.95
  target, because DDXPlus often leaves out the defining finding; **0.3318 [0.3245, 0.3386] at 50%
  and 0.1369 [0.1317, 0.1420] at 25%**, because most rules need two or three findings together.
  This is why the fused ranking matters as well as the rules.
- **Weakest spot:** unstable angina at 25% evidence, top-1 0.6896 [0.6726, 0.7059] (it is confused
  with stable angina); its top-3 is 0.9997. Every condition's top-3 is at least 0.993.
- **Calibration:** the raw top-label ECE is 0.0398 [0.0394, 0.0402] at full evidence (the bar was
  0.10), and 0.0785 [0.0766, 0.0806] at 25%. No probability is shown to the user.

**Caveats to state with any number:**
- On complete histories every model reaches about 99.9% top-1, because DDXPlus is synthetic and
  each patient's evidence comes only from their condition (R-16). That is why the protocol scores
  50% and 25% of the history, with random masks: they measure robustness to missing answers, not
  history-taking.
- The test split is a second sample from the same generator as validation: it checks that nothing
  was fitted to validation, not that the system works on real patients.
- The system is closed-world (R-13): a patient with pneumonia is forced into one of the 14
  conditions.
- The graph's weight α was chosen on validation.

## 6. Questions the audience may ask

- **"Is it accurate?"** On synthetic data it is, at the level of the figures in §5, and synthetic
  data overstates it. The test split was opened once and nothing was tuned on it. There is no
  clinical validation and no real patient has been seen.
- **"Does the knowledge graph help?"** Not measurably on DDXPlus: the fused system equals the
  transformer alone, and the graph's own score is circular because the graph was built from
  DDXPlus's definitions (R-12). Its value is for what the model has never seen: aortic
  dissection has no training patients, and only the graph and the rules can rank it (GC-003).
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
- **"Are the evidence passages relevant?"** We do not know. Two team members were to judge them
  blind and had no time, so EXP-011 stops at "built and measured for cost", the hybrid search stays
  by default, and the page says the relevance is unjudged. The laptop's query vector also overlaps
  the full-sentence one by only 0.40 at the median (top 10).
- **"Why no language-model explanation?"** We built it and measured it first: on 50 patients, 74%
  of its claims were not supported by the case (the bar was under 20%), so the template stays
  (EXP-009/010). The check that stopped it is lexical, which we disclose.
- **"Did you look at the test set more than once?"** No. It was opened once, on 2026-11-09, by a
  run that refused to start unless the code was frozen and two recorded digests reproduced, and
  the opening is recorded (`docs/08` EXP-012).
- **"Can I use it commercially?"** No. BODHI-S and the embedding model are CC-BY-NC, and 844 of the
  2,067 corpus articles are non-commercial; every source is credited on the page.
- **"What about the patient history panel?"** It is display only: choosing a patient changes nothing in the
  ranking, the findings or the explanation (tested, and checked on the page). A Synthea patient is not a
  DDXPlus patient, so the card's age and sex are not tied to the case beside it, and the page says it is
  context. Synthea's own labels are shown as they come, social history included.
