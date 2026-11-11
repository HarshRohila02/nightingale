# Prompt: generate the final presentation slides (2026-11-15)

**How to use it:**
1. Fill in the `[TO ADD]` items below that only the team knows: the talk's length and the marking
   scheme (from the announcement), the presenters' names, and the references (copy them from
   `docs/reading-list.pdf`, which checked each against Crossref or arXiv).
2. Paste everything from the line **"PROMPT STARTS HERE"** to the end into a slide generator:
   Claude, Gamma, Microsoft Copilot in PowerPoint, or Google Slides with Gemini.
3. Afterwards, check every number on the slides against `docs/08-experiment-log.md` (EXP-012,
   `[TEST]`) and `docs/15-prototype-demo.md` §5.

The prompt is self-contained: the generator needs no other file. Every figure below is a **test-split
figure** (the split was opened once, on 2026-11-09), copied with its interval from `docs/08`.

---

**PROMPT STARTS HERE**

You are preparing a **16:9 slide deck** for the final presentation of a university project. The project
is **Nightingale — Clinical Intelligence & Reasoning System**: a four-person team's research prototype
that ranks the possible causes of **acute chest pain**, flags the dangerous ones, and shows why. The
audience is faculty examiners in computer science and engineering. The talk is `[TO ADD: length]`
minutes plus a live demo of about 5 minutes and questions, shared by four presenters (P1–P4). The
marking scheme is `[TO ADD: from the announcement]`; until it is known, weigh the deck toward the
results, the honest limits and the demo.

### Presenters by slide
Put the presenter's label in each slide's speaker notes (P1 knowledge graph; P2 data and machine
learning; P3 clinical safety, fusion, evidence and explanation; P4 platform, evaluation and
integration):
- **P4:** slides 1–3, 10–12, 14, 15 (and drives the demo, slide 13);
- **P2:** slides 4, 7;
- **P1:** slide 5, and the "what to ask next" part of slide 9;
- **P3:** slides 6, 8, 9.

### Rules you must follow
1. **Use only the facts, numbers and references given below.** Do not add statistics, citations,
   authors, years or claims of your own. If a slide needs something that is not given, write
   `[TO ADD]` instead.
2. **Copy every number exactly, with its interval** in square brackets and its label (for example
   "test, 25% of the history"). Never round away an interval, and never drop a caveat given with a
   number.
3. **Safety wording:** the system **ranks and flags; it never diagnoses.** Never present any treatment
   or drug as advice. Put this disclaimer on slide 2 and in the footer of every results slide:
   *"Decision support only, not a diagnosis. Synthetic data."*
4. **Be honest, and keep the order the results come in.** The comparative claims mostly failed, and
   the deck says so on its own slide (slide 12). Do not soften a "not supported" into a "partly".
5. **Design:** clean and academic; white or very light background, one dark text colour, one accent
   colour (deep blue or teal); **red only for red flags and safety**. At most 6 bullets per slide and
   about 12 words per bullet. Body text at least 18 pt, tables at least 14 pt. Number every slide.
   Diagrams as simple boxes and arrows; no clip-art and no stock photos of doctors.
6. Add **speaker notes** to every slide from the "Notes" given.
7. References on the slides: short form "(Author Year)". The full list goes on the last slide:
   `[TO ADD: copy from docs/reading-list.pdf]`.

---

### Slide 1 · Title
- Nightingale — Clinical Intelligence & Reasoning System
- Ranking the causes of acute chest pain, with the dangerous ones flagged first
- `[TO ADD: team, supervisor, course, date 2026-11-15]`

**Notes (P4):** One sentence: a decision-support prototype that ranks 14 causes of acute chest pain,
built and tested on synthetic patients. Say at once that it is not for patient care.

### Slide 2 · The problem and what the system is not
- Chest pain: many causes; a few cannot be missed (heart attack, embolism, dissection)
- The system **ranks and flags**; the clinician decides
- It knows **14 conditions only**: anything else is forced into one (closed world)
- Built and tested on **synthetic** patients: no real patient was seen
- Non-commercial academic use only

**Notes (P4):** Read the disclaimer: "Decision support only, not a diagnosis. Synthetic data." The
headline metric is **must-not-miss recall@3**: of the patients whose true cause is a dangerous one,
how many have it in the top three.

### Slide 3 · What we built
Diagram, left to right: **findings entered** → **small transformer** + **cardiac knowledge graph** →
**fusion** (logarithmic opinion pool) → **red-flag rules first** → **safety layer** → **output**:
ranked causes, why (✓ supports, ✗ against, ? not asked), what to ask next, tests to consider,
cited evidence.
- Live re-ranking: about 14 to 17 ms for the whole pipeline (p95)
- Every failed component is named on screen; the disclaimer is on every screen

**Notes (P4):** The red-flag rules are independent of the model on purpose: they still fire if the
model fails. Evidence and questions never change the ranking.

### Slide 4 · The data
- **DDXPlus** (CC-BY-4.0): synthetic patients; we keep 13 chest-pain conditions
- Our splits: train 255,900 · validation 33,963 · **test 34,657** (opened once)
- **BODHI-S** (CC-BY-NC-4.0, Eka Care): enriches 4 conditions of the graph
- **Synthea**: 1,013 synthetic patient histories, shown beside a case (never ranked on)
- **PubMed Central Open Access**: 2,067 articles, 26,190 passages, each with its own licence
- Aortic dissection has **no training patient**: only the graph and the rules can rank it

**Notes (P2):** Why synthetic: no real patient data may enter the project. The cost: DDXPlus draws
each patient's evidence only from their condition's list, so models look nearly perfect on a full
history (slide 12). That is the data, not skill.

### Slide 5 · The knowledge graph
- Built from DDXPlus's condition knowledge base, enriched from BODHI-S, plus hand-authored aortic dissection
- Every edge records its source; 129 nodes, 321 edges
- Score: a naive-Bayes log-likelihood over the graph; shows ✓ ✗ ? for each candidate
- **On DDXPlus the graph's own accuracy is circular**: DDXPlus was generated from the same definitions
- Case GC-003: dissection ranks **first** with a red flag, though the model never saw one

**Notes (P1):** Say "circular" aloud: any graph-only figure on DDXPlus flatters the graph. The golden
cases are the evidence that counts for the graph.

### Slide 6 · Red flags and the safety layer
- Rules after published patterns (ADD-RS, Wells, Braunwald); every must-not-miss condition has one
- Red-flag sensitivity, test: **0.8285 [0.8229, 0.8344]** at 100% of the history (target 0.95)
- At 50%: **0.3318 [0.3245, 0.3386]**; at 25%: **0.1369 [0.1317, 0.1420]**
- Strict flag precision at 100%: **0.5089 [0.5047, 0.5128]**: about half the flags name another condition
- Flags rank first, which **lowers top-1**: 0.8346 [0.8303, 0.8382] against 0.9987 [0.9983, 0.9991] for the model alone
- No treatment or drug is ever shown; the safety layer removes any such line

**Notes (P3):** The rules need two or three findings together, so a short history rarely fires them.
That is why the ranking must hold up on its own. The flags are a safety net, deliberately noisy.

### Slide 7 · The ranker: a small transformer, trained on partial histories
- FT-Transformer style: 225,645 parameters; about 1.4 ms a patient on the laptop CPU
- Trained on **masked copies** of histories, and told which questions were asked
- Safety first (D-13): it replaced logistic regression only after passing a rule fixed before training
- At 25% of the history the **plain** models fail the safety targets; the masked-trained ones meet them
- Against XGBoost trained the same way it is **level**, and we say so

**Notes (P2):** A real chart is the plain-versus-masked table on slide 11. The supervisor asked which
deep model suits the approach and at what cost: this is the answer, with its price.

### Slide 8 · Fusion: what we found
- A logarithmic opinion pool of the model and the graph; weight α = 0.05, chosen by a pre-registered sweep
- **The fusion adds nothing measurable** over the transformer alone (test, 25%): MRR 0.9639 [0.9623, 0.9652] both ways
- Top-3: 0.9979 [0.9974, 0.9983] fused, 0.9980 [0.9975, 0.9984] model alone
- Its value is for what the model cannot score: aortic dissection
- A negative result, reported as one

**Notes (P3):** The pool gives a condition outside the model the graph's own posterior, so dissection
can rank first. On DDXPlus nothing else is gained, and any gain would be circular anyway.

### Slide 9 · Explanation, evidence and what to ask next
- **What to ask next** (graph only): rest pain and the crescendo pattern, for unstable angina against MI
- **Tests to consider**: investigations only, each from a cited guideline (for example ECG, serial troponin)
- **Evidence**: PMC excerpts with citation, link and licence; diagnosis only, treatment screened out
- **The relevance of the passages was never judged**: the judges had no time, so the page says so
- The laptop's query vector overlaps the full-sentence one by only **0.40** at the median (top 10)
- A language-model explainer was built and tried (validation, 50 patients): **0.741 [0.663, 0.803]** of its claims were unsupported (the bar was under 0.20), so the template stays

**Notes (P1 for the questions, P3 for the rest):** Say what we did not do: no relevance judging, and the
language model failed its own checkpoint. Those are findings, not omissions.

### Slide 10 · How we evaluated
- Protocol **written and frozen before any model was trained** (`docs/05`), amended only through a log
- Every system scored at **100%, 50% and 25%** of each patient's history, by fixed random masks (digests recorded first)
- Code and configuration **frozen** (tag `v0.9-freeze`) before the test split was touched
- The test split (34,657 patients) was **opened once**, on 2026-11-09; the run refused to start unless two recorded digests reproduced
- Every figure carries its 95% interval (1,000 bootstrap resamples) and its level

**Notes (P4):** The 100% level cannot separate systems (every comparator is perfect), so 50% and 25% were added
before any result. Nothing was tuned on the test split.

### Slide 11 · Results (1): safety at 25% of the history (test)
Table, 34,657 patients, 95% intervals:

| System | Must-not-miss recall@3 | Dangerous false-negative rate | Top-3 |
|---|---|---|---|
| Plain logistic regression | 0.9198 [0.9156, 0.9238] | 0.0272 [0.0249, 0.0297] | 0.9436 [0.9412, 0.9461] |
| Plain XGBoost | 0.8710 [0.8659, 0.8764] | 0.0629 [0.0592, 0.0665] | 0.9035 [0.9003, 0.9068] |
| Transformer alone | 0.9986 [0.9980, 0.9991] | 0.0001 [0.0000, 0.0002] | 0.9988 [0.9985, 0.9991] |
| **Full system** | **0.9986 [0.9980, 0.9991]** | 0.0002 [0.0000, 0.0004] | 0.9988 [0.9985, 0.9992] |

- Targets: recall@3 ≥ 0.95 and dangerous false-negative rate ≤ 0.02
- Both plain models miss them; the systems trained on masked histories meet them
- **The finding that carries the project:** train on partial histories

**Notes (P4):** The plain models were trained on complete histories only. Footer: "Decision support only, not a diagnosis. Synthetic data, test split."

### Slide 12 · Results (2): the pre-registered claims, as they stand
| Claim | Verdict (test) |
|---|---|
| H1, H2: beats B1 at full history | **not supported** (every comparator misses 0 of 34,657) |
| H1-R, H2-R: beats B1 at 25% | **not supported** (level with the masked-trained models, behind them on MRR) |
| H6: deep model beats the same-trained models | **not supported** (beats logistic regression, level with XGBoost) |
| H4: removing the graph hurts must-not-miss recall | supported by the rule, **but hollow**: recall does not change (0.0000); the point estimate passes because flags lower top-1 |
| H5: ECE below 0.10 at full history | supported: 0.0099 [0.0094, 0.0103] (unprimed arm), 0.0398 [0.0394, 0.0402] (primed arm) |
| Minimum (beats the prevalence baseline) | met |
| Reduced-evidence Target | met by one arm only, by a handful of patients (40 against 56 misses of 34,657) |

- The same verdicts as on validation: nothing changed between the two splits

**Notes (P4):** Read the table as it is. R-16: on DDXPlus every model reaches about 99.9% top-1 at full
history, so a comparison there tests nothing. Do not quote H4 as evidence that the graph helps.

### Slide 13 · Live demo (about 5 minutes)
Follow `docs/15-prototype-demo.md` §4: GC-003 (dissection first, though the model never saw one);
GC-001 (live re-ranking, the model's reasons, what to ask next, evidence); build a history from nothing;
GC-002 and GC-004; the patient-history panel (display only); the credits. Say aloud: the evidence's
relevance is unjudged; the form and ranking do not change when a patient history is shown.

**Notes (P4 drives):** Resume Aura the day before; if it is paused, the screen says "Running without:
graph_backend" and the ranking is identical.

### Slide 14 · Limits, ethics and licences
- Synthetic data only; **no clinical validation**, no user study
- Closed world: 14 conditions; the graph's accuracy on DDXPlus is circular; α was chosen on validation
- By-sex results are near-equal because DDXPlus's demographics are synthetic: **not evidence of fairness**
- Free text is ignored by design (tested); the disclaimer is on every screen, API result and figure
- **Non-commercial**: BODHI-S and the embedding model are CC-BY-NC, and 844 of 2,067 articles are
- Credits: DDXPlus, BODHI-S (Eka Care), Synthea, PubMed Central, S-PubMedBert

**Notes (P3):** The compliance checklists are ticked with evidence in `docs/03` §7 and `docs/04` §8.

### Slide 15 · What we learned
- The protocol, written first, made honest negatives possible
- Training on partial histories mattered more than the model family or the fusion
- A fusion of a model and a knowledge graph needs data that is not circular to show a gain
- What we did not do: judged the evidence, a clinician's review, real patients
- Thank you; questions

**Notes (P4):** Close on what the evidence supports and no more. Offer the backup slides.

---

### Backup slides (for questions)
- **B1 · Where the model errs:** at 25%, unstable angina top-1 **0.6896 [0.6726, 0.7059]** (confused with stable angina; its top-3 is 0.9997); every condition's top-3 is at least 0.993.
- **B2 · Calibration:** raw top-label ECE 0.0398 [0.0394, 0.0402] at full history and 0.0785 [0.0766, 0.0806] at 25%; no probability is shown to the user, because a short hand-written case is near-certain under any calibration.
- **B3 · By sex:** A0′ top-1 at 25% is 0.9375 [0.9338, 0.9410] for women and 0.9368 [0.9330, 0.9408] for men; the near-equality is an artifact of the data.
- **B4 · Cost:** 225,645 parameters; about 1.4 ms a patient on the laptop CPU; the whole pipeline about 14 to 17 ms (p95).

### Reference slide
`[TO ADD: the numbered list from docs/reading-list.pdf]`
