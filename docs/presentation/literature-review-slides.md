# Literature review: instructions and content for the slides

**For:** Claude in PowerPoint (the add-in) · **Project:** Nightingale, a diagnosis helper for acute
chest pain · **Version:** 2026-10-15

---

## Instructions for Claude in PowerPoint

1. **Make at most 3 slides** for this section:
   - **Slide A:** literature review, part 1 (papers 1–2: decision support, and why chest pain);
   - **Slide B:** literature review, part 2 (papers 3–5: safety, rules and our dataset) and the
     summary of what each paper gave us;
   - **Slide C:** the research gap.

   If the deck already has slides with these titles, edit them instead of adding new ones.
2. **Keep it simple.** Short bullets of about 12 words or fewer, plain words, no jargon. A student
   must be able to explain each bullet aloud in one or two sentences.
3. **At most 5 bullets per slide.** Use a small table where one is given below.
4. **Cite briefly on the slide**, e.g. "(Sutton 2020)". Put the full references on the deck's
   existing references slide; if it has none, put them in Slide C's speaker notes.
5. **Copy names, years and facts exactly as written here.**
   - Do not add papers, statistics or claims that are not in this file.
   - If space is short, cut words, not facts.
6. **Add the speaker notes** given for each slide.
7. **Keep the deck's existing theme, fonts and colours.**
8. **Somewhere in this section, include the line:** "Nightingale is decision support, not a
   diagnosis."

---

## Slide A · Literature Review (1/2): Decision support and why chest pain

**Title:** Literature Review: What We Learned (1/2)

**Bullets:**
- **Sutton et al. 2020 (our first paper):** computer tools that help doctors decide (CDSS).
- They reduce errors, but black-box AI and too many alerts reduce trust.
- **Lesson:** build a helper that explains itself and leaves the decision to the doctor.
- **Gulati et al. 2021 (chest-pain guideline):** chest pain is common in emergency care and
  can be deadly.
- **Lesson:** chest pain is high-stakes and has clear rules, so it is a good fit for a helper.

**Speaker notes:**
- Our first paper was Sutton et al. 2020, an overview of clinical decision support systems. It
  shows these tools help doctors, but doctors distrust black-box AI and ignore tools that raise
  too many alerts.
- So we decided to build an explainable diagnosis helper.
- We then chose acute chest pain using the 2021 chest-pain guideline: doctors must quickly separate
  deadly causes, such as a heart attack or a blood clot in the lung, from harmless ones like
  reflux.

---

## Slide B · Literature Review (2/2): Safety, rules and our dataset

**Title:** Literature Review: What We Learned (2/2)

**Bullets:**
- **Pope et al. 2000:** some heart-attack patients were wrongly sent home, so **safety first**.
- **Six et al. 2008, the HEART score:** a simple 5-point chest-pain risk score, the basis for our
  red-flag rules.
- **Fansi Tchango et al. 2022, DDXPlus:** 1.3 million synthetic patients, 49 diseases, a ranked
  list of diagnoses.
- **Our design follows DDXPlus:** a ranked list of causes, its data, and its disease–symptom
  knowledge.
- **We added aortic dissection,** which DDXPlus lacks, from medical sources.

**Small table (if space allows):**

| Paper | What it gave us |
|---|---|
| Sutton 2020 | An explainable helper; the doctor decides |
| Gulati 2021 | Why chest pain |
| Pope 2000 | Safety first |
| Six 2008 | Rule-based red flags |
| DDXPlus 2022 | The dataset and the design |

**Speaker notes:**
- Pope et al. showed that real heart-attack patients were sometimes missed, so our system ranks
  dangerous conditions first.
- The HEART score showed that doctors already use simple rules for chest pain, so our red-flag
  alarms follow published rules.
- DDXPlus is our dataset and our design reference. It is a large synthetic dataset, so there are
  no privacy issues. Like it, our system outputs a ranked list of possible causes, not one answer.
- We kept its 13 chest-pain conditions, about 256,000 training and 34,000 validation patients, and
  added aortic dissection ourselves.

---

## Slide C · Research Gap

**Title:** Research Gap

**Statement (large text):** Current tools help doctors but are often unexplained, raise too many
alerts, or are tested only on complete patient data. No reviewed system ranks the causes of chest
pain safely, explainably, with incomplete histories, and including a condition missing from its
training data.

**Table:**

| Gap | How Nightingale fills it |
|---|---|
| Black-box AI: no reasons shown (Sutton 2020) | A medical knowledge graph and rules show the reasons |
| Accuracy over safety (Pope 2000) | Dangerous conditions are kept at the top; red flags first |
| Too many alerts (Sutton 2020) | Red flags follow published rules; false alarms measured |
| Tested only on complete data (DDXPlus 2022) | Tested with 100%, 50% and 25% of the history |
| AI can't suggest an unseen disease (DDXPlus has no aortic dissection) | The knowledge graph adds it |

**Closing line:** Nightingale is decision support, not a diagnosis: a safe, explainable
chest-pain helper.

**Speaker notes:**
- Putting it together: decision-support tools are useful but have known problems.
- Our project addresses each one:
  - it explains itself;
  - it puts dangerous conditions first;
  - it keeps alerts meaningful;
  - it works even when the doctor has asked only some of the questions;
  - it can still suggest aortic dissection, which the training data does not contain.

---

## References (for the references slide)

1. Sutton RT, Pincock D, Baumgart DC, Sadowski DC, Fedorak RN, Kroeker KI. An overview of clinical decision support systems: benefits, risks, and strategies for success. *npj Digit Med.* 2020;3:17. doi:10.1038/s41746-020-0221-y
2. Gulati M, Levy PD, Mukherjee D, et al. 2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR Guideline for the Evaluation and Diagnosis of Chest Pain. *Circulation.* 2021;144(22):e368–e454. doi:10.1161/CIR.0000000000001029
3. Pope JH, Aufderheide TP, Ruthazer R, et al. Missed diagnoses of acute cardiac ischemia in the emergency department. *N Engl J Med.* 2000;342:1163–1170. doi:10.1056/NEJM200004203421603
4. Six AJ, Backus BE, Kelder JC. Chest pain in the emergency room: value of the HEART score. *Neth Heart J.* 2008;16:191–196. doi:10.1007/BF03086144
5. Fansi Tchango A, Goel R, Wen Z, Martel J, Ghosn J. DDXPlus: A New Dataset for Automatic Medical Diagnosis. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks.* doi:10.52202/068431-2270
