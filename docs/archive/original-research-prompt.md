# Deep Research Prompt — AI-Assisted Differential Diagnosis Using Patient Knowledge Graphs + Medical RAG

## Research Goal

Conduct a **deep, comprehensive, current research study** on the following proposed academic project:

> **AI-Assisted Differential Diagnosis and Clinical Decision Support System using Patient-Centric Knowledge Graphs, Medical Knowledge Graphs, Machine Learning, and Evidence-Grounded RAG**

The intended project is a **research/academic prototype for assisting qualified doctors**, not an autonomous diagnostic system and not a replacement for clinical judgment.

The system should allow a doctor to upload or enter a patient's relevant medical information—such as demographics, symptoms, medical history, medications, allergies, vital signs, laboratory results, examination findings, imaging/report findings, and clinical notes—and then transform that heterogeneous information into a structured patient representation.

The system should combine that patient representation with a curated medical knowledge graph and retrieval-augmented medical literature/guideline evidence to:

1. identify relevant clinical concepts,
2. construct a patient-specific knowledge graph,
3. retrieve medically relevant diseases/conditions,
4. rank a **differential diagnosis** rather than assert a definitive diagnosis,
5. show supporting and contradictory evidence for each candidate,
6. identify missing information that could help distinguish between candidates,
7. retrieve relevant medical guidelines/literature,
8. provide traceable reasoning paths and source citations,
9. flag safety-critical or high-risk findings,
10. present the result to the doctor for final clinical judgment.

The core research question should be:

> **Can combining a patient-specific knowledge graph with a medical knowledge graph, machine-learning-based diagnostic ranking, and evidence-grounded RAG improve the accuracy, factuality, explainability, evidence traceability, and safety of differential-diagnosis assistance compared with conventional ML, LLM-only, or text-RAG approaches?**

---

# 1. Research the Problem and Clinical Use Case

Investigate:

- What is differential diagnosis and how clinicians construct it.
- How doctors integrate:
  - symptoms,
  - past medical history,
  - medications,
  - allergies,
  - family history,
  - social history,
  - vital signs,
  - laboratory results,
  - imaging findings,
  - physical examination,
  - clinical notes.
- Where clinical decision-support systems currently help.
- Existing limitations of conventional clinical decision-support systems.
- Limitations of LLMs in clinical reasoning.
- Hallucination, unsupported claims, stale knowledge, poor calibration, and reasoning errors in medical AI.
- Why structured medical knowledge graphs may complement LLMs.
- Why RAG may complement knowledge graphs.
- Whether a hybrid KG + RAG + ML architecture is supported by recent research.

Clearly distinguish:

- diagnosis,
- differential diagnosis,
- clinical decision support,
- risk prediction,
- triage,
- patient education.

The project should be positioned as **decision support for physicians**, not autonomous diagnosis.

---

# 2. Investigate the Best System Architecture

Research and recommend the most practical architecture for a 4-person student team with approximately 3–6 months of development time.

Compare at least these architectures:

### Architecture A — Traditional ML

```text
Patient data
→ feature extraction
→ ML classifier
→ disease ranking
```

### Architecture B — LLM only

```text
Patient information
→ LLM
→ differential diagnosis
```

### Architecture C — RAG + LLM

```text
Patient information
→ document retrieval
→ LLM
→ differential diagnosis
```

### Architecture D — Knowledge Graph + ML

```text
Patient information
→ patient KG
→ medical KG
→ graph reasoning / ML
→ disease ranking
```

### Architecture E — KG + RAG + ML + LLM

```text
Patient information
→ Clinical NLP
→ Patient KG
→ Medical KG
→ Candidate generation
→ ML ranking
→ Evidence retrieval/RAG
→ Reasoning/validation
→ LLM explanation
→ Doctor dashboard
```

Determine which architecture should be the **recommended final architecture** and explain why.

---

# 3. Design the Patient Knowledge Graph

Research how to represent a patient as a temporary or access-controlled graph.

Consider nodes such as:

- Patient
- Symptom
- Disease/condition
- Medication
- Allergy
- Laboratory test
- Laboratory result
- Vital sign
- Imaging finding
- Procedure
- Diagnosis
- Risk factor
- Family history
- Social history
- Clinical observation
- Clinical note
- Temporal event

Consider relationships such as:

```text
Patient
  ├── HAS_SYMPTOM → Symptom
  ├── HAS_HISTORY → Disease
  ├── TAKES → Medication
  ├── ALLERGIC_TO → Drug/Substance
  ├── HAS_LAB_RESULT → LabResult
  ├── HAS_VITAL → Vital
  ├── HAS_FINDING → Finding
  ├── HAS_PROCEDURE → Procedure
  └── HAS_FAMILY_HISTORY → Disease
```

Research how to represent:

- positive findings,
- negative findings,
- uncertain findings,
- historical findings,
- resolved findings,
- temporal relationships,
- severity,
- frequency,
- onset,
- duration,
- confidence,
- source,
- provenance.

For example, determine how the graph should distinguish:

> "Patient has chest pain"

from:

> "Patient denies chest pain"

and:

> "Patient previously had chest pain in 2022."

Investigate standards such as:

- FHIR,
- SNOMED CT,
- LOINC,
- RxNorm,
- ICD,
- UMLS,
- OMOP/OHDSI.

Recommend the best standards for the student prototype.

---

# 4. Design the Medical Knowledge Graph

Research how to construct a medical KG containing:

### Entities

- Diseases
- Symptoms
- Signs
- Drugs
- Drug classes
- Treatments
- Procedures
- Laboratory findings
- Imaging findings
- Risk factors
- Contraindications
- Adverse effects
- Complications
- Differential diagnoses
- Clinical guidelines
- Medical evidence
- Research papers

### Relationships

Examples:

```text
Disease → HAS_SYMPTOM → Symptom
Disease → HAS_SIGN → Sign
Disease → RISK_FACTOR → RiskFactor
Disease → DIAGNOSED_BY → Test
Disease → ASSOCIATED_WITH → LabFinding
Disease → ASSOCIATED_WITH → ImagingFinding
Disease → TREATED_BY → Drug
Drug → CONTRAINDICATED_IN → Disease
Drug → HAS_ADVERSE_EFFECT → Effect
Disease → DIFFERENTIAL_WITH → Disease
Disease → COMPLICATION → Disease
```

Research:

- ontology design,
- schema design,
- RDF vs property graph,
- Neo4j vs RDF triple stores,
- Cypher vs SPARQL,
- OWL,
- ontology reasoning,
- graph validation,
- provenance,
- temporal/versioned medical knowledge.

Recommend one practical graph technology.

---

# 5. Research Medical Knowledge Sources and Licensing

Deeply investigate current access, licensing, restrictions, update frequency, and academic suitability of:

- UMLS
- SNOMED CT
- ICD-10 / ICD-11
- MedDRA
- RxNorm
- LOINC
- MeSH
- DrugBank
- PubChem
- NDF-RT
- HPO
- OMIM where appropriate
- FHIR terminology resources
- publicly available medical KGs
- guideline repositories
- WHO resources
- NIH/NLM resources
- CDC resources
- NICE guidelines
- other authoritative guideline sources

For every source provide:

| Resource | Content | Access | License | Commercial restrictions | Academic suitability | Role in project |
|---|---|---|---|---|---|---|

Explicitly identify:

- completely open/public-domain resources,
- free but registration-required resources,
- research-only resources,
- restricted resources,
- resources that should NOT be redistributed.

Do not recommend using proprietary data simply because it is technically useful.

Verify all licensing information from official sources.

---

# 6. Research Patient-Level Clinical Datasets

Deeply investigate:

- MIMIC-IV
- MIMIC-III
- eICU
- PhysioNet datasets
- other de-identified EHR datasets
- synthetic patient datasets
- clinical case report datasets
- diagnosis prediction datasets
- laboratory datasets
- clinical note datasets
- medical imaging datasets where relevant.

For each, determine:

- what information it contains,
- whether diagnoses are available,
- whether patient history is longitudinal,
- whether notes are available,
- whether labs/vitals/medications are available,
- access requirements,
- data-use agreement,
- suitability for a student project,
- preprocessing difficulty,
- computational requirements,
- potential leakage problems.

Prioritize datasets that allow a realistic experiment where the final diagnosis can be hidden and the system can generate a differential diagnosis.

---

# 7. Research Medical QA and Reasoning Datasets

Investigate:

- MedQA
- MedMCQA
- PubMedQA
- MMLU medical subsets
- MedExQA
- medical reasoning datasets
- clinical case datasets
- differential-diagnosis datasets
- medical exam datasets
- clinical decision-support benchmarks.

Explain which datasets are useful for:

- general medical QA,
- medical reasoning,
- evidence retrieval,
- diagnosis ranking,
- clinical NLP,
- entity extraction,
- evaluation only.

Do not assume a medical QA dataset is appropriate for patient-level diagnosis prediction.

---

# 8. Clinical NLP Pipeline

Research how to convert unstructured clinical text into structured patient facts.

Required pipeline:

```text
Clinical document
→ sentence segmentation
→ clinical NER
→ entity linking
→ negation detection
→ temporality detection
→ assertion status
→ relation extraction
→ normalization
→ patient KG
```

Research tools/models such as:

- ClinicalBERT
- BioBERT
- PubMedBERT
- SapBERT
- MedSpaCy
- scispaCy
- Hugging Face biomedical models
- UMLS linking tools
- SNOMED terminology services.

Investigate how to identify:

- diseases,
- symptoms,
- drugs,
- lab values,
- procedures,
- anatomical findings,
- family history,
- temporal expressions,
- negation.

Recommend a practical approach rather than simply listing models.

---

# 9. Patient Entity Linking and Medical Concept Normalization

Research how terms such as:

```text
"high blood pressure"
"HTN"
"hypertension"
```

can be mapped to the same standardized concept.

Investigate:

- UMLS concept linking,
- SNOMED CT normalization,
- SapBERT,
- lexical matching,
- embedding similarity,
- hybrid entity linking.

Explain how concept IDs should be stored in the KG.

---

# 10. Differential Diagnosis Engine

This is the core ML/reasoning problem.

Research several approaches:

### Rule-based

```text
IF fever + cough + infiltrate
THEN pneumonia becomes relevant
```

### Feature-based ML

- Logistic Regression
- Random Forest
- XGBoost
- SVM
- neural networks

### Deep learning

- MLP
- transformer
- clinical transformer

### Knowledge graph approaches

- graph traversal
- path scoring
- Personalized PageRank
- KG embeddings
- TransE
- RotatE
- DistMult
- ComplEx

### Graph neural networks

- GCN
- GraphSAGE
- R-GCN
- GAT

### Hybrid neuro-symbolic approaches

Research how structured graph reasoning and ML can be combined.

Determine which approaches are realistically implementable by students.

Recommend:

- one primary approach,
- one baseline,
- one stretch approach.

---

# 11. Diagnostic Candidate Ranking

Design a scoring system that considers:

- matching symptoms,
- absent symptoms,
- lab abnormalities,
- vital abnormalities,
- imaging findings,
- medical history,
- risk factors,
- age,
- sex,
- medications,
- contraindications,
- temporal information,
- evidence strength,
- conflicting evidence.

Research whether the system should produce:

- ranking scores,
- probabilities,
- confidence scores,
- calibrated probabilities.

Do not call a raw ML score a probability unless it is properly calibrated.

Research:

- calibration,
- Brier score,
- Expected Calibration Error,
- reliability diagrams.

---

# 12. Evidence and RAG Layer

Research how to retrieve medical evidence for each candidate diagnosis.

Possible sources:

- clinical guidelines,
- PubMed,
- PubMed Central,
- systematic reviews,
- trusted medical references,
- drug databases,
- authoritative government/medical organizations.

Design:

```text
Candidate diagnosis
+
patient findings
→
retrieval query
→
document/chunk retrieval
→
reranking
→
evidence
→
LLM synthesis
```

Compare:

- vector RAG,
- keyword/BM25 retrieval,
- hybrid retrieval,
- KG retrieval,
- GraphRAG,
- KG-RAG.

Recommend the best approach for the prototype.

---

# 13. KG + RAG Integration

Research exactly how KG and RAG should interact.

Compare:

### KG-first

```text
Patient KG
→ candidate diseases
→ RAG evidence retrieval
```

### RAG-first

```text
Patient information
→ document retrieval
→ KG validation
```

### Parallel

```text
Patient
 ↙      ↘
KG      RAG
 ↘      ↙
Reasoning
```

### Iterative

```text
Patient
→ KG
→ RAG
→ KG
→ RAG
→ final answer
```

Recommend one architecture based on accuracy, explainability, complexity, and compute.

---

# 14. LLM Role

Research whether the LLM should:

- extract structured information,
- generate retrieval queries,
- summarize evidence,
- explain diagnostic rankings,
- identify missing information,
- validate reasoning,
- generate the final doctor-facing report.

Explicitly investigate whether the LLM should be allowed to generate medical facts not present in retrieved evidence.

Recommend guardrails.

The LLM should ideally act as an **evidence synthesizer and interface**, not as an unconstrained source of medical knowledge.

---

# 15. Missing Information / Diagnostic Gap Detection

This is a major proposed feature.

Research how the system could identify information that would help distinguish between competing diagnoses.

Example:

```text
Candidate A: Pneumonia
Candidate B: Pulmonary embolism

Missing information:
- recent immobilization?
- recent surgery?
- unilateral leg swelling?
- relevant imaging?
- specific laboratory marker?
```

Research:

- information gain,
- entropy reduction,
- feature importance,
- Bayesian diagnosis,
- active learning,
- decision trees,
- clinical question selection.

Determine whether this can be implemented realistically.

---

# 16. Contradiction Detection

Research how the system can show:

### Supporting evidence

```text
✓ Fever
✓ Elevated WBC
✓ Imaging finding
```

### Contradicting evidence

```text
✗ No relevant symptom
✗ Normal test
```

### Missing evidence

```text
? Test not available
? Clinical history unknown
```

Research whether KG reasoning, rules, ML feature attribution, or a combination should generate this.

---

# 17. Explainability

The final output should be explainable to a physician.

Research:

- KG reasoning paths,
- evidence citations,
- feature importance,
- SHAP,
- counterfactual explanations,
- rule traces,
- provenance,
- evidence grading.

The system should ideally be able to answer:

> "Why is diagnosis A ranked above diagnosis B?"

and:

> "Which patient findings support this candidate?"

and:

> "Which medical sources support this conclusion?"

and:

> "What evidence contradicts it?"

---

# 18. Safety and Clinical Risk

This is a medical decision-support system, so conduct a serious safety analysis.

Research:

- clinical AI safety,
- hallucination,
- automation bias,
- over-reliance,
- distribution shift,
- dataset bias,
- calibration,
- false negatives,
- false positives,
- missing data,
- contradictory records,
- outdated guidelines,
- incorrect entity linking,
- incorrect graph relationships,
- incorrect RAG retrieval,
- prompt injection through uploaded documents,
- malicious clinical text,
- privacy,
- de-identification,
- access control,
- audit logs.

Determine appropriate safeguards.

The system must:

- clearly identify itself as decision support,
- preserve physician responsibility,
- avoid claiming certainty,
- surface uncertainty,
- show evidence,
- expose contradictions,
- support clinician override,
- log important reasoning/provenance,
- avoid unnecessary patient-identifying information.

---

# 19. Regulatory and Ethical Considerations

Research the current regulatory landscape relevant to a student-built clinical decision-support prototype, especially:

- India,
- United States,
- EU if relevant.

Investigate:

- medical device/software considerations,
- clinical decision support regulation,
- data protection,
- HIPAA where applicable,
- GDPR where applicable,
- India's Digital Personal Data Protection framework,
- ethical approval requirements,
- institutional review boards,
- data-use agreements.

Clearly distinguish:

> academic research prototype

from:

> clinically deployable medical device.

Do not assume that using de-identified public data makes a system automatically safe for clinical deployment.

---

# 20. Evaluation Framework

Design a rigorous evaluation framework.

### Diagnostic performance

- Top-1 accuracy
- Top-3 accuracy
- Top-5 accuracy
- MRR
- Precision@K
- Recall@K
- macro/micro F1

### Evidence retrieval

- Recall@K
- Precision@K
- MRR
- nDCG

### Factuality

- supported claim rate
- unsupported claim rate
- hallucination rate
- citation correctness

### KG reasoning

- path correctness
- relation accuracy
- graph retrieval recall
- link prediction metrics
- Hits@1
- Hits@3
- Hits@10
- MRR

### Calibration

- Brier score
- ECE
- reliability curves

### Safety

- dangerous false-negative rate
- unsafe recommendation rate
- escalation accuracy
- contradiction detection
- uncertainty detection

### Explainability

Expert evaluation of:

- reasoning validity,
- evidence relevance,
- citation correctness,
- usefulness,
- clarity.

---

# 21. Baseline and Ablation Experiments

Design experiments comparing:

### Baseline 1

```text
ML only
```

### Baseline 2

```text
LLM only
```

### Baseline 3

```text
RAG + LLM
```

### Baseline 4

```text
Medical KG + graph reasoning
```

### Proposed

```text
Patient KG
+
Medical KG
+
ML ranking
+
RAG
+
LLM explanation
```

Ablate:

- remove patient KG,
- remove medical KG,
- remove RAG,
- remove ML ranking,
- remove contradiction detection,
- remove provenance,
- remove missing-information module.

Determine which components actually improve performance.

This should be a central part of the research because otherwise it will be difficult to demonstrate that the architecture is better than a simpler system.

---

# 22. Dataset Construction Strategy

Design concrete schemas for:

### Patient case

```text
patient_id
age
sex
symptoms
history
medications
allergies
vitals
labs
imaging_findings
clinical_notes
final_diagnosis
```

### KG triple

```text
subject
relation
object
source
source_type
confidence
timestamp
version
```

### Evidence record

```text
candidate_diagnosis
patient_finding
supporting_source
evidence_text
source_url
publication_date
evidence_strength
```

### Evaluation case

```text
case_id
patient_features
ground_truth_diagnosis
differential_diagnoses
relevant_evidence
missing_information
```

Explain how to avoid train/test leakage, especially when multiple records belong to the same patient or related clinical episodes.

---

# 23. Compute and Model Selection

Research practical compute requirements for:

- CPU-only development,
- laptop GPU,
- RTX-class local GPU,
- cloud GPU,
- LLM APIs,
- open-source local models.

Compare:

- fine-tuning,
- LoRA/QLoRA,
- frozen pretrained models,
- retrieval-only approaches.

Determine what should be trained from scratch and what should use pretrained models.

Strongly prioritize approaches that can realistically run within a student project budget.

---

# 24. Recommended Technology Stack

Evaluate and recommend tools for:

### Backend

- Python
- FastAPI

### ML

- PyTorch
- scikit-learn
- Hugging Face Transformers

### Clinical NLP

- scispaCy
- MedSpaCy
- biomedical transformers
- UMLS/SNOMED terminology tools

### Knowledge Graph

- Neo4j
- RDFLib / RDF stores as alternatives

### Vector retrieval

- FAISS
- Qdrant
- Chroma
- pgvector

### RAG

- LangChain
- LlamaIndex
- custom pipeline

Do not recommend a framework merely because it is popular; explain whether it is actually necessary.

---

# 25. Four-Person Team Architecture

Design a realistic division:

### Person 1 — Medical KG

- ontology
- data ingestion
- graph schema
- Neo4j
- graph reasoning

### Person 2 — Clinical NLP + ML

- NER
- entity linking
- structured patient extraction
- diagnostic ranking
- ML experiments

### Person 3 — RAG + LLM

- medical document retrieval
- embeddings
- RAG
- LLM integration
- evidence synthesis

### Person 4 — Backend + UI + Evaluation

- FastAPI
- frontend
- integration
- safety layer
- evaluation framework
- testing

Explain dependencies between workstreams.

---

# 26. 3–6 Month Implementation Plan

Create a detailed weekly plan.

Include:

- Week 1: requirements and literature review
- Weeks 2–4: datasets and KG
- Weeks 4–6: patient NLP
- Weeks 6–8: baseline ML
- Weeks 8–10: KG reasoning
- Weeks 10–12: RAG
- Weeks 12–14: integration
- Weeks 14–16: safety and explainability
- Weeks 16–20: evaluation
- Weeks 20–24: research experiments, paper/report, polish

Adjust the schedule based on actual complexity.

---

# 27. MVP Definition

Define the smallest version that demonstrates the core research contribution.

The MVP should ideally support:

1. Upload/enter a structured patient case.
2. Extract clinical concepts from text.
3. Normalize medical concepts.
4. Build a patient-specific KG.
5. Query a medical KG.
6. Generate candidate diagnoses.
7. Rank candidates.
8. Retrieve supporting evidence.
9. Display supporting/contradicting findings.
10. Show reasoning paths.
11. Show sources.
12. Show uncertainty and limitations.
13. Keep the physician as final decision-maker.

Define which advanced features should be postponed.

---

# 28. Stretch Goals

Evaluate feasibility of:

- GNN-based reasoning
- KG embeddings
- multimodal imaging analysis
- ECG integration
- Hindi-English clinical NLP
- longitudinal patient graphs
- temporal reasoning
- personalized guideline retrieval
- drug interaction reasoning
- counterfactual diagnosis
- active question generation
- agentic workflows
- FHIR integration
- clinician feedback loops.

Rank these by value vs implementation difficulty.

---

# 29. What NOT to Build

Explicitly identify features that are unrealistic or unsafe for a 3–6 month student project, such as:

- training a medical LLM from scratch,
- building a universal medical KG from scratch,
- autonomous diagnosis,
- autonomous treatment prescription,
- direct patient deployment,
- real hospital deployment,
- unrestricted EHR integration,
- claiming clinical efficacy without clinical validation,
- processing real identifiable patient data without proper authorization,
- huge multimodal medical foundation models.

---

# 30. Research Novelty

Identify realistic research contributions.

Potential directions:

1. Patient-specific KG + medical KG integration.
2. KG-guided differential diagnosis ranking.
3. KG + RAG evidence-grounded clinical reasoning.
4. Contradiction-aware diagnostic ranking.
5. Missing-information detection.
6. Explainable diagnostic paths.
7. Provenance-aware medical RAG.
8. Comparison of LLM-only vs RAG vs KG vs KG+RAG.
9. Calibration and uncertainty in KG-enhanced diagnostic assistance.
10. Neuro-symbolic clinical reasoning.

For each potential novelty, determine:

- how original it is,
- existing related work,
- implementation difficulty,
- evaluation requirements,
- potential for a student paper,
- whether it is sufficiently novel for a major project.

Do not falsely claim novelty; compare against recent literature.

---

# 31. Literature Review

Conduct a comprehensive literature review covering at minimum:

- Clinical Decision Support Systems
- Medical Knowledge Graphs
- Patient Knowledge Graphs
- Clinical NLP
- Medical entity linking
- Biomedical transformers
- Medical LLMs
- Medical RAG
- GraphRAG
- KG-RAG
- Neuro-symbolic reasoning
- Knowledge graph embeddings
- GNNs for healthcare
- Differential diagnosis models
- AI-assisted diagnosis
- Explainable AI in healthcare
- Uncertainty/calibration
- Clinical AI safety

Prioritize:

1. peer-reviewed papers,
2. official documentation,
3. official dataset pages,
4. authoritative medical organizations,
5. primary research papers,
6. high-quality surveys.

Avoid relying heavily on blogs or marketing pages.

---

# 32. Current-State Verification

The research should be **current as of August 2026**.

Verify:

- dataset availability,
- licensing,
- software versions where relevant,
- current medical KG resources,
- recent research,
- current regulatory information,
- current open-source models,
- current clinical AI literature.

For every important recommendation, provide the source and date where possible.

---

# 33. Final Required Output

Produce a detailed research report with the following sections:

1. Executive Summary
2. Refined Problem Statement
3. Clinical Use Case
4. Why This Problem Matters
5. Existing Systems and Literature
6. Proposed Architecture
7. Patient Knowledge Graph Design
8. Medical Knowledge Graph Design
9. KG + RAG Integration
10. Clinical NLP Pipeline
11. Diagnostic Ranking/Reasoning
12. Missing Information Detection
13. Contradiction Detection
14. LLM Role
15. Safety Architecture
16. Dataset Comparison
17. Licensing/Access Matrix
18. Recommended Dataset Combination
19. Recommended Models
20. Technology Stack
21. MVP
22. Stretch Goals
23. Evaluation Framework
24. Baselines
25. Ablation Experiments
26. Research Novelty
27. Risks and Failure Modes
28. Privacy/Ethics/Regulatory Issues
29. Team Division
30. 3–6 Month Timeline
31. Hardware/Compute Requirements
32. Estimated Budget
33. Step-by-Step Implementation Plan
34. Final Recommended Architecture
35. Final Research Questions/Hypotheses
36. References

---

# 34. Most Important Research Requirement

Do not merely describe technologies individually.

The research must answer:

> **Exactly how should Patient KG + Medical KG + ML + RAG + LLM work together in one practical system?**

Give a concrete end-to-end example such as:

```text
Doctor uploads patient history
        ↓
Clinical NLP
        ↓
Medical entity extraction
        ↓
Negation + temporal detection
        ↓
Concept normalization
        ↓
Patient Knowledge Graph
        ↓
Medical Knowledge Graph retrieval
        ↓
Candidate disease generation
        ↓
ML/graph-based ranking
        ↓
Contradiction analysis
        ↓
Missing-information analysis
        ↓
Medical guideline/literature retrieval
        ↓
Evidence synthesis
        ↓
LLM explanation
        ↓
Safety/consistency validation
        ↓
Doctor dashboard
        ↓
Doctor makes final decision
```

For every arrow, explain:

- what happens,
- what model/tool performs it,
- what data it consumes,
- what it outputs,
- what can go wrong,
- how it is evaluated.

---

# 35. Final Recommendation Requirement

At the end, do not give 10 equally weighted possibilities.

Make a **clear recommendation** for a 4-person student team:

- exact architecture,
- exact KG technology,
- exact primary datasets,
- exact clinical NLP approach,
- exact baseline,
- exact diagnostic-ranking approach,
- exact RAG approach,
- exact LLM strategy,
- exact evaluation strategy,
- what to implement first,
- what to leave as stretch goals.

The goal is to produce a research-backed blueprint that a 4-person student team can immediately begin implementing while still having a meaningful AI/ML research contribution.
