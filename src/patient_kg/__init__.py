"""The patient knowledge graph from Synthea's synthetic patients (task 1d).

* :mod:`src.patient_kg.synthea` reads Synthea's CSV output into per-patient records;
* :mod:`src.patient_kg.mapping` is the table from Synthea's SNOMED codes to the ``RF:*`` concepts,
  each code checked in Synthea's module files at the pinned release;
* :mod:`src.patient_kg.graph` builds one patient's graph as of a day, and reads its findings back.

Synthea runs on Kaggle (notebooks/kaggle_synthea.ipynb, D-7), never on the laptop; the builder is
scripts/build_patient_kg.py. Nothing here is wired into the ranking (docs/17 T7).
"""
