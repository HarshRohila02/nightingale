"""Nightingale rag module: evidence retrieval from the PMC Open Access subset (P3, task 3a).

- ``vocabulary``: what is searched for, per condition and per finding (hand-curated).
- ``eutils``: a rate-limited NCBI E-utilities client, for the Kaggle notebook.
- ``corpus``: PMC's JATS XML into articles and passages; licences; chunking.
- ``screening``: keeping treatment out of the evidence (FR-6.5): which sentences and titles may
  ever be stored or shown.
- ``build``: the Kaggle build, search to bundle, with NCBI and the model passed in (tested offline).
- ``bundle``: the folder the notebook writes and the laptop reads, checked against its manifest.
- ``text``: the BM25 tokenizer, phrase mentions and excerpts.
- ``bm25``: Okapi BM25 in NumPy.
- ``retriever``: ``HybridRetriever`` (BM25 + dense, fused by reciprocal rank) and
  ``open_retriever``, which degrades to no evidence, reporting ``rag``, without a bundle.

Nothing here imports a model: the corpus is embedded on Kaggle (D-7) and the laptop only reads
the vectors (``notebooks/kaggle_retrieval_corpus.ipynb``, docs/11 §4.6). The package imports
nothing at load, so the notebook can use its standard-library parts before NumPy is even needed.
Evidence is shown beside a ranking, never used to make one, and is about diagnosis only.
"""
