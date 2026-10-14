"""Okapi BM25 over the passages, in NumPy (task 3a).

Written here rather than taken from ``rank-bm25`` because nothing is installed on the laptop (D-4,
2026-10-13), and because a score a test can compute by hand is worth more than a dependency.

For a query Q and a passage D, with the parameters of Robertson and Zaragoza (2009) that most
systems default to, **k1 = 1.2 and b = 0.75**:

    score(D, Q) = Σ_{t ∈ Q} idf(t) · f(t, D) · (k1 + 1) / (f(t, D) + k1 · (1 − b + b · |D| / avgdl))
    idf(t)      = ln(1 + (N − n(t) + 0.5) / (n(t) + 0.5))

* ``f(t, D)`` is how often term t occurs in D, ``|D|`` D's length in terms, ``avgdl`` the mean
  length, N the number of passages and ``n(t)`` how many contain t.
* The idf is Lucene's, which stays positive: the classic ``ln((N − n + 0.5) / (n + 0.5))`` goes
  negative for a term in more than half the passages and then punishes a passage for containing
  it.
* Q is a **set**: a term repeated in the query counts once.
* Terms come from :func:`src.rag.text.tokenize`; a query term absent from every passage scores
  nothing.

The index is term-major (each term's postings contiguous, like a CSC matrix), so a query costs
one slice per query term.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from src.rag.text import tokenize

__all__ = ["B", "K1", "BM25Index"]

K1 = 1.2
B = 0.75


class BM25Index:
    """BM25 over a fixed list of documents, each already a list of terms.

    Args:
        documents: One term list per passage (``tokenize(text)``), in passage order.
        k1, b: The BM25 parameters; the defaults are fixed, not tuned (T-6's spirit: nothing in
            retrieval is tuned before EXP-011 has a labelled set to tune on, validation only).
    """

    def __init__(self, documents: Sequence[Sequence[str]], *, k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.vocabulary: dict[str, int] = {}
        term_ids: list[int] = []
        doc_ids: list[int] = []
        counts: list[int] = []
        lengths = np.zeros(self.n_docs, dtype=np.float64)
        for d, terms in enumerate(documents):
            lengths[d] = len(terms)
            for term, count in Counter(terms).items():
                term_ids.append(self.vocabulary.setdefault(term, len(self.vocabulary)))
                doc_ids.append(d)
                counts.append(count)
        order = np.argsort(np.asarray(term_ids, dtype=np.int64), kind="stable")
        self._docs = np.asarray(doc_ids, dtype=np.int64)[order]
        self._tf = np.asarray(counts, dtype=np.float64)[order]
        per_term = np.bincount(np.asarray(term_ids, dtype=np.int64), minlength=len(self.vocabulary))
        self._ptr = np.concatenate([[0], np.cumsum(per_term)])
        self.doc_freq = per_term
        self.lengths = lengths
        self.avgdl = float(lengths.mean()) if self.n_docs else 0.0
        n = per_term.astype(np.float64)
        self.idf = np.log1p((self.n_docs - n + 0.5) / (n + 0.5))
        # The length normaliser of every passage, computed once: k1 · (1 − b + b · |D| / avgdl).
        self._norm = k1 * (1.0 - b + b * lengths / self.avgdl) if self.avgdl > 0 else lengths * 0.0

    @classmethod
    def from_texts(cls, texts: Iterable[str], **kwargs: float) -> BM25Index:
        return cls([tokenize(t) for t in texts], **kwargs)

    def query_terms(self, query: str | Sequence[str]) -> list[str]:
        """The query's distinct terms that occur somewhere in the index, in query order."""
        terms = tokenize(query) if isinstance(query, str) else list(query)
        return [t for t in dict.fromkeys(terms) if t in self.vocabulary]

    def scores(self, query: str | Sequence[str]) -> np.ndarray:
        """BM25 of every passage for ``query`` (a text, or terms already tokenized)."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        for term in self.query_terms(query):
            t = self.vocabulary[term]
            lo, hi = self._ptr[t], self._ptr[t + 1]
            docs, tf = self._docs[lo:hi], self._tf[lo:hi]
            # Each passage appears once in a term's postings, so a plain fancy-index add is exact.
            out[docs] += self.idf[t] * tf * (self.k1 + 1.0) / (tf + self._norm[docs])
        return out
