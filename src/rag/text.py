"""Text handling for retrieval: the BM25 tokenizer, phrase mentions and excerpts (task 3a).

Standard library only, so the Kaggle notebook and the laptop share it.

**The tokenizer is deliberately simple, and stated exactly** so a BM25 score can be computed by
hand (tests/test_retrieval.py does):

1. ``casefold`` the text, then take the runs of ASCII letters and digits (``[a-z0-9]+``). Hyphens
   and apostrophes split words: "gastro-oesophageal" → "gastro", "oesophageal"; "Boerhaave's" →
   "boerhaave", "s".
2. Drop single letters and a short list of English function words (:data:`STOPWORDS`). Digits are
   kept, even alone: "type 2 diabetes".
3. Fold British spellings into American: "oe" and "ae" become "e" anywhere in a word ("oedema" →
   "edema", "ischaemia" → "ischemia", "oesophageal" → "esophageal"). The fold is applied to the
   query and the passages alike, so where it mangles a word ("boerhaave" → "berhaave") both sides
   are mangled the same way, and nothing is lost unless two distinct medical words collide.
4. Strip one plural "s" from words of four letters or more, except after "s", "u" or "i"
   ("attacks" → "attack", but "loss", "embolus", "pericarditis", "diagnosis" stay whole).

There is **no stemming** beyond that ("painful" and "pain" stay distinct) and no synonym
expansion: the dense half of the hybrid (src/rag/retriever.py) is what matches meaning rather than
words.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import lru_cache

__all__ = [
    "STOPWORDS",
    "excerpt",
    "mentions_any",
    "phrase_tokens",
    "sentences",
    "tokenize",
]

_WORD = re.compile(r"[a-z0-9]+")
_FOLD = re.compile(r"[oa]e")

_STOPWORD_TEXT = """
    an and are as at be been being but by can could did do does for from had has have he her
    his how if in into is it its may might more most no nor not of on or our she should so such
    than that the their them then there these they this those to was we were what when where
    which while who whom why will with would you your
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())
"""Function words dropped by :func:`tokenize`. Negations ("no", "not") go too: BM25 counts words
and cannot read a negation anyway, so keeping them would only reward passages that negate."""


@lru_cache(maxsize=1 << 18)
def _term(word: str) -> str | None:
    """One casefolded word as its BM25 term, or None for a dropped word (rules 2–4).

    Cached: a corpus repeats a few tens of thousands of distinct words millions of times, and the
    cache turns the per-word rules into one dictionary lookup (the retriever's start-up, EXP-011).
    """
    if word in STOPWORDS or (len(word) == 1 and not word.isdigit()):
        return None
    word = _FOLD.sub("e", word)
    if len(word) >= 4 and word.endswith("s") and word[-2] not in "sui":
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """The text as BM25 terms, in order (the rules are in the module docstring)."""
    return [term for word in _WORD.findall(text.casefold()) if (term := _term(word)) is not None]


def phrase_tokens(phrase: str) -> tuple[str, ...]:
    """A phrase as the token sequence a passage must contain for :func:`mentions_any`."""
    return tuple(tokenize(phrase))


def mentions_any(
    tokens: Sequence[str],
    phrases: Iterable[tuple[str, ...]],
    token_set: set[str] | frozenset[str] | None = None,
) -> bool:
    """Whether ``tokens`` contains one of ``phrases`` as consecutive tokens.

    Matching whole tokens is what keeps "stable angina" out of a passage about "unstable angina":
    "unstable" is one token, not "un" + "stable". ``token_set`` (``set(tokens)``) is an optional
    shortcut: a phrase with a token the passage lacks cannot match, which rules out most passages
    without scanning them.
    """
    present = token_set if token_set is not None else set(tokens)
    joined: str | None = None
    for phrase in phrases:
        if not phrase or not all(t in present for t in phrase):
            continue
        if len(phrase) == 1:
            return True
        if joined is None:
            joined = " " + " ".join(tokens) + " "
        if " " + " ".join(phrase) + " " in joined:
            return True
    return False


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def sentences(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` character spans of the sentences in ``text``: a split after ".", "!" or
    "?" followed by a capital, a digit or a bracket. Crude ("e.g. Smith" splits), but only used to
    choose an excerpt, never to judge a claim."""
    spans = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def excerpt(text: str, query_terms: Iterable[str], limit: int = 320) -> str:
    """The sentence of ``text`` that holds most of ``query_terms``, for showing beside a citation.

    The sentence with the most distinct query terms wins, the earliest on a tie; one shorter than
    a third of ``limit`` is extended by the sentences after it. The result is cut at a word
    boundary to at most ``limit`` characters, and "…" marks text left out at either end. It is a
    verbatim substring of the passage, apart from the marks: selected, never generated.
    """
    text = " ".join(text.split())
    if not text:
        return ""
    terms = set(query_terms)
    spans = sentences(text) or [(0, len(text))]
    best = max(
        range(len(spans)),
        key=lambda i: (len(terms & set(tokenize(text[spans[i][0] : spans[i][1]]))), -i),
    )
    start, end = spans[best]
    following = best + 1
    while end - start < limit // 3 and following < len(spans):
        end = spans[following][1]
        following += 1
    piece = text[start:end]
    cut_short = len(piece) > limit
    if cut_short:
        cut = piece.rfind(" ", 0, limit - 1)
        piece = piece[: cut if cut > 0 else limit - 1].rstrip(" ,;:")
    prefix = "…" if start > 0 else ""
    suffix = "…" if cut_short or end < len(text) else ""
    return prefix + piece + suffix
