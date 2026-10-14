"""Turning PMC Open Access articles into passages (task 3a): JATS parsing, licences, chunking.

Standard library only. The Kaggle notebook (``notebooks/kaggle_retrieval_corpus.ipynb``) runs it on
what NCBI's EFetch returns; the tests run it on small invented XML, so the parsing is checked
before the notebook ever runs.

* :func:`parse_articles` reads a ``<pmc-articleset>`` (JATS XML) into :class:`Article` records,
  or the reason each one was dropped. **An article is kept only with a Creative Commons licence
  stated in the XML** (:func:`licence_code`), on top of the search's ``"open access"[filter]``:
  only the PMC Open Access subset may be kept (docs/03 §1.4). Retractions, corrections and
  expressions of concern are dropped by article type.
* Sections are kept for what they say about a condition: methods, statistics and the
  administrative sections (funding, conflicts, author contributions…) are dropped by title
  (:data:`DROPPED_SECTIONS`); tables, figures, formulas and citation markers are not text.
* :func:`chunk_spans` cuts a section into the configured windows (``retrieval.chunk_size`` 512,
  ``chunk_overlap`` 64), counted in the embedding model's own WordPiece tokens. It works on the
  tokenizer's character offsets, so every passage is an exact substring of its section.
* :func:`article_passages` chunks only the runs of sentences free of treatment language
  (src/rag/screening.py): **no passage holds a treatment sentence** (FR-6.5).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "DROPPED_ARTICLE_TYPES",
    "DROPPED_SECTIONS",
    "Article",
    "article_passages",
    "article_url",
    "chunk_spans",
    "licence_code",
    "normalise_pmcid",
    "parse_articles",
]

DROPPED_SECTIONS = re.compile(
    r"method|material|statistic|acknowledg|funding|financial|author.{0,3}contribution|conflict|"
    r"competing|disclosure|ethic|consent|data.{0,3}availab|availability|supplementa|abbreviation|"
    r"reference|declaration|limitation|"
    r"treat|therap|manag|surg|intervention|pharmacolog|medication|prophyla|prevention|"
    r"follow.{0,3}up|outcome",
    re.IGNORECASE,
)
"""Section titles whose text is dropped. A methods section describes a study, not a condition;
the administrative ones describe the paper. "Limitations" goes too: it is about the study, and
reads as a claim about the condition when quoted alone. **So do the sections about treatment,
prevention and outcome** ("Management", "Treatment and follow-up"…): evidence is about diagnosis
only (FR-6.5), and what the remaining sections say about treatment is screened sentence by
sentence (src/rag/screening.py)."""

DROPPED_ARTICLE_TYPES: frozenset[str] = frozenset(
    {"correction", "retraction", "expression-of-concern", "addendum", "erratum", "removal-notice"}
)
"""JATS ``article-type`` values that are notices about other papers, not evidence."""

_SKIPPED_ELEMENTS = frozenset(
    {
        "table-wrap",
        "table-wrap-group",
        "table",
        "fig",
        "fig-group",
        "disp-formula",
        "xref",
        "supplementary-material",
        "graphic",
        "inline-graphic",
        "media",
        "fn-group",
        "ref-list",
        "label",
    }
)
_EMPTY_BRACKETS = re.compile(r"[\[(][\s,;–—-]*[\])]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:)\]])")
_CC = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z-]+)/?(\d(?:\.\d)?)?", re.I)
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


@dataclass
class Article:
    """One PMC article, as kept for the corpus.

    ``sections`` holds ``(title, text)`` pairs in reading order, the abstract first; ``title`` is
    "Abstract" for it and "Body" for text outside any section.
    """

    pmcid: str
    title: str
    licence: str
    licence_url: str | None = None
    pmid: str | None = None
    doi: str | None = None
    journal: str | None = None
    year: int | None = None
    authors: str | None = None
    article_type: str | None = None
    sections: list[tuple[str, str]] = field(default_factory=list)

    @property
    def url(self) -> str:
        return article_url(self.pmcid)

    def metadata(self) -> dict[str, Any]:
        """Everything but the text: one line of the bundle's ``articles.jsonl``."""
        record = asdict(self)
        del record["sections"]
        record["url"] = self.url
        return record


def article_url(pmcid: str) -> str:
    """The article's page on PMC, which carries its full citation and licence."""
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"


def normalise_pmcid(value: str) -> str | None:
    """ "PMC1234567" from "PMC1234567", "pmc1234567" or the bare "1234567", else None."""
    digits = value.strip().upper().removeprefix("PMC")
    return f"PMC{digits}" if digits.isdigit() else None


def licence_code(url_or_text: str | None) -> str | None:
    """A Creative Commons licence as a short code ("CC BY 4.0", "CC BY-NC-ND 3.0", "CC0 1.0"), from
    its URL, or failing that from its name in a sentence ("…under the Creative Commons
    Attribution-NonCommercial 4.0 International License", "CC BY-NC 4.0"); None for anything
    else, including a licence the text only hints at."""
    if not url_or_text:
        return None
    match = _CC.search(url_or_text)
    if match:
        kind, name, version = match.group(1).lower(), match.group(2).lower(), match.group(3)
        if kind == "publicdomain":
            if name != "zero":
                return None
            code = "CC0"
        else:
            parts = name.split("-")
            if parts[0] != "by" or not set(parts) <= {"by", "nc", "nd", "sa"}:
                return None
            code = "CC " + name.upper()
        return f"{code} {version}" if version else code
    return _licence_from_text(url_or_text)


_CC_SHORT = re.compile(r"\bCC[ -]?(BY(?:[ -](?:NC|ND|SA))*)(?: (\d\.\d))?\b")
_CC_LONG = re.compile(
    r"creative\s+commons\s+attribution((?:[\s-]+(?:non-?commercial|no-?deriv\w*|share-?alike))*)"
    r"(?:[\s-]+(?:license\s+)?(\d\.\d))?",
    re.IGNORECASE,
)
_CC_ZERO = re.compile(r"\bCC0\b|creative\s+commons\s+zero|public\s+domain\s+dedication", re.I)


def _licence_from_text(text: str) -> str | None:
    short = _CC_SHORT.search(text)
    if short:
        code = "CC " + short.group(1).upper().replace(" ", "-")
        return f"{code} {short.group(2)}" if short.group(2) else code
    long = _CC_LONG.search(text)
    if long:
        parts = long.group(1).lower()
        code = "CC BY" + "".join(
            suffix
            for word, suffix in (("commercial", "-NC"), ("deriv", "-ND"), ("share", "-SA"))
            if word in parts
        )
        return f"{code} {long.group(2)}" if long.group(2) else code
    if _CC_ZERO.search(text):
        return "CC0"
    return None


def _local(tag: Any) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _text(element: ET.Element | None) -> str:
    """An element's text, without tables, figures, formulas or citation markers, whitespace
    collapsed and the brackets left empty by removed citations dropped."""
    if element is None:
        return ""
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if _local(child.tag) not in _SKIPPED_ELEMENTS:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(element)
    text = " ".join("".join(parts).split())
    text = _EMPTY_BRACKETS.sub("", text)
    return " ".join(_SPACE_BEFORE_PUNCT.sub(r"\1", text).split())


def _find(element: ET.Element, *path: str) -> ET.Element | None:
    """The first descendant matching ``path`` of local names (namespaces ignored)."""
    if not path:
        return element
    for child in element.iter():
        if child is not element and _local(child.tag) == path[0]:
            found = _find(child, *path[1:])
            if found is not None:
                return found
    return None


def _children(element: ET.Element, name: str) -> Iterator[ET.Element]:
    return (c for c in element if _local(c.tag) == name)


def _ids(meta: ET.Element) -> dict[str, str]:
    ids: dict[str, str] = {}
    for node in meta.iter():
        if _local(node.tag) == "article-id" and node.text:
            ids.setdefault(node.get("pub-id-type", ""), node.text.strip())
    return ids


def _licence(meta: ET.Element) -> tuple[str | None, str | None]:
    """(code, url) of the article's Creative Commons licence, or (None, None). The URL is the
    licence's canonical one when the XML gives it, else None."""
    permissions = _find(meta, "permissions")
    if permissions is None:
        return None, None
    for node in permissions.iter():
        if _local(node.tag) != "license":
            continue
        candidates = [node.get(_XLINK_HREF), node.get("href")]
        candidates += [c.text for c in node.iter() if _local(c.tag) == "license_ref"]
        candidates += [c.get(_XLINK_HREF) for c in node.iter() if _local(c.tag) == "ext-link"]
        candidates.append(_text(node))
        for candidate in candidates:
            code = licence_code(candidate)
            if code:
                found = _CC.search(candidate or "")
                return code, ("https://" + found.group(0).rstrip("/") + "/") if found else None
    return None, None


def _year(meta: ET.Element) -> int | None:
    years = [
        int(y.text)
        for date in meta.iter()
        if _local(date.tag) == "pub-date"
        for y in _children(date, "year")
        if y.text and y.text.strip().isdigit()
    ]
    return min(years) if years else None


def _authors(meta: ET.Element) -> str | None:
    """ "Surname AB" for the first author, "et al." after it when there are more."""
    names = []
    for contrib in meta.iter():
        if _local(contrib.tag) == "contrib" and contrib.get("contrib-type", "author") == "author":
            name = _find(contrib, "name")
            if name is None:
                collab = _find(contrib, "collab")
                if collab is not None and _text(collab):
                    names.append(_text(collab))
                continue
            surname = _text(_find(name, "surname"))
            given = _text(_find(name, "given-names"))
            initials = "".join(w[0] for w in re.split(r"[\s.-]+", given) if w)
            if surname:
                names.append(f"{surname} {initials}".strip())
    if not names:
        return None
    return names[0] + (" et al." if len(names) > 1 else "")


def _abstract(meta: ET.Element) -> str:
    texts = []
    for abstract in meta.iter():
        if _local(abstract.tag) != "abstract" or abstract.get("abstract-type") in {
            "graphical",
            "teaser",
            "toc",
        }:
            continue
        for node in abstract.iter():
            if _local(node.tag) == "p" and _text(node):
                texts.append(_text(node))
        if texts:
            break
    return " ".join(texts)


def _sections(body: ET.Element) -> list[tuple[str, str]]:
    """``(top-level section title, text)`` pairs, nested sections folded into their top-level
    one; a section is dropped when its own title or an ancestor's matches DROPPED_SECTIONS."""
    out: list[tuple[str, str]] = []

    def paragraphs(node: ET.Element, dropped: bool) -> list[str]:
        texts = []
        for child in node:
            name = _local(child.tag)
            if name == "p":
                if not dropped and _text(child):
                    texts.append(_text(child))
            elif name == "sec":
                title = _text(next(_children(child, "title"), None))
                texts += paragraphs(child, dropped or bool(DROPPED_SECTIONS.search(title)))
            elif name not in _SKIPPED_ELEMENTS and name != "title":
                texts += paragraphs(child, dropped)
        return texts

    loose = [_text(p) for p in _children(body, "p") if _text(p)]
    if loose:
        out.append(("Body", " ".join(loose)))
    for sec in _children(body, "sec"):
        title_node = next(_children(sec, "title"), None)
        title = _text(title_node) or "Body"
        if DROPPED_SECTIONS.search(title):
            continue
        text = " ".join(paragraphs(sec, False))
        if text:
            out.append((title, text))
    return out


def _cap(sections: list[tuple[str, str]], max_chars: int) -> list[tuple[str, str]]:
    """The sections in order until ``max_chars`` characters. The one that crosses the budget is
    cut at its last sentence end inside it, or at a space when that would lose more than half."""
    kept, used = [], 0
    for title, text in sections:
        room = max_chars - used
        if room <= 0:
            break
        if len(text) > room:
            sentence_end = text.rfind(". ", 0, room)
            if sentence_end >= room // 2:
                text = text[: sentence_end + 1]
            else:
                space = text.rfind(" ", 0, room)
                text = text[:space] if space > 0 else text[:room]
        if text:
            kept.append((title, text))
            used += len(text)
    return kept


def parse_articles(
    xml: bytes | str, *, max_chars: int = 20_000
) -> tuple[list[Article], list[dict[str, str]]]:
    """The articles of one EFetch response (``<pmc-articleset>``), and the ones dropped.

    Args:
        xml: The response body.
        max_chars: How much text to keep per article, the abstract first (a cap on the corpus's
            size, and on how far one long review can crowd out the rest).

    Returns:
        ``(kept, dropped)``; each dropped entry is ``{"pmcid", "reason"}``.

    Raises:
        xml.etree.ElementTree.ParseError: when the response is not XML at all; the notebook then
            fetches the batch one article at a time.
    """
    root = ET.fromstring(xml)
    articles = (
        [root]
        if _local(root.tag) == "article"
        else [child for child in root if _local(child.tag) == "article"]
    )
    kept: list[Article] = []
    dropped: list[dict[str, str]] = []
    for element in articles:
        meta = _find(element, "front", "article-meta")
        if meta is None:
            dropped.append({"pmcid": "", "reason": "no article-meta"})
            continue
        ids = _ids(meta)
        # PMC has written the id as "pmc", "pmcid" and, digits only, "pmc-uid". A wrong guess
        # cannot mislabel an article: the build keeps only articles whose id it asked for.
        pmcid = normalise_pmcid(ids.get("pmcid") or ids.get("pmc") or ids.get("pmc-uid") or "")
        if pmcid is None:
            dropped.append({"pmcid": "", "reason": "no PMC id"})
            continue
        article_type = element.get("article-type")
        if article_type in DROPPED_ARTICLE_TYPES:
            dropped.append({"pmcid": pmcid, "reason": f"article type {article_type}"})
            continue
        code, url = _licence(meta)
        if code is None:
            dropped.append({"pmcid": pmcid, "reason": "no Creative Commons licence in the XML"})
            continue
        title = _text(_find(meta, "title-group", "article-title"))
        if not title or re.match(r"(retraction|retracted|expression of concern)\b", title, re.I):
            dropped.append({"pmcid": pmcid, "reason": "no title, or a retraction notice"})
            continue
        sections = []
        abstract = _abstract(meta)
        if abstract:
            sections.append(("Abstract", abstract))
        body = _find(element, "body")
        if body is not None:
            sections += _sections(body)
        sections = _cap(sections, max_chars)
        if not sections:
            dropped.append({"pmcid": pmcid, "reason": "no abstract or body text"})
            continue
        journal = _find(element, "journal-title")
        kept.append(
            Article(
                pmcid=pmcid,
                title=title,
                licence=code,
                licence_url=url,
                pmid=ids.get("pmid"),
                doi=ids.get("doi"),
                journal=_text(journal) or None,
                year=_year(meta),
                authors=_authors(meta),
                article_type=article_type,
                sections=sections,
            )
        )
    return kept, dropped


def chunk_spans(
    offsets: Sequence[tuple[int, int]], window: int, overlap: int
) -> list[tuple[int, int]]:
    """Character spans of overlapping windows of ``window`` tokens, ``overlap`` shared.

    Args:
        offsets: Each token's ``(start, end)`` characters in the text, in order, as a fast
            tokenizer's ``return_offsets_mapping`` gives them (without special tokens).
        window: Tokens per window. The notebook passes ``chunk_size - 2``: the model adds [CLS]
            and [SEP], and ``chunk_size`` (512) is BERT's whole input, so nothing is truncated.
        overlap: Tokens each window shares with the one before (``chunk_overlap``, 64).

    Windows begin and end at word boundaries (a token preceded by a gap begins a word), so
    re-tokenizing a passage gives back the same tokens and never more than ``window``; a window
    with no boundary inside it (one "word" longer than the window) is cut where it falls. The last
    window may be shorter.

    Raises:
        ValueError: unless ``0 <= overlap < window``.
    """
    if not 0 <= overlap < window:
        raise ValueError(f"need 0 <= overlap < window, got overlap {overlap}, window {window}")
    n = len(offsets)
    if n == 0:
        return []

    def begins_word(i: int) -> bool:
        return i == 0 or offsets[i][0] > offsets[i - 1][1]

    spans = []
    start = 0
    while True:
        end = min(start + window, n)  # exclusive
        if end < n:
            snapped = next((j for j in range(end, start, -1) if begins_word(j)), None)
            if snapped is not None:
                end = snapped
        spans.append((offsets[start][0], offsets[end - 1][1]))
        if end >= n:
            return spans
        nxt = max(end - overlap, start + 1)
        nxt = next((j for j in range(nxt, end) if begins_word(j)), nxt)
        start = nxt


def article_passages(
    article: Article,
    offsets_of: Callable[[str], Sequence[tuple[int, int]]],
    window: int,
    overlap: int,
    min_chars: int | None = None,
) -> list[dict[str, str]]:
    """The article's passages: ``{"pmcid", "section", "text"}``, in reading order.

    Each section is first cut into its runs of consecutive sentences free of treatment language
    (:func:`src.rag.screening.diagnostic_spans`), and only those runs are chunked
    (:func:`chunk_spans`), so **every passage is a verbatim, contiguous piece of one section and
    holds no treatment sentence** (FR-6.5).

    Args:
        article: A kept article.
        offsets_of: A text's token character offsets, without special tokens: in the notebook the
            embedding model's fast tokenizer (``return_offsets_mapping``); in the tests a
            whitespace split.
        window, overlap: As for :func:`chunk_spans`.
        min_chars: The shortest run kept (default
            :data:`src.rag.screening.MIN_SEGMENT_CHARS`).
    """
    from src.rag.screening import MIN_SEGMENT_CHARS, diagnostic_spans

    floor = MIN_SEGMENT_CHARS if min_chars is None else min_chars
    out = []
    for section, text in article.sections:
        for start, end in diagnostic_spans(text, floor):
            segment = text[start:end]
            for lo, hi in chunk_spans(offsets_of(segment), window, overlap):
                piece = segment[lo:hi].strip()
                if piece:
                    out.append({"pmcid": article.pmcid, "section": section, "text": piece})
    return out
