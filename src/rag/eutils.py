"""A small, polite client for NCBI's E-utilities and PMC's OA service (task 3a).

Standard library only (``urllib``), for the Kaggle notebook; nothing on the laptop calls it. The
tests drive it with a fake opener and clock, so its rate limit and retries are checked offline.

Rules it keeps (docs/03 §1.4):

* **At most 3 requests a second**, NCBI's limit without an API key, across every call made
  through one client: requests are spaced at least :data:`MIN_INTERVAL` apart.
* **No API key and no credentials.** It sends ``tool`` (the project's name) and, only when the
  owner fills it in, ``email``: NCBI asks for a contact address so it can write before blocking a
  misbehaving tool. Nothing else about anyone is sent.
* On HTTP 429 or a server error it waits and retries (2, 4, 8… seconds, at most :data:`RETRIES`
  times): Kaggle's machines share addresses, so another user's traffic can trip NCBI's limit.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["EUTILS", "MIN_INTERVAL", "OA_SERVICE", "RETRIES", "EUtils", "EUtilsError"]

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OA_SERVICE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
MIN_INTERVAL = 0.35
"""Seconds between requests: a little over 1/3, so never more than three in any second."""
RETRIES = 5
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

Opener = Callable[[str, float], tuple[int, bytes]]
"""(url, timeout) -> (HTTP status, body). The default is urllib's; tests pass a fake."""


class EUtilsError(RuntimeError):
    """A request that still failed after every retry, or an answer NCBI marked as an error."""


def _urllib_opener(url: str, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": "nightingale-coursework/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read() or b""


class EUtils:
    """E-utilities over one rate limit.

    Args:
        tool: Sent as ``tool``, as NCBI asks.
        email: Sent as ``email`` only when not empty: the owner's choice, never filled in by code.
        opener, clock, sleep: Injected by the tests.
        timeout: Seconds per request.
    """

    def __init__(
        self,
        tool: str = "nightingale",
        email: str | None = None,
        *,
        opener: Opener | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 120.0,
        min_interval: float = MIN_INTERVAL,
        retries: int = RETRIES,
    ) -> None:
        self.tool = tool
        self.email = email or None
        self._open = opener or _urllib_opener
        self._clock = clock
        self._sleep = sleep
        self.timeout = timeout
        self.min_interval = min_interval
        self.retries = retries
        self._last: float | None = None
        self.requests = 0

    def _params(self, params: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in params.items() if v is not None}
        out["tool"] = self.tool
        if self.email:
            out["email"] = self.email
        return out

    def get(self, url: str, params: dict[str, Any]) -> bytes:
        """The body of ``url?params``, after the rate limit, retrying what can be retried.

        Raises:
            EUtilsError: after :attr:`retries` retries, or at once for another HTTP error.
        """
        full = url + "?" + urllib.parse.urlencode(self._params(params), safe=",")
        for attempt in range(self.retries + 1):
            if self._last is not None:
                wait = self.min_interval - (self._clock() - self._last)
                if wait > 0:
                    self._sleep(wait)
            self._last = self._clock()
            self.requests += 1
            try:
                status, body = self._open(full, self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                status, body = -1, str(error).encode()
            if status == 200:
                return body
            if status not in RETRY_STATUSES and status != -1:
                raise EUtilsError(f"HTTP {status} from {url}: {body[:200]!r}")
            if attempt < self.retries:
                self._sleep(min(2.0 ** (attempt + 1), 60.0))
        raise EUtilsError(f"{url} still failing after {self.retries} retries (last: {status})")

    def esearch(
        self,
        term: str,
        *,
        db: str = "pmc",
        retmax: int = 100,
        mindate: str | None = None,
        maxdate: str | None = None,
        sort: str | None = "relevance",
    ) -> dict[str, Any]:
        """ESearch's ``esearchresult``: ``count``, ``idlist`` (PMC's numeric ids, without "PMC"),
        ``querytranslation`` and any warnings, as NCBI returns them.

        Raises:
            EUtilsError: when NCBI reports an error for the query.
        """
        params = {
            "db": db,
            "term": term,
            "retmax": retmax,
            "retmode": "json",
            "sort": sort,
            "datetype": "pdat" if mindate or maxdate else None,
            "mindate": mindate,
            "maxdate": maxdate,
        }
        if mindate and not maxdate:
            params["maxdate"] = "3000"
        data = json.loads(self.get(EUTILS + "esearch.fcgi", params))
        result = data.get("esearchresult", {})
        if "ERROR" in result or "error" in data:
            raise EUtilsError(f"ESearch error for {term!r}: {result.get('ERROR') or data}")
        return result

    def efetch(self, ids: Sequence[str], *, db: str = "pmc") -> bytes:
        """EFetch's XML for ``ids`` (for PMC, a ``<pmc-articleset>`` of JATS articles)."""
        if not ids:
            raise ValueError("efetch needs at least one id")
        return self.get(EUTILS + "efetch.fcgi", {"db": db, "id": ",".join(ids), "retmode": "xml"})

    def oa_record(self, pmcid: str) -> dict[str, Any]:
        """What PMC's OA service says of one article: ``{"open_access": bool, "licence",
        "retracted", "error"}``. An article outside the Open Access subset comes back with
        ``open_access`` False and NCBI's error code (``idIsNotOpenAccess``)."""
        root = ET.fromstring(self.get(OA_SERVICE, {"id": pmcid}))
        error = root.find("error")
        if error is not None:
            return {"open_access": False, "error": error.get("code") or (error.text or "").strip()}
        record = root.find("./records/record")
        if record is None:
            return {"open_access": False, "error": "no record"}
        return {
            "open_access": True,
            "licence": record.get("license"),
            "retracted": (record.get("retracted") or "no").lower() == "yes",
            "error": None,
        }
