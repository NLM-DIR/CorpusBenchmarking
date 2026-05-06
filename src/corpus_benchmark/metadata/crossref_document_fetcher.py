import json
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

from corpus_benchmark.models.config import APP_NAME_VER
from corpus_benchmark.models.corpus import DocumentIdentifierType
from corpus_benchmark.metadata.document_fetcher import DocumentMetadataFetcher
from corpus_benchmark.metadata.eutils_journal_fetchers import ISSN
from corpus_benchmark.registry import register_document_fetcher

logger = logging.getLogger(__name__)

@register_document_fetcher("crossref_doi")
class CrossrefDOIFetcher(DocumentMetadataFetcher):
    """Queries the Crossref REST API for metadata using DOIs.

    Crossref supports exact DOI lookup through ``/works/{doi}``, but for many DOIs
    it is more efficient to query the ``/works`` endpoint with repeated ``doi``
    filters. This implementation batches DOIs into modest URL-safe chunks, uses
    the Crossref polite pool when ``mailto`` is provided, and honors the
    rate-limit headers returned by Crossref.
    """

    CROSSREF_WORKS_URL = "https://api.crossref.org/works"
    DEFAULT_BATCH_SIZE = 50
    DEFAULT_WAIT_SECONDS = 1.0
    MAX_RETRIES = 3

    def __init__(
        self,
        *,
        mailto: Optional[str] = None,
        user_agent: str = APP_NAME_VER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        timeout: int = 30,
    ):
        self.mailto = mailto
        self.user_agent = user_agent
        self.batch_size = batch_size
        self.wait_seconds = wait_seconds
        self.timeout = timeout
        self._last_request = 0.0
        self._dynamic_wait_seconds = wait_seconds

    @property
    def supported_id_type(self) -> DocumentIdentifierType:
        return DocumentIdentifierType.DOI

    def fetch(self, dois: List[str]) -> List[Dict[str, Any]]:
        if not dois:
            return []

        normalized_dois = self._dedupe_preserve_order(DocumentIdentifierType.DOI.normalize(doi) for doi in dois if doi)

        results: List[Dict[str, Any]] = []
        chunks = [normalized_dois[i : i + self.batch_size] for i in range(0, len(normalized_dois), self.batch_size)]

        for i, chunk in enumerate(chunks):
            records = self._fetch_chunk(chunk)
            results.extend(records)
            logger.info("Crossref DOI Fetcher: processed chunk %s/%s", i + 1, len(chunks))

        return results

    def _fetch_chunk(self, dois: List[str]) -> List[Dict[str, Any]]:
        """Fetch a DOI chunk using the Crossref works endpoint.

        A query such as ``filter=doi:10.x/a,doi:10.x/b`` returns metadata for
        those exact DOIs. Missing or non-Crossref DOIs simply do not appear in
        the response.
        """
        if not dois:
            return []

        params = {
            "filter": ",".join(f"doi:{doi}" for doi in dois),
            "rows": str(len(dois)),
        }
        if self.mailto:
            params["mailto"] = self.mailto

        url = f"{self.CROSSREF_WORKS_URL}?{urllib.parse.urlencode(params)}"
        data = self._get_json_with_retries(url)
        if not data:
            return []

        items = data.get("message", {}).get("items", [])
        records = []
        seen = set()

        for item in items:
            record = self._parse_work(item)
            doi = record["identifiers"][DocumentIdentifierType.DOI]
            if doi in dois and doi not in seen:
                records.append(record)
                seen.add(doi)

        # Fallback: if Crossref ever returns fewer records than expected because
        # of filter behavior or URL length, exact /works/{doi} lookup recovers
        # the missing records without abandoning batching for the normal case.
        missing = [doi for doi in dois if doi not in seen]
        if missing and len(dois) > 1:
            for doi in missing:
                item = self._fetch_one(doi)
                if item:
                    records.append(self._parse_work(item))

        return records

    def _fetch_one(self, doi: str) -> Optional[Dict[str, Any]]:
        encoded_doi = urllib.parse.quote(doi, safe="")
        params = {}
        if self.mailto:
            params["mailto"] = self.mailto
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{self.CROSSREF_WORKS_URL}/{encoded_doi}{query}"
        data = self._get_json_with_retries(url)
        if not data:
            return None
        return data.get("message")

    def _get_json_with_retries(self, url: str) -> Optional[Dict[str, Any]]:
        for attempt in range(self.MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            req = urllib.request.Request(url, headers=self._headers())

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    self._last_request = time.monotonic()
                    self._update_rate_limit_from_headers(response.headers)
                    return json.loads(response.read().decode("utf-8"))

            except urllib.error.HTTPError as e:
                self._last_request = time.monotonic()
                if e.code in (429, 503):
                    retry_after = self._parse_retry_after(e.headers.get("Retry-After"))
                    sleep_for = retry_after if retry_after is not None else self._backoff_seconds(attempt)
                    logger.warning(
                        "Crossref DOI Fetcher rate limited (%s); retrying after %.1fs", e.code, sleep_for
                    )
                    time.sleep(sleep_for)
                    continue

                if e.code == 404:
                    return None

                logger.error("Crossref DOI Fetcher HTTP Error %s: %s", e.code, e.reason)
                return None

            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                if attempt >= self.MAX_RETRIES:
                    logger.error("Crossref DOI Fetcher error: %s", e)
                    return None
                sleep_for = self._backoff_seconds(attempt)
                logger.warning("Crossref DOI Fetcher transient error: %s; retrying after %.1fs", e, sleep_for)
                time.sleep(sleep_for)

        return None

    def _headers(self) -> Dict[str, str]:
        user_agent = self.user_agent
        if self.mailto and "mailto:" not in user_agent:
            user_agent = f"{user_agent} (mailto:{self.mailto})"

        return {
            "Accept": "application/json",
            "User-Agent": user_agent,
        }

    def _wait_for_rate_limit(self):
        elapsed = time.monotonic() - self._last_request
        wait_seconds = max(self.wait_seconds, self._dynamic_wait_seconds)
        if elapsed < wait_seconds:
            time.sleep(wait_seconds - elapsed)

    def _update_rate_limit_from_headers(self, headers):
        limit = headers.get("X-Rate-Limit-Limit")
        interval = headers.get("X-Rate-Limit-Interval")

        if not limit or not interval:
            return

        try:
            limit_val = int(limit)
            interval_seconds = self._parse_interval_seconds(interval)
            if limit_val > 0 and interval_seconds is not None:
                # Add a small cushion so multiple local processes or network
                # jitter do not accidentally exceed the advertised window.
                self._dynamic_wait_seconds = (interval_seconds / limit_val) * 1.10
        except ValueError:
            return

    def _parse_interval_seconds(self, interval: str) -> Optional[float]:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", interval)
        if match:
            return float(match.group(1))
        return None

    def _parse_retry_after(self, retry_after: Optional[str]) -> Optional[float]:
        if not retry_after:
            return None
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            return None

    def _backoff_seconds(self, attempt: int) -> float:
        return min(60.0, (2**attempt) * self.wait_seconds + random.uniform(0.0, 0.25))

    def _parse_work(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        doi = DocumentIdentifierType.DOI.normalize(msg.get("DOI", ""))
        identifiers = {DocumentIdentifierType.DOI: doi}

        # TODO Unclear which key is used
        pmid = self._first_assertion_value(msg, "pubmed")
        if not pmid:
            pmid = self._first_assertion_value(msg, "pmid")
        if pmid:
            identifiers[DocumentIdentifierType.PMID] = DocumentIdentifierType.PMID.normalize(pmid)

        pmcid = self._first_assertion_value(msg, "pmcid")
        if pmcid:
            identifiers[DocumentIdentifierType.PMCID] = DocumentIdentifierType.PMCID.normalize(pmcid)

        journal = self._first(msg.get("short-container-title"))
        if not journal:
            logger.debug("Crossref DOI Fetcher falling back to full journal title")
            journal = self._first(msg.get("container-title"))

        return {
            "identifiers": identifiers,
            "journal": journal,
            "pub_year": self._issued_year(msg),
            "journal_metadata": self._parse_journal_metadata(msg),
        }

    def _issued_year(self, msg: Dict[str, Any]) -> Optional[str]:
        for key in ("published-print", "published-online", "published", "issued"):
            date_parts = msg.get(key, {}).get("date-parts")
            if date_parts and date_parts[0]:
                return str(date_parts[0][0])
        return None

    def _first(self, value: Any) -> Optional[Any]:
        if isinstance(value, list) and value:
            return value[0]
        return value

    def _parse_journal_metadata(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        full_title = self._first(msg.get("container-title"))
        abbreviation = self._first(msg.get("short-container-title"))
        variants = []
        for value in self._as_list(msg.get("container-title")) + self._as_list(msg.get("short-container-title")):
            if value not in (full_title, abbreviation):
                variants.append(value)

        identifiers: dict[str, Any] = {}
        issns = self._dedupe_preserve_order(msg.get("ISSN") or [])
        if issns:
            identifiers[ISSN] = issns

        return {
            "identifiers": identifiers,
            "name": full_title,
            "abbreviation": abbreviation,
            "name_variants": variants,
        }

    def _as_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _first_assertion_value(self, msg: Dict[str, Any], name: str) -> Optional[str]:
        for assertion in msg.get("assertion", []) or []:
            if str(assertion.get("name", "")).lower() == name:
                value = assertion.get("value")
                return str(value) if value else None
        return None

    def _dedupe_preserve_order(self, values) -> List[str]:
        seen = set()
        deduped = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped
