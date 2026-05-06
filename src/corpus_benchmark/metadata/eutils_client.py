import random
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import logging

from corpus_benchmark.models.config import APP_NAME_VER

logger = logging.getLogger(__name__)

CHUNK_SIZE = 250
WAIT_SECONDS = 0.4
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class EUtilsClient:
    """Small NCBI E-utilities client with shared rate limiting."""

    _last_request_by_identity: dict[tuple[str | None, str | None], float] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        tool: str = APP_NAME_VER,
        email: str | None = None,
        wait_seconds: float = WAIT_SECONDS,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.tool = tool
        self.email = email
        self.wait_seconds = wait_seconds
        self.timeout = timeout
        self.max_retries = 3

    def get_xml(self, endpoint: str, params: dict[str, str]) -> ET.Element:
        return self._request_xml(endpoint, params, method="GET")

    def post_xml(self, endpoint: str, params: dict[str, str]) -> ET.Element:
        return self._request_xml(endpoint, params, method="POST")

    def _request_xml(self, endpoint: str, params: dict[str, str], *, method: str) -> ET.Element:
        query_params = dict(params)
        query_params["tool"] = self.tool
        if self.email:
            query_params["email"] = self.email
        if self.api_key:
            query_params["api_key"] = self.api_key

        base_url = f"{EUTILS_BASE_URL}/{endpoint}.fcgi"
        if method == "POST":
            url = base_url
            data = urllib.parse.urlencode(query_params).encode("utf-8")
            request = urllib.request.Request(url, data=data, method="POST")
        else:
            url = f"{base_url}?{urllib.parse.urlencode(query_params)}"
            request = urllib.request.Request(url, method="GET")

        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    self._mark_request()
                    return ET.fromstring(response.read())
            except urllib.error.HTTPError as e:
                self._mark_request()
                if e.code not in (429, 500, 502, 503, 504) or attempt >= self.max_retries:
                    raise
                retry_after = self._parse_retry_after(e.headers.get("Retry-After"))
                sleep_for = retry_after if retry_after is not None else self._backoff_seconds(attempt)
                logger.warning("NCBI E-utilities HTTP %s; retrying after %.1fs", e.code, sleep_for)
                time.sleep(sleep_for)
            except (urllib.error.URLError, TimeoutError) as e:
                self._mark_request()
                if attempt >= self.max_retries:
                    raise
                sleep_for = self._backoff_seconds(attempt)
                logger.warning(
                    "NCBI E-utilities transient error: %s; retrying after %.1fs",
                    e,
                    sleep_for,
                )
                time.sleep(sleep_for)

        raise RuntimeError("Unreachable E-utilities retry state")

    def _identity(self) -> tuple[str | None, str | None]:
        return (self.api_key, self.email)

    def _wait_for_rate_limit(self) -> None:
        identity = self._identity()
        last_request = self._last_request_by_identity.get(identity, 0.0)
        elapsed = time.monotonic() - last_request
        if elapsed < self.wait_seconds:
            time.sleep(self.wait_seconds - elapsed)

    def _mark_request(self) -> None:
        self._last_request_by_identity[self._identity()] = time.monotonic()

    def _parse_retry_after(self, retry_after: str | None) -> float | None:
        if not retry_after:
            return None
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            return None

    def _backoff_seconds(self, attempt: int) -> float:
        return min(60.0, (2**attempt) * self.wait_seconds + random.uniform(0.0, 0.25))
