from __future__ import annotations

import logging
import re
from typing import Any
import xml.etree.ElementTree as ET

from corpus_benchmark.metadata.eutils_client import EUtilsClient
from corpus_benchmark.metadata.journal_fetcher import JournalMetadataFetcher
from corpus_benchmark.metadata.normalizers import normalize_issn

logger = logging.getLogger(__name__)

NLM_UNIQUE_ID = "NLMUNIQUEID"
ISSN = "ISSN"
ABBREVIATION = "ABBREVIATION"
NAME = "NAME"

CATALOG_FETCH_CHUNK_SIZE = 250
CATALOG_SEARCH_CHUNK_SIZE = 50
POST_ID_THRESHOLD = 200


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def _clean_title(value: str | None) -> str | None:
    value = _clean_text(value)
    if value and value.endswith("."):
        value = value[:-1].rstrip()
    return value or None


def _dedupe_preserve_order(values: list[str | None]) -> list[str]:
    seen = set()
    deduped: list[str] = []
    for value in values:
        value = _clean_text(value)
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _normalize_match_text(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value).strip().casefold()


def _format_issn_for_query(value: str) -> str:
    normalized = normalize_issn(value)
    return f"{normalized[:4]}-{normalized[4:]}"


def parse_nlm_catalog_record(element: ET.Element) -> dict[str, Any] | None:
    nlm_unique_id = _clean_text(element.findtext("./NlmUniqueID"))
    if not nlm_unique_id:
        return None

    name = _clean_title(element.findtext("./TitleMain/Title"))
    abbreviation = _clean_text(element.findtext("./MedlineTA"))

    issns: list[str | None] = []
    for issn_element in element.findall("./ISSN"):
        if issn_element.attrib.get("ValidYN", "Y") != "N":
            issns.append(_clean_text(issn_element.text))
    issns.extend(_clean_text(element.text) for element in element.findall("./ISSNLinking"))

    alternate_titles = [_clean_title(title_element.text) for title_element in element.findall("./TitleAlternate/Title")]
    variants = _dedupe_preserve_order(alternate_titles)
    canonical_values = {_normalize_match_text(value) for value in (name, abbreviation) if value}
    variants = [variant for variant in variants if _normalize_match_text(variant) not in canonical_values]

    mesh_topics = _dedupe_preserve_order([descriptor.text for descriptor in element.findall("./MeshHeadingList/MeshHeading/DescriptorName")])

    record = {
        "identifiers": {
            NLM_UNIQUE_ID: nlm_unique_id,
            ISSN: _dedupe_preserve_order(issns),
        },
        "name": name,
        "abbreviation": abbreviation,
        "name_variants": variants,
        "mesh_topics": mesh_topics,
    }
    return record


def parse_nlm_catalog_records(root: ET.Element) -> list[dict[str, Any]]:
    if root.tag == "NLMCatalogRecord":
        record_elements = [root]
    else:
        record_elements = root.findall("./NLMCatalogRecord")
    return [record for record in (parse_nlm_catalog_record(element) for element in record_elements) if record is not None]


class NlmCatalogNlmUniqueIDFetcher(JournalMetadataFetcher):
    """Resolve NLM catalog records directly by NLM Unique ID."""

    def __init__(self, client: EUtilsClient | None = None, **client_params: Any):
        self.client = client or EUtilsClient(**client_params)

    @property
    def supported_key(self) -> str:
        return NLM_UNIQUE_ID

    def fetch(self, identifiers: list[str]) -> list[dict[str, Any]]:
        ids = _dedupe_preserve_order(identifiers)
        if not ids:
            return []

        results: list[dict[str, Any]] = []
        chunks = [ids[i : i + CATALOG_FETCH_CHUNK_SIZE] for i in range(0, len(ids), CATALOG_FETCH_CHUNK_SIZE)]

        for i, chunk in enumerate(chunks):
            try:
                root = self._fetch_chunk(chunk)
                results.extend(parse_nlm_catalog_records(root))
                logger.info(
                    "NLM catalog NlmUniqueID fetcher: processed chunk %s/%s",
                    i + 1,
                    len(chunks),
                )
            except Exception as e:
                logger.error("NLM catalog NlmUniqueID fetcher error: %s", e)

        return results

    def _fetch_chunk(self, chunk: list[str]) -> ET.Element:
        params = {
            "db": "nlmcatalog",
            "id": ",".join(chunk),
            "retmode": "xml",
        }
        if len(chunk) > POST_ID_THRESHOLD:
            return self.client.post_xml("efetch", params)
        return self.client.get_xml("efetch", params)


class _NlmCatalogSearchFetcher(JournalMetadataFetcher):
    """Resolve journals by an NLM catalog search field, then efetch details."""

    search_field: str
    key: str

    def __init__(
        self,
        client: EUtilsClient | None = None,
        *,
        fetcher: NlmCatalogNlmUniqueIDFetcher | None = None,
        **client_params: Any,
    ):
        self.client = client or EUtilsClient(**client_params)
        self.fetcher = fetcher or NlmCatalogNlmUniqueIDFetcher(client=self.client)

    @property
    def supported_key(self) -> str:
        return self.key

    def fetch(self, identifiers: list[str]) -> list[dict[str, Any]]:
        values = _dedupe_preserve_order([self._prepare_value(value) for value in identifiers])
        if not values:
            return []

        nlm_ids: list[str] = []
        chunks = [values[i : i + CATALOG_SEARCH_CHUNK_SIZE] for i in range(0, len(values), CATALOG_SEARCH_CHUNK_SIZE)]
        for i, chunk in enumerate(chunks):
            try:
                root = self.client.get_xml(
                    "esearch",
                    {
                        "db": "nlmcatalog",
                        "retmode": "xml",
                        "retmax": str(max(20, len(chunk) * 10)),
                        "term": self._or_query(chunk),
                    },
                )
                nlm_ids.extend(id_element.text for id_element in root.findall("./IdList/Id") if id_element.text)
                logger.info(
                    "NLM catalog %s search fetcher: processed chunk %s/%s",
                    self.search_field,
                    i + 1,
                    len(chunks),
                )
            except Exception as e:
                logger.error("NLM catalog %s search fetcher error: %s", self.search_field, e)

        return self.fetcher.fetch(_dedupe_preserve_order(nlm_ids))

    def _prepare_value(self, value: str) -> str | None:
        return _clean_text(value)

    def _or_query(self, values: list[str]) -> str:
        clauses = []
        for value in values:
            escaped = value.replace('"', " ")
            clauses.append(f'("{escaped}"[{self.search_field}])')
        return " OR ".join(clauses)


class NlmCatalogISSNFetcher(_NlmCatalogSearchFetcher):
    search_field = "issn"
    key = ISSN

    def _prepare_value(self, value: str) -> str | None:
        try:
            return _format_issn_for_query(value)
        except ValueError:
            logger.debug("Skipping invalid ISSN search value: %r", value)
            return None


class NlmCatalogAbbreviationFetcher(_NlmCatalogSearchFetcher):
    search_field = "ta"
    key = ABBREVIATION


class NlmCatalogFullNameFetcher(_NlmCatalogSearchFetcher):
    search_field = "jo"
    key = NAME
