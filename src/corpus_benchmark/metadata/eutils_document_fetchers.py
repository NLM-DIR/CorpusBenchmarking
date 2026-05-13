import logging
import random
import re
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

from corpus_benchmark.models.corpus import DocumentIdentifierType
from corpus_benchmark.metadata.document_fetcher import DocumentMetadataFetcher
from corpus_benchmark.metadata.eutils_client import EUtilsClient
from corpus_benchmark.metadata.eutils_journal_fetchers import ISSN, NLM_UNIQUE_ID
from corpus_benchmark.registry import register_document_fetcher

logger = logging.getLogger(__name__)

CHUNK_SIZE = 250
POST_ID_THRESHOLD = 200


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", value).strip()
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


def _get_xml_for_id_chunk(
    client: EUtilsClient,
    endpoint: str,
    params: dict[str, str],
    id_count: int,
) -> ET.Element:
    if id_count > POST_ID_THRESHOLD:
        return client.post_xml(endpoint, params)
    return client.get_xml(endpoint, params)


@register_document_fetcher("pubmed_eutils")
class PubMedFetcher(DocumentMetadataFetcher):
    """Queries NCBI eUtils for metadata using PubMed IDs."""

    def __init__(self, client: EUtilsClient | None = None, **client_params: Any):
        self.client = client or EUtilsClient(**client_params)

    @property
    def supported_id_type(self) -> DocumentIdentifierType:
        return DocumentIdentifierType.PMID

    def fetch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []

        results = []
        random.shuffle(pmids)
        chunks = [pmids[i : i + CHUNK_SIZE] for i in range(0, len(pmids), CHUNK_SIZE)]

        for i, chunk in enumerate(chunks):
            try:
                root = _get_xml_for_id_chunk(
                    self.client,
                    "efetch",
                    {
                        "db": "pubmed",
                        "id": ",".join(chunk),
                        "retmode": "xml",
                    },
                    len(chunk),
                )
                for article in root.findall("./PubmedArticle"):
                    results.append(self._parse_article(article))
                logger.info("PubMed Fetcher: processed chunk %s/%s", i + 1, len(chunks))
            except Exception as e:
                logger.error("PubMed Fetcher error: %s", e)
        return results

    def _parse_article(self, element: ET.Element) -> Dict[str, Any]:
        pmid = element.findtext("./MedlineCitation/PMID")
        pmc = element.findtext("./PubmedData/ArticleIdList/ArticleId[@IdType='pmc']")
        doi = element.findtext("./PubmedData/ArticleIdList/ArticleId[@IdType='doi']")

        journal = element.findtext("./MedlineCitation/Article/Journal/ISOAbbreviation")
        if not journal:
            journal = element.findtext("./MedlineCitation/Article/Journal/Title")
        journal = _clean_text(journal)

        year = element.findtext(
            "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year"
        )
        if not year:
            medline_date = element.findtext(
                "./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate"
            )
            year = medline_date[:4] if medline_date else None

        pmid = DocumentIdentifierType.PMID.normalize(pmid)
        identifiers = {DocumentIdentifierType.PMID: pmid}
        if not pmc is None:
            identifiers[DocumentIdentifierType.PMCID] = (
                DocumentIdentifierType.PMCID.normalize(pmc)
            )
        if not doi is None:
            identifiers[DocumentIdentifierType.DOI] = (
                DocumentIdentifierType.DOI.normalize(doi)
            )
        record = {
            "identifiers": identifiers,
            "journal": journal,
            "pub_year": year,
            "mesh_topics": self._parse_mesh_headings(element),
            "journal_metadata": self._parse_journal_metadata(element),
        }
        # print(f"PubMedFetcher._parse_article() returning: {record}")
        return record

    def _parse_mesh_headings(self, element: ET.Element) -> list[str]:
        return _dedupe_preserve_order(
            [
                descriptor.text
                for descriptor in element.findall(
                    "./MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName"
                )
            ]
        )

    def _parse_journal_metadata(self, element: ET.Element) -> Dict[str, Any]:
        article_journal = element.find("./MedlineCitation/Article/Journal")
        medline_journal_info = element.find("./MedlineCitation/MedlineJournalInfo")

        issns: list[str | None] = []
        if article_journal is not None:
            issns.append(article_journal.findtext("./ISSN"))
        if medline_journal_info is not None:
            issns.append(medline_journal_info.findtext("./ISSNLinking"))

        nlm_unique_id = (
            medline_journal_info.findtext("./NlmUniqueID")
            if medline_journal_info is not None
            else None
        )
        medline_ta = (
            medline_journal_info.findtext("./MedlineTA")
            if medline_journal_info is not None
            else None
        )
        title = article_journal.findtext("./Title") if article_journal is not None else None
        iso_abbreviation = (
            article_journal.findtext("./ISOAbbreviation")
            if article_journal is not None
            else None
        )

        identifiers: dict[str, Any] = {}
        if nlm_unique_id:
            identifiers[NLM_UNIQUE_ID] = nlm_unique_id
        normalized_issns = _dedupe_preserve_order(issns)
        if normalized_issns:
            identifiers[ISSN] = normalized_issns

        variants = _dedupe_preserve_order([medline_ta])
        if iso_abbreviation and medline_ta and iso_abbreviation == medline_ta:
            variants = []

        return {
            "identifiers": identifiers,
            "name": _clean_text(title),
            "abbreviation": _clean_text(iso_abbreviation or medline_ta),
            "name_variants": variants,
        }


@register_document_fetcher("pmc_eutils_efetch")
class PMCEFetchFetcher(DocumentMetadataFetcher):
    """Queries NCBI eUtils efetch for metadata using PMC IDs."""

    def __init__(self, client: EUtilsClient | None = None, **client_params: Any):
        self.client = client or EUtilsClient(**client_params)

    @property
    def supported_id_type(self) -> DocumentIdentifierType:
        return DocumentIdentifierType.PMCID

    def fetch(self, pmcids: List[str]) -> List[Dict[str, Any]]:
        if not pmcids:
            return []

        pmcids = [DocumentIdentifierType.PMCID.normalize(pmcid) for pmcid in pmcids]
        numeric_ids = [pmcid[3:] for pmcid in pmcids]
        results: list[dict[str, Any]] = []
        chunks = [
            numeric_ids[i : i + CHUNK_SIZE]
            for i in range(0, len(numeric_ids), CHUNK_SIZE)
        ]

        for i, chunk in enumerate(chunks):
            try:
                root = _get_xml_for_id_chunk(
                    self.client,
                    "efetch",
                    {
                        "db": "pmc",
                        "id": ",".join(chunk),
                        "retmode": "xml",
                    },
                    len(chunk),
                )
                articles = [root] if root.tag == "article" else root.findall("./article")
                for article in articles:
                    results.append(self._parse_article(article))
                logger.info("PMC efetch Fetcher: processed chunk %s/%s", i + 1, len(chunks))
            except Exception as e:
                logger.error("PMC efetch Fetcher error: %s", e)
        return results

    def _parse_article(self, element: ET.Element) -> Dict[str, Any]:
        article_meta = element.find("./front/article-meta")
        journal_meta = element.find("./front/journal-meta")

        pmcid = self._article_id(article_meta, "pmcid")
        pmid = self._article_id(article_meta, "pmid")
        doi = self._article_id(article_meta, "doi")

        identifiers = {DocumentIdentifierType.PMCID: DocumentIdentifierType.PMCID.normalize(pmcid)}
        if pmid:
            identifiers[DocumentIdentifierType.PMID] = DocumentIdentifierType.PMID.normalize(pmid)
        if doi:
            identifiers[DocumentIdentifierType.DOI] = DocumentIdentifierType.DOI.normalize(doi)

        abbreviation = self._journal_id(journal_meta, "nlm-ta")
        iso_abbreviation = self._journal_id(journal_meta, "iso-abbrev")
        name = _clean_text(journal_meta.findtext("./journal-title-group/journal-title") if journal_meta is not None else None)
        journal = _clean_text(abbreviation or iso_abbreviation or name)

        record = {
            "identifiers": identifiers,
            "journal": journal,
            "pub_year": self._publication_year(article_meta),
            "journal_metadata": {
                "identifiers": self._journal_identifiers(journal_meta),
                "name": name,
                "abbreviation": _clean_text(abbreviation or iso_abbreviation),
                "name_variants": _dedupe_preserve_order([iso_abbreviation]),
            },
        }
        return record

    def _article_id(self, article_meta: ET.Element | None, pub_id_type: str) -> str | None:
        if article_meta is None:
            return None
        return _clean_text(
            article_meta.findtext(f"./article-id[@pub-id-type='{pub_id_type}']")
        )

    def _journal_id(self, journal_meta: ET.Element | None, journal_id_type: str) -> str | None:
        if journal_meta is None:
            return None
        return _clean_text(
            journal_meta.findtext(f"./journal-id[@journal-id-type='{journal_id_type}']")
        )

    def _journal_identifiers(self, journal_meta: ET.Element | None) -> dict[str, Any]:
        if journal_meta is None:
            return {}
        issns = _dedupe_preserve_order(
            [issn_element.text for issn_element in journal_meta.findall("./issn")]
        )
        return {ISSN: issns} if issns else {}

    def _publication_year(self, article_meta: ET.Element | None) -> str | None:
        if article_meta is None:
            return None
        for pub_type in ("ppub", "epub", "collection"):
            year = article_meta.findtext(f"./pub-date[@pub-type='{pub_type}']/year")
            if year:
                return _clean_text(year)
        return _clean_text(article_meta.findtext("./pub-date/year"))
