from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from corpus_benchmark.metadata.eutils_document_fetchers import PMCEFetchFetcher, PubMedFetcher
from corpus_benchmark.metadata.eutils_journal_fetchers import (
    ISSN,
    NLM_UNIQUE_ID,
    parse_nlm_catalog_records,
)
from corpus_benchmark.models.corpus import DocumentIdentifierType

EXAMPLES = Path(__file__).resolve().parents[1] / "tests/data"


def test_pubmed_efetch_parser_returns_article_mesh_and_journal_metadata() -> None:
    root = ET.parse(EXAMPLES / "efetch_pubmed.xml").getroot()
    article = root.find("./PubmedArticle")
    assert article is not None

    record = PubMedFetcher()._parse_article(article)

    assert record["identifiers"][DocumentIdentifierType.PMID] == "23525089"
    assert record["identifiers"][DocumentIdentifierType.PMCID] == "PMC3631456"
    assert record["journal"] == "Nat Immunol"
    assert "Mitophagy" in record["mesh_topics"]
    assert "Orthomyxoviridae Infections" in record["mesh_topics"]

    journal_metadata = record["journal_metadata"]
    assert journal_metadata["identifiers"][NLM_UNIQUE_ID] == "100941354"
    assert journal_metadata["identifiers"][ISSN] == ["1529-2916", "1529-2908"]
    assert journal_metadata["name"] == "Nature immunology"
    assert journal_metadata["abbreviation"] == "Nat Immunol"


def test_pmc_efetch_parser_returns_issns_for_journal_resolution() -> None:
    root = ET.parse(EXAMPLES / "efetch_pmc.xml").getroot()
    article = root.find("./article")
    assert article is not None

    record = PMCEFetchFetcher()._parse_article(article)

    assert record["identifiers"][DocumentIdentifierType.PMCID] == "PMC3631456"
    assert record["identifiers"][DocumentIdentifierType.PMID] == "23525089"
    assert record["identifiers"][DocumentIdentifierType.DOI] == "10.1038/ni.2563"
    assert record["journal"] == "Nat Immunol"
    assert record["pub_year"] == "2013"

    journal_metadata = record["journal_metadata"]
    assert journal_metadata["identifiers"][ISSN] == ["1529-2908", "1529-2916"]
    assert journal_metadata["name"] == "Nature immunology"
    assert journal_metadata["abbreviation"] == "Nat Immunol"
    assert journal_metadata["name_variants"] == ["Nat. Immunol."]


def test_nlm_catalog_parser_returns_full_journal_records() -> None:
    root = ET.parse(EXAMPLES / "efetch_nlmcatalog_multi.xml").getroot()

    records = parse_nlm_catalog_records(root)

    assert len(records) == 2
    nature_immunology = records[0]
    assert nature_immunology["identifiers"][NLM_UNIQUE_ID] == "100941354"
    assert nature_immunology["identifiers"][ISSN] == ["1529-2908", "1529-2916"]
    assert nature_immunology["name"] == "Nature immunology"
    assert nature_immunology["abbreviation"] == "Nat Immunol"
    assert nature_immunology["name_variants"] == ["Nature Immun"]
    assert nature_immunology["mesh_topics"] == [
        "Immune System",
        "Immunity",
        "Immunotherapy",
    ]

    nature_communications = records[1]
    assert nature_communications["identifiers"][NLM_UNIQUE_ID] == "101528555"
    assert nature_communications["identifiers"][ISSN] == ["2041-1723"]
    assert nature_communications["mesh_topics"] == ["Biological Science Disciplines"]
