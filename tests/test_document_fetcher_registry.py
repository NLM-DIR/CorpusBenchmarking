from __future__ import annotations

from typing import Any

import pytest

from corpus_benchmark import registry
from corpus_benchmark.builtins import register_builtins
from corpus_benchmark.metadata.document_fetcher import DocumentMetadataFetcher
from corpus_benchmark.metadata.eutils_journal_fetchers import ABBREVIATION, ISSN, NLM_UNIQUE_ID
from corpus_benchmark.metadata.journal_fetcher import JournalMetadataFetcher
from corpus_benchmark.metadata.json_record_store import JsonRecordStore, normalize_issn, normalize_nlm_unique_id
from corpus_benchmark.models.config import BatteryConfig, LoaderSpec, MetricSpec, WorkspaceConfig
from corpus_benchmark.models.corpus import Document, DocumentIdentifierType
from corpus_benchmark.workspace import GlobalWorkspace


class EmptyPMIDFetcher(DocumentMetadataFetcher):
    @property
    def supported_id_type(self) -> DocumentIdentifierType:
        return DocumentIdentifierType.PMID

    def fetch(self, identifiers: list[str]) -> list[dict[str, Any]]:
        return []


class RaisingPMIDFetcher(DocumentMetadataFetcher):
    @property
    def supported_id_type(self) -> DocumentIdentifierType:
        return DocumentIdentifierType.PMID

    def fetch(self, identifiers: list[str]) -> list[dict[str, Any]]:
        raise RuntimeError("primary failed")


class FallbackPMIDFetcher(DocumentMetadataFetcher):
    @property
    def supported_id_type(self) -> DocumentIdentifierType:
        return DocumentIdentifierType.PMID

    def fetch(self, identifiers: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "identifiers": {DocumentIdentifierType.PMID: identifier},
                "journal": "Fallback Journal",
                "pub_year": "2025",
            }
            for identifier in identifiers
        ]


class DOIFetcher(DocumentMetadataFetcher):
    @property
    def supported_id_type(self) -> DocumentIdentifierType:
        return DocumentIdentifierType.DOI

    def fetch(self, identifiers: list[str]) -> list[dict[str, Any]]:
        return []


class JournalAwarePMIDFetcher(DocumentMetadataFetcher):
    @property
    def supported_id_type(self) -> DocumentIdentifierType:
        return DocumentIdentifierType.PMID

    def fetch(self, identifiers: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "identifiers": {DocumentIdentifierType.PMID: identifier},
                "journal": "Nat Immunol",
                "pub_year": "2013",
                "journal_metadata": {
                    "identifiers": {NLM_UNIQUE_ID: "100941354"},
                    "name": "Nature immunology",
                    "abbreviation": "Nat Immunol",
                },
            }
            for identifier in identifiers
        ]


class AbbreviationOnlyPMIDFetcher(DocumentMetadataFetcher):
    @property
    def supported_id_type(self) -> DocumentIdentifierType:
        return DocumentIdentifierType.PMID

    def fetch(self, identifiers: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "identifiers": {DocumentIdentifierType.PMID: identifier},
                "journal": "Nat Immunol",
                "pub_year": "2013",
                "journal_metadata": {
                    "identifiers": {},
                    "name": "Nature immunology",
                    "abbreviation": "Nat Immunol",
                },
            }
            for identifier in identifiers
        ]


class StaticJournalFetcher(JournalMetadataFetcher):
    def __init__(self, key: str):
        self.key = key
        self.calls: list[list[str]] = []

    @property
    def supported_key(self) -> str:
        return self.key

    def fetch(self, identifiers: list[str]) -> list[dict[str, Any]]:
        self.calls.append(list(identifiers))
        return [
            {
                "identifiers": {
                    NLM_UNIQUE_ID: "100941354",
                    ISSN: ["1529-2908", "1529-2916"],
                },
                "name": "Nature immunology",
                "abbreviation": "Nat Immunol",
                "name_variants": ["Nature Immun"],
                "mesh_topics": ["Immune System", "Immunity", "Immunotherapy"],
            }
        ]


def make_document_store(tmp_path):
    return JsonRecordStore(
        tmp_path / "metadata.json",
        identifier_types={
            DocumentIdentifierType.PMID,
            DocumentIdentifierType.PMCID,
            DocumentIdentifierType.DOI,
        },
        fields={"journal", "pub_year", "journal_id", "mesh_topics"},
        field_policies={
            "journal": "strict",
            "pub_year": "strict",
            "journal_id": "strict",
            "mesh_topics": "set_union",
        },
        identifier_normalizers={
            DocumentIdentifierType.PMID: DocumentIdentifierType.PMID.normalize,
            DocumentIdentifierType.PMCID: DocumentIdentifierType.PMCID.normalize,
            DocumentIdentifierType.DOI: DocumentIdentifierType.DOI.normalize,
        },
    )


def make_journal_store(tmp_path):
    return JsonRecordStore(
        tmp_path / "journals.json",
        identifier_types={
            NLM_UNIQUE_ID,
            ISSN,
        },
        fields={"name", "abbreviation", "name_variants", "mesh_topics"},
        field_policies={
            "name": "replace",
            "abbreviation": "replace",
            "name_variants": "set_union",
            "mesh_topics": "set_union",
        },
        identifier_normalizers={
            NLM_UNIQUE_ID: normalize_nlm_unique_id,
            ISSN: normalize_issn,
        },
    )


def test_register_builtins_populates_document_fetcher_registry() -> None:
    register_builtins()

    assert "pubmed_eutils" in registry.DOCUMENT_FETCHERS
    assert "pmc_eutils" in registry.DOCUMENT_FETCHERS
    assert "pmc_eutils_efetch" in registry.DOCUMENT_FETCHERS
    assert "crossref_doi" in registry.DOCUMENT_FETCHERS


def test_workspace_config_validate_rejects_unknown_fetcher() -> None:
    config = WorkspaceConfig(document_fetchers={"pmid": [LoaderSpec("missing_test_fetcher")]})

    with pytest.raises(ValueError, match="Unknown document fetcher"):
        config.validate()


def test_workspace_config_validate_rejects_identifier_type_mismatch(monkeypatch) -> None:
    monkeypatch.setitem(registry.DOCUMENT_FETCHERS, "test_doi_fetcher", DOIFetcher)
    config = WorkspaceConfig(document_fetchers={"pmid": [LoaderSpec("test_doi_fetcher")]})

    with pytest.raises(ValueError, match="supports 'doi'"):
        config.validate()


def test_battery_config_validate_rejects_unknown_metric() -> None:
    config = BatteryConfig(metrics=[MetricSpec(metric_name="missing_metric")])

    with pytest.raises(ValueError, match="Unknown metric"):
        config.validate()


def test_battery_config_validate_rejects_unknown_metric_bundle() -> None:
    config = BatteryConfig(
        metrics=[
            MetricSpec(
                metric_name="document_count",
                target_bundles=["missing_bundle"],
            )
        ]
    )

    with pytest.raises(ValueError, match="unknown bundle"):
        config.validate()


def test_workspace_uses_configured_fetcher_fallbacks(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(registry.DOCUMENT_FETCHERS, "test_empty_pmid", EmptyPMIDFetcher)
    monkeypatch.setitem(registry.DOCUMENT_FETCHERS, "test_fallback_pmid", FallbackPMIDFetcher)

    config = WorkspaceConfig(
        document_store_filename=str(tmp_path / "metadata.json"),
        corpora_download_dir=str(tmp_path / "corpora"),
        terminology_dir=str(tmp_path / "terminologies"),
        document_fetchers={
            "pmid": [
                LoaderSpec("test_empty_pmid"),
                LoaderSpec("test_fallback_pmid"),
            ]
        },
    )
    store = make_document_store(tmp_path)
    workspace = GlobalWorkspace(document_store=store, workspace_config=config)
    metadata = workspace.get_document_metadata(
        [
            Document(
                document_id="doc-1",
                identifiers={DocumentIdentifierType.PMID: "123"},
            )
        ]
    )

    assert metadata["doc-1"]["journal"] == "Fallback Journal"
    assert metadata["doc-1"]["pub_year"] == "2025"


def test_workspace_uses_fallback_when_primary_fetcher_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(registry.DOCUMENT_FETCHERS, "test_raising_pmid", RaisingPMIDFetcher)
    monkeypatch.setitem(registry.DOCUMENT_FETCHERS, "test_fallback_pmid", FallbackPMIDFetcher)

    config = WorkspaceConfig(
        document_store_filename=str(tmp_path / "metadata.json"),
        corpora_download_dir=str(tmp_path / "corpora"),
        terminology_dir=str(tmp_path / "terminologies"),
        document_fetchers={
            "pmid": [
                LoaderSpec("test_raising_pmid"),
                LoaderSpec("test_fallback_pmid"),
            ]
        },
    )
    store = make_document_store(tmp_path)
    workspace = GlobalWorkspace(document_store=store, workspace_config=config)
    metadata = workspace.get_document_metadata(
        [
            Document(
                document_id="doc-1",
                identifiers={DocumentIdentifierType.PMID: "123"},
            )
        ]
    )

    assert metadata["doc-1"]["journal"] == "Fallback Journal"


def test_workspace_creates_journal_record_and_links_document(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(registry.DOCUMENT_FETCHERS, "test_journal_aware_pmid", JournalAwarePMIDFetcher)

    config = WorkspaceConfig(
        document_store_filename=str(tmp_path / "metadata.json"),
        journal_store_filename=str(tmp_path / "journals.json"),
        corpora_download_dir=str(tmp_path / "corpora"),
        terminology_dir=str(tmp_path / "terminologies"),
        document_fetchers={"pmid": [LoaderSpec("test_journal_aware_pmid")]},
    )
    document_store = make_document_store(tmp_path)
    journal_store = make_journal_store(tmp_path)
    nlm_fetcher = StaticJournalFetcher(NLM_UNIQUE_ID)
    workspace = GlobalWorkspace(
        document_store=document_store,
        journal_store=journal_store,
        workspace_config=config,
    )
    workspace.journal_fetchers = {NLM_UNIQUE_ID: nlm_fetcher}

    metadata = workspace.get_document_metadata(
        [
            Document(
                document_id="doc-1",
                identifiers={DocumentIdentifierType.PMID: "123"},
            )
        ]
    )

    journal_record = journal_store.get(NLM_UNIQUE_ID, "100941354")
    assert journal_record is not None
    assert metadata["doc-1"]["journal_id"] == journal_record.record_id
    assert document_store.get(DocumentIdentifierType.PMID, "123").data["journal_id"] == journal_record.record_id
    assert journal_record.identifiers[ISSN] == ["15292908", "15292916"]
    assert journal_record.data["mesh_topics"] == ["Immune System", "Immunity", "Immunotherapy"]
    assert nlm_fetcher.calls == [["100941354"]]


def test_workspace_adds_new_journal_identifiers_to_text_matched_record(tmp_path) -> None:
    config = WorkspaceConfig(
        document_store_filename=str(tmp_path / "metadata.json"),
        journal_store_filename=str(tmp_path / "journals.json"),
        corpora_download_dir=str(tmp_path / "corpora"),
        terminology_dir=str(tmp_path / "terminologies"),
    )
    document_store = make_document_store(tmp_path)
    journal_store = make_journal_store(tmp_path)
    workspace = GlobalWorkspace(
        document_store=document_store,
        journal_store=journal_store,
        workspace_config=config,
    )
    existing = journal_store.upsert(
        identifiers={NLM_UNIQUE_ID: "100941354"},
        data={
            "name": "Nature immunology",
            "abbreviation": "Nat Immunol",
        },
    )

    journal_record = workspace._upsert_journal_metadata(
        {
            "identifiers": {ISSN: ["1529-2908", "1529-2916"]},
            "name": "Nature immunology",
            "abbreviation": "Nat Immunol",
        }
    )

    assert journal_record is not None
    assert journal_record.record_id == existing.record_id
    assert journal_store.count() == 1
    assert journal_store.get(ISSN, "1529-2908").record_id == existing.record_id
    assert journal_store.get(NLM_UNIQUE_ID, "100941354").identifiers[ISSN] == [
        "15292908",
        "15292916",
    ]


def test_workspace_resolves_journal_by_abbreviation_when_document_has_no_journal_id(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(registry.DOCUMENT_FETCHERS, "test_abbreviation_only_pmid", AbbreviationOnlyPMIDFetcher)

    config = WorkspaceConfig(
        document_store_filename=str(tmp_path / "metadata.json"),
        journal_store_filename=str(tmp_path / "journals.json"),
        corpora_download_dir=str(tmp_path / "corpora"),
        terminology_dir=str(tmp_path / "terminologies"),
        document_fetchers={"pmid": [LoaderSpec("test_abbreviation_only_pmid")]},
    )
    document_store = make_document_store(tmp_path)
    journal_store = make_journal_store(tmp_path)
    abbreviation_fetcher = StaticJournalFetcher(ABBREVIATION)
    workspace = GlobalWorkspace(
        document_store=document_store,
        journal_store=journal_store,
        workspace_config=config,
    )
    workspace.journal_fetchers = {ABBREVIATION: abbreviation_fetcher}

    metadata = workspace.get_document_metadata(
        [
            Document(
                document_id="doc-1",
                identifiers={DocumentIdentifierType.PMID: "123"},
            )
        ]
    )

    journal_record = journal_store.get(NLM_UNIQUE_ID, "100941354")
    assert journal_record is not None
    assert metadata["doc-1"]["journal_id"] == journal_record.record_id
    assert abbreviation_fetcher.calls == [["Nat Immunol"]]
