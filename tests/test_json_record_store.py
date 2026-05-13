from __future__ import annotations

import json

import pytest

from corpus_benchmark.metadata.json_record_store import (
    IdentifierConflictError,
    JsonRecordStore,
    MetadataConflictError,
    StoreFormatError,
    UnknownFieldError,
)
from corpus_benchmark.models.corpus import DocumentIdentifierType


def make_document_store(tmp_path, *, autoload: bool = True) -> JsonRecordStore:
    return JsonRecordStore(
        tmp_path / "records.json",
        identifier_types={
            DocumentIdentifierType.PMID,
            DocumentIdentifierType.PMCID,
            DocumentIdentifierType.DOI,
        },
        fields={"journal", "pub_year", "mesh_topics"},
        field_policies={"mesh_topics": "set_union"},
        identifier_normalizers={
            DocumentIdentifierType.PMID: DocumentIdentifierType.PMID.normalize,
            DocumentIdentifierType.PMCID: DocumentIdentifierType.PMCID.normalize,
            DocumentIdentifierType.DOI: DocumentIdentifierType.DOI.normalize,
        },
        autoload=autoload,
    )


def test_upsert_merges_new_identifiers_for_same_record(tmp_path) -> None:
    store = make_document_store(tmp_path)

    first = store.upsert(
        identifiers={
            DocumentIdentifierType.PMID: "123",
            DocumentIdentifierType.DOI: "https://doi.org/10.1000/example",
        },
        data={"journal": "Example Journal", "pub_year": "2024"},
    )

    second = store.upsert(
        identifiers={
            DocumentIdentifierType.PMID: "123",
            DocumentIdentifierType.PMCID: "456",
        },
        data={"mesh_topics": ["A", "B"]},
    )

    assert second.record_id == first.record_id
    assert store.count() == 1

    record = store.get(DocumentIdentifierType.PMCID, "PMC456")
    assert record is not None
    assert record.identifiers == {
        "DOI": ["10.1000/example"],
        "PMCID": ["PMC456"],
        "PMID": ["123"],
    }
    assert record.data == {
        "journal": "Example Journal",
        "mesh_topics": ["A", "B"],
        "pub_year": "2024",
    }


def test_upsert_rejects_identifier_bridge_between_two_records(tmp_path) -> None:
    store = make_document_store(tmp_path)
    store.upsert(
        identifiers={DocumentIdentifierType.PMID: "123"},
        data={"journal": "Journal A"},
    )
    store.upsert(
        identifiers={DocumentIdentifierType.DOI: "10.1000/b"},
        data={"journal": "Journal B"},
    )

    with pytest.raises(IdentifierConflictError):
        store.upsert(
            identifiers={
                DocumentIdentifierType.PMID: "123",
                DocumentIdentifierType.DOI: "10.1000/b",
            }
        )


def test_upsert_rejects_unknown_fields_and_strict_conflicts(tmp_path) -> None:
    store = make_document_store(tmp_path)

    with pytest.raises(UnknownFieldError):
        store.upsert(
            identifiers={DocumentIdentifierType.PMID: "123"},
            data={"title": "Unexpected"},
        )

    store.upsert(
        identifiers={DocumentIdentifierType.PMID: "123"},
        data={"journal": "Journal A"},
    )

    with pytest.raises(MetadataConflictError):
        store.upsert(
            identifiers={DocumentIdentifierType.PMID: "123"},
            data={"journal": "Journal B"},
        )


def test_save_and_autoload_round_trip_records(tmp_path) -> None:
    store = make_document_store(tmp_path)
    record = store.upsert(
        identifiers={
            DocumentIdentifierType.PMID: "123",
            DocumentIdentifierType.PMCID: "456",
        },
        data={"journal": "Example Journal"},
    )

    assert store.dirty is True
    store.save()
    assert store.dirty is False

    reloaded = make_document_store(tmp_path)
    assert reloaded.count() == 1

    by_pmcid = reloaded.get(DocumentIdentifierType.PMCID, "456")
    assert by_pmcid is not None
    assert by_pmcid.record_id == record.record_id
    assert by_pmcid.identifiers == {
        "PMCID": ["PMC456"],
        "PMID": ["123"],
    }
    assert by_pmcid.data == {"journal": "Example Journal"}

    assert (
        reloaded.upsert(
            identifiers={DocumentIdentifierType.DOI: "10.1000/example"},
            data={"pub_year": "2024"},
        ).record_id
        == 2
    )


def test_new_empty_skips_autoload_and_can_overwrite_existing_file(tmp_path) -> None:
    original = make_document_store(tmp_path)
    original.upsert(
        identifiers={DocumentIdentifierType.PMID: "123"},
        data={"journal": "Original Journal"},
    )
    original.save()

    replacement = JsonRecordStore.new_empty(
        tmp_path / "records.json",
        identifier_types={
            DocumentIdentifierType.PMID,
            DocumentIdentifierType.PMCID,
            DocumentIdentifierType.DOI,
        },
        fields={"journal", "pub_year", "mesh_topics"},
        field_policies={"mesh_topics": "set_union"},
        identifier_normalizers={
            DocumentIdentifierType.PMID: DocumentIdentifierType.PMID.normalize,
            DocumentIdentifierType.PMCID: DocumentIdentifierType.PMCID.normalize,
            DocumentIdentifierType.DOI: DocumentIdentifierType.DOI.normalize,
        },
    )
    assert replacement.count() == 0

    replacement.upsert(
        identifiers={DocumentIdentifierType.PMID: "999"},
        data={"journal": "Replacement Journal"},
    )
    replacement.save()

    reloaded = make_document_store(tmp_path)
    assert reloaded.get(DocumentIdentifierType.PMID, "123") is None
    assert reloaded.get(DocumentIdentifierType.PMID, "999") is not None


def test_load_rejects_invalid_json_store_format(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "records": [
                    {
                        "record_id": 1,
                        "identifiers": {"PMID": ["123"]},
                        "data": {"title": "Unexpected"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnknownFieldError):
        make_document_store(tmp_path)

    path.write_text(json.dumps({"format_version": 99, "records": []}), encoding="utf-8")

    with pytest.raises(StoreFormatError):
        make_document_store(tmp_path)
