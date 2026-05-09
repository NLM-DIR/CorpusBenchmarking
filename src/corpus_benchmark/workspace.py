from __future__ import annotations
import logging
import inspect
from typing import Any, Dict


from corpus_benchmark.acquisition import AcquisitionManager
from corpus_benchmark.builtins import register_builtins
from corpus_benchmark.metadata.document_fetcher import DocumentMetadataFetcher
from corpus_benchmark.metadata.eutils_client import EUtilsClient
from corpus_benchmark.metadata.json_record_store import JsonRecordStore, StoredRecord
from corpus_benchmark.models.config import LoaderSpec, WorkspaceConfig
from corpus_benchmark.models.corpus import Document, DocumentIdentifierType
from corpus_benchmark.models.terminologies import TerminologyResource
from corpus_benchmark.metadata.journal_metadata import (
    JournalRecordStore,
    normalize_journal_match_text,
    as_list,
    journal_info_from_document_data,
)
from corpus_benchmark.registry import DOCUMENT_FETCHERS
from utils.text_utils import clean_text

logger = logging.getLogger(__name__)


class GlobalWorkspace:
    """Manages persistent, cross-run resources like caches and downloaded files."""

    document_store: JsonRecordStore
    acquisition_manager: AcquisitionManager
    workspace_config: WorkspaceConfig
    terminologies: dict[str, TerminologyResource]

    def __init__(
        self,
        document_store: JsonRecordStore,
        workspace_config: WorkspaceConfig,
        journal_record_store: JournalRecordStore | None = None,
    ):
        self.document_store = document_store
        self.workspace_config = workspace_config
        self.acquisition_manager = AcquisitionManager(workspace_config)
        self.terminologies = {}
        self.fetchers = build_document_fetchers(workspace_config.document_fetchers)
        self.journal_record_store = journal_record_store

    def get_document_metadata(self, documents: list[Document]) -> Dict[str, Dict[str, Any]]:
        # TODO REfactor this so that it runs one Fetcher at a time, resolving all of the documents it can
        self._attach_known_document_identifiers(documents)
        missing_ids = {id_type: set() for id_type in self.fetchers.keys()}
        # 1. Check store
        for doc in documents:
            for id_type, id_val in doc.identifiers.items():
                record = self._get_document_record(id_type, id_val)
                # print(f"Metadata for {id_type}:{id_val} returned {record}")
                if not record and id_type in self.fetchers:
                    missing_ids[id_type].add(id_val)

        for id_type, missing_ids_by_type in missing_ids.items():
            if len(missing_ids_by_type) > 0:
                logger.info("Fetching %s IDs of type %s", len(missing_ids_by_type), id_type)

        # 2. Fetch missing items using configured primary/fallback fetchers and add new records to the store
        for id_type, id_set in missing_ids.items():
            remaining_ids = set(id_set)
            for fetcher in self.fetchers[id_type]:
                if not remaining_ids:
                    break
                try:
                    fetched_records = fetcher.fetch(list(remaining_ids))
                except Exception as e:
                    logger.warning(
                        "Document fetcher %s failed for %s IDs of type %s: %s",
                        type(fetcher).__name__,
                        len(remaining_ids),
                        id_type,
                        e,
                    )
                    continue
                self._add_document_records(fetched_records)
                remaining_ids = {id_val for id_val in remaining_ids if self._get_document_record(id_type, id_val) is None}
            if remaining_ids:
                logger.warning(
                    "Could not fetch metadata for %s %s IDs using configured fetchers",
                    len(remaining_ids),
                    id_type,
                )
                logger.debug(f"IDs without metadata: {remaining_ids}")

        # 3. Resolve journals for the retrieved document metadata.
        document_records_by_doc_id: dict[str, StoredRecord | None] = {}
        for doc in documents:
            record = None
            for id_type, id_val in doc.identifiers.items():
                record = self.document_store.get(id_type, id_val)
                # print(f"Metadata for {id_type}:{id_val} returned {record}")
                if record:
                    break  # Found it!
            document_records_by_doc_id[doc.document_id] = record

        self._resolve_journals_for_document_records([record for record in document_records_by_doc_id.values() if record is not None])

        # 4. Get metadata for realzies
        doc_metadata = {}
        for doc_id, record in document_records_by_doc_id.items():
            if record is None:
                doc_metadata[doc_id] = {}
            else:
                latest_record = self.document_store.get_by_record_id(record.record_id)
                doc_metadata[doc_id] = self._format_stored_record(latest_record)

        logger.debug("Resolved metadata for %s documents", len(doc_metadata))
        return doc_metadata

    def _attach_known_document_identifiers(self, documents: list[Document]) -> None:
        for doc in documents:
            if not doc.identifiers:
                continue
            if any(self.document_store.get(id_type, id_val) is not None for id_type, id_val in doc.identifiers.items()):
                self.document_store.upsert(identifiers=doc.identifiers)

    def _get_document_record(self, id_type: DocumentIdentifierType, id_val: str) -> Dict[str, Any] | None:
        record = self.document_store.get(id_type, id_val)
        if record is None:
            return None
        return self._format_stored_record(record)

    def _format_stored_record(self, record: StoredRecord) -> Dict[str, Any]:
        metadata = dict(record.data)
        identifiers: dict[DocumentIdentifierType | str, str | list[str]] = {}
        for raw_id_type, values in record.identifiers.items():
            try:
                id_type: DocumentIdentifierType | str = DocumentIdentifierType(raw_id_type.lower())
            except ValueError:
                id_type = raw_id_type.lower()
            identifiers[id_type] = values[0] if len(values) == 1 else values
        metadata["identifiers"] = identifiers

        # Populate journal name from journal store if missing in document record
        if metadata.get("journal") is None and metadata.get("journal_id") is not None:
            journal_record = self.journal_record_store.get_journal_metadata_by_id(metadata["journal_id"])
            if journal_record:
                metadata["journal"] = journal_record.get("abbreviation")

        return metadata

    def _add_document_records(self, new_records: list[Dict[str, Any]]) -> None:
        updated = 0
        for new_record in new_records:
            identifiers = new_record.get("identifiers", {})
            journal_metadata = new_record.get("journal_metadata")
            data = {key: value for key, value in new_record.items() if key not in {"identifiers", "journal_metadata"}}
            data = self._reconcile_document_data_for_existing_record(identifiers, data)
            document_record = self.document_store.upsert(identifiers=identifiers, data=data)
            journal_record = self.journal_record_store.upsert_journal_metadata(journal_metadata)
            if journal_record is not None:
                document_data = self._document_journal_link_data(document_record, journal_record)
                document_record = self.document_store.upsert(
                    identifiers=document_record.identifiers,
                    data=document_data,
                )
            updated += 1
        if updated > 0:
            logger.info("Updated %s metadata records", updated)
            self.document_store.save()
            if self.journal_record_store is not None:
                self.journal_record_store.save()

    def _resolve_journals_for_document_records(self, document_records: list[StoredRecord]) -> None:
        if self.journal_record_store is None or not document_records:
            return

        self.journal_record_store.refresh_incomplete_journal_records()
        self._attach_journal_ids_from_store(document_records)

        unresolved_infos: list[dict[str, Any]] = []
        for document_record in document_records:
            current_record = self.document_store.get_by_record_id(document_record.record_id)
            if self.journal_record_store.journal_record_id_exists(current_record.data.get("journal_id")):
                continue
            journal_info = journal_info_from_document_data(current_record.data)
            if journal_info is not None:
                unresolved_infos.append(journal_info)

        self.journal_record_store.resolve_journal_infos(unresolved_infos)
        self.journal_record_store.refresh_incomplete_journal_records()
        self._attach_journal_ids_from_store(document_records)
        self.document_store.save()
        self.journal_record_store.save()

    def _find_existing_document_record_for_identifiers(self, identifiers: dict[Any, Any]) -> StoredRecord | None:
        matching_record_ids: set[int] = set()
        for id_type, values in identifiers.items():
            for value in as_list(values):
                try:
                    record = self.document_store.get(id_type, value)
                except ValueError:
                    continue
                if record is not None:
                    matching_record_ids.add(record.record_id)
        if len(matching_record_ids) == 1:
            return self.document_store.get_by_record_id(next(iter(matching_record_ids)))
        return None

    def _reconcile_document_data_for_existing_record(
        self,
        identifiers: dict[Any, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        existing = self._find_existing_document_record_for_identifiers(identifiers)
        if existing is None or "journal" not in data or "journal" not in existing.data:
            return reconcile_document_pub_year_for_existing_record(existing, data)

        incoming_journal = clean_text(data.get("journal"))
        existing_journal = clean_text(existing.data.get("journal"))
        if not incoming_journal or not existing_journal:
            return reconcile_document_pub_year_for_existing_record(existing, data)

        if normalize_journal_match_text(incoming_journal) == normalize_journal_match_text(existing_journal):
            reconciled = dict(data)
            reconciled["journal"] = existing.data["journal"]
            return reconcile_document_pub_year_for_existing_record(existing, reconciled)

        return reconcile_document_pub_year_for_existing_record(existing, data)

    def _document_journal_link_data(
        self,
        document_record: StoredRecord,
        journal_record: StoredRecord,
    ) -> dict[str, Any]:
        """Once a journal is linked, we set the journal name to None to avoid conflicts."""
        return {"journal_id": journal_record.record_id, "journal": None}

    def _attach_journal_ids_from_store(self, document_records: list[StoredRecord]) -> None:
        if self.journal_record_store is None:
            return

        for document_record in document_records:
            current_record = self.document_store.get_by_record_id(document_record.record_id)
            if self.journal_record_store.journal_record_id_exists(current_record.data.get("journal_id")):
                continue

            journal_info = journal_info_from_document_data(current_record.data)
            if journal_info is None:
                continue

            journal_record = self.journal_record_store.find_journal_record_for_info(journal_info)
            if journal_record is None:
                continue

            document_data = self._document_journal_link_data(current_record, journal_record)
            self.document_store.upsert(
                identifiers=current_record.identifiers,
                data=document_data,
            )


def build_document_fetchers(configured_fetchers: dict[str, list[LoaderSpec]]) -> dict[DocumentIdentifierType, list[DocumentMetadataFetcher]]:
    register_builtins()
    eutils_client = EUtilsClient()
    fetchers: dict[DocumentIdentifierType, list[DocumentMetadataFetcher]] = {}

    for raw_id_type, fetcher_specs in configured_fetchers.items():
        raw_id_type_value = getattr(raw_id_type, "value", str(raw_id_type)).lower()
        id_type = DocumentIdentifierType(raw_id_type_value)
        fetchers[id_type] = []
        for fetcher_spec in fetcher_specs:
            if fetcher_spec.name not in DOCUMENT_FETCHERS:
                available = ", ".join(sorted(DOCUMENT_FETCHERS)) or "<none>"
                raise ValueError(f"Unknown document fetcher '{fetcher_spec.name}' for {id_type}. " f"Available document fetchers: {available}")

            fetcher_cls = DOCUMENT_FETCHERS[fetcher_spec.name]
            params = dict(fetcher_spec.params)
            if "client" in inspect.signature(fetcher_cls).parameters and "client" not in params and not params:
                params["client"] = eutils_client
            fetcher = fetcher_cls(**params)

            if fetcher.supported_id_type != id_type:
                raise ValueError(f"Document fetcher '{fetcher_spec.name}' supports " f"{fetcher.supported_id_type}, but it was configured for {id_type}.")
            fetchers[id_type].append(fetcher)

    return fetchers


def reconcile_document_pub_year_for_existing_record(
    existing: StoredRecord | None,
    data: dict[str, Any],
) -> dict[str, Any]:
    if existing is None or "pub_year" not in data or "pub_year" not in existing.data:
        return data

    incoming_year = clean_text(data.get("pub_year"))
    existing_year = clean_text(existing.data.get("pub_year"))
    if incoming_year and existing_year and incoming_year != existing_year:
        reconciled = dict(data)
        reconciled["pub_year"] = existing.data["pub_year"]
        return reconciled

    return data
