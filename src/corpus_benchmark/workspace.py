from __future__ import annotations
import inspect
import logging
import re
from typing import Any, Dict


from corpus_benchmark.builtins import register_builtins
from corpus_benchmark.acquisition import AcquisitionManager
from corpus_benchmark.metadata.document_fetcher import DocumentMetadataFetcher
from corpus_benchmark.metadata.eutils_client import EUtilsClient
from corpus_benchmark.metadata.eutils_journal_fetchers import (
    ABBREVIATION,
    ISSN,
    NAME,
    NLM_UNIQUE_ID,
    NlmCatalogAbbreviationFetcher,
    NlmCatalogFullNameFetcher,
    NlmCatalogISSNFetcher,
    NlmCatalogNlmUniqueIDFetcher,
)
from corpus_benchmark.metadata.journal_fetcher import JournalMetadataFetcher
from corpus_benchmark.metadata.json_record_store import JsonRecordStore, StoredRecord
from corpus_benchmark.models.config import LoaderSpec, WorkspaceConfig
from corpus_benchmark.models.corpus import Document, DocumentIdentifierType
from corpus_benchmark.models.terminologies import TerminologyResource
from corpus_benchmark.registry import DOCUMENT_FETCHERS

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
        journal_store: JsonRecordStore | None = None,
    ):
        self.document_store = document_store
        self.journal_store = journal_store
        self.workspace_config = workspace_config
        self.acquisition_manager = AcquisitionManager(workspace_config)
        self.terminologies = {}
        self.fetchers = self._build_document_fetchers(workspace_config.document_fetchers)
        self.journal_fetchers = self._build_journal_fetchers()
        self._ambiguous_journal_match_warnings: set[tuple[int, ...]] = set()
        self._journal_catalog_refresh_attempted: set[str] = set()

    def _build_document_fetchers(self, configured_fetchers: dict[str, list[LoaderSpec]]) -> dict[DocumentIdentifierType, list[DocumentMetadataFetcher]]:
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

    def _build_journal_fetchers(self) -> dict[str, JournalMetadataFetcher]:
        eutils_client = EUtilsClient()
        nlm_unique_id_fetcher = NlmCatalogNlmUniqueIDFetcher(client=eutils_client)
        return {
            NLM_UNIQUE_ID: nlm_unique_id_fetcher,
            ISSN: NlmCatalogISSNFetcher(
                client=eutils_client,
                fetcher=nlm_unique_id_fetcher,
            ),
            ABBREVIATION: NlmCatalogAbbreviationFetcher(
                client=eutils_client,
                fetcher=nlm_unique_id_fetcher,
            ),
            NAME: NlmCatalogFullNameFetcher(
                client=eutils_client,
                fetcher=nlm_unique_id_fetcher,
            ),
        }

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
            journal_record = self.get_journal_metadata_by_id(metadata["journal_id"])
            if journal_record:
                metadata["journal"] = journal_record.get("abbreviation")

        return metadata

    def _get_document_record(self, id_type: DocumentIdentifierType, id_val: str) -> Dict[str, Any] | None:
        record = self.document_store.get(id_type, id_val)
        if record is None:
            return None
        return self._format_stored_record(record)

    def _get_document_stored_record(self, id_type: DocumentIdentifierType, id_val: str) -> StoredRecord | None:
        return self.document_store.get(id_type, id_val)

    def _find_existing_document_record_for_identifiers(self, identifiers: dict[Any, Any]) -> StoredRecord | None:
        matching_record_ids: set[int] = set()
        for id_type, values in identifiers.items():
            for value in self._as_list(values):
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
            return self._reconcile_document_pub_year_for_existing_record(existing, data)

        incoming_journal = self._clean_text(data.get("journal"))
        existing_journal = self._clean_text(existing.data.get("journal"))
        if not incoming_journal or not existing_journal:
            return self._reconcile_document_pub_year_for_existing_record(existing, data)

        if self._normalize_journal_match_text(incoming_journal) == self._normalize_journal_match_text(existing_journal):
            reconciled = dict(data)
            reconciled["journal"] = existing.data["journal"]
            return self._reconcile_document_pub_year_for_existing_record(existing, reconciled)

        return self._reconcile_document_pub_year_for_existing_record(existing, data)

    def _reconcile_document_pub_year_for_existing_record(
        self,
        existing: StoredRecord | None,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if existing is None or "pub_year" not in data or "pub_year" not in existing.data:
            return data

        incoming_year = self._clean_text(data.get("pub_year"))
        existing_year = self._clean_text(existing.data.get("pub_year"))
        if incoming_year and existing_year and incoming_year != existing_year:
            reconciled = dict(data)
            reconciled["pub_year"] = existing.data["pub_year"]
            return reconciled

        return data

    def _add_document_records(self, new_records: list[Dict[str, Any]]) -> None:
        updated = 0
        for new_record in new_records:
            identifiers = new_record.get("identifiers", {})
            journal_metadata = new_record.get("journal_metadata")
            data = {key: value for key, value in new_record.items() if key not in {"identifiers", "journal_metadata"}}
            data = self._reconcile_document_data_for_existing_record(identifiers, data)
            document_record = self.document_store.upsert(identifiers=identifiers, data=data)
            journal_record = self._upsert_journal_metadata(journal_metadata)
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
            if self.journal_store is not None:
                self.journal_store.save()

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, set):
            return sorted(value)
        return [value]

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        if value is None:
            return None
        value = re.sub(r"\s+", " ", str(value)).strip()
        return value or None

    @classmethod
    def _dedupe_strings(cls, values: list[Any]) -> list[str]:
        seen = set()
        deduped: list[str] = []
        for value in values:
            value = cls._clean_text(value)
            if value and value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped

    @staticmethod
    def _canonical_journal_identifier_type(value: Any) -> str:
        return str(value).strip().upper()

    def _journal_identifiers_from_metadata(self, journal_metadata: dict[str, Any]) -> dict[str, Any]:
        raw_identifiers = journal_metadata.get("identifiers", {}) or {}
        identifiers: dict[str, Any] = {}
        for raw_key, raw_values in raw_identifiers.items():
            key = self._canonical_journal_identifier_type(raw_key)
            if key not in {NLM_UNIQUE_ID, ISSN}:
                continue
            values = self._dedupe_strings(self._as_list(raw_values))
            if not values:
                continue
            identifiers[key] = values[0] if key == NLM_UNIQUE_ID and len(values) == 1 else values
        return identifiers

    def _journal_data_from_metadata(
        self,
        journal_metadata: dict[str, Any],
        *,
        keep_empty_lists: bool = False,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {}

        name = self._clean_text(journal_metadata.get("name") or journal_metadata.get("full_name"))
        abbreviation = self._clean_text(journal_metadata.get("abbreviation"))
        if name is not None:
            data["name"] = name
        if abbreviation is not None:
            data["abbreviation"] = abbreviation

        for key in ("name_variants", "mesh_topics"):
            if key not in journal_metadata:
                continue
            values = self._dedupe_strings(self._as_list(journal_metadata.get(key)))
            if values or keep_empty_lists:
                data[key] = values

        return data

    def _upsert_journal_metadata(self, journal_metadata: Any) -> StoredRecord | None:
        if self.journal_store is None or not isinstance(journal_metadata, dict):
            return None

        identifiers = self._journal_identifiers_from_metadata(journal_metadata)
        if len(identifiers) == 0:
            raise ValueError("Journal must have known identifiers")
        existing = self._find_journal_record_for_identifiers(journal_metadata)
        data = self._journal_data_from_metadata(journal_metadata)

        if identifiers:
            if existing is not None:
                merged_identifiers: dict[str, list[Any]] = {key: list(values) for key, values in existing.identifiers.items()}
                for key, values in identifiers.items():
                    merged_identifiers.setdefault(key, []).extend(self._as_list(values))
                return self.journal_store.upsert(identifiers=merged_identifiers, data=data)
            return self.journal_store.upsert(identifiers=identifiers, data=data)
        return existing

    def _add_journal_records(self, journal_records: list[dict[str, Any]]) -> None:
        if self.journal_store is None:
            return

        updated = 0
        for journal_record in journal_records:
            identifiers = self._journal_identifiers_from_metadata(journal_record)
            if not identifiers:
                continue
            data = self._journal_data_from_metadata(journal_record, keep_empty_lists=True)
            self.journal_store.upsert(identifiers=identifiers, data=data)
            updated += 1
        if updated > 0:
            logger.info("Updated %s journal metadata records", updated)
            self.journal_store.save()

    def _attach_known_document_identifiers(self, documents: list[Document]) -> None:
        for doc in documents:
            if not doc.identifiers:
                continue
            if any(self.document_store.get(id_type, id_val) is not None for id_type, id_val in doc.identifiers.items()):
                self.document_store.upsert(identifiers=doc.identifiers)

    @staticmethod
    def _normalize_journal_match_text(value: str) -> str:
        return re.sub(r"[\W_]+", " ", value).strip().casefold()

    def _journal_text_values_from_metadata(self, journal_metadata: dict[str, Any]) -> list[str]:
        values = [
            journal_metadata.get("name"),
            journal_metadata.get("full_name"),
            journal_metadata.get("abbreviation"),
        ]
        values.extend(self._as_list(journal_metadata.get("name_variants")))
        return self._dedupe_strings(values)

    def _journal_text_values_from_record(self, journal_record: StoredRecord) -> list[str]:
        data = journal_record.data
        values = [data.get("name"), data.get("abbreviation")]
        values.extend(self._as_list(data.get("name_variants")))
        return self._dedupe_strings(values)

    def _find_journal_record_for_identifiers(self, journal_metadata: dict[str, Any]) -> StoredRecord | None:
        if self.journal_store is None:
            return None

        matching_record_ids: set[int] = set()
        identifiers = self._journal_identifiers_from_metadata(journal_metadata)
        for id_type, values in identifiers.items():
            for value in self._as_list(values):
                try:
                    record = self.journal_store.get(id_type, value)
                except ValueError:
                    continue
                if record is not None:
                    matching_record_ids.add(record.record_id)

        if len(matching_record_ids) == 1:
            return self.journal_store.get_by_record_id(next(iter(matching_record_ids)))
        if len(matching_record_ids) > 1:
            warning_key = tuple(sorted(matching_record_ids))
            if warning_key not in self._ambiguous_journal_match_warnings:
                self._ambiguous_journal_match_warnings.add(warning_key)
                # TODO Figure out why this error goes away if we rerun it
                logger.warning(
                    "Journal metadata matched multiple journal records: %s",
                    sorted(matching_record_ids),
                )
                logger.warning(f"TEST Ambiguous journal by ID; journal_metadata = {journal_metadata}")
                logger.warning(f"TEST Ambiguous journal by ID; identifiers = {identifiers}")
                logger.warning(f"TEST Ambiguous journal by ID; matching_record_ids = {matching_record_ids}")
        return None

    def _find_journal_record_for_info(self, journal_metadata: dict[str, Any]) -> StoredRecord | None:
        if self.journal_store is None:
            return None

        matching_record_ids: set[int] = set()
        identifiers = self._journal_identifiers_from_metadata(journal_metadata)
        for id_type, values in identifiers.items():
            for value in self._as_list(values):
                try:
                    record = self.journal_store.get(id_type, value)
                except ValueError:
                    continue
                if record is not None:
                    matching_record_ids.add(record.record_id)

        target_texts = None
        if len(matching_record_ids) < 1:
            # TESTING Only try the journal text if we don't already have the record
            # FIXME : this is causing ambiguous errors AND slowing us down
            # TODO after pulling identifiers, if we have it, return it
            # TODO ONLY after recognizing we don't have the identifiers do we check names
            # TODO OR... what if we require an identifier for the journal?
            target_texts = {self._normalize_journal_match_text(value) for value in self._journal_text_values_from_metadata(journal_metadata)}
            if target_texts:
                for record in self.journal_store:
                    record_texts = {self._normalize_journal_match_text(value) for value in self._journal_text_values_from_record(record)}
                    if target_texts & record_texts:
                        matching_record_ids.add(record.record_id)
            if len(matching_record_ids) > 0:
                logger.warning("TEST journal found by text; journal_metadata = {journal_metadata}")

        if len(matching_record_ids) == 1:
            return self.journal_store.get_by_record_id(next(iter(matching_record_ids)))
        if len(matching_record_ids) > 1:
            warning_key = tuple(sorted(matching_record_ids))
            if warning_key not in self._ambiguous_journal_match_warnings:
                self._ambiguous_journal_match_warnings.add(warning_key)
                # TODO Figure out why this error goes away if we rerun it
                logger.warning(
                    "Journal metadata matched multiple journal records: %s",
                    sorted(matching_record_ids),
                )
                logger.warning(f"TEST Ambiguous journal by info; journal_metadata = {journal_metadata}")
                logger.warning(f"TEST Ambiguous journal by info; identifiers = {identifiers}")
                logger.warning(f"TEST Ambiguous journal by info; target_texts = {target_texts}")
                logger.warning(f"TEST Ambiguous journal by info; matching_record_ids = {matching_record_ids}")
        return None

    def _journal_record_id_exists(self, record_id: Any) -> bool:
        if self.journal_store is None or record_id is None:
            return False
        try:
            self.journal_store.get_by_record_id(int(record_id))
            return True
        except (KeyError, TypeError, ValueError):
            return False

    def get_journal_metadata_by_id(self, record_id: Any) -> dict[str, Any] | None:
        if self.journal_store is None or record_id is None:
            return None
        try:
            record = self.journal_store.get_by_record_id(int(record_id))
        except (KeyError, TypeError, ValueError):
            return None
        return self._format_stored_record(record)

    def _journal_info_from_document_data(self, data: dict[str, Any]) -> dict[str, Any] | None:
        journal = self._clean_text(data.get("journal"))
        if not journal or journal == "Unknown":
            return None
        return {
            "name": journal,
            "abbreviation": journal,
            "name_variants": [journal],
        }

    def _document_journal_link_data(
        self,
        document_record: StoredRecord,
        journal_record: StoredRecord,
    ) -> dict[str, Any]:
        """Once a journal is linked, we set the journal name to None to avoid conflicts."""
        return {"journal_id": journal_record.record_id, "journal": None}

    def _attach_journal_ids_from_store(self, document_records: list[StoredRecord]) -> None:
        if self.journal_store is None:
            return

        for document_record in document_records:
            current_record = self.document_store.get_by_record_id(document_record.record_id)
            if self._journal_record_id_exists(current_record.data.get("journal_id")):
                continue

            journal_info = self._journal_info_from_document_data(current_record.data)
            if journal_info is None:
                continue

            journal_record = self._find_journal_record_for_info(journal_info)
            if journal_record is None:
                continue

            document_data = self._document_journal_link_data(current_record, journal_record)
            self.document_store.upsert(
                identifiers=current_record.identifiers,
                data=document_data,
            )

    def _journal_record_needs_catalog_refresh(self, journal_record: StoredRecord) -> bool:
        return "mesh_topics" not in journal_record.data

    def _fetch_and_store_journals(self, key: str, values: list[Any]) -> None:
        if self.journal_store is None:
            return
        fetcher = self.journal_fetchers.get(key)
        values = self._dedupe_strings(values)
        if fetcher is None or not values:
            return

        try:
            fetched_records = fetcher.fetch(values)
        except Exception as e:
            logger.warning(
                "Journal fetcher %s failed for %s values of type %s: %s",
                type(fetcher).__name__,
                len(values),
                key,
                e,
            )
            return
        self._add_journal_records(fetched_records)

    def _refresh_incomplete_journal_records(self) -> None:
        if self.journal_store is None:
            return

        nlm_unique_ids: list[str] = []
        for journal_record in self.journal_store:
            if self._journal_record_needs_catalog_refresh(journal_record):
                nlm_unique_ids.extend(
                    nlm_unique_id for nlm_unique_id in journal_record.identifiers.get(NLM_UNIQUE_ID, []) if nlm_unique_id not in self._journal_catalog_refresh_attempted
                )
        self._journal_catalog_refresh_attempted.update(nlm_unique_ids)
        self._fetch_and_store_journals(NLM_UNIQUE_ID, nlm_unique_ids)

    def _unresolved_journal_values_by_key(
        self,
        journal_infos: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        values_by_key = {
            NLM_UNIQUE_ID: [],
            ISSN: [],
            ABBREVIATION: [],
            NAME: [],
        }
        for journal_info in journal_infos:
            identifiers = self._journal_identifiers_from_metadata(journal_info)
            values_by_key[NLM_UNIQUE_ID].extend(self._as_list(identifiers.get(NLM_UNIQUE_ID)))
            values_by_key[ISSN].extend(self._as_list(identifiers.get(ISSN)))
            values_by_key[ABBREVIATION].append(journal_info.get("abbreviation"))
            values_by_key[NAME].append(journal_info.get("name") or journal_info.get("full_name"))

        return values_by_key

    def _filter_unresolved_journal_values(self, key: str, values: list[Any]) -> list[Any]:
        unresolved: list[Any] = []
        for value in self._dedupe_strings(values):
            if key in {NLM_UNIQUE_ID, ISSN}:
                if self.journal_store is not None:
                    try:
                        if self.journal_store.get(key, value) is not None:
                            continue
                    except ValueError:
                        continue
            else:
                field = "abbreviation" if key == ABBREVIATION else "name"
                if self._find_journal_record_for_info({field: value}) is not None:
                    continue
            unresolved.append(value)
        return unresolved

    def _resolve_journal_infos(self, journal_infos: list[dict[str, Any]]) -> None:
        if self.journal_store is None or not journal_infos:
            return

        for key, values in self._unresolved_journal_values_by_key(journal_infos).items():
            unresolved_values = self._filter_unresolved_journal_values(key, values)
            self._fetch_and_store_journals(key, unresolved_values)

    def _resolve_journals_for_document_records(self, document_records: list[StoredRecord]) -> None:
        if self.journal_store is None or not document_records:
            return

        self._refresh_incomplete_journal_records()
        self._attach_journal_ids_from_store(document_records)

        unresolved_infos: list[dict[str, Any]] = []
        for document_record in document_records:
            current_record = self.document_store.get_by_record_id(document_record.record_id)
            if self._journal_record_id_exists(current_record.data.get("journal_id")):
                continue
            journal_info = self._journal_info_from_document_data(current_record.data)
            if journal_info is not None:
                unresolved_infos.append(journal_info)

        self._resolve_journal_infos(unresolved_infos)
        self._refresh_incomplete_journal_records()
        self._attach_journal_ids_from_store(document_records)
        self.document_store.save()
        self.journal_store.save()

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
                record = self._get_document_stored_record(id_type, id_val)
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
