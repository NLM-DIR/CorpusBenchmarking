import logging
import re
from typing import Any, Dict
from pathlib import Path

from corpus_benchmark.metadata.json_record_store import JsonRecordStore, StoredRecord
from corpus_benchmark.metadata.journal_fetcher import JournalMetadataFetcher
from corpus_benchmark.metadata.normalizers import normalize_issn, normalize_nlm_unique_id
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
from corpus_benchmark.metadata.eutils_client import EUtilsClient
from utils.text_utils import clean_text, dedupe_strings

logger = logging.getLogger(__name__)


class JournalRecordStore:

    def __init__(
        self,
        journal_store: JsonRecordStore | None = None,
    ):
        self.journal_store = journal_store
        self.journal_fetchers = _build_journal_fetchers()
        self._ambiguous_journal_match_warnings: set[tuple[int, ...]] = set()
        self._journal_catalog_refresh_attempted: set[str] = set()

    def get_journal_metadata_by_id(self, record_id: Any) -> dict[str, Any] | None:
        if self.journal_store is None or record_id is None:
            return None
        try:
            record = self.journal_store.get_by_record_id(int(record_id))
        except (KeyError, TypeError, ValueError):
            return None
        return self._format_stored_record(record)

    def _format_stored_record(self, record: StoredRecord) -> Dict[str, Any]:
        metadata = dict(record.data)
        identifiers: dict[str, str | list[str]] = {}
        for raw_id_type, values in record.identifiers.items():
            id_type = raw_id_type.lower()
            identifiers[id_type] = values[0] if len(values) == 1 else values
        metadata["identifiers"] = identifiers
        return metadata

    def upsert_journal_metadata(self, journal_metadata: Any) -> StoredRecord | None:
        if self.journal_store is None or not isinstance(journal_metadata, dict):
            return None

        identifiers = _journal_identifiers_from_metadata(journal_metadata)
        if len(identifiers) == 0:
            raise ValueError("Journal must have known identifiers")
        existing = self._find_journal_record_for_identifiers(journal_metadata)

        if not identifiers:
            return existing

        data = _journal_data_from_metadata(journal_metadata)
        if existing is None:
            return self.journal_store.upsert(identifiers=identifiers, data=data)

        merged_identifiers: dict[str, list[Any]] = {key: list(values) for key, values in existing.identifiers.items()}
        for key, values in identifiers.items():
            merged_identifiers.setdefault(key, []).extend(as_list(values))
        return self.journal_store.upsert(identifiers=merged_identifiers, data=data)

    def _add_journal_records(self, journal_records: list[dict[str, Any]]) -> None:
        if self.journal_store is None:
            return
        updated = 0
        for journal_record in journal_records:
            identifiers = _journal_identifiers_from_metadata(journal_record)
            if not identifiers:
                continue
            data = _journal_data_from_metadata(journal_record, keep_empty_lists=True)
            self.journal_store.upsert(identifiers=identifiers, data=data)
            updated += 1
        if updated > 0:
            logger.info("Updated %s journal metadata records", updated)
            self.journal_store.save()

    def _find_journal_record_for_identifiers(self, journal_metadata: dict[str, Any]) -> StoredRecord | None:
        if self.journal_store is None:
            return None
        matching_record_ids: set[int] = set()
        identifiers = _journal_identifiers_from_metadata(journal_metadata)
        for id_type, values in identifiers.items():
            for value in as_list(values):
                try:
                    record = self.journal_store.get(id_type, value)
                except ValueError:
                    continue
                if record is not None:
                    matching_record_ids.add(record.record_id)

        if len(matching_record_ids) == 1:
            return self.journal_store.get_by_record_id(next(iter(matching_record_ids)))
        if len(matching_record_ids) > 1:
            # TODO Verify this should never happen and raise an error instead
            warning_key = tuple(sorted(matching_record_ids))
            if warning_key not in self._ambiguous_journal_match_warnings:
                self._ambiguous_journal_match_warnings.add(warning_key)
                logger.warning(
                    "Journal metadata matched multiple journal records: %s",
                    sorted(matching_record_ids),
                )
        return None

    def find_journal_record_for_info(self, journal_metadata: dict[str, Any]) -> StoredRecord | None:
        if self.journal_store is None:
            return None
        matching_record_ids: set[int] = set()
        identifiers = _journal_identifiers_from_metadata(journal_metadata)
        for id_type, values in identifiers.items():
            for value in as_list(values):
                try:
                    record = self.journal_store.get(id_type, value)
                except ValueError:
                    continue
                if record is not None:
                    matching_record_ids.add(record.record_id)

        target_texts = None
        if len(matching_record_ids) < 1:
            # Only try the journal text if we don't already have the record
            target_texts = {normalize_journal_match_text(value) for value in _journal_text_values_from_metadata(journal_metadata)}
            if target_texts:
                for record in self.journal_store:
                    record_texts = {normalize_journal_match_text(value) for value in _journal_text_values_from_record(record)}
                    if target_texts & record_texts:
                        matching_record_ids.add(record.record_id)
            if len(matching_record_ids) > 0:
                # TODO If querying journal by text is not needed, remove it
                logger.info("Journal found by text; journal_metadata = {journal_metadata}")

        if len(matching_record_ids) == 1:
            return self.journal_store.get_by_record_id(next(iter(matching_record_ids)))
        if len(matching_record_ids) > 1:
            warning_key = tuple(sorted(matching_record_ids))
            if warning_key not in self._ambiguous_journal_match_warnings:
                self._ambiguous_journal_match_warnings.add(warning_key)
                logger.warning(
                    "Journal metadata matched multiple journal records: %s",
                    sorted(matching_record_ids),
                )
        return None

    def _fetch_and_store_journals(self, key: str, values: list[Any]) -> None:
        if self.journal_store is None:
            return
        fetcher = self.journal_fetchers.get(key)
        values = dedupe_strings(values)
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

    def refresh_incomplete_journal_records(self) -> None:
        if self.journal_store is None:
            return

        nlm_unique_ids: list[str] = []
        for journal_record in self.journal_store:
            if _journal_record_needs_catalog_refresh(journal_record):
                nlm_unique_ids.extend(
                    nlm_unique_id for nlm_unique_id in journal_record.identifiers.get(NLM_UNIQUE_ID, []) if nlm_unique_id not in self._journal_catalog_refresh_attempted
                )
        self._journal_catalog_refresh_attempted.update(nlm_unique_ids)
        self._fetch_and_store_journals(NLM_UNIQUE_ID, nlm_unique_ids)

    def resolve_journal_infos(self, journal_infos: list[dict[str, Any]]) -> None:
        if self.journal_store is None or not journal_infos:
            return

        for key, values in _unresolved_journal_values_by_key(journal_infos).items():
            unresolved_values = self._filter_unresolved_journal_values(key, values)
            self._fetch_and_store_journals(key, unresolved_values)

    def _filter_unresolved_journal_values(self, key: str, values: list[Any]) -> list[Any]:
        unresolved: list[Any] = []
        for value in dedupe_strings(values):
            if key in {NLM_UNIQUE_ID, ISSN}:
                if self.journal_store is not None:
                    try:
                        if self.journal_store.get(key, value) is not None:
                            continue
                    except ValueError:
                        continue
            else:
                field = "abbreviation" if key == ABBREVIATION else "name"
                if self.find_journal_record_for_info({field: value}) is not None:
                    continue
            unresolved.append(value)
        return unresolved

    def journal_record_id_exists(self, record_id: Any) -> bool:
        if self.journal_store is None or record_id is None:
            return False
        try:
            self.journal_store.get_by_record_id(int(record_id))
            return True
        except (KeyError, TypeError, ValueError):
            return False

    def save(self) -> None:
        self.journal_store.save()


def create_journal_record_store(journal_store_filename: str) -> JournalRecordStore:
    journal_store_path = Path(journal_store_filename)
    journal_store_path.parent.mkdir(parents=True, exist_ok=True)
    journal_store = JsonRecordStore(
        journal_store_path,
        identifier_types={
            "NLMUNIQUEID",
            "ISSN",
        },
        fields={
            "name",
            "abbreviation",
            "name_variants",
            "mesh_topics",
        },
        field_policies={
            "name": "replace",
            "abbreviation": "replace",
            "name_variants": "set_union",
            "mesh_topics": "set_union",
        },
        identifier_normalizers={
            "NLMUNIQUEID": normalize_nlm_unique_id,
            "ISSN": normalize_issn,
        },
    )
    return JournalRecordStore(journal_store)


def _build_journal_fetchers() -> dict[str, JournalMetadataFetcher]:
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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def _canonical_journal_identifier_type(value: Any) -> str:
    return str(value).strip().upper()


def normalize_journal_match_text(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value).strip().casefold()


def _journal_identifiers_from_metadata(journal_metadata: dict[str, Any]) -> dict[str, Any]:
    raw_identifiers = journal_metadata.get("identifiers", {}) or {}
    identifiers: dict[str, Any] = {}
    for raw_key, raw_values in raw_identifiers.items():
        key = _canonical_journal_identifier_type(raw_key)
        if key not in {NLM_UNIQUE_ID, ISSN}:
            continue
        values = dedupe_strings(as_list(raw_values))
        if not values:
            continue
        identifiers[key] = values[0] if key == NLM_UNIQUE_ID and len(values) == 1 else values
    return identifiers


def _journal_data_from_metadata(
    journal_metadata: dict[str, Any],
    *,
    keep_empty_lists: bool = False,
) -> dict[str, Any]:
    data: dict[str, Any] = {}

    name = clean_text(journal_metadata.get("name") or journal_metadata.get("full_name"))
    abbreviation = clean_text(journal_metadata.get("abbreviation"))
    if name is not None:
        data["name"] = name
    if abbreviation is not None:
        data["abbreviation"] = abbreviation

    for key in ("name_variants", "mesh_topics"):
        if key not in journal_metadata:
            continue
        values = dedupe_strings(as_list(journal_metadata.get(key)))
        if values or keep_empty_lists:
            data[key] = values

    return data


def _journal_text_values_from_metadata(journal_metadata: dict[str, Any]) -> list[str]:
    values = [
        journal_metadata.get("name"),
        journal_metadata.get("full_name"),
        journal_metadata.get("abbreviation"),
    ]
    values.extend(as_list(journal_metadata.get("name_variants")))
    return dedupe_strings(values)


def _journal_text_values_from_record(journal_record: StoredRecord) -> list[str]:
    data = journal_record.data
    values = [data.get("name"), data.get("abbreviation")]
    values.extend(as_list(data.get("name_variants")))
    return dedupe_strings(values)


def _journal_record_needs_catalog_refresh(journal_record: StoredRecord) -> bool:
    return "mesh_topics" not in journal_record.data


def _unresolved_journal_values_by_key(
    journal_infos: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    values_by_key = {
        NLM_UNIQUE_ID: [],
        ISSN: [],
        ABBREVIATION: [],
        NAME: [],
    }
    for journal_info in journal_infos:
        identifiers = _journal_identifiers_from_metadata(journal_info)
        values_by_key[NLM_UNIQUE_ID].extend(as_list(identifiers.get(NLM_UNIQUE_ID)))
        values_by_key[ISSN].extend(as_list(identifiers.get(ISSN)))
        values_by_key[ABBREVIATION].append(journal_info.get("abbreviation"))
        values_by_key[NAME].append(journal_info.get("name") or journal_info.get("full_name"))

    return values_by_key


def journal_info_from_document_data(data: dict[str, Any]) -> dict[str, Any] | None:
    journal = clean_text(data.get("journal"))
    if not journal or journal == "Unknown":
        return None
    return {
        "name": journal,
        "abbreviation": journal,
        "name_variants": [journal],
    }
