from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]
Normalizer = Callable[[str], str]


class RecordStoreError(Exception):
    """Base exception for JsonRecordStore errors."""


class IdentifierConflictError(RecordStoreError):
    """Raised when identifiers imply conflicting record identities."""


class MetadataConflictError(RecordStoreError):
    """Raised when metadata updates conflict with existing values."""


class UnknownIdentifierTypeError(RecordStoreError):
    """Raised when an identifier type was not declared for this store."""


class UnknownFieldError(RecordStoreError):
    """Raised when a metadata field was not declared for this store."""


class StoreFormatError(RecordStoreError):
    """Raised when a JSON file does not match the expected store format."""


@dataclass(frozen=True)
class StoredRecord:
    """
    A record returned from the store.

    record_id:
        Internal store identifier. Stable within the JSON file, but external
        code should usually use domain identifiers instead.

    identifiers:
        Mapping from identifier type to one or more identifier values.

    data:
        Arbitrary JSON-compatible key/value metadata.
    """

    record_id: int
    identifiers: dict[str, list[str]]
    data: JsonDict


class JsonRecordStore:
    """
    In-memory record store backed by JSON on request.

    This class is intended for workflows where records are loaded once,
    updated in memory, queried many times, and explicitly saved back to disk.

    Design:
      - One JSON file stores one kind of record, e.g. documents OR journals.
      - Records have arbitrary JSON-compatible metadata.
      - Records may have multiple identifiers.
      - Identifier lookup is O(1) through an in-memory index.
      - Identifier uniqueness is enforced in Python.
      - Metadata merge/conflict behavior is controlled by field policies.

    The JSON file format is intentionally human-readable:

        {
          "format_version": 1,
          "next_record_id": 3,
          "records": [
            {
              "record_id": 1,
              "identifiers": {
                "PMID": ["123456"],
                "DOI": ["10.1000/example"]
              },
              "data": {
                "publication_year": 2023,
                "mesh_topics": ["Humans"]
              }
            }
          ]
        }

    Merge policies:

      strict:
          Missing existing value -> add.
          Existing None -> replace with incoming non-None.
          Incoming None -> ignored if existing already has a value.
          Same value -> okay.
          Different non-None value -> conflict.

      set_union:
          Treat existing and incoming values as list-like values and merge
          without duplicates while preserving first-seen order.

      append:
          Append incoming value(s) to a list. Duplicates are preserved.

      replace:
          Replace existing value with incoming value.

    By default, undeclared metadata fields are rejected. Set
    allow_unknown_fields=True if you want arbitrary data keys.
    """

    FORMAT_VERSION = 1

    VALID_FIELD_POLICIES = {
        "strict",
        "strict_case_insensitive",
        "first_seen",
        "set_union",
        "append",
        "replace",
    }

    def __init__(
        self,
        path: str | Path,
        *,
        identifier_types: Iterable[str],
        fields: Iterable[str] | None = None,
        field_policies: Mapping[str, str] | None = None,
        identifier_normalizers: Mapping[str, Normalizer] | None = None,
        allow_unknown_fields: bool = False,
        autoload: bool = True,
    ) -> None:
        self.path = Path(path)
        self.identifier_types = {self._canonical_identifier_type(t) for t in identifier_types}

        self.fields = set(fields or [])
        self.allow_unknown_fields = allow_unknown_fields

        self.field_policies = dict(field_policies or {})
        for field, policy in self.field_policies.items():
            if policy not in self.VALID_FIELD_POLICIES:
                raise ValueError(f"Invalid merge policy {policy!r} for field {field!r}. " f"Expected one of {sorted(self.VALID_FIELD_POLICIES)}.")

        unknown_policy_fields = set(self.field_policies) - self.fields
        if unknown_policy_fields and not allow_unknown_fields:
            raise ValueError("Field policies were provided for undeclared fields: " f"{sorted(unknown_policy_fields)}")

        self.identifier_normalizers: dict[str, Normalizer] = {self._canonical_identifier_type(k): v for k, v in (identifier_normalizers or {}).items()}

        self._records: dict[int, StoredRecord] = {}
        self._identifier_index: dict[tuple[str, str], int] = {}
        self._next_record_id = 1
        self.dirty = False

        if autoload and self.path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def new_empty(
        cls,
        path: str | Path,
        *,
        identifier_types: Iterable[str],
        fields: Iterable[str] | None = None,
        field_policies: Mapping[str, str] | None = None,
        identifier_normalizers: Mapping[str, Normalizer] | None = None,
        allow_unknown_fields: bool = False,
    ) -> JsonRecordStore:
        """
        Create an empty store without loading an existing file.

        This is useful when you intentionally want to overwrite an existing
        file later with save().
        """
        return cls(
            path,
            identifier_types=identifier_types,
            fields=fields,
            field_policies=field_policies,
            identifier_normalizers=identifier_normalizers,
            allow_unknown_fields=allow_unknown_fields,
            autoload=False,
        )

    # ------------------------------------------------------------------
    # Identifier normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_identifier_type(id_type: str) -> str:
        return id_type.strip().upper()

    def _normalize_identifier_value(self, id_type: str, id_value: Any) -> str:
        if id_value is None:
            raise ValueError(f"Identifier value for {id_type!r} cannot be None.")

        value = str(id_value).strip()
        if not value:
            raise ValueError(f"Identifier value for {id_type!r} cannot be empty.")

        normalizer = self.identifier_normalizers.get(id_type)
        if normalizer is not None:
            value = normalizer(value)

        return value

    def _normalize_identifiers(
        self,
        identifiers: Mapping[str, Any],
    ) -> dict[str, set[str]]:
        """
        Normalize input identifiers to:

            {
                "PMID": {"123"},
                "DOI": {"10.123/example"}
            }

        Values may be scalar or iterable. Strings are treated as scalar values.
        """
        normalized: dict[str, set[str]] = {}

        for raw_type, raw_values in identifiers.items():
            id_type = self._canonical_identifier_type(raw_type)

            if id_type not in self.identifier_types:
                raise UnknownIdentifierTypeError(f"Identifier type {raw_type!r} is not allowed. " f"Allowed types: {sorted(self.identifier_types)}")

            if raw_values is None:
                continue

            if isinstance(raw_values, str):
                values = [raw_values]
            else:
                try:
                    values = list(raw_values)
                except TypeError:
                    values = [raw_values]

            for raw_value in values:
                id_value = self._normalize_identifier_value(id_type, raw_value)
                normalized.setdefault(id_type, set()).add(id_value)

        if not normalized:
            raise ValueError("At least one identifier is required.")

        return normalized

    @staticmethod
    def _identifiers_to_lists(
        identifiers: Mapping[str, Iterable[str]],
    ) -> dict[str, list[str]]:
        return {str(id_type): sorted({str(value) for value in values}) for id_type, values in sorted(identifiers.items())}

    # ------------------------------------------------------------------
    # Data validation and merging
    # ------------------------------------------------------------------

    def _validate_data_fields(self, data: Mapping[str, Any]) -> None:
        if self.allow_unknown_fields:
            return

        unknown = set(data) - self.fields
        if unknown:
            raise UnknownFieldError(f"Unknown metadata fields: {sorted(unknown)}. " f"Allowed fields: {sorted(self.fields)}")

    def _merge_data(self, existing: JsonDict, incoming: Mapping[str, Any]) -> JsonDict:
        self._validate_data_fields(incoming)

        merged = dict(existing)

        for key, incoming_value in incoming.items():
            policy = self.field_policies.get(key, "strict")

            if key not in merged:
                merged[key] = incoming_value
                continue

            existing_value = merged[key]

            if policy == "strict":
                if existing_value is None:
                    merged[key] = incoming_value
                elif incoming_value is None:
                    pass
                elif existing_value == incoming_value:
                    pass
                else:
                    raise MetadataConflictError(f"Conflicting value for field {key!r}: " f"existing={existing_value!r}, incoming={incoming_value!r}")

            elif policy == "strict_case_insensitive":
                if existing_value is None:
                    merged[key] = incoming_value
                elif incoming_value is None:
                    pass
                elif str(existing_value).lower() == str(incoming_value).lower():
                    pass
                else:
                    raise MetadataConflictError(f"Conflicting value (case-insensitive) for field {key!r}: " f"existing={existing_value!r}, incoming={incoming_value!r}")

            elif policy == "first_seen":
                if existing_value is None:
                    merged[key] = incoming_value
                # Otherwise, keep existing_value regardless of incoming_value

            elif policy == "set_union":
                merged[key] = self._merge_set_union(existing_value, incoming_value)

            elif policy == "append":
                merged[key] = self._merge_append(existing_value, incoming_value)

            elif policy == "replace":
                merged[key] = incoming_value

            else:
                raise AssertionError(f"Unhandled field policy: {policy!r}")

        return merged

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

    @classmethod
    def _merge_set_union(cls, existing_value: Any, incoming_value: Any) -> list[Any]:
        result: list[Any] = []

        for value in cls._as_list(existing_value) + cls._as_list(incoming_value):
            if value not in result:
                result.append(value)

        return result

    @classmethod
    def _merge_append(cls, existing_value: Any, incoming_value: Any) -> list[Any]:
        return cls._as_list(existing_value) + cls._as_list(incoming_value)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def upsert(
        self,
        *,
        identifiers: Mapping[str, Any],
        data: Mapping[str, Any] | None = None,
    ) -> StoredRecord:
        """
        Insert or update a record.

        If none of the incoming identifiers are known:
            Create a new record.

        If exactly one existing record is matched:
            Merge identifiers and metadata into that record.

        If multiple existing records are matched:
            Raise IdentifierConflictError, because the incoming identifiers
            imply that distinct existing records are actually the same record.
        """
        normalized_ids = self._normalize_identifiers(identifiers)
        incoming_data = dict(data or {})
        self._validate_data_fields(incoming_data)

        matching_record_ids = self._find_record_ids_for_identifiers(normalized_ids)

        if len(matching_record_ids) > 1:
            raise IdentifierConflictError("Incoming identifiers match multiple records: " f"{sorted(matching_record_ids)}")

        if not matching_record_ids:
            record_id = self._next_record_id
            self._next_record_id += 1

            merged_data = self._merge_data({}, incoming_data)
            merged_identifiers = self._identifiers_to_lists(normalized_ids)

            record = StoredRecord(
                record_id=record_id,
                identifiers=merged_identifiers,
                data=merged_data,
            )

            self._records[record_id] = record
            self._index_record(record)
            self.dirty = True
            return record

        record_id = next(iter(matching_record_ids))
        existing_record = self._records[record_id]

        merged_data = self._merge_data(existing_record.data, incoming_data)

        merged_identifier_sets: dict[str, set[str]] = {id_type: set(values) for id_type, values in existing_record.identifiers.items()}
        for id_type, values in normalized_ids.items():
            merged_identifier_sets.setdefault(id_type, set()).update(values)

        merged_identifiers = self._identifiers_to_lists(merged_identifier_sets)

        updated_record = StoredRecord(
            record_id=record_id,
            identifiers=merged_identifiers,
            data=merged_data,
        )

        self._unindex_record(existing_record)
        self._assert_identifiers_available(updated_record)
        self._records[record_id] = updated_record
        self._index_record(updated_record)

        if updated_record != existing_record:
            self.dirty = True

        return updated_record

    def get(self, id_type: str, id_value: Any) -> StoredRecord | None:
        """Return a record by one of its identifiers, or None if absent."""
        id_type_norm = self._canonical_identifier_type(id_type)

        if id_type_norm not in self.identifier_types:
            raise UnknownIdentifierTypeError(f"Identifier type {id_type!r} is not allowed.")

        id_value_norm = self._normalize_identifier_value(id_type_norm, id_value)
        record_id = self._identifier_index.get((id_type_norm, id_value_norm))

        if record_id is None:
            return None

        return self._records[record_id]

    def get_by_record_id(self, record_id: int) -> StoredRecord:
        try:
            return self._records[record_id]
        except KeyError as exc:
            raise KeyError(f"No record with record_id={record_id}") from exc

    def __contains__(self, key: object) -> bool:
        """
        Support:

            ("PMID", "123456") in store
        """
        if not isinstance(key, tuple) or len(key) != 2:
            return False

        id_type, id_value = key
        try:
            return self.get(str(id_type), id_value) is not None
        except (ValueError, UnknownIdentifierTypeError):
            return False

    def __iter__(self) -> Iterator[StoredRecord]:
        """Iterate over all records in record_id order."""
        for record_id in sorted(self._records):
            yield self._records[record_id]

    def values(self) -> Iterator[StoredRecord]:
        """Alias for iterating over all known records."""
        return iter(self)

    def items(self) -> Iterator[tuple[int, StoredRecord]]:
        """Iterate over internal record id plus record."""
        for record_id in sorted(self._records):
            yield record_id, self._records[record_id]

    def count(self) -> int:
        return len(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def delete_record(self, record_id: int) -> None:
        """Delete a record and all identifiers attached to it."""
        record = self.get_by_record_id(record_id)
        self._unindex_record(record)
        del self._records[record_id]
        self.dirty = True

    # ------------------------------------------------------------------
    # Index handling
    # ------------------------------------------------------------------

    def _find_record_ids_for_identifiers(
        self,
        identifiers: Mapping[str, set[str]],
    ) -> set[int]:
        record_ids: set[int] = set()

        for id_type, values in identifiers.items():
            for id_value in values:
                record_id = self._identifier_index.get((id_type, id_value))
                if record_id is not None:
                    record_ids.add(record_id)

        return record_ids

    def _assert_identifiers_available(self, record: StoredRecord) -> None:
        for id_type, values in record.identifiers.items():
            for id_value in values:
                key = (id_type, id_value)
                existing_record_id = self._identifier_index.get(key)

                if existing_record_id is not None and existing_record_id != record.record_id:
                    raise IdentifierConflictError(f"Identifier {id_type}:{id_value} already belongs to " f"record {existing_record_id}, not record {record.record_id}.")

    def _index_record(self, record: StoredRecord) -> None:
        self._assert_identifiers_available(record)

        for id_type, values in record.identifiers.items():
            for id_value in values:
                self._identifier_index[(id_type, id_value)] = record.record_id

    def _unindex_record(self, record: StoredRecord) -> None:
        for id_type, values in record.identifiers.items():
            for id_value in values:
                self._identifier_index.pop((id_type, id_value), None)

    def rebuild_index(self) -> None:
        """Rebuild the in-memory identifier index from stored records."""
        self._identifier_index.clear()
        for record in self._records.values():
            self._index_record(record)

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load records from disk, replacing current in-memory contents.

        If the file does not exist, the store remains empty.
        """
        if not self.path.exists():
            self._records.clear()
            self._identifier_index.clear()
            self._next_record_id = 1
            self.dirty = False
            return

        logging.debug(f'Loading record store from "{self.path}"')
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise StoreFormatError("Store JSON must contain an object at the top level.")

        format_version = payload.get("format_version")
        if format_version != self.FORMAT_VERSION:
            raise StoreFormatError(f"Unsupported format_version={format_version!r}; " f"expected {self.FORMAT_VERSION}.")

        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise StoreFormatError("Store JSON must contain a list field named 'records'.")

        records: dict[int, StoredRecord] = {}

        for raw_record in raw_records:
            record = self._parse_raw_record(raw_record)
            if record.record_id in records:
                raise StoreFormatError(f"Duplicate record_id={record.record_id}.")
            records[record.record_id] = record

        self._records = records
        self._identifier_index.clear()
        self.rebuild_index()

        raw_next_record_id = payload.get("next_record_id")
        if raw_next_record_id is None:
            self._next_record_id = (max(self._records) + 1) if self._records else 1
        else:
            self._next_record_id = int(raw_next_record_id)
            min_next = (max(self._records) + 1) if self._records else 1
            if self._next_record_id < min_next:
                self._next_record_id = min_next

        self.dirty = False

    def _parse_raw_record(self, raw_record: Any) -> StoredRecord:
        if not isinstance(raw_record, dict):
            raise StoreFormatError("Each record must be a JSON object.")

        try:
            record_id = int(raw_record["record_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreFormatError("Each record must have an integer record_id.") from exc

        raw_identifiers = raw_record.get("identifiers")
        if not isinstance(raw_identifiers, dict):
            raise StoreFormatError(f"Record {record_id} must have an identifiers object.")

        identifiers = self._normalize_identifiers(raw_identifiers)
        identifiers_as_lists = self._identifiers_to_lists(identifiers)

        raw_data = raw_record.get("data", {})
        if not isinstance(raw_data, dict):
            raise StoreFormatError(f"Record {record_id} data must be an object.")

        self._validate_data_fields(raw_data)

        return StoredRecord(
            record_id=record_id,
            identifiers=identifiers_as_lists,
            data=dict(raw_data),
        )

    def to_json_payload(self) -> JsonDict:
        """Return the serializable JSON payload for this store."""
        return {
            "format_version": self.FORMAT_VERSION,
            "next_record_id": self._next_record_id,
            "records": [
                {
                    "record_id": record.record_id,
                    "identifiers": record.identifiers,
                    "data": record.data,
                }
                for record in self
            ],
        }

    def save(self, *, force: bool = False, indent: int = 2) -> None:
        """
        Save records to disk.

        Writes are atomic:
          1. write JSON to a temporary file in the same directory
          2. fsync the temporary file
          3. replace the target path with os.replace()

        If force=False and the store is not dirty, no file is written.
        """
        if not force and not self.dirty:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json_payload()

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=indent)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_name, self.path)
            self.dirty = False

        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
