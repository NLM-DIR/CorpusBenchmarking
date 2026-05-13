from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from corpus_benchmark.models.corpus import ALL_CORPUS_SUBSET, BenchmarkCorpus, CorpusSubset, Document
from corpus_benchmark.models.corpus import DocumentIdentifierType

logger = logging.getLogger(__name__)

DOCUMENT_ID = "document_id"


def apply_document_split(corpus: BenchmarkCorpus, split_config: dict[str, Any] | None) -> BenchmarkCorpus:
    """Apply an external split definition to an already-loaded corpus.

    Split definitions are intentionally loader-agnostic. A loader only has to
    produce Document objects with document_id and/or normalized identifiers.
    """
    if not split_config:
        return corpus

    documents = _collect_unique_documents(corpus)
    id_kind = split_config.get("id_type", DOCUMENT_ID)
    split_map = _normalize_split_map(load_split_map(split_config), id_kind)
    require_all_documents = bool(split_config.get("require_all_documents", True))
    allow_unknown_ids = bool(split_config.get("allow_unknown_ids", False))

    split_docs: dict[str, list[Document]] = {split_name: [] for split_name in sorted(set(split_map.values()))}

    for document in documents:
        split_name = _resolve_document_split(document, id_kind, split_map)
        if split_name is None:
            if require_all_documents:
                raise ValueError(
                    f"Document {document.document_id!r} did not match any configured split using id_type={id_kind!r}"
                )
            continue
        split_docs.setdefault(split_name, []).append(document)

    if not allow_unknown_ids:
        known_keys = {
            key
            for document in documents
            for key in _document_split_keys(document, id_kind)
        }
        unknown_ids = sorted(set(split_map) - known_keys)
        if unknown_ids:
            preview = ", ".join(unknown_ids[:10])
            extra = "" if len(unknown_ids) <= 10 else f" ... ({len(unknown_ids)} total)"
            raise ValueError(f"Split definition references IDs that were not loaded: {preview}{extra}")

    logger.info(
        "Applied external document split using id_type=%s: %s",
        id_kind,
        {split_name: len(split_docs_for_name) for split_name, split_docs_for_name in split_docs.items()},
    )
    return BenchmarkCorpus(
        subsets={
            split_name: CorpusSubset(name=split_name, documents=split_docs_for_name)
            for split_name, split_docs_for_name in split_docs.items()
        },
        metadata=corpus.metadata,
    )


def load_split_map(split_config: dict[str, Any]) -> dict[str, str]:
    source = split_config.get("source")
    if source == "files":
        return _load_split_files(split_config)
    if source == "mapping":
        return _load_split_mapping(split_config)
    if "files" in split_config:
        return _load_split_files(split_config)
    if "mapping_path" in split_config:
        return _load_split_mapping(split_config)
    raise ValueError("Split config must specify source='files' with files, or source='mapping' with mapping_path")


def _collect_unique_documents(corpus: BenchmarkCorpus) -> list[Document]:
    documents: list[Document] = []
    seen: set[str] = set()
    for subset_name, subset in corpus.subsets.items():
        if subset_name == ALL_CORPUS_SUBSET:
            continue
        for document in subset.documents:
            if document.document_id in seen:
                continue
            seen.add(document.document_id)
            documents.append(document)
    return documents


def _load_split_files(split_config: dict[str, Any]) -> dict[str, str]:
    files = split_config.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Split file config requires a non-empty files mapping")

    split_map: dict[str, str] = {}
    for split_name, path_text in files.items():
        path = Path(path_text)
        id_column = split_config.get("id_column", 0)
        delimiter = split_config.get("delimiter")
        has_header = bool(split_config.get("header", False))
        for identifier in _iter_column_values(path, id_column, delimiter, has_header):
            _add_split_id(split_map, identifier, str(split_name), path)
    return split_map


def _load_split_mapping(split_config: dict[str, Any]) -> dict[str, str]:
    path = Path(split_config["mapping_path"])
    id_column = split_config.get("id_column", 0)
    split_column = split_config.get("split_column", 1)
    delimiter = split_config.get("delimiter", "\t")
    has_header = bool(split_config.get("header", False))

    split_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter=delimiter)
        for line_index, row in enumerate(reader, start=1):
            if line_index == 1 and has_header:
                continue
            if not row or _is_comment_row(row):
                continue
            try:
                identifier = row[id_column].strip()
                split_name = row[split_column].strip()
            except IndexError as exc:
                raise ValueError(
                    f"Split mapping {path} line {line_index} does not contain columns "
                    f"{id_column} and {split_column}: {row}"
                ) from exc
            _add_split_id(split_map, identifier, split_name, path)
    return split_map


def _iter_column_values(path: Path, column: int, delimiter: str | None, has_header: bool):
    with path.open("r", encoding="utf-8", newline="") as file:
        if delimiter is None:
            for line_index, line in enumerate(file, start=1):
                if line_index == 1 and has_header:
                    continue
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                yield line
            return

        reader = csv.reader(file, delimiter=delimiter)
        for line_index, row in enumerate(reader, start=1):
            if line_index == 1 and has_header:
                continue
            if not row or _is_comment_row(row):
                continue
            try:
                yield row[column].strip()
            except IndexError as exc:
                raise ValueError(f"Split file {path} line {line_index} does not contain column {column}: {row}") from exc


def _add_split_id(split_map: dict[str, str], identifier: str, split_name: str, path: Path) -> None:
    if not identifier or not split_name:
        return
    if identifier in split_map and split_map[identifier] != split_name:
        raise ValueError(
            f"Identifier {identifier!r} appears in multiple splits: {split_map[identifier]!r} and "
            f"{split_name!r} while reading {path}"
        )
    split_map[identifier] = split_name


def _resolve_document_split(document: Document, id_kind: str, split_map: dict[str, str]) -> str | None:
    for key in _document_split_keys(document, id_kind):
        split_name = split_map.get(key)
        if split_name is not None:
            return split_name
    return None


def _document_split_keys(document: Document, id_kind: str) -> list[str]:
    if id_kind == DOCUMENT_ID:
        return [document.document_id]

    try:
        id_type = DocumentIdentifierType(id_kind.lower())
    except ValueError as exc:
        valid = ", ".join([DOCUMENT_ID] + [id_type.value for id_type in DocumentIdentifierType])
        raise ValueError(f"Invalid split id_type {id_kind!r}. Valid values: {valid}") from exc

    value = document.identifiers.get(id_type)
    if not value:
        return []
    return _dedupe([value, id_type.normalize(value)])


def _normalize_split_map(split_map: dict[str, str], id_kind: str) -> dict[str, str]:
    if id_kind == DOCUMENT_ID:
        return split_map
    try:
        id_type = DocumentIdentifierType(id_kind.lower())
    except ValueError as exc:
        valid = ", ".join([DOCUMENT_ID] + [id_type.value for id_type in DocumentIdentifierType])
        raise ValueError(f"Invalid split id_type {id_kind!r}. Valid values: {valid}") from exc

    normalized_map: dict[str, str] = {}
    for identifier, split_name in split_map.items():
        normalized_id = id_type.normalize(identifier)
        if normalized_id in normalized_map and normalized_map[normalized_id] != split_name:
            raise ValueError(
                f"Identifier {identifier!r} normalizes to {normalized_id!r}, which appears in multiple splits"
            )
        normalized_map[normalized_id] = split_name
    return normalized_map


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _is_comment_row(row: list[str]) -> bool:
    return bool(row and row[0].strip().startswith("#"))
