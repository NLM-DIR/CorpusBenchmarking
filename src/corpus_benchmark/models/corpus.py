from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import gzip
import json
import logging
from pathlib import Path
import re
from typing import Any, Optional, TextIO

from corpus_benchmark.models.types import MatchType, LinkRelation

logger = logging.getLogger(__name__)

NIL_RESOURCE = "<NIL>"
ALL_CORPUS_SUBSET = "<ALL>"
CORPUS_JSON_SCHEMA_VERSION = 1


def _enum_value(value: Enum | str | None) -> str | None:
    """Return the JSON-safe value for an enum-like field."""
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _enum_or_none(enum_type: type[Enum], value: str | None) -> Enum | None:
    """Parse a nullable enum field with a clearer error message than Enum(value)."""
    if value is None:
        return None
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(f"Invalid {enum_type.__name__} value {value!r}; expected one of: {allowed}") from exc


def _open_text_for_read(path: str | Path) -> TextIO:
    """Open JSON or JSON.GZ paths for text reading."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_text_for_write(path: str | Path) -> TextIO:
    """Open JSON or JSON.GZ paths for text writing, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


@dataclass(slots=True)
class Link:
    """
    Base class for all link structures.

    This is mainly used for typing so Annotation.link can contain either an
    IdentifierLink or a CompositeLink.
    """

    match_type: Optional[MatchType] = None

    @abstractmethod
    def get_identifier_links(self) -> list[IdentifierLink]:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert the link to a JSON-serializable dictionary."""
        pass

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> Link | None:
        """Reconstruct a Link from its JSON representation."""
        if data is None:
            return None
        link_type = data.get("type")
        if link_type == "identifier":
            return IdentifierLink.from_dict(data)
        if link_type == "composite":
            return CompositeLink.from_dict(data)
        raise ValueError(f"Unknown link type: {link_type!r}")


@dataclass(slots=True)
class IdentifierLink(Link):
    """
    Leaf link to a single identifier.

    identifier=None represents a NIL link.
    """

    identifier: Optional[str] = None
    resource: Optional[str] = None

    def get_identifier_links(self) -> list[IdentifierLink]:
        return [self]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "identifier",
            "match_type": _enum_value(self.match_type),
            "identifier": self.identifier,
            "resource": self.resource,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> IdentifierLink:
        return IdentifierLink(
            match_type=_enum_or_none(MatchType, data.get("match_type")),
            identifier=data.get("identifier"),
            resource=data.get("resource"),
        )

    def __str__(self) -> str:
        return f"IdentifierLink(match_type={self.match_type}, resource={self.resource}, identifier={self.identifier})"


NIL = IdentifierLink(identifier=None, resource=NIL_RESOURCE, match_type=MatchType.NIL)


@dataclass(slots=True)
class CompositeLink(Link):
    """
    Recursive link structure containing other links and the semantic
    relationship between them.
    """

    relation: LinkRelation = LinkRelation.RELATED_SET
    components: list[Link] = field(default_factory=list)

    def get_identifier_links(self) -> list[IdentifierLink]:
        identifier_links: list[IdentifierLink] = []
        for link in self.components:
            identifier_links.extend(link.get_identifier_links())
        return identifier_links

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "composite",
            "match_type": _enum_value(self.match_type),
            "relation": _enum_value(self.relation),
            "components": [component.to_dict() for component in self.components],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CompositeLink:
        return CompositeLink(
            match_type=_enum_or_none(MatchType, data.get("match_type")),
            relation=_enum_or_none(LinkRelation, data.get("relation")) or LinkRelation.RELATED_SET,
            components=[link for link in (Link.from_dict(item) for item in data.get("components", [])) if link is not None],
        )

    def __str__(self) -> str:
        return f"CompositeLink(relation={self.relation}, components={self.components})"


@dataclass(slots=True)
class AnnotationSpan:
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AnnotationSpan:
        return AnnotationSpan(start=int(data["start"]), end=int(data["end"]))


@dataclass(slots=True)
class Annotation:
    mention_id: str
    text: str
    spans: list[AnnotationSpan]
    label: Optional[str]
    link: Optional[Link]
    attributes: dict[str, str] = field(default_factory=dict)

    def get_identifier_links(self) -> list[IdentifierLink]:
        if self.link is None:
            return []
        return self.link.get_identifier_links()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "text": self.text,
            "spans": [span.to_dict() for span in self.spans],
            "label": self.label,
            "link": self.link.to_dict() if self.link is not None else None,
            "attributes": dict(self.attributes),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Annotation:
        return Annotation(
            mention_id=str(data["mention_id"]),
            text=str(data.get("text", "")),
            spans=[AnnotationSpan.from_dict(span) for span in data.get("spans", [])],
            label=data.get("label"),
            link=Link.from_dict(data.get("link")),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass(slots=True)
class Passage:
    passage_id: str
    text: str
    offset: int
    annotations: list[Annotation] = field(default_factory=list)
    infons: dict[str, str] = field(default_factory=dict)

    # TODO Check that the text for each annotation matches the text at the spans

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "text": self.text,
            "offset": self.offset,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
            "infons": dict(self.infons),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Passage:
        return Passage(
            passage_id=str(data["passage_id"]),
            text=str(data.get("text", "")),
            offset=int(data.get("offset", 0)),
            annotations=[Annotation.from_dict(annotation) for annotation in data.get("annotations", [])],
            infons=dict(data.get("infons", {})),
        )


class DocumentIdentifierType(str, Enum):
    PMID = "pmid"
    PMCID = "pmcid"
    DOI = "doi"

    def normalize(self, value: str) -> str:
        """Standardize the formatting of the identifier."""
        if not value:
            return value

        if self == DocumentIdentifierType.PMID:
            value = str(value).strip()
            if not value.isdigit():
                raise ValueError(f"PMID should contain only digits: {value!r}")
            # Strip 'PMID:' prefixes if they somehow got included
            value = re.sub(r"^pmid:\s*", "", value, flags=re.IGNORECASE)
            return value

        if self == DocumentIdentifierType.PMCID:
            # Ensure it is uppercase and starts with 'PMC'
            value = str(value).strip().upper()
            if value.isdigit():
                value = f"PMC{value}"
            if not value.startswith("PMC"):
                raise ValueError(f"PMCID should look like PMC123456: {value!r}")
            return value

        if self == DocumentIdentifierType.DOI:
            value = str(value).strip().lower()
            # Strip common DOI resolver URLs and 'doi:' prefixes
            # Handles: https://doi.org/10... | http://dx.doi.org/10... | doi:10...
            value = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
            value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
            return value

        # Any future types pass through stripped
        return value


@dataclass(slots=True)
class Document:
    document_id: str
    passages: list[Passage] = field(default_factory=list)
    identifiers: dict[DocumentIdentifierType, str] = field(default_factory=dict)
    infons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "passages": [passage.to_dict() for passage in self.passages],
            "identifiers": {_enum_value(key): value for key, value in self.identifiers.items()},
            "infons": dict(self.infons),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Document:
        identifiers: dict[DocumentIdentifierType, str] = {}
        for key, value in data.get("identifiers", {}).items():
            identifier_type = DocumentIdentifierType(key)
            identifiers[identifier_type] = identifier_type.normalize(value)

        return Document(
            document_id=str(data["document_id"]),
            passages=[Passage.from_dict(passage) for passage in data.get("passages", [])],
            identifiers=identifiers,
            infons=dict(data.get("infons", {})),
        )


@dataclass(slots=True)
class CorpusSubset:
    name: str
    documents: list[Document]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "documents": [document.to_dict() for document in self.documents],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CorpusSubset:
        return CorpusSubset(
            name=str(data["name"]),
            documents=[Document.from_dict(document) for document in data.get("documents", [])],
        )


@dataclass(slots=True)
class BenchmarkCorpus:
    subsets: dict[str, CorpusSubset] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rebuild_all_subset()

    def rebuild_all_subset(self) -> None:
        """Recreate the derived <ALL> subset from the named source subsets."""
        all_documents: list[Document] = []
        for subset_name, subset in self.subsets.items():
            if subset_name == ALL_CORPUS_SUBSET:
                continue
            all_documents.extend(subset.documents)
        self.subsets[ALL_CORPUS_SUBSET] = CorpusSubset(ALL_CORPUS_SUBSET, all_documents)
        logger.debug("Constructed ALL corpus subset with %s documents", len(all_documents))

    def source_subsets(self) -> dict[str, CorpusSubset]:
        """Return subsets that should be serialized, excluding the derived <ALL> subset."""
        return {name: subset for name, subset in self.subsets.items() if name != ALL_CORPUS_SUBSET}

    def to_dict(self, *, include_all_subset: bool = False) -> dict[str, Any]:
        """
        Convert the corpus to a JSON-serializable dictionary.

        By default, the derived <ALL> subset is omitted to avoid duplicating every
        document in the cache file. It is reconstructed automatically on load.
        """
        subsets = self.subsets if include_all_subset else self.source_subsets()
        return {
            "schema_version": CORPUS_JSON_SCHEMA_VERSION,
            "metadata": dict(self.metadata),
            "subsets": {name: subset.to_dict() for name, subset in subsets.items()},
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> BenchmarkCorpus:
        schema_version = data.get("schema_version", 0)
        if schema_version not in (0, CORPUS_JSON_SCHEMA_VERSION):
            raise ValueError(f"Unsupported BenchmarkCorpus JSON schema_version={schema_version!r}; " f"this code supports {CORPUS_JSON_SCHEMA_VERSION}.")

        subsets = {name: CorpusSubset.from_dict(subset_data) for name, subset_data in data.get("subsets", {}).items() if name != ALL_CORPUS_SUBSET}
        return BenchmarkCorpus(subsets=subsets, metadata=dict(data.get("metadata", {})))

    def to_json(
        self,
        path: str | Path,
        *,
        include_all_subset: bool = False,
        indent: int | None = 2,
        sort_keys: bool = False,
    ) -> None:
        """
        Write the corpus to a JSON or JSON.GZ cache file.

        Use a .gz suffix for compressed output. Plain .json is more convenient
        for inspection and diffs; .json.gz is usually better for large corpora.
        """
        with _open_text_for_write(path) as handle:
            json.dump(
                self.to_dict(include_all_subset=include_all_subset),
                handle,
                ensure_ascii=False,
                indent=indent,
                sort_keys=sort_keys,
            )
            handle.write("\n")

    @staticmethod
    def from_json(path: str | Path) -> BenchmarkCorpus:
        """Load a BenchmarkCorpus from a JSON or JSON.GZ cache file."""
        with _open_text_for_read(path) as handle:
            data = json.load(handle)
        return BenchmarkCorpus.from_dict(data)


@dataclass(slots=True)
class BenchmarkBattery:
    corpora: dict[str, BenchmarkCorpus] = field(default_factory=dict)

    def to_dict(self, *, include_all_subset: bool = False) -> dict[str, Any]:
        return {
            "schema_version": CORPUS_JSON_SCHEMA_VERSION,
            "corpora": {name: corpus.to_dict(include_all_subset=include_all_subset) for name, corpus in self.corpora.items()},
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> BenchmarkBattery:
        schema_version = data.get("schema_version", 0)
        if schema_version not in (0, CORPUS_JSON_SCHEMA_VERSION):
            raise ValueError(f"Unsupported BenchmarkBattery JSON schema_version={schema_version!r}; " f"this code supports {CORPUS_JSON_SCHEMA_VERSION}.")
        return BenchmarkBattery(corpora={name: BenchmarkCorpus.from_dict(corpus_data) for name, corpus_data in data.get("corpora", {}).items()})

    def to_json(
        self,
        path: str | Path,
        *,
        include_all_subset: bool = False,
        indent: int | None = 2,
        sort_keys: bool = False,
    ) -> None:
        """Write the battery to a JSON or JSON.GZ cache file."""
        with _open_text_for_write(path) as handle:
            json.dump(
                self.to_dict(include_all_subset=include_all_subset),
                handle,
                ensure_ascii=False,
                indent=indent,
                sort_keys=sort_keys,
            )
            handle.write("\n")

    @staticmethod
    def from_json(path: str | Path) -> BenchmarkBattery:
        """Load a BenchmarkBattery from a JSON or JSON.GZ cache file."""
        with _open_text_for_read(path) as handle:
            data = json.load(handle)
        return BenchmarkBattery.from_dict(data)
