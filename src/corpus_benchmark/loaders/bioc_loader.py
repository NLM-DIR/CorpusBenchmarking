from __future__ import annotations

import logging
from collections import Counter
from typing import List
import os

from bioc import biocxml, pubtator

from corpus_benchmark.models.corpus import (
    BenchmarkCorpus,
    CorpusSubset,
    Document,
    DocumentIdentifierType,
    NIL_RESOURCE,
    Passage,
    Annotation,
    AnnotationSpan,
    Link,
    IdentifierLink,
    CompositeLink,
    NIL,
)
from corpus_benchmark.models.types import MatchType
from corpus_benchmark.registry import register_loader
from corpus_benchmark.loaders.parsing import parse_identifier_format_list, parse_qualifier_map, IdentifierFormat
from corpus_benchmark.loaders.splits import apply_document_split

logger = logging.getLogger(__name__)


def _resolve_load_paths(paths: dict[str, str] | None, path: str | None) -> dict[str, str]:
    if paths and path:
        raise ValueError("Configure either loader.params.paths or loader.params.path, not both")
    if paths:
        return paths
    if path:
        return {"all": path}
    raise ValueError("Loader requires either loader.params.paths or loader.params.path")


@register_loader("bioc_xml")
def load_bioc_xml(
    paths: dict[str, str] | None = None,
    path: str | None = None,
    split: dict | None = None,
    doc_id_map: dict[str, str] = {},
    passage_id_infon_key: str | None = None,
    label_infon_key: str = "type",
    id_infon_key: str = "identifier",
    label_map: dict[str, str | None] = {},
    id_format_list: list[list[str]] = [],
    qualifier_map: dict[str, str] = {},
    nil_labels: set[str] = set(),
    default_resource: str | None = None,
    resource_delimiter: str = ":",
    **kwargs,
) -> BenchmarkCorpus:
    """
    Load a BioC XML file and convert it into the internal corpus model.

    Notes:
    - This minimal implementation uses only the first location for each annotation.
    - If your corpus uses discontinuous spans, extend this representation before production use.
    """
    loader = BioCXMLLoader(
        doc_id_map,
        passage_id_infon_key,
        label_infon_key,
        id_infon_key,
        label_map,
        parse_identifier_format_list(id_format_list),
        parse_qualifier_map(qualifier_map),
        nil_labels,
        default_resource,
        resource_delimiter,
        **kwargs,
    )
    load_paths = _resolve_load_paths(paths, path)
    subsets = dict()
    for subset_name, subset_path in load_paths.items():
        subsets[subset_name] = loader.load_subset(subset_name, subset_path)
    corpus = BenchmarkCorpus(
        subsets=subsets,
        metadata={
            "source_format": "BioC XML",
        },
    )
    return apply_document_split(corpus, split)


class Loader:

    def __init__(
        self,
        label_map: dict[str, str | None] = {},
        id_format_list: list[IdentifierFormat] = [],
        qualifier_map: dict[str, MatchType] = {},
        nil_labels: set[str] = set(),
        default_resource: str | None = None,
        resource_delimiter: str = ":",
    ):
        self.label_map = label_map
        self.id_format_list = id_format_list
        self.qualifier_map = qualifier_map
        self.nil_labels = set(str(label) for label in nil_labels)
        self.default_resource = default_resource
        self.resource_delimiter = resource_delimiter

    def get_label(self, label_text) -> str:
        label = self.label_map.get(label_text, label_text)
        if label is None:
            return None
        return str(label)

    def get_identifier(self, identifier_text) -> Link | None:
        if identifier_text is None:
            return None
        link = self._parse_identifier(identifier_text.strip(), self.id_format_list)
        # logger.debug(f'TRACE Loader.parse_identifier(): identifier_text = "{identifier_text}"; link = "{link}"')
        return link

    def _parse_identifier(self, identifier_text: str, identifier_format_list: list[IdentifierFormat] | None) -> Link:
        #logger.debug(f'Loader.parse_identifier(): identifier_text = "{identifier_text}"; identifier_format_list = "{identifier_format_list}"')
        if identifier_format_list is None or len(identifier_format_list) == 0:
            return self._parse_atomic_identifier(identifier_text)
        identifier_format = identifier_format_list[0]
        #logger.debug(f'Loader.parse_identifier(): identifier_text = "{identifier_text}"; identifier_format = "{identifier_format}"')
        remaining_identifier_formats = identifier_format_list[1:]
        match_type = None
        #logger.debug(
        #    f'Loader.parse_identifier(): identifier_text = "{identifier_text}"; identifier_format.qualifier_allowed = "{identifier_format.qualifier_allowed}" type(identifier_format.qualifier_allowed) = "{type(identifier_format.qualifier_allowed)}"'
        #)
        if identifier_format.qualifier_allowed:
            # mapping_debug = [
            #    (qualifier_text, match_type, identifier_text.startswith(qualifier_text))
            #    for qualifier_text, match_type in self.qualifier_map.items()
            # ]
            # logger.debug(f"Loader.parse_identifier(): identifier_text = \"{identifier_text}\"; mapping_debug = \"{mapping_debug}\"")
            mapping = [(len(qualifier_text), match_type) for qualifier_text, match_type in self.qualifier_map.items() if identifier_text.startswith(qualifier_text)]
            if len(mapping) > 0:
                mapping.sort(reverse=True)
                match_length, match_type = mapping[0]
                identifier_text = identifier_text[match_length:]
        #logger.debug(f'Loader.parse_identifier(): identifier_text = "{identifier_text}"; match_type = "{match_type}"')
        identifier_elements = [self._parse_identifier(element_text.strip(), remaining_identifier_formats) for element_text in identifier_text.split(identifier_format.delimiter)]
        #logger.debug(f'Loader.parse_identifier(): identifier_text = "{identifier_text}"; identifier_elements = "{identifier_elements}"')
        if len(identifier_elements) == 1:
            link = identifier_elements[0]
            link.match_type = match_type
            return link
        return CompositeLink(relation=identifier_format.relation, components=identifier_elements, match_type=match_type)

    def _parse_atomic_identifier(self, identifier_text) -> IdentifierLink:
        #logger.debug(f'Loader.parse_atomic_identifier(): identifier_text = "{identifier_text}"')
        identifier_text = identifier_text.strip()
        if identifier_text in self.nil_labels:
            return IdentifierLink(identifier=None, resource=NIL_RESOURCE, match_type=NIL.match_type)
        if self.resource_delimiter in identifier_text:
            fields = identifier_text.split(self.resource_delimiter)
            if len(fields) != 2:
                raise ValueError(f'Identifier "{identifier_text}" cannot be split into exactly 2 fields using the configured delimiter "{self.resource_delimiter}"')
            resource, accession = fields
        elif not self.default_resource is None:
            resource = self.default_resource
            accession = identifier_text
        else:
            resource = None
            accession = identifier_text
        return IdentifierLink(resource=resource, identifier=accession)


class DocIDExtractor:

    def __init__(self, loader: Loader, doc_id_map: dict[str, str] = {}):
        self.loader = loader
        if not isinstance(loader, BioCXMLLoader) and not isinstance(loader, BioCPubtatorLoader):
            raise ValueError(f"Unknown loader type: {type(loader)}")
        self.doc_id_map = doc_id_map

    def get_IDs(self, doc) -> dict[DocumentIdentifierType, str]:
        raw_ids = dict()
        if isinstance(self.loader, BioCXMLLoader):
            raw_ids.update(self.get_BioCXML_IDs(doc))
        elif isinstance(self.loader, BioCPubtatorLoader):
            raw_ids.update(self.get_Pubtator_IDs(doc))
        else:
            raise ValueError(f"Unknown loader type: {type(self.loader)}")
        ids = dict()
        for id_type, doc_id in raw_ids.items():
            try:
                id_type_enum = DocumentIdentifierType(id_type.lower())
            except ValueError:
                logger.warning(f"Unknown identifier type '{id_type}' configured in YAML.")
                continue
            try:
                ids[id_type_enum] = id_type_enum.normalize(doc_id)
            except ValueError:
                logger.warning("Invalid %s identifier value %r configured in YAML.", id_type_enum.value, doc_id)
        return ids

    def get_BioCXML_IDs(self, doc) -> dict[str, str]:
        ids = dict()
        for id_type, location in self.doc_id_map.items():
            match location:
                # Case 1: Simple string match for Document ID
                case "__DOCUMENT_ID__":
                    doc_id = str(doc.id)

                # Case 2: List/Tuple with ["__HEADER_INFON__", "key"]
                case ["__HEADER_INFON__", key] if len(doc.passages) > 0:
                    val = doc.passages[0].infons.get(key)
                    doc_id = str(val) if val is not None else None

                # Case 3: List/Tuple with ["__DOCUMENT_INFON__", "key"]
                case ["__DOCUMENT_INFON__", key]:
                    val = doc.infons.get(key)
                    doc_id = str(val) if val is not None else None

                # Case 4: Catch-all for legacy strings or invalid formats
                case _:
                    raise ValueError(f"Invalid location format or value: {location}")

            if doc_id:
                ids[id_type] = doc_id

        return ids

    def get_Pubtator_IDs(self, doc) -> dict[str, str]:
        ids = dict()
        for id_type, location in self.doc_id_map.items():
            if location != "__DOCUMENT_ID__":
                raise ValueError(f'Pubtator format only supports location="__DOCUMENT_ID__": {location}')
            ids[id_type] = str(doc.pmid)
        return ids


class BioCXMLLoader(Loader):

    def __init__(
        self,
        doc_id_map: dict[str, str] = {},
        passage_id_infon_key: str | None = None,
        label_infon_key: str = "type",
        id_infon_key: str = "identifier",
        label_map: dict[str, str | None] = {},
        id_format_list: list[IdentifierFormat] = [],
        qualifier_map: dict[str, MatchType] = {},
        nil_labels: set[str] = set(),
        default_resource: str | None = None,
        resource_delimiter=":",
        **kwargs,
    ):
        super().__init__(label_map, id_format_list, qualifier_map, nil_labels, default_resource, resource_delimiter)
        self.doc_id_fetcher = DocIDExtractor(self, doc_id_map)
        self.passage_id_infon_key = passage_id_infon_key
        self.label_infon_key = label_infon_key
        self.id_infon_key = id_infon_key

    def load_subset(self, subset_name: str, path: str):
        """
        Load a BioC XML file and convert it into the internal corpus model.
        """
        logger.info(f"Loading subset {subset_name} from {path}")
        if os.path.isfile(path):
            documents, id_type_counts =  self._load_bioc_file(path)
        elif os.path.isdir(path):
            documents, id_type_counts =  self._load_bioc_dir(path)
        else:
            raise ValueError(f"The path '{path}' is neither a valid file nor a directory.")
        return CorpusSubset(name=subset_name, documents=documents)

    def _load_bioc_dir(self, dirname: str) -> List[Document]:
        """
        Iterates through all files in the given directory, 
        calls load_document_file on each, and returns a combined list of Documents.
        """
        all_documents = []
        all_id_type_counts = Counter()
        
        # Iterate over items in the directory
        for item in os.listdir(dirname):
            item_path = os.path.join(dirname, item)
            
            # Only process files (skips subdirectories)
            if os.path.isfile(item_path):
                documents, id_type_counts = self._load_bioc_file(item_path)
                all_documents.extend(documents)
                all_id_type_counts.update(id_type_counts)
                
        logger.info(f"\tLoaded {len(all_documents)} documents")
        for id_type, count in all_id_type_counts.items():
            percentage = 100.0 * count / len(all_documents) if all_documents else 0.0
            logger.info(f"\tID type {id_type} present in {count} / {len(all_documents)} of documents ({percentage:.2f}%)")
        return all_documents, all_id_type_counts

    def _load_bioc_file(self, path: str) -> List[Document]:
        with open(path, "r", encoding="utf-8") as fp:
            collection = biocxml.load(fp)

        # NOTE: This version silently de-duplicates documents based on the document ID
        documents: dict[str, Document] = {}
        id_type_counts = Counter()
        for doc in collection.documents:
            passages: list[Passage] = []
            for passage_index, passage in enumerate(doc.passages):
                new_passage = Passage(
                    passage_id=str(passage.infons.get(self.passage_id_infon_key, passage_index)),
                    text=passage.text,
                    offset=int(passage.offset),
                    annotations=[],
                    infons=passage.infons,
                )
                for ann in passage.annotations:
                    mention = self.get_mention(ann)
                    if not mention is None:
                        new_passage.annotations.append(mention)
                passages.append(new_passage)

            document_id = str(doc.id)
            ids = self.doc_id_fetcher.get_IDs(doc)
            id_type_counts.update(ids.keys())
            documents[document_id] = Document(
                document_id=str(doc.id),
                identifiers=ids,
                passages=passages,
                infons=doc.infons,
            )

        return list(documents.values()), id_type_counts

    def get_mention(self, ann):
        spans = []
        for location in ann.locations:
            span = AnnotationSpan(
                start=int(location.offset),
                end=int(location.offset) + int(location.length),
            )
            spans.append(span)

        label = self.get_label(ann.infons.get(self.label_infon_key))
        if label is None:
            return None

        mention = Annotation(
            mention_id=str(ann.id),
            text=ann.text,
            spans=spans,
            label=label,
            link=self.get_identifier(ann.infons.get(self.id_infon_key)),
            attributes={k: str(v) for k, v in ann.infons.items()},
        )

        return mention


@register_loader("pubtator")
def load_pubtator(
    paths: dict[str, str] | None = None,
    path: str | None = None,
    split: dict | None = None,
    doc_id_map: dict[str, str] = {"pmid": "__DOCUMENT_ID__"},
    label_map: dict[str, str | None] = {},
    id_format_list: list[list[str]] = [],
    qualifier_map: dict[str, str] = {},
    nil_labels: set[str] = set(),
    default_resource: str | None = None,
    resource_delimiter: str = ":",
) -> BenchmarkCorpus:
    """
    Load a BioC XML file and convert it into the internal corpus model.

    Notes:
    - This minimal implementation uses only the first location for each annotation.
    - If your corpus uses discontinuous spans, extend this representation before production use.
    """
    loader = BioCPubtatorLoader(
        doc_id_map,
        label_map,
        parse_identifier_format_list(id_format_list),
        parse_qualifier_map(qualifier_map),
        nil_labels,
        default_resource,
        resource_delimiter,
    )
    load_paths = _resolve_load_paths(paths, path)
    subsets = dict()
    for subset_name, subset_path in load_paths.items():
        subsets[subset_name] = loader.load_subset(subset_name, subset_path)
    corpus = BenchmarkCorpus(
        subsets=subsets,
        metadata={
            "source_format": "Pubtator",
        },
    )
    return apply_document_split(corpus, split)


class BioCPubtatorLoader(Loader):

    def __init__(
        self,
        doc_id_map: dict[str, str] = {"pmid": "__DOCUMENT_ID__"},
        label_map: dict[str, str | None] = {},
        id_format_list: list[IdentifierFormat] = [],
        qualifier_map: dict[str, MatchType] = {},
        nil_labels: set[str] = set(),
        default_resource: str | None = None,
        resource_delimiter: str = ":",
    ):
        super().__init__(label_map, id_format_list, qualifier_map, nil_labels, default_resource, resource_delimiter)
        self.doc_id_fetcher = DocIDExtractor(self, doc_id_map)

    def load_subset(self, subset_name: str, path: str):
        """
        Load a BioC Pubtator file and convert it into the internal corpus model.
        """
        with open(path, "r", encoding="utf-8") as fp:
            collection = pubtator.load(fp)

        id_type_counts = Counter()
        documents: list[Document] = []

        for doc in collection:
            pmid = str(doc.pmid)
            title_passage = Passage(
                passage_id=f"{pmid}_t",
                text=doc.title,
                offset=0,
                annotations=[],
            )
            abstract_offset = len(doc.title) + 1
            abstract_passage = Passage(
                passage_id=f"{pmid}_a",
                text=doc.text,
                offset=abstract_offset,
                annotations=[],
            )
            for ann_index, ann in enumerate(doc.annotations):
                label = self.get_label(ann.type)
                if label is None:
                    continue
                mention = Annotation(
                    mention_id=str(ann_index + 1),
                    text=ann.text,
                    spans=[
                        AnnotationSpan(
                            start=ann.start,
                            end=ann.end,
                        )
                    ],
                    label=label,
                    link=self.get_identifier(ann.id),
                )
                if ann.start < abstract_offset:
                    title_passage.annotations.append(mention)
                else:
                    abstract_passage.annotations.append(mention)
            ids = self.doc_id_fetcher.get_IDs(doc)
            id_type_counts.update(ids.keys())
            documents.append(Document(document_id=pmid, identifiers=ids, passages=[title_passage, abstract_passage]))
        logger.info(f"\tLoaded {len(collection)} documents")
        for id_type, count in id_type_counts.items():
            percentage = 100.0 * count / len(collection)
            logger.info(f"\tID type {id_type} present in {count} / {len(collection)} of documents ({percentage:.2f}%)")
        return CorpusSubset(name=subset_name, documents=documents)
