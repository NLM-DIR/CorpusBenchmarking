"""Loader for simple BRAT/standoff-style biomedical annotation corpora.

This module currently targets the JNLPBA standoff distribution prepared by
Sampo Pyysalo (https://github.com/spyysalo/jnlpba). In that distribution,
each document is represented by two files in the same directory:

* ``<document_id>.txt``: exactly two lines, where line 1 is the title and
  line 2 is the abstract.
* ``<document_id>.ann``: tab-delimited standoff annotations in the standard
  simple BRAT form::

      T1\tprotein 0 7\tExample

Offsets in the annotation file are document-level character offsets over the
concatenated text ``title + "\n" + abstract``. The internal ``Passage`` objects
therefore retain document-level offsets: the title starts at 0 and the abstract
starts at ``len(title) + 1``.
"""

from __future__ import annotations

import logging
import re
from abc import abstractmethod
from pathlib import Path

from corpus_benchmark.models.corpus import (
    Annotation,
    AnnotationSpan,
    BenchmarkCorpus,
    CompositeLink,
    CorpusSubset,
    Document,
    DocumentIdentifierType,
    IdentifierLink,
    Link,
    Passage,
)
from corpus_benchmark.models.types import LinkRelation
from corpus_benchmark.loaders.bioc_loader import Loader
from corpus_benchmark.loaders.splits import apply_document_split
from corpus_benchmark.registry import register_loader

logger = logging.getLogger(__name__)


def _resolve_load_paths(paths: dict[str, str] | None, path: str | None) -> dict[str, str]:
    if paths and path:
        raise ValueError("Configure either loader.params.paths or loader.params.path, not both")
    if paths:
        return paths
    if path:
        return {"all": path}
    raise ValueError("Loader requires either loader.params.paths or loader.params.path")


def read_docid_map(filename: str | Path) -> dict[str, str]:
    """Read a two-column, tab-delimited document identifier map.

    Parameters
    ----------
    filename:
        Path to a text file containing ``source_id<TAB>target_id`` rows.
        Blank lines are ignored.

    Returns
    -------
    dict[str, str]
        Mapping from the identifier used in the standoff file names to the
        desired external identifier, such as a PMID.
    """
    path = Path(filename)
    docid_map: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Expected exactly 2 columns on line " f'{line_index} of file "{path}" but got {len(fields)}: ' f'"{line}"')

            docid_from, docid_to = fields
            docid_map[docid_from.strip()] = docid_to.strip()

    return docid_map


@register_loader("JNLPBA_standoff")
def load_JNLPBA_standoff(
    MUID_PMID_map_path: str,
    paths: dict[str, str] | None = None,
    path: str | None = None,
    split: dict | None = None,
    label_map: dict[str, str | None] = {},
) -> BenchmarkCorpus:
    """Load the JNLPBA standoff corpus into the benchmark corpus model.

    Parameters
    ----------
    paths:
        Mapping from subset name (for example, ``"train"`` or ``"test"``)
        to a directory containing paired ``.txt`` and ``.ann`` files.
    MUID_PMID_map_path:
        Path to a two-column mapping file from the JNLPBA/Medline UI document
        identifiers to the desired external identifiers.
    label_map:
        Optional mapping from source annotation labels to normalized labels.
        A value of ``None`` suppresses that label.
    """
    docid_map = read_docid_map(MUID_PMID_map_path)
    # logger.debug(f"TRACE: len(docid_map) = {len(docid_map)}")

    loader = JNLPBA_StandoffLoader(
        docid_map=docid_map,
        label_map=label_map,
    )

    load_paths = _resolve_load_paths(paths, path)
    subsets = {subset_name: loader.load_subset(subset_name, subset_path) for subset_name, subset_path in load_paths.items()}

    corpus = BenchmarkCorpus(
        subsets=subsets,
        metadata={"source_format": "JNLPBA standoff"},
    )
    return apply_document_split(corpus, split)


@register_loader("AnatEM_standoff")
def load_AnatEM_standoff(
    paths: dict[str, str] | None = None,
    path: str | None = None,
    split: dict | None = None,
    label_map: dict[str, str | None] = {},
) -> BenchmarkCorpus:
    """Load the JNLPBA standoff corpus into the benchmark corpus model.

    Parameters
    ----------
    paths:
        Mapping from subset name (for example, ``"train"`` or ``"test"``)
        to a directory containing paired ``.txt`` and ``.ann`` files.
    label_map:
        Optional mapping from source annotation labels to normalized labels.
        A value of ``None`` suppresses that label.
    """
    loader = AnatEM_StandoffLoader(
        label_map=label_map,
    )

    load_paths = _resolve_load_paths(paths, path)
    subsets = {subset_name: loader.load_subset(subset_name, subset_path) for subset_name, subset_path in load_paths.items()}

    corpus = BenchmarkCorpus(
        subsets=subsets,
        metadata={"source_format": "AnatEM standoff"},
    )
    return apply_document_split(corpus, split)


@register_loader("BRAT_standoff")
def load_BRAT_standoff(
    paths: dict[str, str] | None = None,
    path: str | None = None,
    split: dict | None = None,
    label_map: dict[str, str | None] = {},
    docid_type: str = "pmid",
    docid_regex: str | None = None,
    docid_group: int | str = 1,
    docid_map_path: str | None = None,
    docid_map_invert: bool = False,
    normalization_resource_map: dict[str, str] = {},
    verify_text: bool = True,
) -> BenchmarkCorpus:
    """Load a generic BRAT standoff corpus into the benchmark corpus model.

    This loader supports standard text-bound rows and optional BRAT
    normalization rows, for example ``N1\tReference T1 Taxonomy:9606``.
    """
    docid_map = read_docid_map(docid_map_path) if docid_map_path else {}
    if docid_map_invert:
        docid_map = {mapped_id: source_id for source_id, mapped_id in docid_map.items()}

    loader = BRAT_StandoffLoader(
        label_map=label_map,
        docid_type=docid_type,
        docid_regex=docid_regex,
        docid_group=docid_group,
        docid_map=docid_map,
        normalization_resource_map=normalization_resource_map,
        verify_text=verify_text,
    )

    load_paths = _resolve_load_paths(paths, path)
    subsets = {subset_name: loader.load_subset(subset_name, subset_path) for subset_name, subset_path in load_paths.items()}

    corpus = BenchmarkCorpus(
        subsets=subsets,
        metadata={"source_format": "BRAT standoff"},
    )
    return apply_document_split(corpus, split)


class StandoffLoader(Loader):
    """Load a directory of simple standoff annotation files.

    The loader follows the same high-level model as the BioC loaders:
    each subset becomes a ``CorpusSubset``; each text/annotation file pair
    becomes a ``Document``; the title and abstract become separate passages;
    and each standoff row becomes an ``Annotation`` attached to the passage
    containing its span.
    """

    def __init__(
        self,
        label_map: dict[str, str | None] = {},
        normalization_resource_map: dict[str, str] = {},
        verify_text: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the standoff loader.

        Parameters
        ----------
        docid_type:
            External identifier type to store in each ``Document.identifiers``
            entry, such as PMID.
        docid_map:
            Optional mapping from file-stem document IDs to external IDs.
            Unmapped document IDs are retained as-is.
        label_map:
            Optional mapping from source labels to normalized labels. Labels
            mapped to ``None`` are skipped.
        **kwargs:
            Accepted for compatibility with the generic loader factory pattern;
            currently unused.
        """
        super().__init__(label_map=label_map)
        self.normalization_resource_map = normalization_resource_map
        self.verify_text = verify_text

    @abstractmethod
    def get_ids(self, docid: str) -> dict[DocumentIdentifierType, str]:
        """Return normalized document identifiers for one source document ID."""
        pass

    def load_subset(self, subset_name: str, path: str | Path) -> CorpusSubset:
        """Load all ``.txt``/``.ann`` pairs in a directory.

        Files are processed in lexicographic order for deterministic output.
        Each ``.txt`` file must have a matching ``.ann`` file with the same
        stem. Extra ``.ann`` files are ignored, because the text file defines
        the document set.
        """
        subset_path = Path(path)
        if not subset_path.is_dir():
            raise ValueError(f'Standoff subset path is not a directory: "{subset_path}"')

        logger.info(f"Loading subset {subset_name} from {subset_path}")

        documents: list[Document] = []
        annotation_count = 0

        file_paths = [f for f in subset_path.glob("*.txt") if not f.name.startswith(".")]
        for text_path in sorted(file_paths):
            document_id = text_path.stem
            annotation_path = Path(path) / f"{document_id}.ann"
            doc = self.load_document(document_id, text_path, annotation_path)
            annotation_count += sum(len(p.annotations) for p in doc.passages)
            documents.append(doc)

        logger.info(f"\tLoaded {len(documents)} documents and {annotation_count} annotations")
        return CorpusSubset(name=subset_name, documents=documents)

    def load_document(self, document_id, text_path, annotation_path) -> Document:
        try:
            text = Path(text_path).read_text(encoding="utf-8")
        except Exception as e:
            raise Exception(f"Unable to load document {text_path} due to {e}")

        passage = Passage(
            passage_id=f"{document_id}_t",
            text=text,
            offset=0,
            annotations=[],
        )

        annotations = self.load_annotations(document_id, annotation_path)
        for annotation in annotations:
            if self.verify_text:
                self.verify_annotation_text(annotation, passage, document_id)
            passage.annotations.append(annotation)

        doc = Document(
            document_id=document_id,
            identifiers=self.get_ids(document_id),
            passages=[passage],
        )
        return doc

    @staticmethod
    def _annotation_span(annotation: Annotation) -> AnnotationSpan:
        """Return the single span for loaders that only support simple spans."""
        if len(annotation.spans) != 1:
            raise ValueError(f"Expected exactly one span for annotation {annotation.mention_id!r}, but found {len(annotation.spans)}.")
        return annotation.spans[0]

    def verify_annotation_text(
        self,
        annotation: Annotation,
        passage: Passage,
        document_id: str,
    ) -> None:
        """Verify that the standoff span extracts the recorded mention text."""
        span_texts = []
        for span in annotation.spans:
            local_start = span.start - passage.offset
            local_end = span.end - passage.offset
            span_texts.append(passage.text[local_start:local_end])
        extracted_text = " ".join(span_texts)

        if extracted_text != annotation.text and "".join(span_texts) != annotation.text:
            raise ValueError(
                f"Annotation text mismatch in document {document_id!r}, "
                f"annotation {annotation.mention_id!r}: spans extract {extracted_text!r}, but annotation file "
                f"contains {annotation.text!r}."
            )

    def load_annotations(self, document_id: str, annotation_path: Path) -> list[Annotation]:
        """Load simple standoff annotations for one document.

        Only text-bound annotations are expected. The method intentionally
        rejects discontinuous spans (for example, ``0 4;8 12``), attributes,
        events, and other BRAT extensions because the JNLPBA standoff files
        used here consist of one continuous span per line.
        """
        if not annotation_path.is_file():
            raise ValueError(f'Missing standoff annotation file for document "{document_id}": ' f'"{annotation_path}"')

        annotations: list[Annotation] = []
        annotations_by_id: dict[str, Annotation] = {}
        pending_links: dict[str, list[IdentifierLink]] = {}
        with annotation_path.open("r", encoding="utf-8") as file:
            for line_index, line in enumerate(file, start=1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue

                fields = line.split("\t")
                annotation_kind = fields[0][:1]
                if annotation_kind == "T":
                    if len(fields) != 3:
                        raise ValueError(f"Expected exactly 3 tab-delimited fields on line " f"{line_index} of standoff annotation file " f'{annotation_path}: "{line}"')

                    mention_id, span_descriptor, mention_text = fields
                    label, spans = self.parse_span_descriptor(
                        span_descriptor=span_descriptor,
                        annotation_path=annotation_path,
                        line_index=line_index,
                        line=line,
                    )

                    # Label maps may intentionally suppress labels by mapping
                    # them to None. This mirrors the behavior of the BioC
                    # loaders.
                    if label is None:
                        continue

                    annotation = Annotation(
                        mention_id=str(mention_id),
                        text=str(mention_text),
                        spans=spans,
                        label=label,
                        link=None,
                    )
                    annotations.append(annotation)
                    annotations_by_id[str(mention_id)] = annotation
                    if mention_id in pending_links:
                        annotation.link = self._combine_links(pending_links.pop(mention_id))
                    continue

                if annotation_kind == "N":
                    target_id, link = self.parse_normalization_descriptor(
                        fields=fields,
                        annotation_path=annotation_path,
                        line_index=line_index,
                        line=line,
                    )
                    annotation = annotations_by_id.get(target_id)
                    if annotation is None:
                        pending_links.setdefault(target_id, []).append(link)
                    else:
                        current_links = annotation.link.get_identifier_links() if annotation.link is not None else []
                        annotation.link = self._combine_links([*current_links, link])
                    continue

                # Ignore BRAT attributes, notes, relations, and events. They
                # are not part of the corpus model used by these metrics.

        return annotations

    def parse_span_descriptor(
        self,
        span_descriptor: str,
        annotation_path: Path,
        line_index: int,
        line: str,
    ) -> tuple[str | None, list[AnnotationSpan]]:
        """Parse the middle field of a simple standoff annotation row.

        The expected format is ``<label> <start> <end>`` or a BRAT
        discontinuous span such as ``<label> <start> <end>;<start> <end>``.
        """
        try:
            raw_label, span_text = span_descriptor.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Expected a label and span offsets in the center " f"field on line {line_index} of standoff annotation file " f'{annotation_path}: "{line}"') from exc

        spans: list[AnnotationSpan] = []
        for raw_span in span_text.split(";"):
            fields = raw_span.split()
            if len(fields) != 2:
                raise ValueError(f"Expected start and end offsets in span {raw_span!r} on line " f"{line_index} of standoff annotation file {annotation_path}: " f"{line!r}")
            raw_start, raw_end = fields
            try:
                span_start = int(raw_start)
                span_end = int(raw_end)
            except ValueError as exc:
                raise ValueError(f"Could not parse integer offsets on line {line_index} of " f"standoff annotation file {annotation_path}: {line!r}") from exc

            if span_start < 0 or span_end <= span_start:
                raise ValueError(f"Invalid span [{span_start}, {span_end}) on line " f"{line_index} of standoff annotation file {annotation_path}: " f"{line!r}")
            spans.append(AnnotationSpan(start=span_start, end=span_end))

        return self.get_label(raw_label), spans

    def parse_normalization_descriptor(
        self,
        fields: list[str],
        annotation_path: Path,
        line_index: int,
        line: str,
    ) -> tuple[str, IdentifierLink]:
        if len(fields) < 2:
            raise ValueError(f"Expected a normalization descriptor on line " f"{line_index} of standoff annotation file " f'{annotation_path}: "{line}"')

        descriptor_fields = fields[1].split()
        if len(descriptor_fields) < 3:
            raise ValueError(f"Expected normalization format 'Reference <target> <resource>:<id>' " f"on line {line_index} of standoff annotation file " f'{annotation_path}: "{line}"')

        target_id = descriptor_fields[1]
        resource_and_id = descriptor_fields[2]
        if ":" in resource_and_id:
            resource, identifier = resource_and_id.split(":", 1)
        else:
            resource, identifier = None, resource_and_id
        mapped_resource = self.normalization_resource_map.get(resource, resource)
        return target_id, IdentifierLink(resource=mapped_resource, identifier=identifier)

    @staticmethod
    def _combine_links(links: list[IdentifierLink]) -> Link | None:
        if not links:
            return None
        if len(links) == 1:
            return links[0]
        return CompositeLink(relation=LinkRelation.RELATED_SET, components=list(links))


class JNLPBA_StandoffLoader(StandoffLoader):

    def __init__(
        self,
        docid_map: dict[str, str] = {},
        label_map: dict[str, str | None] = {},
        **kwargs,
    ) -> None:
        super().__init__(label_map=label_map, kwargs=kwargs)
        self.docid_map = docid_map

    def get_ids(self, filename_docid: str) -> dict[DocumentIdentifierType, str]:
        """Return normalized document identifiers for one source document ID."""
        # NOTE JNLPBA uses "-2" for PMIDs that appear twice;
        docid_fields = filename_docid.split("-")
        docid = docid_fields[0]
        docid_mapped = self.docid_map.get(docid)
        if not docid_mapped:
            raise ValueError(f"docid {docid} from filename_docid {filename_docid} not found in docid_map")
        ids = {DocumentIdentifierType.PMID: DocumentIdentifierType.PMID.normalize(docid_mapped)}
        # logger.debug(f"TRACE ids for {docid} = {ids}")
        return ids


class AnatEM_StandoffLoader(StandoffLoader):
    def __init__(
        self,
        label_map: dict[str, str | None] = {},
        **kwargs,
    ) -> None:
        super().__init__(label_map=label_map, kwargs=kwargs)

    def get_ids(self, filename_docid: str) -> dict[DocumentIdentifierType, str]:
        """Return normalized document identifiers for one source document ID."""
        # AnatEM PMID files look like "PMID-8643685.txt"
        if filename_docid.startswith("PMID-"):
            return self.get_typed_ids(DocumentIdentifierType.PMID, filename_docid)
        # AnatEM PMC files look like "PMC-2811192-caption-03.txt"
        if filename_docid.startswith("PMC-"):
            return self.get_typed_ids(DocumentIdentifierType.PMCID, filename_docid)
        raise ValueError(f"Unknown filename docid format: {filename_docid}")

    def get_typed_ids(self, docid_type: DocumentIdentifierType, filename_docid: str) -> dict[DocumentIdentifierType, str]:
        docid_fields = filename_docid.split("-")
        docid = docid_type.normalize(docid_fields[1])
        ids = {docid_type: docid}
        # logger.debug(f"TRACE ids for {docid} = {ids}")
        return ids


class BRAT_StandoffLoader(StandoffLoader):
    def __init__(
        self,
        label_map: dict[str, str | None] = {},
        docid_type: str = "pmid",
        docid_regex: str | None = None,
        docid_group: int | str = 1,
        docid_map: dict[str, str] = {},
        normalization_resource_map: dict[str, str] = {},
        verify_text: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            label_map=label_map,
            normalization_resource_map=normalization_resource_map,
            verify_text=verify_text,
            kwargs=kwargs,
        )
        self.docid_type = DocumentIdentifierType(docid_type.lower())
        self.docid_regex = re.compile(docid_regex) if docid_regex else None
        self.docid_group = docid_group
        self.docid_map = docid_map

    def get_ids(self, filename_docid: str) -> dict[DocumentIdentifierType, str]:
        docid_text = self.docid_map.get(filename_docid, filename_docid)
        if self.docid_regex is not None:
            match = self.docid_regex.search(docid_text)
            if match is None:
                raise ValueError(f"Document ID {filename_docid!r} resolved to {docid_text!r}, which did not match configured docid_regex")
            docid_text = match.group(self.docid_group)
        docid = self.docid_type.normalize(docid_text)
        return {self.docid_type: docid}
