from __future__ import annotations

import logging
from pathlib import Path
import xml.etree.ElementTree as ET

from corpus_benchmark.loaders.splits import apply_document_split
from corpus_benchmark.models.corpus import (
    Annotation,
    AnnotationSpan,
    BenchmarkCorpus,
    CorpusSubset,
    Document,
    DocumentIdentifierType,
    IdentifierLink,
    Passage,
)
from corpus_benchmark.models.types import MatchType
from corpus_benchmark.registry import register_loader

logger = logging.getLogger(__name__)


@register_loader("craft_knowtator")
def load_craft_knowtator(
    text_dir: str,
    annotation_dirs: dict[str, str],
    split: dict | None = None,
    id_mapping_path: str | None = None,
    text_suffix: str = ".txt",
    label_map: dict[str, str | None] = {},
) -> BenchmarkCorpus:
    """Load CRAFT concept annotations from Knowtator XML.

    CRAFT stores full article text separately from concept annotations, with
    each ontology in a separate Knowtator directory. The ontology name is used
    as the annotation label; the concrete ontology concept is stored as an
    IdentifierLink.
    """
    loader = CraftKnowtatorLoader(
        text_dir=Path(text_dir),
        annotation_dirs={name: Path(path) for name, path in annotation_dirs.items()},
        id_mapping_path=Path(id_mapping_path) if id_mapping_path else None,
        text_suffix=text_suffix,
        label_map=label_map,
    )
    corpus = BenchmarkCorpus(
        subsets={"all": loader.load_subset("all")},
        metadata={"source_format": "CRAFT Knowtator"},
    )
    return apply_document_split(corpus, split)


class CraftKnowtatorLoader:
    def __init__(
        self,
        *,
        text_dir: Path,
        annotation_dirs: dict[str, Path],
        id_mapping_path: Path | None = None,
        text_suffix: str = ".txt",
        label_map: dict[str, str | None] = {},
    ):
        self.text_dir = text_dir
        self.annotation_dirs = annotation_dirs
        self.id_mapping = _read_craft_id_mapping(id_mapping_path) if id_mapping_path else {}
        self.text_suffix = text_suffix
        self.label_map = label_map

    def load_subset(self, subset_name: str) -> CorpusSubset:
        if not self.text_dir.is_dir():
            raise ValueError(f"CRAFT text_dir is not a directory: {self.text_dir}")
        if not self.annotation_dirs:
            raise ValueError("CRAFT Knowtator loader requires at least one annotation directory")

        documents = []
        for text_path in sorted(self.text_dir.glob(f"*{self.text_suffix}")):
            doc_id = text_path.name[: -len(self.text_suffix)]
            documents.append(self.load_document(doc_id, text_path))

        annotation_count = sum(len(passage.annotations) for doc in documents for passage in doc.passages)
        logger.info(
            "Loaded CRAFT Knowtator subset %s with %s documents and %s annotations",
            subset_name,
            len(documents),
            annotation_count,
        )
        return CorpusSubset(name=subset_name, documents=documents)

    def load_document(self, doc_id: str, text_path: Path) -> Document:
        text = text_path.read_text(encoding="utf-8")
        annotations = []
        for ontology_name, annotation_dir in self.annotation_dirs.items():
            annotation_path = self._find_annotation_file(annotation_dir, doc_id, text_path.name)
            if annotation_path is None:
                logger.debug("No %s Knowtator annotations found for document %s", ontology_name, doc_id)
                continue
            annotations.extend(self._parse_knowtator_file(annotation_path, ontology_name, text))

        identifiers = {DocumentIdentifierType.PMID: DocumentIdentifierType.PMID.normalize(doc_id)}
        mapping = self.id_mapping.get(doc_id)
        if mapping:
            identifiers.update(mapping)

        return Document(
            document_id=doc_id,
            identifiers=identifiers,
            passages=[
                Passage(
                    passage_id=f"{doc_id}_text",
                    text=text,
                    offset=0,
                    annotations=annotations,
                )
            ],
        )

    def _find_annotation_file(self, annotation_dir: Path, doc_id: str, text_filename: str) -> Path | None:
        candidates = [
            annotation_dir / f"{text_filename}.knowtator.xml",
            annotation_dir / f"{doc_id}.knowtator.xml",
            annotation_dir / f"{doc_id}.xml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _parse_knowtator_file(self, path: Path, ontology_name: str, text: str) -> list[Annotation]:
        root = ET.parse(path).getroot()
        if root.tag == "knowtator-project":
            return self._parse_knowtator2(root, ontology_name, text, path)
        return self._parse_knowtator1(root, ontology_name, text, path)

    def _parse_knowtator1(self, root: ET.Element, ontology_name: str, text: str, path: Path) -> list[Annotation]:
        class_mentions = {}
        for class_mention in root.findall("./classMention"):
            mention_class = class_mention.find("./mentionClass")
            if mention_class is None:
                continue
            class_mentions[class_mention.attrib.get("id")] = {
                "identifier": mention_class.attrib.get("id"),
                "concept_label": (mention_class.text or "").strip() or None,
            }

        annotations = []
        for annotation_el in root.findall("./annotation"):
            mention_el = annotation_el.find("./mention")
            mention_id = mention_el.attrib.get("id") if mention_el is not None else annotation_el.attrib.get("id")
            if not mention_id:
                logger.warning("Skipping Knowtator annotation without mention id in %s", path)
                continue

            spans = _parse_span_elements(annotation_el.findall("./span"), path, mention_id)
            if not spans:
                continue

            concept = class_mentions.get(mention_id, {})
            spanned_text = annotation_el.findtext("./spannedText")
            annotations.append(
                self._make_annotation(
                    mention_id=mention_id,
                    ontology_name=ontology_name,
                    spans=spans,
                    text=spanned_text or _text_for_spans(text, spans),
                    concept_identifier=concept.get("identifier"),
                    concept_label=concept.get("concept_label"),
                )
            )
        return annotations

    def _parse_knowtator2(self, root: ET.Element, ontology_name: str, text: str, path: Path) -> list[Annotation]:
        annotations = []
        for annotation_el in root.findall(".//annotation"):
            mention_id = annotation_el.attrib.get("id")
            if not mention_id:
                logger.warning("Skipping Knowtator-2 annotation without id in %s", path)
                continue

            spans = _parse_span_elements(annotation_el.findall("./span"), path, mention_id)
            if not spans:
                continue

            class_el = annotation_el.find("./class")
            concept_identifier = class_el.attrib.get("id") if class_el is not None else None
            concept_label = class_el.attrib.get("label") if class_el is not None else None
            span_text = " ... ".join((span_el.text or "").strip() for span_el in annotation_el.findall("./span") if span_el.text)

            annotations.append(
                self._make_annotation(
                    mention_id=mention_id,
                    ontology_name=ontology_name,
                    spans=spans,
                    text=span_text or _text_for_spans(text, spans),
                    concept_identifier=concept_identifier,
                    concept_label=concept_label,
                )
            )
        return annotations

    def get_label(self, label_text) -> str:
        label = self.label_map.get(label_text, label_text)
        if label is None:
            return None
        return str(label)

    def _make_annotation(
        self,
        *,
        mention_id: str,
        ontology_name: str,
        spans: list[AnnotationSpan],
        text: str,
        concept_identifier: str | None,
        concept_label: str | None,
    ) -> Annotation:
        return Annotation(
            mention_id=mention_id,
            text=text,
            spans=spans,
            label=self.get_label(ontology_name),
            link=IdentifierLink(
                resource=ontology_name,
                identifier=_normalize_concept_identifier(concept_identifier),
                match_type=MatchType.EXACT,
            ),
            attributes={
                "ontology": ontology_name,
                "concept_id": concept_identifier or "",
                "concept_label": _clean_label(concept_label) or "",
            },
        )


def _parse_span_elements(span_elements: list[ET.Element], path: Path, mention_id: str) -> list[AnnotationSpan]:
    spans = []
    for span_el in span_elements:
        try:
            start = int(span_el.attrib["start"])
            end = int(span_el.attrib["end"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid span for annotation {mention_id!r} in {path}: {ET.tostring(span_el)}") from exc
        if start < 0 or end < start:
            raise ValueError(f"Invalid span [{start}, {end}) for annotation {mention_id!r} in {path}")
        if end == start:
            logger.warning("Skipping zero-length span for annotation %s in %s", mention_id, path)
            continue
        spans.append(AnnotationSpan(start=start, end=end))
    return spans


def _normalize_concept_identifier(identifier: str | None) -> str | None:
    if not identifier:
        return None
    if "/" in identifier:
        identifier = identifier.rsplit("/", 1)[-1]
    if "#" in identifier:
        identifier = identifier.rsplit("#", 1)[-1]
    return identifier.replace("_", ":")


def _clean_label(label: str | None) -> str | None:
    if label is None:
        return None
    return label.strip().strip("'\"") or None


def _text_for_spans(text: str, spans: list[AnnotationSpan]) -> str:
    return " ... ".join(text[span.start : span.end] for span in spans)


def _read_craft_id_mapping(path: Path | None) -> dict[str, dict[DocumentIdentifierType, str]]:
    if path is None:
        return {}

    mapping: dict[str, dict[DocumentIdentifierType, str]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(f"Expected at least 3 columns in CRAFT ID mapping line {line_index}: {line!r}")
            _, pmcid, pmid = fields[:3]
            mapping[pmid] = {
                DocumentIdentifierType.PMID: DocumentIdentifierType.PMID.normalize(pmid),
                DocumentIdentifierType.PMCID: DocumentIdentifierType.PMCID.normalize(pmcid),
            }
    return mapping
