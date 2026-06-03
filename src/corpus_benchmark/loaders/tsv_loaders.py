from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

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
from corpus_benchmark.registry import register_loader


def _resolve_load_paths(paths: dict[str, str] | None, path: str | None) -> dict[str, str]:
    if paths and path:
        raise ValueError("Configure either loader.params.paths or loader.params.path, not both")
    if paths:
        return paths
    if path:
        return {"all": path}
    raise ValueError("Loader requires either loader.params.paths or loader.params.path")


@register_loader("linnaeus_species")
def load_linnaeus_species(
    text_dir: str,
    tags_path: str,
    split: dict | None = None,
    default_resource: str = "NCBITaxon",
) -> BenchmarkCorpus:
    tag_rows: dict[str, list[Annotation]] = defaultdict(list)
    with Path(tags_path).open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 5:
                raise ValueError(f"LINNAEUS tag line {line_index} should have at least 5 columns: {line!r}")
            entity_id, document_id, raw_start, raw_end, mention_text = fields[:5]
            taxon_id = entity_id.rsplit(":", 1)[-1]
            annotation_index = len(tag_rows[document_id]) + 1
            tag_rows[document_id].append(
                Annotation(
                    mention_id=f"T{annotation_index}",
                    text=mention_text,
                    spans=[AnnotationSpan(start=int(raw_start), end=int(raw_end))],
                    label="Species",
                    link=IdentifierLink(resource=default_resource, identifier=taxon_id),
                    attributes={"source_entity_id": entity_id},
                )
            )

    documents: list[Document] = []
    for text_path in sorted(Path(text_dir).glob("*.txt")):
        document_id = text_path.stem
        pmcid = document_id
        if document_id.lower().startswith("pmca"):
            pmcid = document_id[4:]
        text = text_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                document_id=document_id,
                identifiers={DocumentIdentifierType.PMCID: DocumentIdentifierType.PMCID.normalize(pmcid)},
                passages=[
                    Passage(
                        passage_id=f"{document_id}_text",
                        text=text,
                        offset=0,
                        annotations=tag_rows.get(document_id, []),
                    )
                ],
            )
        )

    corpus = BenchmarkCorpus(
        subsets={"all": CorpusSubset(name="all", documents=documents)},
        metadata={"source_format": "LINNAEUS TSV"},
    )
    return apply_document_split(corpus, split)


@register_loader("sentence_label_tsv")
def load_sentence_label_tsv(
    paths: dict[str, str] | None = None,
    path: str | None = None,
    split: dict | None = None,
    docid_type: str = "pmid",
) -> BenchmarkCorpus:
    id_type = DocumentIdentifierType(docid_type.lower())
    subsets: dict[str, CorpusSubset] = {}
    for subset_name, subset_path in _resolve_load_paths(paths, path).items():
        documents = [_load_sentence_label_document(text_path, id_type) for text_path in sorted(Path(subset_path).glob("*.txt"))]
        subsets[subset_name] = CorpusSubset(name=subset_name, documents=documents)

    corpus = BenchmarkCorpus(
        subsets=subsets,
        metadata={"source_format": "sentence label TSV"},
    )
    return apply_document_split(corpus, split)


def _load_sentence_label_document(path: Path, id_type: DocumentIdentifierType) -> Document:
    document_id = path.stem
    text_parts: list[str] = []
    annotations: list[Annotation] = []
    offset = 0
    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                sentence, labels_text = line.rsplit("\t", 1)
            except ValueError as exc:
                raise ValueError(f"Sentence label file {path} line {line_index} should contain sentence<TAB>labels") from exc
            labels = ast.literal_eval(labels_text)
            if not isinstance(labels, list):
                raise ValueError(f"Sentence label file {path} line {line_index} labels should be a list: {labels_text!r}")
            sentence_start = offset
            sentence_end = sentence_start + len(sentence)
            text_parts.append(sentence)
            offset = sentence_end + 1
            for label in labels:
                annotations.append(
                    Annotation(
                        mention_id=f"T{len(annotations) + 1}",
                        text=sentence,
                        spans=[AnnotationSpan(start=sentence_start, end=sentence_end)],
                        label=str(label),
                        link=None,
                    )
                )

    return Document(
        document_id=document_id,
        identifiers={id_type: id_type.normalize(document_id)},
        passages=[
            Passage(
                passage_id=f"{document_id}_sentences",
                text="\n".join(text_parts),
                offset=0,
                annotations=annotations,
            )
        ],
    )


@register_loader("mutationfinder")
def load_mutationfinder(
    paths: dict[str, str],
    gold_paths: dict[str, str],
    split: dict | None = None,
    default_resource: str = "MutationFinder",
) -> BenchmarkCorpus:
    subsets: dict[str, CorpusSubset] = {}
    for subset_name, text_path in paths.items():
        if subset_name not in gold_paths:
            raise ValueError(f"MutationFinder loader is missing gold path for subset {subset_name!r}")
        subsets[subset_name] = _load_mutationfinder_subset(
            subset_name,
            Path(text_path),
            Path(gold_paths[subset_name]),
            default_resource,
        )

    corpus = BenchmarkCorpus(
        subsets=subsets,
        metadata={"source_format": "MutationFinder corpus"},
    )
    return apply_document_split(corpus, split)


def _load_mutationfinder_subset(
    subset_name: str,
    text_path: Path,
    gold_path: Path,
    default_resource: str,
) -> CorpusSubset:
    documents: dict[str, Document] = {}
    for line_index, line in enumerate(text_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            pmid, title, abstract = line.split("\t", 2)
        except ValueError as exc:
            raise ValueError(f"MutationFinder text line {line_index} should contain PMID<TAB>title<TAB>abstract") from exc
        title_passage = Passage(passage_id=f"{pmid}_t", text=title, offset=0, annotations=[])
        abstract_offset = len(title) + 1
        abstract_passage = Passage(passage_id=f"{pmid}_a", text=abstract, offset=abstract_offset, annotations=[])
        documents[pmid] = Document(
            document_id=pmid,
            identifiers={DocumentIdentifierType.PMID: DocumentIdentifierType.PMID.normalize(pmid)},
            passages=[title_passage, abstract_passage],
        )

    for line_index, line in enumerate(gold_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        pmid = fields[0]
        document = documents.get(pmid)
        if document is None:
            raise ValueError(f"MutationFinder gold line {line_index} references unknown PMID {pmid!r}")
        full_text = "\n".join(passage.text for passage in document.passages)
        search_start = 0
        for mutation in [field for field in fields[1:] if field]:
            mention_id = f"T{sum(len(p.annotations) for p in document.passages) + 1}"
            span = _find_mutation_span(full_text, mutation, search_start)
            if span is None:
                target_passage = document.passages[1]
                spans: list[AnnotationSpan] = []
            else:
                start, end = span
                search_start = end
                target_passage = document.passages[0] if start < len(document.passages[0].text) else document.passages[1]
                spans = [AnnotationSpan(start=start, end=end)]
            target_passage.annotations.append(
                Annotation(
                    mention_id=mention_id,
                    text=mutation,
                    spans=spans,
                    label="SequenceVariant",
                    link=IdentifierLink(resource=default_resource, identifier=mutation),
                )
            )

    return CorpusSubset(name=subset_name, documents=list(documents.values()))


def _find_mutation_span(text: str, mutation: str, search_start: int) -> tuple[int, int] | None:
    index = text.find(mutation, search_start)
    if index < 0:
        index = text.find(mutation)
    if index < 0:
        return None
    return index, index + len(mutation)
