from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict

from corpus_benchmark.models.corpus import CorpusSubset, Document, Passage, Annotation, IdentifierLink
from corpus_benchmark.models.filters import AnnotationFilter
from corpus_benchmark.models.terminologies import (
    TerminologyResource,
    TerminologyTopicAnchorCounter,
    load_topic_term_overrides,
)
from corpus_benchmark.workspace import GlobalWorkspace
from utils.text_utils import extract_tokens_from_texts, extract_sentences_from_texts

logger = logging.getLogger(__name__)

PRECISION = 8


@dataclass(slots=True)
class BenchmarkContext:
    """Shared context for metrics, including a cache for expensive computations."""

    workspace: GlobalWorkspace
    cache: dict[str, Any] = field(default_factory=dict)
    usage_counts: Counter[str:int] = field(default_factory=Counter)
    annotation_filters: dict[str, AnnotationFilter] = field(default_factory=dict)

    def get_filter(self, filter_name: str | None) -> AnnotationFilter | None:
        if filter_name is None:
            return None
        if filter_name not in self.annotation_filters:
            available = ", ".join(sorted(self.annotation_filters)) or "<none>"
            raise ValueError(f"Unknown annotation filter '{filter_name}'. " f"Available filters: {available}")
        return self.annotation_filters[filter_name]

    def get_or_compute(self, key: str, factory: Callable[[], Any]) -> Any:
        self.usage_counts[key] += 1
        if key not in self.cache:
            logger.debug("Cache miss for %s", key)
            self.cache[key] = factory()
        else:
            logger.debug("Cache hit for %s", key)
        return self.cache[key]


def get_workspace(target: MetricTarget) -> GlobalWorkspace:
    """Extracts the global workspace from the target's first component context."""
    if not target.components:
        raise ValueError("Cannot extract workspace from an empty MetricTarget")
    return target.components[0][1].workspace


@dataclass(slots=True)
class MetricTarget:
    """Wraps multiple CorpusSubsets and their Contexts to appear as a single target."""

    # TODO Move workspace out of BenchmarkContext to here
    name: str
    components: list[tuple[CorpusSubset, BenchmarkContext]] = field(default_factory=list)


class SingleMetricTarget(MetricTarget):
    """Defines a single-item mock target to re-use cached items"""

    def __init__(self, subset: CorpusSubset, context: BenchmarkContext):
        super().__init__(name=None, components=[(subset, context)])


def get_documents(target: MetricTarget) -> list[Document]:
    documents = []
    for subset, context in target.components:
        subset_docs = context.get_or_compute(f"documents({subset.name})", lambda: list(subset.documents))
        documents.extend(subset_docs)
    return documents


def get_passages(target: MetricTarget) -> list[Passage]:
    passages = []
    for subset, context in target.components:
        subset_passages = context.get_or_compute(
            f"passages({subset.name})",
            lambda: [p for d in get_documents(SingleMetricTarget(subset, context)) for p in d.passages],
        )
        passages.extend(subset_passages)
    return passages


def get_sentences(target: MetricTarget) -> list[str]:
    sentences = []
    for subset, context in target.components:
        subset_sentences = context.get_or_compute(
            f"sentences({subset.name})",
            lambda: extract_sentences_from_texts(
                [passage.text for passage in get_passages(SingleMetricTarget(subset, context))]
            ),
        )
        sentences.extend(subset_sentences)
    return sentences


def get_tokens(target: MetricTarget) -> list[str]:
    tokens = []
    for subset, context in target.components:
        subset_tokens = context.get_or_compute(
            f"tokens({subset.name})",
            lambda: extract_tokens_from_texts(
                [passage.text for passage in get_passages(SingleMetricTarget(subset, context))]
            ),
        )
        tokens.extend(subset_tokens)
    return tokens


def get_annotations(target: MetricTarget, annotation_filter_name: str | None = None) -> list[Annotation]:
    annotations = []
    for subset, context in target.components:
        annotation_filter = context.get_filter(annotation_filter_name)
        if annotation_filter is None:
            subset_annotations = context.get_or_compute(
                f"annotations({subset.name}, {annotation_filter_name})",
                lambda: [
                    annotation
                    for passage in get_passages(SingleMetricTarget(subset, context))
                    for annotation in passage.annotations
                ],
            )
        else:
            subset_annotations = context.get_or_compute(
                f"annotations({subset.name}, {annotation_filter_name})",
                lambda: annotation_filter.filter_annotations(
                    [
                        annotation
                        for passage in get_passages(SingleMetricTarget(subset, context))
                        for annotation in passage.annotations
                    ]
                ),
            )
        annotations.extend(subset_annotations)
    return annotations


def _internal_get_annotations_per_document(
    subset: CorpusSubset, context: BenchmarkContext, annotation_filter_name: str | None = None
) -> list[Annotation]:
    annotations_per_document = []
    annotation_filter = context.get_filter(annotation_filter_name)
    for document in get_documents(SingleMetricTarget(subset, context)):
        annotations_for_document = [annotation for passage in document.passages for annotation in passage.annotations]
        if not annotation_filter is None:
            annotations_for_document = annotation_filter.filter_annotations(annotations_for_document)
        annotations_per_document.append(annotations_for_document)
    return annotations_per_document


def get_annotations_per_document(target: MetricTarget, annotation_filter_name: str | None = None) -> list[Annotation]:
    annotations_per_document = []
    for subset, context in target.components:
        annotations_per_document.extend(
            context.get_or_compute(
                f"annotations_per_document({subset.name}, {annotation_filter_name})",
                lambda: _internal_get_annotations_per_document(subset, context, annotation_filter_name),
            )
        )
    return annotations_per_document


def get_labels(target: MetricTarget, annotation_filter_name: str | None = None) -> list[str]:
    labels = []
    for subset, context in target.components:
        subset_labels = context.get_or_compute(
            f"labels({subset.name}, {annotation_filter_name})",
            lambda: [
                annotation.label
                for annotation in get_annotations(SingleMetricTarget(subset, context), annotation_filter_name)
            ],
        )
        labels.extend(subset_labels)
    return labels


def get_spans(target: MetricTarget, annotation_filter_name: str | None = None) -> list[str]:
    spans = []
    for subset, context in target.components:
        subset_spans = context.get_or_compute(
            f"spans({subset.name}, {annotation_filter_name})",
            lambda: [
                annotation.spans
                for annotation in get_annotations(SingleMetricTarget(subset, context), annotation_filter_name)
            ],
        )
        spans.extend(subset_spans)
    return spans


def get_mentions(target: MetricTarget, annotation_filter_name: str | None = None) -> list[str]:
    mentions = []
    for subset, context in target.components:
        subset_mentions = context.get_or_compute(
            f"mentions({subset.name}, {annotation_filter_name})",
            lambda: [
                annotation.text
                for annotation in get_annotations(SingleMetricTarget(subset, context), annotation_filter_name)
            ],
        )
        mentions.extend(subset_mentions)
    return mentions


def get_mention_tokens(target: MetricTarget, annotation_filter_name: str | None = None) -> list[str]:
    mention_tokens = []
    for subset, context in target.components:
        subset_mention_tokens = context.get_or_compute(
            f"mention_tokens({subset.name}, {annotation_filter_name})",
            lambda: extract_tokens_from_texts(
                get_mentions(SingleMetricTarget(subset, context), annotation_filter_name)
            ),
        )
        mention_tokens.extend(subset_mention_tokens)
    return mention_tokens


def get_identifier_links(target: MetricTarget, annotation_filter_name: str | None = None) -> list[IdentifierLink]:
    identifier_links = []
    for subset, context in target.components:
        subset_identifier_links = context.get_or_compute(
            f"identifier_links({subset.name}, {annotation_filter_name})",
            lambda: [
                identifier_link
                for annotation in get_annotations(SingleMetricTarget(subset, context), annotation_filter_name)
                for identifier_link in annotation.get_identifier_links()
            ],
        )
        identifier_links.extend(subset_identifier_links)
    return identifier_links


def get_identifiers(target: MetricTarget, annotation_filter_name: str | None = None) -> list[str]:
    identifiers = []
    for subset, context in target.components:
        subset_identifiers = context.get_or_compute(
            f"identifiers({subset.name}, {annotation_filter_name})",
            lambda: [
                identifier_link.identifier
                for identifier_link in get_identifier_links(SingleMetricTarget(subset, context), annotation_filter_name)
            ],
        )
        identifiers.extend(subset_identifiers)
    return identifiers


def get_identifier_resources(target: MetricTarget, annotation_filter_name: str | None = None) -> list[str]:
    resources = []
    for subset, context in target.components:
        subset_resources = context.get_or_compute(
            f"identifier_resources({subset.name}, {annotation_filter_name})",
            lambda: [
                identifier_link.resource
                for identifier_link in get_identifier_links(SingleMetricTarget(subset, context), annotation_filter_name)
            ],
        )
        resources.extend(subset_resources)
    return resources


def get_identifier_links_for_terminology(
    target: MetricTarget,
    terminology: TerminologyResource,
    annotation_filter_name: str | None = None,
) -> list[IdentifierLink]:
    identifier_links = []
    terminology_key = terminology.cache_key()
    for subset, context in target.components:
        subset_identifier_links = context.get_or_compute(
            f"identifier_links_for_terminology({subset.name}, {annotation_filter_name}, {terminology_key})",
            lambda: [
                link
                for link in get_identifier_links(SingleMetricTarget(subset, context), annotation_filter_name)
                if link.identifier is not None and terminology.accepts_resource(link.resource)
            ],
        )
        identifier_links.extend(subset_identifier_links)
    return identifier_links


def get_match_types(target: MetricTarget, annotation_filter_name: str | None = None) -> list[str]:
    match_types = []
    for subset, context in target.components:
        subset_match_types = context.get_or_compute(
            f"match_types({subset.name}, {annotation_filter_name})",
            lambda: [
                identifier_link.match_type
                for identifier_link in get_identifier_links(SingleMetricTarget(subset, context), annotation_filter_name)
            ],
        )
        match_types.extend(subset_match_types)
    return match_types


def get_metadata_for_target(target: MetricTarget) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves metadata for all documents in a target.
    Returns a dictionary mapping document_id to its metadata record.
    """
    workspace = get_workspace(target)
    metadata = dict()
    for subset, context in target.components:
        subset_metadata = context.get_or_compute(
            f"metadata({subset.name})", lambda: workspace.get_document_metadata(subset.documents)
        )
        metadata.update(subset_metadata)
    return metadata


def get_terminology_anchor_counter(
    target: MetricTarget,
    terminology: TerminologyResource,
    term_overrides_path: str | None,
) -> TerminologyTopicAnchorCounter:
    if not target.components:
        raise ValueError("terminology anchor counter requires a non-empty metric target")
    cache_key = "terminology_anchor_counter({}, {})".format(terminology.cache_key(), term_overrides_path or "")
    context = target.components[0][1]
    return context.get_or_compute(
        cache_key,
        lambda: TerminologyTopicAnchorCounter(
            terminology,
            term_overrides=load_topic_term_overrides(Path(term_overrides_path)) if term_overrides_path else None,
        ),
    )


def get_high_level_topic_rows(
    target: MetricTarget,
    terminology: TerminologyResource,
    annotation_filter_name: str | None,
    term_overrides_path: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not target.components:
        raise ValueError("high-level terminology topic rows require a non-empty metric target")

    component_key = tuple((subset.name, id(context)) for subset, context in target.components)
    cache_key = "high_level_topic_rows({}, {}, {}, {}, {})".format(
        target.name, component_key, annotation_filter_name, terminology.cache_key(), term_overrides_path or ""
    )
    context = target.components[0][1]
    return context.get_or_compute(
        cache_key,
        lambda: _build_high_level_topic_rows(target, terminology, annotation_filter_name, term_overrides_path),
    )

# TODO This data should probably be lower level for caching
def _build_high_level_topic_rows(
    target: MetricTarget,
    terminology: TerminologyResource,
    annotation_filter_name: str | None,
    term_overrides_path: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identifier_links = get_identifier_links_for_terminology(target, terminology, annotation_filter_name)
    ids = [link.identifier for link in identifier_links if link.identifier is not None]
    unique_ids = _unique_concept_ids(ids, terminology)
    missing_ids = sorted({ui for ui in ids if terminology.get_concept(ui) is None})
    counter = get_terminology_anchor_counter(target, terminology, term_overrides_path)

    if term_overrides_path:
        corpus_counts = counter.count_by_anchor(unique_ids)
        annotation_counts = counter.count_by_anchor(ids)
        global_counts = counter.get_global_counts_by_anchor()
    else:
        corpus_counts = counter.count_by_branch(unique_ids)
        annotation_counts = counter.count_by_branch(ids)
        global_counts = counter.get_global_counts_by_branch()

    all_branches = sorted(set(corpus_counts.keys()) | set(annotation_counts.keys()))
    if term_overrides_path:
        all_branches = sorted(set(all_branches) | set(global_counts.keys()) | set(counter.configured_anchor_topics))

    rows = []
    for branch_code in all_branches:
        count = corpus_counts.get(branch_code, 0.0)
        annotation_count = annotation_counts.get(branch_code, 0.0)
        terminology_total = global_counts.get(branch_code, 0.0)
        terminology_proportion = count / terminology_total if terminology_total > 0 else 0.0
        rows.append(
            {
                "branch_code": branch_code,
                "label": counter.branch_label(branch_code),
                "count": round(count, PRECISION),
                "annotation_count": round(annotation_count, PRECISION),
                "terminology_total_count": round(terminology_total, PRECISION),
                "terminology_proportion": round(terminology_proportion, PRECISION),
                "annotation_proportion": round(annotation_count / len(ids), PRECISION) if ids else 0.0,
            }
        )

    details = {
        "n_input_ids": len(ids),
        "n_unique_input_ids": len(unique_ids),
        "n_missing_ids": len(missing_ids),
        "missing_ids": missing_ids,
        "terminology": terminology.name,
        "resource_aliases": terminology.aliases,
        "term_overrides_path": term_overrides_path,
        "terminology_distribution_entropy": _shannon_entropy(corpus_counts.values()),
        "annotation_distribution_entropy": _shannon_entropy(annotation_counts.values()),
    }
    return rows, details


def _unique_concept_ids(ids: list[str], terminology: TerminologyResource) -> list[str]:
    unique_ids = []
    seen = set()
    for ui in ids:
        concept = terminology.get_concept(ui)
        key = concept.ui if concept is not None else terminology.normalize_identifier(ui)
        if key is None or key in seen:
            continue
        unique_ids.append(key)
        seen.add(key)
    return unique_ids


def _shannon_entropy(counts) -> float:
    total = sum(count for count in counts if count > 0)
    if total <= 0:
        return 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in counts if count > 0)
    return round(entropy, PRECISION)
