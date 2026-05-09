from __future__ import annotations

import logging
import statistics
import math
from collections import Counter, defaultdict

from corpus_benchmark.context import (
    MetricTarget,
    get_documents,
    get_passages,
    get_sentences,
    get_tokens,
    get_annotations,
    get_annotations_per_document,
    get_spans,
    get_mentions,
    get_mention_tokens,
    get_identifiers,
)
from corpus_benchmark.registry import register_subset_metric
from corpus_benchmark.results import SubsetMetricResult
from utils.text_utils import extract_tokens_from_texts

logger = logging.getLogger(__name__)

PRECISION = 8  # Number of decimal places

LENGTH_ZERO = (
    {
        "mean": math.nan,
        "stdev": math.nan,
        "min": math.nan,
        "median": math.nan,
        "max": math.nan,
        "count": 0,
    },
    Counter([]),
)


def calculate_stats(values: list[int]) -> dict[str, float]:
    if len(values) == 0:
        return LENGTH_ZERO
    stats = {
        "mean": round(statistics.mean(values), PRECISION),
        "stdev": round(statistics.stdev(values), PRECISION) if len(values) > 1 else math.nan,
        "min": round(min(values), PRECISION),
        "median": round(statistics.median(values), PRECISION),
        "max": round(max(values), PRECISION),
        "count": round(len(values), PRECISION),
    }
    return stats, Counter(values)


@register_subset_metric("passages_per_document_stats")
def passages_per_document_stats(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    stats, distribution = calculate_stats([len(document.passages) for document in get_documents(target)])
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="passages_per_document_stats",
        value=stats,
        subset_name=target.name,
        details={
            "distribution": dict(distribution),
            "total": distribution.total(),
        },
    )


# TODO Add sentences per document
# TODO Add sentences per passage
# TODO Add tokens per document
# TODO Add tokens per passage
# TODO Add tokens per sentence
# TODO Add tokens per mention


@register_subset_metric("annotations_per_document_stats")
def annotations_per_document_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    stats, distribution = calculate_stats(
        [
            len(annotations_for_document)
            for annotations_for_document in get_annotations_per_document(target, annotation_filter_name)
        ]
    )
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="annotations_per_document_stats",
        value=stats,
        subset_name=target.name,
        # details={
        #    "distribution": dict(distribution),
        #    "total": distribution.total(),
        # },
    )


@register_subset_metric("annotations_per_1000_tokens_stats")
def annotations_per_1000_tokens_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    counts = []
    for document, annotations_for_document in zip(
        get_documents(target),
        get_annotations_per_document(target, annotation_filter_name),
    ):
        token_count = len(extract_tokens_from_texts([passage.text for passage in document.passages]))
        counts.append(len(annotations_for_document) / token_count * 1000 if token_count else 0)
    stats, distribution = calculate_stats(counts)
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="annotations_per_1000_tokens_stats",
        value=stats,
        subset_name=target.name,
    )


@register_subset_metric("unique_mentions_per_document_stats")
def unique_mentions_per_document_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    counts = []
    for annotations_for_document in get_annotations_per_document(target, annotation_filter_name):
        mentions = {annotation.text for annotation in annotations_for_document}
        counts.append(len(mentions))
    stats, distribution = calculate_stats(counts)
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="unique_mentions_per_document_stats",
        value=stats,
        subset_name=target.name,
        # details={
        #    "distribution": dict(distribution),
        #    "total": distribution.total(),
        # },
    )


@register_subset_metric("unique_identifiers_per_document_stats")
def unique_identifiers_per_document_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    counts = []
    for annotations_for_document in get_annotations_per_document(target, annotation_filter_name):
        identifiers_for_document = set()
        for annotation in annotations_for_document:
            for identifier_link in annotation.get_identifier_links():
                identifiers_for_document.add(identifier_link.identifier)
        counts.append(len(identifiers_for_document))
    stats, distribution = calculate_stats(counts)
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="unique_identifiers_per_document_stats",
        value=stats,
        subset_name=target.name,
        # details={
        #    "distribution": dict(distribution),
        #    "total": distribution.total(),
        # },
    )


@register_subset_metric("spans_per_annotation_stats")
def spans_per_annotation_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    stats, distribution = calculate_stats([len(spans) for spans in get_spans(target, annotation_filter_name)])
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="spans_per_annotation_stats",
        value=stats,
        subset_name=target.name,
        # details={
        #    "distribution": dict(distribution),
        #    "total": distribution.total(),
        # },
    )


@register_subset_metric("identifiers_per_annotation_stats")
def identifiers_per_annotation_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    counts = [len(annotation.get_identifier_links()) for annotation in get_annotations(target, annotation_filter_name)]
    stats, distribution = calculate_stats(counts)
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="identifiers_per_annotation_stats",
        value=stats,
        subset_name=target.name,
        # details={
        #    "distribution": dict(distribution),
        #    "total": distribution.total(),
        # },
    )


@register_subset_metric("document_length_stats")
def document_length_stats(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    documents = get_documents(target)
    stats, distribution = calculate_stats([sum(len(passage.text) for passage in doc.passages) for doc in documents])
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="document_length_stats",
        value=stats,
        subset_name=target.name,
        # details={
        #    "distribution": dict(distribution),
        #    "total": distribution.total(),
        # },
    )


@register_subset_metric("passage_length_stats")
def passage_length_stats(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    passages = get_passages(target)
    stats, distribution = calculate_stats([len(passage.text) for passage in passages])
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="passage_length_stats",
        value=stats,
        subset_name=target.name,
        details={
            "distribution": dict(distribution),
            "total": distribution.total(),
        },
    )


@register_subset_metric("sentence_length_stats")
def sentence_length_stats(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    sentences = get_sentences(target)
    stats, distribution = calculate_stats([len(sentence) for sentence in sentences])
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="sentence_length_stats",
        value=stats,
        subset_name=target.name,
        details={
            "distribution": dict(distribution),
            "total": distribution.total(),
        },
    )


@register_subset_metric("token_length_stats")
def token_length_stats(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    tokens = get_tokens(target)
    stats, distribution = calculate_stats([len(token) for token in tokens])
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="token_length_stats",
        value=stats,
        subset_name=target.name,
        details={
            "distribution": dict(distribution),
            "total": distribution.total(),
        },
    )


@register_subset_metric("mention_length_stats")
def mention_length_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    mentions = get_mentions(target, annotation_filter_name)
    stats, distribution = calculate_stats([len(mention) for mention in mentions])
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="mention_length_stats",
        value=stats,
        subset_name=target.name,
        # details={
        #    "distribution": dict(distribution),
        #    "total": distribution.total(),
        # },
    )


@register_subset_metric("mention_token_length_stats")
def mention_token_length_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    mention_tokens = get_mention_tokens(target, annotation_filter_name)
    stats, distribution = calculate_stats([len(mention_token) for mention_token in mention_tokens])
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="mention_token_length_stats",
        value=stats,
        subset_name=target.name,
        details={
            "distribution": dict(distribution),
            "total": distribution.total(),
        },
    )


@register_subset_metric("passage_redundancy_stats")
def passage_redundancy_stats(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    counts = Counter(get_passages(target))
    stats, distribution = calculate_stats(counts.values())
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="passage_redundancy_stats",
        value=stats,
        subset_name=target.name,
        details={
            "distribution": dict(distribution),
            "total": distribution.total(),
        },
    )


@register_subset_metric("sentence_redundancy_stats")
def sentence_redundancy_stats(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    counts = Counter(get_sentences(target))
    stats, distribution = calculate_stats(counts.values())
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="sentence_redundancy_stats",
        value=stats,
        subset_name=target.name,
        details={
            "distribution": dict(distribution),
            "total": distribution.total(),
        },
    )


@register_subset_metric("token_redundancy_stats")
def token_redundancy_stats(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    counts = Counter(get_tokens(target))
    stats, distribution = calculate_stats(counts.values())
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="token_redundancy_stats",
        value=stats,
        subset_name=target.name,
        details={
            "distribution": dict(distribution),
            "total": distribution.total(),
        },
    )


@register_subset_metric("mention_redundancy_stats")
def mention_redundancy_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    counts = Counter(get_mentions(target, annotation_filter_name))
    stats, distribution = calculate_stats(counts.values())
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="mention_redundancy_stats",
        value=stats,
        subset_name=target.name,
        details={
            "distribution": dict(distribution),
            "total": distribution.total(),
        },
    )


@register_subset_metric("mention_token_redundancy_stats")
def mention_token_redundancy_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    counts = Counter(get_mention_tokens(target, annotation_filter_name))
    stats, distribution = calculate_stats(counts.values())
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="mention_token_redundancy_stats",
        value=stats,
        subset_name=target.name,
        details={
            "distribution": dict(distribution),
            "total": distribution.total(),
        },
    )


@register_subset_metric("identifier_redundancy_stats")
def identifier_redundancy_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    counts = Counter(get_identifiers(target, annotation_filter_name))
    stats, distribution = calculate_stats(counts.values())
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="identifier_redundancy_stats",
        value=stats,
        subset_name=target.name,
        # details={
        #    "distribution": dict(distribution),
        #    "total": distribution.total(),
        # },
    )


@register_subset_metric("variation_degree_stats")
def variation_degree_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    ll2texts: defaultdict[str, set[str]] = defaultdict(set)
    for annotation in get_annotations(target, annotation_filter_name):
        ll2texts[(annotation.label, str(annotation.link))].add(annotation.text)
    counts = [len(texts) for texts in ll2texts.values()]
    stats, distribution = calculate_stats(counts)
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="variation_degree_stats",
        value=stats,
        subset_name=target.name,
        # details={
        #    "distribution": dict(distribution),
        #    "total": distribution.total(),
        # },
    )


@register_subset_metric("ambiguity_degree_stats")
def ambiguity_degree_stats(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    text2lls: defaultdict[str, set[str]] = defaultdict(set)
    for annotation in get_annotations(target, annotation_filter_name):
        text2lls[annotation.text].add((annotation.label, str(annotation.link)))
    counts = [len(lls) for lls in text2lls.values()]
    stats, distribution = calculate_stats(counts)
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="ambiguity_degree_stats",
        value=stats,
        subset_name=target.name,
        # details={
        #    "distribution": dict(distribution),
        #    "total": distribution.total(),
        # },
    )
