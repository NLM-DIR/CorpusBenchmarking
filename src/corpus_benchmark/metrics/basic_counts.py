from __future__ import annotations

import logging

from corpus_benchmark.context import (
    MetricTarget,
    get_documents,
    get_passages,
    get_sentences,
    get_tokens,
    get_mentions,
    get_mention_tokens,
    get_identifiers,
)
from corpus_benchmark.registry import register_subset_metric
from corpus_benchmark.results import SubsetMetricResult

logger = logging.getLogger(__name__)


@register_subset_metric("document_count")
def document_count(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    count = len(get_documents(target))
    return SubsetMetricResult(
        result_name=result_name, metric_name="document_count", value=count, subset_name=target.name
    )


@register_subset_metric("passage_count")
def passage_count(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    count = len(get_passages(target))
    return SubsetMetricResult(
        result_name=result_name, metric_name="passage_count", value=count, subset_name=target.name
    )


@register_subset_metric("sentence_count")
def sentence_count(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    count = len(get_sentences(target))
    return SubsetMetricResult(
        result_name=result_name, metric_name="sentence_count", value=count, subset_name=target.name
    )


@register_subset_metric("token_count")
def token_count(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    count = len(get_tokens(target))
    return SubsetMetricResult(result_name=result_name, metric_name="token_count", value=count, subset_name=target.name)


@register_subset_metric("mention_count")
def mention_count(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    count = len(get_mentions(target, annotation_filter_name))
    return SubsetMetricResult(
        result_name=result_name, metric_name="mention_count", value=count, subset_name=target.name
    )


@register_subset_metric("mention_token_count")
def mention_token_count(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    count = len(get_mention_tokens(target, annotation_filter_name))
    return SubsetMetricResult(
        result_name=result_name, metric_name="mention_token_count", value=count, subset_name=target.name
    )


@register_subset_metric("identifier_count")
def identifier_count(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    count = len(get_identifiers(target, annotation_filter_name))
    return SubsetMetricResult(
        result_name=result_name, metric_name="identifier_count", value=count, subset_name=target.name
    )
