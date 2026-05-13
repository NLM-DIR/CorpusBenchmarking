from __future__ import annotations

import logging

from corpus_benchmark.context import (
    BenchmarkContext,
    get_sentences,
    get_tokens,
    get_mentions,
    get_mention_tokens,
    get_identifiers,
)
from corpus_benchmark.models.corpus import CorpusSubset
from corpus_benchmark.registry import register_subset_metric
from corpus_benchmark.results import SubsetMetricResult

logger = logging.getLogger(__name__)


@register_subset_metric("unique_tokens")
def unique_tokens(subset: CorpusSubset, context: BenchmarkContext) -> SubsetMetricResult:
    count = len(set(get_tokens(subset, context)))
    return SubsetMetricResult(metric_name="unique_tokens", value=count, subset_name=subset.name)


@register_subset_metric("unique_sentences")
def unique_sentences(subset: CorpusSubset, context: BenchmarkContext) -> SubsetMetricResult:
    count = len(set(get_sentences(subset, context)))
    return SubsetMetricResult(metric_name="unique_sentences", value=count, subset_name=subset.name)


@register_subset_metric("unique_mentions")
def unique_mentions(
    subset: CorpusSubset, context: BenchmarkContext, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    count = len(set(get_mentions(subset, context, annotation_filter_name)))
    return SubsetMetricResult(metric_name="unique_mentions", value=count, subset_name=subset.name)


@register_subset_metric("unique_mention_tokens")
def unique_mention_tokens(
    subset: CorpusSubset, context: BenchmarkContext, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    count = len(set(get_mention_tokens(subset, context, annotation_filter_name)))
    return SubsetMetricResult(metric_name="unique_mention_tokens", value=count, subset_name=subset.name)


@register_subset_metric("unique_identifiers")
def unique_identifiers(
    subset: CorpusSubset, context: BenchmarkContext, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    count = len(set(get_identifiers(subset, context, annotation_filter_name)))
    return SubsetMetricResult(metric_name="unique_identifiers", value=count, subset_name=subset.name)
