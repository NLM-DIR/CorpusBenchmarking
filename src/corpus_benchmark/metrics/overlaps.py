from __future__ import annotations
from collections import Counter

import logging

from corpus_benchmark.context import MetricTarget, get_tokens, get_mentions, get_mention_tokens, get_identifiers
from corpus_benchmark.registry import register_cross_metric
from corpus_benchmark.results import CrossSubsetMetricResult

logger = logging.getLogger(__name__)

PRECISION = 8  # Number of decimal places


@register_cross_metric("token_overlap")
def token_overlap(target1: MetricTarget, target2: MetricTarget, result_name: str) -> CrossSubsetMetricResult:
    tokens1 = set(get_tokens(target1))
    tokens2 = set(get_tokens(target2))
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    return CrossSubsetMetricResult(
        result_name=result_name,
        metric_name="token_overlap",
        value=round(jaccard, PRECISION),
        subset_name1=target1.name,
        subset_name2=target2.name,
        details={
            f"len({target1.name})": len(tokens1),
            f"len({target2.name})": len(tokens2),
            "intersection": len(intersection),
            "union": len(union),
        },
    )


@register_cross_metric("mention_overlap", supports_annotation_scope=True)
def mention_overlap(
    target1: MetricTarget,
    target2: MetricTarget,
    result_name: str,
    annotation_filter_name: str | None = None,
) -> CrossSubsetMetricResult:
    mentions1 = set(get_mentions(target1, annotation_filter_name))
    mentions2 = set(get_mentions(target2, annotation_filter_name))
    intersection = mentions1.intersection(mentions2)
    union = mentions1.union(mentions2)
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    return CrossSubsetMetricResult(
        result_name=result_name,
        metric_name="mention_overlap",
        value=round(jaccard, PRECISION),
        subset_name1=target1.name,
        subset_name2=target2.name,
        details={
            f"len({target1.name})": len(mentions1),
            f"len({target2.name})": len(mentions2),
            "intersection": len(intersection),
            "union": len(union),
        },
    )


@register_cross_metric("mention_token_overlap", supports_annotation_scope=True)
def mention_token_overlap(
    target1: MetricTarget,
    target2: MetricTarget,
    result_name: str,
    annotation_filter_name: str | None = None,
) -> CrossSubsetMetricResult:
    mention_tokens1 = set(get_mention_tokens(target1, annotation_filter_name))
    mention_tokens2 = set(get_mention_tokens(target2, annotation_filter_name))
    intersection = mention_tokens1.intersection(mention_tokens2)
    union = mention_tokens1.union(mention_tokens2)
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    return CrossSubsetMetricResult(
        result_name=result_name,
        metric_name="mention_token_overlap",
        value=round(jaccard, PRECISION),
        subset_name1=target1.name,
        subset_name2=target2.name,
        details={
            f"len({target1.name})": len(mention_tokens1),
            f"len({target2.name})": len(mention_tokens2),
            "intersection": len(intersection),
            "union": len(union),
        },
    )


@register_cross_metric("identifier_overlap", supports_annotation_scope=True)
def identifier_overlap(
    target1: MetricTarget,
    target2: MetricTarget,
    result_name: str,
    annotation_filter_name: str | None = None,
) -> CrossSubsetMetricResult:
    identifiers1 = set(get_identifiers(target1, annotation_filter_name))
    identifiers2 = set(get_identifiers(target2, annotation_filter_name))
    intersection = identifiers1.intersection(identifiers2)
    union = identifiers1.union(identifiers2)
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    return CrossSubsetMetricResult(
        result_name=result_name,
        metric_name="identifier_overlap",
        value=round(jaccard, PRECISION),
        subset_name1=target1.name,
        subset_name2=target2.name,
        details={
            f"len({target1.name})": len(identifiers1),
            f"len({target2.name})": len(identifiers2),
            "intersection": len(intersection),
            "union": len(union),
        },
    )


def _find_bin(count, bins):
    for low, high in bins:
        if low <= count and (high is None or count < high):
            return (low, high)
    raise ValueError(f"Unable to find bin for count {count} in bins: {bins}")


FREQUENCY_BINS = [
    (1, 2),
    (2, 4),
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
    (256, None),
]


def _overlap_by_frequency(items1, items2):
    counts1 = Counter(items1)
    counts2 = Counter(items2)
    counts_total = counts1 + counts2
    bin_counts = {str(bin): [0, 0] for bin in FREQUENCY_BINS}
    for item, count in counts_total.items():
        bin = _find_bin(count, FREQUENCY_BINS)
        bin_counts[str(bin)][0] += 1
        if counts1.get(item, 0) and counts2.get(item, 0):
            bin_counts[str(bin)][1] += 1
    value = {}
    for bin, (union, intersection) in bin_counts.items():
        value[bin] = round(intersection / union if union > 0 else 0.0, PRECISION)
    return value, bin_counts, counts1, counts2


@register_cross_metric("token_overlap_by_frequency")
def token_overlap_by_frequency(
    target1: MetricTarget,
    target2: MetricTarget,
    result_name: str,
) -> CrossSubsetMetricResult:
    value, bin_counts, tokens1, tokens2 = _overlap_by_frequency(get_tokens(target1), get_tokens(target2))
    return CrossSubsetMetricResult(
        result_name=result_name,
        metric_name="token_overlap_by_frequency",
        value=value,
        subset_name1=target1.name,
        subset_name2=target2.name,
        details={
            f"len({target1.name})": len(tokens1),
            f"len({target2.name})": len(tokens2),
            "bin_counts": bin_counts,
        },
    )


@register_cross_metric("mention_overlap_by_frequency", supports_annotation_scope=True)
def mention_overlap_by_frequency(
    target1: MetricTarget,
    target2: MetricTarget,
    result_name: str,
    annotation_filter_name: str | None = None,
) -> CrossSubsetMetricResult:
    value, bin_counts, mentions1, mentions2 = _overlap_by_frequency(
        get_mentions(target1, annotation_filter_name),
        get_mentions(target2, annotation_filter_name),
    )
    return CrossSubsetMetricResult(
        result_name=result_name,
        metric_name="mention_overlap_by_frequency",
        value=value,
        subset_name1=target1.name,
        subset_name2=target2.name,
        details={
            f"len({target1.name})": len(mentions1),
            f"len({target2.name})": len(mentions2),
            "bin_counts": bin_counts,
        },
    )


@register_cross_metric("mention_token_overlap_by_frequency", supports_annotation_scope=True)
def mention_token_overlap_by_frequency(
    target1: MetricTarget,
    target2: MetricTarget,
    result_name: str,
    annotation_filter_name: str | None = None,
) -> CrossSubsetMetricResult:
    value, bin_counts, mention_tokens1, mention_tokens2 = _overlap_by_frequency(
        get_mention_tokens(target1, annotation_filter_name),
        get_mention_tokens(target2, annotation_filter_name),
    )
    return CrossSubsetMetricResult(
        result_name=result_name,
        metric_name="mention_token_overlap_by_frequency",
        value=value,
        subset_name1=target1.name,
        subset_name2=target2.name,
        details={
            f"len({target1.name})": len(mention_tokens1),
            f"len({target2.name})": len(mention_tokens2),
            "bin_counts": bin_counts,
        },
    )


# TODO Make bins configurable somehow
@register_cross_metric("identifier_overlap_by_frequency", supports_annotation_scope=True)
def identifier_overlap_by_frequency(
    target1: MetricTarget,
    target2: MetricTarget,
    result_name: str,
    annotation_filter_name: str | None = None,
) -> CrossSubsetMetricResult:
    value, bin_counts, identifiers1, identifiers2 = _overlap_by_frequency(
        get_identifiers(target1, annotation_filter_name),
        get_identifiers(target2, annotation_filter_name),
    )
    return CrossSubsetMetricResult(
        result_name=result_name,
        metric_name="identifier_overlap_by_frequency",
        value=value,
        subset_name1=target1.name,
        subset_name2=target2.name,
        details={
            f"len({target1.name})": len(identifiers1),
            f"len({target2.name})": len(identifiers2),
            "bin_counts": bin_counts,
        },
    )

