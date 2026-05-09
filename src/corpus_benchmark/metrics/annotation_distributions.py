from __future__ import annotations

import logging
from collections import Counter

from corpus_benchmark.context import MetricTarget, get_labels, get_identifier_resources, get_match_types
from corpus_benchmark.registry import register_subset_metric
from corpus_benchmark.results import SubsetMetricResult

logger = logging.getLogger(__name__)

PRECISION = 8  # Number of decimal places


def calculate_proportions(counts: Counter[Any, int]) -> dict[str, float]:
    total = counts.total()
    return {
        str(label) if label is not None else "null": (
            round(count / total, PRECISION) if total else 0.0
        )
        for label, count in counts.items()
    }


def normalize_counts(counts: Counter[Any, int]) -> dict[str, int]:
    return {
        str(label) if label is not None else "null": count
        for label, count in counts.items()
    }


@register_subset_metric("label_distribution")
def label_distribution(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    counts = Counter(get_labels(target, annotation_filter_name))
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="label_distribution",
        value=calculate_proportions(counts),
        subset_name=target.name,
        details={
            "counts": normalize_counts(counts),
            "total": counts.total(),
        },
    )


@register_subset_metric("identifier_resource_distribution")
def identifier_resource_distribution(
    target: MetricTarget, result_name: str, annotation_filter_name: str | None = None
) -> SubsetMetricResult:
    counts = Counter(get_identifier_resources(target, annotation_filter_name))
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="identifier_resource_distribution",
        value=calculate_proportions(counts),
        subset_name=target.name,
        details={
            "counts": normalize_counts(counts),
            "total": counts.total(),
        },
    )


@register_subset_metric("match_type_distribution")
def match_type_distribution(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    counts = Counter(get_match_types(target))
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="match_type_distribution",
        value=calculate_proportions(counts),
        subset_name=target.name,
        details={
            "counts": normalize_counts(counts),
            "total": counts.total(),
        },
    )
