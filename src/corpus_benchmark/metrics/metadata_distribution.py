from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Any

from corpus_benchmark.context import MetricTarget, get_documents
from corpus_benchmark.metadata.journal_topics import classify_journal
from corpus_benchmark.registry import register_subset_metric
from corpus_benchmark.results import SubsetMetricResult
from corpus_benchmark.context import get_metadata_for_target

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


@register_subset_metric("journal_distribution")
def journal_distribution(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    metadata = get_metadata_for_target(target)

    journals = []
    for doc in get_documents(target):
        meta = metadata.get(doc.document_id, {})
        journals.append(meta.get("journal") or "Unknown")

    counts = Counter(journals)
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="journal_distribution",
        value=calculate_proportions(counts),
        subset_name=target.name,
        details={"counts": normalize_counts(counts), "total": counts.total()},
    )


@register_subset_metric("journal_topic_distribution")
def journal_topic_distribution(
    target: MetricTarget, result_name: str
) -> SubsetMetricResult:
    metadata = get_metadata_for_target(target)

    topics = []
    for doc in get_documents(target):
        meta = metadata.get(doc.document_id, {})
        topics.append(classify_journal(meta.get("journal") or "Unknown"))

    counts = Counter(topics)
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="journal_topic_distribution",
        value=calculate_proportions(counts),
        subset_name=target.name,
        details={"counts": normalize_counts(counts), "total": counts.total()},
    )


@register_subset_metric("publication_year_distribution")
def publication_year_distribution(target: MetricTarget, result_name: str) -> SubsetMetricResult:
    metadata = get_metadata_for_target(target)

    years = []
    for doc in get_documents(target):
        meta = metadata.get(doc.document_id, {})
        years.append(meta.get("pub_year") or "Unknown")

    counts = Counter(years)
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="publication_year_distribution",
        value=calculate_proportions(counts),
        subset_name=target.name,
        details={"counts": normalize_counts(counts), "total": counts.total()},
    )
