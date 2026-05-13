from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from corpus_benchmark.context import MetricTarget, get_documents, get_metadata_for_target, get_workspace
from corpus_benchmark.models.terminologies import TerminologyTopicAnchorCounter
from corpus_benchmark.models.terminologies import get_journal_topic_anchor_counter
from corpus_benchmark.models.terminologies import load_topic_term_overrides
from corpus_benchmark.models.terminologies import topic_treetop_names
from corpus_benchmark.registry import register_subset_metric
from corpus_benchmark.results import SubsetMetricResult

logger = logging.getLogger(__name__)

PRECISION = 8  # Number of decimal places


def calculate_proportions(counts: Counter[Any]) -> dict[str, float]:
    total = counts.total()
    return {str(label) if label is not None else "null": (round(count / total, PRECISION) if total else 0.0) for label, count in counts.items()}


def normalize_counts(counts: Counter[Any]) -> dict[str, int | float]:
    return {str(label) if label is not None else "null": count for label, count in counts.items()}


def _add_weighted_counts(
    target: Counter[str],
    source: dict[str, float],
    weight: float = 1.0,
) -> None:
    for name, count in source.items():
        target[name] += count * weight


def _normalize_topic_counts(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    return {name: count / total for name, count in counts.items()}


def _get_article_topic_anchor_counter(
    target: MetricTarget,
    terminology,
    terminology_name: str,
    article_topic_path: str,
) -> TerminologyTopicAnchorCounter:
    if not target.components:
        raise ValueError("article topic distribution requires a non-empty metric target")

    cache_key = (
        "article_topic_anchor_counter",
        terminology_name,
        article_topic_path,
    )
    context = target.components[0][1]
    return context.get_or_compute(
        repr(cache_key),
        lambda: TerminologyTopicAnchorCounter(
            terminology,
            term_overrides=load_topic_term_overrides(Path(article_topic_path)),
        ),
    )


def _journal_topic_counts_for_metadata(
    root_counter: TerminologyTopicAnchorCounter,
    journal_record: dict[str, Any] | None,
) -> dict[str, float]:
    if journal_record is None:
        return {}

    journal_name = journal_record.get("name") or journal_record.get("abbreviation") or "Unknown"
    mesh_counts = root_counter.topic_counts(journal_record.get("mesh_topics", []) or [])
    if mesh_counts:
        return _normalize_topic_counts(mesh_counts)
    return _normalize_topic_counts(root_counter.fallback_topic_counts(journal_name))


def _article_topic_counts_with_fallback(
    article_mesh_topics: list[str],
    article_counter: TerminologyTopicAnchorCounter,
    journal_fallback_counts: dict[str, float],
) -> Counter[str]:
    if not article_mesh_topics:
        return Counter(journal_fallback_counts or {"Unknown": 1.0})

    counts: Counter[str] = Counter()
    topic_weight = 1.0 / len(article_mesh_topics)
    for mesh_topic in article_mesh_topics:
        topic_counts = _normalize_topic_counts(article_counter.topic_anchor_counts(mesh_topic))
        if topic_counts:
            _add_weighted_counts(counts, topic_counts, topic_weight)
        elif journal_fallback_counts:
            _add_weighted_counts(counts, journal_fallback_counts, topic_weight)
        else:
            counts["Unknown"] += topic_weight
    return counts


@register_subset_metric("journal_distribution", requires_metadata=True)
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


@register_subset_metric("journal_topic_distribution", requires_metadata=True)
def journal_topic_distribution(target: MetricTarget, result_name: str, terminology_name: str = "mesh") -> SubsetMetricResult:
    workspace = get_workspace(target)
    terminology = workspace.get_terminology(terminology_name)
    metadata = get_metadata_for_target(target)
    topic_treetop_cache: dict[str, list[str]] = {}

    topics = []
    for doc in get_documents(target):
        meta = metadata.get(doc.document_id, {})
        journal_record = workspace.journal_record_store.get_journal_metadata_by_id(meta.get("journal_id"))
        journal_treetops: set[str] = set()
        for mesh_topic in (journal_record or {}).get("mesh_topics", []):
            if mesh_topic not in topic_treetop_cache:
                topic_treetop_cache[mesh_topic] = topic_treetop_names(
                    terminology,
                    mesh_topic,
                )
            journal_treetops.update(topic_treetop_cache[mesh_topic])
        topics.extend(sorted(journal_treetops) or ["Unknown"])

    counts = Counter(topics)
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="journal_topic_distribution",
        value=calculate_proportions(counts),
        subset_name=target.name,
        details={"counts": normalize_counts(counts), "total": counts.total()},
    )


@register_subset_metric("journal_MeSH_topic_distribution", requires_metadata=True)
def journal_MeSH_topic_distribution(
    target: MetricTarget,
    result_name: str,
    terminology_name: str = "mesh",
    journal_topics_path: str = "configs/journal_topics.yaml",
    journal_name_topics_path: str = "configs/journal_name_topic.json",
) -> SubsetMetricResult:
    workspace = get_workspace(target)
    terminology = workspace.get_terminology(terminology_name)
    metadata = get_metadata_for_target(target)
    root_counter = get_journal_topic_anchor_counter(
        target,
        terminology,
        terminology_name,
        journal_topics_path,
        journal_name_topics_path,
    )

    counts: Counter[str] = Counter()
    for doc in get_documents(target):
        meta = metadata.get(doc.document_id, {})
        journal_record = workspace.journal_record_store.get_journal_metadata_by_id(meta.get("journal_id"))
        if journal_record is None:
            counts["Unknown"] += 1.0
            continue

        journal_name = journal_record.get("name") or journal_record.get("abbreviation") or "Unknown"
        root_counts = root_counter.counts_for_record_topics(
            journal_name,
            journal_record.get("mesh_topics", []) or [],
        )
        if root_counts:
            _add_weighted_counts(counts, root_counts)
        else:
            counts["Unknown"] += 1.0

    return SubsetMetricResult(
        result_name=result_name,
        metric_name="journal_MeSH_topic_distribution",
        value=calculate_proportions(counts),
        subset_name=target.name,
        details={"counts": normalize_counts(counts), "total": counts.total()},
    )


@register_subset_metric("article_MeSH_topic_distribution", requires_metadata=True)
def article_MeSH_topic_distribution(
    target: MetricTarget,
    result_name: str,
    terminology_name: str = "mesh",
    article_topic_path: str = "configs/article_topics.yaml",
    journal_topics_path: str = "configs/journal_topics.yaml",
    journal_name_topics_path: str = "configs/journal_name_topic.json",
) -> SubsetMetricResult:
    workspace = get_workspace(target)
    terminology = workspace.get_terminology(terminology_name)
    metadata = get_metadata_for_target(target)
    article_counter = _get_article_topic_anchor_counter(
        target,
        terminology,
        terminology_name,
        article_topic_path,
    )
    journal_counter = get_journal_topic_anchor_counter(
        target,
        terminology,
        terminology_name,
        journal_topics_path,
        journal_name_topics_path,
    )

    counts: Counter[str] = Counter()
    for doc in get_documents(target):
        meta = metadata.get(doc.document_id, {})
        journal_record = workspace.journal_record_store.get_journal_metadata_by_id(meta.get("journal_id"))
        journal_fallback_counts = _journal_topic_counts_for_metadata(journal_counter, journal_record)
        article_counts = _article_topic_counts_with_fallback(
            meta.get("mesh_topics", []) or [],
            article_counter,
            journal_fallback_counts,
        )
        _add_weighted_counts(counts, article_counts)

    return SubsetMetricResult(
        result_name=result_name,
        metric_name="article_MeSH_topic_distribution",
        value=calculate_proportions(counts),
        subset_name=target.name,
        details={"counts": normalize_counts(counts), "total": counts.total()},
    )


@register_subset_metric("publication_year_distribution", requires_metadata=True)
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
