from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Any

from corpus_benchmark.context import MetricTarget, get_documents, get_workspace
from corpus_benchmark.registry import register_subset_metric
from corpus_benchmark.results import SubsetMetricResult
from corpus_benchmark.context import get_metadata_for_target
from corpus_benchmark.models.terminologies import TerminologyResource

logger = logging.getLogger(__name__)

PRECISION = 8  # Number of decimal places


def calculate_proportions(counts: Counter[Any, int]) -> dict[str, float]:
    total = counts.total()
    return {str(label) if label is not None else "null": (round(count / total, PRECISION) if total else 0.0) for label, count in counts.items()}


def normalize_counts(counts: Counter[Any, int]) -> dict[str, int]:
    return {str(label) if label is not None else "null": count for label, count in counts.items()}


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
def journal_topic_distribution(target: MetricTarget, result_name: str, terminology_name: str = "mesh") -> SubsetMetricResult:
    workspace = get_workspace(target)
    terminology = _get_terminology(workspace.terminologies, terminology_name)
    metadata = get_metadata_for_target(target)
    topic_treetop_cache: dict[str, list[str]] = {}

    topics = []
    for doc in get_documents(target):
        meta = metadata.get(doc.document_id, {})
        journal_record = workspace.get_journal_metadata_by_id(meta.get("journal_id"))
        journal_treetops: set[str] = set()
        for mesh_topic in (journal_record or {}).get("mesh_topics", []):
            if mesh_topic not in topic_treetop_cache:
                topic_treetop_cache[mesh_topic] = _mesh_topic_treetop_names(
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


def _get_terminology(terminologies: dict[str, TerminologyResource], terminology_name: str | None) -> TerminologyResource:
    if terminology_name and terminology_name in terminologies:
        return terminologies[terminology_name]
    if terminology_name:
        available = ", ".join(sorted(terminologies)) or "<none>"
        raise ValueError(f"journal_topic_distribution requires loaded terminology " f"{terminology_name!r}. Available terminologies: {available}")
    if len(terminologies) == 1:
        return next(iter(terminologies.values()))
    available = ", ".join(sorted(terminologies)) or "<none>"
    raise ValueError("journal_topic_distribution requires terminology_name when multiple " f"terminologies are loaded. Available terminologies: {available}")


def _mesh_topic_treetop_names(terminology: TerminologyResource, mesh_topic_name: str) -> list[str]:
    treetop_names: set[str] = set()
    for ui in terminology.get_concept_ids_by_name(mesh_topic_name):
        for concept in terminology.resolve_to_tree_concepts(ui):
            for tree_number in concept.tree_numbers:
                treetop = tree_number[0]
                treetop_name = terminology.treetop_names.get(treetop)
                if treetop_name:
                    treetop_names.add(treetop_name)
    return sorted(treetop_names)


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
