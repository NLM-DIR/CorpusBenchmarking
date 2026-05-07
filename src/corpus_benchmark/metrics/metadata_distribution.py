from __future__ import annotations

import logging
from collections import Counter
import json
from pathlib import Path
from typing import Any

import yaml

from corpus_benchmark.context import MetricTarget, get_documents, get_metadata_for_target, get_workspace
from corpus_benchmark.metadata.journal_MeSH_topics import JournalMeSHTopicRootCounter
from corpus_benchmark.registry import register_subset_metric
from corpus_benchmark.results import SubsetMetricResult
from corpus_benchmark.models.terminologies import TerminologyResource

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
) -> None:
    for name, count in source.items():
        target[name] += count


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


@register_subset_metric("journal_MeSH_topic_distribution")
def journal_MeSH_topic_distribution(
    target: MetricTarget,
    result_name: str,
    terminology_name: str = "mesh",
    journal_topics_path: str = "configs/journal_topics.yaml",
    journal_name_topics_path: str = "configs/journal_name_topic.json",
) -> SubsetMetricResult:
    workspace = get_workspace(target)
    terminology = _get_terminology(workspace.terminologies, terminology_name)
    metadata = get_metadata_for_target(target)
    root_counter = _get_journal_mesh_topic_root_counter(
        target,
        terminology,
        terminology_name,
        journal_topics_path,
        journal_name_topics_path,
    )

    counts: Counter[str] = Counter()
    for doc in get_documents(target):
        meta = metadata.get(doc.document_id, {})
        journal_record = workspace.get_journal_metadata_by_id(meta.get("journal_id"))
        if journal_record is None:
            counts["Unknown"] += 1.0
            continue

        journal_name = journal_record.get("name") or journal_record.get("abbreviation") or "Unknown"
        root_counts = root_counter.root_counts(
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


def _get_journal_mesh_topic_root_counter(
    target: MetricTarget,
    terminology: TerminologyResource,
    terminology_name: str,
    journal_topics_path: str,
    journal_name_topics_path: str,
) -> JournalMeSHTopicRootCounter:
    if not target.components:
        raise ValueError("journal_MeSH_topic_distribution requires a non-empty metric target")

    cache_key = (
        "journal_MeSH_topic_distribution.counter",
        terminology_name,
        journal_topics_path,
        journal_name_topics_path,
    )
    context = target.components[0][1]
    return context.get_or_compute(
        repr(cache_key),
        lambda: JournalMeSHTopicRootCounter(
            terminology,
            _load_mesh_term_overrides(Path(journal_topics_path)),
            _load_journal_name_topics(Path(journal_name_topics_path)),
        ),
    )


def _load_mesh_term_overrides(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping of journal topics to MeSH terms")

    overrides: dict[str, str] = {}
    for topic, mesh_terms in payload.items():
        if not isinstance(topic, str):
            raise ValueError(f"{path} contains a non-string journal topic: {topic!r}")
        if not isinstance(mesh_terms, list):
            raise ValueError(f"{path} topic {topic!r} must contain a list of MeSH terms")
        for mesh_term in mesh_terms:
            if not isinstance(mesh_term, str):
                raise ValueError(f"{path} topic {topic!r} contains a non-string MeSH term: {mesh_term!r}")
            existing_topic = overrides.get(mesh_term)
            if existing_topic is not None and existing_topic != topic:
                raise ValueError(f"{path} maps MeSH term {mesh_term!r} to both {existing_topic!r} and {topic!r}")
            overrides[mesh_term] = topic
    return overrides


def _load_journal_name_topics(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON mapping of journal names to topic lists")

    journal_topics: dict[str, list[str]] = {}
    for journal_name, topics in payload.items():
        if not isinstance(journal_name, str):
            raise ValueError(f"{path} contains a non-string journal name: {journal_name!r}")
        if not isinstance(topics, list) or not topics:
            raise ValueError(f"{path} journal {journal_name!r} must contain a non-empty list of topics")
        journal_topics[journal_name] = []
        for topic in topics:
            if not isinstance(topic, str):
                raise ValueError(f"{path} journal {journal_name!r} contains a non-string topic: {topic!r}")
            journal_topics[journal_name].append(topic)
    return journal_topics


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
