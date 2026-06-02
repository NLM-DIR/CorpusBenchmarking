from __future__ import annotations

import logging
from typing import Any

from corpus_benchmark.context import (
    MetricTarget,
    get_high_level_topic_rows,
    get_identifier_links_for_terminology,
    get_terminology_anchor_counter,
)
from corpus_benchmark.models.terminologies import TerminologyResource
from corpus_benchmark.registry import register_terminology_metric
from corpus_benchmark.results import SubsetMetricResult

logger = logging.getLogger(__name__)

PRECISION = 8


def _term_overrides_path(params: dict[str, Any], annotation_filter_name: str | None) -> str | None:
    paths_by_scope = params.get("term_override_paths_by_entity_scope") or {}
    if paths_by_scope:
        if not isinstance(paths_by_scope, dict):
            raise ValueError(
                "term_override_paths_by_entity_scope must be a mapping of entity scope names to YAML paths"
            )
        scope_key = annotation_filter_name or "all"
        configured_path = paths_by_scope.get(scope_key)
        if configured_path:
            return str(configured_path)

    configured_path = (
        params.get("term_overrides_path") or params.get("term_override_path") or params.get("topic_terms_path")
    )
    return str(configured_path) if configured_path else None


# TODO terminology_concept_coverage and annotation_topic_coverage return the same information, with the dashboard using them differently; change it to only return the relevant info for each


@register_terminology_metric("terminology_concept_coverage", supports_annotation_scope=True)
def terminology_concept_coverage(
    target: MetricTarget,
    result_name: str,
    terminology: TerminologyResource,
    annotation_filter_name: str | None = None,
    **params,
) -> SubsetMetricResult:
    term_overrides_path = _term_overrides_path(params, annotation_filter_name)
    rows, details = get_high_level_topic_rows(target, terminology, annotation_filter_name, term_overrides_path)
    details = dict(details)
    details["distribution_entropy"] = details["terminology_distribution_entropy"]
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="terminology_concept_coverage",
        subset_name=target.name,
        value=rows,
        details=details,
    )


@register_terminology_metric("annotation_topic_coverage", supports_annotation_scope=True)
def annotation_topic_coverage(
    target: MetricTarget,
    result_name: str,
    terminology: TerminologyResource,
    annotation_filter_name: str | None = None,
    **params,
) -> SubsetMetricResult:
    term_overrides_path = _term_overrides_path(params, annotation_filter_name)
    rows, details = get_high_level_topic_rows(target, terminology, annotation_filter_name, term_overrides_path)
    details = dict(details)
    details["distribution_entropy"] = details["annotation_distribution_entropy"]
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="annotation_topic_coverage",
        subset_name=target.name,
        value=rows,
        details=details,
    )


@register_terminology_metric("concept_depth_counts", supports_annotation_scope=True)
def concept_depth_counts(
    target: MetricTarget,
    result_name: str,
    terminology: TerminologyResource,
    annotation_filter_name: str | None = None,
    **params,
) -> SubsetMetricResult:
    term_overrides_path = _term_overrides_path(params, annotation_filter_name)
    identifier_links = get_identifier_links_for_terminology(target, terminology, annotation_filter_name)
    ids = [link.identifier for link in identifier_links if link.identifier is not None]
    counter = get_terminology_anchor_counter(target, terminology, term_overrides_path)

    corpus_counts = counter.count_by_depth(ids)
    global_counts = counter.get_global_counts_by_depth()
    corpus_total = sum(corpus_counts.values())

    all_depths = sorted(set(corpus_counts.keys()) | set(global_counts.keys()))
    rows = []
    for d in all_depths:
        c_count = corpus_counts.get(d, 0.0)
        m_count = global_counts.get(d, 0.0)
        rows.append(
            {
                "depth": d,
                "count": round(c_count, PRECISION),
                "terminology_total_count": round(m_count, PRECISION),
                "terminology_proportion": round(c_count / corpus_total, PRECISION) if corpus_total > 0 else 0.0,
            }
        )

    missing_ids = sorted({ui for ui in ids if terminology.get_concept(ui) is None})
    return SubsetMetricResult(
        result_name=result_name,
        metric_name="concept_depth_counts",
        subset_name=target.name,
        value=rows,
        details={
            "n_input_ids": len(ids),
            "n_missing_ids": len(missing_ids),
            "missing_ids": missing_ids,
            "terminology": terminology.name,
            "resource_aliases": terminology.aliases,
        },
    )
