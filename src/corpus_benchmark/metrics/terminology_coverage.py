from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from corpus_benchmark.context import MetricTarget, get_identifier_links
from corpus_benchmark.models.terminologies import TerminologyResource
from corpus_benchmark.models.terminologies import TerminologyTopicAnchorCounter
from corpus_benchmark.models.terminologies import load_topic_term_overrides
from corpus_benchmark.registry import register_terminology_metric
from corpus_benchmark.results import SubsetMetricResult

logger = logging.getLogger(__name__)

PRECISION = 8
_ANCHOR_COUNTER_CACHE: dict[tuple[int, int, int, str], TerminologyTopicAnchorCounter] = {}

# TODO make these metrics resource-aware: only do ID lookups in the associated terminology
# TODO report IDs not found as <resource>:<accession>


def _identifier_links_for_terminology(
    target: MetricTarget,
    terminology: TerminologyResource,
    annotation_filter_name: str | None,
):
    return [link for link in get_identifier_links(target, annotation_filter_name) if link.identifier is not None and terminology.accepts_resource(link.resource)]


def _unique_concept_ids(ids: list[str], terminology: TerminologyResource) -> list[str]:
    unique_ids = []
    seen = set()
    for ui in ids:
        concept = terminology.get_concept(ui)
        key = concept.ui if concept is not None else terminology.normalize_identifier(ui)
        if key is None or key in seen:
            continue
        unique_ids.append(key)
        seen.add(key)
    return unique_ids


def _term_overrides_path(params: dict[str, Any], annotation_filter_name: str | None) -> str | None:
    paths_by_scope = params.get("term_override_paths_by_entity_scope") or {}
    if paths_by_scope:
        if not isinstance(paths_by_scope, dict):
            raise ValueError("term_override_paths_by_entity_scope must be a mapping of entity scope names to YAML paths")
        scope_key = annotation_filter_name or "all"
        configured_path = paths_by_scope.get(scope_key)
        if configured_path:
            return str(configured_path)

    configured_path = params.get("term_overrides_path") or params.get("term_override_path") or params.get("topic_terms_path")
    return str(configured_path) if configured_path else None


def _get_anchor_counter(
    target: MetricTarget,
    terminology: TerminologyResource,
    annotation_filter_name: str | None,
    params: dict[str, Any],
) -> TerminologyTopicAnchorCounter:
    term_overrides_path = _term_overrides_path(params, annotation_filter_name)
    cache_key = (id(terminology), len(terminology.concepts), id(terminology.concepts), term_overrides_path or "")

    def build_counter() -> TerminologyTopicAnchorCounter:
        term_overrides = None
        if term_overrides_path:
            term_overrides = load_topic_term_overrides(Path(term_overrides_path))
        return TerminologyTopicAnchorCounter(terminology, term_overrides=term_overrides)

    if cache_key not in _ANCHOR_COUNTER_CACHE:
        _ANCHOR_COUNTER_CACHE[cache_key] = build_counter()
    return _ANCHOR_COUNTER_CACHE[cache_key]


@register_terminology_metric("high_level_concept_counts", supports_annotation_scope=True)
def high_level_concept_counts(target: MetricTarget, result_name: str, terminology: TerminologyResource, annotation_filter_name: str | None = None, **params) -> SubsetMetricResult:
    identifier_links = _identifier_links_for_terminology(target, terminology, annotation_filter_name)
    ids = [link.identifier for link in identifier_links if link.identifier is not None]
    unique_ids = _unique_concept_ids(ids, terminology)
    missing_ids = [ui for ui in ids if terminology.get_concept(ui) is None]
    missing_ids = sorted(list(set(missing_ids)))
    term_overrides_path = _term_overrides_path(params, annotation_filter_name)
    counter = _get_anchor_counter(target, terminology, annotation_filter_name, params)

    if term_overrides_path:
        corpus_counts = counter.count_by_anchor(unique_ids)
        annotation_counts = counter.count_by_anchor(ids)
        global_counts = counter.get_global_counts_by_anchor()
    else:
        corpus_counts = counter.count_by_branch(unique_ids)
        annotation_counts = counter.count_by_branch(ids)
        global_counts = counter.get_global_counts_by_branch()

    all_branches = sorted(set(corpus_counts.keys()) | set(annotation_counts.keys()))
    if term_overrides_path:
        all_branches = sorted(set(all_branches) | set(global_counts.keys()) | set(counter.configured_anchor_topics))
    rows = []
    for branch_code in all_branches:
        count = corpus_counts.get(branch_code, 0.0)
        annotation_count = annotation_counts.get(branch_code, 0.0)
        terminology_total = global_counts.get(branch_code, 0.0)
        terminology_proportion = count / terminology_total if terminology_total > 0 else 0.0
        treetop = branch_code if term_overrides_path else branch_code.split(".")[0]
        treetop_name = counter.branch_label(branch_code) if term_overrides_path else terminology.treetop_names.get(treetop) or counter.branch_label(branch_code)
        rounded_terminology_proportion = round(terminology_proportion, PRECISION)

        rows.append(
            {
                "branch_code": branch_code,
                "label": counter.branch_label(branch_code),
                "count": round(count, PRECISION),
                "annotation_count": round(annotation_count, PRECISION),
                "terminology_total_count": round(terminology_total, PRECISION),
                "terminology_proportion": rounded_terminology_proportion,
                "annotation_proportion": round(annotation_count / len(ids), PRECISION) if ids else 0.0,
            }
        )

    return SubsetMetricResult(
        result_name=result_name,
        metric_name="high_level_concept_counts",
        subset_name=target.name,
        value=rows,
        details={
            "n_input_ids": len(ids),
            "n_unique_input_ids": len(unique_ids),
            "n_missing_ids": len(missing_ids),
            "missing_ids": missing_ids,
            "terminology": terminology.name,
            "resource_aliases": terminology.aliases,
            "term_overrides_path": term_overrides_path,
        },
    )


@register_terminology_metric("concept_depth_counts", supports_annotation_scope=True)
def concept_depth_counts(target: MetricTarget, result_name: str, terminology: TerminologyResource, annotation_filter_name: str | None = None, **params) -> SubsetMetricResult:
    identifier_links = _identifier_links_for_terminology(target, terminology, annotation_filter_name)
    ids = [link.identifier for link in identifier_links if link.identifier is not None]
    counter = TerminologyTopicAnchorCounter(terminology)

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
