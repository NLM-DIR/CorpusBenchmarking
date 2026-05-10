from __future__ import annotations

import logging

from corpus_benchmark.context import MetricTarget, get_identifier_links
from corpus_benchmark.models.terminologies import TerminologyResource
from corpus_benchmark.models.terminologies import TerminologyTopicAnchorCounter
from corpus_benchmark.registry import register_terminology_metric
from corpus_benchmark.results import SubsetMetricResult

logger = logging.getLogger(__name__)

PRECISION = 8

# TODO make these metrics resource-aware: only do ID lookups in the associated terminology
# TODO report IDs not found as <resource>:<accession>

def _identifier_links_for_terminology(
    target: MetricTarget,
    terminology: TerminologyResource,
    annotation_filter_name: str | None,
):
    return [
        link
        for link in get_identifier_links(target, annotation_filter_name)
        if link.identifier is not None and terminology.accepts_resource(link.resource)
    ]


@register_terminology_metric("high_level_concept_counts", supports_annotation_scope=True)
def high_level_concept_counts(target: MetricTarget, result_name: str, terminology: TerminologyResource, annotation_filter_name: str | None = None, **params) -> SubsetMetricResult:
    identifier_links = _identifier_links_for_terminology(target, terminology, annotation_filter_name)
    ids = [link.identifier for link in identifier_links if link.identifier is not None]
    missing_ids = [ui for ui in ids if terminology.get_concept(ui) is None]
    missing_ids = sorted(list(set(missing_ids)))
    counter = TerminologyTopicAnchorCounter(terminology)

    corpus_counts = counter.count_by_branch(ids)
    global_counts = counter.get_global_counts_by_branch()

    all_branches = sorted(corpus_counts.keys())
    rows = []
    for branch_code in all_branches:
        count = corpus_counts.get(branch_code, 0.0)
        terminology_total = global_counts.get(branch_code, 0.0)
        proportion = count / terminology_total if terminology_total > 0 else 0.0

        rows.append(
            {
                "branch_code": branch_code,
                "label": counter.branch_label(branch_code),
                "treetop": branch_code.split(".")[0],
                "treetop_name": terminology.treetop_names.get(branch_code.split(".")[0]) or counter.branch_label(branch_code),
                "count": round(count, PRECISION),
                "terminology_total_count": round(terminology_total, PRECISION),
                "mesh_total_count": round(terminology_total, PRECISION),
                "proportion": round(proportion, PRECISION),
            }
        )

    return SubsetMetricResult(
        result_name=result_name,
        metric_name="high_level_concept_counts",
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


@register_terminology_metric("concept_depth_counts", supports_annotation_scope=True)
def concept_depth_counts(target: MetricTarget, result_name: str, terminology: TerminologyResource, annotation_filter_name: str | None = None, **params) -> SubsetMetricResult:
    identifier_links = _identifier_links_for_terminology(target, terminology, annotation_filter_name)
    ids = [link.identifier for link in identifier_links if link.identifier is not None]
    counter = TerminologyTopicAnchorCounter(terminology)

    corpus_counts = counter.count_by_depth(ids)
    global_counts = counter.get_global_counts_by_depth()

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
                "mesh_total_count": round(m_count, PRECISION),
                "proportion": round(c_count / m_count, PRECISION) if m_count > 0 else 0.0,
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
