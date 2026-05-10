from __future__ import annotations

from corpus_benchmark.context import BenchmarkContext, MetricTarget
from corpus_benchmark.metrics.terminology_coverage import high_level_concept_counts
from corpus_benchmark.models.corpus import Annotation, AnnotationSpan, CorpusSubset, Document, IdentifierLink, Passage
from corpus_benchmark.models.filters import AnnotationFilter
from corpus_benchmark.models.terminologies import TerminologyConcept, TerminologyResource, TerminologyTopicAnchorCounter


def _target() -> MetricTarget:
    annotations = [
        Annotation(
            mention_id="a1",
            text="cell",
            spans=[AnnotationSpan(0, 4)],
            label="Cell",
            link=IdentifierLink(identifier="CL:0002", resource="CL"),
        ),
        Annotation(
            mention_id="a2",
            text="chemical",
            spans=[AnnotationSpan(5, 13)],
            label="Chemical",
            link=IdentifierLink(identifier="CHEBI:1", resource="CHEBI"),
        ),
    ]
    passage = Passage("p1", "cell chemical", 0, annotations)
    subset = CorpusSubset("train", [Document("doc1", [passage])])
    context = BenchmarkContext(
        workspace=object(),
        annotation_filters={"cell": AnnotationFilter(labels={"Cell"})},
    )
    return MetricTarget("Example_corpus", [(subset, context)])


def test_terminology_metric_filters_identifiers_by_resource_and_scope() -> None:
    terminology = TerminologyResource(
        name="cell_ontology",
        concepts={
            "CL:0001": TerminologyConcept(ui="CL:0001", name="cell root"),
            "CL:0002": TerminologyConcept(ui="CL:0002", name="cell child", parent_ids=["CL:0001"]),
        },
        tree_to_ids={"CL:0001": ["CL:0001"]},
        treetop_names={"CL:0001": "cell root"},
        resource_aliases=["CL"],
        id_prefix="CL",
    )

    result = high_level_concept_counts(_target(), "high_level_concept_counts", terminology)
    scoped = high_level_concept_counts(
        _target(),
        "high_level_concept_counts",
        terminology,
        annotation_filter_name="cell",
    )

    assert result.details["n_input_ids"] == 1
    assert result.details["n_missing_ids"] == 0
    assert scoped.details["n_input_ids"] == 1
    assert scoped.value[0]["branch_code"] == "CL:0001"
    assert scoped.value[0]["proportion"] == 0.5


def test_terminology_topic_anchor_counter_uses_configured_anchor_ids() -> None:
    terminology = TerminologyResource(
        name="example",
        concepts={
            "R1": TerminologyConcept(ui="R1", name="Root A"),
            "A1": TerminologyConcept(ui="A1", name="Anchor A", parent_ids=["R1"]),
            "T1": TerminologyConcept(ui="T1", name="Term One", parent_ids=["A1"]),
        },
    )

    counter = TerminologyTopicAnchorCounter(terminology, anchor_ids={"A1": "Broad A"})

    assert counter.topic_anchor_counts("Term One") == {"Broad A": 1.0}
