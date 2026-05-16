from __future__ import annotations

from pathlib import Path

from corpus_benchmark.context import BenchmarkContext, MetricTarget
from corpus_benchmark.metrics.terminology_coverage import concept_depth_counts, high_level_concept_counts
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


def _target_for_link(identifier: str, resource: str, label: str = "Entity") -> MetricTarget:
    annotation = Annotation(
        mention_id="a1",
        text="entity",
        spans=[AnnotationSpan(0, 6)],
        label=label,
        link=IdentifierLink(identifier=identifier, resource=resource),
    )
    passage = Passage("p1", "entity", 0, [annotation])
    subset = CorpusSubset("train", [Document("doc1", [passage])])
    context = BenchmarkContext(
        workspace=object(),
        annotation_filters={"entity": AnnotationFilter(labels={label})},
    )
    return MetricTarget("Example_corpus", [(subset, context)])


def _target_for_links(links: list[IdentifierLink], label: str = "Entity") -> MetricTarget:
    annotations = [
        Annotation(
            mention_id=f"a{i}",
            text="entity",
            spans=[AnnotationSpan(i * 10, i * 10 + 6)],
            label=label,
            link=link,
        )
        for i, link in enumerate(links, start=1)
    ]
    passage = Passage("p1", "entity entity entity", 0, annotations)
    subset = CorpusSubset("train", [Document("doc1", [passage])])
    context = BenchmarkContext(
        workspace=object(),
        annotation_filters={"entity": AnnotationFilter(labels={label})},
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


def test_high_level_concept_counts_uses_configured_term_overrides(tmp_path: Path) -> None:
    terminology = TerminologyResource(
        name="example",
        concepts={
            "R1": TerminologyConcept(ui="R1", name="Root A"),
            "A1": TerminologyConcept(ui="A1", name="Anchor A", parent_ids=["R1"]),
            "T1": TerminologyConcept(ui="T1", name="Term One", parent_ids=["A1"]),
        },
        resource_aliases=["EX"],
    )
    mapping_path = tmp_path / "mappings.yaml"
    mapping_path.write_text("Broad A:\n- Anchor A\n", encoding="utf-8")

    result = high_level_concept_counts(
        _target_for_link("T1", "EX"),
        "high_level_concept_counts",
        terminology,
        term_overrides_path=str(mapping_path),
    )

    assert result.value[0]["branch_code"] == "Broad A"
    assert result.value[0]["label"] == "Broad A"
    assert result.details["term_overrides_path"] == str(mapping_path)


def test_high_level_concept_counts_selects_scope_specific_term_overrides(tmp_path: Path) -> None:
    terminology = TerminologyResource(
        name="example",
        concepts={
            "R1": TerminologyConcept(ui="R1", name="Root A"),
            "A1": TerminologyConcept(ui="A1", name="Anchor A", parent_ids=["R1"]),
            "T1": TerminologyConcept(ui="T1", name="Term One", parent_ids=["A1"]),
        },
        resource_aliases=["EX"],
    )
    mapping_path = tmp_path / "entity_mappings.yaml"
    mapping_path.write_text("Broad A:\n- Anchor A\n", encoding="utf-8")
    target = _target_for_link("T1", "EX")

    unscoped = high_level_concept_counts(
        target,
        "high_level_concept_counts",
        terminology,
        term_override_paths_by_entity_scope={"entity": str(mapping_path)},
    )
    scoped = high_level_concept_counts(
        target,
        "high_level_concept_counts",
        terminology,
        annotation_filter_name="entity",
        term_override_paths_by_entity_scope={"entity": str(mapping_path)},
    )

    assert unscoped.value[0]["branch_code"] == "R1"
    assert unscoped.details["term_overrides_path"] is None
    assert scoped.value[0]["branch_code"] == "Broad A"
    assert scoped.details["term_overrides_path"] == str(mapping_path)


def test_high_level_concept_counts_uses_unique_corpus_concepts_for_recall() -> None:
    terminology = TerminologyResource(
        name="example",
        concepts={
            "R1": TerminologyConcept(ui="R1", name="Root A"),
            "T1": TerminologyConcept(ui="T1", name="Term One", parent_ids=["R1"]),
        },
        resource_aliases=["EX"],
    )
    target = _target_for_links(
        [
            IdentifierLink(identifier="T1", resource="EX"),
            IdentifierLink(identifier="T1", resource="EX"),
            IdentifierLink(identifier="T1", resource="EX"),
        ]
    )

    result = high_level_concept_counts(target, "high_level_concept_counts", terminology)

    assert result.details["n_input_ids"] == 3
    assert result.details["n_unique_input_ids"] == 1
    assert result.value[0]["count"] == 1
    assert result.value[0]["annotation_count"] == 3
    assert result.value[0]["proportion"] == 0.5
    assert result.value[0]["annotation_proportion"] == 1.0


def test_high_level_concept_counts_annotation_proportion_uses_all_identifiers() -> None:
    terminology = TerminologyResource(
        name="example",
        concepts={
            "R1": TerminologyConcept(ui="R1", name="Root A"),
            "T1": TerminologyConcept(ui="T1", name="Term One", parent_ids=["R1"]),
        },
        resource_aliases=["EX"],
    )
    target = _target_for_links(
        [
            IdentifierLink(identifier="T1", resource="EX"),
            IdentifierLink(identifier="T1", resource="EX"),
            IdentifierLink(identifier="OLD:1", resource="EX"),
        ]
    )

    result = high_level_concept_counts(target, "high_level_concept_counts", terminology)
    row = result.value[0]

    assert result.details["n_input_ids"] == 3
    assert result.details["n_missing_ids"] == 1
    assert row["count"] == 1
    assert row["proportion"] == 0.5
    assert row["annotation_count"] == 2
    assert row["annotation_proportion"] == round(2 / 3, 8)


def test_concept_depth_counts_reports_annotation_depth_distribution() -> None:
    terminology = TerminologyResource(
        name="example",
        concepts={
            "R1": TerminologyConcept(ui="R1", name="Root A"),
            "T1": TerminologyConcept(ui="T1", name="Term One", parent_ids=["R1"]),
            "T2": TerminologyConcept(ui="T2", name="Term Two", parent_ids=["T1"]),
        },
        resource_aliases=["EX"],
    )
    target = _target_for_links(
        [
            IdentifierLink(identifier="T1", resource="EX"),
            IdentifierLink(identifier="T1", resource="EX"),
            IdentifierLink(identifier="T2", resource="EX"),
        ]
    )

    result = concept_depth_counts(target, "concept_depth_counts", terminology)
    proportions = {row["depth"]: row["proportion"] for row in result.value}

    assert proportions[1] == 0.0
    assert proportions[2] == round(2 / 3, 8)
    assert proportions[3] == round(1 / 3, 8)
    assert sum(proportions.values()) == 1.0
