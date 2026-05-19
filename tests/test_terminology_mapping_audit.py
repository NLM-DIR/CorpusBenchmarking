from __future__ import annotations

from corpus_benchmark.models.terminologies import TerminologyConcept, TerminologyResource
from corpus_benchmark.audits.terminology_mapping_audit import build_terminology_mapping_audit


def test_build_terminology_mapping_audit_reports_concept_mappings_and_totals() -> None:
    terminology = TerminologyResource(
        name="example",
        concepts={
            "R1": TerminologyConcept(ui="R1", name="Root A"),
            "A1": TerminologyConcept(ui="A1", name="Anchor A", parent_ids=["R1"]),
            "T1": TerminologyConcept(ui="T1", name="Term One", parent_ids=["A1"]),
            "T2": TerminologyConcept(ui="T2", name="Term Two", parent_ids=["R1"]),
        },
    )

    audit = build_terminology_mapping_audit(
        terminology,
        {
            "Anchor A": "Broad A",
            "Missing Anchor": "Broad Missing",
        },
    )

    mappings = {item["concept_id"]: item["anchor_counts"] for item in audit["concept_mappings"]}
    mapping_sums = {item["concept_id"]: item["anchor_count_sum"] for item in audit["concept_mappings"]}
    assert mappings["A1"] == {"Broad A": 1.0}
    assert mappings["T1"] == {"Broad A": 1.0}
    assert mappings["R1"] == {}
    assert mappings["T2"] == {}
    assert mapping_sums["A1"] == 1.0
    assert mapping_sums["T2"] == 0.0
    assert audit["high_level_totals"] == {"Broad A": 2.0, "Broad Missing": 0.0}
    assert audit["configured_topics"] == ["Broad A", "Broad Missing"]


def test_build_terminology_mapping_audit_reports_normalized_mixed_mapped_concepts() -> None:
    terminology = TerminologyResource(
        name="mesh_like",
        concepts={
            "D1": TerminologyConcept(ui="D1", name="Disease Descriptor One", tree_numbers=["C01.001"]),
            "D2": TerminologyConcept(ui="D2", name="Disease Descriptor Two", tree_numbers=["C01.002"]),
            "O1": TerminologyConcept(ui="O1", name="Chemical Descriptor", tree_numbers=["D08.001"]),
            "S1": TerminologyConcept(ui="S1", name="Mixed Supplemental", mapped_ui_ids=["D1", "O1", "D2"]),
            "S2": TerminologyConcept(ui="S2", name="Out Of Scope Supplemental", mapped_ui_ids=["O1"]),
        },
        tree_to_ids={"C01.001": ["D1"], "C01.002": ["D2"], "D08.001": ["O1"]},
    )

    audit = build_terminology_mapping_audit(
        terminology,
        {
            "Disease Descriptor One": "Topic One",
            "Disease Descriptor Two": "Topic Two",
        },
    )

    mappings = {item["concept_id"]: item for item in audit["concept_mappings"]}
    assert mappings["S1"]["anchor_counts"] == {"Topic One": 0.5, "Topic Two": 0.5}
    assert mappings["S1"]["anchor_count_sum"] == 1.0
    assert mappings["S2"]["anchor_counts"] == {}
    assert mappings["S2"]["anchor_count_sum"] == 0
    assert audit["high_level_totals"] == {"Topic One": 1.5, "Topic Two": 1.5}
