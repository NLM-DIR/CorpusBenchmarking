from __future__ import annotations

from corpus_benchmark.models.terminologies import TerminologyConcept, TerminologyResource
from corpus_benchmark.terminology_mapping_audit import build_terminology_mapping_audit


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
    assert mappings["A1"] == {"Broad A": 1.0}
    assert mappings["T1"] == {"Broad A": 1.0}
    assert mappings["R1"] == {}
    assert mappings["T2"] == {}
    assert audit["high_level_totals"] == {"Broad A": 2.0, "Broad Missing": 0.0}
    assert audit["configured_topics"] == ["Broad A", "Broad Missing"]
