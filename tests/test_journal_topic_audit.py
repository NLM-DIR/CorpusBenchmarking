from __future__ import annotations

from corpus_benchmark.audits.journal_topic_audit import build_journal_topic_audit
from corpus_benchmark.audits.journal_topic_audit import build_mesh_topic_root_counts
from corpus_benchmark.audits.journal_topic_audit import build_mesh_term_root_frequencies
from corpus_benchmark.models.terminologies import TerminologyConcept
from corpus_benchmark.models.terminologies import TerminologyResource


def _test_terminology() -> TerminologyResource:
    return TerminologyResource(
        name="mesh",
        concepts={
            "R1": TerminologyConcept(ui="R1", name="Root A"),
            "R2": TerminologyConcept(ui="R2", name="Root B"),
            "T1": TerminologyConcept(ui="T1", name="Term One", parent_ids=["R1"]),
            "T2": TerminologyConcept(ui="T2", name="Term Two", parent_ids=["R2"]),
        },
    )


def test_build_mesh_topic_root_counts_uses_journal_name_topics_when_mesh_topics_are_empty() -> None:
    terminology = _test_terminology()
    journal_records = [
        {
            "record_id": "journal-1",
            "data": {"name": "Mesh Journal", "mesh_topics": ["Term One", "Term Two"]},
        },
        {
            "record_id": "journal-2",
            "data": {"name": "Fallback Journal", "mesh_topics": []},
        },
        {
            "record_id": "journal-3",
            "data": {"name": "Unmapped Journal", "mesh_topics": []},
        },
    ]
    metadata_records = [
        {"record_id": "doc-1", "data": {"journal_id": "journal-1"}},
        {"record_id": "doc-2", "data": {"journal_id": "journal-1"}},
        {"record_id": "doc-3", "data": {"journal_id": "journal-2"}},
        {"record_id": "doc-4", "data": {"journal_id": "journal-2"}},
        {"record_id": "doc-5", "data": {"journal_id": "journal-2"}},
        {"record_id": "doc-6", "data": {"journal_id": "journal-3"}},
    ]

    assert build_mesh_topic_root_counts(
        journal_records,
        metadata_records,
        terminology,
        {},
        {"Fallback Journal": ["Fallback A", "Fallback B"]},
    ) == {
        "Fallback A": 1.5,
        "Fallback B": 1.5,
        "Root A": 1.0,
        "Root B": 1.0,
    }


def test_build_journal_topic_audit_uses_journal_name_topics_when_mesh_topics_are_empty() -> None:
    terminology = _test_terminology()
    audit_records = build_journal_topic_audit(
        [
            {
                "record_id": "journal-1",
                "data": {"name": "Fallback Journal", "mesh_topics": []},
            },
        ],
        [
            {"record_id": "doc-1", "data": {"journal_id": "journal-1"}},
        ],
        terminology,
        {},
        {"Fallback Journal": ["Fallback A", "Fallback B"]},
    )

    assert audit_records[0]["mesh_root_counts"] == {
        "Fallback A": 0.5,
        "Fallback B": 0.5,
    }


def test_build_mesh_term_root_frequencies_uses_per_journal_fractional_counts() -> None:
    terminology = _test_terminology()
    journal_records = [
        {
            "record_id": "journal-1",
            "data": {"mesh_topics": ["Term One", "Term Two"]},
        },
        {
            "record_id": "journal-2",
            "data": {"mesh_topics": ["Term One"]},
        },
        {
            "record_id": "unused-journal",
            "data": {"mesh_topics": ["Unused Term"]},
        },
    ]
    metadata_records = [
        {"record_id": "doc-1", "data": {"journal_id": "journal-1"}},
        {"record_id": "doc-2", "data": {"journal_id": "journal-1"}},
        {"record_id": "doc-3", "data": {"journal_id": "journal-2"}},
    ]

    assert build_mesh_term_root_frequencies(journal_records, metadata_records, terminology, {}) == {
        "Term One": {
            "frequency": 2.0,
            "roots": {"Root A": 2.0},
        },
        "Term Two": {
            "frequency": 1.0,
            "roots": {"Root B": 1.0},
        },
    }
