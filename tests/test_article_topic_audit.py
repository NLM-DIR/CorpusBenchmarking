from __future__ import annotations

from corpus_benchmark.article_topic_audit import build_article_topic_audit
from corpus_benchmark.article_topic_audit import build_article_topic_root_counts
from corpus_benchmark.article_topic_audit import build_unmapped_article_mesh_term_frequencies
from corpus_benchmark.models.terminologies import TerminologyConcept
from corpus_benchmark.models.terminologies import TerminologyResource


def _test_terminology() -> TerminologyResource:
    return TerminologyResource(
        name="mesh",
        concepts={
            "T1": TerminologyConcept(ui="T1", name="Term One"),
            "T2": TerminologyConcept(ui="T2", name="Journal Term"),
        },
    )


def test_build_article_topic_audit_falls_back_only_for_unmapped_article_terms() -> None:
    terminology = _test_terminology()
    journal_records = [
        {
            "record_id": "journal-1",
            "data": {"name": "Mesh Journal", "mesh_topics": ["Journal Term"]},
        },
        {
            "record_id": "journal-2",
            "data": {"name": "Fallback Journal", "mesh_topics": []},
        },
    ]
    metadata_records = [
        {
            "record_id": "doc-1",
            "data": {"journal_id": "journal-1", "mesh_topics": ["Term One", "Unmapped Term"]},
        },
        {
            "record_id": "doc-2",
            "data": {"journal_id": "journal-2", "mesh_topics": ["Unmapped Term"]},
        },
        {
            "record_id": "doc-3",
            "data": {"journal_id": "journal-1", "mesh_topics": []},
        },
        {
            "record_id": "doc-4",
            "data": {"journal_id": None, "mesh_topics": ["Unmapped Term"]},
        },
    ]

    audit_records = build_article_topic_audit(
        journal_records,
        metadata_records,
        terminology,
        {"Term One": "Article Topic"},
        {"Journal Term": "Journal Topic"},
        {"Fallback Journal": ["Name Topic A", "Name Topic B"]},
        include_document_id=True,
    )

    assert audit_records[0] == {
        "document_id": "doc-1",
        "article_mesh_topics": ["Term One", "Unmapped Term"],
        "unmapped_article_mesh_topics": ["Unmapped Term"],
        "journal_id": "journal-1",
        "journal_name": "Mesh Journal",
        "source": "article_mesh+journal_mesh",
        "fallback_source": "journal_mesh",
        "article_root_counts_without_fallback": {"Article Topic": 0.5},
        "fallback_root_counts": {"Journal Topic": 1.0},
        "root_counts": {"Article Topic": 0.5, "Journal Topic": 0.5},
    }
    assert audit_records[1]["source"] == "journal_name"
    assert audit_records[1]["root_counts"] == {"Name Topic A": 0.5, "Name Topic B": 0.5}
    assert audit_records[2]["source"] == "journal_mesh"
    assert audit_records[2]["root_counts"] == {"Journal Topic": 1.0}
    assert audit_records[3]["source"] == "unknown"
    assert audit_records[3]["root_counts"] == {"Unknown": 1.0}


def test_build_article_topic_root_counts_can_report_with_and_without_fallback() -> None:
    terminology = _test_terminology()
    journal_records = [
        {
            "record_id": "journal-1",
            "data": {"name": "Mesh Journal", "mesh_topics": ["Journal Term"]},
        },
    ]
    metadata_records = [
        {
            "record_id": "doc-1",
            "data": {"journal_id": "journal-1", "mesh_topics": ["Term One", "Unmapped Term"]},
        },
        {
            "record_id": "doc-2",
            "data": {"journal_id": "journal-1", "mesh_topics": []},
        },
    ]

    common_args = (
        journal_records,
        metadata_records,
        terminology,
        {"Term One": "Article Topic"},
        {"Journal Term": "Journal Topic"},
        {},
    )

    assert build_article_topic_root_counts(*common_args, include_fallback=True) == {
        "Journal Topic": 1.5,
        "Article Topic": 0.5,
    }
    assert build_article_topic_root_counts(*common_args, include_fallback=False) == {
        "Article Topic": 0.5,
    }


def test_build_unmapped_article_mesh_term_frequencies_uses_per_article_fractional_counts() -> None:
    terminology = _test_terminology()
    metadata_records = [
        {
            "record_id": "doc-1",
            "data": {"mesh_topics": ["Term One", "Unmapped Term"]},
        },
        {
            "record_id": "doc-2",
            "data": {"mesh_topics": ["Unmapped Term"]},
        },
    ]

    assert build_unmapped_article_mesh_term_frequencies(
        metadata_records,
        terminology,
        {"Term One": "Article Topic"},
    ) == {"Unmapped Term": 1.5}
