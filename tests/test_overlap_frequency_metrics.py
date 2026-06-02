from __future__ import annotations

from corpus_benchmark.context import BenchmarkContext, MetricTarget
from corpus_benchmark.metrics.overlaps import (
    identifier_overlap_by_frequency,
    mention_overlap_by_frequency,
    mention_token_overlap_by_frequency,
    token_overlap_by_frequency,
)
from corpus_benchmark.models.corpus import Annotation, AnnotationSpan, CorpusSubset, Document, IdentifierLink, Passage


def _target(name: str, text: str, annotations: list[tuple[str, str]]) -> MetricTarget:
    passage_annotations = [
        Annotation(
            mention_id=f"{name}-{index}",
            text=mention,
            spans=[AnnotationSpan(index, index + len(mention))],
            label="Entity",
            link=IdentifierLink(identifier=identifier, resource="EX"),
        )
        for index, (mention, identifier) in enumerate(annotations)
    ]
    subset = CorpusSubset(name, [Document(f"{name}-doc", [Passage(f"{name}-passage", text, 0, passage_annotations)])])
    return MetricTarget(name, [(subset, BenchmarkContext(workspace=object()))])


def test_token_overlap_by_frequency_reports_jaccard_per_frequency_bin() -> None:
    train = _target("train", "shared rare trainonly trainonly repeated repeated repeated", [])
    test = _target("test", "shared rare testonly testonly repeated repeated", [])

    result = token_overlap_by_frequency(train, test, "token_overlap_by_frequency")

    assert result.value["(2, 4)"] == 0.5
    assert result.value["(4, 16)"] == 1.0
    assert result.details["bin_counts"]["(2, 4)"] == [4, 2]
    assert result.details["bin_counts"]["(4, 16)"] == [1, 1]


def test_scoped_overlap_by_frequency_metrics_report_jaccard_per_frequency_bin() -> None:
    train = _target(
        "train",
        "alpha beta gamma",
        [
            ("shared phrase", "ID:1"),
            ("trainunique", "ID:2"),
            ("trainunique", "ID:2"),
            ("repeat repeat", "ID:3"),
        ],
    )
    test = _target(
        "test",
        "alpha delta gamma",
        [
            ("shared phrase", "ID:1"),
            ("testunique", "ID:4"),
            ("testunique", "ID:4"),
            ("repeat repeat", "ID:3"),
        ],
    )

    mention_result = mention_overlap_by_frequency(train, test, "mention_overlap_by_frequency")
    mention_token_result = mention_token_overlap_by_frequency(train, test, "mention_token_overlap_by_frequency")
    identifier_result = identifier_overlap_by_frequency(train, test, "identifier_overlap_by_frequency")

    assert mention_result.value["(2, 4)"] == 0.5
    assert mention_result.details["bin_counts"]["(2, 4)"] == [4, 2]
    assert mention_token_result.value["(2, 4)"] == 0.5
    assert mention_token_result.details["bin_counts"]["(2, 4)"] == [4, 2]
    assert identifier_result.value["(2, 4)"] == 0.5
    assert identifier_result.details["bin_counts"]["(2, 4)"] == [4, 2]
