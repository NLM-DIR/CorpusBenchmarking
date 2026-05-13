from __future__ import annotations

from corpus_benchmark.context import BenchmarkContext, MetricTarget
from corpus_benchmark.metrics.basic_stats import (
    annotations_per_document_stats,
    sentences_per_document_stats,
    tokens_per_document_stats,
)
from corpus_benchmark.metrics.basic_counts import token_count
from corpus_benchmark.models.corpus import Annotation, AnnotationSpan, CorpusSubset, Document, Passage
from corpus_benchmark.models.filters import AnnotationFilter

def _target() -> MetricTarget:
    annotations1 = [
        Annotation(
            mention_id="a1",
            text="cell",
            spans=[AnnotationSpan(0, 4)],
            label="Cell",
            link=None,
        ),
        Annotation(
            mention_id="a2",
            text="chemical",
            spans=[AnnotationSpan(5, 13)],
            label="Chemical",
            link=None,
        ),
    ]
    passage1 = Passage("p1", "cell chemical", 0, annotations1)
    
    annotations2 = [
        Annotation(
            mention_id="a3",
            text="cell",
            spans=[AnnotationSpan(0, 4)],
            label="Cell",
            link=None,
        ),
    ]
    passage2 = Passage("p2", "cell", 0, annotations2)
    
    subset = CorpusSubset("train", [
        Document("doc1", [passage1]),
        Document("doc2", [passage2])
    ])
    context = BenchmarkContext(
        workspace=object(),
        annotation_filters={"cell": AnnotationFilter(labels={"Cell"})},
    )
    return MetricTarget("Example_corpus", [(subset, context)])

def test_annotations_per_document_stats() -> None:
    target = _target()
    result = annotations_per_document_stats(target, "annotations_per_document_stats")
    
    # doc1 has 2, doc2 has 1. Mean = 1.5, Max = 2, Min = 1, Count = 2
    assert result.value["mean"] == 1.5
    assert result.value["max"] == 2
    assert result.value["min"] == 1
    assert result.value["count"] == 2

def test_annotations_per_document_stats_scoped() -> None:
    target = _target()
    result = annotations_per_document_stats(target, "annotations_per_document_stats", annotation_filter_name="cell")
    
    # doc1 has 1 (cell), doc2 has 1 (cell). Mean = 1.0
    assert result.value["mean"] == 1.0
    assert result.value["count"] == 2

def test_token_count() -> None:
    target = _target()
    result = token_count(target, "token_count")
    
    # passage1: "cell chemical" -> 2 tokens
    # passage2: "cell" -> 1 token
    # Total = 3
    assert result.value == 3

def test_sentences_per_document_stats() -> None:
    target = _target()
    result = sentences_per_document_stats(target, "sentences_per_document_stats")
    
    # doc1: "cell chemical" -> 1 sentence
    # doc2: "cell" -> 1 sentence
    # Mean = 1.0, Count = 2
    assert result.value["mean"] == 1.0
    assert result.value["count"] == 2

def test_tokens_per_document_stats() -> None:
    target = _target()
    result = tokens_per_document_stats(target, "tokens_per_document_stats")
    
    # doc1: "cell chemical" -> 2 tokens
    # doc2: "cell" -> 1 token
    # Mean = 1.5, Count = 2
    assert result.value["mean"] == 1.5
    assert result.value["count"] == 2
