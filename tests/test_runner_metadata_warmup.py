from __future__ import annotations

from types import SimpleNamespace

from corpus_benchmark.context import BenchmarkContext, MetricTarget
from corpus_benchmark.metrics import metadata_distribution  # noqa: F401
from corpus_benchmark.models.corpus import CorpusSubset, Document
from corpus_benchmark.registry import SUBSET_METRICS
from corpus_benchmark.registry import CROSS_METRICS, TERMINOLOGY_METRICS
from corpus_benchmark.runner import PlannedMetricExecution, _warm_metadata_cache


class FakeWorkspace:
    def __init__(self) -> None:
        self.calls = 0

    def get_document_metadata(self, documents: list[Document]) -> dict[str, dict[str, str]]:
        self.calls += 1
        return {document.document_id: {"journal": "Example Journal"} for document in documents}


def test_metadata_distribution_metrics_declare_metadata_dependency() -> None:
    assert getattr(SUBSET_METRICS["journal_distribution"], "requires_metadata") is True
    assert getattr(SUBSET_METRICS["journal_topic_distribution"], "requires_metadata") is True
    assert getattr(SUBSET_METRICS["journal_MeSH_topic_distribution"], "requires_metadata") is True
    assert getattr(SUBSET_METRICS["article_MeSH_topic_distribution"], "requires_metadata") is True
    assert getattr(SUBSET_METRICS["publication_year_distribution"], "requires_metadata") is True


def test_scoped_metrics_declare_annotation_scope_support() -> None:
    assert getattr(SUBSET_METRICS["label_distribution"], "supports_annotation_scope") is True
    assert getattr(SUBSET_METRICS["annotations_per_document_stats"], "supports_annotation_scope") is True
    assert getattr(CROSS_METRICS["mention_overlap"], "supports_annotation_scope") is True
    assert getattr(TERMINOLOGY_METRICS["high_level_concept_counts"], "supports_annotation_scope") is True
    assert not getattr(CROSS_METRICS["token_overlap"], "supports_annotation_scope", False)


def test_warm_metadata_cache_populates_context_once_per_target() -> None:
    workspace = FakeWorkspace()
    subset = CorpusSubset("train", [Document("doc-1")])
    context = BenchmarkContext(workspace=workspace)
    target = MetricTarget("Example_corpus", [(subset, context)])
    metric_spec = SimpleNamespace(result_name="journal_distribution")

    executions = [
        PlannedMetricExecution(
            metric_spec=metric_spec,
            metric=lambda *_args, **_kwargs: None,
            targets=(target,),
            requires_metadata=True,
        ),
        PlannedMetricExecution(
            metric_spec=metric_spec,
            metric=lambda *_args, **_kwargs: None,
            targets=(target,),
            requires_metadata=True,
        ),
    ]

    _warm_metadata_cache(executions)

    assert workspace.calls == 1
    assert context.cache["metadata(train)"] == {"doc-1": {"journal": "Example Journal"}}
