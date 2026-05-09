from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from corpus_benchmark.builtins import register_builtins
from corpus_benchmark.context import BenchmarkContext, MetricTarget
from corpus_benchmark.models.config import BatteryConfig, DatasetBundle, BenchmarkConfig
from corpus_benchmark.models.corpus import BenchmarkCorpus, DocumentIdentifierType
from corpus_benchmark.registry import (
    LOADERS,
    TERMINOLOGY_LOADERS,
    SUBSET_METRICS,
    CROSS_METRICS,
    TERMINOLOGY_METRICS,
)
from corpus_benchmark.workspace import GlobalWorkspace
from corpus_benchmark.metadata.json_record_store import JsonRecordStore
from corpus_benchmark.metadata.journal_metadata import create_journal_record_store

logger = logging.getLogger(__name__)


def _resolve_bundle(bundle: DatasetBundle, corpora: dict, contexts: dict) -> MetricTarget:
    """Helper to convert a DatasetBundle config into an actionable MetricTarget."""
    components = []
    for ref in bundle.subsets:
        corpus = corpora[ref.corpus_name]
        subset = corpus.subsets[ref.subset_name]
        context = contexts[ref.corpus_name]
        components.append((subset, context))
    return MetricTarget(name=bundle.name, components=components)


def _load_corpus(workspace: GlobalWorkspace, benchmark_name: str, benchmark_config: BenchmarkConfig) -> BenchmarkCorpus:
    # 1. Make sure files are ready. We check this first to ensure that
    # if acquisition was interrupted, we re-acquire before trusting any cache.
    logger.info(f"Loading corpus {benchmark_name}")
    was_ready = workspace.acquisition_manager.ensure_corpus_ready(benchmark_name, benchmark_config)

    # 2. Try to load from cache
    cache_path = Path(benchmark_config.cache_filename) if benchmark_config.cache_filename else None
    if was_ready and cache_path and cache_path.exists():
        try:
            logger.info(f'Loading corpus "{benchmark_name}" from cache at {cache_path}')
            return BenchmarkCorpus.from_json(cache_path)
        except Exception as e:
            logger.warning(f'Could not load cache at {cache_path} for corpus "{benchmark_name}". Starting fresh. Error was {e}')

    # 3. Load from corpus-specific formats
    loader_name = benchmark_config.loader.name
    if loader_name not in LOADERS:
        available = ", ".join(sorted(LOADERS)) or "<none>"
        raise ValueError(f"Unknown loader '{benchmark_config.loader.name}'. Available loaders: {available}")
    loader = LOADERS[loader_name]
    benchmark_corpus = loader(**benchmark_config.loader.params)
    # 4. Try to save to cache
    if cache_path:
        logger.info(f'Saving corpus "{benchmark_name}" to cache at {cache_path}')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_corpus.to_json(cache_path)
    return benchmark_corpus


def _create_document_record_store(document_store_filename: str) -> JsonRecordStore:
    document_store_path = Path(document_store_filename)
    document_store_path.parent.mkdir(parents=True, exist_ok=True)
    document_store = JsonRecordStore(
        document_store_path,
        identifier_types={
            DocumentIdentifierType.PMID,
            DocumentIdentifierType.PMCID,
            DocumentIdentifierType.DOI,
        },
        fields={
            "pub_year",
            "journal",
            "journal_id",
            "mesh_topics",
        },
        field_policies={
            "pub_year": "strict",
            "journal": "replace",
            "journal_id": "strict",
            "mesh_topics": "set_union",
        },
        identifier_normalizers={
            DocumentIdentifierType.PMID: DocumentIdentifierType.PMID.normalize,
            DocumentIdentifierType.PMCID: DocumentIdentifierType.PMCID.normalize,
            DocumentIdentifierType.DOI: DocumentIdentifierType.DOI.normalize,
        },
    )
    return document_store


def run_benchmark(battery_config: BatteryConfig) -> list[Any]:
    register_builtins()
    battery_config.validate()

    document_store = _create_document_record_store(battery_config.workspace.document_store_filename)
    journal_record_store = create_journal_record_store(battery_config.workspace.journal_store_filename)
    workspace = GlobalWorkspace(
        document_store=document_store,
        journal_record_store=journal_record_store,
        workspace_config=battery_config.workspace,
    )
    # Initialize terminologies
    for term_name, term_config in battery_config.terminologies.items():
        logger.info("Loading terminology %s", term_name)
        loader_name = term_config.name
        if loader_name not in TERMINOLOGY_LOADERS:
            available = ", ".join(sorted(TERMINOLOGY_LOADERS)) or "<none>"
            raise ValueError(f"Unknown terminology loader '{loader_name}'. Available loaders: {available}")
        loader = TERMINOLOGY_LOADERS[loader_name]
        workspace.terminologies[term_name] = loader(workspace.workspace_config, **term_config.params)

    # Load corpora
    corpora: dict[str, BenchmarkCorpus] = dict()
    contexts: dict[str, BenchmarkContext] = dict()
    for benchmark_name, benchmark_config in battery_config.corpora.items():
        benchmark_corpus = _load_corpus(workspace, benchmark_name, benchmark_config)
        document_count = sum(len(corpus_subset.documents) for corpus_subset in benchmark_corpus.subsets.values())
        logger.info(
            "Loaded %s documents in %s subsets",
            document_count,
            len(benchmark_corpus.subsets),
        )
        for filter_name, filter in benchmark_config.annotation_filters.items():
            logger.debug(
                'Runner annotation filter "%s" has definition "%s"',
                filter_name,
                filter,
            )
        corpora[benchmark_name] = benchmark_corpus
        contexts[benchmark_name] = BenchmarkContext(
            workspace=workspace,
            annotation_filters=benchmark_config.annotation_filters,
        )

    # TODO: pre-cache any necessary document metadata
    # TODO: pre-cache any necessary journal metadata

    # Run metrics
    logger.info(
        "Metrics configured: %s",
        [metric_spec.metric_name for metric_spec in battery_config.metrics],
    )
    results: list[Any] = []
    for metric_spec in battery_config.metrics:
        if not metric_spec.enabled:
            continue

        logger.info("Calculating metric %s", metric_spec.result_name)
        if metric_spec.metric_name in SUBSET_METRICS:
            metric = SUBSET_METRICS[metric_spec.metric_name]
            for bundle_name in metric_spec.target_bundles:
                logger.debug(
                    "Calculating metric %s on bundle %s",
                    metric_spec.result_name,
                    bundle_name,
                )
                bundle = battery_config.bundles[bundle_name]
                target = _resolve_bundle(bundle, corpora, contexts)
                logger.debug("... bundle resolved")

                # Execute metric
                result = metric(
                    target,
                    metric_spec.result_name,
                    **getattr(metric_spec, "params", {}),
                )
                logger.debug("... metric calculated")
                results.append(result)
        elif metric_spec.metric_name in CROSS_METRICS:
            metric = CROSS_METRICS[metric_spec.metric_name]
            suite = battery_config.comparison_suites[metric_spec.comparison_suite]
            for bundle1_name, bundle2_name in suite.bundle_pairs:
                logger.debug(
                    "Calculating metric %s on bundles %s and %s",
                    metric_spec.result_name,
                    bundle1_name,
                    bundle2_name,
                )
                bundle1 = battery_config.bundles[bundle1_name]
                bundle2 = battery_config.bundles[bundle2_name]
                target1 = _resolve_bundle(bundle1, corpora, contexts)
                target2 = _resolve_bundle(bundle2, corpora, contexts)
                logger.debug("... bundles resolved")

                # Execute cross-metric (requires metrics designed for two targets)
                result = metric(
                    target1,
                    target2,
                    metric_spec.result_name,
                    **getattr(metric_spec, "params", {}),
                )
                logger.debug("... metric calculated")
                results.append(result)
        elif metric_spec.metric_name in TERMINOLOGY_METRICS:
            metric = TERMINOLOGY_METRICS[metric_spec.metric_name]
            params = getattr(metric_spec, "params", {})
            term_name = params.get("terminology_name")
            if not term_name or term_name not in workspace.terminologies:
                # Fallback to the first loaded terminology if only one exists
                if len(workspace.terminologies) == 1:
                    term_name = list(workspace.terminologies.keys())[0]
                    terminology = workspace.terminologies[term_name]
                else:
                    available = ", ".join(sorted(workspace.terminologies)) or "<none>"
                    raise ValueError(f"Metric {metric_spec.metric_name} requires a terminology_name param matching a loaded terminology. " f"Available terminologies: {available}")
            else:
                terminology = workspace.terminologies[term_name]

            for bundle_name in metric_spec.target_bundles:
                logger.debug(
                    "Calculating metric %s on bundle %s",
                    metric_spec.result_name,
                    bundle_name,
                )
                bundle = battery_config.bundles[bundle_name]
                target = _resolve_bundle(bundle, corpora, contexts)
                logger.debug("... bundle resolved")
                result = metric(
                    target,
                    metric_spec.result_name,
                    terminology=terminology,
                    **params,
                )
                logger.debug("... metric calculated")
                results.append(result)
        else:
            available_metrics = []
            available_metrics.extend(SUBSET_METRICS)
            available_metrics.extend(CROSS_METRICS)
            available_metrics.extend(TERMINOLOGY_METRICS)
            available = ", ".join(sorted(available_metrics)) or "<none>"
            raise ValueError(f"Unknown metric '{metric_spec.metric_name}'. Available metrics: {available}")

    document_store.save()
    journal_record_store.save()

    # Display context usage
    logger.debug("Context usage:")
    for benchmark_name, benchmark_context in contexts.items():
        logger.debug("%s:", benchmark_name)
        for context_key, usage_count in benchmark_context.usage_counts.items():
            logger.debug("  %s: %s", context_key, usage_count)

    return results
