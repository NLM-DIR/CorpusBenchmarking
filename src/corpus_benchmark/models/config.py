from __future__ import annotations

from collections import Counter
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from corpus_benchmark.models.filters import AnnotationFilter

logger = logging.getLogger(__name__)

APP_NAME_VER = "CorpusBenchmarking/0.1"

@dataclass(slots=True)
class LoaderSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


def _coerce_loader_spec(data: LoaderSpec | str | dict[str, Any]) -> LoaderSpec:
    if isinstance(data, LoaderSpec):
        return data
    if isinstance(data, str):
        return LoaderSpec(name=data)
    return LoaderSpec(name=str(data["name"]), params=dict(data.get("params", {})))


@dataclass(slots=True)
class AcquisitionSpec:
    source_urls: list[str] = field(default_factory=list)
    format: str | None = None
    converter: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AcquisitionSpec":
        urls = data.get("source_urls", [])
        # Support both 'source_url' (single) and 'source_urls' (list) for convenience
        if "source_url" in data:
            urls.append(data["source_url"])
        logger.debug("Parsed acquisition spec with %s source URLs", len(urls))
        return cls(
            source_urls=urls,
            format=data.get("format"),
            converter=data.get("converter"),
        )


@dataclass(slots=True)
class SubsetRef:
    """Explicit pointer to a specific subset within a specific corpus."""

    corpus_name: str
    subset_name: str


@dataclass(slots=True)
class DatasetBundle:
    """A logical grouping of subsets to be treated as a single unit by a metric."""

    name: str
    subsets: list[SubsetRef] = field(default_factory=list)


@dataclass(slots=True)
class ComparisonSuite:
    """A named collection defining pairs of DatasetBundles to compare."""

    name: str
    bundle_pairs: list[tuple[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class MetricSpec:
    metric_name: str
    result_name: str | None = None

    # Replaced 'subsets' with explicit bundle or comparison suite references
    target_bundles: list[str] | None = None
    comparison_suite: str | None = None

    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.result_name is None:
            self.result_name = self.metric_name


@dataclass(slots=True)
class BenchmarkConfig:
    name: str
    loader: LoaderSpec
    annotation_filters: dict[str, AnnotationFilter] = field(default_factory=dict)
    acquisition: AcquisitionSpec | None = None
    cache_filename: str | None = None


@dataclass(slots=True)
class WorkspaceConfig:
    """Global configuration for the benchmarking workspace and caches."""

    document_store_filename: str = "data/metadata.json"
    journal_store_filename: str = "data/journals.json"
    corpora_download_dir: str = "corpora/"
    terminology_dir: str = "data/terminologies/"
    document_fetchers: dict[str, list[LoaderSpec]] = field(
        default_factory=lambda: {
            "pmid": [LoaderSpec("pubmed_eutils")],
            "pmcid": [LoaderSpec("pmc_eutils_efetch")],
        }
    )

    def __post_init__(self):
        normalized_fetchers = {}
        for id_type, specs in self.document_fetchers.items():
            id_type_value = getattr(id_type, "value", str(id_type)).lower()
            if isinstance(specs, (LoaderSpec, str, dict)):
                specs = [specs]
            normalized_fetchers[id_type_value] = [_coerce_loader_spec(spec) for spec in specs]
        self.document_fetchers = normalized_fetchers

    def validate(self) -> None:
        from corpus_benchmark.builtins import register_builtins
        from corpus_benchmark.models.corpus import DocumentIdentifierType
        from corpus_benchmark.registry import DOCUMENT_FETCHERS

        register_builtins()

        if not str(self.document_store_filename).strip():
            raise ValueError("workspace.document_store_filename must not be empty.")
        if Path(self.document_store_filename).name == "":
            raise ValueError("workspace.document_store_filename must point to a file.")
        if not str(self.journal_store_filename).strip():
            raise ValueError("workspace.journal_store_filename must not be empty.")
        if Path(self.journal_store_filename).name == "":
            raise ValueError("workspace.journal_store_filename must point to a file.")
        if not str(self.corpora_download_dir).strip():
            raise ValueError("workspace.corpora_download_dir must not be empty.")
        if not str(self.terminology_dir).strip():
            raise ValueError("workspace.terminology_dir must not be empty.")

        for raw_id_type, fetcher_specs in self.document_fetchers.items():
            try:
                id_type = DocumentIdentifierType(str(raw_id_type).lower())
            except ValueError as exc:
                allowed = ", ".join(item.value for item in DocumentIdentifierType)
                raise ValueError(f"Unknown workspace.document_fetchers identifier type " f"{raw_id_type!r}. Expected one of: {allowed}") from exc

            for fetcher_spec in fetcher_specs:
                if fetcher_spec.name not in DOCUMENT_FETCHERS:
                    available = ", ".join(sorted(DOCUMENT_FETCHERS)) or "<none>"
                    raise ValueError(f"Unknown document fetcher {fetcher_spec.name!r} for " f"identifier type {id_type.value!r}. Available document " f"fetchers: {available}")

                fetcher_cls = DOCUMENT_FETCHERS[fetcher_spec.name]
                try:
                    fetcher = fetcher_cls(**fetcher_spec.params)
                except TypeError as exc:
                    raise ValueError(f"Invalid params for document fetcher " f"{fetcher_spec.name!r}: {exc}") from exc

                if fetcher.supported_id_type != id_type:
                    raise ValueError(f"Document fetcher {fetcher_spec.name!r} supports " f"{fetcher.supported_id_type.value!r}, but it was " f"configured for {id_type.value!r}.")


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    filename: str | None = None


@dataclass(slots=True)
class BatteryConfig:
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    corpora: dict[str, BenchmarkConfig] = field(default_factory=dict)
    terminologies: dict[str, LoaderSpec] = field(default_factory=dict)
    bundles: dict[str, DatasetBundle] = field(default_factory=dict)
    comparison_suites: dict[str, ComparisonSuite] = field(default_factory=dict)
    metrics: list[MetricSpec] = field(default_factory=list)
    output_path: str | None = None
    entity_scope_config: str | None = None

    def __post_init__(self):
        # Ensure all metric result_names are unique
        metric_result_name_counts = Counter(metric.result_name for metric in self.metrics)
        non_unique = {name for name, count in metric_result_name_counts.items() if count > 1}
        if len(non_unique) > 0:
            raise ValueError(f"Metric result names must be unique: {non_unique}")
        logger.debug("Validated battery config with %s metrics", len(self.metrics))

    def validate(self) -> None:
        from corpus_benchmark.builtins import register_builtins
        from corpus_benchmark.registry import (
            CROSS_METRICS,
            LOADERS,
            SUBSET_METRICS,
            TERMINOLOGY_LOADERS,
            TERMINOLOGY_METRICS,
        )

        register_builtins()
        self.workspace.validate()

        for corpus_name, corpus_config in self.corpora.items():
            loader_name = corpus_config.loader.name
            if loader_name not in LOADERS:
                available = ", ".join(sorted(LOADERS)) or "<none>"
                raise ValueError(f"Corpus {corpus_name!r} uses unknown loader " f"{loader_name!r}. Available loaders: {available}")

        for term_name, term_config in self.terminologies.items():
            loader_name = term_config.name
            if loader_name not in TERMINOLOGY_LOADERS:
                available = ", ".join(sorted(TERMINOLOGY_LOADERS)) or "<none>"
                raise ValueError(f"Terminology {term_name!r} uses unknown loader " f"{loader_name!r}. Available terminology loaders: {available}")

        for bundle_name, bundle in self.bundles.items():
            if not bundle.subsets:
                raise ValueError(f"Bundle {bundle_name!r} must contain at least one subset reference.")
            for ref in bundle.subsets:
                if ref.corpus_name not in self.corpora:
                    available = ", ".join(sorted(self.corpora)) or "<none>"
                    raise ValueError(f"Bundle {bundle_name!r} references unknown corpus " f"{ref.corpus_name!r}. Available corpora: {available}")

        for suite_name, suite in self.comparison_suites.items():
            if not suite.bundle_pairs:
                raise ValueError(f"Comparison suite {suite_name!r} must contain at least one bundle pair.")
            for bundle1, bundle2 in suite.bundle_pairs:
                for bundle_name in (bundle1, bundle2):
                    if bundle_name not in self.bundles:
                        available = ", ".join(sorted(self.bundles)) or "<none>"
                        raise ValueError(f"Comparison suite {suite_name!r} references " f"unknown bundle {bundle_name!r}. Available bundles: " f"{available}")

        for metric_spec in self.metrics:
            if not metric_spec.enabled:
                continue

            metric_name = metric_spec.metric_name
            if metric_name in SUBSET_METRICS:
                self._validate_metric_target_bundles(metric_spec)
            elif metric_name in CROSS_METRICS:
                if not metric_spec.comparison_suite:
                    raise ValueError(f"Cross metric {metric_name!r} requires comparison_suite.")
                if metric_spec.comparison_suite not in self.comparison_suites:
                    available = ", ".join(sorted(self.comparison_suites)) or "<none>"
                    raise ValueError(
                        f"Metric {metric_name!r} references unknown comparison " f"suite {metric_spec.comparison_suite!r}. Available " f"comparison suites: {available}"
                    )
            elif metric_name in TERMINOLOGY_METRICS:
                self._validate_metric_target_bundles(metric_spec)
                term_name = metric_spec.params.get("terminology_name")
                if term_name is not None and term_name not in self.terminologies:
                    available = ", ".join(sorted(self.terminologies)) or "<none>"
                    raise ValueError(f"Metric {metric_name!r} references unknown terminology " f"{term_name!r}. Available terminologies: {available}")
                if term_name is None and len(self.terminologies) != 1:
                    raise ValueError(f"Metric {metric_name!r} requires params.terminology_name " "unless exactly one terminology is configured.")
            else:
                available_metrics = []
                available_metrics.extend(SUBSET_METRICS)
                available_metrics.extend(CROSS_METRICS)
                available_metrics.extend(TERMINOLOGY_METRICS)
                available = ", ".join(sorted(available_metrics)) or "<none>"
                raise ValueError(f"Unknown metric {metric_name!r}. Available metrics: {available}")

    def _validate_metric_target_bundles(self, metric_spec: MetricSpec) -> None:
        if not metric_spec.target_bundles:
            raise ValueError(f"Metric {metric_spec.metric_name!r} requires target_bundles.")
        for bundle_name in metric_spec.target_bundles:
            if bundle_name not in self.bundles:
                available = ", ".join(sorted(self.bundles)) or "<none>"
                raise ValueError(f"Metric {metric_spec.metric_name!r} references unknown " f"bundle {bundle_name!r}. Available bundles: {available}")
