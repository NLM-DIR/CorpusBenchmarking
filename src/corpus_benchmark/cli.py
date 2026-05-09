from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from corpus_benchmark.models.config import BatteryConfig, WorkspaceConfig, BenchmarkConfig, LoaderSpec, AcquisitionSpec, MetricSpec, DatasetBundle, SubsetRef, ComparisonSuite, LoggingConfig
from corpus_benchmark.models.filters import AnnotationFilter
from corpus_benchmark.runner import run_benchmark

logger = logging.getLogger(__name__)

def setup_logging(config: LoggingConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        format=config.format,
        filename=config.filename,
    )

def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    with open(path, "r", encoding="utf-8") as fp:
        raw_config: dict[str, Any] = yaml.safe_load(fp)

    annotation_filters = {
        name: AnnotationFilter.from_config_dict(raw_filter)
        for name, raw_filter in raw_config.get("filters", {}).items()
    }

    # Load acquisition safely
    acq_raw = raw_config.get("acquisition")
    acquisition = AcquisitionSpec.from_dict(acq_raw) if acq_raw else None

    return BenchmarkConfig(
        name=str(raw_config["name"]),
        loader=LoaderSpec(**raw_config["loader"]),
        annotation_filters=annotation_filters,
        acquisition=acquisition, 
        cache_filename=raw_config.get("cache_filename")
    )

def load_battery_config(path: str | Path) -> BatteryConfig:
    with open(path, "r", encoding="utf-8") as fp:
        raw_config: dict[str, Any] = yaml.safe_load(fp)

    # Load workspace config (using default if missing from YAML)
    workspace_dict = raw_config.get("workspace", {})
    workspace_config = WorkspaceConfig(**workspace_dict)

    # Load logging config
    logging_dict = raw_config.get("logging", {})
    logging_config = LoggingConfig(**logging_dict)

    # Load Corpora
    corpora = {
        corpus_name: load_benchmark_config(corpus_path)
        for corpus_name, corpus_path in raw_config.get("corpora", {}).items()
    }

    # Load Terminologies
    terminologies = {
        term_name: LoaderSpec(**term_dict)
        for term_name, term_dict in raw_config.get("terminologies", {}).items()
    }

    # Load Dataset Bundles
    bundles = {
        bundle_name: DatasetBundle(
            name=bundle_name,
            subsets=[SubsetRef(**ref_dict) for ref_dict in subset_list]
        )
        for bundle_name, subset_list in raw_config.get("bundles", {}).items()
    }

    # Load Comparison Suites
    comparison_suites = {
        suite_name: ComparisonSuite(
            name=suite_name,
            # Ensure YAML lists of 2 items are safely cast to Python tuples
            bundle_pairs=[(pair[0], pair[1]) for pair in pairs_list]
        )
        for suite_name, pairs_list in raw_config.get("comparison_suites", {}).items()
    }

    # 4. Initialize the full config
    return BatteryConfig(
        workspace=workspace_config,
        logging=logging_config,
        corpora=corpora,
        terminologies=terminologies,
        bundles=bundles,
        comparison_suites=comparison_suites,
        metrics=[MetricSpec(**metric) for metric in raw_config.get("metrics", [])],
        output_path=raw_config.get("output_path"),
        entity_scope_config=raw_config.get("entity_scope_config"),
    )

def group_results(results, result_dicts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    results_dict: dict[str, list[dict[str, Any]]] = dict()
    for result, result_dict in zip(results, result_dicts):
        key = result.result_key()
        if not key in results_dict:
            results_dict[key] = list()
        results_dict[key].append(result_dict)
    return results_dict


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m corpus_benchmark.cli <config.yaml>", file=sys.stderr)
        raise SystemExit(2)

    battery_config = load_battery_config(Path(sys.argv[1]))
    battery_config.validate()
    setup_logging(battery_config.logging)

    logger.info(f"Loading battery config from {sys.argv[1]}")
    logger.info(f"Loaded {len(battery_config.corpora)} corpora, {len(battery_config.metrics)} metrics")

    results = run_benchmark(battery_config)
    result_dicts = [r.to_dict() for r in results]
    payload = group_results(results, result_dicts)

    if battery_config.output_path:
        output_path = Path(battery_config.output_path)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    logger.info("Done.")

if __name__ == "__main__":
    main()
