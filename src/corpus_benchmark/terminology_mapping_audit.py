from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from corpus_benchmark.builtins import register_builtins
from corpus_benchmark.cli import load_battery_config
from corpus_benchmark.models.config import LoaderSpec, WorkspaceConfig
from corpus_benchmark.models.terminologies import TerminologyResource
from corpus_benchmark.models.terminologies import TerminologyTopicAnchorCounter
from corpus_benchmark.models.terminologies import load_topic_term_overrides
from corpus_benchmark.registry import TERMINOLOGY_LOADERS

PRECISION = 4
DEFAULT_CONFIG_PATH = Path("configs/terminology_coverage.yaml")


def _load_terminology(
    workspace_config: WorkspaceConfig,
    terminology_name: str,
    terminology_spec: LoaderSpec,
) -> TerminologyResource:
    register_builtins()

    loader = TERMINOLOGY_LOADERS.get(terminology_spec.name)
    if loader is None:
        available = ", ".join(sorted(TERMINOLOGY_LOADERS)) or "<none>"
        raise ValueError(
            f"Unknown terminology loader {terminology_spec.name!r} for {terminology_name!r}. "
            f"Available terminology loaders: {available}"
        )
    return loader(workspace_config, **terminology_spec.params)


def _round_counts(counts: dict[str, float]) -> dict[str, float]:
    return {
        name: round(count, PRECISION)
        for name, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def build_terminology_mapping_audit(
    terminology: TerminologyResource,
    term_overrides: dict[str, str],
) -> dict[str, Any]:
    counter = TerminologyTopicAnchorCounter(terminology, term_overrides=term_overrides)
    configured_topics = sorted(counter.configured_anchor_topics)

    concept_mappings = []
    for concept in sorted(terminology.concepts.values(), key=lambda item: (item.name.casefold(), item.ui)):
        counts = counter.concept_anchor_counts(concept.ui)
        concept_mappings.append(
            {
                "concept_id": concept.ui,
                "name": concept.name,
                "anchor_counts": _round_counts(counts),
            }
        )

    high_level_totals = counter.get_global_counts_by_anchor()
    for topic in configured_topics:
        high_level_totals.setdefault(topic, 0.0)

    return {
        "terminology": terminology.name,
        "n_concepts": len(terminology.concepts),
        "configured_topics": configured_topics,
        "concept_mappings": concept_mappings,
        "high_level_totals": _round_counts(high_level_totals),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit how terminology concepts map to configured high-level topics, "
            "and summarize total high-level mappings across the full terminology."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Battery config containing workspace paths and terminology loader definitions.",
    )
    parser.add_argument(
        "--terminology-name",
        required=True,
        help="Terminology entry name from the config to load.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="YAML mapping of high-level topic names to lists of terminology terms.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    battery_config = load_battery_config(args.config)
    terminology_spec = battery_config.terminologies.get(args.terminology_name)
    if terminology_spec is None:
        available = ", ".join(sorted(battery_config.terminologies)) or "<none>"
        raise ValueError(
            f"Config {args.config} does not define terminology {args.terminology_name!r}. "
            f"Available terminologies: {available}"
        )

    terminology = _load_terminology(
        battery_config.workspace,
        args.terminology_name,
        terminology_spec,
    )
    term_overrides = load_topic_term_overrides(args.mapping)
    audit = build_terminology_mapping_audit(terminology, term_overrides)
    audit["terminology_config_name"] = args.terminology_name
    audit["mapping_path"] = str(args.mapping)

    payload = json.dumps(audit, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main(sys.argv[1:])
