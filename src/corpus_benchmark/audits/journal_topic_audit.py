from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

from corpus_benchmark.audits.audit_helpers import (
    add_weighted_counts,
    load_json_records,
    load_journal_name_topics,
    load_mesh_term_overrides,
    load_terminology,
    round_floats,
    write_json_payload,
)
from corpus_benchmark.cli import load_battery_config
from corpus_benchmark.metadata.journal_topics import classify_journal
from corpus_benchmark.models.config import LoaderSpec
from corpus_benchmark.models.terminologies import (
    TerminologyTopicAnchorCounter,
    topic_treetop_names,
    TerminologyResource,
)

DEFAULT_CONFIG_PATH = Path("configs/metadata_stats.yaml")


def build_journal_topic_audit(
    journal_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    terminology: TerminologyResource,
    mesh_term_overrides: dict[str, str],
    journal_name_topics: dict[str, list[str]],
    include_journal_id: bool = False,
) -> list[dict[str, Any]]:
    journal_id_counts = Counter(
        record.get("data", {}).get("journal_id")
        for record in metadata_records
        if record.get("data", {}).get("journal_id") is not None
    )
    root_counter = TerminologyTopicAnchorCounter(
        terminology,
        term_overrides=mesh_term_overrides,
        fallback_name_topics=journal_name_topics,
    )
    mesh_treetop_cache: dict[str, list[str]] = {}

    audit_records = []
    for record in journal_records:
        journal_id = record.get("record_id")
        journal_data = record.get("data", {})
        full_name = journal_data.get("name") or journal_data.get("abbreviation") or "Unknown"

        mesh_topics = journal_data.get("mesh_topics", [])
        for mesh_topic in mesh_topics or []:
            if mesh_topic not in mesh_treetop_cache:
                mesh_treetop_cache[mesh_topic] = topic_treetop_names(
                    terminology,
                    mesh_topic,
                )

        mesh_root_counts = root_counter.counts_for_record_topics(full_name, mesh_topics or [])

        record_payload = {
            "journal_full_name": full_name,
            "metadata_usage_count": journal_id_counts[journal_id],
            "mesh_topics": mesh_topics,
            "classify_journal_topic": classify_journal(full_name),
            "mesh_root_counts": round_floats(mesh_root_counts),
        }
        if include_journal_id:
            record_payload["journal_id"] = journal_id

        audit_records.append(record_payload)

    return sorted(
        audit_records,
        key=lambda item: (-item["metadata_usage_count"], item["journal_full_name"]),
    )


def _build_weighted_mesh_topic_counts(
    journal_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    terminology: TerminologyResource,
    mesh_term_overrides: dict[str, str],
    journal_name_topics: dict[str, list[str]],
) -> dict[str, float]:
    journal_id_counts = Counter(
        record.get("data", {}).get("journal_id")
        for record in metadata_records
        if record.get("data", {}).get("journal_id") is not None
    )
    root_counter = TerminologyTopicAnchorCounter(
        terminology,
        term_overrides=mesh_term_overrides,
        fallback_name_topics=journal_name_topics,
    )
    total_counts: dict[str, float] = {}

    for record in journal_records:
        journal_usage_count = journal_id_counts[record.get("record_id")]
        if journal_usage_count == 0:
            continue

        mesh_topics = record.get("data", {}).get("mesh_topics", []) or []
        journal_data = record.get("data", {})
        full_name = journal_data.get("name") or journal_data.get("abbreviation") or "Unknown"
        journal_counts = root_counter.counts_for_record_topics(full_name, mesh_topics)
        add_weighted_counts(total_counts, journal_counts, float(journal_usage_count))

    return {
        name: round_floats(count)
        for name, count in sorted(
            total_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def build_mesh_topic_root_counts(
    journal_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    terminology: TerminologyResource,
    mesh_term_overrides: dict[str, str],
    journal_name_topics: dict[str, list[str]],
) -> dict[str, float]:
    return _build_weighted_mesh_topic_counts(
        journal_records,
        metadata_records,
        terminology,
        mesh_term_overrides,
        journal_name_topics,
    )


def build_mesh_term_root_frequencies(
    journal_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    terminology: TerminologyResource,
    mesh_term_overrides: dict[str, str],
) -> dict[str, dict[str, Any]]:
    journal_id_counts = Counter(
        record.get("data", {}).get("journal_id")
        for record in metadata_records
        if record.get("data", {}).get("journal_id") is not None
    )
    root_counter = TerminologyTopicAnchorCounter(
        terminology,
        term_overrides=mesh_term_overrides,
    )
    total_counts: dict[str, dict[str, Any]] = {}

    for record in journal_records:
        journal_usage_count = journal_id_counts[record.get("record_id")]
        if journal_usage_count == 0:
            continue

        mesh_topics = record.get("data", {}).get("mesh_topics", []) or []
        if not mesh_topics:
            continue

        topic_weight = float(journal_usage_count) / len(mesh_topics)
        for mesh_topic in mesh_topics:
            topic_counts = total_counts.setdefault(
                mesh_topic,
                {
                    "frequency": 0.0,
                    "roots": {},
                },
            )
            topic_counts["frequency"] += topic_weight
            root_counts = root_counter.topic_anchor_counts(mesh_topic)
            add_weighted_counts(topic_counts["roots"], root_counts, topic_weight)

    return {
        mesh_topic: {
            "frequency": round_floats(counts["frequency"]),
            "roots": {
                root: round_floats(count)
                for root, count in sorted(
                    counts["roots"].items(),
                    key=lambda item: (-item[1], item[0]),
                )
            },
        }
        for mesh_topic, counts in sorted(
            total_counts.items(),
            key=lambda item: (-item[1]["frequency"], item[0]),
        )
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Output per-journal topic mappings from data/journals.json, "
            "metadata usage counts from data/metadata.json, and MeSH treetop topics."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Battery config containing workspace paths and the MeSH terminology loader.",
    )
    parser.add_argument(
        "--terminology-name",
        default="mesh",
        help="Terminology entry name from the config to use for MeSH lookups.",
    )
    parser.add_argument(
        "--journal-topics",
        type=Path,
        required=True,
        help="YAML mapping of journal topic names to lists of MeSH terms.",
    )
    parser.add_argument(
        "--journal-name-topics",
        type=Path,
        required=True,
        help="JSON mapping of journal names without MeSH topics to high-level topic names.",
    )
    parser.add_argument(
        "--journals",
        type=Path,
        help="Override the journal JSON record-store path.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Override the document metadata JSON record-store path.",
    )
    parser.add_argument(
        "--include-journal-id",
        action="store_true",
        help="Include the journal record_id in the output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--mesh-root-counts-output",
        type=Path,
        help="Write weighted MeSH root topic counts to this path.",
    )
    parser.add_argument(
        "--mesh-term-root-frequencies-output",
        type=Path,
        help="Write per-MeSH-term journal frequencies and root contributions to this path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    battery_config = load_battery_config(args.config)
    workspace_config = battery_config.workspace
    terminology_spec = battery_config.terminologies.get(args.terminology_name)

    journal_path = args.journals or Path(workspace_config.journal_store_filename)
    metadata_path = args.metadata or Path(workspace_config.document_store_filename)
    mesh_term_overrides = load_mesh_term_overrides(args.journal_topics)
    journal_name_topics = load_journal_name_topics(args.journal_name_topics)

    terminology = load_terminology(
        workspace_config,
        args.terminology_name,
        terminology_spec,
        default_spec=LoaderSpec(name="mesh_xml", params={"year": 2026}),
    )
    journal_records = load_json_records(journal_path)
    metadata_records = load_json_records(metadata_path)

    audit_records = build_journal_topic_audit(
        journal_records,
        metadata_records,
        terminology,
        mesh_term_overrides,
        journal_name_topics,
        include_journal_id=args.include_journal_id,
    )
    mesh_root_counts = build_mesh_topic_root_counts(
        journal_records,
        metadata_records,
        terminology,
        mesh_term_overrides,
        journal_name_topics,
    )
    mesh_term_root_frequencies = build_mesh_term_root_frequencies(
        journal_records,
        metadata_records,
        terminology,
        mesh_term_overrides,
    )
    write_json_payload(args.output, audit_records, sort_keys=True)
    write_json_payload(args.mesh_root_counts_output, mesh_root_counts)
    write_json_payload(args.mesh_term_root_frequencies_output, mesh_term_root_frequencies)


if __name__ == "__main__":
    main(sys.argv[1:])
