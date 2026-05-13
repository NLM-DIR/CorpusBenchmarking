from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from corpus_benchmark.builtins import register_builtins
from corpus_benchmark.cli import load_battery_config
from corpus_benchmark.journal_topic_audit import _load_json_records
from corpus_benchmark.journal_topic_audit import _load_terminology
from corpus_benchmark.journal_topic_audit import _round_floats
from corpus_benchmark.models.config import LoaderSpec, WorkspaceConfig
from corpus_benchmark.models.terminologies import TerminologyResource
from corpus_benchmark.models.terminologies import TerminologyTopicAnchorCounter
from corpus_benchmark.models.terminologies import load_name_topic_fallbacks
from corpus_benchmark.models.terminologies import load_topic_term_overrides

DEFAULT_CONFIG_PATH = Path("configs/metadata_stats.yaml")


def _add_weighted_counts(
    target: dict[str, float],
    source: dict[str, float],
    weight: float,
) -> None:
    for name, count in source.items():
        target[name] = target.get(name, 0.0) + count * weight


def _normalize_topic_counts(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    return {name: count / total for name, count in counts.items()}


def _topic_counts_and_unmapped(
    mesh_topics: list[str],
    counter: TerminologyTopicAnchorCounter,
) -> tuple[dict[str, float], list[str]]:
    if not mesh_topics:
        return {}, []

    topic_weight = 1.0 / len(mesh_topics)
    counts: dict[str, float] = {}
    unmapped = []
    for mesh_topic in mesh_topics:
        topic_counts = _normalize_topic_counts(counter.topic_anchor_counts(mesh_topic))
        if topic_counts:
            _add_weighted_counts(counts, topic_counts, topic_weight)
        else:
            unmapped.append(mesh_topic)
    return counts, unmapped


def _journal_fallback_counts(
    journal_data: dict[str, Any] | None,
    journal_counter: TerminologyTopicAnchorCounter,
) -> tuple[dict[str, float], str]:
    if journal_data is None:
        return {}, "unknown"

    journal_name = journal_data.get("name") or journal_data.get("abbreviation") or "Unknown"
    mesh_counts = _normalize_topic_counts(journal_counter.topic_counts(journal_data.get("mesh_topics", []) or []))
    if mesh_counts:
        return mesh_counts, "journal_mesh"

    name_counts = _normalize_topic_counts(journal_counter.fallback_topic_counts(journal_name))
    if name_counts:
        return name_counts, "journal_name"
    return {}, "unknown"


def _source_label(article_mesh_topics: list[str], unmapped: list[str], fallback_source: str) -> str:
    if not article_mesh_topics:
        return fallback_source
    if not unmapped:
        return "article_mesh"
    if len(unmapped) == len(article_mesh_topics):
        return fallback_source
    return f"article_mesh+{fallback_source}"


def build_article_topic_audit(
    journal_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    terminology: TerminologyResource,
    article_mesh_term_overrides: dict[str, str],
    journal_mesh_term_overrides: dict[str, str],
    journal_name_topics: dict[str, list[str]],
    include_document_id: bool = False,
) -> list[dict[str, Any]]:
    article_counter = TerminologyTopicAnchorCounter(
        terminology,
        term_overrides=article_mesh_term_overrides,
    )
    journal_counter = TerminologyTopicAnchorCounter(
        terminology,
        term_overrides=journal_mesh_term_overrides,
        fallback_name_topics=journal_name_topics,
    )
    journals_by_id = {record.get("record_id"): record.get("data", {}) for record in journal_records}

    audit_records = []
    for record in metadata_records:
        article_data = record.get("data", {})
        journal_id = article_data.get("journal_id")
        journal_data = journals_by_id.get(journal_id)
        journal_name = (journal_data or {}).get("name") or (journal_data or {}).get("abbreviation") or article_data.get("journal") or "Unknown"
        article_mesh_topics = article_data.get("mesh_topics", []) or []

        article_counts, unmapped = _topic_counts_and_unmapped(article_mesh_topics, article_counter)
        fallback_counts, fallback_source = _journal_fallback_counts(journal_data, journal_counter)

        root_counts = dict(article_counts)
        if article_mesh_topics:
            fallback_weight = len(unmapped) / len(article_mesh_topics)
            if fallback_weight and fallback_counts:
                _add_weighted_counts(root_counts, fallback_counts, fallback_weight)
            elif fallback_weight:
                root_counts["Unknown"] = root_counts.get("Unknown", 0.0) + fallback_weight
        elif fallback_counts:
            _add_weighted_counts(root_counts, fallback_counts, 1.0)
        else:
            root_counts["Unknown"] = 1.0

        payload = {
            "article_mesh_topics": article_mesh_topics,
            "unmapped_article_mesh_topics": unmapped,
            "journal_id": journal_id,
            "journal_name": journal_name,
            "source": _source_label(article_mesh_topics, unmapped, fallback_source),
            "fallback_source": fallback_source,
            "article_root_counts_without_fallback": _round_floats(article_counts),
            "fallback_root_counts": _round_floats(fallback_counts),
            "root_counts": _round_floats(root_counts),
        }
        if include_document_id:
            payload["document_id"] = record.get("record_id")
        audit_records.append(payload)

    return audit_records


def build_article_topic_root_counts(
    journal_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    terminology: TerminologyResource,
    article_mesh_term_overrides: dict[str, str],
    journal_mesh_term_overrides: dict[str, str],
    journal_name_topics: dict[str, list[str]],
    include_fallback: bool = True,
) -> dict[str, float]:
    total_counts: dict[str, float] = {}
    for record in build_article_topic_audit(
        journal_records,
        metadata_records,
        terminology,
        article_mesh_term_overrides,
        journal_mesh_term_overrides,
        journal_name_topics,
    ):
        counts = record["root_counts"] if include_fallback else record["article_root_counts_without_fallback"]
        _add_weighted_counts(total_counts, counts, 1.0)

    return {
        name: _round_floats(count)
        for name, count in sorted(
            total_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def build_unmapped_article_mesh_term_frequencies(
    metadata_records: list[dict[str, Any]],
    terminology: TerminologyResource,
    article_mesh_term_overrides: dict[str, str],
) -> dict[str, float]:
    article_counter = TerminologyTopicAnchorCounter(
        terminology,
        term_overrides=article_mesh_term_overrides,
    )
    term_counts: Counter[str] = Counter()
    for record in metadata_records:
        mesh_topics = record.get("data", {}).get("mesh_topics", []) or []
        if not mesh_topics:
            continue
        topic_weight = 1.0 / len(mesh_topics)
        for mesh_topic in mesh_topics:
            if not article_counter.topic_anchor_counts(mesh_topic):
                term_counts[mesh_topic] += topic_weight

    return {
        name: _round_floats(count)
        for name, count in sorted(
            term_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Output per-article topic mappings from data/metadata.json, including unmapped article MeSH terms and journal fallback contributions.")
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
        "--article-topics",
        type=Path,
        required=True,
        help="YAML mapping of article topic names to lists of MeSH terms.",
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
        help="JSON mapping of journal names without usable MeSH topics to high-level topic names.",
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
        "--include-document-id",
        action="store_true",
        help="Include the metadata record_id in the per-article output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write per-article JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--topic-root-counts-output",
        type=Path,
        help="Write aggregate article topic counts with fallback to this path.",
    )
    parser.add_argument(
        "--topic-root-counts-without-fallback-output",
        type=Path,
        help="Write aggregate article topic counts before journal fallback to this path.",
    )
    parser.add_argument(
        "--unmapped-terms-output",
        type=Path,
        help="Write weighted frequencies of unmapped article MeSH terms to this path.",
    )
    return parser.parse_args(argv)


def _load_workspace_terminology(
    workspace_config: WorkspaceConfig,
    terminology_name: str,
    terminology_spec: LoaderSpec | None,
) -> TerminologyResource:
    register_builtins()
    return _load_terminology(workspace_config, terminology_name, terminology_spec)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    battery_config = load_battery_config(args.config)
    workspace_config = battery_config.workspace
    terminology_spec = battery_config.terminologies.get(args.terminology_name)

    journal_path = args.journals or Path(workspace_config.journal_store_filename)
    metadata_path = args.metadata or Path(workspace_config.document_store_filename)
    article_mesh_term_overrides = load_topic_term_overrides(args.article_topics)
    journal_mesh_term_overrides = load_topic_term_overrides(args.journal_topics)
    journal_name_topics = load_name_topic_fallbacks(args.journal_name_topics)

    terminology = _load_workspace_terminology(
        workspace_config,
        args.terminology_name,
        terminology_spec,
    )
    journal_records = _load_json_records(journal_path)
    metadata_records = _load_json_records(metadata_path)

    audit_records = build_article_topic_audit(
        journal_records,
        metadata_records,
        terminology,
        article_mesh_term_overrides,
        journal_mesh_term_overrides,
        journal_name_topics,
        include_document_id=args.include_document_id,
    )
    root_counts = build_article_topic_root_counts(
        journal_records,
        metadata_records,
        terminology,
        article_mesh_term_overrides,
        journal_mesh_term_overrides,
        journal_name_topics,
        include_fallback=True,
    )
    root_counts_without_fallback = build_article_topic_root_counts(
        journal_records,
        metadata_records,
        terminology,
        article_mesh_term_overrides,
        journal_mesh_term_overrides,
        journal_name_topics,
        include_fallback=False,
    )
    unmapped_terms = build_unmapped_article_mesh_term_frequencies(
        metadata_records,
        terminology,
        article_mesh_term_overrides,
    )

    payload = json.dumps(audit_records, indent=2, sort_keys=True)
    root_counts_payload = json.dumps(root_counts, indent=2)
    root_counts_without_fallback_payload = json.dumps(root_counts_without_fallback, indent=2)
    unmapped_terms_payload = json.dumps(unmapped_terms, indent=2)

    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    if args.topic_root_counts_output:
        args.topic_root_counts_output.write_text(root_counts_payload + "\n", encoding="utf-8")
    else:
        print(root_counts_payload)

    if args.topic_root_counts_without_fallback_output:
        args.topic_root_counts_without_fallback_output.write_text(root_counts_without_fallback_payload + "\n", encoding="utf-8")
    else:
        print(root_counts_without_fallback_payload)

    if args.unmapped_terms_output:
        args.unmapped_terms_output.write_text(unmapped_terms_payload + "\n", encoding="utf-8")
    else:
        print(unmapped_terms_payload)


if __name__ == "__main__":
    main(sys.argv[1:])
