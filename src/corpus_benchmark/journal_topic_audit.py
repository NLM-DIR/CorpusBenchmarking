from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import yaml

from corpus_benchmark.builtins import register_builtins
from corpus_benchmark.cli import load_battery_config
from corpus_benchmark.metadata.journal_topics import classify_journal
from corpus_benchmark.metrics.metadata_distribution import _mesh_topic_treetop_names
from corpus_benchmark.models.config import LoaderSpec, WorkspaceConfig
from corpus_benchmark.models.terminologies import TerminologyConcept
from corpus_benchmark.models.terminologies import TerminologyResource
from corpus_benchmark.registry import TERMINOLOGY_LOADERS

DEFAULT_CONFIG_PATH = Path("configs/metadata_stats.yaml")


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a JSON record-store 'records' list")
    return records


def load_mesh_term_overrides(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping of journal topics to MeSH terms")

    overrides: dict[str, str] = {}
    for topic, mesh_terms in payload.items():
        if not isinstance(topic, str):
            raise ValueError(f"{path} contains a non-string journal topic: {topic!r}")
        if not isinstance(mesh_terms, list):
            raise ValueError(f"{path} topic {topic!r} must contain a list of MeSH terms")
        for mesh_term in mesh_terms:
            if not isinstance(mesh_term, str):
                raise ValueError(f"{path} topic {topic!r} contains a non-string MeSH term: {mesh_term!r}")
            existing_topic = overrides.get(mesh_term)
            if existing_topic is not None and existing_topic != topic:
                raise ValueError(f"{path} maps MeSH term {mesh_term!r} to both {existing_topic!r} and {topic!r}")
            overrides[mesh_term] = topic
    return overrides


def load_journal_name_topics(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON mapping of journal names to topic lists")

    journal_topics: dict[str, list[str]] = {}
    for journal_name, topics in payload.items():
        if not isinstance(journal_name, str):
            raise ValueError(f"{path} contains a non-string journal name: {journal_name!r}")
        if not isinstance(topics, list) or not topics:
            raise ValueError(f"{path} journal {journal_name!r} must contain a non-empty list of topics")
        journal_topics[journal_name] = []
        for topic in topics:
            if not isinstance(topic, str):
                raise ValueError(f"{path} journal {journal_name!r} contains a non-string topic: {topic!r}")
            journal_topics[journal_name].append(topic)
    return journal_topics


def _load_terminology(
    workspace_config: WorkspaceConfig,
    terminology_name: str,
    terminology_spec: LoaderSpec | None,
) -> TerminologyResource:
    register_builtins()

    spec = terminology_spec or LoaderSpec(name="mesh_xml", params={"year": 2026})
    loader = TERMINOLOGY_LOADERS.get(spec.name)
    if loader is None:
        available = ", ".join(sorted(TERMINOLOGY_LOADERS)) or "<none>"
        raise ValueError(f"Unknown terminology loader {spec.name!r} for {terminology_name!r}. " f"Available terminology loaders: {available}")
    return loader(workspace_config, **spec.params)


def _build_mesh_concept_overrides(
    terminology: TerminologyResource,
    mesh_term_overrides: dict[str, str],
) -> dict[str, str]:
    mesh_concept_overrides: dict[str, str] = {}
    for mesh_term, topic in mesh_term_overrides.items():
        for concept_id in terminology.get_concept_ids_by_name(mesh_term):
            mesh_concept_overrides[concept_id] = topic
    return mesh_concept_overrides


def build_journal_topic_audit(
    journal_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    terminology: TerminologyResource,
    mesh_term_overrides: dict[str, str],
    journal_name_topics: dict[str, list[str]],
) -> list[dict[str, Any]]:
    journal_id_counts = Counter(record.get("data", {}).get("journal_id") for record in metadata_records if record.get("data", {}).get("journal_id") is not None)
    mesh_concept_overrides = _build_mesh_concept_overrides(terminology, mesh_term_overrides)
    mesh_treetop_cache: dict[str, list[str]] = {}
    root_id_cache: dict[str, dict[str, float]] = {}
    root_name_cache: dict[str, dict[str, float]] = {}

    audit_records = []
    for record in journal_records:
        journal_id = record.get("record_id")
        journal_data = record.get("data", {})
        full_name = journal_data.get("name") or journal_data.get("abbreviation") or "Unknown"

        mesh_topics = journal_data.get("mesh_topics", [])
        for mesh_topic in mesh_topics or []:
            if mesh_topic not in mesh_treetop_cache:
                mesh_treetop_cache[mesh_topic] = _mesh_topic_treetop_names(
                    terminology,
                    mesh_topic,
                )

        if mesh_topics:
            mesh_root_counts = _journal_mesh_topic_counts(
                terminology,
                mesh_topics,
                root_id_cache,
                root_name_cache,
                _mesh_topic_root_counts_by_name,
                mesh_concept_overrides,
            )
        else:
            mesh_root_counts = _journal_name_topic_root_counts(full_name, journal_name_topics)

        audit_records.append(
            {
                "journal_id": journal_id,
                "journal_full_name": full_name,
                "metadata_usage_count": journal_id_counts[journal_id],
                "mesh_topics": mesh_topics,
                "classify_journal_topic": classify_journal(full_name),
                "mesh_root_counts": mesh_root_counts,
            }
        )

    return sorted(
        audit_records,
        key=lambda item: (-item["metadata_usage_count"], item["journal_full_name"]),
    )


def _add_weighted_counts(
    target: dict[str, float],
    source: dict[str, float],
    weight: float,
) -> None:
    for name, count in source.items():
        target[name] = target.get(name, 0.0) + count * weight


def _add_weighted_nested_counts(
    target: dict[str, dict[str, float]],
    source: dict[str, dict[str, float]],
    weight: float,
) -> None:
    for topic, child_counts in source.items():
        topic_counts = target.setdefault(topic, {})
        _add_weighted_counts(topic_counts, child_counts, weight)


def _topic_parent_ids(concept: TerminologyConcept) -> list[str]:
    if concept.parent_ids:
        return concept.parent_ids
    return concept.mapped_ui_ids


def _mesh_topic_ancestor_counts_by_id(
    terminology: TerminologyResource,
    ui: str,
    cache: dict[str, dict[str, float]],
    active: set[str],
    mesh_concept_overrides: dict[str, str],
) -> dict[str, float]:
    if ui in cache:
        return cache[ui]
    if ui in mesh_concept_overrides:
        cache[ui] = {mesh_concept_overrides[ui]: 1.0}
        return cache[ui]
    if ui in active:
        return {}

    concept = terminology.get_concept(ui)
    if concept is None:
        cache[ui] = {}
        return cache[ui]

    active.add(ui)
    counts = {concept.name: 1.0}
    parent_ids = _topic_parent_ids(concept)
    if parent_ids:
        parent_weight = 1.0 / len(parent_ids)
        for parent_id in parent_ids:
            parent_counts = _mesh_topic_ancestor_counts_by_id(
                terminology,
                parent_id,
                cache,
                active,
                mesh_concept_overrides,
            )
            _add_weighted_counts(counts, parent_counts, parent_weight)
    active.remove(ui)

    cache[ui] = counts
    return counts


def _mesh_topic_root_counts_by_id(
    terminology: TerminologyResource,
    ui: str,
    cache: dict[str, dict[str, float]],
    active: set[str],
    mesh_concept_overrides: dict[str, str],
) -> dict[str, float]:
    if ui in cache:
        return cache[ui]
    if ui in mesh_concept_overrides:
        cache[ui] = {mesh_concept_overrides[ui]: 1.0}
        return cache[ui]
    if ui in active:
        return {}

    concept = terminology.get_concept(ui)
    if concept is None:
        cache[ui] = {}
        return cache[ui]

    active.add(ui)
    parent_ids = _topic_parent_ids(concept)
    if parent_ids:
        parent_weight = 1.0 / len(parent_ids)
        counts: dict[str, float] = {}
        for parent_id in parent_ids:
            parent_counts = _mesh_topic_root_counts_by_id(
                terminology,
                parent_id,
                cache,
                active,
                mesh_concept_overrides,
            )
            _add_weighted_counts(counts, parent_counts, parent_weight)
    else:
        counts = {concept.name: 1.0}
    active.remove(ui)

    cache[ui] = counts
    return counts


def _mesh_topic_root_counts_by_name(
    terminology: TerminologyResource,
    mesh_topic_name: str,
    id_cache: dict[str, dict[str, float]],
    name_cache: dict[str, dict[str, float]],
    mesh_concept_overrides: dict[str, str],
) -> dict[str, float]:
    if mesh_topic_name in name_cache:
        return name_cache[mesh_topic_name]

    concept_ids = terminology.get_concept_ids_by_name(mesh_topic_name)
    counts: dict[str, float] = {}
    if concept_ids:
        concept_weight = 1.0 / len(concept_ids)
        for concept_id in concept_ids:
            concept_counts = _mesh_topic_root_counts_by_id(
                terminology,
                concept_id,
                id_cache,
                set(),
                mesh_concept_overrides,
            )
            _add_weighted_counts(counts, concept_counts, concept_weight)

    name_cache[mesh_topic_name] = counts
    return counts


def _is_topic_boundary(
    concept: TerminologyConcept | None,
    ui: str,
    mesh_concept_overrides: dict[str, str],
) -> bool:
    return concept is not None and (ui in mesh_concept_overrides or not _topic_parent_ids(concept))


def _mesh_topic_root_child_counts_by_id(
    terminology: TerminologyResource,
    ui: str,
    cache: dict[str, dict[str, dict[str, float]]],
    root_cache: dict[str, dict[str, float]],
    active: set[str],
    mesh_concept_overrides: dict[str, str],
) -> dict[str, dict[str, float]]:
    if ui in cache:
        return cache[ui]
    if ui in active:
        return {}

    concept = terminology.get_concept(ui)
    if concept is None:
        cache[ui] = {}
        return cache[ui]

    if ui in mesh_concept_overrides:
        cache[ui] = {mesh_concept_overrides[ui]: {concept.name: 1.0}}
        return cache[ui]

    parent_ids = _topic_parent_ids(concept)
    if not parent_ids:
        cache[ui] = {concept.name: {concept.name: 1.0}}
        return cache[ui]

    active.add(ui)
    counts: dict[str, dict[str, float]] = {}
    parent_weight = 1.0 / len(parent_ids)
    for parent_id in parent_ids:
        parent_concept = terminology.get_concept(parent_id)
        if _is_topic_boundary(parent_concept, parent_id, mesh_concept_overrides):
            parent_root_counts = _mesh_topic_root_counts_by_id(
                terminology,
                parent_id,
                root_cache,
                active,
                mesh_concept_overrides,
            )
            for topic, root_count in parent_root_counts.items():
                topic_counts = counts.setdefault(topic, {})
                topic_counts[concept.name] = topic_counts.get(concept.name, 0.0) + root_count * parent_weight
        else:
            parent_counts = _mesh_topic_root_child_counts_by_id(
                terminology,
                parent_id,
                cache,
                root_cache,
                active,
                mesh_concept_overrides,
            )
            _add_weighted_nested_counts(counts, parent_counts, parent_weight)
    active.remove(ui)

    cache[ui] = counts
    return counts


def _journal_mesh_topic_counts(
    terminology: TerminologyResource,
    mesh_topics: list[str],
    id_cache: dict[str, dict[str, float]],
    name_cache: dict[str, dict[str, float]],
    count_func,
    mesh_concept_overrides: dict[str, str],
) -> dict[str, float]:
    if not mesh_topics:
        return {}

    topic_weight = 1.0 / len(mesh_topics)
    journal_counts: dict[str, float] = {}
    for mesh_topic in mesh_topics:
        topic_counts = count_func(
            terminology,
            mesh_topic,
            id_cache,
            name_cache,
            mesh_concept_overrides,
        )
        _add_weighted_counts(journal_counts, topic_counts, topic_weight)
    return {
        name: count
        for name, count in sorted(
            journal_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def _journal_name_topic_root_counts(
    journal_name: str,
    journal_name_topics: dict[str, list[str]],
) -> dict[str, float]:
    topics = journal_name_topics.get(journal_name, [])
    if not topics:
        return {}

    topic_weight = 1.0 / len(topics)
    return {
        topic: topic_weight
        for topic in sorted(topics)
    }


def _build_weighted_mesh_topic_counts(
    journal_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    terminology: TerminologyResource,
    mesh_term_overrides: dict[str, str],
    journal_name_topics: dict[str, list[str]],
    count_func,
) -> dict[str, float]:
    journal_id_counts = Counter(record.get("data", {}).get("journal_id") for record in metadata_records if record.get("data", {}).get("journal_id") is not None)
    mesh_concept_overrides = _build_mesh_concept_overrides(terminology, mesh_term_overrides)
    id_cache: dict[str, dict[str, float]] = {}
    name_cache: dict[str, dict[str, float]] = {}
    total_counts: dict[str, float] = {}

    for record in journal_records:
        journal_usage_count = journal_id_counts[record.get("record_id")]
        if journal_usage_count == 0:
            continue

        mesh_topics = record.get("data", {}).get("mesh_topics", []) or []
        if mesh_topics:
            journal_counts = _journal_mesh_topic_counts(
                terminology,
                mesh_topics,
                id_cache,
                name_cache,
                count_func,
                mesh_concept_overrides,
            )
        else:
            journal_data = record.get("data", {})
            full_name = journal_data.get("name") or journal_data.get("abbreviation") or "Unknown"
            journal_counts = _journal_name_topic_root_counts(full_name, journal_name_topics)
        _add_weighted_counts(total_counts, journal_counts, float(journal_usage_count))

    return {
        name: count
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
        _mesh_topic_root_counts_by_name,
    )


def build_mesh_term_root_frequencies(
    journal_records: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    terminology: TerminologyResource,
    mesh_term_overrides: dict[str, str],
) -> dict[str, dict[str, Any]]:
    journal_id_counts = Counter(record.get("data", {}).get("journal_id") for record in metadata_records if record.get("data", {}).get("journal_id") is not None)
    mesh_concept_overrides = _build_mesh_concept_overrides(terminology, mesh_term_overrides)
    id_cache: dict[str, dict[str, float]] = {}
    name_cache: dict[str, dict[str, float]] = {}
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
            root_counts = _mesh_topic_root_counts_by_name(
                terminology,
                mesh_topic,
                id_cache,
                name_cache,
                mesh_concept_overrides,
            )
            _add_weighted_counts(topic_counts["roots"], root_counts, topic_weight)

    return {
        mesh_topic: {
            "frequency": counts["frequency"],
            "roots": {
                root: count
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
        description=("Output per-journal topic mappings from data/journals.json, " "metadata usage counts from data/metadata.json, and MeSH treetop topics.")
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

    terminology = _load_terminology(
        workspace_config,
        args.terminology_name,
        terminology_spec,
    )
    journal_records = _load_json_records(journal_path)
    metadata_records = _load_json_records(metadata_path)

    audit_records = build_journal_topic_audit(
        journal_records,
        metadata_records,
        terminology,
        mesh_term_overrides,
        journal_name_topics,
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
    payload = json.dumps(audit_records, indent=2, sort_keys=True)
    root_counts_payload = json.dumps(mesh_root_counts, indent=2)
    term_root_frequencies_payload = json.dumps(mesh_term_root_frequencies, indent=2)

    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    if args.mesh_root_counts_output:
        args.mesh_root_counts_output.write_text(root_counts_payload + "\n", encoding="utf-8")
    else:
        print(root_counts_payload)

    if args.mesh_term_root_frequencies_output:
        args.mesh_term_root_frequencies_output.write_text(term_root_frequencies_payload + "\n", encoding="utf-8")
    else:
        print(term_root_frequencies_payload)


if __name__ == "__main__":
    main(sys.argv[1:])
