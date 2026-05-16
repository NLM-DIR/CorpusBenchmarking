from __future__ import annotations

import collections
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)
_TerminologyCacheSignature = tuple[int, int]
_NAME_INDEX_CACHE: Dict[int, tuple[_TerminologyCacheSignature, Dict[str, List[str]]]] = {}
_ID_INDEX_CACHE: Dict[int, tuple[_TerminologyCacheSignature, Dict[str, str]]] = {}
_DEPTH_CACHE: Dict[int, tuple[_TerminologyCacheSignature, Dict[str, int]]] = {}
_TOP_ANCESTOR_CACHE: Dict[int, tuple[_TerminologyCacheSignature, Dict[str, List[str]]]] = {}
_GLOBAL_BRANCH_COUNT_CACHE: Dict[int, tuple[int, Dict[str, float]]] = {}
_GLOBAL_DEPTH_COUNT_CACHE: Dict[int, tuple[int, Dict[int, float]]] = {}
NOT_FOUND_LIMIT = 10
PROGRESS_LOG_MIN_TOTAL = 10_000
PROGRESS_LOG_INTERVAL = 50_000


@dataclass(slots=True)
class TerminologyConcept:
    ui: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    tree_numbers: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    scope_note: Optional[str] = None
    mapped_ui_ids: List[str] = field(default_factory=list)
    alt_ids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TerminologyResource:
    name: str
    concepts: Dict[str, TerminologyConcept] = field(default_factory=dict)
    tree_to_ids: Dict[str, List[str]] = field(default_factory=dict)
    treetop_names: Dict[str, str] = field(default_factory=dict)
    resource_aliases: List[str] = field(default_factory=list)
    id_prefix: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.resource_aliases:
            self.resource_aliases = [self.name]

    @property
    def aliases(self) -> List[str]:
        return list(getattr(self, "resource_aliases", None) or [self.name])

    @property
    def prefix(self) -> Optional[str]:
        return getattr(self, "id_prefix", None)

    def _cache_signature(self) -> _TerminologyCacheSignature:
        return (len(self.concepts), id(self.concepts))

    @staticmethod
    def normalize_identifier(ui: str | None) -> str | None:
        if ui is None:
            return None
        value = str(ui).strip()
        if not value:
            return None
        return value.replace("_", ":")

    @staticmethod
    def normalize_resource(resource: str | None) -> str | None:
        if resource is None:
            return None
        value = str(resource).strip()
        if not value:
            return None
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def accepts_resource(self, resource: str | None) -> bool:
        normalized = self.normalize_resource(resource)
        if normalized is None:
            return False
        return normalized in {self.normalize_resource(alias) for alias in self.aliases}

    def _candidate_ids(self, ui: str | None) -> List[str]:
        normalized = self.normalize_identifier(ui)
        if normalized is None:
            return []
        candidates = [normalized]
        if self.prefix:
            prefix = self.prefix.rstrip(":")
            if not normalized.casefold().startswith(prefix.casefold() + ":"):
                candidates.append(f"{prefix}:{normalized}")
        return candidates

    def get_concept(self, ui: str) -> Optional[TerminologyConcept]:
        id_index = self._id_index()
        for candidate in self._candidate_ids(ui):
            primary_id = id_index.get(candidate)
            if primary_id is not None:
                return self.concepts.get(primary_id)
        return None

    def _id_index(self) -> Dict[str, str]:
        cache_key = id(self)
        signature = self._cache_signature()
        cached = _ID_INDEX_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            logger.debug("Reusing terminology ID index for %s (%s concepts)", self.name, len(self.concepts))
            return cached[1]

        start = time.perf_counter()
        logger.debug("Building terminology ID index for %s (%s concepts)", self.name, len(self.concepts))
        index: Dict[str, str] = {}
        for concept in self.concepts.values():
            ids = [concept.ui, *getattr(concept, "alt_ids", [])]
            for candidate in ids:
                for normalized_candidate in self._candidate_ids(candidate):
                    index[normalized_candidate] = concept.ui
        _ID_INDEX_CACHE[cache_key] = (signature, index)
        logger.debug(
            "Built terminology ID index for %s with %s identifiers in %.2fs",
            self.name,
            len(index),
            time.perf_counter() - start,
        )
        return index

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.split()).casefold()

    def _name_index(self) -> Dict[str, List[str]]:
        cache_key = id(self)
        signature = self._cache_signature()
        cached = _NAME_INDEX_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            logger.debug("Reusing terminology name index for %s (%s concepts)", self.name, len(self.concepts))
            return cached[1]

        start = time.perf_counter()
        logger.debug("Building terminology name index for %s (%s concepts)", self.name, len(self.concepts))
        index: Dict[str, List[str]] = {}
        for concept in self.concepts.values():
            index.setdefault(self._normalize_name(concept.name), []).append(concept.ui)
        _NAME_INDEX_CACHE[cache_key] = (signature, index)
        logger.debug(
            "Built terminology name index for %s with %s names in %.2fs",
            self.name,
            len(index),
            time.perf_counter() - start,
        )
        return index

    def get_concept_ids_by_name(self, name: str) -> List[str]:
        """Return concept identifiers whose preferred name exactly matches name."""
        return list(self._name_index().get(self._normalize_name(name), []))

    def resolve_to_tree_concepts(self, ui: str) -> List[TerminologyConcept]:
        """
        Resolve an input ID to one or more concepts that carry tree numbers.
        (Similar to resolve_descriptor_records in MeSH)
        """
        concept = self.get_concept(ui)
        if concept is None:
            return []
        tree_numbers = getattr(concept, "tree_numbers", [])
        parent_ids = getattr(concept, "parent_ids", [])
        mapped_ui_ids = getattr(concept, "mapped_ui_ids", [])
        if tree_numbers or parent_ids or not mapped_ui_ids:
            return [concept]

        resolved = []
        for mapped_id in mapped_ui_ids:
            mapped = self.get_concept(mapped_id)
            if mapped and getattr(mapped, "tree_numbers", []):
                resolved.append(mapped)
        return resolved

    def depth_for_concept(self, concept: TerminologyConcept) -> int:
        cache_key = id(self)
        signature = self._cache_signature()
        cached = _DEPTH_CACHE.get(cache_key)
        if cached is None or cached[0] != signature:
            cached = (signature, {})
            _DEPTH_CACHE[cache_key] = cached
        depth_cache = cached[1]
        if concept.ui not in depth_cache:
            depth_cache[concept.ui] = self._depth_for_concept(concept, set())
        return depth_cache[concept.ui]

    def _depth_for_concept(self, concept: TerminologyConcept, seen: set[str]) -> int:
        if concept.ui in seen:
            return 1
        seen.add(concept.ui)
        tree_numbers = getattr(concept, "tree_numbers", [])
        parent_ids = getattr(concept, "parent_ids", [])
        if tree_numbers:
            return min(len(tree.split(".")) for tree in tree_numbers)
        if not parent_ids:
            return 1
        parent_depths = [
            self._depth_for_concept(parent, seen.copy())
            for parent_id in parent_ids
            for parent in [self.get_concept(parent_id)]
            if parent is not None and parent.ui != concept.ui
        ]
        return (min(parent_depths) + 1) if parent_depths else 1

    def top_ancestor_ids(self, ui: str) -> List[str]:
        cache_key = id(self)
        signature = self._cache_signature()
        cached = _TOP_ANCESTOR_CACHE.get(cache_key)
        if cached is None or cached[0] != signature:
            cached = (signature, {})
            _TOP_ANCESTOR_CACHE[cache_key] = cached
        ancestor_cache = cached[1]
        normalized = self.normalize_identifier(ui) or ui
        if normalized not in ancestor_cache:
            ancestor_cache[normalized] = self._top_ancestor_ids(normalized, set())
        return ancestor_cache[normalized]

    def _top_ancestor_ids(self, ui: str, seen: set[str]) -> List[str]:
        if ui in seen:
            return []
        seen.add(ui)
        concept = self.get_concept(ui)
        if concept is None:
            return []
        tree_numbers = getattr(concept, "tree_numbers", [])
        parent_ids = getattr(concept, "parent_ids", [])
        if tree_numbers:
            roots = []
            for tree in tree_numbers:
                root = tree.split(".")[0]
                for candidate in self.tree_to_ids.get(root, []):
                    if candidate not in roots:
                        roots.append(candidate)
            return roots or [concept.ui]
        if not parent_ids:
            return [concept.ui]
        roots = []
        for parent_id in parent_ids:
            for root in self._top_ancestor_ids(parent_id, seen.copy()):
                if root not in roots:
                    roots.append(root)
        return roots or [concept.ui]


def _add_weighted_counts(
    target: dict[str, float],
    source: Mapping[str, float],
    weight: float,
) -> None:
    for name, count in source.items():
        target[name] = target.get(name, 0.0) + count * weight


def _get_not_found_text(not_found: set[str]) -> str:
    not_found_list = list(not_found)[:NOT_FOUND_LIMIT]
    not_found_text = ", ".join(f"\"{not_found_item}\"" for not_found_item in not_found_list)
    if len(not_found_list) < len(not_found):
        more_count = len(not_found) - len(not_found_list)
        not_found_text += f" ...(+{more_count} more)"
    return not_found_text


def _topic_parent_ids(concept: TerminologyConcept) -> list[str]:
    if concept.parent_ids:
        return concept.parent_ids
    return concept.mapped_ui_ids


def _safe_len(values: Iterable[object]) -> int | None:
    try:
        return len(values)  # type: ignore[arg-type]
    except TypeError:
        return None


def _should_log_progress(total: int | None) -> bool:
    return total is None or total >= PROGRESS_LOG_MIN_TOTAL


def _progress_interval(total: int | None) -> int:
    if total is None:
        return PROGRESS_LOG_INTERVAL
    return max(PROGRESS_LOG_MIN_TOTAL, min(PROGRESS_LOG_INTERVAL, total // 10 or total))


def _log_count_start(operation: str, terminology: TerminologyResource, total: int | None) -> None:
    if _should_log_progress(total):
        logger.info(
            "Starting %s for terminology %s (%s input IDs, %s concepts)",
            operation,
            terminology.name,
            total if total is not None else "unknown",
            len(terminology.concepts),
        )
    else:
        logger.debug(
            "Starting %s for terminology %s (%s input IDs, %s concepts)",
            operation,
            terminology.name,
            total,
            len(terminology.concepts),
        )


def _log_count_progress(
    operation: str,
    terminology: TerminologyResource,
    processed: int,
    total: int | None,
    start: float,
) -> None:
    if _should_log_progress(total):
        logger.info(
            "%s for terminology %s processed %s/%s input IDs in %.1fs",
            operation,
            terminology.name,
            processed,
            total if total is not None else "unknown",
            time.perf_counter() - start,
        )


def _log_count_finish(
    operation: str,
    terminology: TerminologyResource,
    processed: int,
    total: int | None,
    result_count: int,
    not_found_count: int,
    start: float,
) -> None:
    message_args = (
        operation,
        terminology.name,
        processed,
        total if total is not None else "unknown",
        result_count,
        not_found_count,
        time.perf_counter() - start,
    )
    if _should_log_progress(total):
        logger.info(
            "Finished %s for terminology %s after %s/%s input IDs: %s result keys, %s missing IDs in %.1fs",
            *message_args,
        )
    else:
        logger.debug(
            "Finished %s for terminology %s after %s/%s input IDs: %s result keys, %s missing IDs in %.1fs",
            *message_args,
        )


class TerminologyTopicAnchorCounter:
    """Count terminology concepts under broad anchor topics."""

    def __init__(
        self,
        terminology: TerminologyResource,
        anchor_ids: Mapping[str, str] | Sequence[str] | None = None,
        term_overrides: Mapping[str, str] | None = None,
        fallback_name_topics: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.terminology = terminology
        self.anchor_labels = self._build_anchor_labels(anchor_ids)
        self.configured_anchor_topics = self._build_configured_anchor_topics(anchor_ids, term_overrides or {})
        self.term_concept_overrides = self._build_term_concept_overrides(term_overrides or {})
        self.term_tree_overrides = self._build_term_tree_overrides(self.term_concept_overrides)
        self.term_tree_override_index = self._build_term_tree_override_index(self.term_tree_overrides)
        self.fallback_name_topics = {
            name: list(topics)
            for name, topics in (fallback_name_topics or {}).items()
        }
        self.topic_id_cache: dict[str, dict[str, float]] = {}
        self.topic_name_cache: dict[str, dict[str, float]] = {}
        self.global_anchor_counts_cache: tuple[int, Dict[str, float]] | None = None

    @property
    def has_configured_anchors(self) -> bool:
        return bool(self.anchor_labels or self.term_concept_overrides)

    def _build_anchor_labels(
        self,
        anchor_ids: Mapping[str, str] | Sequence[str] | None,
    ) -> dict[str, str]:
        if anchor_ids is None:
            return {}
        if isinstance(anchor_ids, Mapping):
            items = anchor_ids.items()
        elif isinstance(anchor_ids, str):
            items = ((anchor_ids, None),)
        else:
            items = ((anchor_id, None) for anchor_id in anchor_ids)

        labels: dict[str, str] = {}
        for anchor_id, label in items:
            concept = self.terminology.get_concept(anchor_id)
            normalized_id = concept.ui if concept is not None else self.terminology.normalize_identifier(anchor_id)
            if normalized_id is None:
                continue
            labels[normalized_id] = label or (concept.name if concept is not None else normalized_id)
        return labels

    def _build_configured_anchor_topics(
        self,
        anchor_ids: Mapping[str, str] | Sequence[str] | None,
        term_overrides: Mapping[str, str],
    ) -> list[str]:
        topics: list[str] = []
        for topic in term_overrides.values():
            if topic not in topics:
                topics.append(topic)
        if isinstance(anchor_ids, Mapping):
            for anchor_id, label in anchor_ids.items():
                concept = self.terminology.get_concept(anchor_id)
                topic = label or (concept.name if concept is not None else self.terminology.normalize_identifier(anchor_id))
                if topic and topic not in topics:
                    topics.append(topic)
        return topics

    def _build_term_concept_overrides(self, term_overrides: Mapping[str, str]) -> dict[str, str]:
        concept_overrides: dict[str, str] = {}
        for term, topic in term_overrides.items():
            for concept_id in self.terminology.get_concept_ids_by_name(term):
                concept_overrides[concept_id] = topic
        return concept_overrides

    def _build_term_tree_overrides(self, concept_overrides: Mapping[str, str]) -> list[tuple[str, str]]:
        tree_overrides: list[tuple[str, str]] = []
        for concept_id, topic in concept_overrides.items():
            concept = self.terminology.get_concept(concept_id)
            if concept is None:
                continue
            for tree_number in concept.tree_numbers:
                tree_overrides.append((tree_number, topic))
        return sorted(tree_overrides, key=lambda item: len(item[0]), reverse=True)

    def _build_term_tree_override_index(self, tree_overrides: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for tree_number, topic in tree_overrides:
            index.setdefault(tree_number, []).append(topic)
        return index

    def _matching_tree_override_topics(self, tree_number: str) -> list[str]:
        topics: list[str] = []
        current = tree_number
        while current:
            topics.extend(self.term_tree_override_index.get(current, []))
            if "." not in current:
                break
            current = current.rsplit(".", 1)[0]
        return topics

    def counts_for_record_topics(
        self,
        record_name: str,
        topic_names: Sequence[str] | None,
    ) -> dict[str, float]:
        if topic_names:
            return self.topic_counts(topic_names)
        return self.fallback_topic_counts(record_name)

    def topic_counts(self, topic_names: Sequence[str]) -> dict[str, float]:
        if not topic_names:
            return {}
        topic_weight = 1.0 / len(topic_names)
        counts: dict[str, float] = {}
        for topic_name in topic_names:
            topic_counts = self.topic_anchor_counts(topic_name)
            _add_weighted_counts(counts, topic_counts, topic_weight)
        return {
            name: count
            for name, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        }

    def topic_anchor_counts(self, topic_name: str) -> dict[str, float]:
        if topic_name in self.topic_name_cache:
            return self.topic_name_cache[topic_name]

        concept_ids = self.terminology.get_concept_ids_by_name(topic_name)
        counts: dict[str, float] = {}
        if concept_ids:
            concept_weight = 1.0 / len(concept_ids)
            for concept_id in concept_ids:
                concept_counts = self._topic_anchor_counts_by_id(concept_id, set())
                _add_weighted_counts(counts, concept_counts, concept_weight)

        self.topic_name_cache[topic_name] = counts
        return counts

    def concept_anchor_counts(self, concept_id: str) -> dict[str, float]:
        return self._topic_anchor_counts_by_id(concept_id, set())

    def fallback_topic_counts(self, record_name: str) -> dict[str, float]:
        topics = self.fallback_name_topics.get(record_name, [])
        if not topics:
            return {}

        topic_weight = 1.0 / len(topics)
        return {
            topic: topic_weight
            for topic in sorted(topics)
        }

    def _topic_anchor_counts_by_id(
        self,
        ui: str,
        active: set[str],
    ) -> dict[str, float]:
        concept = self.terminology.get_concept(ui)
        cache_key = concept.ui if concept is not None else ui
        if cache_key in self.topic_id_cache:
            return self.topic_id_cache[cache_key]
        if cache_key in self.term_concept_overrides:
            self.topic_id_cache[cache_key] = {self.term_concept_overrides[cache_key]: 1.0}
            return self.topic_id_cache[cache_key]
        if cache_key in self.anchor_labels:
            self.topic_id_cache[cache_key] = {self.anchor_labels[cache_key]: 1.0}
            return self.topic_id_cache[cache_key]
        if cache_key in active:
            return {}

        if concept is None:
            self.topic_id_cache[cache_key] = {}
            return self.topic_id_cache[cache_key]

        tree_numbers = getattr(concept, "tree_numbers", [])
        if tree_numbers and self.term_tree_overrides:
            tree_counts: dict[str, float] = {}
            matched_tree_count = 0
            for tree_number in tree_numbers:
                matching_topics = self._matching_tree_override_topics(tree_number)
                if matching_topics:
                    matched_tree_count += 1
                    topic_weight = 1.0 / len(matching_topics)
                    for topic in matching_topics:
                        tree_counts[topic] = tree_counts.get(topic, 0.0) + topic_weight
            if tree_counts:
                tree_weight = 1.0 / matched_tree_count
                self.topic_id_cache[cache_key] = {
                    topic: count * tree_weight
                    for topic, count in tree_counts.items()
                }
                return self.topic_id_cache[cache_key]

        active.add(cache_key)
        parent_ids = _topic_parent_ids(concept)
        if parent_ids:
            parent_weight = 1.0 / len(parent_ids)
            counts: dict[str, float] = {}
            for parent_id in parent_ids:
                parent_counts = self._topic_anchor_counts_by_id(parent_id, active)
                _add_weighted_counts(counts, parent_counts, parent_weight)
        else:
            counts = {} if self.has_configured_anchors else {concept.name: 1.0}
        active.remove(cache_key)

        self.topic_id_cache[cache_key] = counts
        return counts

    def count_by_branch(self, ids: Iterable[str], total: int | None = None) -> Dict[str, float]:
        if total is None:
            total = _safe_len(ids)
        start = time.perf_counter()
        progress_interval = _progress_interval(total)
        _log_count_start("branch counts", self.terminology, total)
        counts = collections.defaultdict(float)
        not_found: set[str] = set()
        processed = 0
        for processed, ui in enumerate(ids, start=1):
            concepts = self.terminology.resolve_to_tree_concepts(ui)
            if not concepts:
                not_found.add(ui)
                if processed % progress_interval == 0:
                    _log_count_progress("branch counts", self.terminology, processed, total, start)
                continue
            keys = [
                key
                for concept in concepts
                for key in self.terminology.top_ancestor_ids(concept.ui)
            ]
            if not keys:
                if processed % progress_interval == 0:
                    _log_count_progress("branch counts", self.terminology, processed, total, start)
                continue
            weight = 1.0 / len(keys)
            for key in keys:
                counts[key] += weight
            if processed % progress_interval == 0:
                _log_count_progress("branch counts", self.terminology, processed, total, start)
        if len(not_found) > 0:
            logger.warning("No concept found for: {}".format(_get_not_found_text(not_found)))
        result = dict(sorted(counts.items()))
        _log_count_finish("branch counts", self.terminology, processed, total, len(result), len(not_found), start)
        return result

    def count_by_anchor(self, ids: Iterable[str], total: int | None = None) -> Dict[str, float]:
        if total is None:
            total = _safe_len(ids)
        start = time.perf_counter()
        progress_interval = _progress_interval(total)
        _log_count_start("anchor counts", self.terminology, total)
        counts = collections.defaultdict(float)
        not_found: set[str] = set()
        processed = 0
        for processed, ui in enumerate(ids, start=1):
            concepts = self.terminology.resolve_to_tree_concepts(ui)
            if not concepts:
                not_found.add(ui)
                if processed % progress_interval == 0:
                    _log_count_progress("anchor counts", self.terminology, processed, total, start)
                continue
            concept_weight = 1.0 / len(concepts)
            for concept in concepts:
                anchor_counts = self._topic_anchor_counts_by_id(concept.ui, set())
                _add_weighted_counts(counts, anchor_counts, concept_weight)
            if processed % progress_interval == 0:
                _log_count_progress("anchor counts", self.terminology, processed, total, start)
        if len(not_found) > 0:
            logger.warning("No concept found for: {}".format(_get_not_found_text(not_found)))
        result = dict(sorted(counts.items()))
        _log_count_finish("anchor counts", self.terminology, processed, total, len(result), len(not_found), start)
        return result

    def _anchor_counts_by_tree_overrides(self, concepts: Iterable[TerminologyConcept]) -> Dict[str, float]:
        total = _safe_len(concepts)
        start = time.perf_counter()
        progress_interval = _progress_interval(total)
        _log_count_start("tree override anchor counts", self.terminology, total)
        counts = collections.defaultdict(float)
        processed = 0
        for processed, concept in enumerate(concepts, start=1):
            tree_numbers = getattr(concept, "tree_numbers", [])
            if not tree_numbers:
                if processed % progress_interval == 0:
                    _log_count_progress("tree override anchor counts", self.terminology, processed, total, start)
                continue
            tree_counts: dict[str, float] = {}
            matched_tree_count = 0
            for tree_number in tree_numbers:
                matching_topics = self._matching_tree_override_topics(tree_number)
                if matching_topics:
                    matched_tree_count += 1
                    topic_weight = 1.0 / len(matching_topics)
                    for topic in matching_topics:
                        tree_counts[topic] = tree_counts.get(topic, 0.0) + topic_weight
            if not tree_counts:
                if processed % progress_interval == 0:
                    _log_count_progress("tree override anchor counts", self.terminology, processed, total, start)
                continue
            tree_weight = 1.0 / matched_tree_count
            for topic, count in tree_counts.items():
                counts[topic] += count * tree_weight
            if processed % progress_interval == 0:
                _log_count_progress("tree override anchor counts", self.terminology, processed, total, start)
        result = dict(sorted(counts.items()))
        _log_count_finish(
            "tree override anchor counts",
            self.terminology,
            processed,
            total,
            len(result),
            0,
            start,
        )
        return result

    def count_by_depth(self, ids: Iterable[str], total: int | None = None) -> Dict[int, float]:
        if total is None:
            total = _safe_len(ids)
        start = time.perf_counter()
        progress_interval = _progress_interval(total)
        _log_count_start("depth counts", self.terminology, total)
        counts = collections.defaultdict(float)
        not_found: set[str] = set()
        processed = 0
        for processed, ui in enumerate(ids, start=1):
            concepts = self.terminology.resolve_to_tree_concepts(ui)
            if not concepts:
                not_found.add(ui)
                if processed % progress_interval == 0:
                    _log_count_progress("depth counts", self.terminology, processed, total, start)
                continue
            depths = [self.terminology.depth_for_concept(concept) for concept in concepts]
            weight = 1.0 / len(depths)
            for depth in depths:
                counts[depth] += weight
            if processed % progress_interval == 0:
                _log_count_progress("depth counts", self.terminology, processed, total, start)
        if len(not_found) > 0:
            logger.warning("No concept found for: {}".format(_get_not_found_text(not_found)))
        result = dict(sorted(counts.items()))
        _log_count_finish("depth counts", self.terminology, processed, total, len(result), len(not_found), start)
        return result

    def get_global_counts_by_branch(self) -> Dict[str, float]:
        cache_key = id(self.terminology)
        concept_count = len(self.terminology.concepts)
        cached = _GLOBAL_BRANCH_COUNT_CACHE.get(cache_key)
        if cached is not None and cached[0] == concept_count:
            logger.info(
                "Reusing cached global branch counts for terminology %s (%s concepts, %s result keys)",
                self.terminology.name,
                concept_count,
                len(cached[1]),
            )
            return cached[1]
        logger.info(
            "Computing global branch counts for terminology %s by iterating %s concepts",
            self.terminology.name,
            concept_count,
        )
        target_ids = (c.ui for c in self.terminology.concepts.values())
        counts = self.count_by_branch(target_ids, total=concept_count)
        _GLOBAL_BRANCH_COUNT_CACHE[cache_key] = (concept_count, counts)
        return counts

    def get_global_counts_by_anchor(self) -> Dict[str, float]:
        concept_count = len(self.terminology.concepts)
        if self.global_anchor_counts_cache is not None and self.global_anchor_counts_cache[0] == concept_count:
            logger.info(
                "Reusing cached global anchor counts for terminology %s (%s concepts, %s result keys)",
                self.terminology.name,
                concept_count,
                len(self.global_anchor_counts_cache[1]),
            )
            return self.global_anchor_counts_cache[1]
        logger.info(
            "Computing global anchor counts for terminology %s by iterating %s concepts",
            self.terminology.name,
            concept_count,
        )
        if self.term_tree_overrides:
            tree_concepts = (
                tree_concept
                for concept in self.terminology.concepts.values()
                for tree_concept in self.terminology.resolve_to_tree_concepts(concept.ui)
            )
            counts = self._anchor_counts_by_tree_overrides(tree_concepts)
        else:
            target_ids = (c.ui for c in self.terminology.concepts.values())
            counts = self.count_by_anchor(target_ids, total=concept_count)
        self.global_anchor_counts_cache = (concept_count, counts)
        return counts

    def get_global_counts_by_depth(self) -> Dict[int, float]:
        cache_key = id(self.terminology)
        concept_count = len(self.terminology.concepts)
        cached = _GLOBAL_DEPTH_COUNT_CACHE.get(cache_key)
        if cached is not None and cached[0] == concept_count:
            logger.info(
                "Reusing cached global depth counts for terminology %s (%s concepts, %s result keys)",
                self.terminology.name,
                concept_count,
                len(cached[1]),
            )
            return cached[1]
        logger.info(
            "Computing global depth counts for terminology %s by iterating %s concepts",
            self.terminology.name,
            concept_count,
        )
        target_ids = (c.ui for c in self.terminology.concepts.values())
        counts = self.count_by_depth(target_ids, total=concept_count)
        _GLOBAL_DEPTH_COUNT_CACHE[cache_key] = (concept_count, counts)
        return counts

    def branch_label(self, branch_code: str) -> str:
        concept = self.terminology.get_concept(branch_code)
        if concept is not None:
            return concept.name
        for concept in self.terminology.concepts.values():
            if branch_code in getattr(concept, "tree_numbers", []):
                return concept.name
        return self.terminology.treetop_names.get(branch_code, branch_code)

    def topic_treetop_names(self, topic_name: str) -> list[str]:
        return topic_treetop_names(self.terminology, topic_name)


def load_topic_term_overrides(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping of broad topics to terminology terms")

    overrides: dict[str, str] = {}
    for topic, terms in payload.items():
        if not isinstance(topic, str):
            raise ValueError(f"{path} contains a non-string topic: {topic!r}")
        if not isinstance(terms, list):
            raise ValueError(f"{path} topic {topic!r} must contain a list of terminology terms")
        for term in terms:
            if not isinstance(term, str):
                raise ValueError(f"{path} topic {topic!r} contains a non-string terminology term: {term!r}")
            existing_topic = overrides.get(term)
            if existing_topic is not None and existing_topic != topic:
                raise ValueError(f"{path} maps terminology term {term!r} to both {existing_topic!r} and {topic!r}")
            overrides[term] = topic
    return overrides


def load_name_topic_fallbacks(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON mapping of names to topic lists")

    name_topics: dict[str, list[str]] = {}
    for name, topics in payload.items():
        if not isinstance(name, str):
            raise ValueError(f"{path} contains a non-string name: {name!r}")
        if not isinstance(topics, list) or not topics:
            raise ValueError(f"{path} name {name!r} must contain a non-empty list of topics")
        name_topics[name] = []
        for topic in topics:
            if not isinstance(topic, str):
                raise ValueError(f"{path} name {name!r} contains a non-string topic: {topic!r}")
            name_topics[name].append(topic)
    return name_topics


def get_journal_topic_anchor_counter(
    target,
    terminology: TerminologyResource,
    terminology_name: str,
    topic_terms_path: str,
    journal_name_topics_path: str,
) -> TerminologyTopicAnchorCounter:
    if not target.components:
        raise ValueError("journal topic distribution requires a non-empty metric target")

    cache_key = (
        "journal_topic_anchor_counter",
        terminology_name,
        topic_terms_path,
        journal_name_topics_path,
    )
    context = target.components[0][1]
    return context.get_or_compute(
        repr(cache_key),
        lambda: TerminologyTopicAnchorCounter(
            terminology,
            term_overrides=load_topic_term_overrides(Path(topic_terms_path)),
            fallback_name_topics=load_name_topic_fallbacks(Path(journal_name_topics_path)),
        ),
    )


def topic_treetop_names(terminology: TerminologyResource, topic_name: str) -> list[str]:
    treetop_names: set[str] = set()
    for ui in terminology.get_concept_ids_by_name(topic_name):
        for concept in terminology.resolve_to_tree_concepts(ui):
            for tree_number in concept.tree_numbers:
                treetop = tree_number.split(".")[0]
                treetop_name = terminology.treetop_names.get(treetop)
                if treetop_name:
                    treetop_names.add(treetop_name)
    return sorted(treetop_names)
