from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Iterable

from corpus_benchmark.models.corpus import (
    Annotation,
    CompositeLink,
    Link,
)
from corpus_benchmark.models.types import (
    LinkRelation,
    MatchType,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnnotationFilter:
    """
    Filters annotations by:
      - annotation label
      - link relations appearing anywhere in the recursive link structure
      - match types appearing anywhere in the recursive link structure

    Semantics:
      - labels=None means allow any label
      - link_relations=None means allow any link relation pattern
      - match_types=None means allow any match type pattern

    Special handling:
      - if link_relations is an empty set, only annotations with NO CompositeLink
        anywhere in their link structure are accepted
      - if match_types is an empty set, only annotations with NO match_type
        anywhere in their link structure are accepted
    """

    labels: set[str] | None = None
    link_relations: set[LinkRelation] | None = None
    match_types: set[MatchType] | None = None

    @classmethod
    def from_config_dict(cls, raw: dict[str, Any]) -> AnnotationFilter:
        """
        Construct an AnnotationFilter from a YAML/config dictionary.

        Expected keys:
          - labels: list[str] | omitted
          - link_relations: list[str] | omitted
          - match_types: list[str] | omitted

        Omitted means "allow any".
        Empty list means "allow none".
        """
        labels_raw = raw.get("labels", None)
        link_relations_raw = raw.get("link_relations", None)
        match_types_raw = raw.get("match_types", None)

        labels: set[str] | None
        if labels_raw is None:
            labels = None
        else:
            labels = {str(x) for x in labels_raw}

        link_relations: set[LinkRelation] | None
        if link_relations_raw is None:
            link_relations = None
        else:
            link_relations = {cls._parse_link_relation(x) for x in link_relations_raw}

        match_types: set[MatchType] | None
        if match_types_raw is None:
            match_types = None
        else:
            match_types = {cls._parse_match_type(x) for x in match_types_raw}

        return cls(
            labels=labels,
            link_relations=link_relations,
            match_types=match_types,
        )

    @staticmethod
    def _parse_link_relation(value: str | LinkRelation) -> LinkRelation:
        if isinstance(value, LinkRelation):
            return value
        try:
            return LinkRelation(value)
        except ValueError as e:
            valid = ", ".join(x.value for x in LinkRelation)
            raise ValueError(f"Invalid LinkRelation '{value}'. Valid values: {valid}") from e

    @staticmethod
    def _parse_match_type(value: str | MatchType) -> MatchType:
        if isinstance(value, MatchType):
            return value
        try:
            return MatchType(value)
        except ValueError as e:
            valid = ", ".join(x.value for x in MatchType)
            raise ValueError(f"Invalid MatchType '{value}'. Valid values: {valid}") from e

    def accepts(self, annotation: Annotation) -> bool:
        """
        Return True if the annotation passes the filter.
        """
        if self.labels is not None:
            if annotation.label is None or annotation.label not in self.labels:
                return False

        relations_found = self._collect_link_relations(annotation.link)
        match_types_found = self._collect_match_types(annotation.link)

        if self.link_relations is not None:
            if not self._accepts_found_values(
                found=relations_found,
                allowed=self.link_relations,
            ):
                return False

        if self.match_types is not None:
            if not self._accepts_found_values(
                found=match_types_found,
                allowed=self.match_types,
            ):
                return False

        return True

    def rejects(self, annotation: Annotation) -> bool:
        return not self.accepts(annotation)

    def filter_annotations(
        self,
        annotations: Iterable[Annotation],
    ) -> list[Annotation]:
        """
        Return only annotations accepted by the filter.
        """
        return [ann for ann in annotations if self.accepts(ann)]

    @staticmethod
    def _accepts_found_values[T](found: set[T], allowed: set[T]) -> bool:
        """
        Filtering rule:
          - allowed == empty set => accept only when nothing is found
          - otherwise => every found value must be in allowed
        """
        if len(allowed) == 0:
            return len(found) == 0
        return found.issubset(allowed)

    def _collect_link_relations(self, link: Link | None) -> set[LinkRelation]:
        found: set[LinkRelation] = set()
        self._walk_link_relations(link, found)
        return found

    def _walk_link_relations(
        self,
        link: Link | None,
        found: set[LinkRelation],
    ) -> None:
        if link is None:
            return

        if isinstance(link, CompositeLink):
            found.add(link.relation)
            for component in link.components:
                self._walk_link_relations(component, found)

    def _collect_match_types(self, link: Link | None) -> set[MatchType]:
        found: set[MatchType] = set()
        self._walk_match_types(link, found)
        return found

    def _walk_match_types(
        self,
        link: Link | None,
        found: set[MatchType],
    ) -> None:
        if link is None:
            return

        if link.match_type is not None:
            found.add(link.match_type)

        if isinstance(link, CompositeLink):
            for component in link.components:
                self._walk_match_types(component, found)

    def __str__(self):
        return f"AnnotationFilter(labels = {self.labels}, link_relations = {self.link_relations}, match_types = {self.match_types})"
