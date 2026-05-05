from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
_NAME_INDEX_CACHE: Dict[int, tuple[int, Dict[str, List[str]]]] = {}


@dataclass(slots=True)
class TerminologyConcept:
    ui: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    tree_numbers: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    scope_note: Optional[str] = None
    mapped_ui_ids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TerminologyResource:
    name: str
    concepts: Dict[str, TerminologyConcept] = field(default_factory=dict)
    tree_to_ids: Dict[str, List[str]] = field(default_factory=dict)
    treetop_names: Dict[str, str] = field(default_factory=dict)

    def get_concept(self, ui: str) -> Optional[TerminologyConcept]:

        return self.concepts.get(ui)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.split()).casefold()

    def _name_index(self) -> Dict[str, List[str]]:
        cache_key = id(self)
        concept_count = len(self.concepts)
        cached = _NAME_INDEX_CACHE.get(cache_key)
        if cached is not None and cached[0] == concept_count:
            return cached[1]

        index: Dict[str, List[str]] = {}
        for concept in self.concepts.values():
            index.setdefault(self._normalize_name(concept.name), []).append(concept.ui)
        _NAME_INDEX_CACHE[cache_key] = (concept_count, index)
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
            logger.warning("No concept found for %s", ui)
            return []
        if concept.tree_numbers:
            return [concept]

        resolved = []
        for mapped_id in concept.mapped_ui_ids:
            mapped = self.get_concept(mapped_id)
            if mapped and mapped.tree_numbers:
                resolved.append(mapped)
        if len(resolved) == 0:
            logger.warning("Resolved %s to %s tree concepts", ui, len(resolved))
        else:
            logger.debug("Resolved %s to %s tree concepts", ui, len(resolved))
        return resolved
