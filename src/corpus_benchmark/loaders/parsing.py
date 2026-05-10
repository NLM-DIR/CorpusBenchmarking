from __future__ import annotations

import logging
from dataclasses import dataclass

from corpus_benchmark.models.types import LinkRelation, MatchType
from utils.text_utils import str_to_bool

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IdentifierFormat:
    delimiter: str
    relation: LinkRelation
    qualifier_allowed: bool


def parse_identifier_format_list(
    id_format_list: list[list[str]],
) -> list[IdentifierFormat]:
    logger.debug("Parsing %s identifier format definitions", len(id_format_list))
    if len(id_format_list) == 0:
        return []
    return [parse_identifier_format(id_format) for id_format in id_format_list]


def parse_identifier_format(id_format: list[str]) -> IdentifierFormat:
    logger.debug("Parsing identifier format %s", id_format)
    if len(id_format) != 3:
        raise ValueError()
    delimiter = id_format[0]
    relation = LinkRelation(id_format[1])
    qualifier_allowed = str_to_bool(id_format[2])
    return IdentifierFormat(delimiter, relation, qualifier_allowed)


def parse_qualifier_map(qualifier_map: dict[str, str]) -> dict[str, MatchType]:
    return {qualifier_text: MatchType(match_type_text) for qualifier_text, match_type_text in qualifier_map.items()}
