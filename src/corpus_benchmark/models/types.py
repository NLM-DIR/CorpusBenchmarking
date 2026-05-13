from enum import Enum
from dataclasses import dataclass, field


class MatchType(str, Enum):
    EXACT = "exact"
    RELATED = "related"
    APPROXIMATE = "approximate"
    NIL = "NIL"


class LinkRelation(str, Enum):
    DISTRIBUTIVE = "distributive"  # e.g. "endothelial and epithelial cells"
    INTERSECTIVE = "intersective"  # e.g. "inherited muscle disorder"
    RELATED_SET = "related_set"  # e.g. CellLink-style related match set
    ALTERNATIVE = "alternative"  # optional: true either/or ambiguity
