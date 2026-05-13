from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

# TODO rename subset -> target

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class SubsetMetricResult:
    result_name: str
    metric_name: str
    subset_name: str
    value: Any
    details: dict[str, Any] = field(default_factory=dict)
    scopes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def result_key(self) -> str:
        return self.subset_name

    def to_dict(self) -> dict[str, Any]:
        result = {
            "metric_name": self.metric_name,
            "value": self.value,
        }
        if len(self.details) > 0:
            result["details"] = self.details
        if self.scopes:
            result["scopes"] = self.scopes
        return result

@dataclass(slots=True)
class CrossSubsetMetricResult:
    result_name: str
    metric_name: str
    value: Any
    subset_name1: str
    subset_name2: str
    details: dict[str, Any] = field(default_factory=dict)
    scopes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def result_key(self) -> str:
        return f"({self.subset_name1}, {self.subset_name2})"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "metric_name": self.metric_name,
            "value": self.value,
        }
        if len(self.details) > 0:
            result["details"] = self.details
        if self.scopes:
            result["scopes"] = self.scopes
        return result
