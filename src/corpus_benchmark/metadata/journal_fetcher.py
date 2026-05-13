from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class JournalMetadataFetcher(ABC):
    """Base class for resolving journals into JsonRecordStore-ready records."""

    @property
    @abstractmethod
    def supported_key(self) -> str:
        """The input key this fetcher accepts, e.g. NLMUNIQUEID or ISSN."""
        pass

    @abstractmethod
    def fetch(self, identifiers: list[str]) -> list[dict[str, Any]]:
        """Return journal records. Records must include an identifiers dict."""
        pass
