from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List

from corpus_benchmark.models.corpus import DocumentIdentifierType

logger = logging.getLogger(__name__)

# TODO Add configuration so user can specify which document fetcher to use
# TODO Make sure each fetcher documents clearly how many API calls are made per batch
# TODO Add a representation and metadata for journals
# TODO Have each document fetcher API call grab all the info it can for journals
# TODO Consider adding Fetchers for more APIs, e.g., Europe PMC
# TODO Consider adding a eUtils/esearch fetcher to support DOI lookup


# Abstract Base Class for Fetchers
class DocumentMetadataFetcher(ABC):
    @property
    @abstractmethod
    def supported_id_type(self) -> DocumentIdentifierType:
        pass

    @abstractmethod
    def fetch(self, identifiers: List[str]) -> List[Dict[str, Any]]:
        """
        Return standard document records.

        Records must include the ``identifiers`` dict in the output. They may
        also include a ``journal_metadata`` dict; GlobalWorkspace consumes that
        side payload when a journal record store is configured.
        """
        pass
