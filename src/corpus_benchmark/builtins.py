from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BUILTINS_REGISTERED = False


def register_builtins() -> None:
    """Import built-in plugins so their registry decorators run exactly once."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    logger.debug("Registering built-in corpus_benchmark plugins")

    import corpus_benchmark.loaders  # noqa: F401
    import corpus_benchmark.metadata.crossref_document_fetcher  # noqa: F401
    import corpus_benchmark.metadata.eutils_document_fetchers  # noqa: F401
    import corpus_benchmark.metrics  # noqa: F401

    _BUILTINS_REGISTERED = True
