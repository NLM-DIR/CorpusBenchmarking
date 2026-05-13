from __future__ import annotations

from collections.abc import Callable
import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

LOADERS: dict[str, Callable[..., Any]] = {}
TERMINOLOGY_LOADERS: dict[str, Callable[..., Any]] = {}
CONVERTERS: dict[str, Callable[..., Any]] = {}
DOCUMENT_FETCHERS: dict[str, type[Any]] = {}
SUBSET_METRICS: dict[str, Callable[..., Any]] = {}
CROSS_METRICS: dict[str, Callable[..., Any]] = {}
TERMINOLOGY_METRICS: dict[str, Callable[..., Any]] = {}


def _wrap_with_logging(kind: str, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        logger.debug("Calling %s '%s' from %s", kind, name, func.__module__)
        result = func(*args, **kwargs)
        logger.debug("Finished %s '%s' from %s", kind, name, func.__module__)
        return result

    return wrapped


# TODO Change function "register_loader" to "register_corpus_loader"


def register_loader(name: str):
    """Register a corpus loader under a stable symbolic name."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in LOADERS:
            raise ValueError(f"Loader '{name}' is already registered.")
        logger.debug("Registering loader '%s' from %s", name, func.__module__)
        wrapped = _wrap_with_logging("loader", name, func)
        LOADERS[name] = wrapped
        return wrapped

    return decorator


def register_terminology_loader(name: str):
    """Register a terminology loader under a stable symbolic name."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in TERMINOLOGY_LOADERS:
            raise ValueError(f"Terminology loader '{name}' is already registered.")
        logger.debug("Registering terminology loader '%s' from %s", name, func.__module__)
        wrapped = _wrap_with_logging("terminology loader", name, func)
        TERMINOLOGY_LOADERS[name] = wrapped
        return wrapped

    return decorator


def register_converter(name: str):
    """Register a custom acquisition converter/mapper."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in CONVERTERS:
            raise ValueError(f"Converter '{name}' is already registered.")
        logger.debug("Registering converter '%s' from %s", name, func.__module__)
        wrapped = _wrap_with_logging("converter", name, func)
        CONVERTERS[name] = wrapped
        return wrapped

    return decorator


def register_document_fetcher(name: str):
    """Register a document metadata fetcher class under a stable symbolic name."""

    def decorator(fetcher_cls: type[Any]) -> type[Any]:
        if name in DOCUMENT_FETCHERS:
            raise ValueError(f"Document fetcher '{name}' is already registered.")
        logger.debug(
            "Registering document fetcher '%s' from %s",
            name,
            fetcher_cls.__module__,
        )
        DOCUMENT_FETCHERS[name] = fetcher_cls
        return fetcher_cls

    return decorator


def register_subset_metric(
    name: str,
    *,
    requires_metadata: bool = False,
    supports_annotation_scope: bool = False,
):
    """Register a metric under a stable symbolic name."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in SUBSET_METRICS:
            raise ValueError(f"Metric '{name}' is already registered.")
        logger.debug("Registering subset metric '%s' from %s", name, func.__module__)
        wrapped = _wrap_with_logging("subset metric", name, func)
        setattr(wrapped, "requires_metadata", requires_metadata)
        setattr(wrapped, "supports_annotation_scope", supports_annotation_scope)
        SUBSET_METRICS[name] = wrapped
        return wrapped

    return decorator


def register_cross_metric(name: str, *, supports_annotation_scope: bool = False):
    """Register a metric under a stable symbolic name."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in CROSS_METRICS:
            raise ValueError(f"Metric '{name}' is already registered.")
        logger.debug("Registering cross metric '%s' from %s", name, func.__module__)
        wrapped = _wrap_with_logging("cross metric", name, func)
        setattr(wrapped, "supports_annotation_scope", supports_annotation_scope)
        CROSS_METRICS[name] = wrapped
        return wrapped

    return decorator


def register_terminology_metric(name: str, *, supports_annotation_scope: bool = False):
    """Register a terminology metric under a stable symbolic name."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in TERMINOLOGY_METRICS:
            raise ValueError(f"Terminology metric '{name}' is already registered.")
        logger.debug("Registering terminology metric '%s' from %s", name, func.__module__)
        wrapped = _wrap_with_logging("terminology metric", name, func)
        setattr(wrapped, "supports_annotation_scope", supports_annotation_scope)
        TERMINOLOGY_METRICS[name] = wrapped
        return wrapped

    return decorator
