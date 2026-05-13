from __future__ import annotations

import gzip
import logging
import shutil
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from corpus_benchmark.models.config import BenchmarkConfig, WorkspaceConfig, APP_NAME_VER
from corpus_benchmark.registry import CONVERTERS
from utils.download import download_file

logger = logging.getLogger(__name__)

# Canonical format names used internally.  "none" means download only.
_ARCHIVE_FORMAT_ALIASES: dict[str, str] = {
    "": "none",
    "none": "none",
    "raw": "none",
    "download": "none",
    "download_only": "none",
    "no_extract": "none",
    "auto": "auto",
    "zip": "zip",
    "tar": "tar",
    "tar.gz": "tar.gz",
    "tgz": "tar.gz",
    "gz": "gz",
    "gzip": "gz",
}


class AcquisitionManager:
    def __init__(self, workspace_config: WorkspaceConfig):
        self.download_dir = Path(workspace_config.corpora_download_dir)

    def ensure_corpus_ready(self, corpus_name: str, config: BenchmarkConfig) -> bool:
        """Ensure that the required corpus files exist locally.

        Returns:
            True if the corpus was already ready, False if acquisition was performed.

        Acquisition supports two source-url styles:
        ... (omitted docstring rest for brevity in this thought, but I will provide full) ...
        """
        corpus_dir = self.download_dir / corpus_name
        sentinel_file = corpus_dir / ".acquisition_done"

        paths = _expected_loader_paths(config)
        all_exist = bool(paths) and all(Path(path).exists() for path in paths.values())

        if all_exist:
            # If no acquisition spec is provided, we assume the user has
            # manually placed the files and we trust all_exist.
            if not config.acquisition or sentinel_file.exists():
                return True

        if not config.acquisition:
            raise FileNotFoundError(
                f"Files for corpus '{corpus_name}' are missing, and no acquisition spec was provided."
            )

        logger.info("Acquiring corpus '%s'...", corpus_name)
        if sentinel_file.exists():
            sentinel_file.unlink()

        corpus_dir.mkdir(parents=True, exist_ok=True)

        # Download each source and keep the extraction format associated with
        # that specific downloaded file.
        default_format = getattr(config.acquisition, "format", None)
        downloaded_files: list[tuple[Path, str]] = []

        for source in config.acquisition.source_urls:
            url, fmt = _normalize_source_spec(source, default_format)
            filename = _download_filename(url)
            dest_path = corpus_dir / filename

            if not dest_path.exists():
                logger.info("  Downloading %s -> %s", url, dest_path)
                # NOTE: some repos (e.g. Zenodo) insist on a meaningful user_agent
                download_file(url, dest_path, user_agent=APP_NAME_VER)
            else:
                logger.info("  Reusing downloaded file %s", dest_path)

            downloaded_files.append((dest_path, fmt))

        # Extract archives, using a per-source format when provided.  Plain files
        # are skipped when format is "none" or auto-detection finds no archive.
        for file_path, fmt in downloaded_files:
            _extract_downloaded_file(file_path, corpus_dir, fmt)

        # Run custom converter/mapper if specified.
        if config.acquisition.converter:
            converter_func = CONVERTERS.get(config.acquisition.converter)
            if not converter_func:
                raise ValueError(f"Converter '{config.acquisition.converter}' not found in registry.")
            logger.info("  Running converter '%s'", config.acquisition.converter)
            converter_func(corpus_dir, config)

        # Final validation: check that expected loader paths exist after
        # acquisition is complete.
        for label, path in paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"After acquisition, expected file '{path}' for '{label}' is still missing!"
                )

        sentinel_file.touch()
        logger.info("  Corpus '%s' acquired and content verified", corpus_name)
        return False


def _normalize_source_spec(source: Any, default_format: str | None) -> tuple[str, str]:
    """Return ``(url, format)`` for a source specification.

    ``source`` may be either a plain URL string or a mapping with a required
    ``url`` key and an optional ``format`` key.  The source-specific format, if
    present, overrides ``default_format``.
    """
    if isinstance(source, str):
        return source, _normalize_archive_format(default_format)

    if isinstance(source, Mapping):
        if "url" not in source:
            raise ValueError(f"Acquisition source mapping is missing required 'url': {source!r}")
        url = source["url"]
        if not isinstance(url, str):
            raise TypeError(f"Acquisition source 'url' must be a string: {source!r}")
        fmt = source.get("format", default_format)
        return url, _normalize_archive_format(fmt)

    raise TypeError(
        "Each acquisition source must be either a URL string or a mapping "
        f"with a 'url' field; got {type(source).__name__}: {source!r}"
    )


def _normalize_archive_format(fmt: Any) -> str:
    """Normalize user-facing archive format names to canonical values."""
    if fmt is None:
        return "none"

    if not isinstance(fmt, str):
        raise TypeError(f"Archive format must be a string or None; got {type(fmt).__name__}: {fmt!r}")

    key = fmt.strip().lower().replace("_", "-")
    key = key.replace("-", "_") if key in {"download-only", "no-extract"} else key
    key = key.replace("_", "-") if key not in _ARCHIVE_FORMAT_ALIASES else key

    # Handle common spellings after normalizing whitespace/case.
    if key in {"download-only", "download_only"}:
        key = "download_only"
    elif key in {"no-extract", "no_extract"}:
        key = "no_extract"

    try:
        return _ARCHIVE_FORMAT_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_ARCHIVE_FORMAT_ALIASES))
        raise ValueError(f"Unsupported acquisition format {fmt!r}. Supported formats: {supported}") from exc


def _download_filename(url: str) -> str:
    """Derive a local filename from a URL.

    Query strings and fragments are ignored.  Percent-encoded path components
    are decoded so that URLs ending in ``file%20name.zip`` produce a readable
    local filename.
    """
    parsed = urlparse(url)
    filename = unquote(Path(parsed.path).name)
    if not filename:
        raise ValueError(f"Could not determine filename from acquisition URL: {url!r}")
    return filename


def _extract_downloaded_file(file_path: Path, output_dir: Path, fmt: str) -> None:

    """Extract or skip one downloaded file according to ``fmt``."""
    if fmt == "auto":
        inferred_fmt = _infer_archive_format(file_path)
        if inferred_fmt == "none":
            logger.info("  Not extracting %s (auto-detected as raw file)", file_path)
            return
        fmt = inferred_fmt

    if fmt == "none":
        logger.info("  Not extracting %s", file_path)
        return

    logger.info("  Extracting %s as %s", file_path, fmt)

    if fmt == "zip":
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
    elif fmt in {"tar", "tar.gz"}:
        # mode "r:*" handles uncompressed tar, gzip-compressed tar, and other
        # tar compression variants supported by Python's tarfile module.
        with tarfile.open(file_path, "r:*") as tar_ref:
            tar_ref.extractall(output_dir)
    elif fmt == "gz":
        output_path = _gunzip_output_path(file_path, output_dir)
        with gzip.open(file_path, "rb") as src, output_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        # This should be unreachable if formats are normalized at input time.
        raise ValueError(f"Unsupported acquisition format: {fmt!r}")


def _infer_archive_format(file_path: Path) -> str:
    """Infer archive format from filename suffixes.

    Returns ``none`` when the filename does not look like a supported archive.
    """
    name = file_path.name.lower()

    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "tar.gz"
    if name.endswith(".tar"):
        return "tar"
    if name.endswith(".gz"):
        return "gz"

    return "none"


def _gunzip_output_path(file_path: Path, output_dir: Path) -> Path:
    """Return the output path for a single-file gzip archive."""
    if file_path.name.lower().endswith(".gz"):
        output_name = file_path.name[:-3]
    else:
        output_name = file_path.stem

    if not output_name:
        raise ValueError(f"Could not determine gunzip output filename for {file_path}")

    return output_dir / output_name


def _expected_loader_paths(config: BenchmarkConfig) -> dict[str, str]:
    params = config.loader.params
    paths = dict(params.get("paths", {}))
    if "path" in params:
        paths["path"] = params["path"]

    split_config = params.get("split") or {}
    for split_name, split_path in (split_config.get("files") or {}).items():
        paths[f"split:{split_name}"] = split_path
    if "mapping_path" in split_config:
        paths["split:mapping"] = split_config["mapping_path"]
    return paths
