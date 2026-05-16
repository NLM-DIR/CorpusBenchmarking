from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from corpus_benchmark.builtins import register_builtins
from corpus_benchmark.models.config import LoaderSpec, WorkspaceConfig
from corpus_benchmark.models.terminologies import (
    TerminologyResource,
    load_name_topic_fallbacks,
    load_topic_term_overrides,
)
from corpus_benchmark.registry import TERMINOLOGY_LOADERS

PRECISION = 4


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a JSON record-store 'records' list")
    return records


def load_mesh_term_overrides(path: Path) -> dict[str, str]:
    return load_topic_term_overrides(path)


def load_journal_name_topics(path: Path) -> dict[str, list[str]]:
    return load_name_topic_fallbacks(path)


def load_terminology(
    workspace_config: WorkspaceConfig,
    terminology_name: str,
    terminology_spec: LoaderSpec | None,
    *,
    default_spec: LoaderSpec | None = None,
) -> TerminologyResource:
    register_builtins()

    spec = terminology_spec or default_spec
    if spec is None:
        raise ValueError(f"No terminology loader configured for {terminology_name!r}")

    loader = TERMINOLOGY_LOADERS.get(spec.name)
    if loader is None:
        available = ", ".join(sorted(TERMINOLOGY_LOADERS)) or "<none>"
        raise ValueError(
            f"Unknown terminology loader {spec.name!r} for {terminology_name!r}. "
            f"Available terminology loaders: {available}"
        )
    return loader(workspace_config, **spec.params)


def round_floats(data: Any) -> Any:
    if isinstance(data, float):
        return round(data, PRECISION)
    if isinstance(data, dict):
        return {k: round_floats(v) for k, v in data.items()}
    if isinstance(data, list):
        return [round_floats(v) for v in data]
    return data


def round_counts(counts: dict[str, float]) -> dict[str, float]:
    return {
        name: round(count, PRECISION)
        for name, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def add_weighted_counts(
    target: dict[str, float],
    source: dict[str, float],
    weight: float,
) -> None:
    for name, count in source.items():
        target[name] = target.get(name, 0.0) + count * weight


def write_json_payload(path: Path | None, payload: Any, *, sort_keys: bool = False) -> None:
    text = json.dumps(payload, indent=2, sort_keys=sort_keys)
    if path:
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
