from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


INDEPENDENT_VARIABLES = [
    "journal_distribution_entropy",
    "article_topic_distribution_entropy",
    "journal_topic_distribution_entropy",
    "temporal_coverage_range",
]

BASE_VARIABLES = [
    "annotations_per_1000_tokens",
    "unique_identifiers_per_document",
    "ambiguity",
    "variation",
    "entity_type_label_count",
]

DEPENDENT_VARIABLES = [
    "terminology_coverage_entropy",
    "annotation_topic_coverage_entropy",
    "train_test_token_overlap",
    "train_test_mention_token_overlap",
    "train_test_mention_string_overlap",
    "train_test_identifier_overlap",
]

VARIABLES = INDEPENDENT_VARIABLES + BASE_VARIABLES + DEPENDENT_VARIABLES


def norm_corpus_name(name: str) -> str:
    name = name.lower()
    for suffix in ("_corpus", "_train", "_test", "_dev"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"[^a-z0-9]", "", name)


def display_corpus_name(name: str) -> str:
    return name.replace("_corpus", "").replace("_", "-")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(metrics: list[dict[str, Any]], metric_name: str) -> dict[str, Any] | None:
    return next((item for item in metrics if item.get("metric_name") == metric_name), None)


def metric_payload(metrics: list[dict[str, Any]], metric_name: str, scope: str) -> dict[str, Any] | None:
    item = metric(metrics, metric_name)
    if item is None:
        return None
    if scope == "all":
        return item
    return (item.get("scopes") or {}).get(scope)


def stat_value(metrics: list[dict[str, Any]], metric_name: str, stat_name: str, scope: str) -> float | None:
    payload = metric_payload(metrics, metric_name, scope)
    if not payload:
        return None
    value = payload.get("value")
    if not isinstance(value, dict):
        return None
    return as_number(value.get(stat_name))


def details_value(metrics: list[dict[str, Any]], metric_name: str, detail_name: str, scope: str = "all") -> float | None:
    payload = metric_payload(metrics, metric_name, scope)
    if not payload:
        return None
    details = payload.get("details") or {}
    return as_number(details.get(detail_name))


def as_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def label_count(metrics: list[dict[str, Any]], scope: str) -> float | None:
    payload = metric_payload(metrics, "label_distribution", scope)
    if not payload:
        return None
    value = payload.get("value") or {}
    if not isinstance(value, dict) or not value:
        return None
    return float(len([name for name, fraction in value.items() if name not in ("Unknown", None) and fraction]))


def temporal_range(metrics: list[dict[str, Any]]) -> float | None:
    payload = metric_payload(metrics, "publication_year_distribution", "all")
    if not payload:
        return None
    years: list[int] = []
    for key, value in (payload.get("value") or {}).items():
        if key in ("Unknown", None) or not value:
            continue
        try:
            years.append(int(key))
        except (TypeError, ValueError):
            continue
    if not years:
        return None
    return float(max(years) - min(years))


def overlap_corpus_key(key: str) -> str:
    match = re.match(r"\((\w+?)_(?:train|test|dev)", key)
    return norm_corpus_name(match.group(1) if match else key.strip("()"))


def overlap_value(metrics: list[dict[str, Any]], metric_name: str, scope: str) -> float | None:
    payload = metric_payload(metrics, metric_name, scope)
    if not payload and metric_name == "token_overlap":
        payload = metric_payload(metrics, metric_name, "all")
    if not payload:
        return None
    if scope != "all" and metric_name != "token_overlap":
        details = payload.get("details") or {}
        if details.get("union") == 0:
            return None
    return as_number(payload.get("value"))


def terminology_entropy(metrics: list[dict[str, Any]], metric_name: str, scope: str) -> float | None:
    values = []
    for item in metrics:
        if item.get("metric_name") != metric_name:
            continue
        payload = item if scope == "all" else (item.get("scopes") or {}).get(scope)
        if not payload:
            continue
        if (payload.get("details") or {}).get("n_input_ids") == 0:
            continue
        value = as_number((payload.get("details") or {}).get("distribution_entropy"))
        if value is not None:
            values.append(value)
    return mean(values) if values else None


def collect_scopes(*raw_sources: dict[str, Any]) -> list[str]:
    scopes = {"all"}
    for raw in raw_sources:
        for metrics in raw.values():
            if not isinstance(metrics, list):
                continue
            for item in metrics:
                scopes.update((item.get("scopes") or {}).keys())
    return ["all"] + sorted(scope for scope in scopes if scope != "all")


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denominator = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_denominator == 0 or y_denominator == 0:
        return None
    return numerator / (x_denominator * y_denominator)


def format_cell(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{value:.8g}"


def build_rows(
    basic: dict[str, list[dict[str, Any]]],
    metadata: dict[str, list[dict[str, Any]]],
    terminology: dict[str, list[dict[str, Any]]],
    overlap: dict[str, list[dict[str, Any]]],
    scope: str,
) -> list[dict[str, float | str | None]]:
    overlap_by_corpus = {overlap_corpus_key(key): value for key, value in overlap.items()}
    rows = []
    for raw_corpus_name, basic_metrics in sorted(basic.items(), key=lambda item: display_corpus_name(item[0])):
        corpus_key = norm_corpus_name(raw_corpus_name)
        metadata_metrics = metadata.get(raw_corpus_name, [])
        terminology_metrics = terminology.get(raw_corpus_name, [])
        overlap_metrics = overlap_by_corpus.get(corpus_key, [])
        n_labels = label_count(basic_metrics, scope)
        has_scope = scope == "all" or n_labels is not None
        row: dict[str, float | str | None] = {
            "corpus": display_corpus_name(raw_corpus_name),
            "journal_distribution_entropy": details_value(metadata_metrics, "journal_distribution", "entropy"),
            "article_topic_distribution_entropy": details_value(metadata_metrics, "article_MeSH_topic_distribution", "entropy"),
            "journal_topic_distribution_entropy": details_value(metadata_metrics, "journal_MeSH_topic_distribution", "entropy"),
            "temporal_coverage_range": temporal_range(metadata_metrics),
            "annotations_per_1000_tokens": stat_value(basic_metrics, "annotations_per_1000_tokens_stats", "mean", scope) if has_scope else None,
            "unique_identifiers_per_document": stat_value(basic_metrics, "unique_identifiers_per_document_stats", "mean", scope) if has_scope else None,
            "ambiguity": stat_value(basic_metrics, "ambiguity_degree_stats", "mean", scope) if has_scope else None,
            "variation": stat_value(basic_metrics, "variation_degree_stats", "mean", scope) if has_scope else None,
            "entity_type_label_count": n_labels,
            "terminology_coverage_entropy": terminology_entropy(terminology_metrics, "terminology_concept_coverage", scope) if has_scope else None,
            "annotation_topic_coverage_entropy": terminology_entropy(terminology_metrics, "annotation_topic_coverage", scope) if has_scope else None,
            "train_test_token_overlap": overlap_value(overlap_metrics, "token_overlap", scope),
            "train_test_mention_token_overlap": overlap_value(overlap_metrics, "mention_token_overlap", scope) if has_scope else None,
            "train_test_mention_string_overlap": overlap_value(overlap_metrics, "mention_overlap", scope) if has_scope else None,
            "train_test_identifier_overlap": overlap_value(overlap_metrics, "identifier_overlap", scope) if has_scope else None,
        }
        rows.append(row)
    return rows


def correlation_row(rows: list[dict[str, float | str | None]], dependent_variable: str) -> dict[str, float | str | None]:
    result: dict[str, float | str | None] = {"corpus": f"correlation_with_{dependent_variable}"}
    ys_by_row = [as_number(row.get(dependent_variable)) for row in rows]
    for variable in VARIABLES:
        pairs = [
            (as_number(row.get(variable)), y)
            for row, y in zip(rows, ys_by_row)
            if as_number(row.get(variable)) is not None and y is not None
        ]
        if variable == dependent_variable and len(pairs) >= 2:
            result[variable] = 1.0
            continue
        result[variable] = pearson([x for x, _ in pairs], [y for _, y in pairs]) if pairs else None
    return result


def write_tables(
    output_path: Path,
    basic: dict[str, list[dict[str, Any]]],
    metadata: dict[str, list[dict[str, Any]]],
    terminology: dict[str, list[dict[str, Any]]],
    overlap: dict[str, list[dict[str, Any]]],
) -> None:
    scopes = collect_scopes(basic, terminology, overlap)
    header = ["corpus"] + VARIABLES
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for index, scope in enumerate(scopes):
            if index:
                handle.write("\n")
            handle.write(f"Entity type:\t{scope}\n")
            handle.write("\t".join(header) + "\n")
            rows = build_rows(basic, metadata, terminology, overlap, scope)
            for row in rows:
                handle.write("\t".join(format_cell(row.get(column)) for column in header) + "\n")
            for dependent_variable in DEPENDENT_VARIABLES:
                row = correlation_row(rows, dependent_variable)
                handle.write("\t".join(format_cell(row.get(column)) for column in header) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write corpus/entity-scope correlation tables as tab-delimited text.")
    parser.add_argument("--basic", type=Path, default=Path("output/basic_corpus_stats.json"))
    parser.add_argument("--metadata", type=Path, default=Path("output/metadata_stats.json"))
    parser.add_argument("--terminology", type=Path, default=Path("output/terminology_coverage_stats.json"))
    parser.add_argument("--overlap", type=Path, default=Path("output/overlap_stats.json"))
    parser.add_argument("--output", type=Path, default=Path("output/correlation_tables.tsv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_tables(
        args.output,
        load_json(args.basic),
        load_json(args.metadata),
        load_json(args.terminology),
        load_json(args.overlap),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
