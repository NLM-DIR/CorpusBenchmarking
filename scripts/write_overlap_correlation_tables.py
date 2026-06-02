from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


FREQUENCY_METRICS = [
    ("mention_overlap_by_frequency", "mention_overlap"),
    ("identifier_overlap_by_frequency", "identifier_overlap"),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def norm_corpus_name(name: str) -> str:
    name = name.lower()
    for suffix in ("_corpus", "_train", "_test", "_dev"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"[^a-z0-9]", "", name)


def corpus_from_split_pair(split_pair: str) -> str:
    match = re.match(r"\(([^,]+?)(?:_train|_test|_dev)\s*,", split_pair)
    if match:
        return norm_corpus_name(match.group(1))
    return norm_corpus_name(split_pair.strip("()").split(",", 1)[0])


def metric(metrics: list[dict[str, Any]], metric_name: str) -> dict[str, Any] | None:
    return next((item for item in metrics if item.get("metric_name") == metric_name), None)


def article_topic_entropy_by_corpus(metadata: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    entropies = {}
    for corpus_name, metrics in metadata.items():
        topic_metric = metric(metrics, "article_MeSH_topic_distribution")
        if not topic_metric:
            continue
        entropy = (topic_metric.get("details") or {}).get("entropy")
        if entropy is not None:
            entropies[norm_corpus_name(corpus_name)] = float(entropy)
    return entropies


def scope_payloads(frequency_metric: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted((frequency_metric.get("scopes") or {}).items())


def overall_payload(overall_metric: dict[str, Any] | None, scope: str) -> dict[str, Any] | None:
    if overall_metric is None:
        return None
    if scope == "all":
        return overall_metric
    return (overall_metric.get("scopes") or {}).get(scope)


def parse_bin(bin_label: str) -> tuple[int, float]:
    low, high = ast.literal_eval(bin_label)
    return int(low), float("inf") if high is None else float(high)


def sorted_bin_counts(bin_counts: dict[str, list[int]]) -> list[tuple[str, list[int]]]:
    return sorted(bin_counts.items(), key=lambda item: parse_bin(item[0]))


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def jaccard(intersection_count: int | float | None, union_count: int | float | None) -> float:
    if not union_count:
        return 0.0
    return float(intersection_count or 0) / float(union_count)


def share(count: int | float | None, denominator: int | float | None) -> float:
    if not denominator:
        return 0.0
    return float(count or 0) / float(denominator)


def make_row(
    split_pair: str,
    scope: str,
    bin_label: str,
    union_count: int | float | None,
    intersection_count: int | float | None,
    overall_union: int | float,
    overall_intersection: int | float,
    article_entropy: float | None,
) -> list[Any]:
    return [
        split_pair,
        scope,
        bin_label,
        union_count,
        intersection_count,
        jaccard(intersection_count, union_count),
        share(union_count, overall_union),
        share(intersection_count, overall_intersection),
        share(intersection_count, overall_union),
        article_entropy,
    ]


def table_rows(
    overlap: dict[str, list[dict[str, Any]]],
    entropy_by_corpus: dict[str, float],
    frequency_metric_name: str,
    overall_metric_name: str,
) -> list[list[Any]]:
    rows = []
    for split_pair, metrics in sorted(overlap.items()):
        frequency_metric = metric(metrics, frequency_metric_name)
        if frequency_metric is None:
            continue
        overall_metric = metric(metrics, overall_metric_name)
        article_entropy = entropy_by_corpus.get(corpus_from_split_pair(split_pair))

        for scope, payload in scope_payloads(frequency_metric):
            overall = overall_payload(overall_metric, scope)
            if overall is None:
                continue
            details = overall.get("details") or {}
            overall_union = details.get("union") or 0
            overall_intersection = details.get("intersection") or 0
            if overall_union == 0:
                continue
            rows.append(
                make_row(
                    split_pair,
                    scope,
                    "Overall",
                    overall_union,
                    overall_intersection,
                    overall_union,
                    overall_intersection,
                    article_entropy,
                )
            )

            bin_counts = (payload.get("details") or {}).get("bin_counts") or {}
            for bin_label, counts in sorted_bin_counts(bin_counts):
                union_count = counts[0] if len(counts) > 0 else None
                intersection_count = counts[1] if len(counts) > 1 else None
                rows.append(
                    make_row(
                        split_pair,
                        scope,
                        bin_label,
                        union_count,
                        intersection_count,
                        overall_union,
                        overall_intersection,
                        article_entropy,
                    )
                )
    return rows


def write_tables(
    output_path: Path,
    overlap: dict[str, list[dict[str, Any]]],
    metadata: dict[str, list[dict[str, Any]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entropy_by_corpus = article_topic_entropy_by_corpus(metadata)
    header = [
        "Split pair",
        "Entity scope",
        "Bin",
        "Union count",
        "Intersection count",
        "Jaccard",
        "Union share",
        "Intersection share",
        "Contribution to Overall Jaccard",
        "Article topic distribution entropy",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for index, (frequency_metric_name, overall_metric_name) in enumerate(FREQUENCY_METRICS):
            if index:
                handle.write("\n")
            handle.write(f"Metric:\t{frequency_metric_name}\n")
            handle.write("\t".join(header) + "\n")
            for row in table_rows(overlap, entropy_by_corpus, frequency_metric_name, overall_metric_name):
                handle.write("\t".join(format_value(value) for value in row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write overlap frequency bin count tables with article topic entropy."
    )
    parser.add_argument("--overlap", type=Path, default=Path("output/overlap_stats.json"))
    parser.add_argument("--metadata", type=Path, default=Path("output/metadata_stats.json"))
    parser.add_argument("--output", type=Path, default=Path("output/overlap_correlation_tables.tsv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_tables(args.output, load_json(args.overlap), load_json(args.metadata))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
