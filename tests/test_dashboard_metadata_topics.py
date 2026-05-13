from __future__ import annotations

import json

from corpus_benchmark.dashboard.builder import build_metadata_panels
from corpus_benchmark.dashboard.metadata import load_metadata_stats


def test_load_metadata_stats_keeps_journal_and_article_topics_separate(tmp_path) -> None:
    stats_path = tmp_path / "metadata_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "Example_corpus": [
                    {"metric_name": "journal_distribution", "value": {"Journal A": 1.0}},
                    {"metric_name": "publication_year_distribution", "value": {"2024": 1.0}},
                    {"metric_name": "journal_MeSH_topic_distribution", "value": {"Journal Topic": 1.0}},
                    {"metric_name": "article_MeSH_topic_distribution", "value": {"Article Topic": 0.75, "Unknown": 0.25}},
                ]
            }
        ),
        encoding="utf-8",
    )

    metadata = load_metadata_stats(stats_path)

    assert metadata["example"]["topic_dist"] == {"Journal Topic": 100.0}
    assert metadata["example"]["article_topic_dist"] == {"Article Topic": 75.0}


def test_build_metadata_panels_adds_article_topics_pane() -> None:
    corpora = [
        {
            "name": "Example",
            "metadata": {
                "journal": {"n_journals": 1, "top1_name": "Journal A", "top1_pct": 100.0, "top3_pct": 100.0},
                "year": {"mode_year": 2024, "year_min": 2024, "year_max": 2024, "span": 0, "decades": {2020: 100.0}, "year_pcts": {2024: 100.0}},
                "topic_dist": {"Journal Topic": 100.0},
                "article_topic_dist": {"Article Topic": 100.0},
                "has_metadata": True,
            },
        }
    ]

    tabs, panels = build_metadata_panels(corpora, ["#123456"])

    assert "Article topics" in tabs
    assert 'id="p11"' in panels
    assert "Article topic distribution per corpus" in panels
    assert "Article Topic" in panels
    assert "topic-heatmap" in panels
    assert "hm-cell" in panels
