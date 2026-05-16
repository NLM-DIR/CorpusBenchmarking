from __future__ import annotations

from corpus_benchmark.dashboard import PALETTE, _entity_profile_data, build_html, build_terminology_panels


def _corpus(
    name: str,
    label_counts: dict[str, int],
    *,
    doc_count: int = 10,
    has_ids: bool = True,
) -> dict:
    total_ann = sum(label_counts.values())
    return {
        "name": name,
        "raw_name": f"{name}_corpus",
        "doc_count": doc_count,
        "token_count": 100,
        "n_types": len(label_counts),
        "types": list(label_counts),
        "label_counts": label_counts,
        "total_ann": total_ann,
        "ann_per_doc": total_ann / doc_count,
        "men_per_doc": total_ann / doc_count,
        "ids_per_doc": 1.5 if has_ids else None,
        "has_ids": has_ids,
        "id_status": "MESH" if has_ids else "none",
        "id_class": "yes" if has_ids else "no",
        "id_vocab": "MESH" if has_ids else "none",
        "ambiguity": 1.0,
        "variation": 2.0,
        "entropy": 1.0,
    }


def test_entity_profile_data_filters_and_recomputes_label_metrics() -> None:
    corpora = [
        _corpus("Mixed", {"Disease": 30, "Chemical": 70}, doc_count=10),
        _corpus("ChemicalOnly", {"Chemical": 20}, doc_count=5),
    ]
    for index, corpus in enumerate(corpora):
        corpus["color_index"] = index

    config = {
        "entity_scopes": {
            "all": {"label": "All annotations", "include_all": True},
            "disease": {"label": "Diseases", "labels": ["Disease"]},
        }
    }

    profiles = _entity_profile_data(corpora, PALETTE, config)

    assert profiles["all"]["nCorpora"] == 2
    assert profiles["disease"]["nCorpora"] == 1
    assert profiles["disease"]["nWithIds"] == 1
    assert profiles["disease"]["ann"]["labels"] == ["Mixed"]
    assert profiles["disease"]["ann"]["data"] == [3.0]
    assert profiles["disease"]["types"]["data"] == [1]
    assert "ChemicalOnly" not in profiles["disease"]["tableRows"]


def test_entity_profile_data_uses_nested_scoped_metrics() -> None:
    corpora = [
        _corpus("Mixed", {"Disease": 30, "Chemical": 70}, doc_count=10),
        _corpus("TextOnly", {"Disease": 5}, doc_count=5, has_ids=False),
    ]
    corpora[0]["metric_results"] = [
        {
            "metric_name": "label_distribution",
            "value": {"Disease": 0.3, "Chemical": 0.7},
            "details": {"counts": {"Disease": 30, "Chemical": 70}, "total": 100},
            "scopes": {
                "disease": {
                    "value": {"Disease": 1.0},
                    "details": {"counts": {"Disease": 30}, "total": 30},
                }
            },
        },
        {
            "metric_name": "annotations_per_document_stats",
            "value": {"mean": 10.0},
            "scopes": {"disease": {"value": {"mean": 3.0}}},
        },
        {
            "metric_name": "annotations_per_1000_tokens_stats",
            "value": {"mean": 100.0},
            "scopes": {"disease": {"value": {"mean": 30.0}}},
        },
        {
            "metric_name": "unique_identifiers_per_document_stats",
            "value": {"mean": 5.0},
            "scopes": {"disease": {"value": {"mean": 2.0}}},
        },
        {
            "metric_name": "identifier_resource_distribution",
            "value": {"MESH": 1.0},
            "scopes": {"disease": {"value": {"MESH": 1.0}}},
        },
        {
            "metric_name": "ambiguity_degree_stats",
            "value": {"mean": 1.0},
            "scopes": {"disease": {"value": {"mean": 1.25}}},
        },
        {
            "metric_name": "variation_degree_stats",
            "value": {"mean": 2.0},
            "scopes": {"disease": {"value": {"mean": 3.5}}},
        },
    ]
    corpora[1]["metric_results"] = [
        {
            "metric_name": "identifier_resource_distribution",
            "value": {},
            "scopes": {"disease": {"value": {}}},
        }
    ]
    for index, corpus in enumerate(corpora):
        corpus["color_index"] = index

    config = {
        "entity_scopes": {
            "all": {"label": "All annotations", "include_all": True},
            "disease": {"label": "Diseases", "labels": ["Disease"]},
        }
    }

    profiles = _entity_profile_data(corpora, PALETTE, config)

    assert profiles["disease"]["ann"]["data"] == [3.0, 1.0]
    assert profiles["disease"]["ann1k"]["data"] == [30.0, 0]
    assert profiles["disease"]["ids"]["data"] == [2.0, 0]
    assert profiles["disease"]["amb"]["labels"] == ["Mixed"]
    assert profiles["disease"]["amb"]["data"] == [1.25]
    assert profiles["disease"]["variation"]["data"] == [3.5]


def test_terminology_panel_chart_helper_uses_chartjs_config_labels() -> None:
    _, panels = build_terminology_panels(
        {
            "Example": {
                "by_scope": {
                    "all": [
                            {
                                "corpus": "Example",
                                "display_name": "Example",
                                "terminology": "mesh",
                                "terminology_label": "MeSH",
                                "series_label": "Example / mesh",
                                "n_input_ids": 1,
                                "n_missing_ids": 0,
                                "n_unique_missing_ids": 0,
                                "unique_missing": 0,
                                "missing_pct": 0.0,
                                "coverage_pct": 100.0,
                                "mean_depth": "1.00",
                            "high_level": [
                                {
                                    "branch_code": "D000001",
                                    "label": "Root",
                                    "count": 1,
                                    "terminology_total_count": 10,
                                    "proportion": 0.1,
                                }
                            ],
                            "depth": [
                                {
                                    "depth": 1,
                                    "count": 1,
                                    "terminology_total_count": 10,
                                    "proportion": 0.1,
                                }
                            ],
                        }
                    ]
                }
            }
        }
    )

    assert "const labels = config && config.data && config.data.labels;" in panels
    assert "termDepthCharts" in panels
    assert "termTerminologyCoverageCharts" in panels
    assert "tmc3_${i}" in panels
    assert "tmc4_${i}" in panels
    assert "!config.labels" not in panels


def test_dashboard_panel_labels_order_and_removed_cascade_tab() -> None:
    corpora = [_corpus("Example", {"Disease": 5})]
    corpora[0]["overlap"] = {
        "train_size": 10,
        "test_size": 5,
        "token_overlap": 0.5,
        "mention_token_overlap": 0.4,
        "mention_overlap": 0.3,
        "identifier_overlap": 0.2,
    }
    corpora[0]["metadata"] = {
        "has_metadata": True,
        "journal": {"n_journals": 1, "top1_pct": 100, "top3_pct": 100},
        "year": {
            "year_min": 2020,
            "year_max": 2021,
            "mode_year": 2020,
            "decades": {2020: 100.0},
            "year_pcts": {2020: 50.0, 2021: 50.0},
        },
        "topic_dist": {"Oncology": 100.0},
        "article_topic_dist": {"Oncology": 100.0},
    }
    corpora[0]["terminology"] = {
        "by_scope": {
            "all": [
                {
                    "display_name": "Example",
                    "terminology": "mesh",
                    "terminology_label": "MeSH",
                    "series_label": "Example / MeSH",
                    "n_input_ids": 5,
                    "n_missing_ids": 1,
                    "unique_missing": 1,
                    "missing_pct": 20.0,
                    "coverage_pct": 80.0,
                    "mean_depth": 2.0,
                    "branches": {
                        "C": {
                            "label": "Diseases",
                            "count": 1,
                            "annotation_count": 4,
                            "proportion": 0.1,
                            "annotation_proportion": 0.8,
                            "total": 10,
                        }
                    },
                    "depth": {"2": {"count": 4, "proportion": 0.8, "total": 10}},
                }
            ]
        }
    }

    html = build_html(corpora)

    assert html.index("data-p=\"p5\">Summary table") < html.index("data-p=\"p1\">Annotation density")
    assert "Identifier density" in html
    assert "Lexical / conceptual structure" in html
    assert "Deprecated terms" in html
    assert "Terminology coverage" in html
    assert "Annotation topic coverage" in html
    assert "Cascade view" not in html
    assert html.index("Overlap cascade") > html.index("Train-test overlap")
    assert html.index("Article topics") < html.index("Journal topics")
    assert "Unique missing IDs" not in html
    assert "Resolvable identifier rate" in html
    assert "Current (%)" in html
    assert "Deprecated (%)" in html
