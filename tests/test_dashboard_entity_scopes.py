from __future__ import annotations

from corpus_benchmark.dashboard import PALETTE, _entity_profile_data


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
