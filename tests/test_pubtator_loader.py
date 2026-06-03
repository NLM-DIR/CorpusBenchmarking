from __future__ import annotations

import gzip

from corpus_benchmark.loaders.bioc_loader import load_pubtator


def test_pubtator_loader_reads_gzip_and_maps_label_resources(tmp_path) -> None:
    path = tmp_path / "corpus.PubTator.gz"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write(
            "123|t|Title\n"
            "123|a|Abstract text\n"
            "123\t0\t5\tTitle\tGeneOrGeneProduct\t1,2\n\n"
        )

    corpus = load_pubtator(
        path=str(path),
        id_format_list=[[",", "distributive", "False"]],
        id_resource_map={"Gene": "NCBIGene"},
        label_map={"GeneOrGeneProduct": "Gene"},
    )

    annotation = corpus.subsets["all"].documents[0].passages[0].annotations[0]

    assert annotation.label == "Gene"
    assert [(link.resource, link.identifier) for link in annotation.get_identifier_links()] == [
        ("NCBIGene", "1"),
        ("NCBIGene", "2"),
    ]


def test_pubtator_loader_maps_delimited_semantic_type_labels(tmp_path) -> None:
    path = tmp_path / "corpus.PubTator"
    path.write_text(
        "123|t|Title\n"
        "123|a|Abstract text\n"
        "123\t0\t5\tTitle\tT116,T123\tC1\n\n",
        encoding="utf-8",
    )

    corpus = load_pubtator(
        path=str(path),
        default_resource="UMLS",
        label_delimiter=",",
        multi_label_strategy="first",
        label_map={"T116": "Gene", "T123": "Chemical"},
    )

    annotation = corpus.subsets["all"].documents[0].passages[0].annotations[0]

    assert annotation.label == "Gene"
    assert [(link.resource, link.identifier) for link in annotation.get_identifier_links()] == [
        ("UMLS", "C1")
    ]
