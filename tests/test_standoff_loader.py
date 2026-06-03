from __future__ import annotations

from corpus_benchmark.loaders.standoff_loader import load_BRAT_standoff
from corpus_benchmark.models.corpus import DocumentIdentifierType


def test_brat_standoff_loads_normalization_links(tmp_path) -> None:
    subset_dir = tmp_path / "train"
    subset_dir.mkdir()
    (subset_dir / "12345.txt").write_text("Homo sapiens and Gouldian finches\n", encoding="utf-8")
    (subset_dir / "12345.ann").write_text(
        "T1\tSpecies 0 12\tHomo sapiens\n"
        "N1\tReference T1 Taxonomy:9606\t\n"
        "T2\tSpecies 17 25;26 33\tGouldian finches\n"
        "N2\tReference T2 Taxonomy:44316\t\n",
        encoding="utf-8",
    )

    corpus = load_BRAT_standoff(
        paths={"train": str(subset_dir)},
        normalization_resource_map={"Taxonomy": "NCBITaxon"},
    )

    document = corpus.subsets["train"].documents[0]
    first_annotation = document.passages[0].annotations[0]
    second_annotation = document.passages[0].annotations[1]

    assert document.identifiers == {DocumentIdentifierType.PMID: "12345"}
    assert first_annotation.label == "Species"
    assert [(link.resource, link.identifier) for link in first_annotation.get_identifier_links()] == [
        ("NCBITaxon", "9606")
    ]
    assert [(span.start, span.end) for span in second_annotation.spans] == [(17, 25), (26, 33)]
    assert [(link.resource, link.identifier) for link in second_annotation.get_identifier_links()] == [
        ("NCBITaxon", "44316")
    ]
