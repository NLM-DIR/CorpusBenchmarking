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


def test_brat_standoff_can_map_filename_docids_before_regex(tmp_path) -> None:
    subset_dir = tmp_path / "test"
    subset_dir.mkdir()
    (subset_dir / "BC2GM000019979.txt").write_text("Example gene\n", encoding="utf-8")
    (subset_dir / "BC2GM000019979.ann").write_text("T1\tGENE 8 12\tgene\n", encoding="utf-8")
    idmap_path = tmp_path / "test.idmap"
    idmap_path.write_text("P00084003A0741 BC2GM000019979\n", encoding="utf-8")

    corpus = load_BRAT_standoff(
        paths={"test": str(subset_dir)},
        docid_regex=r"^P0*([0-9]+)[A-Z][0-9]+[a-z]*$",
        docid_map_path=str(idmap_path),
        docid_map_invert=True,
        label_map={"GENE": "Gene"},
    )

    document = corpus.subsets["test"].documents[0]

    assert document.document_id == "BC2GM000019979"
    assert document.identifiers == {DocumentIdentifierType.PMID: "84003"}
    assert document.passages[0].annotations[0].label == "Gene"
