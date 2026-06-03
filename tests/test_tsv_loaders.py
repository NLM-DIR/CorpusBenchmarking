from __future__ import annotations

from corpus_benchmark.loaders.tsv_loaders import load_linnaeus_species, load_mutationfinder
from corpus_benchmark.models.corpus import DocumentIdentifierType


def test_linnaeus_species_loader_reads_tags_and_pmcids(tmp_path) -> None:
    text_dir = tmp_path / "txt"
    text_dir.mkdir()
    (text_dir / "pmcA123.txt").write_text("Yeast likes humans.", encoding="utf-8")
    tags_path = tmp_path / "filtered_tags.tsv"
    tags_path.write_text(
        "#entity id\tdocument\tstart\tend\ttext\tcomment\n"
        "species:ncbi:4932\tpmcA123\t0\t5\tYeast\t\n",
        encoding="utf-8",
    )

    corpus = load_linnaeus_species(str(text_dir), str(tags_path))
    document = corpus.subsets["all"].documents[0]
    annotation = document.passages[0].annotations[0]

    assert document.identifiers == {DocumentIdentifierType.PMCID: "PMC123"}
    assert annotation.label == "Species"
    assert [(link.resource, link.identifier) for link in annotation.get_identifier_links()] == [
        ("NCBITaxon", "4932")
    ]


def test_mutationfinder_loader_reads_normalized_mutation_gold(tmp_path) -> None:
    text_path = tmp_path / "devo_set.txt"
    gold_path = tmp_path / "devo_gold_std.txt"
    text_path.write_text(
        "123\tA mutant title K12A\tThe normalized mutation Q34R is present.\n",
        encoding="utf-8",
    )
    gold_path.write_text("123\tK12A\tMISSING1\tQ34R\n", encoding="utf-8")

    corpus = load_mutationfinder(paths={"dev": str(text_path)}, gold_paths={"dev": str(gold_path)})
    document = corpus.subsets["dev"].documents[0]
    annotations = document.passages[0].annotations + document.passages[1].annotations

    assert document.identifiers == {DocumentIdentifierType.PMID: "123"}
    assert [annotation.text for annotation in annotations] == ["K12A", "MISSING1", "Q34R"]
    assert [(span.start, span.end) for span in annotations[0].spans] == [(15, 19)]
    assert annotations[1].spans == []
    assert [(link.resource, link.identifier) for link in annotations[2].get_identifier_links()] == [
        ("MutationFinder", "Q34R")
    ]
