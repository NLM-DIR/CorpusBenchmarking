from __future__ import annotations

import pickle

from corpus_benchmark.loaders.terminology_loaders import load_mesh_xml, load_obo
from corpus_benchmark.models.config import WorkspaceConfig
from corpus_benchmark.models.terminologies import TerminologyConcept, TerminologyResource


def test_load_mesh_xml_normalizes_major_topic_marker_in_supplemental_mappings(tmp_path) -> None:
    descriptor_path = tmp_path / "desc.xml"
    supplemental_path = tmp_path / "supp.xml"
    terminology_dir = tmp_path / "terminologies"

    descriptor_path.write_text(
        """<DescriptorRecordSet>
<DescriptorRecord>
  <DescriptorUI>D009652</DescriptorUI>
  <DescriptorName><String>Norpregnadienes</String></DescriptorName>
  <TreeNumberList><TreeNumber>D04.210</TreeNumber></TreeNumberList>
</DescriptorRecord>
</DescriptorRecordSet>""",
        encoding="utf-8",
    )
    supplemental_path.write_text(
        """<SupplementalRecordSet>
<SupplementalRecord>
  <SupplementalRecordUI>C033273</SupplementalRecordUI>
  <SupplementalRecordName><String>Gestodene</String></SupplementalRecordName>
  <HeadingMappedToList>
    <HeadingMappedTo>
      <DescriptorReferredTo><DescriptorUI>*D009652</DescriptorUI></DescriptorReferredTo>
    </HeadingMappedTo>
  </HeadingMappedToList>
</SupplementalRecord>
</SupplementalRecordSet>""",
        encoding="utf-8",
    )

    terminology = load_mesh_xml(
        WorkspaceConfig(terminology_dir=str(terminology_dir)),
        name="mesh_test",
        year=2026,
        descriptor_path=str(descriptor_path),
        supplemental_path=str(supplemental_path),
    )

    supplemental = terminology.get_concept("C033273")
    assert supplemental is not None
    assert supplemental.mapped_ui_ids == ["D009652"]
    assert [concept.ui for concept in terminology.resolve_to_tree_concepts("C033273")] == ["D009652"]


def test_load_mesh_xml_repairs_cached_major_topic_markers(tmp_path) -> None:
    terminology_dir = tmp_path / "terminologies"
    terminology_dir.mkdir()
    cache_path = terminology_dir / "mesh_test.pkl"
    cached = TerminologyResource(
        name="mesh_test",
        concepts={
            "D009652": TerminologyConcept(ui="D009652", name="Norpregnadienes", tree_numbers=["D04.210"]),
            "C033273": TerminologyConcept(ui="C033273", name="Gestodene", mapped_ui_ids=["*D009652"]),
        },
    )
    with cache_path.open("wb") as fh:
        pickle.dump(cached, fh)

    terminology = load_mesh_xml(WorkspaceConfig(terminology_dir=str(terminology_dir)), name="mesh_test", year=2026)

    supplemental = terminology.get_concept("C033273")
    assert supplemental is not None
    assert supplemental.mapped_ui_ids == ["D009652"]
    assert [concept.ui for concept in terminology.resolve_to_tree_concepts("C033273")] == ["D009652"]

    with cache_path.open("rb") as fh:
        repaired_cache = pickle.load(fh)
    assert repaired_cache.get_concept("C033273").mapped_ui_ids == ["D009652"]


def test_load_obo_parses_terms_relationships_alt_ids_and_skips_obsolete(tmp_path) -> None:
    obo_path = tmp_path / "mini.obo"
    terminology_dir = tmp_path / "terminologies"
    obo_path.write_text(
        """format-version: 1.2

[Term]
id: TEST:0001
name: root

[Term]
id: TEST:0002
name: child
alt_id: TEST:ALT2
synonym: "child synonym" EXACT []
is_a: TEST:0001 ! root

[Term]
id: TEST:0003
name: obsolete
is_obsolete: true
""",
        encoding="utf-8",
    )

    terminology = load_obo(
        WorkspaceConfig(terminology_dir=str(terminology_dir)),
        name="test_obo",
        path=str(obo_path),
        prefix="TEST",
        resource_aliases=["TEST"],
    )

    assert terminology.get_concept("0002").name == "child"
    assert terminology.get_concept("TEST:ALT2").ui == "TEST:0002"
    assert terminology.get_concept("TEST:0003") is None
    assert terminology.top_ancestor_ids("TEST:0002") == ["TEST:0001"]
    assert terminology.depth_for_concept(terminology.get_concept("TEST:0002")) == 2
