from __future__ import annotations

from pathlib import Path

import corpus_benchmark.acquisition as acquisition


def test_download_directory_downloads_files_within_listing(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_download(url: str, dest_path: Path, **kwargs) -> None:
        calls.append((url, dest_path))
        if dest_path.name.endswith(".listing.html"):
            dest_path.write_text(
                """
                <a href="../">Parent Directory</a>
                <a href="123.BioC.XML">123.BioC.XML</a>
                <a href="BioC.dtd">BioC.dtd</a>
                <a href="nested/">nested/</a>
                <a href="https://example.org/outside.txt">outside</a>
                """,
                encoding="utf-8",
            )
        else:
            dest_path.write_text("downloaded", encoding="utf-8")

    monkeypatch.setattr(acquisition, "download_file", fake_download)

    acquisition._download_directory("https://example.org/pub/NLMGene/Corpus/", tmp_path, "agent")

    assert (tmp_path / ".Corpus.listing.html").exists()
    assert (tmp_path / "Corpus" / "123.BioC.XML").read_text(encoding="utf-8") == "downloaded"
    assert (tmp_path / "Corpus" / "BioC.dtd").read_text(encoding="utf-8") == "downloaded"
    assert not (tmp_path / "Corpus" / "outside.txt").exists()
