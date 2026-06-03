from __future__ import annotations

from corpus_benchmark.loaders.splits import load_split_map


def test_split_files_can_read_whitespace_separated_id_lists(tmp_path) -> None:
    train_path = tmp_path / "train.txt"
    test_path = tmp_path / "test.txt"
    train_path.write_text("1 2\n3\n", encoding="utf-8")
    test_path.write_text("4 5", encoding="utf-8")

    split_map = load_split_map(
        {
            "source": "files",
            "delimiter": "whitespace",
            "id_column": "all",
            "files": {
                "train": str(train_path),
                "test": str(test_path),
            },
        }
    )

    assert split_map == {
        "1": "train",
        "2": "train",
        "3": "train",
        "4": "test",
        "5": "test",
    }
