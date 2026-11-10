from __future__ import annotations

import csv

import pytest

from rodali_attacker.cd_dataset import (
    DatasetFormatError,
    load_cd_dataset,
    load_cd_records,
    sample_pairs,
)


def _write_tsv(path, columns, rows, encoding="utf-8"):
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def benchmark_files(tmp_path):
    records = tmp_path / "cddb.tsv"
    dpl = tmp_path / "cddb_DPL.tsv"
    ndpl = tmp_path / "cddb_NDPL.tsv"
    columns = ["artist", "category", "genre", "title", "tracks", "year", "id"]
    _write_tsv(
        records,
        columns,
        [
            {"artist": "Artist", "category": "rock", "genre": "Rock", "title": "Album", "tracks": "One|Two|||", "year": "2001", "id": "10"},
            {"artist": "artist", "category": "misc", "genre": "", "title": "album", "tracks": "One|Two|||", "year": "", "id": "20"},
            {"artist": "Other", "category": "jazz", "genre": "Jazz", "title": "Else", "tracks": "Three|||", "year": "1999", "id": "30"},
        ],
    )
    _write_tsv(dpl, ["id1", "id2"], [{"id1": "10", "id2": "20"}], "ascii")
    _write_tsv(ndpl, ["id1", "id2"], [{"id1": "10", "id2": "30"}], "ascii")
    return records, dpl, ndpl


def test_loads_and_joins_actual_schema(benchmark_files):
    positives, negatives = load_cd_dataset(*benchmark_files)
    assert positives[0].label == 1
    assert negatives[0].label == 0
    assert positives[0].record_id_a == "10"
    assert positives[0].record_a["tracks"] == ["One", "Two"]
    assert "id" not in positives[0].record_a


def test_rejects_schema_drift_and_unknown_ids(benchmark_files, tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("made_up\tid\nvalue\t1\n", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="Unexpected columns"):
        load_cd_records(bad)

    records, dpl, _ = benchmark_files
    unknown = tmp_path / "unknown.tsv"
    _write_tsv(unknown, ["id1", "id2"], [{"id1": "10", "id2": "999"}], "ascii")
    with pytest.raises(DatasetFormatError, match="Unknown record"):
        load_cd_dataset(records, dpl, unknown)


def test_stratified_sampling_is_reproducible(benchmark_files):
    positives, negatives = load_cd_dataset(*benchmark_files)
    first = sample_pairs(positives, negatives, per_class=1, seed=42)
    second = sample_pairs(positives, negatives, per_class=1, seed=42)
    assert first == second
    assert {pair.label for pair in first} == {0, 1}
