"""Loader for HPI FreeDB CD Dataset 1 and its benchmark pair lists."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CD_COLUMNS = ("artist", "category", "genre", "title", "tracks", "year", "id")
PAIR_COLUMNS = ("id1", "id2")


class DatasetFormatError(ValueError):
    """A benchmark file does not satisfy the observed Dataset 1 contract."""


@dataclass(frozen=True, slots=True)
class CDPair:
    """One labeled benchmark pair joined to its two source records."""

    pair_id: str
    record_id_a: str
    record_id_b: str
    record_a: dict[str, Any]
    record_b: dict[str, Any]
    label: int


def _require_header(path: Path, actual: list[str] | None, expected: tuple[str, ...]) -> None:
    if actual != list(expected):
        raise DatasetFormatError(
            f"Unexpected columns in {path}: expected {list(expected)!r}, got {actual!r}"
        )


def _matcher_record(row: dict[str, str]) -> dict[str, Any]:
    """Map a raw CD row to the generic matcher input without its arbitrary ID.

    Empty track slots at the end of the source's pipe-delimited track field are
    omitted to avoid wasting model context; non-empty values are unchanged.
    """

    tracks = row["tracks"].rstrip("|")
    return {
        "artist": row["artist"],
        "category": row["category"],
        "genre": row["genre"],
        "title": row["title"],
        "tracks": tracks.split("|") if tracks else [],
        "year": row["year"],
    }


def load_cd_records(path: str | Path) -> dict[str, dict[str, str]]:
    """Load the UTF-8, quoted TSV record file, indexed by its ``id`` column."""

    source = Path(path)
    records: dict[str, dict[str, str]] = {}
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", quotechar='"')
            _require_header(source, reader.fieldnames, CD_COLUMNS)
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise DatasetFormatError(f"Malformed row {line_number} in {source}")
                record_id = row["id"]
                if not record_id:
                    raise DatasetFormatError(f"Empty id at row {line_number} in {source}")
                if record_id in records:
                    raise DatasetFormatError(f"Duplicate id {record_id!r} in {source}")
                records[record_id] = row
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(f"{source} is not valid UTF-8") from exc
    return records


def load_pair_list(
    path: str | Path,
    records: dict[str, dict[str, str]],
    *,
    label: int,
) -> list[CDPair]:
    """Load an ASCII-compatible pair TSV and join IDs to source records."""

    if label not in {0, 1}:
        raise ValueError("label must be 0 or 1")
    source = Path(path)
    pairs: list[CDPair] = []
    try:
        with source.open("r", encoding="ascii", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", quotechar='"')
            _require_header(source, reader.fieldnames, PAIR_COLUMNS)
            for line_number, row in enumerate(reader, start=2):
                id_a, id_b = row["id1"], row["id2"]
                missing = [record_id for record_id in (id_a, id_b) if record_id not in records]
                if missing:
                    raise DatasetFormatError(
                        f"Unknown record id(s) {missing!r} at row {line_number} in {source}"
                    )
                pairs.append(
                    CDPair(
                        pair_id=f"cd_{'dup' if label else 'nondup'}_{id_a}_{id_b}",
                        record_id_a=id_a,
                        record_id_b=id_b,
                        record_a=_matcher_record(records[id_a]),
                        record_b=_matcher_record(records[id_b]),
                        label=label,
                    )
                )
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(f"{source} is not valid ASCII") from exc
    return pairs


def load_cd_dataset(
    records_path: str | Path,
    duplicate_pairs_path: str | Path,
    nonduplicate_pairs_path: str | Path,
) -> tuple[list[CDPair], list[CDPair]]:
    """Load and join the positive and negative halves of CD Dataset 1."""

    records = load_cd_records(records_path)
    positives = load_pair_list(duplicate_pairs_path, records, label=1)
    negatives = load_pair_list(nonduplicate_pairs_path, records, label=0)
    return positives, negatives


def sample_pairs(
    positives: list[CDPair],
    negatives: list[CDPair],
    *,
    per_class: int | None,
    seed: int,
) -> list[CDPair]:
    """Return a deterministic stratified sample, shuffled across classes."""

    if per_class is not None and per_class < 0:
        raise ValueError("per_class must be non-negative")
    rng = random.Random(seed)
    if per_class is None:
        chosen = [*positives, *negatives]
    else:
        if per_class > len(positives) or per_class > len(negatives):
            raise ValueError(
                f"Requested {per_class} per class, but only {len(positives)} positive "
                f"and {len(negatives)} negative pairs are available"
            )
        chosen = [*rng.sample(positives, per_class), *rng.sample(negatives, per_class)]
    rng.shuffle(chosen)
    return chosen
