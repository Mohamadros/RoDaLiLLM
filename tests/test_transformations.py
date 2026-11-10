from __future__ import annotations

import pytest

from rodali_attacker import RecordTransformer
from rodali_attacker.models import ValidationStatus


@pytest.fixture
def record():
    return {
        "name": "Nike Air Force sneaker",
        "brand": "Nike",
        "description": "White sneaker, comfortable everyday shoe",
        "model_number": "AF1-42",
        "size": "EU 42",
        "color": "White",
    }


def test_protected_fields_and_original_are_unchanged(record):
    original = record.copy()
    transformer = RecordTransformer()
    result = transformer.transform(record, "case_change", seed=1)
    assert record == original
    for field in transformer.protected_fields & record.keys():
        assert result.transformed_record[field] == record[field]


def test_fixed_seed_is_deterministic(record):
    transformer = RecordTransformer()
    first = transformer.transform(record, "char_insert", seed=7)
    second = transformer.transform(record, "char_insert", seed=7)
    assert first == second
    assert first.validation_status is ValidationStatus.REVIEW_REQUIRED


def test_candidates_are_deduplicated(record):
    candidates = RecordTransformer().generate_candidates(
        record, ["case_change", "punctuation_change", "char_delete"], 10, seed=3
    )
    rendered = [repr(sorted(item.transformed_record.items())) for item in candidates]
    assert len(rendered) == len(set(rendered))


def test_combined_history_and_budget(record):
    result = RecordTransformer().transform(
        record, ["case_change", "punctuation_change"], seed=1, intensity="medium"
    )
    assert [step.name for step in result.history] == ["case_change", "punctuation_change"]
    with pytest.raises(ValueError, match="budget"):
        RecordTransformer().transform(
            record,
            ["case_change", "punctuation_change", "char_delete"],
            seed=1,
            intensity="medium",
        )


def test_invalid_strategy_is_clear(record):
    with pytest.raises(ValueError, match="Unknown strategy"):
        RecordTransformer().transform(record, "teleport", seed=1)


def test_embedded_variant_change_is_rejected():
    transformer = RecordTransformer(protected_fields=set(), optional_fields={"description"})
    result = transformer.transform(
        {"description": "Phone 128GB black"}, "omit_optional", seed=1
    )
    assert result.validation_status is ValidationStatus.REJECTED
    assert any("critical" in note for note in result.validation_notes)


def test_german_and_english_language_variation():
    transformer = RecordTransformer(protected_fields={"brand"})
    english = transformer.transform(
        {"brand": "Nike", "description": "Comfortable white sneaker"},
        "language_variation",
        seed=2,
    )
    german = transformer.transform(
        {"brand": "Nike", "description": "Bequemer weißer Schuh"},
        "case_change",
        seed=2,
    )
    assert "Sneaker" not in english.transformed_record["description"]
    assert german.transformed_record["description"] != "Bequemer weißer Schuh"
    assert english.validation_status is ValidationStatus.SAFE


def test_empty_and_missing_fields_are_safe():
    result = RecordTransformer().transform({}, "case_change", seed=1)
    assert result.transformed_record == {}
    assert result.validation_status is ValidationStatus.REJECTED


def test_controlled_abbreviation_requires_explicit_mapping():
    unmapped = RecordTransformer().transform(
        {"name": "Nike Air Force 1"}, "controlled_abbreviation", seed=1
    )
    mapped = RecordTransformer(abbreviations={"Air Force": "AF"}).transform(
        {"name": "Nike Air Force sneaker"}, "controlled_abbreviation", seed=1
    )
    assert unmapped.validation_status is ValidationStatus.REJECTED
    assert mapped.transformed_record["name"] == "Nike AF sneaker"
