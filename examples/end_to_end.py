"""Offline end-to-end demonstration; no API key or network request required."""

from __future__ import annotations

import json
from pathlib import Path

from rodali_attacker import LLMMatcher, MockProvider, RecordTransformer
from rodali_attacker.models import ValidationStatus


def main() -> None:
    record_a = {
        "name": "Nike Air Force 1 '07",
        "brand": "Nike",
        "description": "White sneaker, size EU 42",
        "size": "EU 42",
        "color": "White",
    }
    record_b = {
        "name": "Nike AF1 Low '07",
        "brand": "Nike",
        "description": "Triple White, EU size 42",
        "size": "EU 42",
        "color": "White",
    }
    # A mock makes the workflow runnable offline. Replace it with
    # OpenAIProvider() for a real evaluation.
    matcher = LLMMatcher(
        MockProvider(responder=lambda _prompt: "MATCH"),
        model="offline-demo",
        strategy="rule_guided",
    )
    baseline = matcher.match(record_a, record_b, pair_id="pair_001")

    transformer = RecordTransformer(
        abbreviations={"Air Force 1": "AF1"},
        protected_fields={"brand", "size", "color"},
    )
    candidates = transformer.generate_candidates(
        record_b,
        ["case_change", "controlled_abbreviation", "punctuation_change", "char_transpose"],
        number_of_candidates=6,
        seed=42,
    )

    rows = []
    print("Pair ID | Transformation | Original | New | Ground truth | Validation")
    for index, candidate in enumerate(candidates, 1):
        if candidate.validation_status is ValidationStatus.REJECTED:
            continue
        prediction = matcher.match(
            record_a, candidate.transformed_record, pair_id=f"pair_001_t{index:02d}"
        )
        row = {
            "pair_id": prediction.pair_id,
            "transformation": candidate.transformation_name,
            "original_prediction": baseline.prediction,
            "new_prediction": prediction.prediction,
            "ground_truth": "MATCH",
            "transformation_metadata": candidate.to_dict(),
        }
        rows.append(row)
        print(
            f"{prediction.pair_id} | {candidate.transformation_name} | "
            f"{baseline.prediction} | {prediction.prediction} | MATCH | "
            f"{candidate.validation_status.value}"
        )

    destination = Path("results/demo_predictions.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} rows to {destination}")


if __name__ == "__main__":
    main()
