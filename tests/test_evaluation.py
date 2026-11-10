from __future__ import annotations

import csv
import json

from rodali_attacker import LLMMatcher, MockProvider
from rodali_attacker.cd_dataset import CDPair
from rodali_attacker.evaluation import calculate_metrics, evaluate_pairs, save_evaluation
from rodali_attacker.prompts import build_prompt


def _pairs():
    return [
        CDPair(f"pair_{index}", str(index), str(index + 10), {"artist": "A"}, {"artist": "B"}, label)
        for index, label in enumerate([1, 0, 0, 1], 1)
    ]


def test_cd_prompt_is_domain_specific_and_contains_no_label():
    prompt, version = build_prompt({"artist": "A"}, {"artist": "B"}, "cd_benchmark")
    assert version == "cd-benchmark-v1"
    assert "FreeDB CD benchmark" in prompt
    assert "ground_truth" not in prompt
    assert '"label"' not in prompt


def test_individual_calls_metrics_and_saved_outputs(tmp_path):
    provider = MockProvider(["MATCH", "NON_MATCH", "MATCH", "NON_MATCH"])
    matcher = LLMMatcher(provider, model="mock-cd", strategy="cd_benchmark")
    rows = evaluate_pairs(matcher, _pairs())
    assert len(provider.prompts) == 4
    metrics = calculate_metrics(rows)
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1

    csv_path, metrics_path = save_evaluation(rows, metrics, tmp_path / "run.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["record_id_a"] == "1"
    assert saved[0]["model"] == "mock-cd"
    assert saved[0]["prompt_version"] == "cd-benchmark-v1"
    assert json.loads(metrics_path.read_text())["false_positives"] == 1


def test_failed_predictions_are_reported_and_excluded_from_metrics():
    matcher = LLMMatcher(MockProvider(["unexpected"]), model="mock", strategy="cd_benchmark")
    rows = evaluate_pairs(matcher, [_pairs()[0]])
    metrics = calculate_metrics(rows)
    assert metrics.total_pairs == 1
    assert metrics.evaluated_pairs == 0
    assert metrics.failed_predictions == 1
