from __future__ import annotations

import csv
import json

import pytest

from rodali_attacker import LLMMatcher, MockProvider
from rodali_attacker.llm_matcher import parse_prediction
from rodali_attacker.prompts import build_prompt
from rodali_attacker.providers import ProviderError


def test_rule_prompt_is_stable_and_contains_missing_and_empty_attributes():
    prompt, version = build_prompt({"name": "Shoe", "description": ""}, {"name": "Schuh"}, "rule_guided")
    assert version == "rule-guided-v1"
    assert '"description": ""' in prompt
    assert "Different sizes, colors, capacities" in prompt
    assert prompt.index('"description"') < prompt.index('"name"')


@pytest.mark.parametrize(
    ("response", "expected"),
    [("MATCH", "MATCH"), (" non-match\n", "NON_MATCH"), ("NON_MATCH", "NON_MATCH")],
)
def test_parse_prediction(response, expected):
    assert parse_prediction(response) == expected


@pytest.mark.parametrize("response", ["yes", "MATCH because...", "", "maybe MATCH"])
def test_parse_prediction_rejects_unexpected_response(response):
    with pytest.raises(ValueError):
        parse_prediction(response)


def test_match_returns_structured_parse_error():
    matcher = LLMMatcher(MockProvider(["probably"]), model="mock")
    result = matcher.match({"name": "A"}, {"name": "B"})
    assert not result.succeeded
    assert result.prediction is None
    assert result.error == "Invalid model response: Expected exactly MATCH or NON_MATCH"


def test_api_failure_is_captured_without_secret_details():
    matcher = LLMMatcher(MockProvider(error=ProviderError("token=secret")), model="mock")
    result = matcher.match({}, {})
    assert result.error == "Provider request failed: ProviderError"
    assert "secret" not in result.error


def test_batch_processing_and_stable_ids():
    matcher = LLMMatcher(MockProvider(["MATCH", "NON_MATCH"]), model="mock")
    results = matcher.match_batch([("given", {"x": 1}, {"x": 1}), ({"x": 1}, {"x": 2})])
    assert [result.prediction for result in results] == ["MATCH", "NON_MATCH"]
    assert results[0].pair_id == "given"
    again = LLMMatcher(MockProvider(["MATCH"]), model="mock").match({"x": 1}, {"x": 2})
    assert again.pair_id == results[1].pair_id


def test_save_jsonl_and_csv(tmp_path):
    matcher = LLMMatcher(MockProvider(["MATCH"]), model="mock")
    result = matcher.match({}, {})
    jsonl = matcher.save_results([result], tmp_path / "nested" / "out.jsonl")
    csv_path = matcher.save_results([result], tmp_path / "out.csv")
    assert json.loads(jsonl.read_text().strip())["prediction"] == "MATCH"
    with csv_path.open(newline="") as handle:
        assert next(csv.DictReader(handle))["prediction"] == "MATCH"


def test_simple_prompt_and_mock_capture():
    provider = MockProvider(["MATCH"])
    matcher = LLMMatcher(provider, model="mock", strategy="simple")
    matcher.match({"name": "A"}, {"name": "B"})
    assert "Return exactly MATCH or NON_MATCH" in provider.prompts[0]
