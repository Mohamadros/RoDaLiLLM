"""High-level product matching, batch execution, and result persistence."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .models import MatchPrediction
from .prompts import build_prompt
from .providers import LLMProvider

LOGGER = logging.getLogger(__name__)


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{prefix}_{digest}"


def parse_prediction(response: str) -> str:
    """Parse the deliberately constrained output; reject explanatory prose."""

    normalized = response.strip().upper().replace("-", "_")
    if normalized in {"MATCH", "NON_MATCH"}:
        return normalized
    raise ValueError("Expected exactly MATCH or NON_MATCH")


class LLMMatcher:
    """Ask a configured LLM provider for exact-product match decisions."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str,
        strategy: str = "rule_guided",
        temperature: float | None = 0.0,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self.provider = provider
        self.model = model
        self.strategy = strategy
        self.temperature = temperature

    def match(
        self,
        record_a: Mapping[str, Any],
        record_b: Mapping[str, Any],
        *,
        pair_id: str | None = None,
    ) -> MatchPrediction:
        prompt, version = build_prompt(record_a, record_b, self.strategy)
        record_a_id = _stable_id("record", record_a)
        record_b_id = _stable_id("record", record_b)
        resolved_pair_id = pair_id or _stable_id(
            "pair", {"record_a_id": record_a_id, "record_b_id": record_b_id}
        )
        try:
            raw = self.provider.complete(
                prompt, model=self.model, temperature=self.temperature
            )
            prediction = parse_prediction(raw)
            return MatchPrediction(
                resolved_pair_id,
                prediction,
                self.model,
                version,
                record_a_id,
                record_b_id,
                raw_response=raw,
            )
        except ValueError as exc:
            LOGGER.warning("Matching failed for pair %s: %s", resolved_pair_id, type(exc).__name__)
            return MatchPrediction(
                resolved_pair_id,
                None,
                self.model,
                version,
                record_a_id,
                record_b_id,
                raw_response=locals().get("raw"),
                error=f"Invalid model response: {exc}",
            )
        except Exception as exc:
            # Provider messages can contain request data or sensitive details.
            LOGGER.warning("Provider request failed for pair %s: %s", resolved_pair_id, type(exc).__name__)
            return MatchPrediction(
                resolved_pair_id,
                None,
                self.model,
                version,
                record_a_id,
                record_b_id,
                error=f"Provider request failed: {type(exc).__name__}",
            )

    def match_batch(
        self,
        pairs: Iterable[
            tuple[Mapping[str, Any], Mapping[str, Any]]
            | tuple[str, Mapping[str, Any], Mapping[str, Any]]
        ],
        *,
        max_workers: int = 1,
    ) -> list[MatchPrediction]:
        """Match pairs in input order with bounded concurrency."""

        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        def run(pair: Sequence[object]) -> MatchPrediction:
            if len(pair) == 2:
                return self.match(pair[0], pair[1])  # type: ignore[arg-type]
            if len(pair) == 3:
                return self.match(pair[1], pair[2], pair_id=str(pair[0]))  # type: ignore[arg-type]
            raise ValueError("Each pair must contain (a, b) or (pair_id, a, b)")

        materialized = list(pairs)
        if max_workers == 1:
            return [run(pair) for pair in materialized]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(run, materialized))

    @staticmethod
    def save_results(results: Iterable[MatchPrediction], path: str | Path) -> Path:
        """Save predictions as JSONL or CSV according to the filename suffix."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = [result.to_dict() for result in results]
        if destination.suffix.lower() == ".jsonl":
            with destination.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif destination.suffix.lower() == ".csv":
            fieldnames = list(MatchPrediction.__dataclass_fields__)
            with destination.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            raise ValueError("Output path must end in .jsonl or .csv")
        return destination
