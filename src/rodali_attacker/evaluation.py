"""Baseline evaluation of labeled record pairs through the existing matcher."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .cd_dataset import CDPair
from .llm_matcher import LLMMatcher


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    pair_id: str
    record_id_a: str
    record_id_b: str
    ground_truth: int
    prediction: int | None
    prediction_text: str | None
    model: str
    prompt_version: str
    error: str | None


@dataclass(frozen=True, slots=True)
class Metrics:
    total_pairs: int
    evaluated_pairs: int
    failed_predictions: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate_pairs(matcher: LLMMatcher, pairs: list[CDPair]) -> list[EvaluationRow]:
    """Send pairs one at a time; labels never enter matcher records or prompts."""

    rows: list[EvaluationRow] = []
    for pair in pairs:
        result = matcher.match(pair.record_a, pair.record_b, pair_id=pair.pair_id)
        prediction = {"MATCH": 1, "NON_MATCH": 0}.get(result.prediction or "")
        rows.append(
            EvaluationRow(
                pair_id=pair.pair_id,
                record_id_a=pair.record_id_a,
                record_id_b=pair.record_id_b,
                ground_truth=pair.label,
                prediction=prediction,
                prediction_text=result.prediction,
                model=result.model,
                prompt_version=result.prompt_version,
                error=result.error,
            )
        )
    return rows


def calculate_metrics(rows: list[EvaluationRow]) -> Metrics:
    """Calculate binary metrics over successful predictions only."""

    evaluated = [row for row in rows if row.prediction is not None]
    tp = sum(row.ground_truth == 1 and row.prediction == 1 for row in evaluated)
    tn = sum(row.ground_truth == 0 and row.prediction == 0 for row in evaluated)
    fp = sum(row.ground_truth == 0 and row.prediction == 1 for row in evaluated)
    fn = sum(row.ground_truth == 1 and row.prediction == 0 for row in evaluated)
    count = len(evaluated)
    accuracy = (tp + tn) / count if count else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(
        total_pairs=len(rows),
        evaluated_pairs=count,
        failed_predictions=len(rows) - count,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def save_evaluation(
    rows: list[EvaluationRow], metrics: Metrics, output_path: str | Path
) -> tuple[Path, Path]:
    """Save pair-level CSV and a sibling JSON metrics summary."""

    destination = Path(output_path)
    if destination.suffix.lower() != ".csv":
        raise ValueError("Evaluation output must use a .csv suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvaluationRow.__dataclass_fields__)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    metrics_path = destination.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination, metrics_path
