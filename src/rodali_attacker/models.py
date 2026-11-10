"""Shared data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ValidationStatus(StrEnum):
    """Conservative assessment of a transformation."""

    SAFE = "SAFE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class MatchPrediction:
    """Structured result of one LLM matching request."""

    pair_id: str
    prediction: str | None
    model: str
    prompt_version: str
    record_a_id: str
    record_b_id: str
    raw_response: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.prediction in {"MATCH", "NON_MATCH"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TransformationStep:
    """One operation in a transformation history."""

    name: str
    parameters: dict[str, Any]
    changed_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransformationResult:
    """A transformed record plus traceability and validation metadata."""

    original_record: dict[str, Any]
    transformed_record: dict[str, Any]
    transformation_name: str
    parameters: dict[str, Any]
    changed_fields: tuple[str, ...]
    random_seed: int | None
    validation_status: ValidationStatus
    validation_notes: tuple[str, ...] = ()
    history: tuple[TransformationStep, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["validation_status"] = self.validation_status.value
        return data

