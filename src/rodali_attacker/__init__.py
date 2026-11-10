"""Public API for the RoDaLi LLM attacker prototype."""

from .cd_dataset import CDPair, DatasetFormatError, load_cd_dataset, sample_pairs
from .evaluation import Metrics, calculate_metrics, evaluate_pairs, save_evaluation
from .llm_matcher import LLMMatcher
from .models import (
    MatchPrediction,
    TransformationResult,
    ValidationStatus,
)
from .providers import MockProvider, OpenAIProvider
from .transformations import RecordTransformer

__all__ = [
    "CDPair",
    "DatasetFormatError",
    "LLMMatcher",
    "MatchPrediction",
    "MockProvider",
    "OpenAIProvider",
    "RecordTransformer",
    "TransformationResult",
    "ValidationStatus",
    "Metrics",
    "calculate_metrics",
    "evaluate_pairs",
    "load_cd_dataset",
    "sample_pairs",
    "save_evaluation",
]

