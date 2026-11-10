"""Versioned prompt construction, separate from model transport."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

PROMPT_VERSIONS = {
    "simple": "simple-v1",
    "rule_guided": "rule-guided-v1",
}

_SIMPLE = """Decide whether Product A and Product B describe the exact same real-world product.
Return exactly MATCH or NON_MATCH. Do not add any other text."""

_RULE_GUIDED = """Decide whether Product A and Product B describe the exact same real-world product.

Rules:
- Different sizes, colors, capacities, models, configurations, or package quantities are not matches.
- Spelling, capitalization, formatting, abbreviations, or word order may differ without changing identity.
- Missing attributes alone are not proof of a mismatch.
- Conflicting identifying attributes are evidence of a non-match.
- A match requires positive evidence of equivalence.

Return exactly MATCH or NON_MATCH. Do not add any other text."""


def serialize_record(record: Mapping[str, Any]) -> str:
    """Serialize attributes predictably for prompts and auditability."""

    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def build_prompt(
    record_a: Mapping[str, Any], record_b: Mapping[str, Any], strategy: str
) -> tuple[str, str]:
    """Return the prompt and its stable version identifier."""

    if strategy not in PROMPT_VERSIONS:
        choices = ", ".join(sorted(PROMPT_VERSIONS))
        raise ValueError(f"Unknown prompt strategy {strategy!r}; choose from {choices}")
    instructions = _SIMPLE if strategy == "simple" else _RULE_GUIDED
    prompt = (
        f"{instructions}\n\nProduct A:\n{serialize_record(record_a)}"
        f"\n\nProduct B:\n{serialize_record(record_b)}"
    )
    return prompt, PROMPT_VERSIONS[strategy]

