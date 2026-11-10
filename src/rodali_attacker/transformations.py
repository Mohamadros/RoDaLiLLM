"""Conservative, reproducible adversarial product-record transformations."""

from __future__ import annotations

import copy
import random
import re
import string
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .models import (
    TransformationResult,
    TransformationStep,
    ValidationStatus,
)

DEFAULT_PROTECTED_FIELDS = frozenset(
    {
        "brand",
        "model_number",
        "gtin",
        "ean",
        "sku",
        "size",
        "color",
        "capacity",
        "configuration",
        "package_quantity",
    }
)

INTENSITY_BUDGETS = {"low": 1, "medium": 2, "high": 3}

# Numbers, common units, product-like alphanumeric identifiers, and known
# variant words are treated as critical when embedded in otherwise free text.
_CRITICAL_RE = re.compile(
    r"(?ix)\b(?:"
    r"\d+(?:[.,]\d+)?\s*(?:gb|tb|mb|kg|g|mg|ml|l|cm|mm|inch|in|eu|w|v|mah|hz)"
    r"|\d+"
    r"|(?=\w*[a-z])(?=\w*\d)[a-z0-9][a-z0-9._/-]*"
    r"|black|white|red|blue|green|silver|gold|schwarz|weiß|weiss|rot|blau|grün|gruen"
    r")\b"
)
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}")


def _critical_tokens(value: str) -> tuple[str, ...]:
    return tuple(sorted(match.group(0).casefold() for match in _CRITICAL_RE.finditer(value)))


def _changed_fields(before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key)))


class RecordTransformer:
    """Generate traceable record variants without silently changing identity.

    ``SAFE`` means only that explicit syntactic rules passed. It is not a proof
    of semantic equivalence. Character edits and omissions require human review.
    """

    def __init__(
        self,
        protected_fields: Iterable[str] = DEFAULT_PROTECTED_FIELDS,
        *,
        abbreviations: Mapping[str, str] | None = None,
        optional_fields: Iterable[str] = ("description",),
    ) -> None:
        self.protected_fields = frozenset(protected_fields)
        self.abbreviations = dict(abbreviations or {})
        self.optional_fields = frozenset(optional_fields)
        self._strategies: dict[
            str, Callable[[dict[str, Any], random.Random], tuple[dict[str, Any], dict[str, Any]]]
        ] = {
            "case_change": self._case_change,
            "whitespace_variation": self._whitespace_variation,
            "whitespace_normalization": self._whitespace_variation,
            "punctuation_change": self._punctuation_change,
            "phrase_reorder": self._phrase_reorder,
            "controlled_abbreviation": self._controlled_abbreviation,
            "language_variation": self._language_variation,
            "omit_optional": self._omit_optional,
            "char_delete": self._char_delete,
            "char_insert": self._char_insert,
            "char_transpose": self._char_transpose,
            "char_substitute": self._char_substitute,
            "repeated_char_removal": self._repeated_char_removal,
        }

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(self._strategies)

    def transform(
        self,
        record: dict[str, Any],
        strategy: str | Sequence[str],
        seed: int | None = None,
        *,
        intensity: str = "low",
    ) -> TransformationResult:
        """Apply one strategy or an ordered combination to a copied record."""

        if intensity not in INTENSITY_BUDGETS:
            raise ValueError(f"Unknown intensity {intensity!r}; use low, medium, or high")
        requested = [strategy] if isinstance(strategy, str) else list(strategy)
        if not requested:
            raise ValueError("At least one strategy is required")
        unknown = [name for name in requested if name not in self._strategies]
        if unknown:
            raise ValueError(f"Unknown strategy: {unknown[0]}")
        budget = INTENSITY_BUDGETS[intensity]
        if len(requested) > budget:
            raise ValueError(
                f"Transformation budget exceeded: {intensity} intensity permits "
                f"at most {budget} transformation(s)"
            )

        original = copy.deepcopy(record)
        current = copy.deepcopy(record)
        rng = random.Random(seed)
        history: list[TransformationStep] = []
        all_parameters: dict[str, Any] = {"intensity": intensity}
        for index, name in enumerate(requested):
            before = copy.deepcopy(current)
            current, parameters = self._strategies[name](current, rng)
            changed = _changed_fields(before, current)
            history.append(TransformationStep(name, parameters, changed))
            all_parameters[f"step_{index + 1}"] = parameters

        changed = _changed_fields(original, current)
        status, notes = self._validate(original, current, requested, changed)
        return TransformationResult(
            original_record=original,
            transformed_record=current,
            transformation_name="+".join(requested),
            parameters=all_parameters,
            changed_fields=changed,
            random_seed=seed,
            validation_status=status,
            validation_notes=notes,
            history=tuple(history),
        )

    def generate_candidates(
        self,
        record: dict[str, Any],
        strategies: list[str],
        number_of_candidates: int = 10,
        seed: int | None = None,
        *,
        intensity: str = "low",
    ) -> list[TransformationResult]:
        """Generate distinct candidates with derived, reproducible seeds."""

        if number_of_candidates < 0:
            raise ValueError("number_of_candidates must be non-negative")
        if not strategies and number_of_candidates:
            raise ValueError("At least one strategy is required")
        master = random.Random(seed)
        candidates: list[TransformationResult] = []
        seen: set[str] = set()
        attempts = max(number_of_candidates * 10, len(strategies))
        for index in range(attempts):
            if len(candidates) >= number_of_candidates:
                break
            derived_seed = master.randrange(0, 2**32)
            if intensity == "low":
                selected: str | Sequence[str] = strategies[index % len(strategies)]
            else:
                count = min(INTENSITY_BUDGETS[intensity], len(strategies))
                selected = master.sample(strategies, master.randint(1, count))
            result = self.transform(record, selected, derived_seed, intensity=intensity)
            canonical = repr(sorted(result.transformed_record.items()))
            if result.changed_fields and canonical not in seen:
                seen.add(canonical)
                candidates.append(result)
        return candidates

    def _mutable_text_fields(self, record: Mapping[str, Any]) -> list[str]:
        return [
            key
            for key, value in record.items()
            if key not in self.protected_fields and isinstance(value, str) and value
        ]

    def _pick_field(self, record: Mapping[str, Any], rng: random.Random) -> str | None:
        fields = self._mutable_text_fields(record)
        return rng.choice(fields) if fields else None

    def _validate(
        self,
        original: Mapping[str, Any],
        transformed: Mapping[str, Any],
        strategies: Sequence[str],
        changed: tuple[str, ...],
    ) -> tuple[ValidationStatus, tuple[str, ...]]:
        notes: list[str] = []
        protected_changes = sorted(set(changed) & self.protected_fields)
        if protected_changes:
            return (
                ValidationStatus.REJECTED,
                (f"Protected fields changed: {', '.join(protected_changes)}",),
            )
        for field in changed:
            before, after = original.get(field), transformed.get(field)
            if isinstance(before, str) and isinstance(after, str):
                if _critical_tokens(before) != _critical_tokens(after):
                    return (
                        ValidationStatus.REJECTED,
                        (f"Embedded identity-critical token changed in {field}",),
                    )
        ambiguous = {
            "char_delete",
            "char_insert",
            "char_transpose",
            "char_substitute",
            "repeated_char_removal",
            "phrase_reorder",
            "omit_optional",
        }
        if not changed:
            notes.append("No applicable change was found")
            return ValidationStatus.REJECTED, tuple(notes)
        if ambiguous.intersection(strategies):
            notes.append("Semantic equivalence requires human review")
            return ValidationStatus.REVIEW_REQUIRED, tuple(notes)
        notes.append("Protected fields and embedded critical tokens are unchanged")
        return ValidationStatus.SAFE, tuple(notes)

    def _case_change(self, record: dict[str, Any], rng: random.Random):
        field = self._pick_field(record, rng)
        if field:
            choices = [record[field].upper(), record[field].lower(), record[field].title()]
            alternatives = [value for value in choices if value != record[field]]
            if alternatives:
                record[field] = rng.choice(alternatives)
        return record, {"field": field, "mode": "case"}

    def _whitespace_variation(self, record: dict[str, Any], rng: random.Random):
        field = self._pick_field(record, rng)
        if field:
            normalized = " ".join(record[field].split())
            record[field] = normalized.replace(" ", "  ", 1) if " " in normalized else normalized
        return record, {"field": field}

    def _punctuation_change(self, record: dict[str, Any], rng: random.Random):
        field = self._pick_field(record, rng)
        if field:
            value = record[field]
            match = re.search(r"[,;:]", value)
            if match:
                record[field] = value[: match.start()] + " -" + value[match.end() :]
            else:
                words = value.split(maxsplit=1)
                if len(words) == 2:
                    record[field] = f"{words[0]}: {words[1]}"
        return record, {"field": field}

    def _phrase_reorder(self, record: dict[str, Any], rng: random.Random):
        field = self._pick_field(record, rng)
        if field:
            parts = [part.strip() for part in re.split(r"[,;]", record[field]) if part.strip()]
            if len(parts) > 1:
                parts = parts[1:] + parts[:1]
                record[field] = ", ".join(parts)
        return record, {"field": field}

    def _controlled_abbreviation(self, record: dict[str, Any], rng: random.Random):
        applicable: list[tuple[str, str, str]] = []
        for field in self._mutable_text_fields(record):
            for source, replacement in self.abbreviations.items():
                if re.search(rf"\b{re.escape(source)}\b", record[field], re.IGNORECASE):
                    applicable.append((field, source, replacement))
        if not applicable:
            return record, {"field": None, "evidence": "no configured alias"}
        field, source, replacement = rng.choice(applicable)
        record[field] = re.sub(
            rf"\b{re.escape(source)}\b", replacement, record[field], count=1, flags=re.IGNORECASE
        )
        return record, {"field": field, "source": source, "replacement": replacement}

    def _language_variation(self, record: dict[str, Any], rng: random.Random):
        # Deliberately limited to meaning-preserving, non-variant nouns/adjectives.
        translations = {
            "sneaker": "Turnschuh",
            "shoe": "Schuh",
            "with": "mit",
            "new": "neu",
            "Turnschuh": "sneaker",
            "Schuh": "shoe",
            "mit": "with",
            "neu": "new",
        }
        applicable: list[tuple[str, str, str]] = []
        for field in self._mutable_text_fields(record):
            for source, replacement in translations.items():
                if re.search(rf"\b{re.escape(source)}\b", record[field], re.IGNORECASE):
                    applicable.append((field, source, replacement))
        if not applicable:
            return record, {"field": None}
        field, source, replacement = rng.choice(applicable)
        record[field] = re.sub(
            rf"\b{re.escape(source)}\b", replacement, record[field], count=1, flags=re.IGNORECASE
        )
        return record, {"field": field, "source": source, "replacement": replacement}

    def _omit_optional(self, record: dict[str, Any], rng: random.Random):
        fields = [field for field in self.optional_fields if record.get(field)]
        field = rng.choice(fields) if fields else None
        if field:
            record[field] = ""
        return record, {"field": field}

    def _editable_word(self, record: Mapping[str, Any], rng: random.Random):
        candidates: list[tuple[str, re.Match[str]]] = []
        for field in self._mutable_text_fields(record):
            critical_spans = [match.span() for match in _CRITICAL_RE.finditer(record[field])]
            for match in _WORD_RE.finditer(record[field]):
                if not any(match.start() < end and match.end() > start for start, end in critical_spans):
                    candidates.append((field, match))
        return rng.choice(candidates) if candidates else (None, None)

    def _replace_word(self, record, rng, operation):
        field, match = self._editable_word(record, rng)
        if field is None or match is None:
            return record, {"field": None}
        word = match.group(0)
        changed, details = operation(word, rng)
        record[field] = record[field][: match.start()] + changed + record[field][match.end() :]
        return record, {"field": field, "original": word, "replacement": changed, **details}

    def _char_delete(self, record, rng):
        def op(word, randomizer):
            index = randomizer.randrange(len(word))
            return word[:index] + word[index + 1 :], {"index": index}

        return self._replace_word(record, rng, op)

    def _char_insert(self, record, rng):
        def op(word, randomizer):
            index = randomizer.randrange(len(word) + 1)
            character = randomizer.choice(string.ascii_lowercase)
            return word[:index] + character + word[index:], {"index": index, "character": character}

        return self._replace_word(record, rng, op)

    def _char_transpose(self, record, rng):
        def op(word, randomizer):
            index = randomizer.randrange(len(word) - 1)
            changed = word[:index] + word[index + 1] + word[index] + word[index + 2 :]
            return changed, {"index": index}

        return self._replace_word(record, rng, op)

    def _char_substitute(self, record, rng):
        def op(word, randomizer):
            index = randomizer.randrange(len(word))
            choices = string.ascii_lowercase.replace(word[index].lower(), "")
            character = randomizer.choice(choices)
            return word[:index] + character + word[index + 1 :], {"index": index, "character": character}

        return self._replace_word(record, rng, op)

    def _repeated_char_removal(self, record, rng):
        applicable: list[tuple[str, re.Match[str]]] = []
        for field in self._mutable_text_fields(record):
            applicable.extend((field, match) for match in re.finditer(r"([A-Za-z])\1+", record[field], re.I))
        if not applicable:
            return record, {"field": None}
        field, match = rng.choice(applicable)
        value = record[field]
        record[field] = value[: match.start()] + match.group(1) + value[match.end() :]
        return record, {"field": field, "sequence": match.group(0)}
