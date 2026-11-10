# RoDaLi LLM Matcher and Record Transformations

This repository is a first-iteration prototype for the RoDaLi Attacker C seminar
project. It has two reusable components:

- `LLMMatcher` builds versioned exact-product prompts, calls an interchangeable
  provider, strictly parses `MATCH`/`NON_MATCH`, processes batches, and writes
  CSV or JSONL results.
- `RecordTransformer` creates reproducible product-record variants while keeping
  configured fields immutable and recording validation status and history.

The code requires Python 3.11 or newer. Tests and the example are fully offline.
No experimental performance claims are made.

## Installation

On macOS or Linux, from the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Live LLM configuration

Install the optional SDK and put the key in your shell environment or copy the
safe template to an uncommitted `.env` file:

```bash
python -m pip install -e '.[openai]'
export OPENAI_API_KEY='your-key-here'
# Alternatively: cp .env.example .env, then edit .env locally.
```

```python
from rodali_attacker import LLMMatcher, OpenAIProvider

matcher = LLMMatcher(
    OpenAIProvider(timeout=30, max_retries=2),
    model="gpt-4.1-mini",
    strategy="rule_guided",  # or "simple"
)
result = matcher.match(record_a, record_b, pair_id="pair_001")
print(result.prediction, result.error)
```

The SDK loads `OPENAI_API_KEY` from the environment. Provider errors are returned
without their original message so credentials or request contents are not copied
into logs. Temperature `0` is requested by default; pass `temperature=None` for
models that do not support it. API/model behavior ultimately determines
reproducibility.

## Transformations

```python
from rodali_attacker import RecordTransformer

transformer = RecordTransformer(
    protected_fields={"brand", "model_number", "gtin", "sku", "size", "color",
                      "capacity", "configuration", "package_quantity"},
    abbreviations={"Air Force 1": "AF1"},  # evidence-backed project mapping
)
result = transformer.transform(record_a, "case_change", seed=42)
candidates = transformer.generate_candidates(
    record_a,
    ["case_change", "punctuation_change", "char_transpose"],
    number_of_candidates=5,
    seed=42,
)
```

Strategies cover character deletion/insertion/transposition/substitution,
repeated-character removal, case, whitespace, punctuation, phrase reordering,
controlled abbreviations, a small English/German mapping, and optional-field
omission. Low, medium, and high intensity permit at most one, two, and three
sequential steps respectively.

Every result includes the original and transformed record, changed fields,
parameters, seed, history, validation notes, and one of:

- `SAFE`: explicit syntactic protection rules passed.
- `REVIEW_REQUIRED`: plausible but semantic equivalence needs a person.
- `REJECTED`: a protected/embedded critical value changed or no valid edit existed.

These labels are workflow aids, not proof of semantic correctness. Review all
research data, especially product-specific aliases and free-text transformations.

## Offline demo and tests

```bash
python examples/end_to_end.py
pytest
```

The demo obtains a mock baseline, generates variants, filters rejected candidates,
matches the remaining records, prints a comparison, and writes
`results/demo_predictions.jsonl`. It demonstrates integration only and does not
test real LLM behavior.

Use `matcher.save_results(results, "results/predictions.jsonl")` (or `.csv`) for
direct prediction export. Stable IDs are hashes of complete canonicalized records.

## Known limitations

- Identity protection is conservative lexical validation, not semantic proof.
- Critical-token recognition covers common numbers, units, alphanumeric IDs, and
  color terms; domain-specific variants need explicit fields and human review.
- Abbreviation aliases are caller-supplied because they are domain dependent.
- Concurrent batches assume the provider client supports concurrent calls; use
  the default `max_workers=1` when uncertain.
- Live-provider behavior and credentials are not exercised by the offline suite.
