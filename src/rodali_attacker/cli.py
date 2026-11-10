"""Command-line entry point for the original-record CD baseline experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .cd_dataset import load_cd_dataset, sample_pairs
from .evaluation import calculate_metrics, evaluate_pairs, save_evaluation
from .llm_matcher import LLMMatcher
from .providers import MockProvider, OpenAIProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the LLM matcher on original HPI CD Dataset 1 records."
    )
    parser.add_argument("--records", type=Path, default=Path("data/raw/cddb.tsv"))
    parser.add_argument("--dpl", type=Path, default=Path("data/raw/cddb_DPL.tsv"))
    parser.add_argument("--ndpl", type=Path, default=Path("data/raw/cddb_NDPL.tsv"))
    parser.add_argument("--output", type=Path, default=Path("results/cd_baseline.csv"))
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate N duplicate and N non-duplicate pairs (default: all pairs).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly enable paid OpenAI requests; otherwise use the offline mock.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positives, negatives = load_cd_dataset(args.records, args.dpl, args.ndpl)
    pairs = sample_pairs(
        positives, negatives, per_class=args.sample_per_class, seed=args.seed
    )
    if args.live:
        provider = OpenAIProvider(timeout=args.timeout, max_retries=args.max_retries)
        model = args.model
    else:
        # Dry-run behavior exercises the pipeline without pretending that a
        # heuristic approximates a real model's benchmark quality.
        provider = MockProvider(responder=lambda _prompt: "NON_MATCH")
        model = "offline-mock"
    matcher = LLMMatcher(
        provider, model=model, strategy="cd_benchmark", temperature=0.0
    )
    rows = evaluate_pairs(matcher, pairs)
    metrics = calculate_metrics(rows)
    csv_path, metrics_path = save_evaluation(rows, metrics, args.output)
    print(json.dumps(asdict(metrics), indent=2, sort_keys=True))
    print(f"Saved pair results to {csv_path}")
    print(f"Saved metrics to {metrics_path}")
    if not args.live:
        print("Offline mock only: no API requests were issued.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
