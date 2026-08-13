"""Path-agnostic command line interface for supplementary evaluators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .climatenet_portability import evaluate_climatenet_portability
from .matched_scoring import evaluate_matched_scoring
from .physical_metric_ablation import evaluate_physical_metric_ablation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a reviewer-requested development control from an external JSON data contract."
    )
    parser.add_argument(
        "study",
        choices=("climatenet-portability", "matched-scoring", "physical-metric-ablation"),
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSON following supplementary/README.md")
    parser.add_argument("--output", type=Path, required=True, help="Destination for aggregate JSON")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20262202)
    parser.add_argument("--backend", default="acorn", help="Backend key for matched-scoring")
    parser.add_argument("--full-system", default="cem", help="Full CEM candidate-pool key for matched-scoring")
    return parser


def evaluate(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    if args.study == "climatenet-portability":
        return evaluate_climatenet_portability(payload, resamples=args.resamples, seed=args.seed)
    if args.study == "matched-scoring":
        return evaluate_matched_scoring(
            payload,
            backend=args.backend,
            full_system=args.full_system,
            resamples=args.resamples,
            seed=args.seed,
        )
    return evaluate_physical_metric_ablation(payload, resamples=args.resamples, seed=args.seed)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = evaluate(args, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
