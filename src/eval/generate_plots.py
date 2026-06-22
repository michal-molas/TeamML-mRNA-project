#!/usr/bin/env python3
"""Generate evaluation plots for model subdirectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plotting.eval_plots import generate_eval_plots


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate distribution, correlation, and TE-improvement plots for an "
            "evaluation directory containing model subdirectories."
        )
    )
    parser.add_argument(
        "eval_dir",
        type=Path,
        help="Directory containing one subdirectory per model, each with sequences.csv and scores.csv.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    return parser


def main() -> None:
    args = _get_parser().parse_args()
    result = generate_eval_plots(args.eval_dir, progress=None if args.quiet else print)
    print(
        f"Saved plots for {result.model_count} model(s), {result.row_count} row(s) "
        f"to {result.output_dir}"
    )


if __name__ == "__main__":
    main()
