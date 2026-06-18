#!/usr/bin/env python3
"""Generate evaluation plots for model subdirectories."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plotting.eval_plots import generate_eval_plots


class _Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.reset = "\033[0m" if enabled else ""
        self.bold = "\033[1m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""
        self.cyan = "\033[36m" if enabled else ""
        self.green = "\033[32m" if enabled else ""
        self.magenta = "\033[35m" if enabled else ""
        self.yellow = "\033[33m" if enabled else ""


def _use_color(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


class _ProgressBar:
    def __init__(self, total: int, style: _Style) -> None:
        self.total = max(total, 1)
        self.style = style
        self.count = 0
        self.width = 28
        self.previous_len = 0
        self.interactive = sys.stdout.isatty()

    def __call__(self, message: str) -> None:
        self.count = min(self.count + 1, self.total)
        filled = round(self.width * self.count / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"{self.style.cyan}[{bar}]{self.style.reset} "
            f"{self.count:>2}/{self.total:<2} {message}"
        )
        if self.interactive:
            padding = " " * max(self.previous_len - len(line), 0)
            sys.stdout.write(f"\r{line}{padding}")
            sys.stdout.flush()
            self.previous_len = len(line)
            if self.count >= self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return
        print(line, flush=True)


def _count_model_dirs(eval_dir: Path) -> int:
    if not eval_dir.is_dir():
        return 0
    return sum(
        1
        for child in eval_dir.iterdir()
        if child.is_dir()
        and child.name != "eval_plots"
        and (child / "sequences.csv").exists()
        and (child / "scores.csv").exists()
    )


def _progress_total(eval_dir: Path) -> int:
    model_count = _count_model_dirs(eval_dir)
    return 3 * model_count + 14


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
        help="Only print the final output path.",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize CLI output. Default: auto.",
    )
    return parser


def main() -> None:
    args = _get_parser().parse_args()
    style = _Style(_use_color(args.color))
    progress = None
    if not args.quiet:
        print(f"{style.magenta}{style.bold}mRNA plots{style.reset}", flush=True)
        progress = _ProgressBar(_progress_total(args.eval_dir), style)

    result = generate_eval_plots(args.eval_dir, progress=progress)
    prefix = f"{style.green}{style.bold}Done.{style.reset}" if not args.quiet else "Done:"
    print(
        f"{prefix} Saved plots for {result.model_count} model(s), {result.row_count} row(s) "
        f"to {style.yellow}{result.output_dir}{style.reset}",
        flush=True,
    )


if __name__ == "__main__":
    main()
