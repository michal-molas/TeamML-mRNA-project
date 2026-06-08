import argparse
from itertools import product
import os
from typing import Callable
import yaml

import pandas as pd

import plots


# Plot functions should take a DataFrame, save path, and optional config
PLOT_FUNCTIONS: dict[str, Callable[[dict[str, dict[str, pd.DataFrame]], str, dict], None]] = {
    "gt_vs_generated_distribution": plots.gt_vs_generated_distribution,
}


def calculate_global_stats(scores_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in scores_df.columns if col not in ["id", "sample"]]

    stats_df = scores_df[feature_cols].agg(["mean", "min", "max"])
    return stats_df.transpose().rename(columns={"index": "feature"})


def calculate_cds_stats(scores_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in scores_df.columns if col not in ["id", "sample"]]

    stats_df = (
        scores_df
        .groupby("id")[feature_cols]
        .agg(["mean", "min", "max"])
    )

    # flatten multi-index columns
    stats_df.columns = [
        f"{feature}_{stat}"
        for feature, stat in stats_df.columns
    ]

    return stats_df.reset_index()


def make_plots(
    stats: dict[str, dict[str, pd.DataFrame]],
    save_dir: str,
    config: dict,
) -> None:
    # Create subdirectory for plots
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Generate plots based on config
    for plot_name, plot_config in config.items():
        plot_func = PLOT_FUNCTIONS.get(plot_name)
        if plot_func is None:
            print(f"Unknown plot type: {plot_name}. Skipping.")
            continue
        print(f"Generating plot: {plot_name}...")
        plot_func(stats, plots_dir, plot_config)


def _save_stats(
    stats: dict[str, dict[str, pd.DataFrame]],
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    for sample_type, stats_type in product(
        ["gt", "generated"],
        ["cds", "global"],
    ):
        stats_path = os.path.join(save_dir, f"{sample_type}_{stats_type}_stats.csv")
        stats[sample_type][stats_type].to_csv(stats_path)
        print(f"Saved {sample_type} {stats_type} stats to {stats_path}")


def _load_scores(scores_csv: str) -> pd.DataFrame:
    return pd.read_csv(scores_csv)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate statistics for mRNA sequences.")
    parser.add_argument("--config", type=str)
    parser.add_argument("--scores_csv", type=str)
    parser.add_argument("--output_dir", type=str)
    return parser


def main() -> None:
    parser = _get_parser()
    args = parser.parse_args()

    config = _load_config(args.config)
    scores = _load_scores(args.scores_csv)

    gt_scores = scores[scores["sample"] == "gt"]
    generated_scores = scores[scores["sample"] != "gt"]

    stats: dict[str, dict[str, pd.DataFrame]] = {"gt": {}, "generated": {}}

    # Calculate stats aggregated by starting CDS
    stats["gt"]["cds"] = calculate_cds_stats(gt_scores)
    stats["generated"]["cds"] = calculate_cds_stats(generated_scores)

    # Calculate global stats
    stats["gt"]["global"] = calculate_global_stats(gt_scores)
    stats["generated"]["global"] = calculate_global_stats(generated_scores)

    # Save results
    output_dir = config.get("eval_dir") or args.output_dir
    if output_dir is None:
        raise ValueError("No output directory specified in config or arguments.")

    _save_stats(stats, output_dir)

    # Create and save plots
    if "plots" in config:
        make_plots(stats, args.output_dir, config["plots"])


if __name__ == "__main__":
    main()