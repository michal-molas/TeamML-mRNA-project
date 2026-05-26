import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def gt_vs_generated_distribution(
    stats: dict[str, dict[str, pd.DataFrame]],
    save_dir: str,
    config: dict,
) -> None:
    gt_metrics = stats["gt"]["cds"]
    generated_metrics = stats["generated"]["cds"]

    metrics = config["metrics"]  # List of metric columns to compare

    # if metrics is empty, default to all columns except id and sample
    if not metrics:
        metrics = [col for col in gt_metrics.columns if col not in ["id", "sample"] and "mean" in col]

    plot_dir = f"{save_dir}/gt_vs_generated_distribution"
    os.makedirs(plot_dir, exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.kdeplot(gt_metrics[metric], label="Ground Truth", fill=True)
        sns.kdeplot(generated_metrics[metric], label="Generated", fill=True)
        plt.title(f"Distribution of {metric} for GT vs Generated")
        plt.legend()
        plt.tight_layout()

        save_path = f"{plot_dir}/{metric}.png"
        plt.savefig(save_path)
        plt.close()