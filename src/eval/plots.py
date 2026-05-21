
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def _gt_vs_generated_distribution(
    gt_metrics: pd.DataFrame,
    generated_metrics: pd.DataFrame,
    metric: str,
    save_dir: str,
) -> None:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(gt_metrics[metric], label="Ground Truth", fill=True)
    sns.kdeplot(generated_metrics[metric], label="Generated", fill=True)
    plt.title(f"Distribution of {metric} for GT vs Generated")
    plt.legend()
    plt.tight_layout()

    save_path = f"{save_dir}/gt_vs_generated_{metric}.png"
    plt.savefig(save_path)
    plt.close()


def gt_vs_generated_distribution(
    eval_results: dict[str, pd.DataFrame | dict[str, pd.DataFrame]],
    config: dict,
    save_dir: str,
) -> None:
    gt_metrics = eval_results["cds_metrics"]["gt"]
    generated_metrics = eval_results["cds_metrics"]["generated"]

    metrics = config["metrics"]  # List of metric columns to compare

    for metric in metrics:
        _gt_vs_generated_distribution(gt_metrics, generated_metrics, metric, save_dir)


def feature_distribution(
    eval_results: dict[str, pd.DataFrame | dict[str, pd.DataFrame]],
    config: dict,
    save_dir: str,
) -> None:
    table = config.get("table", "features")  # Default to "features" table if not specified
    df = eval_results.get(table)
    if df is None:
        print(f"Table {table} not found in eval results, cannot generate plot.")
        return
    feature = config.get("feature", "cds_length")  # Default to "cds_length" if not specified

    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    save_path = f"{save_dir}/feature_distribution_{feature}.png"
    plt.savefig(save_path)
    plt.close()