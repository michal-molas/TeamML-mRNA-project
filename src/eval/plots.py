
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def feature_distribution(
    eval_results: dict[str, pd.DataFrame],
    config: dict,
    save_path: str,
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
    plt.savefig(save_path)
    plt.close()