import argparse
from dataclasses import dataclass
import os
from datetime import datetime
from functools import reduce
from typing import Iterable

import pandas as pd
import yaml
import numpy as np

from eval import FeatureExtractor, ScoringModel, Aggregator, EvalConfig
from schemas import SAMPLE_CSV_COLS, SAMPLE_INDEX_COLS

from feature_extractors import (
    StringStatisticsExtractor,
)
from scoring_models import (
    RiboNN,
    RNAfold,
    UTRLM,
)
from aggregators import (
    SimpleAverageAggregator,
)
from plots import (
    feature_distribution,
)


EVAL_DIR = "data/evals"

FEATURE_EXTRACTORS = {
    "StringStatistics": StringStatisticsExtractor
}

SCORING_MODELS = {
    "RiboNN": RiboNN,
    "RNAfold": RNAfold,
    "UTRLM": UTRLM,
}

AGGREGATORS = {
    "SimpleAverage": SimpleAverageAggregator
}

PLOTS = {
    "feature_distribution": feature_distribution
}


def load_scoring_model(config: dict) -> ScoringModel:
    scoring_model = SCORING_MODELS.get(config["type"])
    if scoring_model is None:
        raise ValueError(f"Unknown scoring model type: {config['type']}")
    kwargs = {key: value for key, value in config.items() if key != "type"}
    return scoring_model(**kwargs)


def load_feature_extractor(config: dict) -> FeatureExtractor:
    feature_extractor = FEATURE_EXTRACTORS.get(config["type"])
    if feature_extractor is None:
        raise ValueError(f"Unknown feature extractor type: {config['type']}")
    return feature_extractor()


def load_aggregator(config: dict) -> Aggregator:
    aggregator = AGGREGATORS.get(config["type"])
    if aggregator is None:
        raise ValueError(f"Unknown aggregator type: {config['type']}")
    return aggregator()


def validate_samples_schema(samples: pd.DataFrame) -> None:
    missing_cols = [col for col in SAMPLE_CSV_COLS if col not in samples.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in samples DataFrame: {missing_cols}")


def merge_feature_tables(feature_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # join all feature tables on SAMPLE_INDEX_COLS
    feature_tables_list = list(feature_tables.values())
    return reduce(lambda left, right: pd.merge(left, right, on=SAMPLE_INDEX_COLS), feature_tables_list)


def merge_oracle_tables(oracle_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # join all oracle tables on SAMPLE_INDEX_COLS
    orcale_tables_list = list(oracle_tables.values())
    return reduce(lambda left, right: pd.merge(left, right, on=SAMPLE_INDEX_COLS), orcale_tables_list)


def merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Unified function to merge any tables on SAMPLE_INDEX_COLS"""
    tables_list = list(tables.values())
    return reduce(lambda left, right: pd.merge(left, right, on=SAMPLE_INDEX_COLS), tables_list)


def merge_cds_metrics(metrics: dict[str, dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """
    input dict:
    {
        "aggregator_name": {
            "gt": pd.DataFrame,
            "generated": pd.DataFrame,
        },
        ...
    }
    output dict:
    {
        "gt": pd.DataFrame,
        "generated": pd.DataFrame,
    }
    """
    merged_metrics = {}
    for group in ["gt", "generated"]:
        group_tables = [metrics[agg_name][group] for agg_name in metrics]
        merged_metrics[group] = reduce(lambda left, right: pd.merge(left, right, on=SAMPLE_INDEX_COLS), group_tables)
    return merged_metrics





def load_config(config_path: str) -> EvalConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    print(f"Loaded config: {config_dict}")
    return EvalConfig(**config_dict)


def save_results(
    eval_results: dict[str, pd.DataFrame | dict[str, pd.DataFrame]],
    save_dir: str,
    plots: list[dict],
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    eval_results["samples"].to_csv(f"{save_dir}/samples.csv", index=False)
    print(f"Saved samples to {save_dir}/samples.csv")

    eval_results["features"].to_csv(f"{save_dir}/features.csv", index=False)
    print(f"Saved features to {save_dir}/features.csv")

    for agg_name, metrics in eval_results["cds_metrics"].items():
        metrics.to_csv(f"{save_dir}/cds_metrics_{agg_name}.csv", index=False)
        print(f"Saved CDS metrics for {agg_name} to {save_dir}/cds_metrics_{agg_name}.csv")

    for agg_name, metrics in eval_results["global_metrics"].items():
        metrics.to_csv(f"{save_dir}/global_metrics_{agg_name}.csv", index=False)
        print(f"Saved global metrics for {agg_name} to {save_dir}/global_metrics_{agg_name}.csv")

    if plots is not None:
        plot_dir = f"{save_dir}/plots"
        os.makedirs(plot_dir, exist_ok=True)

        for plot_config in plots:
            name = plot_config["type"]
            plot_func = PLOTS.get(name)
            if plot_func is None:
                print(f"Unknown plot type: {name}, skipping...")
                continue
            print(f"Generating plot: {name}...")
            save_path = f"{plot_dir}/{name}.png"
            plot_func(eval_results, plot_config, save_path)
            print(f"Saved plot {name} to {save_path}")


def preprocess_samples_df(samples: pd.DataFrame) -> pd.DataFrame:
    # TODO: extract this to generation
    samples.fillna("", inplace=True)
    samples["generated"] = samples["sample"] != "gt"
    return samples


def calculate_col_metrics(scores_grouped, col: str) -> dict[str, np.ndarray]:
    """Calculate metrics for given column"""
    metrics = {
        "id": scores_grouped[col].mean().index,  # use index from any metric
        "mean": scores_grouped[col].mean().values,
        "std": scores_grouped[col].std().values,
        "min": scores_grouped[col].min().values,
        "max": scores_grouped[col].max().values,
    }
    return metrics



def calculate_cds_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Return 
    """
    # group by 'id'
    data_cols = [col for col in scores.columns if col not in SAMPLE_INDEX_COLS]
    scores_grouped = scores.groupby("id")

    # col -> metric -> np.ndarray
    col_metrics: dict[str, dict[str, np.ndarray]] = {}
    for col in data_cols:
        col_metrics[col] = calculate_col_metrics(scores_grouped, col)

    # convert to DataFrame
    metrics_df = pd.DataFrame({
        "id": col_metrics[data_cols[0]]["id"],  # use id from first column
    })
    print("metrics_df after adding id:")
    print(metrics_df.head())
    for col in data_cols:
        metrics_df[f"{col}_mean"] = col_metrics[col]["mean"]
        metrics_df[f"{col}_std"] = col_metrics[col]["std"]
        metrics_df[f"{col}_min"] = col_metrics[col]["min"]
        metrics_df[f"{col}_max"] = col_metrics[col]["max"]
    return metrics_df


def calculate_global_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    data_cols = [col for col in scores.columns if col not in SAMPLE_INDEX_COLS]
    metrics = {}
    for col in data_cols:
        metrics[f"{col}_mean"] = scores[col].mean()
        metrics[f"{col}_std"] = scores[col].std()
        metrics[f"{col}_min"] = scores[col].min()
        metrics[f"{col}_max"] = scores[col].max()
    return pd.DataFrame([metrics])


def run_evaluation(
    samples: pd.DataFrame,
    feature_extractors: Iterable[FeatureExtractor],
    scoring_models: Iterable[ScoringModel],
    # cds_aggregators: Iterable[Aggregator],
    # global_aggregators: Iterable[Aggregator],
    config: EvalConfig,
) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:

    validate_samples_schema(samples)
    samples = preprocess_samples_df(samples)

    feature_tables = {}
    scoring_model_tables = {}

    for extractor in feature_extractors:
        print(f"Extracting features using {extractor.__class__.__name__}...")
        feature_tables[extractor.__class__.__name__] = extractor.extract_features(samples)

    for scoring_model in scoring_models:
        print(f"Scoring samples using {scoring_model.__class__.__name__}...")
        scoring_model_tables[scoring_model.__class__.__name__] = scoring_model.score(samples)

    print("FEATURE TABLES:")
    print(feature_tables)

    # features = merge_feature_tables(feature_tables)
    # oracle_scores = merge_oracle_tables(scoring_model_tables)

    features = merge_tables(feature_tables | scoring_model_tables)
    features['generated'] = samples['generated']  # add generated column to features for later analysis

    print("Merged features:")
    print(features)

    gt_features = features[features["generated"] == False]
    generated_features = features[features["generated"] == True]

    gt_cds_metrics = calculate_cds_metrics(gt_features)
    generated_cds_metrics = calculate_cds_metrics(generated_features)

    gt_global_metrics = calculate_global_metrics(gt_features)
    generated_global_metrics = calculate_global_metrics(generated_features)

    print("Generated CDS Metrics:")
    print(generated_cds_metrics)

    return {
        "samples": samples,
        "features": features,
        "cds_metrics": {
            "gt": gt_cds_metrics,
            "generated": generated_cds_metrics,
        },
        "global_metrics": {
            "gt": gt_global_metrics,
            "generated": generated_global_metrics,
        },
    }


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_csv", type=str, required=True, help="Path to the input samples CSV file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation config YAML file.")
    parser.add_argument("--eval_dir", type=str, default=EVAL_DIR, help="Default directory for evals")
    parser.add_argument("--save_dir", type=str, help="Specified concrete directory for this eval run (overrides --eval_dir)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to evaluate (for testing)")

    return parser


def main() -> None:
    parser = _get_parser()
    args = parser.parse_args()

    samples_df = pd.read_csv(args.samples_csv)
    print(f"Loaded {len(samples_df)} samples from {args.samples_csv}")

    if args.max_samples is not None:
        samples_df = samples_df.head(args.max_samples)
        print(f"Using only the first {args.max_samples} samples for evaluation")

    config = load_config(args.config)
    print(f"Loaded evaluation config from {args.config}")

    feature_extractors = [load_feature_extractor(fe_config) for fe_config in config.feature_extractors]
    scoring_models = [load_scoring_model(sm_config) for sm_config in config.scoring_models]
    # aggregators = [load_aggregator(agg_config) for agg_config in config.aggregators]

    eval_results = run_evaluation(
        samples=samples_df,
        feature_extractors=feature_extractors,
        scoring_models=scoring_models,
        # aggregators=aggregators,
        config=config,
    )

    print("Evaluation completed. Results:")
    for k, v in eval_results.items():
        if isinstance(v, pd.DataFrame):
            print(k)
            print(v.head())
        else:
            print(f"{k}: {v}")

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = f"{args.eval_dir}/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    save_results(
        eval_results=eval_results,
        save_dir=save_dir,
        plots=config.plots,
    )
    print(f"Saved evaluation results to {save_dir}")



if __name__ == "__main__":
    main()
