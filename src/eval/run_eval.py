import argparse
from dataclasses import dataclass
import os
from datetime import datetime
from functools import reduce
from typing import Iterable

import pandas as pd
import yaml

from eval import FeatureExtractor, ScoringModel, Aggregator, EvalConfig
from schemas import SAMPLE_CSV_COLS, SAMPLE_INDEX_COLS

from feature_extractors import StringStatisticsExtractor
from scoring_models import RiboNN
from aggregators import SimpleAverageAggregator
from plots import feature_distribution


EVAL_DIR = "data/evals"

FEATURE_EXTRACTORS = {
    "StringStatistics": StringStatisticsExtractor
}

SCORING_MODELS = {
    "RiboNN": RiboNN,
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


def run_evaluation(
    samples: pd.DataFrame,
    feature_extractors: Iterable[FeatureExtractor],
    scoring_models: Iterable[ScoringModel],
    aggregators: Iterable[Aggregator],
    config: EvalConfig,
) -> dict[str, pd.DataFrame]:

    validate_samples_schema(samples)

    feature_tables = {}
    scoring_model_tables = {}


    for extractor in feature_extractors:
        print(f"Extracting features using {extractor.__class__.__name__}...")
        feature_tables[extractor.__class__.__name__] = extractor.extract_features(samples)

    for scoring_model in scoring_models:
        print(f"Scoring samples using {scoring_model.__class__.__name__}...")
        scoring_model_tables[scoring_model.__class__.__name__] = scoring_model.score(samples)

    for aggregator in aggregators:
        print(f"Aggregating scores using {aggregator.__class__.__name__}...")

    print("FEATURE TABLES:")
    print(feature_tables)

    features = merge_feature_tables(feature_tables)
    oracle_scores = merge_oracle_tables(scoring_model_tables)

    print("Merged features:")
    print(features)

    return {
        "samples": samples,
        "features": features,
        "orcale_scores": oracle_scores,
        # "metrics": metrics
    }


def load_config(config_path: str) -> EvalConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    print(f"Loaded config: {config_dict}")
    return EvalConfig(**config_dict)


def save_results(
    eval_results: dict[str, pd.DataFrame],
    save_dir: str,
    plots: list[dict],
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    for key, df in eval_results.items():
        df.to_csv(f"{save_dir}/{key}.csv", index=False)
        print(f"Saved {key} to {save_dir}/{key}.csv")

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



def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_csv", type=str, required=True, help="Path to the input samples CSV file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation config YAML file.")
    parser.add_argument("--eval_dir", type=str, default=EVAL_DIR, help="Default directory for evals")
    parser.add_argument("--save_dir", type=str, help="Specified concrete directory for this eval run (overrides --eval_dir)")

    return parser


def main() -> None:
    parser = _get_parser()
    args = parser.parse_args()

    samples_df = pd.read_csv(args.samples_csv)
    print(f"Loaded {len(samples_df)} samples from {args.samples_csv}")

    config = load_config(args.config)
    print(f"Loaded evaluation config from {args.config}")

    feature_extractors = [load_feature_extractor(fe_config) for fe_config in config.feature_extractors]
    scoring_models = [load_scoring_model(sm_config) for sm_config in config.scoring_models]
    aggregators = [load_aggregator(agg_config) for agg_config in config.aggregators]

    eval_results = run_evaluation(
        samples=samples_df,
        feature_extractors=feature_extractors,
        scoring_models=scoring_models,
        aggregators=aggregators,
        config=config,
    )

    print("Evaluation completed. Results:")
    for key, df in eval_results.items():
        print(f"{key}:", df.head())

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
