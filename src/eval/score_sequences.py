import argparse
import os
import yaml
from math import ceil

import pandas as pd
from tqdm import tqdm

from . import scorers


INDEX_COLS = ["id", "sample"]
SEQUENCE_COLS = ["utr5", "cds", "utr3"]

EVAL_DIR = "../../data/evals"

SCORERS = {
    "string_statistics": scorers.StringStatisticsScorer,
    "ribonn": scorers.RiboNNScorer,
    "rnafold": scorers.RNAfoldScorer,
    # "utrlm": scorers.UTRLMScorer,
}

RIBONN_BATCH_SIZE = 16


def _save_scores(scores_df: pd.DataFrame, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "scores.csv")
    scores_df.to_csv(output_path, index=False)
    print(f"Saved scores to {output_path}")


def _sanitize_samples_df(samples_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = set(INDEX_COLS + SEQUENCE_COLS)
    missing_columns = required_columns - set(samples_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in samples CSV: {missing_columns}")
    samples_df = samples_df.fillna("")
    return samples_df[INDEX_COLS + SEQUENCE_COLS]


def _with_index_cols(scores_df: pd.DataFrame, samples_df: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical index columns by row order for scorer outputs."""
    score_cols = [col for col in scores_df.columns if col not in INDEX_COLS]
    return pd.concat(
        [
            samples_df[INDEX_COLS].reset_index(drop=True),
            scores_df[score_cols].reset_index(drop=True),
        ],
        axis=1,
    )


def _load_samples(samples_csv: str, max_samples: int | None = None) -> pd.DataFrame:
    samples_df = pd.read_csv(samples_csv)
    if max_samples is not None:
        samples_df = samples_df.head(max_samples)
    return samples_df


def _load_scorers(scorers_config: dict) -> list[scorers.Scorer]:
    """Instantiate scorer objects based on the provided configuration."""
    scorers_list = []
    for scorer_name, scorer_params in scorers_config.items():
        scorer_cls = SCORERS.get(scorer_name)
        if scorer_cls is None:
            # raise Warning(f"Unknown scorer: {scorer_name}. Skipping.")
            continue
        scorer = scorer_cls(**scorer_params)
        scorers_list.append(scorer)
        print(f"Initialized scorer: {scorer_name} with params: {scorer_params}")
    return scorers_list


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation config YAML file.")
    parser.add_argument("--samples_csv", type=str, help="Path to the input samples CSV file.")
    parser.add_argument("--eval_dir", type=str, default=EVAL_DIR, help="Default directory for evals")
    parser.add_argument("--save_dir", type=str, help="Specified concrete directory for this eval run (overrides --eval_dir)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to evaluate (for testing)")

    return parser


def main() -> None:
    parser = _get_parser()
    args = parser.parse_args()

    config: dict = _load_config(args.config)

    print(f"Loaded config from {args.config}: {config}")

    samples_csv = args.samples_csv or config.get("samples_csv")
    if samples_csv is None:
        raise ValueError("No samples CSV provided. Please specify --samples_csv or include it in the config file.")

    max_samples = args.max_samples or config.get("max_samples")
    print(f"Using samples CSV: {samples_csv} with max_samples={max_samples}")

    raw_samples_df = _load_samples(samples_csv, max_samples)
    samples_df = _sanitize_samples_df(raw_samples_df)

    scorers_list: list[scorers.Scorer] = _load_scorers(config.get("scorers", {}))
    scores_dfs: dict[str, pd.DataFrame] = {}

    for scorer in scorers_list:
        print(f"Scoring with {scorer.name}...")

        if scorer.name == "RiboNN":
            print("Doing batched scoring for RiboNN...")

            batch_scores = []
            n_batches = ceil(len(samples_df) / RIBONN_BATCH_SIZE)
            iterator = enumerate(range(0, len(samples_df), RIBONN_BATCH_SIZE), start=1)

            for batch_idx, start in tqdm(iterator, total=n_batches, desc=f"Scoring with {scorer.name}"):
                # print(f"Scoring batch {batch_idx}/{n_batches} (samples {start} to {min(start + RIBONN_BATCH_SIZE, len(samples_df)) - 1})...")

                scores_batch_df = scorer.score_df(
                    samples_df.iloc[start:start + RIBONN_BATCH_SIZE],
                    index_cols=INDEX_COLS,
                    progress_bar=True,
                )

                batch_scores.append(scores_batch_df)

            scores_df = pd.concat(batch_scores, ignore_index=True)
            scores_dfs[scorer.name] = _with_index_cols(scores_df, samples_df)
        else:
            scores_df = scorer.score_df(samples_df, index_cols=INDEX_COLS, progress_bar=True)
            scores_dfs[scorer.name] = _with_index_cols(scores_df, samples_df)

    # Merge all scores into a single DataFrame
    final_scores_df = samples_df[INDEX_COLS].copy()
    for scorer_name, scores_df in scores_dfs.items():
        final_scores_df = final_scores_df.merge(scores_df, on=INDEX_COLS)

    save_dir = config.get("save_dir") or args.save_dir or args.eval_dir
    _save_scores(final_scores_df, save_dir)


if __name__ == "__main__":
    main()
