import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import yaml


def _get_eval_dir(config: dict) -> Path:
    if "eval_dir" in config and config["eval_dir"]:
        eval_dir = Path(config["eval_dir"])
        print(f"Using specified eval directory from config: {eval_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = Path("evals") / f"eval_{timestamp}"
        eval_dir.mkdir(parents=True, exist_ok=False)
        print(f"Created eval directory: {eval_dir}")
    return eval_dir


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation config YAML file.")
    return parser


def main() -> None:
    parser = _get_parser()
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config path: {config_path}")

    # If eval dir is specified in config, use it. Otherwise, create a new one with timestamp.
    eval_dir = _get_eval_dir(config)
    eval_dir = eval_dir.resolve()
    repo_root = Path(__file__).resolve().parents[2]

    print("Running score_sequences.py...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval.score_sequences",
            "--config",
            str(config_path),
            "--eval_dir",
            str(eval_dir),
        ],
        cwd=repo_root,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval.calculate_statistics",
            "--config",
            str(config_path),
            "--scores_csv",
            str(eval_dir / "scores.csv"),
            "--output_dir",
            str(eval_dir),
        ],
        cwd=repo_root,
        check=True,
    )

    print(f"Saved eval results to: {eval_dir}")


if __name__ == "__main__":
    main()
