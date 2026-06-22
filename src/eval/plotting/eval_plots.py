"""Plot suites for model evaluation directories.

The expected input layout is an evaluation directory containing one subdirectory
per model. Each model directory must contain ``sequences.csv`` and ``scores.csv``.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "teamml_mrna_plot_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

try:
    import numpy as np
    import pandas as pd

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing plotting dependency. Run this script in the project environment "
        "(for example: venv/bin/python src/eval/generate_plots.py <eval_dir>). "
        f"Original error: {exc}"
    ) from exc


SEQUENCE_COLUMNS = ("utr5", "cds", "utr3")
SEQUENCES_REQUIRED_COLUMNS = {"id", "sample", *SEQUENCE_COLUMNS}
SCORES_REQUIRED_COLUMNS = {"id", "sample"}
TE_COLUMN_CANDIDATES = ("ribonn_te", "predicted_te", "te")
COHORT_GT = "Ground truth"
COHORT_GENERATED = "Generated"
DIVERSITY_KMER_SIZE = 3
DIVERSITY_SEQUENCE_REGION = "utr_context"
COHORT_COLORS = {
    COHORT_GT: "#2563eb",
    COHORT_GENERATED: "#16a34a",
}
DIVERSITY_COLORS = {
    "Generated vs generated": "#0f766e",
    "Generated vs ground truth": "#be123c",
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    integer_bins: bool = False


METRICS = (
    MetricSpec("utr5_length", "5' UTR length (nt)", integer_bins=True),
    MetricSpec("cds_length", "CDS length (nt)", integer_bins=True),
    MetricSpec("utr3_length", "3' UTR length (nt)", integer_bins=True),
    MetricSpec("utr5_gc_content", "5' UTR GC content"),
    MetricSpec("cds_gc_content", "CDS GC content"),
    MetricSpec("utr3_gc_content", "3' UTR GC content"),
    MetricSpec("ribonn_te", "Predicted TE"),
)
UTR_LENGTH_METRICS = (
    MetricSpec("utr5_length", "UTR5 length (nt)", integer_bins=True),
    MetricSpec("utr3_length", "UTR3 length (nt)", integer_bins=True),
)
UTR_GC_METRICS = (
    MetricSpec("utr5_gc_content", "UTR5 GC content"),
    MetricSpec("utr3_gc_content", "UTR3 GC content"),
)
MODEL_GRID_COLUMNS = 3
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class EvalPlotResult:
    output_dir: Path
    model_count: int
    row_count: int


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def generate_eval_plots(eval_dir: Path, progress: ProgressCallback | None = None) -> EvalPlotResult:
    """Generate the full plot suite for ``eval_dir``."""

    eval_dir = eval_dir.resolve()
    output_dir = eval_dir / "eval_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, "Output directory ready")

    metrics = load_eval_metrics(eval_dir, progress=progress)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    _save_summary(metrics, output_dir)
    _emit(progress, "Saved metrics and summary tables")

    improvements = calculate_te_improvements(metrics)
    improvements.to_csv(output_dir / "te_improvements.csv", index=False)
    _emit(progress, "Computed TE improvements")

    sequence_diversity = calculate_sequence_diversity(eval_dir, progress=progress)
    sequence_diversity.to_csv(output_dir / "sequence_diversity.csv", index=False)
    _save_sequence_diversity_summary(sequence_diversity, output_dir)
    _emit(progress, "Computed sequence diversity")

    _plot_per_model(metrics, improvements, output_dir / "per_model", progress=progress)
    _plot_merged(metrics, improvements, sequence_diversity, output_dir / "merged", progress=progress)

    return EvalPlotResult(
        output_dir=output_dir,
        model_count=int(metrics["model"].nunique()),
        row_count=len(metrics),
    )


def load_eval_metrics(eval_dir: Path, progress: ProgressCallback | None = None) -> pd.DataFrame:
    """Load all model subdirectories and return one tidy metrics table."""

    model_dirs = _discover_model_dirs(eval_dir)
    records = []
    for model_idx, model_dir in enumerate(model_dirs, start=1):
        model_metrics = _load_model_metrics(model_dir)
        records.append(model_metrics)
        _emit(progress, f"Loaded model {model_idx}/{len(model_dirs)}: {model_dir.name}")

    metrics = pd.concat(records, ignore_index=True)
    metric_columns = [metric.name for metric in METRICS]
    metrics[metric_columns] = metrics[metric_columns].apply(pd.to_numeric, errors="coerce")
    return metrics


def calculate_te_improvements(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare generated TE against each CDS ground-truth TE by model and id."""

    records: list[dict[str, object]] = []
    for (model, cds_id), group in metrics.groupby(["model", "id"], sort=True, dropna=False):
        gt_te = group.loc[group["cohort"] == COHORT_GT, "ribonn_te"].dropna()
        generated_te = group.loc[group["cohort"] == COHORT_GENERATED, "ribonn_te"].dropna()
        if gt_te.empty or generated_te.empty:
            continue
        gt_value = float(gt_te.mean())
        generated_mean = float(generated_te.mean())
        generated_best = float(generated_te.max())
        records.append(
            {
                "model": model,
                "id": cds_id,
                "gt_te": gt_value,
                "generated_mean_te": generated_mean,
                "generated_best_te": generated_best,
                "mean_te_improvement": generated_mean - gt_value,
                "best_te_improvement": generated_best - gt_value,
                "n_generated": int(generated_te.size),
            }
        )

    return pd.DataFrame.from_records(records)


def calculate_sequence_diversity(
    eval_dir: Path,
    *,
    kmer_size: int = DIVERSITY_KMER_SIZE,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Measure UTR-context sequence diversity for generated samples grouped by CDS."""

    records: list[dict[str, object]] = []
    model_dirs = _discover_model_dirs(eval_dir)
    for model_idx, model_dir in enumerate(model_dirs, start=1):
        records.extend(_calculate_model_sequence_diversity(model_dir, kmer_size=kmer_size))
        _emit(progress, f"Sequence diversity {model_idx}/{len(model_dirs)}: {model_dir.name}")

    columns = [
        "model",
        "cds",
        "representative_id",
        "id_count",
        "n_generated",
        "n_ground_truth",
        "kmer_size",
        "sequence_region",
        "generated_pair_count",
        "generated_gt_pair_count",
        "mean_generated_pairwise_distance",
        "median_generated_pairwise_distance",
        "min_generated_pairwise_distance",
        "max_generated_pairwise_distance",
        "mean_generated_to_gt_distance",
        "median_generated_to_gt_distance",
        "min_generated_to_gt_distance",
        "max_generated_to_gt_distance",
        "unique_generated_sequences",
        "generated_unique_rate",
        "mean_generated_utr_length",
        "mean_ground_truth_utr_length",
    ]
    diversity = pd.DataFrame.from_records(records, columns=columns)
    if not diversity.empty:
        diversity = diversity.sort_values(["model", "representative_id", "cds"]).reset_index(drop=True)
    return diversity


def _discover_model_dirs(eval_dir: Path) -> list[Path]:
    if not eval_dir.exists():
        raise FileNotFoundError(f"eval_dir does not exist: {eval_dir}")
    if not eval_dir.is_dir():
        raise NotADirectoryError(f"eval_dir is not a directory: {eval_dir}")

    model_dirs = [
        child
        for child in sorted(eval_dir.iterdir())
        if child.is_dir()
        and child.name != "eval_plots"
        and (child / "sequences.csv").exists()
        and (child / "scores.csv").exists()
    ]
    if not model_dirs:
        raise FileNotFoundError(
            f"No model subdirectories with both sequences.csv and scores.csv found under {eval_dir}"
        )
    return model_dirs


def _load_model_metrics(model_dir: Path) -> pd.DataFrame:
    sequences_path = model_dir / "sequences.csv"
    scores_path = model_dir / "scores.csv"
    sequences = pd.read_csv(sequences_path)
    scores = pd.read_csv(scores_path)

    _require_columns(sequences, SEQUENCES_REQUIRED_COLUMNS, sequences_path)
    _require_columns(scores, SCORES_REQUIRED_COLUMNS, scores_path)
    te_column = _resolve_te_column(scores, scores_path)

    sequence_metrics = sequences[["id", "sample", *SEQUENCE_COLUMNS]].copy()
    sequence_metrics["_id_key"] = sequence_metrics["id"].map(_key)
    sequence_metrics["_sample_key"] = sequence_metrics["sample"].map(_key)
    for column in SEQUENCE_COLUMNS:
        cleaned = sequence_metrics[column].map(_clean_sequence)
        sequence_metrics[f"{column}_length"] = cleaned.map(len)
        sequence_metrics[f"{column}_gc_content"] = cleaned.map(_gc_content)

    te_scores = scores[["id", "sample", te_column]].copy()
    te_scores["_id_key"] = te_scores["id"].map(_key)
    te_scores["_sample_key"] = te_scores["sample"].map(_key)
    te_scores = te_scores.rename(columns={te_column: "ribonn_te"})

    merged = sequence_metrics.merge(
        te_scores[["_id_key", "_sample_key", "ribonn_te"]],
        on=["_id_key", "_sample_key"],
        how="left",
        validate="one_to_one",
    )
    merged.insert(0, "model", model_dir.name)
    merged["cohort"] = np.where(
        merged["sample"].map(_key).str.lower() == "gt",
        COHORT_GT,
        COHORT_GENERATED,
    )
    return merged[
        [
            "model",
            "id",
            "sample",
            "cohort",
            "utr5_length",
            "cds_length",
            "utr3_length",
            "utr5_gc_content",
            "cds_gc_content",
            "utr3_gc_content",
            "ribonn_te",
        ]
    ]


def _calculate_model_sequence_diversity(
    model_dir: Path,
    *,
    kmer_size: int,
) -> list[dict[str, object]]:
    sequences_path = model_dir / "sequences.csv"
    sequences = pd.read_csv(sequences_path)
    _require_columns(sequences, SEQUENCES_REQUIRED_COLUMNS, sequences_path)

    prepared = sequences[["id", "sample", *SEQUENCE_COLUMNS]].copy()
    prepared["cohort"] = np.where(
        prepared["sample"].map(_key).str.lower() == "gt",
        COHORT_GT,
        COHORT_GENERATED,
    )
    prepared["_cds_key"] = prepared["cds"].map(_clean_sequence)
    prepared["_utr5"] = prepared["utr5"].map(_clean_sequence)
    prepared["_utr3"] = prepared["utr3"].map(_clean_sequence)
    prepared["_utr_length"] = prepared["_utr5"].map(len) + prepared["_utr3"].map(len)
    prepared["_signature"] = prepared.apply(
        lambda row: _utr_context_signature(row["_utr5"], row["_utr3"], kmer_size=kmer_size),
        axis=1,
    )
    prepared["_exact_utr_context"] = prepared["_utr5"] + "|" + prepared["_utr3"]

    records: list[dict[str, object]] = []
    for cds_key, group in prepared.groupby("_cds_key", sort=True, dropna=False):
        generated = group[group["cohort"] == COHORT_GENERATED]
        ground_truth = group[group["cohort"] == COHORT_GT]
        generated_signatures = generated["_signature"].tolist()
        ground_truth_signatures = ground_truth["_signature"].tolist()

        generated_pairwise = _pairwise_jaccard_distances(generated_signatures)
        generated_to_gt = _cross_jaccard_distances(generated_signatures, ground_truth_signatures)
        ids = sorted(group["id"].map(_key).unique())
        records.append(
            {
                "model": model_dir.name,
                "cds": cds_key,
                "representative_id": ids[0] if ids else "",
                "id_count": len(ids),
                "n_generated": int(len(generated)),
                "n_ground_truth": int(len(ground_truth)),
                "kmer_size": int(kmer_size),
                "sequence_region": DIVERSITY_SEQUENCE_REGION,
                "generated_pair_count": int(len(generated_pairwise)),
                "generated_gt_pair_count": int(len(generated_to_gt)),
                **_distance_summary(generated_pairwise, prefix="generated_pairwise"),
                **_distance_summary(generated_to_gt, prefix="generated_to_gt"),
                "unique_generated_sequences": int(generated["_exact_utr_context"].nunique(dropna=True)),
                "generated_unique_rate": _safe_divide(
                    float(generated["_exact_utr_context"].nunique(dropna=True)),
                    float(len(generated)),
                ),
                "mean_generated_utr_length": _safe_mean(generated["_utr_length"]),
                "mean_ground_truth_utr_length": _safe_mean(ground_truth["_utr_length"]),
            }
        )

    return records


def _utr_context_signature(utr5: str, utr3: str, *, kmer_size: int) -> frozenset[str]:
    tokens: set[str] = set()
    for label, sequence in (("utr5", utr5), ("utr3", utr3)):
        normalized = sequence.replace("U", "T")
        tokens.update(f"{label}:{kmer}" for kmer in _kmers(normalized, kmer_size=kmer_size))
    return frozenset(tokens)


def _kmers(sequence: str, *, kmer_size: int) -> set[str]:
    valid_sequence = "".join(base for base in sequence if base in {"A", "C", "G", "T"})
    if not valid_sequence:
        return set()
    if len(valid_sequence) < kmer_size:
        return {valid_sequence}
    return {
        valid_sequence[start : start + kmer_size]
        for start in range(0, len(valid_sequence) - kmer_size + 1)
    }


def _pairwise_jaccard_distances(signatures: list[frozenset[str]]) -> list[float]:
    return [
        _jaccard_distance(left, right)
        for left_idx, left in enumerate(signatures)
        for right in signatures[left_idx + 1 :]
    ]


def _cross_jaccard_distances(
    left_signatures: list[frozenset[str]],
    right_signatures: list[frozenset[str]],
) -> list[float]:
    return [_jaccard_distance(left, right) for left in left_signatures for right in right_signatures]


def _jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 0.0
    union_size = len(left | right)
    if union_size == 0:
        return 0.0
    return 1.0 - (len(left & right) / union_size)


def _distance_summary(values: list[float], *, prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"mean_{prefix}_distance": math.nan,
            f"median_{prefix}_distance": math.nan,
            f"min_{prefix}_distance": math.nan,
            f"max_{prefix}_distance": math.nan,
        }
    array = np.asarray(values, dtype=float)
    return {
        f"mean_{prefix}_distance": float(np.mean(array)),
        f"median_{prefix}_distance": float(np.median(array)),
        f"min_{prefix}_distance": float(np.min(array)),
        f"max_{prefix}_distance": float(np.max(array)),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return math.nan
    return numerator / denominator


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    return float(numeric.mean())


def _require_columns(df: pd.DataFrame, required_columns: set[str], path: Path) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required column(s): {missing}")


def _resolve_te_column(scores: pd.DataFrame, path: Path) -> str:
    for column in TE_COLUMN_CANDIDATES:
        if column in scores.columns:
            return column

    numeric_columns = [
        column
        for column in scores.columns
        if column not in SCORES_REQUIRED_COLUMNS and pd.api.types.is_numeric_dtype(scores[column])
    ]
    if len(numeric_columns) == 1:
        return numeric_columns[0]

    candidates = ", ".join(TE_COLUMN_CANDIDATES)
    raise ValueError(f"{path} is missing a TE score column. Expected one of: {candidates}")


def _key(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _clean_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _gc_content(sequence: str) -> float:
    bases = [base for base in sequence if base in {"A", "C", "G", "T", "U"}]
    if not bases:
        return math.nan
    gc_count = sum(base in {"G", "C"} for base in bases)
    return gc_count / len(bases)


def _save_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = [metric.name for metric in METRICS]
    summary = (
        metrics.groupby(["model", "cohort"], dropna=False)[metric_columns]
        .agg(["count", "mean", "median", "min", "max"])
        .round(6)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "summary.csv", index=False)


def _save_sequence_diversity_summary(diversity: pd.DataFrame, output_dir: Path) -> None:
    if diversity.empty:
        pd.DataFrame().to_csv(output_dir / "sequence_diversity_summary.csv", index=False)
        return

    summary_columns = [
        "mean_generated_pairwise_distance",
        "mean_generated_to_gt_distance",
        "generated_unique_rate",
        "mean_generated_utr_length",
        "mean_ground_truth_utr_length",
    ]
    summary = (
        diversity.groupby("model", dropna=False)[summary_columns]
        .agg(["count", "mean", "median", "min", "max"])
        .round(6)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "sequence_diversity_summary.csv", index=False)


def _plot_per_model(
    metrics: pd.DataFrame,
    improvements: pd.DataFrame,
    output_dir: Path,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(metrics["model"].unique())
    for model_idx, model in enumerate(models, start=1):
        model_dir = output_dir / _slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_metrics = metrics[metrics["model"] == model]

        _plot_distribution_grid(
            model_metrics,
            model_dir / "distributions.png",
            title=f"{_display_name(model)}: generated vs ground truth distributions",
            split_by_model=False,
        )
        _plot_correlation_heatmap(
            model_metrics,
            model_dir / "correlation_heatmap.png",
            title=f"{_display_name(model)}: metric correlations",
        )
        _plot_pairwise_scatters(
            model_metrics,
            model_dir / "correlation_scatters",
            title_prefix=f"{_display_name(model)}",
            merged=False,
        )

        model_improvements = improvements[improvements["model"] == model]
        _plot_te_improvement(
            model_improvements,
            model_dir / "te_improvement_by_cds.png",
            title=f"{_display_name(model)}: generated TE improvement over ground truth",
        )
        _emit(progress, f"Per-model plots {model_idx}/{len(models)}: {model}")


def _plot_merged(
    metrics: pd.DataFrame,
    improvements: pd.DataFrame,
    sequence_diversity: pd.DataFrame,
    output_dir: Path,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_scatter_dir = output_dir / "correlation_scatters"
    if stale_scatter_dir.exists():
        shutil.rmtree(stale_scatter_dir)

    _plot_grouped_distribution_by_model(
        metrics,
        UTR_LENGTH_METRICS,
        output_dir / "utr_length_distributions.png",
        title="UTR length distributions: generated vs ground truth",
        ylabel="UTR density",
    )
    _emit(progress, "Merged plot: UTR lengths")

    _plot_grouped_distribution_by_model(
        metrics,
        UTR_GC_METRICS,
        output_dir / "utr_gc_content_distributions.png",
        title="UTR GC content distributions: generated vs ground truth",
        ylabel="UTR density",
    )
    _emit(progress, "Merged plot: UTR GC content")

    _plot_metric_distribution_grid(
        metrics,
        MetricSpec("ribonn_te", "Predicted TE"),
        output_dir / "predicted_te_distributions_by_model.png",
        title="Predicted TE distributions by model: generated vs ground truth",
    )
    _emit(progress, "Merged plot: predicted TE")

    _plot_correlation_heatmap(
        metrics,
        output_dir / "correlation_heatmap.png",
        title="Metric correlations across all models",
    )
    _emit(progress, "Merged plot: correlation heatmap")

    _plot_merged_te_improvement(
        improvements,
        output_dir / "best_te_improvement_by_model.png",
        title="Best generated TE improvement vs ground truth by model",
        value_column="best_te_improvement",
        ylabel="Best generated TE - ground-truth TE",
    )
    _emit(progress, "Merged plot: best TE improvement")

    _plot_focused_scatter_by_model(
        metrics,
        "utr5_length",
        "utr3_length",
        output_dir / "utr3_length_vs_utr5_length.png",
        title="UTR3 length vs UTR5 length",
        xlabel="UTR5 length (nt)",
        ylabel="UTR3 length (nt)",
    )
    _emit(progress, "Merged plot: UTR3 vs UTR5")

    _plot_sequence_diversity_distributions(
        sequence_diversity,
        output_dir / "sequence_diversity_distributions.png",
        title="Sequence diversity by model and CDS",
    )
    _emit(progress, "Merged plot: sequence diversity distributions")

    _plot_diversity_vs_gt_distance_scatter(
        sequence_diversity,
        output_dir / "sequence_diversity_vs_ground_truth_distance.png",
        title="Generated diversity vs distance from ground truth by CDS",
    )
    _emit(progress, "Merged plot: diversity vs ground truth")

    _plot_sequence_diversity_heatmap(
        sequence_diversity,
        "mean_generated_pairwise_distance",
        output_dir / "generated_pairwise_diversity_by_cds.png",
        title="Generated pairwise sequence diversity by CDS",
        colorbar_label="Mean generated-vs-generated k-mer Jaccard distance",
    )
    _emit(progress, "Merged plot: generated pairwise diversity heatmap")

    _plot_sequence_diversity_heatmap(
        sequence_diversity,
        "mean_generated_to_gt_distance",
        output_dir / "generated_to_ground_truth_distance_by_cds.png",
        title="Generated distance from ground truth by CDS",
        colorbar_label="Mean generated-vs-ground-truth k-mer Jaccard distance",
    )
    _emit(progress, "Merged plot: generated to ground truth heatmap")


def _plot_distribution_grid(
    df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    split_by_model: bool,
) -> None:
    models = sorted(df["model"].unique()) if split_by_model else [None]
    nrows = len(METRICS)
    ncols = len(models)
    fig_width = max(5.0 * ncols, 8.0)
    fig_height = max(2.4 * nrows, 10.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)

    for row_idx, metric in enumerate(METRICS):
        values = _finite_values(df, metric.name)
        bins = _hist_bins(values, bins=40, integer_bins=metric.integer_bins)
        for col_idx, model in enumerate(models):
            ax = axes[row_idx][col_idx]
            subset = df if model is None else df[df["model"] == model]
            _draw_histograms(ax, subset, metric.name, bins)
            if row_idx == 0 and model is not None:
                ax.set_title(_display_name(model), fontsize=10)
            ax.set_xlabel(metric.label)
            ax.set_ylabel("Density")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.97, 0.975))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_distribution_by_model(
    df: pd.DataFrame,
    metrics: tuple[MetricSpec, ...],
    output_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(models),
        figsize=(max(3.2 * len(models), 9.0), max(2.8 * len(metrics), 5.5)),
        squeeze=False,
    )

    for row_idx, metric in enumerate(metrics):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx][col_idx]
            subset = df[df["model"] == model]
            bins = _hist_bins(_finite_values(subset, metric.name), bins=40, integer_bins=metric.integer_bins)
            _draw_histograms(ax, subset, metric.name, bins)
            if row_idx == 0:
                ax.set_title(_display_name(model), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xlabel(metric.label)
            ax.grid(axis="y", alpha=0.2)

    _add_shared_legend(fig, axes[0][0])
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.97, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_distribution_grid(
    df: pd.DataFrame,
    metric: MetricSpec,
    output_path: Path,
    *,
    title: str,
) -> None:
    models = sorted(df["model"].unique())
    ncols = min(MODEL_GRID_COLUMNS, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.25 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    bins = _hist_bins(_finite_values(df, metric.name), bins=40, integer_bins=metric.integer_bins)

    for model_idx, model in enumerate(models):
        ax = axes[model_idx // ncols][model_idx % ncols]
        subset = df[df["model"] == model]
        _draw_histograms(ax, subset, metric.name, bins)
        _draw_cohort_means(ax, subset, metric.name)
        ax.set_title(_display_name(model), fontsize=11)
        ax.set_xlabel(metric.label)
        if model_idx % ncols == 0:
            ax.set_ylabel("Density")
        ax.grid(axis="y", alpha=0.2)

    _hide_unused_axes(axes, len(models))
    _add_shared_legend(fig, axes[0][0])
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.97, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_cohort_means(ax: plt.Axes, df: pd.DataFrame, column: str) -> None:
    for cohort in (COHORT_GENERATED, COHORT_GT):
        values = _finite_values(df[df["cohort"] == cohort], column)
        if values.empty:
            continue
        ax.axvline(
            float(values.mean()),
            color=COHORT_COLORS[cohort],
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )


def _finite_values(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    return values[np.isfinite(values)]


def _hist_bins(values: pd.Series, *, bins: int, integer_bins: bool) -> np.ndarray | int:
    if values.empty:
        return bins
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    if integer_bins:
        start = math.floor(min_value) - 0.5
        stop = math.ceil(max_value) + 0.5
        bin_count = int(min(max(stop - start, 1), bins))
        return np.linspace(start, stop, bin_count + 1)
    return np.linspace(min_value, max_value, bins + 1)


def _draw_histograms(ax: plt.Axes, df: pd.DataFrame, column: str, bins: np.ndarray | int) -> None:
    for cohort in (COHORT_GENERATED, COHORT_GT):
        values = _finite_values(df[df["cohort"] == cohort], column)
        if values.empty:
            continue
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.28,
            linewidth=1.3,
            color=COHORT_COLORS[cohort],
            label=cohort,
        )
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.2,
            color=COHORT_COLORS[cohort],
        )
    ax.grid(axis="y", alpha=0.25)


def _add_shared_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)


def _hide_unused_axes(axes: np.ndarray, used_count: int) -> None:
    for ax in axes.flat[used_count:]:
        ax.set_visible(False)


def _plot_correlation_heatmap(df: pd.DataFrame, output_path: Path, *, title: str) -> None:
    metric_columns = [metric.name for metric in METRICS]
    corr = _safe_correlation_matrix(df, metric_columns)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    image = ax.imshow(corr.to_numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [_short_metric_label(metric.name) for metric in METRICS]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)

    for row_idx in range(corr.shape[0]):
        for col_idx in range(corr.shape[1]):
            value = corr.iat[row_idx, col_idx]
            if pd.isna(value):
                label = ""
            else:
                label = f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _safe_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    corr = pd.DataFrame(np.nan, index=columns, columns=columns, dtype=float)
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    for x_name in columns:
        for y_name in columns:
            if x_name == y_name:
                values = numeric[x_name].dropna()
                if len(values) >= 2 and values.nunique() >= 2:
                    corr.loc[x_name, y_name] = 1.0
                continue
            pair = numeric[[x_name, y_name]].dropna()
            if len(pair) < 2 or pair[x_name].nunique() < 2 or pair[y_name].nunique() < 2:
                continue
            corr.loc[x_name, y_name] = float(pair[x_name].corr(pair[y_name]))
    return corr


def _plot_pairwise_scatters(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    title_prefix: str,
    merged: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_by_name = {metric.name: metric for metric in METRICS}
    for x_name, y_name in combinations(metric_by_name, 2):
        x_metric = metric_by_name[x_name]
        y_metric = metric_by_name[y_name]
        pair_df = df[["model", "cohort", x_name, y_name]].copy()
        pair_df[x_name] = pd.to_numeric(pair_df[x_name], errors="coerce")
        pair_df[y_name] = pd.to_numeric(pair_df[y_name], errors="coerce")
        pair_df = pair_df.dropna(subset=[x_name, y_name])

        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        if merged:
            _draw_merged_scatter(ax, pair_df, x_name, y_name)
        else:
            _draw_cohort_scatter(ax, pair_df, x_name, y_name)

        ax.set_xlabel(x_metric.label)
        ax.set_ylabel(y_metric.label)
        ax.set_title(f"{title_prefix}: {_short_metric_label(x_name)} vs {_short_metric_label(y_name)}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{_slug(x_name)}_vs_{_slug(y_name)}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _draw_cohort_scatter(ax: plt.Axes, df: pd.DataFrame, x_name: str, y_name: str) -> None:
    for cohort in (COHORT_GENERATED, COHORT_GT):
        subset = df[df["cohort"] == cohort]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_name],
            subset[y_name],
            s=18 if cohort == COHORT_GENERATED else 28,
            alpha=0.38 if cohort == COHORT_GENERATED else 0.78,
            color=COHORT_COLORS[cohort],
            label=cohort,
            edgecolors="none",
        )


def _draw_merged_scatter(ax: plt.Axes, df: pd.DataFrame, x_name: str, y_name: str) -> None:
    models = sorted(df["model"].unique())
    cmap = plt.get_cmap("tab10")
    for model_idx, model in enumerate(models):
        for cohort, marker in ((COHORT_GENERATED, "o"), (COHORT_GT, "^")):
            subset = df[(df["model"] == model) & (df["cohort"] == cohort)]
            if subset.empty:
                continue
            ax.scatter(
                subset[x_name],
                subset[y_name],
                s=16 if cohort == COHORT_GENERATED else 30,
                alpha=0.35 if cohort == COHORT_GENERATED else 0.85,
                color=cmap(model_idx % 10),
                marker=marker,
                label=f"{_display_name(model)} {cohort}",
                edgecolors="none",
            )


def _plot_te_improvement(df: pd.DataFrame, output_path: Path, *, title: str) -> None:
    if df.empty:
        _plot_no_data(output_path, title, "No CDS groups have both ground-truth and generated TE scores.")
        return

    ordered = df.sort_values("best_te_improvement", ascending=False).reset_index(drop=True)
    x = np.arange(len(ordered))
    width = 0.42
    fig_width = max(9.0, min(0.22 * len(ordered), 28.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.bar(
        x - width / 2,
        ordered["mean_te_improvement"],
        width=width,
        color="#0f766e",
        label="Generated mean - GT",
    )
    ax.bar(
        x + width / 2,
        ordered["best_te_improvement"],
        width=width,
        color="#be123c",
        label="Generated best - GT",
    )
    ax.axhline(0, color="#111827", linewidth=1.0)
    _format_cds_axis(ax, ordered["id"].astype(str).tolist())
    ax.set_ylabel("Predicted TE improvement")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_merged_te_improvement(
    df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    value_column: str,
    ylabel: str,
) -> None:
    if df.empty:
        _plot_no_data(output_path, title, "No CDS groups have both ground-truth and generated TE scores.")
        return

    models = sorted(df["model"].unique())
    ncols = min(MODEL_GRID_COLUMNS, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 3.4 * nrows),
        sharey=True,
        squeeze=False,
    )
    for model_idx, model in enumerate(models):
        ax = axes[model_idx // ncols][model_idx % ncols]
        ordered = (
            df[df["model"] == model]
            .sort_values(value_column, ascending=True)
            .reset_index(drop=True)
        )
        x = np.arange(len(ordered))
        values = ordered[value_column]
        colors = np.where(values >= 0, "#22c55e", "#ef4444")
        ax.bar(x, values, color=colors, width=0.9, linewidth=0)
        ax.axhline(0, color="#111827", linewidth=0.9)
        ax.set_title(_display_name(model), fontsize=11)
        ax.set_xlabel("IDs sorted by best improvement")
        if model_idx % ncols == 0:
            ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        _annotate_improvement_summary(ax, ordered, value_column)

    _hide_unused_axes(axes, len(models))
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _annotate_improvement_summary(ax: plt.Axes, df: pd.DataFrame, value_column: str) -> None:
    values = pd.to_numeric(df[value_column], errors="coerce").dropna()
    if values.empty:
        return
    improved_count = int((values > 0).sum())
    best = float(values.max())
    median = float(values.median())
    text = f"Improved: {improved_count}/{len(values)}\nBest: {best:.4f}\nMedian: {median:.4f}"
    ax.text(
        0.02,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )


def _plot_focused_scatter_by_model(
    df: pd.DataFrame,
    x_name: str,
    y_name: str,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    models = sorted(df["model"].unique())
    ncols = min(MODEL_GRID_COLUMNS, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.7 * ncols, 3.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for model_idx, model in enumerate(models):
        ax = axes[model_idx // ncols][model_idx % ncols]
        subset = df[df["model"] == model]
        _draw_cohort_scatter(ax, subset, x_name, y_name)
        _annotate_correlations(ax, subset, x_name, y_name)
        ax.set_title(_display_name(model), fontsize=11)
        ax.set_xlabel(xlabel)
        if model_idx % ncols == 0:
            ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    _hide_unused_axes(axes, len(models))
    _add_shared_legend(fig, axes[0][0])
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.97, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _annotate_correlations(ax: plt.Axes, df: pd.DataFrame, x_name: str, y_name: str) -> None:
    lines = []
    for cohort in (COHORT_GENERATED, COHORT_GT):
        subset = df[df["cohort"] == cohort][[x_name, y_name]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(subset) < 2 or subset[x_name].nunique() < 2 or subset[y_name].nunique() < 2:
            continue
        corr = float(subset[x_name].corr(subset[y_name]))
        lines.append(f"{cohort}: r={corr:.6f}, n={len(subset)}")
    if not lines:
        return
    ax.text(
        0.02,
        0.96,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.88},
    )


def _plot_sequence_diversity_distributions(
    diversity: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    if diversity.empty:
        _plot_no_data(output_path, title, "No sequence diversity records were available.")
        return

    models = sorted(diversity["model"].unique())
    ncols = min(MODEL_GRID_COLUMNS, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 3.45 * nrows),
        sharey=True,
        squeeze=False,
    )

    labels = ["Generated\nvs generated", "Generated\nvs GT"]
    colors = [
        DIVERSITY_COLORS["Generated vs generated"],
        DIVERSITY_COLORS["Generated vs ground truth"],
    ]
    for model_idx, model in enumerate(models):
        ax = axes[model_idx // ncols][model_idx % ncols]
        subset = diversity[diversity["model"] == model]
        values = [
            _finite_values(subset, "mean_generated_pairwise_distance").to_numpy(dtype=float),
            _finite_values(subset, "mean_generated_to_gt_distance").to_numpy(dtype=float),
        ]
        _draw_diversity_violins(ax, values, labels, colors)
        ax.set_title(_display_name(model), fontsize=11)
        if model_idx % ncols == 0:
            ax.set_ylabel("k-mer Jaccard distance")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.22)
        _annotate_diversity_medians(ax, values)

    _hide_unused_axes(axes, len(models))
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_diversity_violins(
    ax: plt.Axes,
    values: list[np.ndarray],
    labels: list[str],
    colors: list[str],
) -> None:
    positions = np.arange(1, len(values) + 1)
    non_empty = [(position, series, color) for position, series, color in zip(positions, values, colors) if len(series)]
    if non_empty:
        parts = ax.violinplot(
            [series for _, series, _ in non_empty],
            positions=[position for position, _, _ in non_empty],
            widths=0.72,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        for body, (_, _, color) in zip(parts["bodies"], non_empty):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.28)
        ax.boxplot(
            [series for _, series, _ in non_empty],
            positions=[position for position, _, _ in non_empty],
            widths=0.22,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "#334155", "linewidth": 1.0},
            medianprops={"color": "#111827", "linewidth": 1.4},
            whiskerprops={"color": "#334155", "linewidth": 0.9},
            capprops={"color": "#334155", "linewidth": 0.9},
        )

    for position, series, color in zip(positions, values, colors):
        if not len(series):
            continue
        x = np.full(len(series), position, dtype=float)
        offsets = _deterministic_offsets(len(series), width=0.18)
        ax.scatter(x + offsets, series, s=6, color=color, alpha=0.18, linewidths=0)

    ax.set_xticks(positions, labels=labels, fontsize=8)


def _deterministic_offsets(count: int, *, width: float) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    repeated = np.arange(count) % 19
    return ((repeated / 18.0) - 0.5) * width


def _annotate_diversity_medians(ax: plt.Axes, values: list[np.ndarray]) -> None:
    lines = []
    labels = ["G-G", "G-GT"]
    for label, series in zip(labels, values):
        if len(series):
            lines.append(f"{label} median: {np.median(series):.3f}")
    if not lines:
        return
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.88},
    )


def _plot_diversity_vs_gt_distance_scatter(
    diversity: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    x_name = "mean_generated_pairwise_distance"
    y_name = "mean_generated_to_gt_distance"
    if diversity.empty or diversity[[x_name, y_name]].dropna().empty:
        _plot_no_data(output_path, title, "No CDS groups have both generated pairwise and generated-to-GT distances.")
        return

    models = sorted(diversity["model"].unique())
    ncols = min(MODEL_GRID_COLUMNS, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 3.8 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    scatter = None
    for model_idx, model in enumerate(models):
        ax = axes[model_idx // ncols][model_idx % ncols]
        subset = diversity[diversity["model"] == model].copy()
        subset[x_name] = pd.to_numeric(subset[x_name], errors="coerce")
        subset[y_name] = pd.to_numeric(subset[y_name], errors="coerce")
        subset["generated_unique_rate"] = pd.to_numeric(subset["generated_unique_rate"], errors="coerce")
        subset = subset.dropna(subset=[x_name, y_name])
        if not subset.empty:
            scatter = ax.scatter(
                subset[x_name],
                subset[y_name],
                c=subset["generated_unique_rate"],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=18,
                alpha=0.55,
                linewidths=0,
            )
            _annotate_diversity_correlation(ax, subset, x_name, y_name)
        ax.plot([0, 1], [0, 1], color="#94a3b8", linewidth=0.9, linestyle="--", alpha=0.8)
        ax.set_title(_display_name(model), fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean generated-vs-generated distance")
        if model_idx % ncols == 0:
            ax.set_ylabel("Mean generated-vs-GT distance")
        ax.grid(alpha=0.22)

    fig.suptitle(title, fontsize=14, y=0.995)
    _hide_unused_axes(axes, len(models))
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.11, top=0.90, wspace=0.18, hspace=0.35)
    if scatter is not None:
        colorbar_axis = fig.add_axes([0.88, 0.18, 0.018, 0.64])
        fig.colorbar(scatter, cax=colorbar_axis, label="Generated unique-sequence rate")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _annotate_diversity_correlation(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_name: str,
    y_name: str,
) -> None:
    if len(df) < 2 or df[x_name].nunique() < 2 or df[y_name].nunique() < 2:
        return
    corr = float(df[x_name].corr(df[y_name]))
    ax.text(
        0.03,
        0.97,
        f"r={corr:.3f}\nn={len(df)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.88},
    )


def _plot_sequence_diversity_heatmap(
    diversity: pd.DataFrame,
    value_column: str,
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
) -> None:
    if diversity.empty or value_column not in diversity.columns:
        _plot_no_data(output_path, title, "No sequence diversity records were available.")
        return

    plot_df = diversity[["model", "cds", value_column]].copy()
    plot_df[value_column] = pd.to_numeric(plot_df[value_column], errors="coerce")
    pivot = plot_df.pivot_table(index="model", columns="cds", values=value_column, aggfunc="mean")
    if pivot.empty or pivot.to_numpy(dtype=float).size == 0 or np.isnan(pivot.to_numpy(dtype=float)).all():
        _plot_no_data(output_path, title, f"No finite values were available for {value_column}.")
        return

    column_order = pivot.mean(axis=0, skipna=True).sort_values(ascending=True).index
    pivot = pivot.loc[sorted(pivot.index), column_order]
    matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))

    fig_width = max(8.5, min(0.025 * pivot.shape[1], 15.0))
    fig_height = max(2.8, 0.45 * pivot.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f1f5f9")
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_yticks(np.arange(len(pivot.index)), labels=[_display_name(model) for model in pivot.index])
    ax.set_xlabel(f"{pivot.shape[1]} CDS groups sorted by mean distance")
    ax.set_ylabel("Model")
    ax.set_title(title)
    if pivot.shape[1] > 1:
        ax.set_xticks([0, pivot.shape[1] - 1], labels=["Low", "High"])
    else:
        ax.set_xticks([0], labels=["CDS"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_cds_axis(ax: plt.Axes, labels: list[str]) -> None:
    ax.set_xlim(-0.75, max(len(labels) - 0.25, 0.25))
    if len(labels) <= 30:
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=90, fontsize=7)
    else:
        tick_count = min(20, len(labels))
        tick_positions = np.linspace(0, len(labels) - 1, tick_count, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(position + 1) for position in tick_positions], fontsize=8)


def _plot_no_data(output_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _short_metric_label(metric_name: str) -> str:
    labels = {
        "utr5_length": "5' length",
        "cds_length": "CDS length",
        "utr3_length": "3' length",
        "utr5_gc_content": "5' GC",
        "cds_gc_content": "CDS GC",
        "utr3_gc_content": "3' GC",
        "ribonn_te": "TE",
    }
    return labels.get(metric_name, metric_name)


def _display_name(value: object) -> str:
    return str(value).replace("_", " ")


def _slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return slug.strip("_") or "value"
