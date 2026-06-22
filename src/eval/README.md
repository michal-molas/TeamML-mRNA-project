# Evaluation

Evaluation is an explicit three-step process:

1. Generate sequences for each model.
2. Score each model's sequences.
3. Generate plots and comparison tables across all scored models.

Run the commands below from the repository root.

## Directory layout

Each evaluation has one subdirectory per model. Generation and scoring create the
two files consumed by the plotting step:

```text
data/evals/<evaluation>/
  <model>/
    sequences.csv
    scores.csv
  eval_plots/
```

`eval_plots/` is created by step 3. Repeat steps 1 and 2 with another `<model>`
directory to include that model in the comparison.

The examples below use:

```bash
EVAL_DIR=data/evals/my_eval
MODEL=transformer_pretrained
```

## 1. Generate samples

The input CSV must contain:

```text
id,utr5,cds,utr3
```

Generate one ground-truth row and five model samples per input sequence:

```bash
python src/eval/generate_samples.py \
  transformer \
  data/models/utr53_pretrained.pt \
  data/pretraining/small_test.csv \
  "$EVAL_DIR/$MODEL/sequences.csv" \
  --samples_per_cds 5 \
  --max_samples 100
```

`model_type` can be `transformer`, `indigo`, or `loarm`. The generated
`sequences.csv` contains:

```text
id,sample,utr5,cds,utr3
```

For each input row it contains `sample=gt` plus generated rows named
`sample_0`, `sample_1`, and so on.

## 2. Score sequences

[`configs/default.yml`](configs/default.yml) contains reusable scorer settings.
Input and output paths are passed on the command line so the same configuration
can score every model:

```bash
python src/eval/score_sequences.py \
  --config src/eval/configs/default.yml \
  --samples_csv "$EVAL_DIR/$MODEL/sequences.csv" \
  --save_dir "$EVAL_DIR/$MODEL"
```

This writes:

```text
data/evals/my_eval/transformer_pretrained/scores.csv
```

To compare another model, choose a new `MODEL` name and repeat steps 1 and 2.
Every model directory must contain both `sequences.csv` and `scores.csv`.

## 3. Generate plots

Run the plotting entry point once for the evaluation root:

```bash
python src/eval/generate_plots.py "$EVAL_DIR"
```

It discovers every model subdirectory containing both required CSV files and
writes results under `$EVAL_DIR/eval_plots/`:

```text
eval_plots/
  metrics.csv
  summary.csv
  te_improvements.csv
  sequence_diversity.csv
  sequence_diversity_summary.csv
  per_model/
  merged/
```

The outputs include normalized sequence and predicted-TE metrics, cohort
summaries, mean and best TE improvements over ground truth, sequence-diversity
statistics, per-model plots, and merged model-comparison plots.

## SLURM

Edit the configuration block in `src/eval/submit_generate.sub`, including the
evaluation name, model name, checkpoint, input path, and sampling parameters.
Then submit it without command-line arguments:

```bash
./slurm_wrap.py src/eval/submit_generate.sub
```

After generation completes, set the matching evaluation and model names in
`src/eval/submit_eval.sub`, then submit scoring:

```bash
./slurm_wrap.py src/eval/submit_eval.sub
```

Repeat those submissions for each model. Once every scoring job is complete,
run step 3 from the repository root to generate the comparison plots.
