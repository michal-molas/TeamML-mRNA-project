# Eval

Local commands below are run from the repository root unless noted otherwise.

## 1. Generate samples

Input CSV must have these columns:

```text
id,utr5,cds,utr3
```

Generate an eval-ready CSV:

```bash
python src/eval/generate_samples.py transformer CHECKPOINT.pt input.csv data/generated/samples.csv \
  --samples_per_cds 5 \
  --max_samples 100
```

`model_type` can be `transformer`, `indigo`, or `loarm`.

The output CSV has:

```text
id,sample,utr5,cds,utr3
```

It contains one ground-truth row per input sequence with `sample=gt`, plus generated rows like `sample_0`, `sample_1`, etc.

### Generate on a cluster

Use the SLURM script:

```bash
cd src/eval
./slurm_wrap.py sbatch submit.sub
```

Before submitting, edit the generation command in `submit.sub`, for example:

```bash
srun python3 generate_samples.py \
  transformer \
  ../../data/models/utr53_pretrained.pt \
  ../../data/pretraining/small_test.csv \
  ../../data/generated/samples.csv \
  --samples_per_cds 3 \
  --max_samples 20
```

The paths above are relative to `src/eval`, because the job is submitted from that directory.

## 2. Score with RiboNN

Create or edit a config YAML, for example `src/eval/configs/ribonn.yml`:

```yaml
samples_csv: data/generated/samples.csv
eval_dir: data/evals/my_eval

scorers:
  ribonn:
    weights_folder: RiboNN/models/human
    top_k_models_to_use: 5
```

Run scoring:

```bash
python src/eval/score_sequences.py --config src/eval/configs/ribonn.yml
```

This writes:

```text
data/evals/my_eval/scores.csv
```

### Default config example

The checked-in default config is `src/eval/configs/default.yml`. It already contains `samples_csv`, `eval_dir`, `string_statistics`, and `ribonn` settings.

To run scoring and aggregation with it:

```bash
cd src/eval
python run_eval.py --config configs/default.yml
```

This writes results to the `eval_dir` set in `configs/default.yml`.

## 3. Aggregate scores

Run:

```bash
python src/eval/calculate_statistics.py \
  --config src/eval/configs/ribonn.yml \
  --scores_csv data/evals/my_eval/scores.csv \
  --output_dir data/evals/my_eval
```

This writes aggregate CSV files such as:

```text
data/evals/my_eval/gt_global_stats.csv
data/evals/my_eval/generated_global_stats.csv
data/evals/my_eval/gt_cds_stats.csv
data/evals/my_eval/generated_cds_stats.csv
```
