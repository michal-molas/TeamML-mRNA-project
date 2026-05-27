#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"

MODEL_PATH="${MODEL_PATH:-data/models/lo_arm_smoke.pt}"
OUTPUT_CSV="${OUTPUT_CSV:-data/generated/lo_arm_smoke_samples.csv}"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model checkpoint not found: $MODEL_PATH" >&2
    echo "Run src/lo_arm/run_train.sh first, or set MODEL_PATH=/path/to/checkpoint.pt." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"

"$PYTHON_BIN" -m lo_arm.sample \
    --model_path "$MODEL_PATH" \
    --dataset_csv "${DATASET_CSV:-data/pretraining/small_test.csv}" \
    --output_csv "$OUTPUT_CSV" \
    --samples_per_cds "${SAMPLES_PER_CDS:-1}" \
    --max_samples "${MAX_SAMPLES:-2}" \
    --max_utr5_len "${MAX_UTR5_LEN:-30}" \
    --max_cds_len "${MAX_CDS_LEN:-500}" \
    --max_utr3_len "${MAX_UTR3_LEN:-30}" \
    --greedy_order \
    --greedy_value \
    "$@"
