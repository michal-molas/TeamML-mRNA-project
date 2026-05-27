#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"

mkdir -p "$(dirname "${OUTPUT_PATH:-data/models/lo_arm_smoke.pt}")"

"$PYTHON_BIN" -m lo_arm.train \
    --train_csv_path "${TRAIN_CSV_PATH:-data/pretraining/small_test.csv}" \
    --test_csv_path "${TEST_CSV_PATH:-data/pretraining/small_test.csv}" \
    --output_path "${OUTPUT_PATH:-data/models/lo_arm_smoke.pt}" \
    --max_utr5_len "${MAX_UTR5_LEN:-30}" \
    --max_cds_len "${MAX_CDS_LEN:-500}" \
    --max_utr3_len "${MAX_UTR3_LEN:-30}" \
    --n_layers "${N_LAYERS:-1}" \
    --d_model "${D_MODEL:-16}" \
    --n_heads "${N_HEADS:-4}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --epochs "${EPOCHS:-1}" \
    --max_val_batches "${MAX_VAL_BATCHES:-1}" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    "$@"
