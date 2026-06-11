# LO-ARM for conditional mRNA UTR generation

This package implements a Learning-Order Autoregressive Model (LO-ARM) for
conditional mRNA UTR generation. The model is conditioned on a CDS sequence and
learns both what nucleotide k-mer token to generate and which target position to
generate next.

The implementation is adapted to the existing repo task rather than molecular
graph generation: it samples a UTR layout from an empirical prior, then infills
only the real 5'UTR/3'UTR k-mer slots. Padding is bookkeeping, not a modeled
target value.

## Data layout

The tokenizer follows the existing k-mer convention used by the baseline
Transformer code. For `k=3`, the vocabulary contains all DNA 3-mers plus special
tokens:

- `<PAD>`: padding in the CDS prefix and unused target canvas slots. It is not
  part of the default loss or order-policy action space.
- `<BOS>`: beginning of the conditional prefix.
- `<EOS>`: end of the generated UTR target.
- `<CDS>`: marker before CDS tokens.
- `<UTR5>`: marker before target generation starts.
- `<UTR3>`: separator between 5'UTR and 3'UTR target regions.
- `<MASK>`: LO-ARM sampling mask meaning "this target slot is not generated
  yet".

Each example is represented as a fixed-length prefix plus a fixed-width target
canvas. The target canvas contains only real UTR k-mer slots followed by
padding:

```text
prefix = <BOS>, <CDS>, cds_tokens, <UTR5>, <PAD>...
target = reversed_utr5_tokens, utr3_tokens, <PAD>...
```

Each sample also carries a token-space layout tuple
`(total_len, utr5_len, cds_len, utr3_len)`, a `target_order_mask`, and
`target_region_ids`. Only `target_order_mask=True` slots participate in LO-ARM
ordering and reconstruction. Prefix padding and target-side `<PAD>` slots are
hidden from attention.

5'UTR tokens are stored reversed during training so generation proceeds outward
from the CDS-proximal side, matching the baseline dataset convention. At decode
time the generated 5'UTR segment is reversed back.

## Model flow

`LoArmTransformer` is a non-causal Transformer encoder. At every forward pass it
sees the full CDS prefix plus the current target canvas:

```text
input = prefix + masked_or_partially_unmasked_target
```

The encoder output is sliced to target positions only. Three heads are applied
to those target states:

- `value_head`: predicts a categorical distribution over token values for each
  target slot.
- `order_head`: predicts the model order-policy
  `p_theta(z_i | z_<i, x_z<i)`, i.e. which still-masked target slot should be
  generated next given the partially generated canvas.
- `posterior_head`: predicts the training-time variational posterior
  `q_phi(z_i | z_<i, x)`, which sees the fully observed target and proposes
  latent generation orders.

The model is non-causal because the visible/masked canvas itself controls what
information is available. Generated target slots are visible; ungenerated slots
contain `<MASK>`. Token, position, and region embeddings are summed so the model
knows whether each visible target slot belongs to 5'UTR or 3'UTR.

## Training objective

Training uses the LO-ARM variational objective with a two-sample
REINFORCE leave-one-out (RLOO) estimator. The default objective is the scheduled
alpha-beta ELBO; `--objective_mode elbo` keeps the original ELBO coefficients on
the new PAD-free masks, while `--objective_mode legacy_full_canvas` restores the
old full-canvas baseline for debugging.

For each batch:

1. Run the model on the fully observed target to obtain posterior logits.
2. Sample two complete real-target-slot permutations with Gumbel-top-k.
3. Sample one partial generation depth `n_previous`.
4. Build two partial target canvases by revealing the first `n_previous` slots
   from each sampled permutation and masking all remaining slots with `<MASK>`.
5. Run the model on each partial canvas.
6. For the next generation step, exactly sum over all still-masked real candidate
   slots:
   - the model order-policy log probability,
   - the value log probability of the true target token,
   - the posterior correction term.
7. Use the two sampled paths to form the RLOO autodiff objective.

`compute_lo_arm_loss()` returns a minimization loss plus detached logging
metrics, including stochastic negative ELBO.

## Generation

Generation starts from a CDS prefix and a sampled layout. The layout is drawn
from the checkpoint's empirical training prior, conditioned on the tokenized CDS
length when possible. The target canvas masks real UTR slots and leaves padding
as `<PAD>`:

```text
target = <MASK>...<MASK>, <PAD>...
```

At each step:

1. Run the model on `prefix + target`.
2. Mask out already-filled slots and all PAD slots in the order-policy logits.
3. Choose or sample the next target slot.
4. Choose or sample the token value for that slot.
5. Write the sampled value into the target canvas.

After all real slots are filled, the sampled layout determines the split:
the first `utr5_len` target tokens are decoded as 5'UTR and reversed back, and
the next `utr3_len` tokens are decoded as 3'UTR.

The sampling CLI writes the same CSV shape used elsewhere in the repo:

```text
id,sample,cds,utr5,utr3
```

## Usage

Quick smoke training:

```bash
src/lo_arm/run_train.sh
```

Quick smoke generation, using the checkpoint produced by the smoke training
script:

```bash
src/lo_arm/run_generate.sh
```

The wrappers are intentionally small and can be configured with environment
variables:

```bash
TRAIN_CSV_PATH=data/pretraining/dataset_with_utr3/train.csv \
TEST_CSV_PATH=data/pretraining/dataset_with_utr3/test.csv \
OUTPUT_PATH=data/models/lo_arm_pretrained.pt \
MAX_UTR5_LEN=200 \
MAX_CDS_LEN=500 \
MAX_UTR3_LEN=200 \
N_LAYERS=4 \
D_MODEL=256 \
N_HEADS=8 \
OBJECTIVE_MODE=alpha_beta \
BATCH_SIZE=16 \
EPOCHS=5 \
src/lo_arm/run_train.sh
```

```bash
MODEL_PATH=data/models/lo_arm_pretrained.pt \
DATASET_CSV=data/pretraining/dataset_with_utr3/test.csv \
OUTPUT_CSV=data/generated/lo_arm_samples.csv \
SAMPLES_PER_CDS=3 \
MAX_SAMPLES=100 \
MAX_UTR5_LEN=200 \
MAX_CDS_LEN=500 \
MAX_UTR3_LEN=200 \
ORDER_TOP_P=0.9 \
src/lo_arm/run_generate.sh
```

Direct module entrypoints are also available:

```bash
PYTHONPATH=src venv/bin/python -m lo_arm.train --help
PYTHONPATH=src venv/bin/python -m lo_arm.sample --help
```

For cluster training, `submit.sub` mirrors the distributed SLURM style used by
InDIGO pretraining. It launches four tasks over four GPUs with `srun`; each task
gets a disjoint shard of the shuffled training indices, validation metrics are
reduced across ranks, and only rank 0 writes checkpoints and logs to W&B.
The same training entrypoint also works with `torchrun`; it uses NCCL when CUDA
is available and falls back to Gloo for CPU-only distributed smoke tests.

## Future directions

- Add a factorized novelty layout prior behind the same layout-prior interface.
- Load tokenizer and data limits from the checkpoint during sampling by default,
  and validate explicit CLI overrides against `model.config`.
- Add deterministic or averaged validation so checkpoint selection is less noisy
  than a single stochastic ELBO estimate.
- Expand order diagnostics, such as selected-slot histograms, UTR5/UTR3
  generation timing, and entropy of selected slots.
- Add larger training presets and optional DDP once one-GPU training behavior is
  stable.
- Add benchmarking against the baseline Transformer and InDIGO in a separate
  pass.
- Add RiboNN-guided fine-tuning after the base LO-ARM objective and sampler are
  stable.

## Current verification

The focused test suite covers vocabulary invariants, layout-prior sampling,
target encode/decode, Gumbel-top-k permutations, partial masks, objective
toggles, finite differentiable loss, target-pad attention behavior, and sampling
smoke checks:

```bash
venv/bin/python -m pytest tests/test_lo_arm.py
```
