# LO-ARM for conditional mRNA UTR generation

This package implements a Learning-Order Autoregressive Model (LO-ARM) for
conditional mRNA UTR generation. The model is conditioned on a CDS sequence and
learns both what nucleotide k-mer token to generate and which target position to
generate next.

The implementation is adapted to the existing repo task rather than molecular
graph generation: it generates a fixed target canvas containing the reversed
5'UTR, optional 3'UTR content, and terminal/padding tokens.

## Data layout

The tokenizer follows the existing k-mer convention used by the baseline
Transformer code. For `k=3`, the vocabulary contains all DNA 3-mers plus special
tokens:

- `<PAD>`: learned post-EOS padding value in the target canvas, and padding in
  the CDS prefix.
- `<BOS>`: beginning of the conditional prefix.
- `<EOS>`: end of the generated UTR target.
- `<CDS>`: marker before CDS tokens.
- `<UTR5>`: marker before target generation starts.
- `<UTR3>`: separator between 5'UTR and 3'UTR target regions.
- `<MASK>`: LO-ARM sampling mask meaning "this target slot is not generated
  yet".

Each example is represented as a fixed-length prefix plus a fixed-length target:

```text
prefix = <BOS>, <CDS>, cds_tokens, <UTR5>, <PAD>...
target = reversed_utr5_tokens, <UTR3>, utr3_tokens, <EOS>, <PAD>...
```

Only target slots participate in LO-ARM ordering. Prefix padding is hidden from
attention with `prefix_padding_mask`; target-side `<PAD>` tokens are not hidden,
because the model must learn where post-EOS padding belongs.

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
contain `<MASK>`.

## Training objective

Training uses the LO-ARM variational objective with a two-sample
REINFORCE leave-one-out (RLOO) estimator.

For each batch:

1. Run the model on the fully observed target to obtain posterior logits.
2. Sample two complete target-slot permutations with Gumbel-top-k.
3. Sample one partial generation depth `n_previous`.
4. Build two partial target canvases by revealing the first `n_previous` slots
   from each sampled permutation and masking all remaining slots with `<MASK>`.
5. Run the model on each partial canvas.
6. For the next generation step, exactly sum over all still-masked candidate
   slots:
   - the model order-policy log probability,
   - the value log probability of the true target token,
   - the posterior correction term.
7. Use the two sampled paths to form the RLOO autodiff objective.

`compute_lo_arm_loss()` returns a minimization loss plus detached logging
metrics, including stochastic negative ELBO.

## Generation

Generation starts from a CDS prefix and an all-`<MASK>` target canvas:

```text
target = <MASK>, <MASK>, ..., <MASK>
```

At each step:

1. Run the model on `prefix + target`.
2. Mask out already-filled target slots in the order-policy logits.
3. Choose or sample the next target slot.
4. Choose or sample the token value for that slot.
5. Write the sampled value into the target canvas.

After all slots are filled, the target is decoded by:

1. Cropping at the first `<EOS>`.
2. Splitting at `<UTR3>`.
3. Detokenizing both regions.
4. Reversing the 5'UTR region back to normal orientation.

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
src/lo_arm/run_generate.sh
```

Direct module entrypoints are also available:

```bash
PYTHONPATH=src venv/bin/python -m lo_arm.train --help
PYTHONPATH=src venv/bin/python -m lo_arm.sample --help
```

For cluster training, `submit.sub` mirrors the existing one-GPU SLURM style used
elsewhere in the repository.

## Future directions

- Load tokenizer and data limits from the checkpoint during sampling by default,
  and validate explicit CLI overrides against `model.config`.
- Mask impossible target values during training and generation: allow k-mers,
  `<UTR3>`, `<EOS>`, and `<PAD>`; disallow `<BOS>`, `<CDS>`, `<UTR5>`, and
  `<MASK>`.
- Add deterministic or averaged validation so checkpoint selection is less noisy
  than a single stochastic ELBO estimate.
- Add order diagnostics, such as selected-slot histograms, UTR5/UTR3/PAD
  generation timing, and entropy of selected slots.
- Add larger training presets and optional DDP once one-GPU training behavior is
  stable.
- Add benchmarking against the baseline Transformer and InDIGO in a separate
  pass.
- Add RiboNN-guided fine-tuning after the base LO-ARM objective and sampler are
  stable.

## Current verification

The focused test suite covers vocabulary invariants, target encode/decode,
Gumbel-top-k permutations, partial masks, finite differentiable loss, target-pad
attention behavior, and sampling smoke checks:

```bash
venv/bin/python -m pytest tests/test_lo_arm.py
```
