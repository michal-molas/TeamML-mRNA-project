import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
RIBONN_ROOT = REPO_ROOT / "RiboNN"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RIBONN_ROOT))
from RiboNN.src.model import RiboNN

RIBONN_MAX_TX_LEN = 1_381 + 11_937  # 13318

# len_after_conv: sequence length after all 10 conv+pool layers for a 13318-length input.
# Derivation:
#   1. initial_conv(k=5,p=0): 13318 -> 13314
#   2. 10 × (conv(k=5,p=0) + maxpool(2,2)): 13314 -> 13310 -> 6655 -> 6651 -> 3325 -> 3321 -> 1660 -> 1656 -> 828 -> 824 -> 412 -> 408 -> 204 -> 200 -> 100 -> 96 -> 48 -> 44 -> 22 -> 18 -> 9
RIBONN_LEN_AFTER_CONV = 9

RIBONN_CONFIG = dict(
    with_NAs=False, # conf.yml
    split_utr5_cds_utr3_channels=False, # conf.yml
    label_codons=True, # conf.yml
    label_utr5=False, # conf.yml
    label_utr3=False, # conf.yml
    label_splice_sites=False, # conf.yml
    label_up_probs=False, # conf.yml
    filters=64, # conf.yml
    conv_stride=1, # conf.yml
    conv_padding=0, # conf.yml
    ln_epsilon=0.007, # conf.yml
    dropout=0.3, # conf.yml
    residual=False, # conf.yml
    activation = "relu", # Can be also silu/leakyrelu
    kernel_size=5, # conf.yml
    num_conv_layers=10, # conf.yml
    len_after_conv=RIBONN_LEN_AFTER_CONV,
    num_targets=78,  # 78 for human, 68 for mouse
    max_shift=0, # conf.yml
    symmetric_shift=True, # conf.yml
)


def load_ribonn(weights_path, device, verbose=False):
    """Load frozen RiboNN weights from the submodule. Returns (model, RIBONN_MAX_TX_LEN)."""

    config = dict(RIBONN_CONFIG)
    model = RiboNN(**config)

    state_dict = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if verbose:
        print(
            f"[ribonn] loaded {weights_path}  missing={len(missing)}  unexpected={len(unexpected)}"
        )

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, RIBONN_MAX_TX_LEN


def build_ribonn_input(
    lm_logits,
    input_ids,
    utr5_lens,
    cds_lens,
    utr3_lens,
    ribonn_max_len,
    label_codons=True,
):
    """
    Convert LM logits into a RiboNN-compatible tensor via Gumbel-softmax.

    Vocab order A=0,U/T=1,C=2,G=3 matches RiboNN channels.
    RiboNN input layout: [ utr5 | cds | utr3 | padding ]  (zero-padded to ribonn_max_len)
    UTR5 is stored reversed in the LM; we flip it back here.

    Returns: (N, num_channels, ribonn_max_len)  — channels: A,U/T,C,G [+ codon_label]
    """
    batch_size = lm_logits.size(0)
    device = lm_logits.device
    num_channels = 5 if label_codons else 4
    out = torch.zeros(batch_size, num_channels, ribonn_max_len, device=device)

    # Nucleotide channels 0-3 already match RiboNN, but we have to drop special-token logits
    soft_nt = F.gumbel_softmax(lm_logits)[:, :, :4]

    for i in range(batch_size):
        utr5_len = int(utr5_lens[i].item())
        cds_len  = int(cds_lens[i].item())
        utr3_len = int(utr3_lens[i].item())

        # UTR5
        utr5_start = 2 + cds_len + 1 # after <BOS>, <CDS>, CDS tokens and <UTR5>
        utr5_end = utr5_start + utr5_len

        utr5_out_start = 0
        utr5_out_end = utr5_len
        out[i, :4, utr5_out_start:utr5_out_end] = soft_nt[i, utr5_start:utr5_end].flip(0).T

        # CDS (Here we use one-hot encoding instead of Gumbel-softmax)
        cds_start = 2 # after <BOS>, <CDS>
        cds_end = cds_start + cds_len

        cds_tokens = input_ids[i, cds_start:cds_end]
        cds_out_start = utr5_len
        cds_out_end = utr5_len + cds_len
        out[i, :4, cds_out_start:cds_out_end] = F.one_hot(cds_tokens, 4).float().T

        # UTR3
        utr3_start = 2 + cds_len + 1 + utr5_len + 1 # after <BOS>, <CDS>, CDS tokens, <UTR5>, UTR5 tokens and <UTR3> token 
        utr3_end = utr3_start + utr3_len

        utr3_out_start = utr5_len + cds_len
        utr3_out_end = utr5_len + cds_len + utr3_len
        out[i, :4, utr3_out_start:utr3_out_end] = soft_nt[i, utr3_start:utr3_end].T

        # Codon-label channel
        if label_codons:
            # Label every first nucleotide of a codon (every 3rd position) in CDS (UTRs should not be labeled)
            for codon_pos in range(cds_out_start, cds_out_end, 3):
                if codon_pos < cds_out_end:
                    out[i, 4, codon_pos] = 1.0

    return out  # (N, num_channels, ribonn_max_len)


def ribonn_input_from_string(
    utr5: str,
    cds: str,
    utr3: str,
    ribonn_max_len: int,
    label_codons: bool = True,
) -> torch.Tensor:
    """
    Convert sequence strings into a RiboNN-compatible tensor.

    Vocab/channel order:
        A = 0
        U/T = 1
        C = 2
        G = 3

    RiboNN input layout:
        [ utr5 | cds | utr3 | padding ]

    Returns:
        Tensor of shape (num_channels, ribonn_max_len)
        where num_channels = 4 or 5 if label_codons=True.
    """
    seq = utr5 + cds + utr3
    total_len = len(seq)

    if total_len > ribonn_max_len:
        raise ValueError(
            f"Sequence length {total_len} exceeds ribonn_max_len={ribonn_max_len}. "
            f"Lengths: utr5={len(utr5)}, cds={len(cds)}, utr3={len(utr3)}"
        )

    num_channels = 5 if label_codons else 4
    out = torch.zeros(num_channels, ribonn_max_len, dtype=torch.float32)

    nt_to_idx = {
        "A": 0,
        "U": 1,
        "T": 1,
        "C": 2,
        "G": 3,
    }

    for pos, nt in enumerate(seq.upper()):
        try:
            channel = nt_to_idx[nt]
        except KeyError:
            raise ValueError(
                f"Invalid nucleotide {nt!r} at position {pos}. "
                "Allowed nucleotides: A, U, T, C, G."
            )

        out[channel, pos] = 1.0

    if label_codons:
        cds_start = len(utr5)
        cds_end = len(utr5) + len(cds)

        # Label every first nucleotide of a codon inside CDS only
        for codon_pos in range(cds_start, cds_end, 3):
            out[4, codon_pos] = 1.0

    return out


def ribonn_predict_using_nested_cross_validation_models(args, device, ribonn_input, batch_width):
    ## RiboNN.src.predict.predict_using_nested_cross_validation_models() ##
    RIBONN_COLUMNS = 78
    run_df = pd.read_csv(args.ribonn_weights_folder + '/runs.csv') 
    all_predictions = torch.zeros((batch_width, RIBONN_COLUMNS), device=device)
    prediction_num = 0
    for test_fold in np.sort(run_df["params.test_fold"].unique()):
        test_fold_str = str(test_fold)
        sub_run_df = run_df.query(
            "`params.test_fold` == @test_fold_str or `params.test_fold` == @test_fold"
        ).reset_index(drop=True)

        ## RiboNN.src.predict.predict_using_models_trained_in_one_fold() ##
        top_k_models_to_use = getattr(args, "top_k_models_to_use", 5)
        sub_run_df = sub_run_df.sort_values("metrics.val_r2", ascending=False).head(top_k_models_to_use)

        for run_id in sub_run_df.run_id:
            # Create a new model
            local_state_dict_path = f"{args.ribonn_weights_folder}/{run_id}/state_dict.pth"
            ribonn_model, _ = load_ribonn(local_state_dict_path, device)
            all_predictions += ribonn_model(ribonn_input)
            prediction_num += 1

    all_predictions /= prediction_num

    return all_predictions
