import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from RiboNN.src.model import RiboNN
from models import MRNACsvDataset, MRNATransformer

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
    device="cuda",
)


def load_ribonn(weights_path, device):
    """Load frozen RiboNN weights from the submodule. Returns (model, RIBONN_MAX_TX_LEN)."""

    config = dict(RIBONN_CONFIG)
    model = RiboNN(**config)

    state_dict = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
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
    tau,
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
    soft_nt = F.gumbel_softmax(lm_logits, tau=tau, hard=False)[:, :, :4]

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


def load_pretrained_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded {checkpoint_path}  missing={len(missing)}  unexpected={len(unexpected)}"
    )


def train(args, device):
    full_dataset = MRNACsvDataset(
        csv_path=args.csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
    )

    val_size = max(1, int(len(full_dataset) * args.val_fraction))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MRNATransformer(
        vocab_size=full_dataset.vocab_size,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.n_layers,
        max_len=full_dataset.max_len,
    ).to(device)

    if args.pretrained_path:
        load_pretrained_weights(model, args.pretrained_path, device)

    # ── Load frozen RiboNN ──────────────────────────────────────────────────
    ribonn_model = None
    ribonn_max_len = None
    if args.ribonn_weights:
        ribonn_model, ribonn_max_len = load_ribonn(args.ribonn_weights, device)
        print(f"[ribonn] frozen scorer active  max_tx_len={ribonn_max_len}")
    else:
        print("[ribonn] --ribonn_weights not set; RiboNN loss disabled")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    best_val_lm = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for step, batch in tqdm(list(enumerate(train_loader))):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            loss_mask = batch["loss_mask"]
            padding_mask = batch["padding_mask"]
            utr5_lens = batch["utr5_len"]
            cds_lens = batch["cds_len"]
            utr3_lens = batch["utr3_len"]

            optimizer.zero_grad()
            lm_logits = model(input_ids, padding_mask=padding_mask)

            # LM loss: masked cross-entropy on generated tokens (same as pretrain)
            lm_loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                target_ids.view(-1),
                reduction="none",
            )
            lm_loss = (lm_loss * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)

            # RiboNN loss: push generated sequences toward higher TE
            ribonn_loss = torch.tensor(0.0, device=device)
            ribonn_input = build_ribonn_input(
                lm_logits,
                input_ids,
                utr5_lens,
                cds_lens,
                utr3_lens,
                ribonn_max_len,
                args.gumbel_tau,
                label_codons=RIBONN_CONFIG["label_codons"],
            )
            te_pred = ribonn_model(ribonn_input)  # (N, 1)
            ribonn_loss = -te_pred.mean()  # maximise TE

            loss = lm_loss + args.lambda_ribonn * ribonn_loss
            loss.backward()
            optimizer.step()

            if args.wandb and step % 10 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lm_loss": lm_loss.item(),
                        "train/ribonn_loss": ribonn_loss.item(),
                    }
                )

        # ── Validation (LM loss only — RiboNN scoring is eval-time) ────────
        model.eval()
        val_lm_sum = val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                target_ids = batch["target_ids"]
                loss_mask = batch["loss_mask"]
                padding_mask = batch["padding_mask"]

                lm_logits = model(input_ids, padding_mask=padding_mask)
                lm = F.cross_entropy(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    target_ids.view(-1),
                    reduction="none",
                )
                lm = (lm * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)
                val_lm_sum += lm.item()
                val_n += 1

        val_lm = val_lm_sum / max(val_n, 1)

        print(
            f"Epoch {epoch} | train_lm={lm_loss.item():.4f} "
            f"train_ribonn={ribonn_loss.item():.4f}  val_lm={val_lm:.4f}"
        )

        if args.wandb:
            wandb.log({"val/lm_loss": val_lm, "epoch": epoch})

        if val_lm < best_val_lm:
            best_val_lm = val_lm
            torch.save(model.state_dict(), args.output_path)
            print(f"[checkpoint] saved → {args.output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, default="../../data/finetuning/ribonn/dataset.csv"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrain.py checkpoint to initialise the LM backbone",
    )
    parser.add_argument("--output_path", type=str, default="finetune_best.pt")
    # RiboNN differentiable scorer
    parser.add_argument(
        "--ribonn_weights",
        type=str,
        default=None,
        help="Path to a RiboNN state_dict.pth weight file",
    )
    parser.add_argument(
        "--lambda_ribonn",
        type=float,
        default=1.0,
        help="Weight of RiboNN loss (negative TE) relative to LM loss",
    )
    parser.add_argument(
        "--gumbel_tau",
        type=float,
        default=1.0,
        help="Gumbel-softmax temperature; lower = sharper / more discrete",
    )
    # Model architecture
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_utr5_len", type=int, default=128)
    parser.add_argument("--max_cds_len", type=int, default=512)
    parser.add_argument("--max_utr3_len", type=int, default=128)
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.init(project="teamml-ribonn-finetune", config=vars(args))

    train(args, device)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
