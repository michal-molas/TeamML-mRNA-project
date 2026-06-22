import argparse
import sys
from dotenv import load_dotenv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from ..common import (
    MRNACsvDataset,
    RIBONN_CONFIG,
    RIBONN_MAX_TX_LEN,
    RIBONN_MAX_UTR5_LEN,
    RiboNNEnsemble,
    load_checkpoint,
    save_checkpoint,
)
from .models import MRNATransformer

def build_ribonn_input(
    lm_logits,
    input_ids,
    utr5_lens,
    cds_lens,
    utr3_lens,
    ribonn_max_len,
    label_codons=True,
    ribonn_max_utr5_len=RIBONN_MAX_UTR5_LEN,
):
    """
    Convert LM logits into a RiboNN-compatible tensor via Gumbel-softmax.

    Vocab order A=0,U/T=1,C=2,G=3 matches RiboNN channels.
    RiboNN input layout for pretrained models:
        [ left padding + utr5 | cds | utr3 + right padding ]
    The CDS start is aligned at ribonn_max_utr5_len.
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

        if utr5_len > ribonn_max_utr5_len:
            raise ValueError(
                f"5' UTR length {utr5_len} exceeds ribonn_max_utr5_len={ribonn_max_utr5_len}."
            )
        ribonn_max_cds_utr3_len = ribonn_max_len - ribonn_max_utr5_len
        if cds_len + utr3_len > ribonn_max_cds_utr3_len:
            raise ValueError(
                f"Combined CDS and 3' UTR length {cds_len + utr3_len} exceeds "
                f"ribonn_max_cds_utr3_len={ribonn_max_cds_utr3_len}."
            )

        # UTR5
        utr5_start = 2 + cds_len + 1 # after <BOS>, <CDS>, CDS tokens and <UTR5>
        utr5_end = utr5_start + utr5_len

        utr5_out_start = ribonn_max_utr5_len - utr5_len
        utr5_out_end = ribonn_max_utr5_len
        out[i, :4, utr5_out_start:utr5_out_end] = soft_nt[i, utr5_start:utr5_end].flip(0).T

        # CDS (Here we use one-hot encoding instead of Gumbel-softmax)
        cds_start = 2 # after <BOS>, <CDS>
        cds_end = cds_start + cds_len

        cds_tokens = input_ids[i, cds_start:cds_end]
        cds_out_start = ribonn_max_utr5_len
        cds_out_end = ribonn_max_utr5_len + cds_len
        out[i, :4, cds_out_start:cds_out_end] = F.one_hot(cds_tokens, 4).float().T

        # UTR3
        utr3_start = 2 + cds_len + 1 + utr5_len + 1 # after <BOS>, <CDS>, CDS tokens, <UTR5>, UTR5 tokens and <UTR3> token 
        utr3_end = utr3_start + utr3_len

        utr3_out_start = ribonn_max_utr5_len + cds_len
        utr3_out_end = ribonn_max_utr5_len + cds_len + utr3_len
        out[i, :4, utr3_out_start:utr3_out_end] = soft_nt[i, utr3_start:utr3_end].T

        # Codon-label channel
        if label_codons:
            # Label every first nucleotide of a codon (every 3rd position) in CDS (UTRs should not be labeled)
            for codon_pos in range(cds_out_start, cds_out_end, 3):
                if codon_pos < cds_out_end:
                    out[i, 4, codon_pos] = 1.0

    return out  # (N, num_channels, ribonn_max_len)


def load_pretrained_weights(model, checkpoint_path, device):
    n_heads = model.transformer.layers[0].self_attn.num_heads
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=device,
        expected_model_type="transformer",
        legacy_config={"n_heads": n_heads},
    )
    missing, unexpected = model.load_state_dict(
        checkpoint.model_state_dict, strict=True
    )
    print(
        f"Loaded {checkpoint_path} format={checkpoint.format_version} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )

def train(args, device):
    pretrained_checkpoint = None
    if args.pretrained_path:
        pretrained_checkpoint = load_checkpoint(
            args.pretrained_path,
            map_location=device,
            expected_model_type="transformer",
            legacy_config={"n_heads": args.n_heads},
        )
    tokenizer_config = (
        pretrained_checkpoint.tokenizer_config if pretrained_checkpoint else {}
    )
    dataset_k = tokenizer_config.get("k", 1)
    only_utr5 = tokenizer_config.get("only_utr5", False)
    if dataset_k != 1:
        raise ValueError(
            "RiboNN fine-tuning currently requires a nucleotide tokenizer (k=1); "
            "k-mer expansion must be implemented before using this checkpoint"
        )

    dataset = MRNACsvDataset(
        csv_path=args.csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        k=dataset_k,
        only_utr5=only_utr5,
    )

    dataset_size = len(dataset)

    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train dataset length: {len(train_dataset)}", file=sys.stderr)
    print(f"Validation dataset length: {len(val_dataset)}", file=sys.stderr)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model_config = (
        dict(pretrained_checkpoint.model_config)
        if pretrained_checkpoint
        else {
            "vocab_size": dataset.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "num_layers": args.n_layers,
            "max_len": dataset.max_len,
        }
    )
    if dataset.vocab_size != model_config["vocab_size"]:
        raise ValueError(
            f"Fine-tuning dataset vocab_size={dataset.vocab_size} does not match "
            f"checkpoint vocab_size={model_config['vocab_size']}"
        )
    if dataset.max_len > model_config["max_len"]:
        raise ValueError(
            f"Fine-tuning dataset max_len={dataset.max_len} exceeds "
            f"checkpoint max_len={model_config['max_len']}"
        )
    model = MRNATransformer(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        nhead=model_config.get("n_heads", model_config.get("nhead")),
        num_layers=model_config["num_layers"],
        max_len=model_config["max_len"],
    ).to(device)
    if pretrained_checkpoint:
        model.load_state_dict(pretrained_checkpoint.model_state_dict, strict=True)

    if not args.ribonn_weights_folder:
        raise ValueError("--ribonn_weights_folder must be set for RiboNN fine-tuning")
    ribonn_ensemble = RiboNNEnsemble(
        args.ribonn_weights_folder,
        device=device,
        top_k=args.ribonn_top_k,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    global_step = 0
    best_val_lm = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            loss_mask = batch["loss_mask"]
            padding_mask = batch["padding_mask"]
            utr5_lens = batch["utr5_len"]
            cds_lens = batch["cds_len"]
            utr3_lens = batch["utr3_len"]
            te_label = batch["te_label"]

            # Devalue sequences with low TE label
            sigmoid_arg = 10 * te_label - 1 
            lm_loss_mask = torch.nn.Sigmoid()(sigmoid_arg)

            optimizer.zero_grad()
            lm_logits = model(input_ids, padding_mask=padding_mask)

            # LM loss
            outputs_flat = lm_logits.view(-1, lm_logits.size(-1))
            targets_flat = target_ids.view(-1)
            
            lm_loss = F.cross_entropy(outputs_flat, targets_flat, reduction='none')

            loss_mask *= lm_loss_mask.view((-1, 1))
            lm_loss = lm_loss * loss_mask.view(-1)
            lm_loss = lm_loss.sum() / (loss_mask.sum() + 1e-8)

            # RiboNN loss
            ribonn_loss = torch.tensor(0.0, device=device)
            ribonn_input = build_ribonn_input(
                lm_logits,
                input_ids,
                utr5_lens,
                cds_lens,
                utr3_lens,
                RIBONN_MAX_TX_LEN,
                label_codons=RIBONN_CONFIG["label_codons"],
            )

            all_predictions = ribonn_ensemble(ribonn_input)
            # TODO: How to compare the TE to the label? 
            #       Can our model output results better than the data (then maybe relu)?
            #       Maybe we should just try to maximize the TE and ignore the label?
            ribonn_loss = F.mse_loss(all_predictions.mean(dim=-1), te_label)

            loss = lm_loss + args.lambda_ribonn * ribonn_loss
            loss.backward()
            optimizer.step()

            global_step += 1

            if args.wandb and step % 100 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lm_loss": lm_loss.item(),
                        "train/ribonn_loss": ribonn_loss.item(),
                    },
                    step=global_step,
                )

        model.eval()
        val_loss_sum = 0
        val_te_sum = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                target_ids = batch["target_ids"]
                loss_mask = batch["loss_mask"]
                padding_mask = batch["padding_mask"]
                utr5_lens = batch["utr5_len"]
                cds_lens = batch["cds_len"]
                utr3_lens = batch["utr3_len"]

                lm_logits = model(input_ids, padding_mask=padding_mask)
                lm = F.cross_entropy(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    target_ids.view(-1),
                    reduction="none",
                )
                lm = (lm * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)
                val_loss_sum += lm.item()

                discrete_logits = F.one_hot(lm_logits.argmax(-1), num_classes=lm_logits.size(-1)).float()
                ribonn_input = build_ribonn_input(
                    discrete_logits,
                    input_ids,
                    utr5_lens,
                    cds_lens,
                    utr3_lens,
                    RIBONN_MAX_TX_LEN,
                    label_codons=RIBONN_CONFIG["label_codons"],
                )

                all_predictions = ribonn_ensemble(ribonn_input)
                val_te_sum += all_predictions.mean().item()
                val_n += 1

        val_lm = val_loss_sum / val_n
        val_te = val_te_sum / val_n

        print(
            # f"Epoch {epoch} | train_lm={lm_loss.item():.4f} "
            # f"train_ribonn={ribonn_loss.item():.4f}  val_lm={val_lm:.4f}  val_te={val_te:.4f}"
            f"Epoch {epoch}"
            f"val_lm={val_lm:.4f}  val_te={val_te:.4f}"
        )

        if args.wandb:
            wandb.log({"val/lm_loss": val_lm, "val/te": val_te}, step=global_step)

        def _save_checkpoint(checkpoint_name=""):
            output_path = args.output_path + checkpoint_name
            save_checkpoint(
                output_path,
                model_type="transformer",
                model_config=model_config,
                model_state_dict=model.state_dict(),
                tokenizer_config={
                    "k": dataset.tokenizer.k,
                    "only_utr5": dataset.tokenizer.only_utr5,
                    "include_u_alias": dataset.tokenizer.include_u_alias,
                },
                data_config={
                    "max_utr5_len": args.max_utr5_len,
                    "max_cds_len": args.max_cds_len,
                    "max_utr3_len": args.max_utr3_len,
                },
                training={
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_lm,
                    "fine_tuned_with_ribonn": True,
                    "lambda_ribonn": args.lambda_ribonn,
                },
            )
            print(f"[checkpoint] saved → {output_path}")

        if args.output_path and val_lm < best_val_lm:
            best_val_lm = val_lm
            _save_checkpoint()
        elif args.output_path and epoch % 20:
            _save_checkpoint(f"_epoch{epoch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, default="../../data/finetuning/ribonn/dataset.csv"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrain.py checkpoint",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--ribonn_weights_folder",
        type=str,
        default=None,
        help="Path to a RiboNN runs.csv weights folder",
    )
    parser.add_argument("--ribonn_top_k", type=int, default=5)
    parser.add_argument(
        "--lambda_ribonn",
        type=float,
        default=1.0,
        help="Weight of RiboNN loss relative to LM loss",
    )

    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_utr5_len", type=int, default=200)
    parser.add_argument("--max_cds_len", type=int, default=500)
    parser.add_argument("--max_utr3_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="teamml-ribonn-finetune")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dotenv()

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            dir='../../logs',
        )

    train(args, device)

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
