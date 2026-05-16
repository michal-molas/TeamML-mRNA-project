import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models import MRNACsvDataset, MRNATransformer
from utils import load_pretrained_weights
from ribonn_utils import (
    load_ribonn,
    RIBONN_CONFIG,
    build_ribonn_input,
    ribonn_predict_using_nested_cross_validation_models,
)


def train(args, device):

    dataset = MRNACsvDataset(
        csv_path=args.csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
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

    model = MRNATransformer(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.n_layers,
        max_len=dataset.max_len,
    ).to(device)

    load_pretrained_weights(model, args.pretrained_path, device)

    # ribonn_model, ribonn_max_len = load_ribonn(args.ribonn_weights, device)
    _, ribonn_max_len = load_ribonn(args.ribonn_weights, device)

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
                ribonn_max_len,
                label_codons=RIBONN_CONFIG["label_codons"],
            )

            all_predictions = ribonn_predict_using_nested_cross_validation_models(
                args=args,
                device=device,
                ribonn_input=ribonn_input,
                batch_width=lm_logits.shape[0],
            )
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
                    ribonn_max_len,
                    label_codons=RIBONN_CONFIG["label_codons"],
                )

                all_predictions = ribonn_predict_using_nested_cross_validation_models(
                    args=args,
                    device=device,
                    ribonn_input=ribonn_input,
                    batch_width=lm_logits.shape[0],
                )
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

        def _save_checkpoint(checkpoint_name = ""):
            torch.save({"model_state_dict": model.state_dict()}, args.output_path + checkpoint_name)
            print(f"[checkpoint] saved → {args.output_path}")

        if val_lm < best_val_lm:
            best_val_lm = val_lm
            _save_checkpoint()
        elif epoch % 20:
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
        "--ribonn_weights",
        type=str,
        default=None,
        help="Path to a RiboNN state_dict.pth weight file",
    )
    parser.add_argument(
        "--ribonn_weights_folder",
        type=str,
        default=None,
        help="Path to a RiboNN state_dict.pth weight file",
    )
    parser.add_argument(
        "--lambda_ribonn",
        type=float,
        default=1.0,
        help="Weight of RiboNN loss relative to LM loss",
    )

    parser.add_argument("--n_layers", type=int, default=4)
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
