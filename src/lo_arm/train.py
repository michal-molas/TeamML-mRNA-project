import argparse
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from .data import MRNALoArmDataset
from .loss import compute_lo_arm_loss
from .model import LoArmConfig, LoArmTransformer


def _to_device(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


@torch.no_grad()
def validate(model, dataloader, mask_id, device, max_batches=None):
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = _to_device(batch, device)
        metrics = compute_lo_arm_loss(model, batch, mask_id)
        losses.append(float(metrics["negative_elbo"].item()))
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def train(args, device):
    train_dataset = MRNALoArmDataset(
        args.train_csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        k=args.k,
    )
    val_dataset = MRNALoArmDataset(
        args.test_csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        k=args.k,
    )
    print(
        f"[dataset] train={len(train_dataset)} skipped={train_dataset.skipped_count} "
        f"val={len(val_dataset)} skipped={val_dataset.skipped_count}",
        file=sys.stderr,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    config = LoArmConfig(
        vocab_size=train_dataset.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=train_dataset.max_len,
        prefix_len=train_dataset.prefix_len,
        target_len=train_dataset.target_len,
        dropout=args.dropout,
    )
    model = LoArmTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_steps = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            metrics = compute_lo_arm_loss(model, batch, train_dataset.mask_id)
            loss = metrics["loss"]
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running += float(loss.item())
            n_steps += 1
            global_step += 1
            if args.wandb and wandb is not None and global_step % args.log_every == 0:
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/negative_elbo": float(metrics["negative_elbo"].item()),
                        "train/n_previous": metrics["n_previous"],
                    },
                    step=global_step,
                )

        train_loss = running / max(n_steps, 1)
        val_loss = validate(
            model,
            val_loader,
            train_dataset.mask_id,
            device,
            max_batches=args.max_val_batches,
        )
        print(f"Epoch {epoch} | train_loss={train_loss:.4f} val_neg_elbo={val_loss:.4f}")
        if args.wandb and wandb is not None:
            wandb.log({"epoch/train_loss": train_loss, "valid/negative_elbo": val_loss}, step=global_step)

        if args.output_path and val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "tokenizer": {
                        "k": args.k,
                        "vocab": train_dataset.vocab,
                        "mask_id": train_dataset.mask_id,
                    },
                    "data": {
                        "max_utr5_len": args.max_utr5_len,
                        "max_cds_len": args.max_cds_len,
                        "max_utr3_len": args.max_utr3_len,
                    },
                },
                args.output_path,
            )
            print(f"[checkpoint] saved {args.output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_path", default="../../data/pretraining/dataset_with_utr3/train.csv")
    parser.add_argument("--test_csv_path", default="../../data/pretraining/dataset_with_utr3/test.csv")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max_utr5_len", type=int, default=200)
    parser.add_argument("--max_cds_len", type=int, default=500)
    parser.add_argument("--max_utr3_len", type=int, default=200)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_val_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="lo-arm-pretrain")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed")
        wandb.init(project=args.wandb_project, config=vars(args), dir="../../logs")

    train(args, device)

    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
