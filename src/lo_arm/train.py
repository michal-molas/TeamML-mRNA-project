import argparse
import os
import random
import subprocess
import sys

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from dotenv import load_dotenv

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    from .data import MRNALoArmDataset
    from .loss import compute_lo_arm_loss
    from .model import LoArmConfig, LoArmTransformer
except ImportError:  # pragma: no cover - supports `python train.py`
    from data import MRNALoArmDataset
    from loss import compute_lo_arm_loss
    from model import LoArmConfig, LoArmTransformer


def _to_device(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _samples_to_batch(samples, device):
    return _to_device(default_collate(samples), device)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def validate(
    model,
    dataset,
    mask_id,
    batch_size,
    device,
    rank=0,
    world_size=1,
    distributed=False,
    max_batches=None,
):
    model.eval()
    loss_sum = 0.0
    count = 0

    indices = list(range(rank, len(dataset), world_size))
    if max_batches is not None:
        indices = indices[: max_batches * batch_size]

    for start in range(0, len(indices), batch_size):
        batch_idx = start // batch_size
        if max_batches is not None and batch_idx >= max_batches:
            break
        samples = [dataset[i] for i in indices[start : start + batch_size]]
        if not samples:
            continue
        batch = _samples_to_batch(samples, device)
        metrics = compute_lo_arm_loss(model, batch, mask_id)
        batch_count = batch["target_ids"].size(0)
        loss_sum += float(metrics["negative_elbo"].item()) * batch_count
        count += batch_count

    if distributed:
        totals = torch.tensor([loss_sum, float(count)], device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        loss_sum, count = totals[0].item(), int(totals[1].item())

    model.train()
    if count == 0:
        return float("nan")
    return loss_sum / count


def _checkpoint_payload(model, config, args, dataset):
    return {
        "model_state_dict": _unwrap_model(model).state_dict(),
        "config": config.to_dict(),
        "tokenizer": {
            "k": args.k,
            "vocab": dataset.vocab,
            "mask_id": dataset.mask_id,
        },
        "data": {
            "max_utr5_len": args.max_utr5_len,
            "max_cds_len": args.max_cds_len,
            "max_utr3_len": args.max_utr3_len,
        },
    }


def train(args, device, rank=0, world_size=1, distributed=False, local_rank=0):
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
    if distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0

        all_indices = list(range(len(train_dataset)))
        rng = random.Random(args.seed + epoch)
        rng.shuffle(all_indices)
        if distributed and len(all_indices) % world_size != 0:
            pad = world_size - (len(all_indices) % world_size)
            all_indices.extend(all_indices[:pad])
        indices = all_indices[rank::world_size]
        n_steps = (len(indices) + args.batch_size - 1) // args.batch_size

        for step in tqdm(range(n_steps), desc=f"epoch {epoch}", disable=(rank != 0)):
            start = step * args.batch_size
            end = min(start + args.batch_size, len(indices))
            samples = [train_dataset[indices[i]] for i in range(start, end)]
            if not samples:
                continue
            batch = _samples_to_batch(samples, device)
            optimizer.zero_grad()
            metrics = compute_lo_arm_loss(model, batch, train_dataset.mask_id)
            loss = metrics["loss"]
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_count = batch["target_ids"].size(0)
            loss_sum += float(loss.item()) * batch_count
            count += batch_count
            global_step += 1
            if args.wandb and wandb is not None and global_step % args.log_every == 0 and rank == 0:
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/negative_elbo": float(metrics["negative_elbo"].item()),
                        "train/n_previous": metrics["n_previous"],
                    },
                    step=global_step,
                )

        if distributed:
            totals = torch.tensor([loss_sum, float(count)], device=device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            loss_sum, count = totals[0].item(), int(totals[1].item())

        train_loss = loss_sum / max(count, 1)
        val_loss = validate(
            model,
            val_dataset,
            train_dataset.mask_id,
            args.batch_size,
            device,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
            max_batches=args.max_val_batches,
        )
        if rank == 0:
            print(f"Epoch {epoch} | train_loss={train_loss:.4f} val_neg_elbo={val_loss:.4f}")
        if args.wandb and wandb is not None and rank == 0:
            wandb.log({"epoch/train_loss": train_loss, "valid/negative_elbo": val_loss}, step=global_step)

        if args.output_path and val_loss < best_val and rank == 0:
            best_val = val_loss
            torch.save(
                _checkpoint_payload(model, config, args, train_dataset),
                args.output_path,
            )
            print(f"[checkpoint] saved {args.output_path}", file=sys.stderr)


def init_distributed_from_environment():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = "0"
        if "MASTER_ADDR" not in os.environ:
            try:
                nodelist = os.environ.get("SLURM_NODELIST", "")
                master = subprocess.check_output(
                    ["scontrol", "show", "hostnames", nodelist],
                    stderr=subprocess.DEVNULL,
                ).decode().splitlines()[0].strip()
            except Exception:
                master = "127.0.0.1"
            os.environ["MASTER_ADDR"] = master
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        distributed = True

    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, rank, world_size, local_rank, device


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

    load_dotenv()

    torch.manual_seed(args.seed)
    distributed, rank, world_size, local_rank, device = init_distributed_from_environment()

    if args.wandb and rank == 0:
        if wandb is None:
            raise RuntimeError("wandb is not installed")
        wandb.init(project=args.wandb_project, config=vars(args), dir="../../logs")

    train(
        args,
        device,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        local_rank=local_rank,
    )

    if args.wandb and wandb is not None and rank == 0:
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
