import argparse
import os
import sys
import functools

import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
from dotenv import load_dotenv

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from ..common import MRNACsvDataset
from .models import MRNATransformer


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        raise RuntimeError(
            "Distributed environment variables not found. "
            "Launch with torchrun."
        )

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        msg = str(device)
        print(f"Rank 0 : {msg}", file=sys.stderr)

        for r in range(1, world_size):
            size = torch.zeros(1, dtype=torch.int64, device=device)
            dist.recv(size, src=r)

            buf = torch.zeros(size.item(), dtype=torch.int64, device=device)
            dist.recv(buf, src=r)

            msg = ''.join(chr(x) for x in buf.cpu().tolist())
            print(f"Rank {r} : {msg}", file=sys.stderr, flush=True)

    else:
        msg = str(device)
        buf = torch.tensor([ord(c) for c in msg], dtype=torch.int64, device=device)
        size = torch.tensor([buf.numel()], dtype=torch.int64, device=device)

        dist.send(size, dst=0)
        dist.send(buf, dst=0)

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Average a scalar tensor across all ranks.
    """
    if not is_dist():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def compute_validation_loss(args, model, valid_dataloader, global_step, device):
    model.eval()
    val_loss_sum = torch.zeros(1, device=device)
    val_batches = torch.zeros(1, device=device)

    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)

            outputs = model(input_ids, padding_mask=padding_mask)

            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = target_ids.reshape(-1)

            loss = F.cross_entropy(outputs_flat, targets_flat, reduction="none")
            loss = (loss * loss_mask.reshape(-1)).sum() / (loss_mask.sum() + 1e-8)

            val_loss_sum += loss.detach()
            val_batches += 1

    local_val_loss = val_loss_sum / torch.clamp(val_batches, min=1.0)
    global_val_loss = reduce_mean(local_val_loss)

    if is_main_process():
        print(f"Validation Loss: {global_val_loss.item()}", file=sys.stderr)
        if args.wandb:
            wandb.log(
                {"valid/loss": global_val_loss.item()},
                step=global_step,
            )

    return global_val_loss.item()


def save_fsdp_checkpoint(model, output_path):
    """
    Save a full, non-sharded state_dict on rank 0 only.
    """
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        save_policy,
    ):
        cpu_state = model.state_dict()

    if is_main_process():
        torch.save({"model_state_dict": cpu_state}, output_path)


def train(args, device, only_utr5=False):
    train_dataset = MRNACsvDataset(
        csv_path=args.train_csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        only_utr5=only_utr5,
    )
    val_dataset = MRNACsvDataset(
        csv_path=args.test_csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        only_utr5=only_utr5,
    )

    if is_main_process():
        print(f"Train dataset length: {len(train_dataset)}", file=sys.stderr)
        print(f"Validation dataset length: {len(val_dataset)}", file=sys.stderr)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        drop_last=False,
    )
    valid_sampler = DistributedSampler(
        val_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=False,
        drop_last=False,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    base_model = MRNATransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.n_layers,
        max_len=max(train_dataset.max_len, val_dataset.max_len),
    )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={nn.TransformerEncoderLayer},
    )

    model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        epoch_loss_sum = torch.zeros(1, device=device)
        epoch_steps = torch.zeros(1, device=device)

        progress = tqdm(
            dataloader,
            disable=not is_main_process(),
            desc=f"Epoch {epoch}",
        )

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(input_ids, padding_mask=padding_mask)

            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = target_ids.reshape(-1)

            loss = F.cross_entropy(outputs_flat, targets_flat, reduction="none")
            loss = (loss * loss_mask.reshape(-1)).sum() / (loss_mask.sum() + 1e-8)

            loss.backward()
            optimizer.step()

            global_step += 1

            detached_loss = loss.detach()
            epoch_loss_sum += detached_loss
            epoch_steps += 1

            if is_main_process():
                progress.set_postfix(loss=f"{detached_loss.item():.4f}")

                if args.wandb and step % 100 == 0:
                    wandb.log(
                        {"train/loss": detached_loss.item()},
                        step=global_step,
                    )

        local_epoch_loss = epoch_loss_sum / torch.clamp(epoch_steps, min=1.0)
        global_epoch_loss = reduce_mean(local_epoch_loss)

        if is_main_process():
            print(f"Epoch {epoch} | loss={global_epoch_loss.item():.4f}")

        valid_loss = compute_validation_loss(
            args=args,
            model=model,
            valid_dataloader=valid_dataloader,
            global_step=global_step,
            device=device,
        )

        if args.output_path and valid_loss < best_loss:
            best_loss = valid_loss
            save_fsdp_checkpoint(model, args.output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_path", type=str, default="../../data/pretraining/dataset_with_utr3/train.csv")
    parser.add_argument("--test_csv_path", type=str, default="../../data/pretraining/dataset_with_utr3/test.csv")
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_utr5_len", type=int, default=200)
    parser.add_argument("--max_cds_len", type=int, default=500)
    parser.add_argument("--max_utr3_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--fsdp_min_num_params", type=int, default=100_000)

    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    load_dotenv()

    if args.wandb and is_main_process():
        wandb.init(
            project="transformer-pretraining",
            config=vars(args),
            dir="../../logs",
        )

    only_utr5 = False 
    try:
        train(args, device, only_utr5)
    finally:
        if args.wandb and is_main_process():
            wandb.finish()
        cleanup_distributed()

if __name__ == "__main__":
    main()
