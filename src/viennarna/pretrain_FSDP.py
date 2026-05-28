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
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from models import MRNACsvDataset, MRNATransformer


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
    """Average a scalar tensor across all ranks."""
    if not is_dist():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def masked_bce_with_logits(logits, targets, mask):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def compute_losses(args, model_outputs, batch):
    logits = model_outputs["logits"]
    target_ids = batch["target_ids"]
    loss_mask = batch["loss_mask"]

    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = target_ids.reshape(-1)

    ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    ce_loss = (ce_loss * loss_mask.reshape(-1)).sum() / (loss_mask.sum() + 1e-8)

    total_loss = ce_loss
    logs = {"loss_ce": ce_loss.detach()}

    mfe_loss = logits.new_tensor(0.0)
    if args.mfe_loss_weight > 0.0 and model_outputs.get("mfe") is not None:
        mfe_label = batch["mfe_label"]
        valid = torch.isfinite(mfe_label)
        if valid.any():
            mfe_loss = F.mse_loss(model_outputs["mfe"][valid], mfe_label[valid])
            total_loss = total_loss + args.mfe_loss_weight * mfe_loss
    logs["loss_mfe"] = mfe_loss.detach()

    paired_loss = logits.new_tensor(0.0)
    if args.paired_loss_weight > 0.0 and model_outputs.get("paired_logits") is not None:
        paired_logits = model_outputs["paired_logits"]
        paired_targets = batch["paired_labels"].clamp(min=0.0, max=1.0)
        structure_mask = batch["structure_loss_mask"]
        if args.paired_generated_only:
            structure_mask = structure_mask * batch["loss_mask"]
        if structure_mask.sum() > 0:
            paired_loss = masked_bce_with_logits(paired_logits, paired_targets, structure_mask)
            total_loss = total_loss + args.paired_loss_weight * paired_loss
    logs["loss_paired"] = paired_loss.detach()
    logs["loss_total"] = total_loss.detach()

    return total_loss, logs


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_validation_loss(args, model, valid_dataloader, global_step, device):
    model.eval()
    val_loss_sum = torch.zeros(1, device=device)
    val_ce_sum = torch.zeros(1, device=device)
    val_mfe_sum = torch.zeros(1, device=device)
    val_paired_sum = torch.zeros(1, device=device)
    val_batches = torch.zeros(1, device=device)

    with torch.no_grad():
        for batch in valid_dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["input_ids"], padding_mask=batch["padding_mask"], return_dict=True)
            loss, logs = compute_losses(args, outputs, batch)

            val_loss_sum += loss.detach()
            val_ce_sum += logs["loss_ce"]
            val_mfe_sum += logs["loss_mfe"]
            val_paired_sum += logs["loss_paired"]
            val_batches += 1

    denom = torch.clamp(val_batches, min=1.0)
    global_val_loss = reduce_mean(val_loss_sum / denom)
    global_val_ce = reduce_mean(val_ce_sum / denom)
    global_val_mfe = reduce_mean(val_mfe_sum / denom)
    global_val_paired = reduce_mean(val_paired_sum / denom)

    if is_main_process():
        print(
            "Validation | "
            f"loss={global_val_loss.item():.4f} "
            f"ce={global_val_ce.item():.4f} "
            f"mfe={global_val_mfe.item():.4f} "
            f"paired={global_val_paired.item():.4f}",
            file=sys.stderr,
        )
        if args.wandb:
            wandb.log(
                {
                    "valid/loss": global_val_loss.item(),
                    "valid/loss_ce": global_val_ce.item(),
                    "valid/loss_mfe": global_val_mfe.item(),
                    "valid/loss_paired": global_val_paired.item(),
                },
                step=global_step,
            )

    return global_val_loss.item()


def save_fsdp_checkpoint(model, output_path):
    """Save a full, non-sharded state_dict on rank 0 only."""
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
    needs_structure_labels = args.paired_loss_weight > 0.0

    train_dataset = MRNACsvDataset(
        csv_path=args.train_csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        only_utr5=only_utr5,
        use_rna_structure_labels=needs_structure_labels,
        mfe_column=args.mfe_column,
        full_structure_column=args.full_structure_column,
        utr5_structure_column=args.utr5_structure_column,
        cds_structure_column=args.cds_structure_column,
        utr3_structure_column=args.utr3_structure_column,
        normalise_mfe_by_length=not args.no_normalise_mfe_by_length,
    )
    val_dataset = MRNACsvDataset(
        csv_path=args.test_csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        only_utr5=only_utr5,
        use_rna_structure_labels=needs_structure_labels,
        mfe_column=args.mfe_column,
        full_structure_column=args.full_structure_column,
        utr5_structure_column=args.utr5_structure_column,
        cds_structure_column=args.cds_structure_column,
        utr3_structure_column=args.utr3_structure_column,
        normalise_mfe_by_length=not args.no_normalise_mfe_by_length,
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
        use_mfe_head=args.mfe_loss_weight > 0.0,
        use_paired_head=args.paired_loss_weight > 0.0,
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
        epoch_ce_sum = torch.zeros(1, device=device)
        epoch_mfe_sum = torch.zeros(1, device=device)
        epoch_paired_sum = torch.zeros(1, device=device)
        epoch_steps = torch.zeros(1, device=device)

        progress = tqdm(
            dataloader,
            disable=not is_main_process(),
            desc=f"Epoch {epoch}",
        )

        for step, batch in enumerate(progress):
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(batch["input_ids"], padding_mask=batch["padding_mask"], return_dict=True)
            loss, logs = compute_losses(args, outputs, batch)

            loss.backward()
            optimizer.step()

            global_step += 1

            detached_loss = loss.detach()
            epoch_loss_sum += detached_loss
            epoch_ce_sum += logs["loss_ce"]
            epoch_mfe_sum += logs["loss_mfe"]
            epoch_paired_sum += logs["loss_paired"]
            epoch_steps += 1

            if is_main_process():
                progress.set_postfix(
                    loss=f"{detached_loss.item():.4f}",
                    ce=f"{logs['loss_ce'].item():.4f}",
                    mfe=f"{logs['loss_mfe'].item():.4f}",
                    paired=f"{logs['loss_paired'].item():.4f}",
                )

                if args.wandb and step % 100 == 0:
                    wandb.log(
                        {
                            "train/loss": detached_loss.item(),
                            "train/loss_ce": logs["loss_ce"].item(),
                            "train/loss_mfe": logs["loss_mfe"].item(),
                            "train/loss_paired": logs["loss_paired"].item(),
                        },
                        step=global_step,
                    )

        denom = torch.clamp(epoch_steps, min=1.0)
        global_epoch_loss = reduce_mean(epoch_loss_sum / denom)
        global_epoch_ce = reduce_mean(epoch_ce_sum / denom)
        global_epoch_mfe = reduce_mean(epoch_mfe_sum / denom)
        global_epoch_paired = reduce_mean(epoch_paired_sum / denom)

        if is_main_process():
            print(
                f"Epoch {epoch} | "
                f"loss={global_epoch_loss.item():.4f} "
                f"ce={global_epoch_ce.item():.4f} "
                f"mfe={global_epoch_mfe.item():.4f} "
                f"paired={global_epoch_paired.item():.4f}",
                file=sys.stderr,
            )

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
    parser.add_argument("--train_csv_path", type=str, default="../../data/pretraining/dataset_with_utr3/train_vienna.csv")
    parser.add_argument("--test_csv_path", type=str, default="../../data/pretraining/dataset_with_utr3/test_vienna.csv")
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

    # Auxiliary ViennaRNA prediction losses. Both are off by default.
    parser.add_argument("--mfe_loss_weight", type=float, default=0.0)
    parser.add_argument("--paired_loss_weight", type=float, default=0.0)
    parser.add_argument("--paired_generated_only", action="store_true")
    parser.add_argument("--no_normalise_mfe_by_length", action="store_true")

    # Optional column overrides. If omitted, the dataset tries common names.
    parser.add_argument("--mfe_column", type=str, default=None)
    parser.add_argument("--full_structure_column", type=str, default=None)
    parser.add_argument("--utr5_structure_column", type=str, default=None)
    parser.add_argument("--cds_structure_column", type=str, default=None)
    parser.add_argument("--utr3_structure_column", type=str, default=None)

    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    load_dotenv()

    if args.wandb and is_main_process():
        wandb.init(
            project="vienna-pretrain-grid",
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
