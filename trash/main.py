from functools import partial
import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from collections import OrderedDict
from datasets import load_dataset, load_from_disk
from transformers import GPT2TokenizerFast
import wandb
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings


class AttentionLayer(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
    ):
        super(AttentionLayer, self).__init__()

        self.ln = nn.LayerNorm(dmodel)

        self.heads = heads

        self.input_projection = nn.Linear(dmodel, 3 * dmodel, bias=False)

        self.output_projection = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, x, attention_mask):
        x = self.ln(x)

        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        key = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        value = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            attention_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                is_causal=True,
            )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


def FeedForward(
    dmodel,
):
    original_hidden_dim = 4 * dmodel
    hidden_dim = int(original_hidden_dim * (2 / 3))

    class SwiGLU(nn.Module):
        def forward(self, x):
            x1, x2 = x.chunk(2, dim=-1)
            return F.silu(x1) * x2

    return nn.Sequential(
        OrderedDict(
            [
                ("ff_layernorm", nn.LayerNorm(dmodel)),
                ("pre_swiglu", nn.Linear(dmodel, 2 * hidden_dim, bias=True)),
                ("swiglu", SwiGLU()),
                ("post_swiglu", nn.Linear(hidden_dim, dmodel, bias=True)),
            ]
        )
    )


class Block(nn.Module):

    def __init__(
        self,
        dmodel,
        heads,
    ):
        super().__init__()
        self.attention_layer = AttentionLayer(dmodel, heads)
        self.feed_forward_layer = FeedForward(dmodel)

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = EmbeddingLayer(
            config.vocab_size, config.d_model, config.max_len
        )
        self.blocks = nn.ModuleList(
            [Block(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        )

        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        output = self.embedding_layer(input_ids)

        for block in self.blocks:
            output = block(output, attention_mask)

        output = self.head(output)
        return output


def collate_tokenize(tokenizer, sequence_length, data):
    text_batch = [element["text"] for element in data]
    tokenized = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=sequence_length + 1,
    )
    input_ids = tokenized["input_ids"]
    tokenized["input_ids"] = input_ids[:, :-1]
    tokenized["target_ids"] = input_ids[:, 1:]
    tokenized["attention_mask"] = tokenized["attention_mask"][:, :-1]
    return tokenized


def get_dataloader(
    batch_size,
    sequence_length,
    split="train",
    buffer_size=10000,
    seed=42,
    num_workers=2,
    rank=0,
    world_size=1,
):
    if split == "train":
        hf_dataset = load_from_disk("/net/tscratch/people/plgjkrajewski/datasets/c4/train")
    else:
        hf_dataset = load_from_disk("/net/tscratch/people/plgjkrajewski/datasets/c4/validation")
    hf_dataset = hf_dataset.to_iterable_dataset(num_shards=64)
    if world_size > 1:
        hf_dataset = hf_dataset.shard(num_shards=world_size, index=rank)
    hf_dataset = hf_dataset.shuffle(buffer_size=buffer_size, seed=seed + rank)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_tokenize, tokenizer, sequence_length),
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return dataloader


def calculate_valid_loss(model, valid_dataloader, device, validation_steps, distributed=False):
    total_loss_sum = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)
    for _, batch in zip(range(validation_steps), valid_dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = model(input_ids)
            mask_loss = F.cross_entropy(
                outputs.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            mask = attention_mask.reshape(-1) == 1
            total_loss_sum += mask_loss[mask].sum()
            total_tokens += mask.sum()

    if distributed:
        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    return (total_loss_sum / total_tokens).item()


def train_model(config, device, rank, world_size, distributed):
    local_batch_size = config.batch_size
    if distributed:
        local_batch_size = config.batch_size // world_size

    dataloader = get_dataloader(local_batch_size, config.seq_length, rank=rank, world_size=world_size)
    valid_dataloader = get_dataloader(
        local_batch_size,
        config.seq_length,
        split="validation",
        rank=rank,
        world_size=world_size,
    )
    validation_steps = int(1e06 // (config.batch_size * config.seq_length))
    model = Transformer(config)

    if distributed:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block},
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            device_id=device,
        )
    else:
        model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    total_steps = int(config.train_steps)
    warmup_steps = int(0.01 * total_steps)
    decay_steps = int(0.10 * total_steps)
    stable_end = total_steps - decay_steps

    def wsd_lr(step):
        if step < warmup_steps:
            return config.learning_rate * (float(step + 1) / float(warmup_steps))
        if step < stable_end:
            return config.learning_rate
        else:
            decay_progress = min(1.0, float(step - stable_end + 1) / float(decay_steps))
            return config.learning_rate * (1.0 - decay_progress)

    model.train()

    for i, batch in zip(range(config.train_steps), dataloader):
        lr = wsd_lr(i)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(input_ids)

        mask_loss = F.cross_entropy(
            outputs.flatten(0, -2),
            target_ids.reshape(-1).long(),
            reduction="none",
        )
        mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
        loss = mask_loss.mean()

        if rank == 0:
            if i % config.log_train_loss_freq == 0:
                print(f"Step:{i}, Train Loss:{loss}")
                wandb.log({"train/loss": loss.item()}, step=i)

            wandb.log({"lr": lr}, step=i)

        if i % config.log_valid_loss_freq == 0:
            valid_loss = calculate_valid_loss(
                model,
                valid_dataloader,
                device,
                validation_steps,
                distributed=distributed,
            )
            if rank == 0:
                print(f"Valid loss:{valid_loss}")
                wandb.log({"valid/loss": valid_loss}, step=i)

        loss.backward()
        optimizer.step()

    final_valid_loss = calculate_valid_loss(
        model,
        valid_dataloader,
        device,
        validation_steps,
        distributed=distributed,
    )
    if rank == 0:
        print(f"Final valid loss:{final_valid_loss}")
        wandb.log({"final/valid_loss": final_valid_loss}, step=config.train_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", "--num_layers", dest="num_layers", type=int, default=4)
    parser.add_argument("--dmodel", "--d_model", dest="d_model", type=int, default=256)
    parser.add_argument("--n_heads", "--num_heads", dest="num_heads", type=int, default=4)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--n_training_steps",
        "--train_steps",
        dest="train_steps",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    tags = [
        f"n_layers={args.num_layers}",
        f"d_model={args.d_model}",
        f"n_heads={args.num_heads}",
        f"batch_size={args.batch_size}",
        f"train_steps={args.train_steps}",
    ]

    config = SimpleNamespace(
        train_steps=args.train_steps,
        vocab_size=50257,
        max_len=256,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        dropout=0.0,
        seq_length=256,
        batch_size=args.batch_size,
        log_train_loss_freq=100,
        log_valid_loss_freq=100
    )
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        print(f"Device type is: {device}. Remember to train on GPU.")

    wandb_config = {
        "n_layers": args.num_layers,
        "d_model": args.d_model,
        "n_heads": args.num_heads,
        "batch_size": args.batch_size,
        "n_training_steps": args.train_steps,
        "learning_rate": args.learning_rate,
    }
    if rank == 0:
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if array_task_id is not None:
            run_name = f"job_{job_id}/task_{array_task_id}/lr_{args.learning_rate}"
        else:
            run_name = f"job_{job_id}"
        wandb.init(
            project="bml-assignment2",
            name=run_name,
            config=wandb_config,
            tags=tags,
        )

    train_model(config, device, rank, world_size, distributed)

    if rank == 0:
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()