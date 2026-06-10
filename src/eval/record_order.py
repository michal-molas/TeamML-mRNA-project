import argparse
import json
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
TRANSFORMER_ROOT = SRC_ROOT / "transformer_training"
LO_ARM_ROOT = SRC_ROOT / "lo_arm"

for path in (str(REPO_ROOT), str(SRC_ROOT), str(TRANSFORMER_ROOT), str(LO_ARM_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.indigo.main import IndigoTransformer
from src.lo_arm.data import MRNALoArmDataset, build_model_inputs
from src.lo_arm.model import LoArmConfig, LoArmTransformer
from src.transformer_training.models import MRNACsvDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["auto", "indigo", "loarm"],
        default="auto",
        help="Checkpoint family. 'auto' detects LO-ARM checkpoints from their config payload.",
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Absolute or relative path to the .pt file")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--samples", "--max_samples", dest="samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample_order", action="store_true", help="Sample order logits instead of using argmax.")
    parser.add_argument("--sample_values", action="store_true", help="Sample value logits instead of using argmax.")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--max_utr5_len", type=int, default=None)
    parser.add_argument("--max_cds_len", type=int, default=None)
    parser.add_argument("--max_utr3_len", type=int, default=None)
    return parser.parse_args()


def clean_seq(value):
    if pd.isna(value):
        return ""
    return str(value).upper().replace("U", "T")


def coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def checkpoint_state_dict(checkpoint):
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def detect_model_type(checkpoint):
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict) and {"prefix_len", "target_len"}.issubset(config):
            return "loarm"
        state_dict = checkpoint_state_dict(checkpoint)
    else:
        state_dict = checkpoint

    if isinstance(state_dict, dict) and "encoder.embedding_layer.token_embedding.weight" in state_dict:
        return "indigo"
    if isinstance(state_dict, dict) and "order_head.weight" in state_dict:
        return "loarm"
    raise ValueError("Could not detect checkpoint type. Pass --model_type explicitly.")


def load_checkpoint_file(path, device):
    print(f"Loading local checkpoint from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find checkpoint at {path}")
    return torch.load(path, map_location=device)


def load_indigo_checkpoint(checkpoint, device):
    state_dict = checkpoint_state_dict(checkpoint)

    token_embedding = state_dict["encoder.embedding_layer.token_embedding.weight"]
    position_embedding = state_dict["encoder.embedding_layer.position_embedding.weight"]
    rel_pos_embedding = state_dict["encoder.blocks.0.attention_layer.relative_positional_embedding.weight"]
    layer_indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("encoder.blocks.") and key.split(".")[2].isdigit()
    }
    d_model = token_embedding.shape[1]
    d_head = rel_pos_embedding.shape[1]
    
    config = SimpleNamespace(
        vocab_size=token_embedding.shape[0],
        d_model=d_model,
        num_heads=d_model // d_head,
        num_layers=max(layer_indices) + 1,
        max_len=position_embedding.shape[0],
    )
    
    model = IndigoTransformer(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_loarm_checkpoint(checkpoint, device):
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError("LO-ARM order recording requires a checkpoint with a saved config payload.")
    config = LoArmConfig(**checkpoint["config"])
    state_dict = checkpoint_state_dict(checkpoint)
    model = LoArmTransformer(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def indigo_dataset_kwargs(args):
    return {
        "max_utr5_len": coalesce(args.max_utr5_len, 200),
        "max_cds_len": coalesce(args.max_cds_len, 500),
        "max_utr3_len": coalesce(args.max_utr3_len, 200),
        "k": coalesce(args.k, 3),
        "only_utr5": False,
    }


def loarm_dataset_kwargs(args, checkpoint):
    data_cfg = checkpoint.get("data", {}) if isinstance(checkpoint, dict) else {}
    tokenizer_cfg = checkpoint.get("tokenizer", {}) if isinstance(checkpoint, dict) else {}
    return {
        "max_utr5_len": coalesce(args.max_utr5_len, data_cfg.get("max_utr5_len"), 200),
        "max_cds_len": coalesce(args.max_cds_len, data_cfg.get("max_cds_len"), 500),
        "max_utr3_len": coalesce(args.max_utr3_len, data_cfg.get("max_utr3_len"), 200),
        "k": coalesce(args.k, tokenizer_cfg.get("k"), 3),
    }

def build_indigo_R(prefix_len, generated_count, ordered_steps, device):
    seq_len = prefix_len + generated_count
    abs_pos = torch.arange(seq_len, dtype=torch.long, device=device)
    rank_by_step = {step: rank for rank, step in enumerate(ordered_steps)}
    for step in range(generated_count):
        abs_pos[prefix_len + step] = prefix_len + rank_by_step[step]
    return torch.sign(abs_pos.unsqueeze(0) - abs_pos.unsqueeze(1)).long()

def sample_logits(logits, temperature=1.0, greedy=True):
    if greedy:
        return torch.argmax(logits, dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def invert_position_to_step_order(position_to_step):
    step_to_position = [None] * len(position_to_step)
    for position, step in enumerate(position_to_step):
        step = int(step)
        if step < 0 or step >= len(position_to_step):
            raise ValueError(f"Invalid generation step {step} in order of length {len(position_to_step)}")
        step_to_position[step] = position
    if any(position is None for position in step_to_position):
        raise ValueError("Generation order is not a complete permutation.")
    return step_to_position


def track_indigo_generation_order(
    model,
    dataset,
    cds,
    device,
    max_target_tokens,
    temperature,
    greedy_order,
    greedy_value,
):
    cds_tokens = dataset.tokenize(cds)
    if len(cds_tokens) > dataset.max_cds_tokens:
        cds_tokens = cds_tokens[:dataset.max_cds_tokens]
    prefix = [dataset.bos_id, dataset.cds_id] + cds_tokens + [dataset.utr5_id]
    
    generated_tokens = []
    ordered_steps = []
    invalid_word_ids = [dataset.pad_id, dataset.bos_id, dataset.cds_id, dataset.utr5_id]

    for step in range(max_target_tokens):
        input_ids = torch.tensor([prefix + generated_tokens], dtype=torch.long, device=device)
        seq_len = input_ids.shape[1]
        if seq_len > model.config.max_len:
            break

        R = build_indigo_R(len(prefix), len(generated_tokens), ordered_steps, device).unsqueeze(0)
        padding_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
        
        H, _, word_logits = model(input_ids, R, padding_mask)
        prediction_pos = len(prefix) + len(generated_tokens) - 1
        
        next_word_logits = word_logits[:, prediction_pos, :].clone()
        next_word_logits[:, invalid_word_ids] = float("-inf")
        next_token = int(sample_logits(next_word_logits, temperature=temperature, greedy=greedy_value).item())

        if step == 0:
            ordered_steps.append(0)
        else:
            _model = model.module if hasattr(model, "module") else model
            h = H[:, prediction_pos, :]
            H_gen = H[:, len(prefix) : len(prefix) + len(generated_tokens), :]
            
            left_keys = _model.position_head_left_proj(H_gen)
            right_keys = _model.position_head_right_proj(H_gen)
            keys = torch.cat([left_keys, right_keys], dim=1)
            
            query = _model.position_head_state_proj(h) + _model.get_embedding_matrix()[next_token].unsqueeze(0)
            position_logits = torch.matmul(query, keys.squeeze(0).T)

            valid = torch.ones(position_logits.shape[-1], dtype=torch.bool, device=device)
            # valid[0] = False
            position_logits = position_logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
            
            slot = int(sample_logits(position_logits, temperature=temperature, greedy=greedy_order).item())
            ref_step = slot % len(generated_tokens)
            ref_pos = ordered_steps.index(ref_step)
            insert_pos = ref_pos if slot < len(generated_tokens) else ref_pos + 1
            ordered_steps.insert(insert_pos, step)

        generated_tokens.append(next_token)
        if next_token == dataset.eos_id:
            break

    return invert_position_to_step_order(ordered_steps)


def mask_loarm_value_logits(value_logits, mask_id):
    logits = value_logits.clone()
    logits[..., mask_id] = float("-inf")
    return logits


def effective_target_length(target_ids, eos_id):
    try:
        return target_ids.index(eos_id) + 1
    except ValueError:
        return len(target_ids)


def track_loarm_generation_order(
    model,
    dataset,
    cds,
    device,
    temperature,
    greedy_order,
    greedy_value,
):
    if dataset.prefix_len != model.config.prefix_len or dataset.target_len != model.config.target_len:
        raise ValueError(
            "LO-ARM dataset canvas does not match checkpoint config: "
            f"dataset prefix/target=({dataset.prefix_len}, {dataset.target_len}), "
            f"checkpoint prefix/target=({model.config.prefix_len}, {model.config.target_len})."
        )

    prefix_ids, prefix_padding_mask = dataset.encode_cds_prefix(cds)
    prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    prefix_padding_mask = torch.tensor(prefix_padding_mask, dtype=torch.bool, device=device).unsqueeze(0)

    target = torch.full((1, dataset.target_len), dataset.mask_id, dtype=torch.long, device=device)
    remaining = torch.ones((1, dataset.target_len), dtype=torch.bool, device=device)
    full_order = []

    for _ in range(dataset.target_len):
        input_ids, padding_mask = build_model_inputs(prefix_ids, prefix_padding_mask, target)
        outputs = model(input_ids, padding_mask)

        order_logits = outputs["order_logits"].masked_fill(~remaining, float("-inf"))
        slot = int(sample_logits(order_logits, temperature=temperature, greedy=greedy_order).item())

        value_logits = mask_loarm_value_logits(outputs["value_logits"], dataset.mask_id)
        selected_value_logits = value_logits[0, slot, :]
        value = int(sample_logits(selected_value_logits.unsqueeze(0), temperature=temperature, greedy=greedy_value).item())

        target[0, slot] = value
        remaining[0, slot] = False
        full_order.append(slot)

    target_ids = target.squeeze(0).detach().cpu().tolist()
    sequence_length = effective_target_length(target_ids, dataset.eos_id)
    order = [position for position in full_order if position < sequence_length]
    return order, full_order, target_ids


def token_count(dataset, seq, reverse=False, max_len=None):
    seq = clean_seq(seq)
    if reverse:
        seq = seq[::-1]
    if max_len is not None:
        seq = seq[:max_len]
    tokenize = getattr(dataset, "tokenize", None)
    if tokenize is None:
        tokenize = dataset.tokenizer.tokenize
    return len(tokenize(seq))


def run_indigo(args, checkpoint, device):
    model = load_indigo_checkpoint(checkpoint, device)
    dataset = MRNACsvDataset(args.input_csv, **indigo_dataset_kwargs(args))

    max_target_tokens = model.config.max_len - (2 + dataset.max_cds_tokens + 1)
    if max_target_tokens <= 0:
        raise ValueError("INDIGO checkpoint max_len is too short for the configured CDS prefix.")

    df = pd.read_csv(args.input_csv).head(args.samples)
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tracking INDIGO Generation Orders"):
        cds_str = clean_seq(row["cds"])
        order = track_indigo_generation_order(
            model,
            dataset,
            cds_str,
            device,
            max_target_tokens,
            temperature=args.temperature,
            greedy_order=not args.sample_order,
            greedy_value=not args.sample_values,
        )
        results.append({
            "id": str(row["id"]),
            "model_type": "indigo",
            "order_format": "step_to_position",
            "sequence_length": len(order),
            "generation_order": order,
            "utr5_token_length": token_count(dataset, row.get("utr5", ""), reverse=True, max_len=dataset.max_utr5_len),
            "utr3_token_length": token_count(dataset, row.get("utr3", ""), max_len=dataset.max_utr3_len),
        })
    return results


def run_loarm(args, checkpoint, device):
    model = load_loarm_checkpoint(checkpoint, device)
    dataset = MRNALoArmDataset(args.input_csv, **loarm_dataset_kwargs(args, checkpoint))

    n_samples = min(args.samples, len(dataset))
    results = []
    for idx in tqdm(range(n_samples), desc="Tracking LO-ARM Generation Orders"):
        sample = dataset.samples[idx]
        order, full_order, target_ids = track_loarm_generation_order(
            model,
            dataset,
            sample["cds"],
            device,
            temperature=args.temperature,
            greedy_order=not args.sample_order,
            greedy_value=not args.sample_values,
        )
        results.append({
            "id": sample["id"],
            "model_type": "loarm",
            "order_format": "step_to_position",
            "sequence_length": len(order),
            "generation_order": order,
            "target_length": dataset.target_len,
            "full_generation_order": full_order,
            "target_ids": target_ids,
            "utr5_token_length": token_count(dataset, sample["utr5"], reverse=True, max_len=dataset.max_utr5_len),
            "utr3_token_length": token_count(dataset, sample["utr3"], max_len=dataset.max_utr3_len),
        })
    return results

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint_file(args.checkpoint_path, device)
    model_type = detect_model_type(checkpoint) if args.model_type == "auto" else args.model_type
    print(f"Recording generation order for model_type={model_type}")

    if model_type == "indigo":
        results = run_indigo(args, checkpoint, device)
    elif model_type == "loarm":
        results = run_loarm(args, checkpoint, device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Successfully saved execution tracks to {args.output_json}")

if __name__ == "__main__":
    main()
