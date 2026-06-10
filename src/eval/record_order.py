import argparse
import json
import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace
from src.indigo.main import IndigoTransformer
from src.transformer_training.models import MRNACsvDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Absolute or relative path to the .pt file")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--samples", type=int, default=100)
    return parser.parse_args()

def load_checkpoint(path, device):
    print(f"Loading local checkpoint from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find checkpoint at {path}")
        
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    
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

def track_generation_order(model, dataset, cds, device, max_target_tokens):
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
        next_token = int(sample_logits(next_word_logits, greedy=True).item())

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
            
            slot = int(sample_logits(position_logits, greedy=True).item())
            ref_step = slot % len(generated_tokens)
            ref_pos = ordered_steps.index(ref_step)
            insert_pos = ref_pos if slot < len(generated_tokens) else ref_pos + 1
            ordered_steps.insert(insert_pos, step)

        generated_tokens.append(next_token)
        if next_token == dataset.eos_id:
            break

    return ordered_steps

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(args.checkpoint_path, device)
    
    dataset = MRNACsvDataset(
        args.input_csv,
        max_utr5_len=200,
        max_cds_len=500,
        max_utr3_len=200,
        k=3,
        only_utr5=False
    )
    
    max_target_tokens = model.config.max_len - (2 + dataset.max_cds_tokens + 1)
    df = pd.read_csv(args.input_csv).head(args.samples)
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tracking Generation Orders"):
        cds_str = str(row["cds"]).upper().replace("U", "T")
        order = track_generation_order(model, dataset, cds_str, device, max_target_tokens)
        results.append({
            "id": str(row["id"]),
            "sequence_length": len(order),
            "generation_order": order
        })
        
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Successfully saved execution tracks to {args.output_json}")

if __name__ == "__main__":
    main()