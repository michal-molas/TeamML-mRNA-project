import argparse
import bisect
import random
import sys

import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.optim import AdamW
from tqdm import tqdm
from dotenv import load_dotenv
from types import SimpleNamespace

sys.path.append('../transformer_training')
from models import MRNACsvDataset

from main import IndigoTransformer


def build_full_R_matrix(prefix_len, target_len, perm):
    """Build relative position matrix R for the full input sequence (vectorized).

    R[i, j] encodes relative position between token i and token j:
      -1: token i is LEFT of token j in the final sequence
       0: i == j
       1: token i is RIGHT of token j

    prefix_len: number of prefix (CDS context) tokens
    target_len: number of target (UTR) tokens
    perm: list where perm[t] is the absolute target index of the token
          generated at step t
    """
    seq_len = prefix_len + target_len
    # Absolute position of each token in the input sequence
    abs_pos = torch.arange(seq_len, dtype=torch.long)
    abs_pos[prefix_len:] = prefix_len + torch.tensor(perm, dtype=torch.long)

    # Vectorized comparison: R[i,j] = sign(abs_pos[j] - abs_pos[i])
    # -1 if abs_pos[i] < abs_pos[j], 0 if equal, 1 if greater
    diff = abs_pos.unsqueeze(0) - abs_pos.unsqueeze(1)  # (seq_len, seq_len)
    R = torch.sign(diff).long()  # -1, 0, or 1
    return R


def compute_position_targets(perm):
    """Compute INDIGO position targets for the generation permutation."""
    position_targets = []
    sorted_placed = [perm[0]]
    gen_idx_map = {perm[0]: 0}

    for i in range(1, len(perm)):
        cur_pos = perm[i]
        insert_idx = bisect.bisect_left(sorted_placed, cur_pos)

        if insert_idx == 0:
            p_indigo = gen_idx_map[sorted_placed[0]]
        elif insert_idx == len(sorted_placed):
            p_indigo = i + gen_idx_map[sorted_placed[-1]]
        else:
            p_indigo = i + gen_idx_map[sorted_placed[insert_idx - 1]]

        position_targets.append(p_indigo)
        sorted_placed.insert(insert_idx, cur_pos)
        gen_idx_map[cur_pos] = i

    return position_targets


def build_training_tensors(prefix_tokens, target_tokens, eos_id):
    """Build permuted training tensors for one sample.

    Returns dict with:
      - input_ids: (seq_len,) prefix + permuted target (no padding)
      - R: (seq_len, seq_len) full relative position matrix
      - word_targets: (target_len,) the token at each generation step
      - pos_targets: (target_len - 1,) the INDIGO position at each step
      - eos_gen_idx: generation index of the EOS token (for position masking)
      - prefix_len: int
      - target_len: int
    """

    # This is the Pre-defined Order (RND)
    # TODO
    target_len = len(target_tokens)
    perm = list(range(target_len))
    random.shuffle(perm)

    permuted_target = [target_tokens[perm[t]] for t in range(target_len)]

    # Track which generation step places the EOS token
    eos_gen_idx = None
    for t in range(target_len):
        if permuted_target[t] == eos_id:
            eos_gen_idx = t
            break

    prefix_len = len(prefix_tokens)
    R = build_full_R_matrix(prefix_len, target_len, perm)
    pos_targets = compute_position_targets(perm)

    input_ids = torch.tensor(prefix_tokens + permuted_target, dtype=torch.long)
    word_targets = torch.tensor(permuted_target, dtype=torch.long)
    pos_targets = torch.tensor(pos_targets, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "R": R,
        "word_targets": word_targets,
        "pos_targets": pos_targets,
        "eos_gen_idx": eos_gen_idx,
        "prefix_len": prefix_len,
        "target_len": target_len,
    }


def collate_indigo_batch(samples, pad_id, eos_id):
    """Collate a list of dataset samples into a padded batch for INDIGO training.

    Each sample is processed through extract_prefix_and_target + build_training_tensors,
    then padded to the maximum lengths in the batch.

    Returns a dict of batched tensors, or None if all samples were skipped.
    """
    batch_tensors = []
    for sample in samples:
        prefix_tokens, target_tokens = extract_prefix_and_target(sample, pad_id)
        if len(target_tokens) < 2:
            continue
        tensors = build_training_tensors(prefix_tokens, target_tokens, eos_id)
        batch_tensors.append(tensors)

    if len(batch_tensors) == 0:
        return None

    B = len(batch_tensors)
    max_seq_len = max(t["input_ids"].size(0) for t in batch_tensors)
    max_target_len = max(t["target_len"] for t in batch_tensors)
    max_pos_steps = max(t["pos_targets"].size(0) for t in batch_tensors)

    input_ids = torch.full((B, max_seq_len), pad_id, dtype=torch.long)
    R = torch.zeros(B, max_seq_len, max_seq_len, dtype=torch.long)
    attention_mask = torch.ones(B, max_seq_len, dtype=torch.bool)  # True = padding
    word_targets = torch.zeros(B, max_target_len, dtype=torch.long)
    pos_targets = torch.zeros(B, max_pos_steps, dtype=torch.long)
    prefix_lens = torch.zeros(B, dtype=torch.long)
    target_lens = torch.zeros(B, dtype=torch.long)
    eos_gen_idxs = torch.full((B,), -1, dtype=torch.long)

    for i, t in enumerate(batch_tensors):
        seq_len = t["input_ids"].size(0)
        target_len = t["target_len"]
        pos_steps = t["pos_targets"].size(0)

        input_ids[i, :seq_len] = t["input_ids"]
        R[i, :seq_len, :seq_len] = t["R"]
        attention_mask[i, :seq_len] = False
        word_targets[i, :target_len] = t["word_targets"]
        if pos_steps > 0:
            pos_targets[i, :pos_steps] = t["pos_targets"]
        prefix_lens[i] = t["prefix_len"]
        target_lens[i] = t["target_len"]
        if t["eos_gen_idx"] is not None:
            eos_gen_idxs[i] = t["eos_gen_idx"]

    return {
        "input_ids": input_ids,
        "R": R,
        "attention_mask": attention_mask,
        "word_targets": word_targets,
        "pos_targets": pos_targets,
        "prefix_lens": prefix_lens,
        "target_lens": target_lens,
        "eos_gen_idxs": eos_gen_idxs,
    }


def compute_indigo_loss_batched(model, batch, device):
    """
    Compute combined word + position prediction loss for a batch.
    """
    input_ids = batch["input_ids"].to(device)
    R = batch["R"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    word_targets = batch["word_targets"].to(device)
    pos_targets_pad = batch["pos_targets"].to(device)
    prefix_lens = batch["prefix_lens"].to(device)
    target_lens = batch["target_lens"].to(device)
    eos_gen_idxs = batch["eos_gen_idxs"].to(device)

    B = input_ids.size(0)
    max_seq_len = input_ids.size(1)
    max_target_len = word_targets.size(1)
    max_pos_steps = pos_targets_pad.size(1)

    # --- Forward pass ---
    H, _, word_logits = model(input_ids, R, attn_mask)
    vocab_size = word_logits.size(-1)
    d_model = H.size(-1)

    # --- Word loss ---
    # For sample b, word predictions are at positions [prefix_lens[b]-1 .. prefix_lens[b]-1+target_lens[b]-1]
    word_positions = (prefix_lens.unsqueeze(1) - 1
                      + torch.arange(max_target_len, device=device).unsqueeze(0))
    word_positions = word_positions.clamp(0, max_seq_len - 1)

    word_logits_gathered = word_logits.gather(
        1, word_positions.unsqueeze(-1).expand(-1, -1, vocab_size)
    )  # (B, max_target_len, vocab_size)

    word_loss_flat = F.cross_entropy(
        word_logits_gathered.reshape(-1, vocab_size),
        word_targets.reshape(-1),
        reduction='none',
    ).reshape(B, max_target_len)

    word_mask = torch.arange(max_target_len, device=device).unsqueeze(0) < target_lens.unsqueeze(1)
    word_loss = (word_loss_flat * word_mask).sum() / word_mask.sum().clamp(min=1)

    # --- Position loss (vectorized over all steps) ---
    pos_loss = torch.tensor(0.0, device=device)

    if max_pos_steps > 0:
        W = model.get_embedding_matrix()
        T = max_pos_steps

        # Gather hidden states for all generated positions
        gen_all_pos = (prefix_lens.unsqueeze(1) + torch.arange(T, device=device)).clamp(0, max_seq_len - 1)
        H_gen_all = H.gather(1, gen_all_pos.unsqueeze(-1).expand(-1, -1, d_model))

        # Projections for position prediction
        H_left_all = model.position_head_left_proj(H_gen_all) # H^T * C
        H_right_all = model.position_head_right_proj(H_gen_all) # H^T * D
        H_state_all = model.position_head_state_proj(H_gen_all) # h_t^T * E

        # Query = state_proj(h) + embedding of token to insert
        z_emb_all = W[word_targets[:, 1 : T + 1].clamp(0, vocab_size - 1)] # W_[y_{t+1}]
        query_all = H_state_all + z_emb_all # (h_t^T * E + W_[y_{t+1}])

        # Keys = [left_keys | right_keys], compute full logit matrix
        keys_full = torch.cat([H_left_all, H_right_all], dim=1) # [H^T * C | H^T * D]
        logits_full = torch.bmm(query_all, keys_full.transpose(-1, -2)) # (h_t^T * E + W_[y_{t+1}]) * [H^T * C | H^T * D]^T

        # Since we are doing this all at once, we need to mask out invalid positions.
        # At step t, only columns [0..t] (left) and [T..T+t] (right) are valid.
        t_idx = torch.arange(T, device=device).unsqueeze(1)
        j_idx = torch.arange(2 * T, device=device).unsqueeze(0)
        left_valid = (j_idx < T) & (j_idx <= t_idx)
        right_valid = (j_idx >= T) & (j_idx <= T + t_idx)
        full_mask = (~(left_valid | right_valid)).unsqueeze(0).expand(B, -1, -1).clone()

        # BOS column is always invalid
        full_mask[:, :, 0] = True

        # Mask the EOS column
        eos_col = (T + eos_gen_idxs).clamp(min=0, max=2 * T - 1)
        t_range = torch.arange(T, device=device).unsqueeze(0)
        eos_active_bt = t_range >= eos_gen_idxs.unsqueeze(1)
        eos_col_3d = eos_col.view(B, 1, 1).expand(B, T, 1)
        eos_scatter = torch.zeros(B, T, 2 * T, dtype=torch.bool, device=device)
        eos_scatter.scatter_(2, eos_col_3d, eos_active_bt.unsqueeze(2))
        full_mask = full_mask | eos_scatter

        # E.g. for T=4, EOS at idx 6, the full mask would look like this:
        #          |--  LEFT KEYS  --|--  RIGHT KEYS --|
        #          |  0   1   2   3  |  4   5   6   7  |
        # ----------------------------------------------
        # Step t=0 | [X] [X] [X] [X] | [ ] [X] [X] [X] |
        # Step t=1 | [X] [ ] [X] [X] | [ ] [ ] [X] [X] |
        # Step t=2 | [X] [ ] [ ] [X] | [ ] [ ] [X] [X] |
        # Step t=3 | [X] [ ] [ ] [ ] | [ ] [ ] [X] [ ] |
        logits_full = logits_full.masked_fill(full_mask, float('-inf'))

        # Map from local indices to global indices
        # Left ones stay the same, right ones are shifted by (T - (t+1))
        n_gen_vals = torch.arange(1, T + 1, device=device).unsqueeze(0)
        is_right_tgt = pos_targets_pad >= n_gen_vals
        target_col_full = torch.where(
            is_right_tgt, pos_targets_pad + (T - n_gen_vals), pos_targets_pad
        ).clamp(min=0, max=2 * T - 1)

        # Mask out invalid steps (padding, left of BOS, right of EOS)
        valid_step = torch.arange(T, device=device).unsqueeze(0) < (target_lens - 1).unsqueeze(1)
        bos_mask = target_col_full == 0
        eos_mask = target_col_full == eos_col.unsqueeze(1)
        valid_mask_pos = valid_step & ~bos_mask & ~eos_mask

        flat_logits = logits_full.reshape(B * T, 2 * T)[valid_mask_pos.reshape(B * T)]
        flat_targets = target_col_full.reshape(B * T)[valid_mask_pos.reshape(B * T)]
        pos_loss = F.cross_entropy(flat_logits, flat_targets)

    return word_loss + pos_loss


def extract_prefix_and_target(sample, pad_id):
    """Extract prefix and target token lists from a dataset sample."""
    input_ids_raw = sample["input_ids"]
    target_ids_raw = sample["target_ids"]
    loss_mask = sample["loss_mask"]

    # Find the first position where loss_mask == 1 (target starts).
    # Cannot use (loss_mask == 0).sum() because padding is also 0.
    ones = (loss_mask == 1).nonzero(as_tuple=True)[0]
    if len(ones) == 0:
        return [], []
    prefix_end = int(ones[0])
    prefix_tokens = input_ids_raw[:prefix_end + 1].tolist()
    target_tokens = target_ids_raw[prefix_end:].tolist()
    target_tokens = [t for t in target_tokens if t != pad_id]
    return prefix_tokens, target_tokens


def compute_validation_loss(args, model, val_dataset, pad_id, eos_id, batch_size, device):
    model.eval()
    val_loss = 0.0
    count = 0
    n_val = min(len(val_dataset), 200)

    with torch.no_grad():
        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            samples = [val_dataset[i] for i in range(start, end)]
            batch = collate_indigo_batch(samples, pad_id, eos_id)
            if batch is None:
                continue

            loss = compute_indigo_loss_batched(model, batch, device)
            batch_actual = batch["input_ids"].size(0)
            val_loss += loss.item() * batch_actual
            count += batch_actual

    if count > 0:
        val_loss /= count
    print(f"Validation Loss: {val_loss:.4f}", file=sys.stderr)
    return val_loss


def train(args, device):
    dataset = MRNACsvDataset(
        csv_path=args.csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        only_utr5=False,
    )

    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train dataset length: {len(train_dataset)}", file=sys.stderr)
    print(f"Validation dataset length: {len(val_dataset)}", file=sys.stderr)

    config = SimpleNamespace(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        num_heads=args.n_heads,
        num_layers=args.n_layers,
        max_len=dataset.max_len,
    )

    model = IndigoTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    pad_id = dataset.pad_id
    eos_id = dataset.eos_id
    global_step = 0
    best_loss = float("inf")

    batch_size = args.batch_size
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        model.train()
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        n_steps = (len(indices) + batch_size - 1) // batch_size
        for step in tqdm(range(n_steps)):
            start = step * batch_size
            end = min(start + batch_size, len(indices))
            samples = [train_dataset[indices[i]] for i in range(start, end)]

            batch = collate_indigo_batch(samples, pad_id, eos_id)
            if batch is None:
                continue

            optimizer.zero_grad()
            loss = compute_indigo_loss_batched(model, batch, device)
            loss.backward()
            optimizer.step()

            global_step += 1
            n_batches += 1
            epoch_loss += loss.item()

            if args.wandb and step % 100 == 0:
                wandb.log({"train/loss": loss.item()}, step=global_step)

        epoch_loss /= max(n_batches, 1)
        print(f"Epoch {epoch} | loss={epoch_loss:.4f}")

        if args.output_path and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model_state_dict": model.state_dict()}, args.output_path)

        val_loss = compute_validation_loss(args, model, val_dataset, pad_id, eos_id, batch_size, device)
        if args.wandb:
            wandb.log({"valid/loss": val_loss}, step=global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="../../data/pretraining/pretraining_refseq.csv")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_utr5_len", type=int, default=200)
    parser.add_argument("--max_cds_len", type=int, default=500)
    parser.add_argument("--max_utr3_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dotenv()

    if args.wandb:
        wandb.init(
            project="indigo-pretrain",
            config=vars(args),
            dir='../../logs',
        )

    train(args, device)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
