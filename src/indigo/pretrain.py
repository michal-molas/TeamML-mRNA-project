import argparse
import sys
import random

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
    """Build relative position matrix R for the full input sequence.

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
    abs_pos = list(range(prefix_len))
    for t in range(target_len):
        abs_pos.append(prefix_len + perm[t])

    R = torch.zeros(seq_len, seq_len, dtype=torch.long)
    for i in range(seq_len):
        for j in range(seq_len):
            if abs_pos[i] < abs_pos[j]:
                R[i, j] = -1
            elif abs_pos[i] > abs_pos[j]:
                R[i, j] = 1
    return R


def compute_position_targets(perm):
    """Compute INDIGO position targets for the generation permutation.

    Returns list of p_indigo values (length target_len - 1).
    At step i (i >= 1), p_indigo encodes where token i is inserted among
    the already-placed tokens 0..i-1.
    """
    position_targets = []
    placed = [perm[0]]

    for i in range(1, len(perm)):
        cur_pos = perm[i]
        sorted_placed = sorted(placed)
        insert_idx = 0
        for j, p in enumerate(sorted_placed):
            if p < cur_pos:
                insert_idx = j + 1

        if insert_idx == 0:
            neighbour_abs = sorted_placed[0]
            neighbour_gen_idx = placed.index(neighbour_abs)
            p_indigo = neighbour_gen_idx
        elif insert_idx == len(sorted_placed):
            neighbour_abs = sorted_placed[-1]
            neighbour_gen_idx = placed.index(neighbour_abs)
            p_indigo = i + neighbour_gen_idx
        else:
            left_neighbour_abs = sorted_placed[insert_idx - 1]
            neighbour_gen_idx = placed.index(left_neighbour_abs)
            p_indigo = i + neighbour_gen_idx

        position_targets.append(p_indigo)
        placed.append(cur_pos)

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


def compute_indigo_loss(model, input_ids, R, prefix_len, word_targets, pos_targets,
                        target_len, eos_gen_idx, device):
    """Compute combined word + position prediction loss for one INDIGO step.

    Uses teacher forcing: the encoder sees the full sequence with the final R
    matrix, then word prediction loss is computed on all target positions
    simultaneously. Position prediction loss is computed per-step.
    """
    input_ids = input_ids.unsqueeze(0).to(device)  # (1, seq_len)
    R = R.unsqueeze(0).to(device)                  # (1, seq_len, seq_len)
    word_targets = word_targets.to(device)
    pos_targets = pos_targets.to(device)

    H, R_out, word_logits = model(input_ids, R)

    # Word prediction loss: at input position (prefix_len - 1 + t), predict
    # the t-th target token (autoregressive shift by 1)
    target_logits = word_logits[0, prefix_len - 1: prefix_len - 1 + target_len, :]
    word_loss = F.cross_entropy(target_logits, word_targets)

    # Position prediction loss: for each step t in [1, target_len),
    # predict where the (t+1)-th token is inserted among the first t tokens
    pos_loss = torch.tensor(0.0, device=device)
    n_pos_steps = len(pos_targets)
    if n_pos_steps > 0:
        for t in range(n_pos_steps):
            # Hidden states of generated tokens 0..t (t+1 tokens)
            H_gen = H[0, prefix_len: prefix_len + t + 1, :].unsqueeze(0)  # (1, t+1, d)
            # Embedding of the token about to be placed
            z_emb = model.get_embedding_matrix()[word_targets[t + 1]]      # (d,)
            # Determine boundary indices among the generated tokens seen so far
            cur_eos_idx = eos_gen_idx if (eos_gen_idx is not None and eos_gen_idx <= t) else None
            # Position logits: (1, 2*(t+1)) — left/right of each existing token
            p_logits = model.position_logits(
                H_gen, z_emb.unsqueeze(0),
                bos_gen_idx=0,              # first gen token is leftmost boundary
                eos_gen_idx=cur_eos_idx,    # EOS boundary (if already placed)
            )  # (1, 2*(t+1))
            target_p = pos_targets[t].unsqueeze(0)
            if target_p.item() < p_logits.size(-1):
                pos_loss = pos_loss + F.cross_entropy(p_logits, target_p)

        pos_loss = pos_loss / n_pos_steps

    return word_loss + pos_loss


def extract_prefix_and_target(sample, pad_id):
    """Extract prefix and target token lists from a dataset sample."""
    input_ids_raw = sample["input_ids"]
    target_ids_raw = sample["target_ids"]
    loss_mask = sample["loss_mask"]

    prefix_end = (loss_mask == 0).sum().item()
    prefix_tokens = input_ids_raw[:prefix_end + 1].tolist()
    target_tokens = target_ids_raw[prefix_end:].tolist()
    target_tokens = [t for t in target_tokens if t != pad_id]
    return prefix_tokens, target_tokens


def compute_validation_loss(args, model, val_dataset, pad_id, eos_id, device):
    model.eval()
    val_loss = 0.0
    count = 0

    with torch.no_grad():
        for idx in range(min(len(val_dataset), 200)):
            sample = val_dataset[idx]
            prefix_tokens, target_tokens = extract_prefix_and_target(sample, pad_id)

            if len(target_tokens) < 2:
                continue

            tensors = build_training_tensors(prefix_tokens, target_tokens, eos_id)
            loss = compute_indigo_loss(
                model,
                tensors["input_ids"],
                tensors["R"],
                tensors["prefix_len"],
                tensors["word_targets"],
                tensors["pos_targets"],
                tensors["target_len"],
                tensors["eos_gen_idx"],
                device,
            )
            val_loss += loss.item()
            count += 1

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

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        model.train()
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        for step, idx in tqdm(list(enumerate(indices))):
            sample = train_dataset[idx]
            prefix_tokens, target_tokens = extract_prefix_and_target(sample, pad_id)

            if len(target_tokens) < 2:
                continue

            tensors = build_training_tensors(prefix_tokens, target_tokens, eos_id)

            optimizer.zero_grad()
            loss = compute_indigo_loss(
                model,
                tensors["input_ids"],
                tensors["R"],
                tensors["prefix_len"],
                tensors["word_targets"],
                tensors["pos_targets"],
                tensors["target_len"],
                tensors["eos_gen_idx"],
                device,
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()

            if args.wandb and step % 100 == 0:
                wandb.log({"train/loss": loss.item()}, step=global_step)

        epoch_loss /= max(len(indices), 1)
        print(f"Epoch {epoch} | loss={epoch_loss:.4f}")

        if args.output_path and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model_state_dict": model.state_dict()}, args.output_path)

        val_loss = compute_validation_loss(args, model, val_dataset, pad_id, eos_id, device)
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
    parser.add_argument("--batch_size", type=int, default=1)
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
