import argparse
import bisect
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dotenv import load_dotenv
import wandb

# ----------------------------
# Imports from your repo
# ----------------------------
# Adjust these two paths to match your repository layout
# sys.path.append("../transformer_training")
# from models import MRNACsvDataset

# # Indigo model (adjust import if yours lives elsewhere)
# from main import IndigoTransformer

# # RiboNN
# sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "RiboNN"))
# from RiboNN.src.model import RiboNN

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "RiboNN"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, "../transformer_training")
from RiboNN.src.model import RiboNN
sys.path.remove(str(Path(__file__).resolve().parents[2] / "RiboNN"))
from transformer_training.models import MRNACsvDataset, MRNATransformer
sys.path.remove(str(Path(__file__).resolve().parents[2]))
sys.path.remove(str(Path(__file__).resolve().parents[2] / "src"))
sys.path.remove("../transformer_training")
from main import IndigoTransformer


# ============================================================
# RiboNN configuration + ensemble (preloaded once)
# ============================================================
RIBONN_MAX_TX_LEN = 1_381 + 11_937  # 13318
RIBONN_LEN_AFTER_CONV = 9

RIBONN_CONFIG = dict(
    with_NAs=False,
    split_utr5_cds_utr3_channels=False,
    label_codons=True,
    label_utr5=False,
    label_utr3=False,
    label_splice_sites=False,
    label_up_probs=False,
    filters=64,
    conv_stride=1,
    conv_padding=0,
    ln_epsilon=0.007,
    dropout=0.3,
    residual=False,
    activation="relu",
    kernel_size=5,
    num_conv_layers=10,
    len_after_conv=RIBONN_LEN_AFTER_CONV,
    num_targets=78,
    max_shift=0,
    symmetric_shift=True,
)


def load_ribonn(weights_path, device, verbose=False):
    model = RiboNN(**dict(RIBONN_CONFIG))
    state_dict = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if verbose:
        print(f"[ribonn] loaded {weights_path} missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class RiboNNEnsemble(torch.nn.Module):
    """
    Load top-k models per fold ONCE from a RiboNN runs.csv folder and average predictions.
    """
    def __init__(self, weights_folder, device, top_k=5, verbose=False):
        super().__init__()
        run_df = pd.read_csv(Path(weights_folder) / "runs.csv")

        model_paths = []
        for test_fold in np.sort(run_df["params.test_fold"].unique()):
            tf_str = str(test_fold)
            sub = run_df.query("`params.test_fold` == @tf_str or `params.test_fold` == @test_fold")
            sub = sub.sort_values("metrics.val_r2", ascending=False).head(top_k)
            for run_id in sub.run_id.tolist():
                model_paths.append(Path(weights_folder) / run_id / "state_dict.pth")

        if verbose:
            print(f"[ribonn] ensemble size={len(model_paths)}", file=sys.stderr)

        self.models = torch.nn.ModuleList([load_ribonn(str(p), device, verbose=verbose) for p in model_paths])
        self.n = len(self.models)
        if self.n == 0:
            raise ValueError(f"No RiboNN models found in {weights_folder}")

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, ribonn_input):
        acc = None
        for m in self.models:
            pred = m(ribonn_input)  # (B, 78)
            acc = pred if acc is None else (acc + pred)
        return acc / self.n


# ============================================================
# InDIGO core: relative matrix R, permutation training, SAO
# ============================================================
def build_full_R_matrix(prefix_len, target_len, perm):
    """
    R[i,j] = sign(abs_pos[i] - abs_pos[j]) where abs_pos encodes insertion order.
    """
    seq_len = prefix_len + target_len
    abs_pos = torch.arange(seq_len, dtype=torch.long)
    abs_pos[prefix_len:] = prefix_len + torch.tensor(perm, dtype=torch.long)
    diff = abs_pos.unsqueeze(0) - abs_pos.unsqueeze(1)  # (seq_len, seq_len)
    return torch.sign(diff).long()


def compute_position_targets(perm):
    """
    Compute InDIGO position targets for a given generation permutation.
    """
    position_targets = []
    sorted_placed = [perm[0]]
    gen_idx_map = {perm[0]: 0}

    for i in range(1, len(perm)):
        cur_pos = perm[i]
        insert_idx = bisect.bisect_left(sorted_placed, cur_pos)

        if insert_idx == 0:
            p_indigo = gen_idx_map[sorted_placed[0]]
        else:
            p_indigo = i + gen_idx_map[sorted_placed[insert_idx - 1]]

        position_targets.append(p_indigo)
        sorted_placed.insert(insert_idx, cur_pos)
        gen_idx_map[cur_pos] = i

    return position_targets


def make_generation_perm(target_len, gen_order):
    """
    gen_order: random | l2r | r2l | inward | outward
    """
    if gen_order == "random":
        perm = list(range(target_len))
        random.shuffle(perm)
    elif gen_order == "l2r":
        perm = list(range(target_len))
    elif gen_order == "r2l":
        perm = list(range(target_len - 1, -1, -1))
    elif gen_order == "outward":
        mid = (target_len - 1) // 2
        perm = [mid]
        left = mid - 1
        right = mid + 1
        while left >= 0 or right < target_len:
            if right < target_len:
                perm.append(right)
                right += 1
            if left >= 0:
                perm.append(left)
                left -= 1
    elif gen_order == "inward":
        perm = []
        left = 0
        right = target_len - 1
        while left <= right:
            perm.append(left)
            if left != right:
                perm.append(right)
            left += 1
            right -= 1
    else:
        raise ValueError(f"Unknown gen_order: {gen_order}")
    return perm


def _extend_R(R_prev, abs_pos_prev, new_abs_pos_val):
    new_col = torch.sign(new_abs_pos_val - abs_pos_prev)
    new_row = -new_col
    R_new = torch.cat([R_prev, new_col.unsqueeze(1)], dim=1)
    new_row_full = torch.cat([new_row, torch.zeros(1, dtype=torch.long, device=R_prev.device)])
    return torch.cat([R_new, new_row_full.unsqueeze(0)], dim=0)


def _score_remaining(beam, remaining_list, log_prob_word_b, W, state_b, keys_b,
                     pos_valid_base, target_tokens, device):
    """
    Compute word + position scores for remaining tokens for one beam.
    """
    t_outer = keys_b.size(0) // 2  # tokens present at outer step

    tokens_rem = torch.tensor([target_tokens[i] for i in remaining_list],
                              dtype=torch.long, device=device)  # (n_rem,)

    word_scores = log_prob_word_b[tokens_rem]  # (n_rem,)

    eos_col = (t_outer + beam["eos_step"]) if beam["eos_step"] is not None else -1
    pos_valid = pos_valid_base.clone()
    if 0 <= eos_col < pos_valid.size(0):
        pos_valid[eos_col] = False

    sorted_placed_t = torch.tensor(beam["sorted_placed"], dtype=torch.long, device=device)
    gen_steps_t = torch.tensor([beam["gen_idx_map"][sp] for sp in beam["sorted_placed"]],
                               dtype=torch.long, device=device)
    remaining_t = torch.tensor(remaining_list, dtype=torch.long, device=device)

    pos_after_insert = (sorted_placed_t.unsqueeze(0) < remaining_t.unsqueeze(1)).sum(dim=1)
    predecessor_idx = torch.clamp(pos_after_insert - 1, min=0)

    ref_gen_step = torch.where(
        pos_after_insert == 0,
        gen_steps_t[0].expand(len(remaining_list)),
        gen_steps_t[predecessor_idx],
    )

    slot_has_key = ref_gen_step < t_outer

    insertion_slots = torch.where(
        pos_after_insert == 0,
        gen_steps_t[0].expand(len(remaining_list)),
        t_outer + gen_steps_t[predecessor_idx],
    )
    clamped_slots = insertion_slots.clamp(0, 2 * t_outer - 1)

    queries = state_b.unsqueeze(0) + W[tokens_rem]
    pos_logits = queries @ keys_b.T
    pos_logits = pos_logits.masked_fill(~pos_valid.unsqueeze(0), float("-inf"))
    log_p_pos = F.log_softmax(pos_logits, dim=-1)
    raw_pos = log_p_pos[torch.arange(len(remaining_list), device=device), clamped_slots]

    pos_scores = torch.where(
        slot_has_key & pos_valid[clamped_slots],
        raw_pos,
        torch.zeros_like(raw_pos),
    )

    return word_scores + pos_scores


def _extend_beam_state(beam, token_idx, new_score, current_t, target_tokens, eos_id, prefix_len, device):
    new_sorted = beam["sorted_placed"].copy()
    ins = bisect.bisect_left(new_sorted, token_idx)
    new_sorted.insert(ins, token_idx)

    new_gen_idx_map = dict(beam["gen_idx_map"])
    new_gen_idx_map[token_idx] = current_t

    eos_step = beam["eos_step"]
    if eos_step is None and target_tokens[token_idx] == eos_id:
        eos_step = current_t

    new_R = _extend_R(beam["R"], beam["abs_pos"], prefix_len + token_idx)
    new_abs_pos = torch.cat([beam["abs_pos"], torch.tensor([prefix_len + token_idx], device=device)])

    return {
        "perm": beam["perm"] + [token_idx],
        "sorted_placed": new_sorted,
        "gen_idx_map": new_gen_idx_map,
        "remaining": beam["remaining"] - {token_idx},
        "score": new_score,
        "eos_step": eos_step,
        "R": new_R,
        "abs_pos": new_abs_pos,
    }


@torch.no_grad()
def beam_search_perms(model, batch_prefix_tokens, batch_target_tokens, eos_id, beam_size, device,
                     n_init=15, tokens_per_step=10):
    """
    SAO: find permutations by beam-search using the model's own scoring.
    Returns: list[N] of list[beam_size] perms, each perm is list[int] of absolute target positions.
    """
    N = len(batch_prefix_tokens)
    prefix_lens = [len(p) for p in batch_prefix_tokens]
    target_lens = [len(t) for t in batch_target_tokens]
    _model = model.module if hasattr(model, "module") else model
    W = _model.get_embedding_matrix()  # (vocab_size, d_model)
    n_init = min(n_init, min(target_lens))

    beams = []
    for sample_idx in range(N):
        prefix_len = prefix_lens[sample_idx]
        target_tokens = batch_target_tokens[sample_idx]
        target_len = target_lens[sample_idx]
        abs_pos_prefix = torch.arange(prefix_len, dtype=torch.long, device=device)

        sample_beams = []
        for _ in range(beam_size):
            step0_perm = sorted(random.sample(range(target_len), n_init))
            R = build_full_R_matrix(prefix_len, n_init, step0_perm).to(device)
            abs_pos = torch.cat([abs_pos_prefix, prefix_len + torch.tensor(step0_perm, dtype=torch.long, device=device)])

            initial_score = 0.0
            eos_step = (n_init - 1) if target_tokens[step0_perm[-1]] == eos_id else None

            sample_beams.append({
                "perm": step0_perm,
                "sorted_placed": step0_perm,
                "gen_idx_map": {idx: k for k, idx in enumerate(step0_perm)},
                "remaining": set(range(target_len)) - set(step0_perm),
                "score": initial_score,
                "eos_step": eos_step,
                "R": R,
                "abs_pos": abs_pos,
            })
        beams.append(sample_beams)

    t = n_init
    while t < max(target_lens):
        all_beams = []
        for sample_idx in range(N):
            if target_lens[sample_idx] <= t:
                continue
            for beam in beams[sample_idx]:
                all_beams.append((sample_idx, beam))

        if len(all_beams) == 0:
            break

        n_all_beams = len(all_beams)
        max_seq_len = max(prefix_lens[sample_idx] + t for sample_idx, _ in all_beams)

        ids_batch = torch.zeros(n_all_beams, max_seq_len, dtype=torch.long, device=device)
        R_batch = torch.zeros(n_all_beams, max_seq_len, max_seq_len, dtype=torch.long, device=device)
        padding_mask_batch = torch.ones(n_all_beams, max_seq_len, dtype=torch.bool, device=device)

        for beam_idx, (sample_idx, beam) in enumerate(all_beams):
            prefix_len = prefix_lens[sample_idx]
            seq_len = prefix_len + t
            tokens = batch_prefix_tokens[sample_idx] + [batch_target_tokens[sample_idx][p] for p in beam["perm"]]
            ids_batch[beam_idx, :seq_len] = torch.tensor(tokens, dtype=torch.long, device=device)
            R_batch[beam_idx, :seq_len, :seq_len] = beam["R"]
            padding_mask_batch[beam_idx, :seq_len] = False

        H_batch_full, _, word_logits_full = model(ids_batch, R_batch, padding_mask_batch)

        new_beams_all = [beams[sample_idx] if target_lens[sample_idx] <= t else None for sample_idx in range(N)]
        sample_offset = 0

        for sample_idx in range(N):
            if target_lens[sample_idx] <= t:
                continue

            prefix_len = prefix_lens[sample_idx]
            target_tokens = batch_target_tokens[sample_idx]
            sample_beams = beams[sample_idx]
            n_beams = len(sample_beams)

            H_batch = H_batch_full[sample_offset:sample_offset + n_beams]
            word_logits_batch = word_logits_full[sample_offset:sample_offset + n_beams]
            sample_offset += n_beams

            h = H_batch[:, prefix_len + t - 1, :]  # (n_beams, d_model)
            log_prob_word = F.log_softmax(word_logits_batch[:, prefix_len + t - 1, :], dim=-1)

            H_gen = H_batch[:, prefix_len:prefix_len + t, :]
            left_keys = _model.position_head_left_proj(H_gen)
            right_keys = _model.position_head_right_proj(H_gen)
            keys = torch.cat([left_keys, right_keys], dim=1)  # (n_beams, 2t, d_model)
            state = _model.position_head_state_proj(h)

            pos_valid_base = torch.ones(2 * t, dtype=torch.bool, device=device)
            pos_valid_base[0] = False

            all_candidates = []
            for b_idx, beam in enumerate(sample_beams):
                remaining_list = list(beam["remaining"])
                scores = _score_remaining(
                    beam, remaining_list, log_prob_word[b_idx], W, state[b_idx], keys[b_idx],
                    pos_valid_base, target_tokens, device
                )
                for k, token_idx in enumerate(remaining_list):
                    all_candidates.append((beam["score"] + scores[k].item(), b_idx, token_idx))

            all_candidates.sort(key=lambda x: -x[0])
            top = all_candidates[:beam_size]

            new_beams = []
            for score, b_idx, token_idx in top:
                extended_beam = _extend_beam_state(
                    sample_beams[b_idx], token_idx, score, t, target_tokens, eos_id, prefix_len, device
                )

                remaining_list_k = list(extended_beam["remaining"])
                n_greedy = min(tokens_per_step - 1,
                               max(0, target_lens[sample_idx] - t - 1),
                               len(remaining_list_k))
                if n_greedy > 0:
                    scores_k = _score_remaining(
                        extended_beam, remaining_list_k, log_prob_word[b_idx], W,
                        state[b_idx], keys[b_idx], pos_valid_base, target_tokens, device
                    )
                    order_k = scores_k.argsort(descending=True).tolist()
                    greedy_tokens = [remaining_list_k[i] for i in order_k[:n_greedy]]

                    new_remaining = extended_beam["remaining"] - set(greedy_tokens)
                    greedy_score = extended_beam["score"] + scores_k[order_k[:n_greedy]].sum().item()

                    will_have_next_outer = (t + tokens_per_step < target_lens[sample_idx])
                    if will_have_next_outer:
                        new_sorted_k = extended_beam["sorted_placed"].copy()
                        new_gim_k = dict(extended_beam["gen_idx_map"])
                        eos_step_k = extended_beam["eos_step"]
                        new_R_k = extended_beam["R"]
                        new_abs_pos_k = extended_beam["abs_pos"]
                        for sk, tok in enumerate(greedy_tokens):
                            current_t = t + 1 + sk
                            ins = bisect.bisect_left(new_sorted_k, tok)
                            new_sorted_k.insert(ins, tok)
                            new_gim_k[tok] = current_t
                            if eos_step_k is None and target_tokens[tok] == eos_id:
                                eos_step_k = current_t
                            new_R_k = _extend_R(new_R_k, new_abs_pos_k, prefix_len + tok)
                            new_abs_pos_k = torch.cat([new_abs_pos_k, torch.tensor([prefix_len + tok], device=device)])
                        extended_beam = {
                            "perm": extended_beam["perm"] + greedy_tokens,
                            "sorted_placed": new_sorted_k,
                            "gen_idx_map": new_gim_k,
                            "remaining": new_remaining,
                            "score": greedy_score,
                            "eos_step": eos_step_k,
                            "R": new_R_k,
                            "abs_pos": new_abs_pos_k,
                        }
                    else:
                        extended_beam = {
                            "perm": extended_beam["perm"] + greedy_tokens,
                            "sorted_placed": extended_beam["sorted_placed"],
                            "gen_idx_map": extended_beam["gen_idx_map"],
                            "remaining": new_remaining,
                            "score": greedy_score,
                            "eos_step": extended_beam["eos_step"],
                            "R": extended_beam["R"],
                            "abs_pos": extended_beam["abs_pos"],
                        }

                new_beams.append(extended_beam)

            new_beams_all[sample_idx] = new_beams

        beams = new_beams_all
        t += tokens_per_step

    return [[beam["perm"] for beam in beams[sample_idx]] for sample_idx in range(N)]


# ============================================================
# Building training tensors + collate (with SAO + metadata for RiboNN)
# ============================================================
def extract_prefix_and_target(sample, pad_id):
    input_ids_raw = sample["input_ids"]
    target_ids_raw = sample["target_ids"]
