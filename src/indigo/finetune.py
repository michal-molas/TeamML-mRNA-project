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
print(str(Path(__file__).resolve()))
print(sys.path)


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
# def build_full_R_matrix(prefix_len, target_len, perm):
#     """
#     R[i,j] = sign(abs_pos[i] - abs_pos[j]) where abs_pos encodes insertion order.
#     """
#     seq_len = prefix_len + target_len
#     abs_pos = torch.arange(seq_len, dtype=torch.long)
#     abs_pos[prefix_len:] = prefix_len + torch.tensor(perm, dtype=torch.long)
#     diff = abs_pos.unsqueeze(0) - abs_pos.unsqueeze(1)  # (seq_len, seq_len)
#     return torch.sign(diff).long()

from pretrain import build_full_R_matrix, compute_position_targets, make_generation_perm, _extend_R, _score_remaining, beam_search_perms, _extend_beam_state


# def compute_position_targets(perm):
#     """
#     Compute InDIGO position targets for a given generation permutation.
#     """
#     position_targets = []
#     sorted_placed = [perm[0]]
#     gen_idx_map = {perm[0]: 0}

#     for i in range(1, len(perm)):
#         cur_pos = perm[i]
#         insert_idx = bisect.bisect_left(sorted_placed, cur_pos)

#         if insert_idx == 0:
#             p_indigo = gen_idx_map[sorted_placed[0]]
#         else:
#             p_indigo = i + gen_idx_map[sorted_placed[insert_idx - 1]]

#         position_targets.append(p_indigo)
#         sorted_placed.insert(insert_idx, cur_pos)
#         gen_idx_map[cur_pos] = i

#     return position_targets


# def make_generation_perm(target_len, gen_order):
#     """
#     gen_order: random | l2r | r2l | inward | outward
#     """
#     if gen_order == "random":
#         perm = list(range(target_len))
#         random.shuffle(perm)
#     elif gen_order == "l2r":
#         perm = list(range(target_len))
#     elif gen_order == "r2l":
#         perm = list(range(target_len - 1, -1, -1))
#     elif gen_order == "outward":
#         mid = (target_len - 1) // 2
#         perm = [mid]
#         left = mid - 1
#         right = mid + 1
#         while left >= 0 or right < target_len:
#             if right < target_len:
#                 perm.append(right)
#                 right += 1
#             if left >= 0:
#                 perm.append(left)
#                 left -= 1
#     elif gen_order == "inward":
#         perm = []
#         left = 0
#         right = target_len - 1
#         while left <= right:
#             perm.append(left)
#             if left != right:
#                 perm.append(right)
#             left += 1
#             right -= 1
#     else:
#         raise ValueError(f"Unknown gen_order: {gen_order}")
#     return perm


# def _extend_R(R_prev, abs_pos_prev, new_abs_pos_val):
#     new_col = torch.sign(new_abs_pos_val - abs_pos_prev)
#     new_row = -new_col
#     R_new = torch.cat([R_prev, new_col.unsqueeze(1)], dim=1)
#     new_row_full = torch.cat([new_row, torch.zeros(1, dtype=torch.long, device=R_prev.device)])
#     return torch.cat([R_new, new_row_full.unsqueeze(0)], dim=0)


# def _score_remaining(beam, remaining_list, log_prob_word_b, W, state_b, keys_b,
#                      pos_valid_base, target_tokens, device):
#     """
#     Compute word + position scores for remaining tokens for one beam.
#     """
#     t_outer = keys_b.size(0) // 2  # tokens present at outer step

#     tokens_rem = torch.tensor([target_tokens[i] for i in remaining_list],
#                               dtype=torch.long, device=device)  # (n_rem,)

#     word_scores = log_prob_word_b[tokens_rem]  # (n_rem,)

#     eos_col = (t_outer + beam["eos_step"]) if beam["eos_step"] is not None else -1
#     pos_valid = pos_valid_base.clone()
#     if 0 <= eos_col < pos_valid.size(0):
#         pos_valid[eos_col] = False

#     sorted_placed_t = torch.tensor(beam["sorted_placed"], dtype=torch.long, device=device)
#     gen_steps_t = torch.tensor([beam["gen_idx_map"][sp] for sp in beam["sorted_placed"]],
#                                dtype=torch.long, device=device)
#     remaining_t = torch.tensor(remaining_list, dtype=torch.long, device=device)

#     pos_after_insert = (sorted_placed_t.unsqueeze(0) < remaining_t.unsqueeze(1)).sum(dim=1)
#     predecessor_idx = torch.clamp(pos_after_insert - 1, min=0)

#     ref_gen_step = torch.where(
#         pos_after_insert == 0,
#         gen_steps_t[0].expand(len(remaining_list)),
#         gen_steps_t[predecessor_idx],
#     )

#     slot_has_key = ref_gen_step < t_outer

#     insertion_slots = torch.where(
#         pos_after_insert == 0,
#         gen_steps_t[0].expand(len(remaining_list)),
#         t_outer + gen_steps_t[predecessor_idx],
#     )
#     clamped_slots = insertion_slots.clamp(0, 2 * t_outer - 1)

#     queries = state_b.unsqueeze(0) + W[tokens_rem]
#     pos_logits = queries @ keys_b.T
#     pos_logits = pos_logits.masked_fill(~pos_valid.unsqueeze(0), float("-inf"))
#     log_p_pos = F.log_softmax(pos_logits, dim=-1)
#     raw_pos = log_p_pos[torch.arange(len(remaining_list), device=device), clamped_slots]

#     pos_scores = torch.where(
#         slot_has_key & pos_valid[clamped_slots],
#         raw_pos,
#         torch.zeros_like(raw_pos),
#     )

#     return word_scores + pos_scores


# def _extend_beam_state(beam, token_idx, new_score, current_t, target_tokens, eos_id, prefix_len, device):
#     new_sorted = beam["sorted_placed"].copy()
#     ins = bisect.bisect_left(new_sorted, token_idx)
#     new_sorted.insert(ins, token_idx)

#     new_gen_idx_map = dict(beam["gen_idx_map"])
#     new_gen_idx_map[token_idx] = current_t

#     eos_step = beam["eos_step"]
#     if eos_step is None and target_tokens[token_idx] == eos_id:
#         eos_step = current_t

#     new_R = _extend_R(beam["R"], beam["abs_pos"], prefix_len + token_idx)
#     new_abs_pos = torch.cat([beam["abs_pos"], torch.tensor([prefix_len + token_idx], device=device)])

#     return {
#         "perm": beam["perm"] + [token_idx],
#         "sorted_placed": new_sorted,
#         "gen_idx_map": new_gen_idx_map,
#         "remaining": beam["remaining"] - {token_idx},
#         "score": new_score,
#         "eos_step": eos_step,
#         "R": new_R,
#         "abs_pos": new_abs_pos,
#     }


# @torch.no_grad()
# def beam_search_perms(model, batch_prefix_tokens, batch_target_tokens, eos_id, beam_size, device,
#                      n_init=15, tokens_per_step=10):
#     """
#     SAO: find permutations by beam-search using the model's own scoring.
#     Returns: list[N] of list[beam_size] perms, each perm is list[int] of absolute target positions.
#     """
#     N = len(batch_prefix_tokens)
#     prefix_lens = [len(p) for p in batch_prefix_tokens]
#     target_lens = [len(t) for t in batch_target_tokens]
#     _model = model.module if hasattr(model, "module") else model
#     W = _model.get_embedding_matrix()  # (vocab_size, d_model)
#     n_init = min(n_init, min(target_lens))

#     beams = []
#     for sample_idx in range(N):
#         prefix_len = prefix_lens[sample_idx]
#         target_tokens = batch_target_tokens[sample_idx]
#         target_len = target_lens[sample_idx]
#         abs_pos_prefix = torch.arange(prefix_len, dtype=torch.long, device=device)

#         sample_beams = []
#         for _ in range(beam_size):
#             step0_perm = sorted(random.sample(range(target_len), n_init))
#             R = build_full_R_matrix(prefix_len, n_init, step0_perm).to(device)
#             abs_pos = torch.cat([abs_pos_prefix, prefix_len + torch.tensor(step0_perm, dtype=torch.long, device=device)])

#             initial_score = 0.0
#             eos_step = (n_init - 1) if target_tokens[step0_perm[-1]] == eos_id else None

#             sample_beams.append({
#                 "perm": step0_perm,
#                 "sorted_placed": step0_perm,
#                 "gen_idx_map": {idx: k for k, idx in enumerate(step0_perm)},
#                 "remaining": set(range(target_len)) - set(step0_perm),
#                 "score": initial_score,
#                 "eos_step": eos_step,
#                 "R": R,
#                 "abs_pos": abs_pos,
#             })
#         beams.append(sample_beams)

#     t = n_init
#     while t < max(target_lens):
#         all_beams = []
#         for sample_idx in range(N):
#             if target_lens[sample_idx] <= t:
#                 continue
#             for beam in beams[sample_idx]:
#                 all_beams.append((sample_idx, beam))

#         if len(all_beams) == 0:
#             break

#         n_all_beams = len(all_beams)
#         max_seq_len = max(prefix_lens[sample_idx] + t for sample_idx, _ in all_beams)

#         ids_batch = torch.zeros(n_all_beams, max_seq_len, dtype=torch.long, device=device)
#         R_batch = torch.zeros(n_all_beams, max_seq_len, max_seq_len, dtype=torch.long, device=device)
#         padding_mask_batch = torch.ones(n_all_beams, max_seq_len, dtype=torch.bool, device=device)

#         for beam_idx, (sample_idx, beam) in enumerate(all_beams):
#             prefix_len = prefix_lens[sample_idx]
#             seq_len = prefix_len + t
#             tokens = batch_prefix_tokens[sample_idx] + [batch_target_tokens[sample_idx][p] for p in beam["perm"]]
#             ids_batch[beam_idx, :seq_len] = torch.tensor(tokens, dtype=torch.long, device=device)
#             R_batch[beam_idx, :seq_len, :seq_len] = beam["R"]
#             padding_mask_batch[beam_idx, :seq_len] = False

#         H_batch_full, _, word_logits_full = model(ids_batch, R_batch, padding_mask_batch)

#         new_beams_all = [beams[sample_idx] if target_lens[sample_idx] <= t else None for sample_idx in range(N)]
#         sample_offset = 0

#         for sample_idx in range(N):
#             if target_lens[sample_idx] <= t:
#                 continue

#             prefix_len = prefix_lens[sample_idx]
#             target_tokens = batch_target_tokens[sample_idx]
#             sample_beams = beams[sample_idx]
#             n_beams = len(sample_beams)

#             H_batch = H_batch_full[sample_offset:sample_offset + n_beams]
#             word_logits_batch = word_logits_full[sample_offset:sample_offset + n_beams]
#             sample_offset += n_beams

#             h = H_batch[:, prefix_len + t - 1, :]  # (n_beams, d_model)
#             log_prob_word = F.log_softmax(word_logits_batch[:, prefix_len + t - 1, :], dim=-1)

#             H_gen = H_batch[:, prefix_len:prefix_len + t, :]
#             left_keys = _model.position_head_left_proj(H_gen)
#             right_keys = _model.position_head_right_proj(H_gen)
#             keys = torch.cat([left_keys, right_keys], dim=1)  # (n_beams, 2t, d_model)
#             state = _model.position_head_state_proj(h)

#             pos_valid_base = torch.ones(2 * t, dtype=torch.bool, device=device)
#             pos_valid_base[0] = False

#             all_candidates = []
#             for b_idx, beam in enumerate(sample_beams):
#                 remaining_list = list(beam["remaining"])
#                 scores = _score_remaining(
#                     beam, remaining_list, log_prob_word[b_idx], W, state[b_idx], keys[b_idx],
#                     pos_valid_base, target_tokens, device
#                 )
#                 for k, token_idx in enumerate(remaining_list):
#                     all_candidates.append((beam["score"] + scores[k].item(), b_idx, token_idx))

#             all_candidates.sort(key=lambda x: -x[0])
#             top = all_candidates[:beam_size]

#             new_beams = []
#             for score, b_idx, token_idx in top:
#                 extended_beam = _extend_beam_state(
#                     sample_beams[b_idx], token_idx, score, t, target_tokens, eos_id, prefix_len, device
#                 )

#                 remaining_list_k = list(extended_beam["remaining"])
#                 n_greedy = min(tokens_per_step - 1,
#                                max(0, target_lens[sample_idx] - t - 1),
#                                len(remaining_list_k))
#                 if n_greedy > 0:
#                     scores_k = _score_remaining(
#                         extended_beam, remaining_list_k, log_prob_word[b_idx], W,
#                         state[b_idx], keys[b_idx], pos_valid_base, target_tokens, device
#                     )
#                     order_k = scores_k.argsort(descending=True).tolist()
#                     greedy_tokens = [remaining_list_k[i] for i in order_k[:n_greedy]]

#                     new_remaining = extended_beam["remaining"] - set(greedy_tokens)
#                     greedy_score = extended_beam["score"] + scores_k[order_k[:n_greedy]].sum().item()

#                     will_have_next_outer = (t + tokens_per_step < target_lens[sample_idx])
#                     if will_have_next_outer:
#                         new_sorted_k = extended_beam["sorted_placed"].copy()
#                         new_gim_k = dict(extended_beam["gen_idx_map"])
#                         eos_step_k = extended_beam["eos_step"]
#                         new_R_k = extended_beam["R"]
#                         new_abs_pos_k = extended_beam["abs_pos"]
#                         for sk, tok in enumerate(greedy_tokens):
#                             current_t = t + 1 + sk
#                             ins = bisect.bisect_left(new_sorted_k, tok)
#                             new_sorted_k.insert(ins, tok)
#                             new_gim_k[tok] = current_t
#                             if eos_step_k is None and target_tokens[tok] == eos_id:
#                                 eos_step_k = current_t
#                             new_R_k = _extend_R(new_R_k, new_abs_pos_k, prefix_len + tok)
#                             new_abs_pos_k = torch.cat([new_abs_pos_k, torch.tensor([prefix_len + tok], device=device)])
#                         extended_beam = {
#                             "perm": extended_beam["perm"] + greedy_tokens,
#                             "sorted_placed": new_sorted_k,
#                             "gen_idx_map": new_gim_k,
#                             "remaining": new_remaining,
#                             "score": greedy_score,
#                             "eos_step": eos_step_k,
#                             "R": new_R_k,
#                             "abs_pos": new_abs_pos_k,
#                         }
#                     else:
#                         extended_beam = {
#                             "perm": extended_beam["perm"] + greedy_tokens,
#                             "sorted_placed": extended_beam["sorted_placed"],
#                             "gen_idx_map": extended_beam["gen_idx_map"],
#                             "remaining": new_remaining,
#                             "score": greedy_score,
#                             "eos_step": extended_beam["eos_step"],
#                             "R": extended_beam["R"],
#                             "abs_pos": extended_beam["abs_pos"],
#                         }

#                 new_beams.append(extended_beam)

#             new_beams_all[sample_idx] = new_beams

#         beams = new_beams_all
#         t += tokens_per_step

#     return [[beam["perm"] for beam in beams[sample_idx]] for sample_idx in range(N)]


# ============================================================
# Building training tensors + collate (with SAO + metadata for RiboNN)
# ============================================================
def extract_prefix_and_target(sample, pad_id):
    input_ids_raw = sample["input_ids"]
    target_ids_raw = sample["target_ids"]
    loss_mask = sample["loss_mask"]

    ones = (loss_mask == 1).nonzero(as_tuple=True)[0]
    if len(ones) == 0:
        return None

    prefix_end = int(ones[0])
    prefix_tokens = input_ids_raw[:prefix_end + 1].tolist()
    target_tokens = target_ids_raw[prefix_end:].tolist()
    target_tokens = [t for t in target_tokens if t != pad_id]
    return prefix_tokens, target_tokens, prefix_end


def build_training_tensors(prefix_tokens, target_tokens, eos_id, gen_order="random", perm=None):
    target_len = len(target_tokens)
    if perm is None:
        perm = make_generation_perm(target_len, gen_order)

    permuted_target = [target_tokens[perm[t]] for t in range(target_len)]

    eos_gen_idx = None
    for t in range(target_len):
        if permuted_target[t] == eos_id:
            eos_gen_idx = t
            break

    prefix_len = len(prefix_tokens)
    R = build_full_R_matrix(prefix_len, target_len, perm)
    pos_targets = compute_position_targets(perm)

    return {
        "input_ids": torch.tensor(prefix_tokens + permuted_target, dtype=torch.long),
        "R": R,
        "word_targets": torch.tensor(permuted_target, dtype=torch.long),
        "pos_targets": torch.tensor(pos_targets, dtype=torch.long),
        "eos_gen_idx": eos_gen_idx,
        "prefix_len": prefix_len,
        "target_len": target_len,
        "perm": torch.tensor(perm, dtype=torch.long),
    }


def collate_indigo_batch(samples, pad_id, eos_id, gen_order="random",
                         model=None, device=None, sao_beam_size=4, sao_tokens_per_step=10):
    """
    Returns a batch dict for InDIGO training, and includes extra fields for RiboNN guidance:
      te_label, utr5_len, cds_len, utr3_len, orig_input_ids, prefix_end, perm_pad
    For SAO: expands the batch by sao_beam_size and duplicates metadata accordingly.
    """
    batch_tensors = []
    meta = []

    if gen_order == "sao":
        assert model is not None and device is not None, "SAO requires model and device"
        model_was_training = model.training
        model.eval()

        samples_with_targets = []
        for sample in samples:
            ext = extract_prefix_and_target(sample, pad_id)
            if ext is None:
                continue
            prefix_tokens, target_tokens, prefix_end = ext
            if len(target_tokens) >= 2:
                samples_with_targets.append((sample, prefix_tokens, target_tokens, prefix_end))

        if not samples_with_targets:
            if model_was_training:
                model.train()
            return None

        all_perms = beam_search_perms(
            model,
            [p for _, p, _, _ in samples_with_targets],
            [t for _, _, t, _ in samples_with_targets],
            eos_id, sao_beam_size, device,
            tokens_per_step=sao_tokens_per_step,
        )

        for (sample, prefix_tokens, target_tokens, prefix_end), perms in zip(samples_with_targets, all_perms):
            for perm in perms:
                batch_tensors.append(build_training_tensors(prefix_tokens, target_tokens, eos_id, perm=perm))
                meta.append({
                    "te_label": sample.get("te_label", None),
                    "utr5_len": sample.get("utr5_len", None),
                    "cds_len": sample.get("cds_len", None),
                    "utr3_len": sample.get("utr3_len", None),
                    "orig_input_ids": sample["input_ids"].clone(),
                    "prefix_end": prefix_end,
                })

        if model_was_training:
            model.train()

    else:
        for sample in samples:
            ext = extract_prefix_and_target(sample, pad_id)
            if ext is None:
                continue
            prefix_tokens, target_tokens, prefix_end = ext
            if len(target_tokens) < 2:
                continue
            batch_tensors.append(build_training_tensors(prefix_tokens, target_tokens, eos_id, gen_order))
            meta.append({
                "te_label": sample.get("te_label", None),
                "utr5_len": sample.get("utr5_len", None),
                "cds_len": sample.get("cds_len", None),
                "utr3_len": sample.get("utr3_len", None),
                "orig_input_ids": sample["input_ids"].clone(),
                "prefix_end": prefix_end,
            })

    if len(batch_tensors) == 0:
        return None

    B = len(batch_tensors)
    max_seq_len = max(t["input_ids"].size(0) for t in batch_tensors)
    max_target_len = max(t["target_len"] for t in batch_tensors)
    max_pos_steps = max(t["pos_targets"].size(0) for t in batch_tensors)

    input_ids = torch.full((B, max_seq_len), pad_id, dtype=torch.long)
    R = torch.zeros(B, max_seq_len, max_seq_len, dtype=torch.long)
    attention_mask = torch.ones(B, max_seq_len, dtype=torch.bool)  # True=padding
    word_targets = torch.full((B, max_target_len), pad_id, dtype=torch.long)
    pos_targets = torch.zeros(B, max_pos_steps, dtype=torch.long)
    prefix_lens = torch.zeros(B, dtype=torch.long)
    target_lens = torch.zeros(B, dtype=torch.long)
    eos_gen_idxs = torch.full((B,), -1, dtype=torch.long)
    perm_pad = torch.full((B, max_target_len), -1, dtype=torch.long)

    # Metadata for RiboNN
    te_label = torch.zeros(B, dtype=torch.float)
    utr5_len = torch.zeros(B, dtype=torch.long)
    cds_len = torch.zeros(B, dtype=torch.long)
    utr3_len = torch.zeros(B, dtype=torch.long)
    prefix_end_t = torch.zeros(B, dtype=torch.long)

    max_orig_len = max(int(m["orig_input_ids"].numel()) for m in meta)
    orig_input_ids = torch.full((B, max_orig_len), pad_id, dtype=torch.long)

    for i, t in enumerate(batch_tensors):
        seq_len = t["input_ids"].size(0)
        tgt_len = t["target_len"]
        pos_steps = t["pos_targets"].size(0)

        input_ids[i, :seq_len] = t["input_ids"]
        R[i, :seq_len, :seq_len] = t["R"]
        attention_mask[i, :seq_len] = False
        word_targets[i, :tgt_len] = t["word_targets"]
        if pos_steps > 0:
            pos_targets[i, :pos_steps] = t["pos_targets"]
        prefix_lens[i] = t["prefix_len"]
        target_lens[i] = t["target_len"]
        if t["eos_gen_idx"] is not None:
            eos_gen_idxs[i] = t["eos_gen_idx"]
        perm_pad[i, :tgt_len] = t["perm"]

        # meta
        prefix_end_t[i] = int(meta[i]["prefix_end"])
        if meta[i]["te_label"] is not None:
            te_label[i] = float(meta[i]["te_label"])
        if meta[i]["utr5_len"] is not None:
            utr5_len[i] = int(meta[i]["utr5_len"])
        if meta[i]["cds_len"] is not None:
            cds_len[i] = int(meta[i]["cds_len"])
        if meta[i]["utr3_len"] is not None:
            utr3_len[i] = int(meta[i]["utr3_len"])

        oi = meta[i]["orig_input_ids"].view(-1)
        orig_input_ids[i, :oi.numel()] = oi

    return {
        "input_ids": input_ids,
        "R": R,
        "attention_mask": attention_mask,
        "word_targets": word_targets,
        "pos_targets": pos_targets,
        "prefix_lens": prefix_lens,
        "target_lens": target_lens,
        "eos_gen_idxs": eos_gen_idxs,
        "perm_pad": perm_pad,

        # RiboNN metadata
        "te_label": te_label,
        "utr5_len": utr5_len,
        "cds_len": cds_len,
        "utr3_len": utr3_len,
        "orig_input_ids": orig_input_ids,
        "prefix_end": prefix_end_t,
        "pad_id": pad_id,
    }


# ============================================================
# RiboNN input assembly from InDIGO predictions
# ============================================================
def build_ribonn_input_from_indigo(nt_probs_abs_target, batch, device, label_codons=True, flip_utr5=True):
    """
    nt_probs_abs_target: (B, Ttarget, 4) nucleotide probabilities in ABS target order.

    Uses batch metadata to place:
      - UTR5: from predicted probs at positions utr5_start..utr5_end in original sequence
      - CDS:  from original (discrete) CDS tokens in orig_input_ids
      - UTR3: from predicted probs at positions utr3_start..utr3_end in original sequence

    Assumes the same special-token layout as your original RiboNN builder:
      <BOS>, <CDS>, CDS..., <UTR5>, UTR5..., <UTR3>, UTR3...
    and that UTR5 is stored reversed in the token stream, so we flip for RiboNN.
    """
    B, Tt, _ = nt_probs_abs_target.shape
    ribonn_max_len = RIBONN_MAX_TX_LEN
    num_channels = 5 if label_codons else 4
    out = torch.zeros(B, num_channels, ribonn_max_len, device=device)

    prefix_end = batch["prefix_end"].to(device)
    utr5_len = batch["utr5_len"].to(device)
    cds_len = batch["cds_len"].to(device)
    utr3_len = batch["utr3_len"].to(device)
    orig_ids = batch["orig_input_ids"].to(device)

    for b in range(B):
        pe = int(prefix_end[b].item())
        u5 = int(utr5_len[b].item())
        cd = int(cds_len[b].item())
        u3 = int(utr3_len[b].item())

        # Full-seq indices (same as your earlier script)
        cds_start = 2
        cds_end = cds_start + cd

        utr5_start = 2 + cd + 1
        utr5_end = utr5_start + u5

        utr3_start = 2 + cd + 1 + u5 + 1
        utr3_end = utr3_start + u3

        def pred_for_full_pos(p_full):
            j = p_full - pe
            if j < 0 or j >= Tt:
                return torch.zeros(4, device=device)
            return nt_probs_abs_target[b, j, :]

        # UTR5 predicted
        if u5 > 0:
            utr5_probs = torch.stack([pred_for_full_pos(p) for p in range(utr5_start, utr5_end)], dim=0)  # (u5,4)
            if flip_utr5:
                utr5_probs = torch.flip(utr5_probs, dims=[0])
            out[b, :4, 0:u5] = utr5_probs.T

        # CDS true one-hot (clamp to nucleotide IDs)
        if cd > 0:
            cds_tokens = orig_ids[b, cds_start:cds_end].clamp(min=0, max=3)
            out[b, :4, u5:u5+cd] = F.one_hot(cds_tokens, 4).float().T

        # UTR3 predicted
        if u3 > 0:
            utr3_probs = torch.stack([pred_for_full_pos(p) for p in range(utr3_start, utr3_end)], dim=0)  # (u3,4)
            out[b, :4, u5+cd:u5+cd+u3] = utr3_probs.T

        # Codon labels
        if label_codons:
            for codon_pos in range(u5, u5 + cd, 3):
                out[b, 4, codon_pos] = 1.0

    return out


# ============================================================
# InDIGO loss (word + position) + RiboNN guidance
# ============================================================
def compute_indigo_loss_with_ribonn(model, batch, device, ribonn_ens=None, lambda_ribonn=0.0, use_gumbel=False):
    """
    Returns: total_loss, indigo_loss, ribonn_loss
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

    # forward
    H, _, word_logits = model(input_ids, R, attn_mask)  # word_logits: (B, S, V)
    vocab_size = word_logits.size(-1)
    d_model = H.size(-1)

    # ---- word loss
    word_positions = (prefix_lens.unsqueeze(1) - 1 +
                      torch.arange(max_target_len, device=device).unsqueeze(0)).clamp(0, max_seq_len - 1)

    word_logits_gathered = word_logits.gather(
        1, word_positions.unsqueeze(-1).expand(-1, -1, vocab_size)
    )  # (B, T, V)

    word_loss_flat = F.cross_entropy(
        word_logits_gathered.reshape(-1, vocab_size),
        word_targets.reshape(-1),
        reduction="none",
    ).reshape(B, max_target_len)

    word_mask = torch.arange(max_target_len, device=device).unsqueeze(0) < target_lens.unsqueeze(1)
    word_loss = (word_loss_flat * word_mask).sum() / word_mask.sum().clamp(min=1)

    # ---- position loss (vectorised like your code)
    pos_loss = torch.tensor(0.0, device=device)
    if max_pos_steps > 0:
        _model = model.module if hasattr(model, "module") else model
        W = _model.get_embedding_matrix()
        T = max_pos_steps

        gen_all_pos = (prefix_lens.unsqueeze(1) + torch.arange(T, device=device)).clamp(0, max_seq_len - 1)
        H_gen_all = H.gather(1, gen_all_pos.unsqueeze(-1).expand(-1, -1, d_model))

        H_left_all = _model.position_head_left_proj(H_gen_all)
        H_right_all = _model.position_head_right_proj(H_gen_all)
        H_state_all = _model.position_head_state_proj(H_gen_all)

        z_emb_all = W[word_targets[:, 1:T+1].clamp(0, vocab_size - 1)]
        query_all = H_state_all + z_emb_all

        keys_full = torch.cat([H_left_all, H_right_all], dim=1)  # (B, 2T, d)
        logits_full = torch.bmm(query_all, keys_full.transpose(-1, -2))  # (B, T, 2T)

        t_idx = torch.arange(T, device=device).unsqueeze(1)
        j_idx = torch.arange(2 * T, device=device).unsqueeze(0)
        left_valid = (j_idx < T) & (j_idx <= t_idx)
        right_valid = (j_idx >= T) & (j_idx <= T + t_idx)
        full_mask = (~(left_valid | right_valid)).unsqueeze(0).expand(B, -1, -1).clone()
        full_mask[:, :, 0] = True  # BOS invalid

        eos_col = (T + eos_gen_idxs).clamp(min=0, max=2 * T - 1)
        t_range = torch.arange(T, device=device).unsqueeze(0)
        eos_active_bt = t_range >= eos_gen_idxs.unsqueeze(1)
        eos_col_3d = eos_col.view(B, 1, 1).expand(B, T, 1)
        eos_scatter = torch.zeros(B, T, 2 * T, dtype=torch.bool, device=device)
        eos_scatter.scatter_(2, eos_col_3d, eos_active_bt.unsqueeze(2))
        full_mask = full_mask | eos_scatter

        logits_full = logits_full.masked_fill(full_mask, float("-inf"))

        n_gen_vals = torch.arange(1, T + 1, device=device).unsqueeze(0)
        is_right_tgt = pos_targets_pad >= n_gen_vals
        target_col_full = torch.where(is_right_tgt, pos_targets_pad + (T - n_gen_vals), pos_targets_pad).clamp(0, 2 * T - 1)

        valid_step = torch.arange(T, device=device).unsqueeze(0) < (target_lens - 1).unsqueeze(1)
        bos_mask = target_col_full == 0
        eos_mask = target_col_full == eos_col.unsqueeze(1)
        valid_mask_pos = valid_step & ~bos_mask & ~eos_mask

        flat_logits = logits_full.reshape(B * T, 2 * T)[valid_mask_pos.reshape(B * T)]
        flat_targets = target_col_full.reshape(B * T)[valid_mask_pos.reshape(B * T)]
        if flat_logits.numel() > 0:
            pos_loss = F.cross_entropy(flat_logits, flat_targets)

    indigo_loss = word_loss + pos_loss

    # ---- RiboNN guidance
    ribonn_loss = torch.tensor(0.0, device=device)
    if ribonn_ens is not None and lambda_ribonn > 0.0:
        perm_pad = batch["perm_pad"].to(device)
        te_label = batch["te_label"].to(device)

        if use_gumbel:
            p_vocab = F.gumbel_softmax(word_logits_gathered, dim=-1)
        else:
            p_vocab = torch.softmax(word_logits_gathered, dim=-1)

        p_nt_step = p_vocab[..., :4]  # (B, T, 4)

        # Scatter per-step probs into ABS target order (B, T, 4)
        p_abs = torch.zeros_like(p_nt_step)
        idx = perm_pad.unsqueeze(-1).expand(-1, -1, 4).clamp(min=0)
        valid = (perm_pad >= 0) & word_mask
        p_abs.scatter_(1, idx, p_nt_step * valid.unsqueeze(-1))

        ribonn_input = build_ribonn_input_from_indigo(
            nt_probs_abs_target=p_abs,
            batch=batch,
            device=device,
            label_codons=RIBONN_CONFIG["label_codons"],
            flip_utr5=True,
        )

        pred_78 = ribonn_ens(ribonn_input)      # (B, 78)
        pred_te = pred_78.mean(dim=-1)          # (B,)
        ribonn_loss = F.mse_loss(pred_te, te_label)

    total = indigo_loss + lambda_ribonn * ribonn_loss
    return total, indigo_loss.detach(), ribonn_loss.detach()


# ============================================================
# Training / validation loops
# ============================================================
def train(args, device):
    dataset = MRNACsvDataset(
        csv_path=args.csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        only_utr5=False,
    )

    # Optional: small subset for local sanity runs
    if args.subset > 0:
        idx = list(range(min(args.subset, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, idx)

    dataset_size = len(dataset)
    val_size = int(args.val_frac * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train dataset length: {len(train_dataset)}", file=sys.stderr)
    print(f"Validation dataset length: {len(val_dataset)}", file=sys.stderr)

    # Indigo model config (matches your indigo code style)
    config = SimpleNamespace(
        vocab_size=dataset.dataset.vocab_size if hasattr(dataset, "dataset") else dataset.vocab_size,
        d_model=args.d_model,
        num_heads=args.n_heads,
        num_layers=args.n_layers,
        max_len=(dataset.dataset.max_len if hasattr(dataset, "dataset") else dataset.max_len),
    )
    model = IndigoTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # dataset pad/eos
    ds_obj = dataset.dataset if hasattr(dataset, "dataset") else dataset
    pad_id = ds_obj.pad_id
    eos_id = ds_obj.eos_id

    # RiboNN ensemble (optional)
    ribonn_ens = None
    if args.lambda_ribonn > 0:
        if not args.ribonn_weights_folder:
            raise ValueError("--ribonn_weights_folder must be set when --lambda_ribonn > 0")
        ribonn_ens = RiboNNEnsemble(args.ribonn_weights_folder, device=device, top_k=args.ribonn_top_k, verbose=args.verbose_ribonn)

    global_step = 0
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        for step, batch_samples in tqdm(enumerate(train_loader), total=len(train_loader)):
            if args.debug_steps > 0 and step >= args.debug_steps:
                break

            # DataLoader gives dict batches by default; we want list of samples for SAO collate.
            # If your DataLoader already yields dicts, convert to list-of-dicts.
            # Here we handle both:
            if isinstance(batch_samples, dict):
                # convert dict-of-tensors to list-of-dicts
                bs = batch_samples["input_ids"].shape[0]
                samples = [{k: batch_samples[k][i] for k in batch_samples.keys()} for i in range(bs)]
            else:
                samples = batch_samples  # already list-of-dicts

            batch = collate_indigo_batch(
                samples,
                pad_id=pad_id,
                eos_id=eos_id,
                gen_order=args.gen_order,
                model=model if args.gen_order == "sao" else None,
                device=device if args.gen_order == "sao" else None,
                sao_beam_size=args.sao_beam_size,
                sao_tokens_per_step=args.sao_tokens_per_step,
            )
            if batch is None:
                continue

            optimizer.zero_grad()
            loss, ind_l, rib_l = compute_indigo_loss_with_ribonn(
                model, batch, device,
                ribonn_ens=ribonn_ens,
                lambda_ribonn=args.lambda_ribonn,
                use_gumbel=args.ribonn_use_gumbel,
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1

            if args.wandb and (step % args.log_every == 0):
                wandb.log({
                    "train/loss": loss.item(),
                    "train/indigo_loss": ind_l.item(),
                    "train/ribonn_loss": rib_l.item(),
                }, step=global_step)

        epoch_loss /= max(n_batches, 1)
        print(f"Epoch {epoch} | train_loss={epoch_loss:.4f}", file=sys.stderr)

        # ---- validation
        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        val_loss_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for step, batch_samples in enumerate(val_loader):
                if args.debug_steps > 0 and step >= args.debug_steps:
                    break

                if isinstance(batch_samples, dict):
                    bs = batch_samples["input_ids"].shape[0]
                    samples = [{k: batch_samples[k][i] for k in batch_samples.keys()} for i in range(bs)]
                else:
                    samples = batch_samples

                batch = collate_indigo_batch(
                    samples,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    gen_order=args.gen_order,  # you can set val gen_order independently if you want
                    model=model if args.gen_order == "sao" else None,
                    device=device if args.gen_order == "sao" else None,
                    sao_beam_size=args.sao_beam_size,
                    sao_tokens_per_step=args.sao_tokens_per_step,
                )
                if batch is None:
                    continue

                loss, ind_l, rib_l = compute_indigo_loss_with_ribonn(
                    model, batch, device,
                    ribonn_ens=ribonn_ens,
                    lambda_ribonn=args.lambda_ribonn,
                    use_gumbel=False,  # deterministic in validation
                )
                val_loss_sum += loss.item()
                val_n += 1

        val_loss = val_loss_sum / max(val_n, 1)
        print(f"Epoch {epoch} | val_loss={val_loss:.4f}", file=sys.stderr)

        if args.wandb:
            wandb.log({"val/loss": val_loss}, step=global_step)

        # checkpoint
        if args.output_path and val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict()}, args.output_path)
            print(f"[checkpoint] saved → {args.output_path}", file=sys.stderr)
        elif args.output_path and (epoch % args.save_every == 0):
            torch.save({"model_state_dict": model.state_dict()}, args.output_path + f"_epoch{epoch}")
            print(f"[checkpoint] saved → {args.output_path}_epoch{epoch}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()

    # data / model
    # parser.add_argument("--csv_path", type=str, default="../../data/finetuning/ribonn/dataset.csv")
    parser.add_argument("--csv_path", type=str, default="data/finetuning/ribonn/dataset.csv")
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)

    parser.add_argument("--max_utr5_len", type=int, default=200)
    parser.add_argument("--max_cds_len", type=int, default=500)
    parser.add_argument("--max_utr3_len", type=int, default=200)

    # training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=20)

    # indigo order
    parser.add_argument("--gen_order", type=str, default="random",
                        choices=["random", "l2r", "r2l", "inward", "outward", "sao"])
    parser.add_argument("--sao_beam_size", type=int, default=4)
    parser.add_argument("--sao_tokens_per_step", type=int, default=10)

    # ribonn guidance
    parser.add_argument("--ribonn_weights_folder", type=str, default=None)
    parser.add_argument("--ribonn_top_k", type=int, default=5)
    parser.add_argument("--lambda_ribonn", type=float, default=0.0)
    parser.add_argument("--ribonn_use_gumbel", action="store_true",
                        help="Use gumbel-softmax for RiboNN path (higher variance). Default uses softmax.")

    # logging/debug
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="indigo-ribonn-finetune")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--subset", type=int, default=0,
                        help="If >0, use only first N samples (local sanity run).")
    parser.add_argument("--debug_steps", type=int, default=0,
                        help="If >0, run only this many batches per epoch (local sanity run).")
    parser.add_argument("--verbose_ribonn", action="store_true")
    args = parser.parse_args()


    print("resolved:", Path(args.csv_path).resolve())

    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args), dir="../../logs")

    train(args, device)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()