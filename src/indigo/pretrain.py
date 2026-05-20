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
        else:
            p_indigo = i + gen_idx_map[sorted_placed[insert_idx - 1]]

        position_targets.append(p_indigo)
        sorted_placed.insert(insert_idx, cur_pos)
        gen_idx_map[cur_pos] = i

    return position_targets


def make_generation_perm(target_len, gen_order):
    """Build a generation permutation for a given order strategy.

    gen_order: 'random' | 'l2r' | 'r2l' | 'inward' | 'outward'
      - random: uniform random shuffle
      - l2r:    left-to-right (0, 1, ..., n-1)
      - r2l:    right-to-left (n-1, ..., 1, 0)
      - outward: center -> out (mid, mid+1, mid-1, mid+2, mid-2, ...)
      - inward:  out -> center (0, n-1, 1, n-2, ..., mid)
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
    """Extend an (n, n) R matrix to (n+1, n+1) by appending one token.

    new_abs_pos_val: int, absolute position of the new token
    abs_pos_prev: LongTensor of shape (n,)
    """
    new_col = torch.sign(new_abs_pos_val - abs_pos_prev)  # R[:, new] = sign(pos_new - pos_i)
    new_row = -new_col                                    # R[new, :] = sign(pos_i - pos_new)
    R_new = torch.cat([R_prev, new_col.unsqueeze(1)], dim=1)
    new_row_full = torch.cat([new_row, torch.zeros(1, dtype=torch.long)])
    return torch.cat([R_new, new_row_full.unsqueeze(0)], dim=0)


def _score_remaining(beam, remaining_list, log_prob_word_b, W, state_b, keys_b,
                     pos_valid_base, target_tokens, device):
    """Compute combined word + position scores for every remaining token in a beam.

    Uses the key matrix from the most recent outer forward pass (size 2*t_outer).
    Tokens that should be inserted relative to a token added *within* the current
    greedy chunk (gen_step >= t_outer) have no corresponding key in keys_b; their
    position score is set to 0 so only the word score contributes.

    Returns: tensor of shape (n_rem,) — does NOT include beam["score"].
    """
    t_outer = keys_b.size(0) // 2  # number of tokens present at the forward-pass step

    tokens_rem = torch.tensor(
        [target_tokens[i] for i in remaining_list],
        dtype=torch.long, device=device,
    )  # (n_rem,)

    word_scores = log_prob_word_b[tokens_rem]  # (n_rem,)

    # Per-beam EOS validity: cannot insert to the right of EOS.
    # The right key of the token placed at gen_step e is at column t_outer + e.
    eos_col = (t_outer + beam["eos_step"]) if beam["eos_step"] is not None else -1
    pos_valid = pos_valid_base.clone()
    if 0 <= eos_col < pos_valid.size(0):
        pos_valid[eos_col] = False

    # Insertion slot for each remaining token given the current sorted order.
    sorted_placed_t = torch.tensor(beam["sorted_placed"], dtype=torch.long, device=device)
    gen_steps_t     = torch.tensor(
        [beam["gen_idx_map"][sp] for sp in beam["sorted_placed"]],
        dtype=torch.long, device=device,
    )
    remaining_t = torch.tensor(remaining_list, dtype=torch.long, device=device)

    pos_after_insert = (sorted_placed_t.unsqueeze(0) < remaining_t.unsqueeze(1)).sum(dim=1)
    predecessor_idx  = torch.clamp(pos_after_insert - 1, min=0)

    # For "before everything": LEFT key of first placed token (column = gen_steps_t[0]).
    # For "after predecessor":  RIGHT key of predecessor  (column = t_outer + gen_steps_t[pred]).
    ref_gen_step = torch.where(
        pos_after_insert == 0,
        gen_steps_t[0].expand(len(remaining_list)),
        gen_steps_t[predecessor_idx],
    )  # (n_rem,)

    # Slots referencing greedy-chunk tokens (ref_gen_step >= t_outer) have no key.
    slot_has_key = ref_gen_step < t_outer

    insertion_slots = torch.where(
        pos_after_insert == 0,
        gen_steps_t[0].expand(len(remaining_list)),
        t_outer + gen_steps_t[predecessor_idx],
    )  # (n_rem,)
    clamped_slots = insertion_slots.clamp(0, 2 * t_outer - 1)

    # Position scores.
    queries   = state_b.unsqueeze(0) + W[tokens_rem]           # (n_rem, d_model)
    pos_logits = queries @ keys_b.T                             # (n_rem, 2*t_outer)
    pos_logits = pos_logits.masked_fill(~pos_valid.unsqueeze(0), float("-inf"))
    log_p_pos  = F.log_softmax(pos_logits, dim=-1)              # (n_rem, 2*t_outer)
    raw_pos    = log_p_pos[torch.arange(len(remaining_list), device=device), clamped_slots]

    # Zero out position score when the slot has no key or the slot itself is masked.
    pos_scores = torch.where(
        slot_has_key & pos_valid[clamped_slots],
        raw_pos,
        torch.zeros_like(raw_pos),
    )

    return word_scores + pos_scores


def _extend_beam_state(beam, token_idx, new_score, current_t, target_tokens, eos_id, prefix_len):
    """Return a new beam dict with token_idx appended at generation step current_t."""
    new_sorted = beam["sorted_placed"].copy()
    ins = bisect.bisect_left(new_sorted, token_idx)
    new_sorted.insert(ins, token_idx)

    new_gen_idx_map = dict(beam["gen_idx_map"])
    new_gen_idx_map[token_idx] = current_t

    eos_step = beam["eos_step"]
    if eos_step is None and target_tokens[token_idx] == eos_id:
        eos_step = current_t

    new_R       = _extend_R(beam["R"], beam["abs_pos"], prefix_len + token_idx)
    new_abs_pos = torch.cat([beam["abs_pos"], torch.tensor([prefix_len + token_idx])])

    return {
        "perm":         beam["perm"] + [token_idx],
        "sorted_placed": new_sorted,
        "gen_idx_map":  new_gen_idx_map,
        "remaining":    beam["remaining"] - {token_idx},
        "score":        new_score,
        "eos_step":     eos_step,
        "R":            new_R,
        "abs_pos":      new_abs_pos,
    }


@torch.no_grad()
def beam_search_perms(model, batch_prefix_tokens, batch_target_tokens, eos_id, beam_size, device, n_init=15, tokens_per_step=10):
    N = len(batch_prefix_tokens)
    prefix_lens = [len(p) for p in batch_prefix_tokens]
    target_lens = [len(t) for t in batch_target_tokens]
    W = model.get_embedding_matrix()  # (vocab_size, d_model)
    n_init = min(n_init, min(target_lens))

    # --------- Step 0 ---------
    # Since we only have 4 nucleotide types, we cannot really determine what are the best first tokens to generate.
    # So instead we will sample n_init starting positions and use them as initial permutation (sorted).

    # beams[sample_idx] = list of beam dicts for sample sample_idx
    beams = []
    for sample_idx in range(N):
        prefix_len = prefix_lens[sample_idx]
        target_tokens = batch_target_tokens[sample_idx]
        target_len = target_lens[sample_idx]
        abs_pos_prefix = torch.arange(prefix_len, dtype=torch.long)

        sample_beams = []
        for _ in range(beam_size):
            step0_perm = sorted(random.sample(range(target_len), n_init))
            R = build_full_R_matrix(prefix_len, n_init, step0_perm)
            abs_pos = torch.cat([
                abs_pos_prefix,
                prefix_len + torch.tensor(step0_perm, dtype=torch.long),
            ])

            # For simplicity, we completely ignore the initial score. This shouldn't be fine, as the initial tokens are random.
            initial_score = 0.0

            eos_step = (n_init - 1) if target_tokens[step0_perm[-1]] == eos_id else None

            sample_beams.append({
                "perm": step0_perm, # Permutation of the target tokens
                "sorted_placed": step0_perm, # Sorted list of placed tokens
                "gen_idx_map": {idx: k for k, idx in enumerate(step0_perm)}, # Mapping from target token index to generation index
                "remaining": set(range(target_len)) - set(step0_perm), # Remaining target tokens
                "score": initial_score, # Score of the beam
                "eos_step": eos_step, # At which step EOS was placed (None if not placed yet)
                "R": R, # R matrix of size (prefix_len + n_init, prefix_len + n_init)
                "abs_pos": abs_pos, # absolute positions of all tokens so far
            })
        beams.append(sample_beams)

    # --- Steps n_init to max_target_len - 1
    # We are dealing with much longer sequences than the authors of the paper,
    # so the slowdown is much bigger then they reported.
    # To mitigate this, we generate tokens_per_step tokens instead of a single one at each step.
    t = n_init
    while t < max(target_lens):
        # Flatten all active beams across all samples into one batch
        all_beams = []  # list of (sample_idx, beam_dict)
        for sample_idx in range(N):
            if target_lens[sample_idx] <= t:
                continue
            for beam in beams[sample_idx]:
                all_beams.append((sample_idx, beam))

        if len(all_beams) == 0: # This can only happen if all target_lens <= t
            break

        n_all_beams = len(all_beams)
        max_seq_len  = max(prefix_lens[sample_idx] + t for sample_idx, _ in all_beams)

        # Run model on all beams batched
        ids_batch = torch.zeros(n_all_beams, max_seq_len, dtype=torch.long, device=device)
        R_batch = torch.zeros(n_all_beams, max_seq_len, max_seq_len, dtype=torch.long, device=device)
        padding_mask_batch = torch.ones(n_all_beams, max_seq_len, dtype=torch.bool, device=device)
        for beam_idx, (sample_idx, beam) in enumerate(all_beams):
            prefix_len = prefix_lens[sample_idx]
            seq_len = prefix_len + t
            tokens = batch_prefix_tokens[sample_idx] + [batch_target_tokens[sample_idx][p] for p in beam["perm"]]
            ids_batch[beam_idx, :seq_len]       = torch.tensor(tokens, dtype=torch.long, device=device)
            R_batch[beam_idx, :seq_len, :seq_len] = beam["R"].to(device)
            padding_mask_batch[beam_idx, :seq_len] = False

        H_batch_full, _, word_logits_full = model(ids_batch, R_batch, padding_mask_batch)

        # Distribute results and score candidates per sample
        
        # Initialize new_beams_all with None for samples that need processing
        # Keep finished samples as they are.
        new_beams_all = [
            beams[sample_idx] if target_lens[sample_idx] <= t else None 
            for sample_idx in range(N)
        ]
        sample_offset = 0  # running index into the flattened batch

        for sample_idx in range(N):
            if target_lens[sample_idx] <= t:
                continue

            prefix_len = prefix_lens[sample_idx]
            target_tokens = batch_target_tokens[sample_idx]
            sample_beams = beams[sample_idx]
            n_beams = len(sample_beams)

            # Slice this sample's rows from the combined forward-pass output
            H_batch = H_batch_full[sample_offset:sample_offset + n_beams]
            word_logits_batch = word_logits_full[sample_offset:sample_offset + n_beams]
            sample_offset += n_beams

            h = H_batch[:, prefix_len + t - 1, :] # (n_beams, d_model)
            log_prob_word = F.log_softmax(word_logits_batch[:, prefix_len + t - 1, :], dim=-1) # (n_beams, vocab)

            # Position keys from all t placed tokens (n_beams, 2t, d_model)
            H_gen = H_batch[:, prefix_len:prefix_len + t, :]
            left_keys  = model.position_head_left_proj(H_gen)
            right_keys = model.position_head_right_proj(H_gen)
            keys  = torch.cat([left_keys, right_keys], dim=1) # (n_beams, 2t, d_model)
            state = model.position_head_state_proj(h) # (n_beams, d_model)

            # Base validity mask: before BOS is always invalid
            pos_valid_base = torch.ones(2 * t, dtype=torch.bool, device=device)
            pos_valid_base[0] = False

            all_candidates = []  # (combined_score, b_idx, token_idx)

            for b_idx, beam in enumerate(sample_beams):
                remaining_list = list(beam["remaining"])
                scores = _score_remaining(
                    beam, remaining_list, log_prob_word[b_idx], W,
                    state[b_idx], keys[b_idx], pos_valid_base, target_tokens, device,
                )
                for k, token_idx in enumerate(remaining_list):
                    all_candidates.append((beam["score"] + scores[k].item(), b_idx, token_idx))

            # Update beams with new best candidates
            all_candidates.sort(key=lambda x: -x[0])
            top = all_candidates[:beam_size]

            new_beams = []
            for score, b_idx, token_idx in top:
                extended_beam = _extend_beam_state(
                    sample_beams[b_idx], token_idx, score, t, target_tokens, eos_id, prefix_len,
                )

                # Greedy extension: place up to tokens_per_step-1 more tokens
                # reusing the same forward-pass outputs (keys, state, log_prob_word).
                # Vectorised: score all remaining tokens once, argsort → greedy order.
                # R/bookkeeping are only rebuilt when another outer forward pass follows;
                # for the final chunk only the perm is needed, so we skip all of it.
                remaining_list_k = list(extended_beam["remaining"])
                n_greedy = min(
                    tokens_per_step - 1,
                    max(0, target_lens[sample_idx] - t - 1),
                    len(remaining_list_k),
                )
                if n_greedy > 0:
                    scores_k = _score_remaining(
                        extended_beam, remaining_list_k, log_prob_word[b_idx], W,
                        state[b_idx], keys[b_idx], pos_valid_base, target_tokens, device,
                    )
                    order_k       = scores_k.argsort(descending=True).tolist()
                    greedy_tokens = [remaining_list_k[i] for i in order_k[:n_greedy]]
                    new_remaining = extended_beam["remaining"] - set(greedy_tokens)
                    greedy_score  = extended_beam["score"] + scores_k[order_k[:n_greedy]].sum().item()

                    will_have_next_outer = (t + tokens_per_step < target_lens[sample_idx])
                    if will_have_next_outer:
                        # Full bookkeeping needed so the next outer pass sees a valid beam.
                        new_sorted_k  = extended_beam["sorted_placed"].copy()
                        new_gim_k     = dict(extended_beam["gen_idx_map"])
                        eos_step_k    = extended_beam["eos_step"]
                        new_R_k       = extended_beam["R"]
                        new_abs_pos_k = extended_beam["abs_pos"]
                        for sk, tok in enumerate(greedy_tokens):
                            current_t = t + 1 + sk
                            ins = bisect.bisect_left(new_sorted_k, tok)
                            new_sorted_k.insert(ins, tok)
                            new_gim_k[tok] = current_t
                            if eos_step_k is None and target_tokens[tok] == eos_id:
                                eos_step_k = current_t
                            new_R_k       = _extend_R(new_R_k, new_abs_pos_k, prefix_len + tok)
                            new_abs_pos_k = torch.cat([new_abs_pos_k, torch.tensor([prefix_len + tok])])
                        extended_beam = {
                            "perm":          extended_beam["perm"] + greedy_tokens,
                            "sorted_placed": new_sorted_k,
                            "gen_idx_map":   new_gim_k,
                            "remaining":     new_remaining,
                            "score":         greedy_score,
                            "eos_step":      eos_step_k,
                            "R":             new_R_k,
                            "abs_pos":       new_abs_pos_k,
                        }
                    else:
                        # Final chunk: skip R/bookkeeping, the perm is all that matters.
                        extended_beam = {
                            "perm":          extended_beam["perm"] + greedy_tokens,
                            "sorted_placed": extended_beam["sorted_placed"],
                            "gen_idx_map":   extended_beam["gen_idx_map"],
                            "remaining":     new_remaining,
                            "score":         greedy_score,
                            "eos_step":      extended_beam["eos_step"],
                            "R":             extended_beam["R"],
                            "abs_pos":       extended_beam["abs_pos"],
                        }

                new_beams.append(extended_beam)

            new_beams_all[sample_idx] = new_beams

        beams = new_beams_all
        t += tokens_per_step

    return [[beam["perm"] for beam in beams[sample_idx]] for sample_idx in range(N)]


def build_training_tensors(prefix_tokens, target_tokens, eos_id, gen_order="random", perm=None):
    """Build permuted training tensors for one sample.

    Returns dict with:
      - input_ids: (seq_len,) prefix + permuted target (no padding)
      - R: (seq_len, seq_len) full relative position matrix
      - word_targets: (target_len,) the token at each generation step
      - pos_targets: (target_len - 1,) the INDIGO position at each step
      - eos_gen_idx: generation index of the EOS token (for position masking)
      - prefix_len: int
      - target_len: int

    perm: if provided, used directly instead of sampling via gen_order.
    """

    target_len = len(target_tokens)
    if perm is None:
        perm = make_generation_perm(target_len, gen_order)

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


def collate_indigo_batch(samples, pad_id, eos_id, gen_order="random",
                         model=None, device=None, sao_beam_size=4, sao_tokens_per_step=1):
    """Collate a list of dataset samples into a padded batch for INDIGO training.

    Each sample is processed through extract_prefix_and_target + build_training_tensors,
    then padded to the maximum lengths in the batch.

    For gen_order='sao', model and device must be supplied.  beam_search_perms is
    called per sample; the resulting beam_size permutations are each built into
    tensors and flattened into the batch (effective batch size *= sao_beam_size).

    Returns a dict of batched tensors, or None if all samples were skipped.
    """
    batch_tensors = []
    if gen_order == "sao":
        samples_with_targets = []
        for sample in samples:
            prefix_tokens, target_tokens = extract_prefix_and_target(sample, pad_id)
            if len(target_tokens) >= 2:
                samples_with_targets.append((prefix_tokens, target_tokens))
        if not samples_with_targets:
            return None
        all_perms = beam_search_perms(
            model,
            [p for p, _ in samples_with_targets],
            [t for _, t in samples_with_targets],
            eos_id, sao_beam_size, device,
            tokens_per_step=sao_tokens_per_step,
        )
        for (prefix_tokens, target_tokens), perms in zip(samples_with_targets, all_perms):
            for p in perms:
                batch_tensors.append(
                    build_training_tensors(prefix_tokens, target_tokens, eos_id, perm=p)
                )
    else:
        for sample in samples:
            prefix_tokens, target_tokens = extract_prefix_and_target(sample, pad_id)
            if len(target_tokens) < 2:
                continue
            tensors = build_training_tensors(prefix_tokens, target_tokens, eos_id, gen_order)
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
            batch = collate_indigo_batch(
                samples, pad_id, eos_id, args.gen_order,
                model=model, device=device, sao_beam_size=args.sao_beam_size,
            )
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

            if args.gen_order == "sao":
                model.eval()
                batch = collate_indigo_batch(
                    samples, pad_id, eos_id, args.gen_order,
                    model=model, device=device, sao_beam_size=args.sao_beam_size,
                    sao_tokens_per_step=args.sao_tokens_per_step,
                )
                model.train()
            else:
                batch = collate_indigo_batch(samples, pad_id, eos_id, args.gen_order)
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
    parser.add_argument("--gen_order", type=str, default="random",
                        choices=["random", "l2r", "r2l", "inward", "outward", "sao"],
                        help="Generation order: random | l2r | r2l | inward | outward | sao")
    parser.add_argument("--sao_beam_size", type=int, default=4,
                        help="Beam size for SAO order search (only used when --gen_order sao)")
    parser.add_argument("--sao_tokens_per_step", type=int, default=1,
                        help="Tokens placed per beam-search forward pass (1=original, K>1 reduces forward passes by K-fold)")
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
