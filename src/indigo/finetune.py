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

def load_pretrained_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded {checkpoint_path}  missing={len(missing)}  unexpected={len(unexpected)}"
    )


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

# copilot wrote this and says that this is faster than what we did (loading a new model in every iteration)
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


from pretrain import build_full_R_matrix, compute_position_targets, make_generation_perm, _extend_R, _score_remaining, beam_search_perms, _extend_beam_state, extract_prefix_and_target


# ============================================================
# Building training tensors + collate (with SAO + metadata for RiboNN)
# ============================================================

# change: returns also perm
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

# change: returns also fields for ribonn
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

    Assumes the same special-token layout:
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

        # Full-seq indices
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

    # ---- position loss 
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
    
    if args.pretrained_path is not None:
        load_pretrained_weights(model, args.pretrained_path, device)
    else:
        print("No path to pretrained model given: training from scratch.")

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
        epoch_indigo_loss = 0.0
        epoch_ribonn_loss = 0.0
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
            epoch_indigo_loss += ind_l.item()
            epoch_ribonn_loss += rib_l.item()
            n_batches += 1

            if args.wandb and (step % args.log_every == 0):
                wandb.log({
                    "train/loss": loss.item(),
                    "train/indigo_loss": ind_l.item(),
                    "train/ribonn_loss": rib_l.item(),
                }, step=global_step)

        epoch_indigo_loss /= max(n_batches, 1)
        epoch_ribonn_loss /= max(n_batches, 1)
        epoch_loss /= max(n_batches, 1)
        print(f"Epoch {epoch} | train_loss={epoch_loss:.4f} indigo_loss={epoch_indigo_loss:.4f} ribonn_loss={epoch_ribonn_loss:.4f}", file=sys.stderr)

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
    parser.add_argument("--csv_path", type=str, default="../../data/finetuning/ribonn/dataset.csv")
    # parser.add_argument("--csv_path", type=str, default="data/finetuning/ribonn/dataset.csv")
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

    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrain.py checkpoint",
    )

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
    parser.add_argument("--wandb_project", type=str, default="indigo-finetuning")
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