import sys
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


COMMON_MFE_COLUMNS = ("mfe", "MFE", "minimum_free_energy")
COMMON_FULL_STRUCTURE_COLUMNS = ("structure", "dot_bracket", "dotbracket")
COMMON_UTR5_STRUCTURE_COLUMNS = ("utr5_structure", "utr5_dot_bracket", "utr5_dotbracket")
COMMON_CDS_STRUCTURE_COLUMNS = ("cds_structure", "cds_dot_bracket", "cds_dotbracket")
COMMON_UTR3_STRUCTURE_COLUMNS = ("utr3_structure", "utr3_dot_bracket", "utr3_dotbracket")


def _first_existing_column(df: pd.DataFrame, explicit: Optional[str], candidates: Tuple[str, ...]) -> Optional[str]:
    if explicit is not None:
        return explicit if explicit in df.columns else None
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _clean_structure(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    structure = str(value).strip()
    if not structure or structure.lower() == "nan":
        return None
    # RNAfold output is sometimes stored as: "....((..)) (-12.3)".
    return structure.split()[0]


def _paired_labels_from_structure(structure: Optional[str], length: int) -> List[float]:
    """Convert dot-bracket notation into 0/1 labels. -1 means unknown/masked."""
    labels = [-1.0] * length
    if structure is None:
        return labels

    usable = min(len(structure), length)
    for i, ch in enumerate(structure[:usable]):
        if ch == ".":
            labels[i] = 0.0
        elif ch in "()[]{}<>":
            labels[i] = 1.0
        else:
            labels[i] = -1.0
    return labels


def _normalise_mfe(value) -> float:
    raw = pd.to_numeric(value, errors="coerce")
    if pd.isna(raw):
        return float("nan")
    return float(raw)


class MRNACsvDataset(Dataset):
    def __init__(
        self,
        csv_path,
        max_utr5_len=2048,
        max_cds_len=8192,
        max_utr3_len=2048,
        only_utr5=False,
        use_rna_structure_labels: bool = False,
        mfe_column: Optional[str] = None,
        full_structure_column: Optional[str] = None,
        utr5_structure_column: Optional[str] = None,
        cds_structure_column: Optional[str] = None,
        utr3_structure_column: Optional[str] = None,
        normalise_mfe_by_length: bool = True,
    ):
        if csv_path.split(".")[-1] == "xlsx":
            dataframe = pd.read_excel(csv_path)
        else:
            dataframe = pd.read_csv(csv_path)

        self.df = dataframe
        self.max_utr5_len = max_utr5_len
        self.max_cds_len = max_cds_len
        self.max_utr3_len = max_utr3_len
        self.only_utr5 = only_utr5
        self.use_rna_structure_labels = use_rna_structure_labels
        self.normalise_mfe_by_length = normalise_mfe_by_length

        # full sequence length - 1 (because we shift by 1 when comparing input_ids and target_ids)
        # <BOS> + <CDS> + cds + <UTR5> + utr5 + [<UTR3> + utr3] + <EOS>
        if self.only_utr5:
            self.max_len = (max_utr5_len + max_cds_len + 4) - 1
        else:
            self.max_len = (max_utr5_len + max_cds_len + max_utr3_len + 5) - 1

        self.vocab = {
            "A": 0,
            "U": 1,
            "T": 1,
            "C": 2,
            "G": 3,
            "<PAD>": 4,
            "<BOS>": 5,
            "<EOS>": 6,
            "<CDS>": 7,
            "<UTR5>": 8,
        }
        if not self.only_utr5:
            self.vocab["<UTR3>"] = 9

        self.vocab_size = len(set(self.vocab.values()))
        self.pad_id = self.vocab["<PAD>"]
        self.bos_id = self.vocab["<BOS>"]
        self.eos_id = self.vocab["<EOS>"]
        self.cds_id = self.vocab["<CDS>"]
        self.utr5_id = self.vocab["<UTR5>"]
        self.utr3_id = self.vocab.get("<UTR3>")

        self.mfe_column = _first_existing_column(self.df, mfe_column, COMMON_MFE_COLUMNS)
        self.full_structure_column = _first_existing_column(
            self.df, full_structure_column, COMMON_FULL_STRUCTURE_COLUMNS
        )
        self.utr5_structure_column = _first_existing_column(
            self.df, utr5_structure_column, COMMON_UTR5_STRUCTURE_COLUMNS
        )
        self.cds_structure_column = _first_existing_column(
            self.df, cds_structure_column, COMMON_CDS_STRUCTURE_COLUMNS
        )
        self.utr3_structure_column = _first_existing_column(
            self.df, utr3_structure_column, COMMON_UTR3_STRUCTURE_COLUMNS
        )

        has_te = "te" in self.df.columns

        self.tokenized_samples = []
        self.skipped_count = 0

        print("Loading dataset...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            # 5'UTR generation should be right-to-left, hence the reverse.
            # Keep lengths in biological order for ViennaRNA structure mapping.
            utr5_bio_str = str(row["utr5"]).upper()[:self.max_utr5_len]
            utr5_str = utr5_bio_str[::-1]
            cds_str = str(row["cds"]).upper()
            utr3_str = str(row["utr3"]).upper()[:self.max_utr3_len]

            utr5_tokens = self.tokenize(utr5_str)
            cds_tokens = self.tokenize(cds_str)
            utr3_tokens = self.tokenize(utr3_str)
            utr5_len = len(utr5_tokens)
            cds_len = len(cds_tokens)
            utr3_len = len(utr3_tokens)

            if cds_len > self.max_cds_len:
                self.skipped_count += 1
                continue

            if utr5_len + cds_len + (0 if self.only_utr5 else utr3_len) >= self.max_len:
                self.skipped_count += 1
                continue

            te = float("nan")
            if has_te:
                raw = pd.to_numeric(row.get("te"), errors="coerce")
                if not pd.isna(raw):
                    te = float(raw)

            mfe = float("nan")
            if self.mfe_column is not None:
                mfe = _normalise_mfe(row.get(self.mfe_column))
                if self.normalise_mfe_by_length and math.isfinite(mfe):
                    seq_nt_len = max(utr5_len + cds_len + utr3_len, 1)
                    mfe = mfe / seq_nt_len

            paired_labels_full = None
            if self.use_rna_structure_labels:
                paired_labels_full = self._build_paired_labels(row, utr5_len, cds_len, utr3_len)

            self.tokenized_samples.append(
                (utr5_tokens, cds_tokens, utr3_tokens, te, mfe, paired_labels_full)
            )

        print(f"[dataset] kept={len(self.tokenized_samples)} skipped={self.skipped_count}")
        if self.mfe_column is not None:
            print(f"[dataset] using MFE column: {self.mfe_column}", file=sys.stderr)
        if self.use_rna_structure_labels:
            print(
                "[dataset] structure columns: "
                f"full={self.full_structure_column}, "
                f"utr5={self.utr5_structure_column}, "
                f"cds={self.cds_structure_column}, "
                f"utr3={self.utr3_structure_column}",
                file=sys.stderr,
            )

    def __len__(self):
        return len(self.tokenized_samples)

    def tokenize(self, seq):
        return [self.vocab.get(n, self.pad_id) for n in seq]

    def _build_position_maps(self, utr5_len: int, cds_len: int, utr3_len: int) -> Dict[int, int]:
        """
        Map biological nucleotide indices into training-token positions.

        ViennaRNA full structure is usually for biological order:
            UTR5 + CDS + UTR3

        Your training order is:
            <BOS> <CDS> CDS <UTR5> reversed(UTR5) [<UTR3> UTR3] <EOS>
        """
        prefix_len = 2 + cds_len + 1
        utr5_start_train = prefix_len
        utr3_start_train = prefix_len + utr5_len + (1 if not self.only_utr5 else 0)

        bio_to_train: Dict[int, int] = {}

        for i in range(utr5_len):
            bio_to_train[i] = utr5_start_train + (utr5_len - 1 - i)

        for i in range(cds_len):
            bio_to_train[utr5_len + i] = 2 + i

        if not self.only_utr5:
            for i in range(utr3_len):
                bio_to_train[utr5_len + cds_len + i] = utr3_start_train + i

        return bio_to_train

    def _build_paired_labels(self, row, utr5_len: int, cds_len: int, utr3_len: int) -> List[float]:
        # full_tokens length before shift can be max_len + 1
        seq_token_len = self.max_len + 1
        paired_labels = [-1.0] * seq_token_len

        if self.full_structure_column is not None:
            full_structure = _clean_structure(row.get(self.full_structure_column))
            bio_len = utr5_len + cds_len + (0 if self.only_utr5 else utr3_len)
            bio_paired = _paired_labels_from_structure(full_structure, bio_len)
            bio_to_train = self._build_position_maps(utr5_len, cds_len, utr3_len)

            for bio_idx, label in enumerate(bio_paired):
                train_idx = bio_to_train.get(bio_idx)
                if train_idx is not None and train_idx < seq_token_len:
                    paired_labels[train_idx] = label
            return paired_labels

        # Region-level structure columns. UTR5 structure is reversed to match reversed UTR5 generation.
        cds_structure = _clean_structure(row.get(self.cds_structure_column)) if self.cds_structure_column else None
        utr5_structure = _clean_structure(row.get(self.utr5_structure_column)) if self.utr5_structure_column else None
        utr3_structure = _clean_structure(row.get(self.utr3_structure_column)) if self.utr3_structure_column else None

        cds_labels = _paired_labels_from_structure(cds_structure, cds_len)
        for i, label in enumerate(cds_labels):
            paired_labels[2 + i] = label

        prefix_len = 2 + cds_len + 1
        utr5_labels_bio = _paired_labels_from_structure(utr5_structure, utr5_len)
        utr5_labels_train = list(reversed(utr5_labels_bio))
        for i, label in enumerate(utr5_labels_train):
            paired_labels[prefix_len + i] = label

        if not self.only_utr5:
            utr3_start = prefix_len + utr5_len + 1
            utr3_labels = _paired_labels_from_structure(utr3_structure, utr3_len)
            for i, label in enumerate(utr3_labels):
                paired_labels[utr3_start + i] = label

        return paired_labels

    def __getitem__(self, idx):
        utr5_tokens, cds_tokens, utr3_tokens, te, mfe, paired_labels_full = self.tokenized_samples[idx]

        # Prefix: <BOS> + <CDS> + CDS_tokens + <UTR5>
        prefix_tokens = [self.bos_id, self.cds_id] + cds_tokens + [self.utr5_id]
        # Target: UTR5_tokens + [<UTR3> + UTR3_tokens] + <EOS>
        target_tokens = utr5_tokens.copy()
        if not self.only_utr5:
            target_tokens += [self.utr3_id] + utr3_tokens
        target_tokens += [self.eos_id]

        full_tokens = prefix_tokens + target_tokens

        generation_start_idx = len(prefix_tokens)
        full_loss_mask = [0] * generation_start_idx + [1] * (len(full_tokens) - generation_start_idx)

        input_ids_list = full_tokens[:-1]
        target_ids_list = full_tokens[1:]
        loss_mask_list = full_loss_mask[1:]

        seq_len = len(input_ids_list)
        padding_length = self.max_len - seq_len

        input_ids = torch.tensor(input_ids_list + [self.pad_id] * padding_length, dtype=torch.long)
        target_ids = torch.tensor(target_ids_list + [self.pad_id] * padding_length, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask_list + [0] * padding_length, dtype=torch.float)

        padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        if padding_length > 0:
            padding_mask[-padding_length:] = True

        te_label = torch.tensor(te, dtype=torch.float)
        mfe_label = torch.tensor(mfe, dtype=torch.float)
        utr5_len = torch.tensor(len(utr5_tokens), dtype=torch.long)
        cds_len = torch.tensor(len(cds_tokens), dtype=torch.long)
        utr3_len = torch.tensor(len(utr3_tokens), dtype=torch.long)

        paired_labels = torch.full((self.max_len,), -1.0, dtype=torch.float)
        structure_loss_mask = torch.zeros(self.max_len, dtype=torch.float)

        if paired_labels_full is not None:
            input_paired = paired_labels_full[:-1][:seq_len]
            paired_labels[:seq_len] = torch.tensor(input_paired, dtype=torch.float)
            structure_loss_mask = (paired_labels >= 0).float()
            paired_labels = paired_labels.clamp(min=0.0)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "padding_mask": padding_mask,
            "te_label": te_label,
            "mfe_label": mfe_label,
            "paired_labels": paired_labels,
            "structure_loss_mask": structure_loss_mask,
            "utr5_len": utr5_len,
            "cds_len": cds_len,
            "utr3_len": utr3_len,
        }


class MRNATransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_layers,
        max_len,
        use_mfe_head: bool = False,
        use_paired_head: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_head = nn.Linear(d_model, vocab_size)

        self.mfe_head = (
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            if use_mfe_head
            else None
        )
        self.paired_head = nn.Linear(d_model, 1) if use_paired_head else None

    def forward(self, x, padding_mask, return_dict: bool = False):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)

        x_emb = self.embedding(x)
        pos_emb = self.pos_embedding(pos)

        hidden = x_emb + pos_emb
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        out = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)

        logits = self.seq_head(out)
        if not return_dict:
            # Backward-compatible behavior for old scripts.
            return logits

        mfe = None
        if self.mfe_head is not None:
            valid = (~padding_mask).float().unsqueeze(-1)
            pooled = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            mfe = self.mfe_head(pooled).squeeze(-1)

        paired_logits = None
        if self.paired_head is not None:
            paired_logits = self.paired_head(out).squeeze(-1)

        return {
            "logits": logits,
            "mfe": mfe,
            "paired_logits": paired_logits,
        }
