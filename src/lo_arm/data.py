import math
import random
from itertools import product

import pandas as pd
import torch
from torch.utils.data import Dataset


REGION_PAD = 0
REGION_SPECIAL = 1
REGION_CDS = 2
REGION_UTR5 = 3
REGION_UTR3 = 4
NUM_REGIONS = 5


class LoArmTokenizer:
    """K-mer tokenizer with a distinct LO-ARM sampling mask token."""

    def __init__(self, k=3, only_utr5=False):
        self.k = k
        self.only_utr5 = only_utr5

        kmers = ["".join(p) for p in product("ATCG", repeat=k)]
        self.vocab = {kmer: i for i, kmer in enumerate(kmers)}

        n_kmer = len(self.vocab)
        self.vocab["<PAD>"] = n_kmer
        self.vocab["<BOS>"] = n_kmer + 1
        self.vocab["<EOS>"] = n_kmer + 2
        self.vocab["<CDS>"] = n_kmer + 3
        self.vocab["<UTR5>"] = n_kmer + 4
        if not only_utr5:
            self.vocab["<UTR3>"] = n_kmer + 5
            self.vocab["<MASK>"] = n_kmer + 6
        else:
            self.vocab["<MASK>"] = n_kmer + 5

        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        self.pad_id = self.vocab["<PAD>"]
        self.bos_id = self.vocab["<BOS>"]
        self.eos_id = self.vocab["<EOS>"]
        self.cds_id = self.vocab["<CDS>"]
        self.utr5_id = self.vocab["<UTR5>"]
        self.utr3_id = self.vocab.get("<UTR3>")
        self.mask_id = self.vocab["<MASK>"]
        self.special_ids = {
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.cds_id,
            self.utr5_id,
            self.mask_id,
        }
        if self.utr3_id is not None:
            self.special_ids.add(self.utr3_id)

    def tokenize(self, seq):
        seq = str(seq).upper().replace("U", "T")
        return [
            self.vocab.get(seq[i : i + self.k], self.pad_id)
            for i in range(0, len(seq) - self.k + 1, self.k)
        ]

    def detokenize(self, ids):
        parts = []
        for idx in ids:
            idx = int(idx)
            if idx in self.special_ids:
                continue
            parts.append(self.id_to_token.get(idx, ""))
        return "".join(parts)


class LayoutPrior:
    """Empirical joint prior over token-space sequence layouts."""

    def __init__(self, layouts):
        self.layouts = [self._normalize(layout) for layout in layouts]
        if not self.layouts:
            raise ValueError("LayoutPrior requires at least one layout")

        self.by_cds_len = {}
        for layout in self.layouts:
            self.by_cds_len.setdefault(layout["cds_len"], []).append(layout)

    @staticmethod
    def _normalize(layout):
        if isinstance(layout, dict):
            utr5_len = int(layout["utr5_len"])
            cds_len = int(layout["cds_len"])
            utr3_len = int(layout["utr3_len"])
            total_len = int(layout.get("total_len", utr5_len + cds_len + utr3_len))
        else:
            if len(layout) != 4:
                raise ValueError("layout tuples must be (total_len, utr5_len, cds_len, utr3_len)")
            total_len, utr5_len, cds_len, utr3_len = [int(value) for value in layout]
        expected_total = utr5_len + cds_len + utr3_len
        if total_len != expected_total:
            total_len = expected_total
        return {
            "total_len": total_len,
            "utr5_len": utr5_len,
            "cds_len": cds_len,
            "utr3_len": utr3_len,
        }

    @classmethod
    def from_dataset(cls, dataset):
        return cls(dataset.layout_tuples)

    def __len__(self):
        return len(self.layouts)

    def to_serializable(self):
        return [dict(layout) for layout in self.layouts]

    def sample(self, cds_len, rng=None):
        rng = rng or random
        cds_len = int(cds_len)
        bucket = self.by_cds_len.get(cds_len)
        if bucket is None:
            nearest = min(self.by_cds_len, key=lambda value: (abs(value - cds_len), value))
            bucket = self.by_cds_len[nearest]
            layout = dict(rng.choice(bucket))
            layout["cds_len"] = cds_len
            layout["total_len"] = layout["utr5_len"] + cds_len + layout["utr3_len"]
            return layout
        return dict(rng.choice(bucket))


class MRNALoArmDataset(Dataset):
    """Dataset for layout-prior LO-ARM conditional UTR infilling."""

    def __init__(
        self,
        csv_path,
        max_utr5_len=200,
        max_cds_len=500,
        max_utr3_len=200,
        only_utr5=False,
        k=3,
    ):
        if str(csv_path).split(".")[-1] == "xlsx":
            dataframe = pd.read_excel(csv_path)
        else:
            dataframe = pd.read_csv(csv_path)

        self.df = dataframe
        self.max_utr5_len = max_utr5_len
        self.max_cds_len = max_cds_len
        self.max_utr3_len = max_utr3_len
        self.only_utr5 = only_utr5
        self.k = k
        self.tokenizer = LoArmTokenizer(k=k, only_utr5=only_utr5)

        self.vocab = self.tokenizer.vocab
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_id
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        self.cds_id = self.tokenizer.cds_id
        self.utr5_id = self.tokenizer.utr5_id
        self.utr3_id = self.tokenizer.utr3_id
        self.mask_id = self.tokenizer.mask_id

        self.max_utr5_tokens = math.ceil(max_utr5_len / k)
        self.max_cds_tokens = math.ceil(max_cds_len / k)
        self.max_utr3_tokens = math.ceil(max_utr3_len / k)
        self.prefix_len = 2 + self.max_cds_tokens + 1
        self.target_len = self.max_utr5_tokens
        if not self.only_utr5:
            self.target_len += self.max_utr3_tokens
        self.max_len = self.prefix_len + self.target_len

        self.samples = []
        self.layout_tuples = []
        self.skipped_count = 0
        for _, row in self.df.iterrows():
            cds_raw = self._clean_seq(row["cds"])
            utr5_raw = self._clean_seq(row["utr5"])
            utr3_raw = "" if only_utr5 else self._clean_seq(row["utr3"])

            cds_tokens = self.tokenizer.tokenize(cds_raw)
            if len(cds_tokens) > self.max_cds_tokens:
                self.skipped_count += 1
                continue

            utr5_tokens = self.tokenizer.tokenize(utr5_raw[::-1][:max_utr5_len])
            utr3_tokens = self.tokenizer.tokenize(utr3_raw[:max_utr3_len])
            if (
                self.pad_id in cds_tokens
                or self.pad_id in utr5_tokens
                or self.pad_id in utr3_tokens
            ):
                self.skipped_count += 1
                continue

            if len(utr5_tokens) > self.max_utr5_tokens or len(utr3_tokens) > self.max_utr3_tokens:
                self.skipped_count += 1
                continue

            real_target_len = len(utr5_tokens) + len(utr3_tokens)
            if real_target_len <= 0 or real_target_len > self.target_len:
                self.skipped_count += 1
                continue

            target_tokens = list(utr5_tokens) + list(utr3_tokens)
            target_tokens += [self.pad_id] * (self.target_len - len(target_tokens))
            target_order_mask = [True] * real_target_len + [False] * (self.target_len - real_target_len)
            target_region_ids = (
                [REGION_UTR5] * len(utr5_tokens)
                + [REGION_UTR3] * len(utr3_tokens)
                + [REGION_PAD] * (self.target_len - real_target_len)
            )
            layout = {
                "total_len": len(utr5_tokens) + len(cds_tokens) + len(utr3_tokens),
                "utr5_len": len(utr5_tokens),
                "cds_len": len(cds_tokens),
                "utr3_len": len(utr3_tokens),
            }

            sample_id = row.get("id", len(self.samples))
            self.samples.append(
                {
                    "id": str(sample_id),
                    "cds": cds_raw,
                    "utr5": utr5_raw,
                    "utr3": utr3_raw,
                    "cds_tokens": cds_tokens,
                    "target_tokens": target_tokens,
                    "target_order_mask": target_order_mask,
                    "target_region_ids": target_region_ids,
                    "layout": layout,
                }
            )
            self.layout_tuples.append(dict(layout))
        self.layout_prior = LayoutPrior(self.layout_tuples) if self.layout_tuples else None

    @staticmethod
    def _clean_seq(value):
        if pd.isna(value):
            return ""
        return str(value).upper().replace("U", "T")

    def __len__(self):
        return len(self.samples)

    def build_prefix(self, cds_tokens):
        prefix = [self.bos_id, self.cds_id] + list(cds_tokens) + [self.utr5_id]
        padding_length = self.prefix_len - len(prefix)
        if padding_length < 0:
            raise ValueError("CDS prefix is longer than prefix canvas")
        prefix_ids = prefix + [self.pad_id] * padding_length
        prefix_padding_mask = [False] * len(prefix) + [True] * padding_length
        prefix_region_ids = (
            [REGION_SPECIAL, REGION_SPECIAL]
            + [REGION_CDS] * len(cds_tokens)
            + [REGION_SPECIAL]
            + [REGION_PAD] * padding_length
        )
        return prefix_ids, prefix_padding_mask, prefix_region_ids

    def encode_cds_prefix(self, cds_seq):
        cds_tokens = self.tokenizer.tokenize(str(cds_seq).upper().replace("U", "T"))
        if len(cds_tokens) > self.max_cds_tokens:
            cds_tokens = cds_tokens[: self.max_cds_tokens]
        prefix_ids, prefix_padding_mask, prefix_region_ids = self.build_prefix(cds_tokens)
        return prefix_ids, prefix_padding_mask, prefix_region_ids

    def decode_target(self, target_ids, layout=None):
        target_ids = [int(x) for x in target_ids]
        if layout is not None:
            utr5_len = int(layout["utr5_len"])
            utr3_len = int(layout["utr3_len"])
            utr5_tokens = target_ids[:utr5_len]
            utr3_tokens = target_ids[utr5_len : utr5_len + utr3_len]
            utr5 = self.tokenizer.detokenize(utr5_tokens)[::-1]
            utr3 = self.tokenizer.detokenize(utr3_tokens)
            return utr5, utr3

        # Legacy full-canvas decode for old checkpoints/tests.
        if self.eos_id in target_ids:
            target_ids = target_ids[: target_ids.index(self.eos_id)]

        if self.utr3_id is not None and self.utr3_id in target_ids:
            split_idx = target_ids.index(self.utr3_id)
            utr5_tokens = target_ids[:split_idx]
            utr3_tokens = target_ids[split_idx + 1 :]
        else:
            utr5_tokens = target_ids
            utr3_tokens = []

        utr5 = self.tokenizer.detokenize(utr5_tokens)[::-1]
        utr3 = self.tokenizer.detokenize(utr3_tokens)
        return utr5, utr3

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prefix_ids, prefix_padding_mask, prefix_region_ids = self.build_prefix(sample["cds_tokens"])
        layout = sample["layout"]
        return {
            "id": sample["id"],
            "cds": sample["cds"],
            "utr5": sample["utr5"],
            "utr3": sample["utr3"],
            "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
            "prefix_padding_mask": torch.tensor(prefix_padding_mask, dtype=torch.bool),
            "prefix_region_ids": torch.tensor(prefix_region_ids, dtype=torch.long),
            "target_ids": torch.tensor(sample["target_tokens"], dtype=torch.long),
            "target_order_mask": torch.tensor(sample["target_order_mask"], dtype=torch.bool),
            "target_region_ids": torch.tensor(sample["target_region_ids"], dtype=torch.long),
            "layout": {key: torch.tensor(value, dtype=torch.long) for key, value in layout.items()},
        }


def build_model_inputs(
    prefix_ids,
    prefix_padding_mask,
    target_canvas,
    prefix_region_ids=None,
    target_region_ids=None,
    target_order_mask=None,
):
    input_ids = torch.cat([prefix_ids, target_canvas], dim=1)
    if target_order_mask is None:
        target_padding_mask = torch.zeros(
            target_canvas.shape, dtype=torch.bool, device=target_canvas.device
        )
    else:
        target_padding_mask = ~target_order_mask.to(device=target_canvas.device)
    padding_mask = torch.cat([prefix_padding_mask, target_padding_mask], dim=1)
    if prefix_region_ids is None or target_region_ids is None:
        return input_ids, padding_mask
    region_ids = torch.cat(
        [
            prefix_region_ids.to(device=target_canvas.device),
            target_region_ids.to(device=target_canvas.device),
        ],
        dim=1,
    )
    return input_ids, padding_mask, region_ids
