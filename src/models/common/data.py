import math
from itertools import product

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


_SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<CDS>", "<UTR5>")


class MRNATokenizer:
    """Non-overlapping DNA k-mer tokenizer shared by autoregressive models."""

    def __init__(self, only_utr5=False, k=1, include_u_alias=True):
        self.only_utr5 = only_utr5
        self.k = k
        self.include_u_alias = include_u_alias

        kmers = ["".join(parts) for parts in product("ATCG", repeat=k)]
        if k == 1 and include_u_alias:
            self.vocab = {"A": 0, "U": 1, "T": 1, "C": 2, "G": 3}
        else:
            self.vocab = {kmer: idx for idx, kmer in enumerate(kmers)}

        next_id = len(kmers)
        for token in _SPECIAL_TOKENS:
            self.vocab[token] = next_id
            next_id += 1
        if not only_utr5:
            self.vocab["<UTR3>"] = next_id

        self.id_to_token = {
            idx: token for token, idx in self.vocab.items() if token != "U"
        }
        self.vocab_size = len(set(self.vocab.values()))
        self.pad_id = self.vocab["<PAD>"]
        self.bos_id = self.vocab["<BOS>"]
        self.eos_id = self.vocab["<EOS>"]
        self.cds_id = self.vocab["<CDS>"]
        self.utr5_id = self.vocab["<UTR5>"]
        self.utr3_id = self.vocab.get("<UTR3>")
        self.special_ids = {
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.cds_id,
            self.utr5_id,
        }
        if self.utr3_id is not None:
            self.special_ids.add(self.utr3_id)

    def tokenize(self, seq):
        seq = str(seq).upper().replace("U", "T")
        return [
            self.vocab.get(seq[i : i + self.k], self.pad_id)
            for i in range(0, len(seq) - self.k + 1, self.k)
        ]

    def detokenize(self, token_ids):
        return "".join(
            self.id_to_token.get(int(token_id), "")
            for token_id in token_ids
            if int(token_id) not in self.special_ids
        )


MRNA_VOCAB = MRNATokenizer(k=1).vocab.copy()


class MRNACsvDataset(Dataset):
    """Autoregressive CDS-conditioned UTR dataset used by Transformer and InDIGO."""

    def __init__(
        self,
        csv_path,
        max_utr5_len=2048,
        max_cds_len=8192,
        max_utr3_len=2048,
        only_utr5=False,
        k=3,
        tokenizer=None,
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
        self.tokenizer = tokenizer or MRNATokenizer(
            k=k, only_utr5=only_utr5, include_u_alias=False
        )
        if self.tokenizer.k != k or self.tokenizer.only_utr5 != only_utr5:
            raise ValueError("Tokenizer configuration must match dataset k and only_utr5")

        self.vocab = self.tokenizer.vocab
        self.vocab_size = self.tokenizer.vocab_size

        max_utr5_tokens = math.ceil(max_utr5_len / k)
        max_cds_tokens = math.ceil(max_cds_len / k)
        max_utr3_tokens = math.ceil(max_utr3_len / k)
        self.max_cds_tokens = max_cds_tokens

        if self.only_utr5:
            self.max_len = (max_utr5_tokens + max_cds_tokens + 4) - 1
        else:
            self.max_len = (
                max_utr5_tokens + max_cds_tokens + max_utr3_tokens + 5
            ) - 1

        self.pad_id = self.tokenizer.pad_id
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        self.cds_id = self.tokenizer.cds_id
        self.utr5_id = self.tokenizer.utr5_id
        self.utr3_id = self.tokenizer.utr3_id

        has_te = "te" in self.df.columns
        self.tokenized_samples = []
        self.skipped_count = 0

        print("Loading dataset...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            utr5_str = str(row["utr5"])[::-1][: self.max_utr5_len].upper()
            cds_str = str(row["cds"]).upper()
            utr3_str = str(row["utr3"])[: self.max_utr3_len].upper()

            utr5_tokens = self.tokenizer.tokenize(utr5_str)
            cds_tokens = self.tokenizer.tokenize(cds_str)
            utr3_tokens = self.tokenizer.tokenize(utr3_str)
            utr5_len = len(utr5_tokens)
            cds_len = len(cds_tokens)
            utr3_len = len(utr3_tokens)

            if cds_len > self.max_cds_tokens:
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

            self.tokenized_samples.append((utr5_tokens, cds_tokens, utr3_tokens, te))

        print(f"[dataset] kept={len(self.tokenized_samples)} skipped={self.skipped_count}")

    def __len__(self):
        return len(self.tokenized_samples)

    def tokenize(self, seq):
        return self.tokenizer.tokenize(seq)

    def __getitem__(self, idx):
        utr5_tokens, cds_tokens, utr3_tokens, te = self.tokenized_samples[idx]

        prefix_tokens = [self.bos_id, self.cds_id] + cds_tokens + [self.utr5_id]
        target_tokens = utr5_tokens.copy()
        if not self.only_utr5:
            target_tokens += [self.utr3_id] + utr3_tokens
        target_tokens += [self.eos_id]

        full_tokens = prefix_tokens + target_tokens
        generation_start_idx = len(prefix_tokens)
        full_loss_mask = [0] * generation_start_idx + [1] * (
            len(full_tokens) - generation_start_idx
        )

        input_ids = full_tokens[:-1]
        target_ids = full_tokens[1:]
        loss_mask = full_loss_mask[1:]

        seq_len = len(input_ids)
        padding_length = self.max_len - seq_len
        input_ids = torch.tensor(
            input_ids + [self.pad_id] * padding_length, dtype=torch.long
        )
        target_ids = torch.tensor(
            target_ids + [self.pad_id] * padding_length, dtype=torch.long
        )
        loss_mask = torch.tensor(loss_mask + [0] * padding_length, dtype=torch.float)

        padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        if padding_length > 0:
            padding_mask[-padding_length:] = True

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "padding_mask": padding_mask,
            "te_label": torch.tensor(te, dtype=torch.float),
            "utr5_len": torch.tensor(len(utr5_tokens), dtype=torch.long),
            "cds_len": torch.tensor(len(cds_tokens), dtype=torch.long),
            "utr3_len": torch.tensor(len(utr3_tokens), dtype=torch.long),
        }
