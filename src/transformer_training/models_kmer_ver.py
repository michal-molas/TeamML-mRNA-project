import math
from itertools import product

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


class MRNACsvDataset(Dataset):
    def __init__(
        self,
        csv_path,
        max_utr5_len=2048,
        max_cds_len=8192,
        max_utr3_len=2048,
        only_utr5=False,
        k=3,
    ):
        dataframe = None
        if (csv_path.split('.')[-1] == 'xlsx'):
            dataframe = pd.read_excel(csv_path)
        else:
            dataframe = pd.read_csv(csv_path)

        self.df = dataframe
        self.max_utr5_len = max_utr5_len
        self.max_cds_len = max_cds_len
        self.max_utr3_len = max_utr3_len
        self.only_utr5 = only_utr5
        self.k = k

        kmers = [''.join(p) for p in product('ATCG', repeat=k)]
        self.vocab = {kmer: i for i, kmer in enumerate(kmers)}

        # Special tokens appended after k-mer tokens
        n_kmer = len(self.vocab)  # == 4^k
        self.vocab['<PAD>'] = n_kmer
        self.vocab['<BOS>'] = n_kmer + 1
        self.vocab['<EOS>'] = n_kmer + 2
        self.vocab['<CDS>'] = n_kmer + 3
        self.vocab['<UTR5>'] = n_kmer + 4
        if not self.only_utr5:
            self.vocab['<UTR3>'] = n_kmer + 5

        self.vocab_size = len(self.vocab)

        # Max lengths in token units (sequences are trimmed in nt units before tokenisation)
        max_utr5_tokens = math.ceil(max_utr5_len / k)
        max_cds_tokens = math.ceil(max_cds_len / k)
        max_utr3_tokens = math.ceil(max_utr3_len / k)
        self.max_cds_tokens = max_cds_tokens

        # full sequence length - 1 (because we shift by 1 when comparing input_ids and target_ids)
        # <BOS> + <CDS> + cds + <UTR5> + utr5 + [<UTR3> + utr3] + <EOS>
        if self.only_utr5:
            self.max_len = (max_utr5_tokens + max_cds_tokens + 4) - 1
        else:
            self.max_len = (max_utr5_tokens + max_cds_tokens + max_utr3_tokens + 5) - 1

        self.pad_id = self.vocab['<PAD>']
        self.bos_id = self.vocab['<BOS>']
        self.eos_id = self.vocab['<EOS>']
        self.cds_id = self.vocab['<CDS>']
        self.utr5_id = self.vocab['<UTR5>']
        self.utr3_id = self.vocab.get('<UTR3>')

        has_te = 'te' in self.df.columns

        self.tokenized_samples = []
        self.skipped_count = 0

        print("Loading dataset...")
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            # 5'UTR generation should be right-to-left, hence the reverse
            # UTRs are trimmed to max length, mrna is skipped if CDS is too long
            utr5_str = str(row["utr5"])[::-1][:self.max_utr5_len].upper()
            cds_str = str(row["cds"]).upper()
            utr3_str = str(row["utr3"])[:self.max_utr3_len].upper()

            utr5_tokens = self.tokenize(utr5_str)
            cds_tokens = self.tokenize(cds_str)
            utr3_tokens = self.tokenize(utr3_str)
            utr5_len = len(utr5_tokens)
            cds_len = len(cds_tokens)
            utr3_len = len(utr3_tokens)

            if cds_len > self.max_cds_tokens:
                self.skipped_count += 1
                continue

            if utr5_len + cds_len + (0 if self.only_utr5 else utr3_len) >= self.max_len:
                self.skipped_count += 1
                continue

            te = float('nan')
            if has_te:
                raw = pd.to_numeric(row.get('te'), errors='coerce')
                if not pd.isna(raw):
                    te = float(raw)

            self.tokenized_samples.append((utr5_tokens, cds_tokens, utr3_tokens, te))

        print(f"[dataset] kept={len(self.tokenized_samples)} skipped={self.skipped_count}")

    def __len__(self):
        return len(self.tokenized_samples)

    def tokenize(self, seq):
        seq = seq.replace('U', 'T')
        return [
            self.vocab.get(seq[i:i + self.k], self.pad_id)
            for i in range(0, len(seq) - self.k + 1, self.k)
        ]

    def __getitem__(self, idx):
        utr5_tokens, cds_tokens, utr3_tokens, te = self.tokenized_samples[idx]

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

        input_ids = full_tokens[:-1]
        target_ids = full_tokens[1:]
        loss_mask = full_loss_mask[1:]

        seq_len = len(input_ids)
        padding_length = self.max_len - seq_len

        input_ids = torch.tensor(input_ids + [self.pad_id] * padding_length, dtype=torch.long)
        target_ids = torch.tensor(target_ids + [self.pad_id] * padding_length, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask + [0] * padding_length, dtype=torch.float)

        padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        if padding_length > 0:
            padding_mask[-padding_length:] = True

        te_label = torch.tensor(te, dtype=torch.float)
        utr5_len = torch.tensor(len(utr5_tokens), dtype=torch.long)
        cds_len = torch.tensor(len(cds_tokens), dtype=torch.long)
        utr3_len = torch.tensor(len(utr3_tokens), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "padding_mask": padding_mask,
            "te_label": te_label,
            "utr5_len": utr5_len,
            "cds_len": cds_len,
            "utr3_len": utr3_len,
        }


class MRNATransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, padding_mask):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)

        x_emb = self.embedding(x)
        pos_emb = self.pos_embedding(pos)

        hidden = x_emb + pos_emb
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        out = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.seq_head(out)
