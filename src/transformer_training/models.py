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
    ):
        self.df = pd.read_csv(csv_path)
        self.max_utr5_len = max_utr5_len
        self.max_cds_len = max_cds_len
        self.max_utr3_len = max_utr3_len
        self.only_utr5 = only_utr5
        # full sequence length - 1 (because we shift by 1 when comparing input_ids and target_ids)
        # <BOS> + <CDS> + cds + <UTR5> + utr5 + [<UTR3> + utr3] + <EOS>
        if self.only_utr5:
            self.max_len = (max_utr5_len + max_cds_len + 4) - 1
        else:
            self.max_len = (max_utr5_len + max_cds_len + max_utr3_len + 5) - 1

        self.vocab = {
            'A': 0,
            'U': 1,
            'T': 1,
            'C': 2,
            'G': 3,
            '<PAD>': 4,
            '<BOS>': 5,
            '<EOS>': 6,
            '<CDS>': 7,
            '<UTR5>': 8,
        }
        if not self.only_utr5:
            self.vocab['<UTR3>'] = 9

        self.vocab_size = len(set(self.vocab.values()))
        self.pad_id = self.vocab['<PAD>']
        self.bos_id = self.vocab['<BOS>']
        self.eos_id = self.vocab['<EOS>']
        self.cds_id = self.vocab['<CDS>']
        self.utr5_id = self.vocab['<UTR5>']
        self.utr3_id = self.vocab.get('<UTR3>')

        has_te = 'te' in self.df.columns

        self.tokenized_samples = []
        self.skipped_count = 0
        self.max_skip_logs = 20

        print("Loading dataset...")
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            if i > 10000:
                break

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

            if cds_len > self.max_cds_len:
                self.skipped_count += 1
                if self.skipped_count <= self.max_skip_logs:
                    print(
                        f"[skip] idx={i} CDS length exceeds limit of {self.max_cds_len}"
                    )
                continue

            if utr5_len + cds_len + (0 if self.only_utr5 else utr3_len) >= self.max_len:
                self.skipped_count += 1
                if self.skipped_count <= self.max_skip_logs:
                    print(
                        f"[skip] idx={i} total length exceeds limit of {self.max_len}"
                    )
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
        return [self.vocab.get(n, self.pad_id) for n in seq]

    def __getitem__(self, idx):
        utr5_tokens, cds_tokens, utr3_tokens, te = self.tokenized_samples[idx]

        # Prefix: <BOS> + <CDS> + CDS_tokens + <UTR5>
        prefix_tokens = [self.bos_id, self.cds_id] + cds_tokens + [self.utr5_id]
        # Target: UTR5_tokens + [<UTR3> + UTR3_tokens] + <EOS>
        target_tokens = utr5_tokens
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
