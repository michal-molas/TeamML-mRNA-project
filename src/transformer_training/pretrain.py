import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import wandb

class MRNACsvDataset(Dataset):
    def __init__(
        self,
        csv_path,
        max_utr5_len=2048,
        max_cds_len=8192,
        max_utr3_len=2048,
    ):
        self.df = pd.read_csv(csv_path)
        self.max_utr5_len = max_utr5_len
        self.max_cds_len = max_cds_len
        self.max_utr3_len = max_utr3_len
        # full sequence length - 1 (because we shift by 1 when comparing input_ids and target_ids)
        self.max_len = (max_utr5_len + max_cds_len + max_utr3_len + 7) - 1
        self.vocab = {
            'A': 0,
            'C': 1,
            'G': 2,
            'U': 3,
            'T': 3,
            '<PAD>': 4,
            '<BOS>': 5,
            '<SEP>': 6,
            '<EOS>': 7,
            '<CDS>': 8,
            '<UTR5>': 9,
            '<UTR3>': 10,
        }
        self.vocab_size = len(set(self.vocab.values()))
        self.pad_id = self.vocab['<PAD>']
        self.bos_id = self.vocab['<BOS>']
        self.sep_id = self.vocab['<SEP>']
        self.eos_id = self.vocab['<EOS>']
        self.cds_id = self.vocab['<CDS>']
        self.utr5_id = self.vocab['<UTR5>']
        self.utr3_id = self.vocab['<UTR3>']

    def __len__(self):
        return len(self.df)

    def tokenize(self, seq):
        return [self.vocab.get(n, self.pad_id) for n in seq]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        utr5_str = str(row["utr5"]).upper()
        cds_str = str(row["cds"]).upper()
        utr3_str = str(row["utr3"]).upper()

        utr5_tokens = self.tokenize(utr5_str)[:self.max_utr5_len]
        cds_tokens = self.tokenize(cds_str)[:self.max_cds_len]
        utr3_tokens = self.tokenize(utr3_str)[:self.max_utr3_len]

        prefix_tokens = [self.bos_id, self.cds_id] + cds_tokens + [self.sep_id]
        target_tokens = [self.utr5_id] + utr5_tokens + [self.sep_id] + [self.utr3_id] + utr3_tokens + [self.eos_id]
        full_tokens = (prefix_tokens + target_tokens)[: self.max_len + 1]

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

        return input_ids, target_ids, loss_mask, padding_mask

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

def train(args, device):
    dataset = MRNACsvDataset(
        csv_path=args.csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MRNATransformer(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.n_layers,
        max_len=dataset.max_len,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        for step, (input_ids, target_ids, loss_mask, padding_mask) in tqdm(list(enumerate(dataloader))):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            
            outputs = model(input_ids, padding_mask=padding_mask)
            
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            
            loss = F.cross_entropy(outputs_flat, targets_flat, reduction='none')
            
            loss = (loss * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)

            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
                if args.wandb:
                    wandb.log({"train/loss": loss.item()})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="../data_preprocessing/data/refseq/refseq_transcripts.csv")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_utr5_len", type=int, default=128)
    parser.add_argument("--max_cds_len", type=int, default=512)
    parser.add_argument("--max_utr3_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.init(project="teamml-project-poc-transformer", config=vars(args))

    train(args, device)

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()