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
    def __init__(self, csv_path, max_len=1024, mask_prob=0.15):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, '<PAD>': 4, '[MASK]': 5}
        self.pad_id = self.vocab['<PAD>']
        self.mask_id = self.vocab['[MASK]']

    def __len__(self):
        return len(self.df)

    def tokenize(self, seq):
        return [self.vocab.get(n, self.pad_id) for n in seq]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        utr5_str = str(row["utr5"]).upper()
        cds_str = str(row["cds"]).upper()

        utr5_tokens = self.tokenize(utr5_str)
        cds_tokens = self.tokenize(cds_str)

        utr5_regions = [0] * len(utr5_tokens)
        cds_regions = [1] * len(cds_tokens)

        input_ids = utr5_tokens + cds_tokens
        region_ids = utr5_regions + cds_regions

        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            region_ids = region_ids[:self.max_len]

        seq_len = len(input_ids)
        padding_length = self.max_len - seq_len

        input_ids = torch.tensor(input_ids + [self.pad_id] * padding_length, dtype=torch.long)
        region_ids = torch.tensor(region_ids + [3] * padding_length, dtype=torch.long)
        
        target_ids = input_ids.clone()

        padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        if padding_length > 0:
            padding_mask[-padding_length:] = True

        loss_mask = torch.zeros(self.max_len, dtype=torch.float)
        
        utr5_len = min(len(utr5_tokens), self.max_len)
        mask_indices = torch.rand(utr5_len) < self.mask_prob
        
        input_ids[:utr5_len][mask_indices] = self.mask_id
        loss_mask[:utr5_len][mask_indices] = 1.0

        return input_ids, region_ids, target_ids, loss_mask, padding_mask

class MaskedRNAGenerator(nn.Module):
    def __init__(self, vocab_size=6, d_model=256, nhead=8, num_layers=4, max_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.region_embedding = nn.Embedding(4, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, region_ids, padding_mask=None):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        
        x_emb = self.embedding(x)
        pos_emb = self.pos_embedding(pos)
        reg_emb = self.region_embedding(region_ids)
        
        hidden = x_emb + pos_emb + reg_emb
        out = self.transformer(hidden, src_key_padding_mask=padding_mask)
        return self.seq_head(out)

def train(args, device):
    dataset = MRNACsvDataset(csv_path=args.csv_path, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MaskedRNAGenerator(
        vocab_size=6, 
        d_model=args.d_model, 
        nhead=args.n_heads, 
        num_layers=args.n_layers,
        max_len=args.max_len
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        print("Epoch", epoch)
        for step, (input_ids, region_ids, target_ids, loss_mask, padding_mask) in tqdm(list(enumerate(dataloader))):
            input_ids = input_ids.to(device)
            region_ids = region_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            
            outputs = model(input_ids, region_ids, padding_mask)
            
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
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.init(project="mrna-masked-lm", config=vars(args))

    train(args, device)

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()