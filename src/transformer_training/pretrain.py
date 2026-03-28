import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from models import MRNACsvDataset, MRNATransformer


def train(args, device, only_utr5=False):
    dataset = MRNACsvDataset(
        csv_path=args.csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        only_utr5=only_utr5,
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
    best_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in tqdm(list(enumerate(dataloader))):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            optimizer.zero_grad()
            
            outputs = model(input_ids, padding_mask=padding_mask)
            
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            
            loss = F.cross_entropy(outputs_flat, targets_flat, reduction='none')
            
            loss = (loss * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if args.wandb and step % 10 == 0:
                wandb.log({"train/loss": loss.item()})
        
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch} | loss={epoch_loss:.4f}")

        if args.output_path and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model_state_dict": model.state_dict()}, args.output_path)

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
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.init(project="teamml-project-poc-transformer", config=vars(args))

    train(args, device, True)

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()