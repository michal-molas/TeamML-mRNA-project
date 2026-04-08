import argparse
import sys

import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
from dotenv import load_dotenv

from models import MRNACsvDataset, MRNATransformer


def compute_validation_loss(args, model, valid_dataloader, global_step, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for step, batch in list(enumerate(valid_dataloader)):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            outputs = model(input_ids, padding_mask=padding_mask)
            
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            
            loss = F.cross_entropy(outputs_flat, targets_flat, reduction='none')
            val_loss += (loss * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)

    val_loss /= len(valid_dataloader)

    print(f"Validation Loss: {val_loss.item()}", file=sys.stderr)
    if args.wandb:
        wandb.log(
            {"valid/loss": val_loss.item()},
            step=global_step,
        )

def train(args, device, only_utr5=False):
    dataset = MRNACsvDataset(
        csv_path=args.csv_path,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        only_utr5=only_utr5,
    )

    dataset_size = len(dataset)

    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train dataset length: {len(train_dataset)}", file=sys.stderr)
    print(f"Validation dataset length: {len(val_dataset)}", file=sys.stderr)

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    model = MRNATransformer(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.n_layers,
        max_len=dataset.max_len,
    ).to(device)

    global_step = 0
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        model.train()

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

            global_step += 1

            epoch_loss += loss.item()
            if args.wandb and step % 100 == 0:
                wandb.log(
                  {"train/loss": loss.item()},
                  step=global_step,
                )
        
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch} | loss={epoch_loss:.4f}")

        if args.output_path and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model_state_dict": model.state_dict()}, args.output_path)
        compute_validation_loss(args, model, valid_dataloader, global_step, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="../../data/pretraining/pretraining_refseq.csv")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_utr5_len", type=int, default=200)
    parser.add_argument("--max_cds_len", type=int, default=500)
    parser.add_argument("--max_utr3_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dotenv()

    if args.wandb:
        wandb.init(
            project="transformer-parameter-grid",
            config=vars(args),
            dir='../../logs',
        )

    only_utr5 = False
    train(args, device, only_utr5)

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()