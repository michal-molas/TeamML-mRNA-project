import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from data import MRNALoArmDataset, build_model_inputs
from loss import mask_value_logits
from model import LoArmConfig, LoArmTransformer


def _sample_from_logits(logits, temperature=1.0, greedy=False):
    if greedy:
        return torch.argmax(logits, dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate_target_canvas(
    model,
    prefix_ids,
    prefix_padding_mask,
    mask_id,
    temperature=1.0,
    greedy_order=False,
    greedy_value=False,
):
    device = next(model.parameters()).device
    prefix_ids = prefix_ids.to(device).unsqueeze(0)
    prefix_padding_mask = prefix_padding_mask.to(device).unsqueeze(0)

    target_len = model.config.target_len
    target = torch.full((1, target_len), mask_id, dtype=torch.long, device=device)
    remaining = torch.ones(1, target_len, dtype=torch.bool, device=device)

    for _ in range(target_len):
        input_ids, padding_mask = build_model_inputs(prefix_ids, prefix_padding_mask, target)
        outputs = model(input_ids, padding_mask)

        order_logits = outputs["order_logits"].masked_fill(~remaining, float("-inf"))
        slot = _sample_from_logits(order_logits, temperature=temperature, greedy=greedy_order)

        value_logits = mask_value_logits(outputs["value_logits"], mask_id)
        selected_value_logits = value_logits[0, slot.item(), :]
        value = _sample_from_logits(
            selected_value_logits.unsqueeze(0),
            temperature=temperature,
            greedy=greedy_value,
        )
        target[0, slot.item()] = value.item()
        remaining[0, slot.item()] = False

    return target.squeeze(0).cpu()


def sample_from_cds(
    model,
    dataset,
    cds_seq,
    temperature=1.0,
    greedy_order=False,
    greedy_value=False,
):
    prefix_ids, prefix_padding_mask = dataset.encode_cds_prefix(cds_seq)
    canvas = generate_target_canvas(
        model=model,
        prefix_ids=torch.tensor(prefix_ids, dtype=torch.long),
        prefix_padding_mask=torch.tensor(prefix_padding_mask, dtype=torch.bool),
        mask_id=dataset.mask_id,
        temperature=temperature,
        greedy_order=greedy_order,
        greedy_value=greedy_value,
    )
    utr5, utr3 = dataset.decode_target(canvas.tolist())
    return {"utr5": utr5, "utr3": utr3, "target_ids": canvas.tolist()}


def _load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = LoArmConfig(**checkpoint["config"])
    model = LoArmTransformer(config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--samples_per_cds", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_utr5_len", type=int, default=200)
    parser.add_argument("--max_cds_len", type=int, default=500)
    parser.add_argument("--max_utr3_len", type=int, default=200)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy_order", action="store_true")
    parser.add_argument("--greedy_value", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_checkpoint(args.model_path, device)
    dataset = MRNALoArmDataset(
        args.dataset_csv,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        k=args.k,
    )

    rows = []
    n = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    for idx in range(n):
        sample = dataset[idx]
        rows.append(
            {
                "id": sample["id"],
                "sample": "gt",
                "cds": sample["cds"],
                "utr5": sample["utr5"],
                "utr3": sample["utr3"],
            }
        )
        for sample_idx in range(args.samples_per_cds):
            generated = sample_from_cds(
                model,
                dataset,
                sample["cds"],
                temperature=args.temperature,
                greedy_order=args.greedy_order,
                greedy_value=args.greedy_value,
            )
            rows.append(
                {
                    "id": sample["id"],
                    "sample": f"sample_{sample_idx}",
                    "cds": sample["cds"],
                    "utr5": generated["utr5"],
                    "utr3": generated["utr3"],
                }
            )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
