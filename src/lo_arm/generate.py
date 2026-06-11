import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

try:
    from .data import (
        REGION_PAD,
        REGION_UTR3,
        REGION_UTR5,
        LayoutPrior,
        MRNALoArmDataset,
        build_model_inputs,
    )
    from .loss import mask_value_logits
    from .model import LoArmConfig, LoArmTransformer
except ImportError:  # pragma: no cover - script execution fallback
    from data import (
        REGION_PAD,
        REGION_UTR3,
        REGION_UTR5,
        LayoutPrior,
        MRNALoArmDataset,
        build_model_inputs,
    )
    from loss import mask_value_logits
    from model import LoArmConfig, LoArmTransformer


def _sample_from_logits(logits, temperature=1.0, greedy=False, top_p=1.0):
    if greedy:
        return torch.argmax(logits, dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    probs = F.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative - sorted_probs < top_p
        keep[..., 0] = True
        sorted_probs = sorted_probs.masked_fill(~keep, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sorted_sample = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(1, sorted_sample).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _layout_prior_from_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    layouts = checkpoint.get("layout_prior")
    if layouts is None:
        layouts = checkpoint.get("data", {}).get("layout_prior")
    if not layouts:
        return None
    return LayoutPrior(layouts)


def _target_metadata(dataset, layout, target_len, device):
    real_len = int(layout["utr5_len"]) + int(layout["utr3_len"])
    if real_len <= 0:
        raise ValueError("sampled layout has no UTR target slots")
    if real_len > target_len:
        raise ValueError(
            f"sampled UTR target length {real_len} exceeds model target_len {target_len}"
        )
    target_order_mask = torch.zeros(1, target_len, dtype=torch.bool, device=device)
    target_order_mask[:, :real_len] = True
    target_region_ids = torch.full((1, target_len), REGION_PAD, dtype=torch.long, device=device)
    target_region_ids[:, : int(layout["utr5_len"])] = REGION_UTR5
    target_region_ids[
        :,
        int(layout["utr5_len"]) : int(layout["utr5_len"]) + int(layout["utr3_len"]),
    ] = REGION_UTR3
    target = torch.full((1, target_len), dataset.pad_id, dtype=torch.long, device=device)
    target[:, :real_len] = dataset.mask_id
    return target, target_order_mask, target_region_ids


@torch.no_grad()
def generate_target_canvas(
    model,
    prefix_ids,
    prefix_padding_mask,
    mask_id,
    pad_id=None,
    prefix_region_ids=None,
    target_region_ids=None,
    target_order_mask=None,
    invalid_value_ids=None,
    temperature=1.0,
    greedy_order=False,
    greedy_value=False,
    order_top_p=0.9,
    return_trace=False,
):
    device = next(model.parameters()).device
    prefix_ids = prefix_ids.to(device).unsqueeze(0)
    prefix_padding_mask = prefix_padding_mask.to(device).unsqueeze(0)
    if prefix_region_ids is not None:
        prefix_region_ids = prefix_region_ids.to(device).unsqueeze(0)

    target_len = model.config.target_len
    if target_order_mask is None:
        target_order_mask = torch.ones(1, target_len, dtype=torch.bool, device=device)
    else:
        target_order_mask = target_order_mask.to(device).unsqueeze(0)
    if target_region_ids is not None:
        target_region_ids = target_region_ids.to(device).unsqueeze(0)
    if pad_id is None:
        pad_id = mask_id
    target = torch.full((1, target_len), pad_id, dtype=torch.long, device=device)
    target[:, target_order_mask.squeeze(0)] = mask_id
    remaining = target_order_mask.clone()
    trace = []

    for step in range(int(target_order_mask.sum().item())):
        built = build_model_inputs(
            prefix_ids,
            prefix_padding_mask,
            target,
            prefix_region_ids=prefix_region_ids,
            target_region_ids=target_region_ids,
            target_order_mask=target_order_mask,
        )
        if len(built) == 2:
            input_ids, padding_mask = built
            region_ids = None
        else:
            input_ids, padding_mask, region_ids = built
        if region_ids is None:
            outputs = model(input_ids, padding_mask)
        else:
            outputs = model(input_ids, padding_mask, region_ids=region_ids)

        order_logits = outputs["order_logits"].masked_fill(~remaining, float("-inf"))
        order_probs = F.softmax(order_logits / temperature, dim=-1)
        slot = _sample_from_logits(
            order_logits,
            temperature=temperature,
            greedy=greedy_order,
            top_p=order_top_p,
        )

        value_logits = mask_value_logits(outputs["value_logits"], mask_id, invalid_value_ids)
        selected_value_logits = value_logits[0, slot.item(), :]
        value = _sample_from_logits(
            selected_value_logits.unsqueeze(0),
            temperature=temperature,
            greedy=greedy_value,
            top_p=1.0,
        )
        target[0, slot.item()] = value.item()
        remaining[0, slot.item()] = False
        if return_trace:
            region_id = None
            if target_region_ids is not None:
                region_id = int(target_region_ids[0, slot.item()].item())
            trace.append(
                {
                    "step": step,
                    "slot": int(slot.item()),
                    "region_id": region_id,
                    "order_prob": float(order_probs[0, slot.item()].item()),
                    "value_id": int(value.item()),
                    "is_pad_slot": False,
                }
            )

    canvas = target.squeeze(0).cpu()
    if return_trace:
        return canvas, trace
    return canvas


def sample_from_cds(
    model,
    dataset,
    cds_seq,
    temperature=1.0,
    greedy_order=False,
    greedy_value=False,
    order_top_p=0.9,
    layout=None,
    layout_prior=None,
    rng=None,
    return_trace=False,
):
    rng = rng or random
    cds_tokens = dataset.tokenizer.tokenize(str(cds_seq).upper().replace("U", "T"))
    if len(cds_tokens) > dataset.max_cds_tokens:
        cds_tokens = cds_tokens[: dataset.max_cds_tokens]
    if layout is None:
        prior = layout_prior or dataset.layout_prior
        if prior is None:
            raise ValueError("sample_from_cds requires a layout or a dataset/checkpoint layout prior")
        layout = prior.sample(len(cds_tokens), rng=rng)
    else:
        layout = {key: int(value) for key, value in layout.items()}
    layout["cds_len"] = len(cds_tokens)
    layout["total_len"] = layout["utr5_len"] + layout["cds_len"] + layout["utr3_len"]

    prefix_ids, prefix_padding_mask, prefix_region_ids = dataset.build_prefix(cds_tokens)
    device = next(model.parameters()).device
    _, target_order_mask, target_region_ids = _target_metadata(
        dataset, layout, model.config.target_len, device=torch.device("cpu")
    )
    target_order_mask = target_order_mask.squeeze(0)
    target_region_ids = target_region_ids.squeeze(0)
    generated = generate_target_canvas(
        model=model,
        prefix_ids=torch.tensor(prefix_ids, dtype=torch.long),
        prefix_padding_mask=torch.tensor(prefix_padding_mask, dtype=torch.bool),
        prefix_region_ids=torch.tensor(prefix_region_ids, dtype=torch.long),
        target_region_ids=target_region_ids,
        target_order_mask=target_order_mask,
        mask_id=dataset.mask_id,
        pad_id=dataset.pad_id,
        invalid_value_ids=dataset.tokenizer.special_ids,
        temperature=temperature,
        greedy_order=greedy_order,
        greedy_value=greedy_value,
        order_top_p=order_top_p,
        return_trace=return_trace,
    )
    if return_trace:
        canvas, trace = generated
    else:
        canvas = generated
        trace = None
    utr5, utr3 = dataset.decode_target(canvas.tolist(), layout=layout)
    result = {
        "utr5": utr5,
        "utr3": utr3,
        "target_ids": canvas.tolist(),
        "layout": dict(layout),
        "generation_steps": int(layout["utr5_len"] + layout["utr3_len"]),
    }
    if return_trace:
        result["order_trace"] = trace
    return result


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
    parser.add_argument("--order_top_p", type=float, default=0.9)
    parser.add_argument("--greedy_order", action="store_true")
    parser.add_argument("--greedy_value", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_checkpoint(args.model_path, device)
    dataset = MRNALoArmDataset(
        args.dataset_csv,
        max_utr5_len=args.max_utr5_len,
        max_cds_len=args.max_cds_len,
        max_utr3_len=args.max_utr3_len,
        k=args.k,
    )
    checkpoint_prior = _layout_prior_from_checkpoint(checkpoint)

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
                order_top_p=args.order_top_p,
                layout_prior=checkpoint_prior,
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
