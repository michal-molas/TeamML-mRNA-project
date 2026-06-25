"""
Usage:

    python generate_samples.py transformer CHECKPOINT.pt input.csv output.csv
    python generate_samples.py indigo CHECKPOINT.pt input.csv output.csv
    python generate_samples.py loarm CHECKPOINT.pt input.csv output.csv

The input CSV must contain id, utr5, cds, and utr3 columns.
The output CSV contains id, sample, utr5, cds, and utr3 columns.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
TRANSFORMER_ROOT = SRC_ROOT / "transformer_training"
LO_ARM_ROOT = SRC_ROOT / "lo_arm"

for path in (str(REPO_ROOT), str(SRC_ROOT), str(TRANSFORMER_ROOT), str(LO_ARM_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.indigo.main import IndigoTransformer
from src.lo_arm.data import MRNALoArmDataset
from src.lo_arm.generate import _load_checkpoint as _load_loarm_checkpoint
from src.lo_arm.generate import sample_from_cds as loarm_sample_from_cds
from src.transformer_training.generate import MRNAInferenceSampler
from src.transformer_training.models import MRNA_VOCAB, MRNACsvDataset, MRNATransformer


INPUT_COLS = ["id", "utr5", "cds", "utr3"]
OUTPUT_COLS = ["id", "sample", "utr5", "cds", "utr3"]


def load_pretrained_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(
        f"Loaded {checkpoint_path}  missing={len(missing)}  unexpected={len(unexpected)}"
    )


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")


def _clean_seq(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).upper().replace("U", "T")


def _load_input_csv(path: str, max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(INPUT_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")
    df = df[INPUT_COLS].copy()
    df["id"] = df["id"].map(lambda value: "" if pd.isna(value) else str(value))
    for col in ["utr5", "cds", "utr3"]:
        df[col] = df[col].map(_clean_seq)
    if max_samples is not None:
        df = df.head(max_samples)
    return df


def _gt_row(row: pd.Series) -> dict[str, str]:
    return {
        "id": row["id"],
        "sample": "gt",
        "utr5": row["utr5"],
        "cds": row["cds"],
        "utr3": row["utr3"],
    }


def _write_samples(rows: list[dict[str, str]], output_csv: str) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLS).fillna("").to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


def _generate_transformer(
    args: argparse.Namespace,
    input_df: pd.DataFrame,
    device: torch.device
) -> list[dict[str, str]]:
    model = MRNATransformer(
        vocab_size=len(set(MRNA_VOCAB.values())),
        d_model=args.d_model,
        nhead=args.n_heads,
        num_layers=args.n_layers,
        max_len=args.max_len,
    ).to(device)
    load_pretrained_weights(model, args.checkpoint_path, device)
    sampler = MRNAInferenceSampler(model, max_len=args.max_len, device=device)

    # TODO: batched inference
    rows = []
    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Generating transformer samples"):
        rows.append(_gt_row(row))
        generated_samples = sampler.generate_from_cds(
            cds=row["cds"],
            k=args.samples_per_cds,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
        )
        for sample_idx, generated in enumerate(generated_samples):
            rows.append(
                {
                    "id": row["id"],
                    "sample": f"sample_{sample_idx}",
                    "utr5": generated["utr5"],
                    "cds": row["cds"],
                    "utr3": generated["utr3"],
                }
            )
    return rows


def _checkpoint_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _loarm_dataset_kwargs(args: argparse.Namespace, checkpoint: dict) -> dict:
    data_cfg = checkpoint.get("data", {}) if isinstance(checkpoint, dict) else {}
    tokenizer_cfg = checkpoint.get("tokenizer", {}) if isinstance(checkpoint, dict) else {}
    return {
        "max_utr5_len": _coalesce(args.max_utr5_len, data_cfg.get("max_utr5_len"), 200),
        "max_cds_len": _coalesce(args.max_cds_len, data_cfg.get("max_cds_len"), 500),
        "max_utr3_len": _coalesce(args.max_utr3_len, data_cfg.get("max_utr3_len"), 200),
        "k": _coalesce(args.k, tokenizer_cfg.get("k"), 3),
    }


def _generate_loarm(args: argparse.Namespace, device: torch.device) -> list[dict[str, str]]:
    model, checkpoint = _load_loarm_checkpoint(args.checkpoint_path, device)
    dataset = MRNALoArmDataset(args.input_csv, **_loarm_dataset_kwargs(args, checkpoint))

    rows = []
    n_samples = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    for idx in tqdm(range(n_samples), desc="Generating LO-ARM samples"):
        sample = dataset[idx]
        rows.append(
            {
                "id": sample["id"],
                "sample": "gt",
                "utr5": sample["utr5"],
                "cds": sample["cds"],
                "utr3": sample["utr3"],
            }
        )
        for sample_idx in range(args.samples_per_cds):
            generated = loarm_sample_from_cds(
                model,
                dataset,
                sample["cds"],
                temperature=args.temperature,
                greedy_order=args.greedy,
                greedy_value=args.greedy,
            )
            rows.append(
                {
                    "id": sample["id"],
                    "sample": f"sample_{sample_idx}",
                    "utr5": generated["utr5"],
                    "cds": sample["cds"],
                    "utr3": generated["utr3"],
                }
            )
    return rows


def _infer_indigo_config(state_dict: dict[str, torch.Tensor]) -> SimpleNamespace:
    token_embedding = state_dict["encoder.embedding_layer.token_embedding.weight"]
    position_embedding = state_dict["encoder.embedding_layer.position_embedding.weight"]
    rel_pos_embedding = state_dict["encoder.blocks.0.attention_layer.relative_positional_embedding.weight"]
    layer_indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("encoder.blocks.") and key.split(".")[2].isdigit()
    }
    d_model = token_embedding.shape[1]
    d_head = rel_pos_embedding.shape[1]
    return SimpleNamespace(
        vocab_size=token_embedding.shape[0],
        d_model=d_model,
        num_heads=d_model // d_head,
        num_layers=max(layer_indices) + 1,
        max_len=position_embedding.shape[0],
    )


def _load_indigo_checkpoint(path: str, device: torch.device) -> IndigoTransformer:
    checkpoint = torch.load(path, map_location=device)
    state_dict = _checkpoint_state_dict(checkpoint)
    model = IndigoTransformer(_infer_indigo_config(state_dict)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_indigo_R(prefix_len: int, generated_count: int, ordered_steps: list[int], device: torch.device):
    seq_len = prefix_len + generated_count
    abs_pos = torch.arange(seq_len, dtype=torch.long, device=device)
    rank_by_step = {step: rank for rank, step in enumerate(ordered_steps)}
    for step in range(generated_count):
        abs_pos[prefix_len + step] = prefix_len + rank_by_step[step]
    return torch.sign(abs_pos.unsqueeze(0) - abs_pos.unsqueeze(1)).long()


def _sample_logits(logits: torch.Tensor, temperature: float, greedy: bool) -> torch.Tensor:
    if greedy:
        return torch.argmax(logits, dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class IndigoSampler:
    def __init__(
        self,
        model: IndigoTransformer,
        dataset: MRNACsvDataset,
        device: torch.device,
        max_target_tokens: int,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.max_target_tokens = max_target_tokens
        self.pad_id = dataset.pad_id
        self.bos_id = dataset.bos_id
        self.eos_id = dataset.eos_id
        self.cds_id = dataset.cds_id
        self.utr5_id = dataset.utr5_id
        self.utr3_id = dataset.utr3_id
        self.invalid_word_ids = [self.pad_id, self.bos_id, self.cds_id, self.utr5_id]
        self.id_to_token = {idx: token for token, idx in dataset.vocab.items()}

    def _prefix(self, cds: str) -> list[int]:
        cds_tokens = self.dataset.tokenize(cds)
        if len(cds_tokens) > self.dataset.max_cds_tokens:
            cds_tokens = cds_tokens[: self.dataset.max_cds_tokens]
        return [self.bos_id, self.cds_id] + cds_tokens + [self.utr5_id]

    def _detokenize(self, token_ids: list[int]) -> str:
        parts = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), "")
            if token in {"<PAD>", "<BOS>", "<EOS>", "<CDS>", "<UTR5>", "<UTR3>"}:
                continue
            parts.append(token)
        return "".join(parts)

    def _parse_target(self, final_tokens: list[int]) -> dict[str, str]:
        if self.eos_id in final_tokens:
            final_tokens = final_tokens[: final_tokens.index(self.eos_id)]

        if self.utr3_id is not None and self.utr3_id in final_tokens:
            split_idx = final_tokens.index(self.utr3_id)
            utr5_tokens = final_tokens[:split_idx]
            utr3_tokens = final_tokens[split_idx + 1 :]
        else:
            utr5_tokens = final_tokens
            utr3_tokens = []

        return {
            "utr5": self._detokenize(utr5_tokens)[::-1],
            "utr3": self._detokenize(utr3_tokens),
        }

    @torch.no_grad()
    def generate_from_cds(self, cds: str, temperature: float, greedy: bool) -> dict[str, str]:
        prefix = self._prefix(cds)
        generated_tokens: list[int] = []
        ordered_steps: list[int] = []

        for step in range(self.max_target_tokens):
            input_ids = torch.tensor(
                [prefix + generated_tokens],
                dtype=torch.long,
                device=self.device,
            )
            seq_len = input_ids.shape[1]
            if seq_len > self.model.config.max_len:
                break

            R = _build_indigo_R(len(prefix), len(generated_tokens), ordered_steps, self.device).unsqueeze(0)
            padding_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=self.device)
            H, _, word_logits = self.model(input_ids, R, padding_mask)
            prediction_pos = len(prefix) + len(generated_tokens) - 1
            next_word_logits = word_logits[:, prediction_pos, :].clone()
            next_word_logits[:, self.invalid_word_ids] = float("-inf")
            next_token = int(_sample_logits(next_word_logits, temperature, greedy).item())

            if step == 0:
                ordered_steps.append(0)
            else:
                _model = self.model.module if hasattr(self.model, "module") else self.model
                h = H[:, prediction_pos, :]
                H_gen = H[:, len(prefix) : len(prefix) + len(generated_tokens), :]
                left_keys = _model.position_head_left_proj(H_gen)
                right_keys = _model.position_head_right_proj(H_gen)
                keys = torch.cat([left_keys, right_keys], dim=1)
                query = _model.position_head_state_proj(h) + _model.get_embedding_matrix()[next_token].unsqueeze(0)
                position_logits = torch.matmul(query, keys.squeeze(0).T)

                valid = torch.ones(position_logits.shape[-1], dtype=torch.bool, device=self.device)
                valid[0] = False
                position_logits = position_logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
                slot = int(_sample_logits(position_logits, temperature, greedy).item())
                ref_step = slot % len(generated_tokens)
                ref_pos = ordered_steps.index(ref_step)
                insert_pos = ref_pos if slot < len(generated_tokens) else ref_pos + 1
                ordered_steps.insert(insert_pos, step)

            generated_tokens.append(next_token)
            if next_token == self.eos_id:
                break

        final_tokens = [generated_tokens[step] for step in ordered_steps]
        return self._parse_target(final_tokens)


def _indigo_dataset_kwargs(args: argparse.Namespace) -> dict:
    return {
        "max_utr5_len": _coalesce(args.max_utr5_len, 200),
        "max_cds_len": _coalesce(args.max_cds_len, 500),
        "max_utr3_len": _coalesce(args.max_utr3_len, 200),
        "k": _coalesce(args.k, 3),
        "only_utr5": False,
    }


def _generate_indigo(
    args: argparse.Namespace,
    input_df: pd.DataFrame,
    device: torch.device,
) -> list[dict[str, str]]:
    model = _load_indigo_checkpoint(args.checkpoint_path, device)
    dataset = MRNACsvDataset(args.input_csv, **_indigo_dataset_kwargs(args))
    max_target_tokens = model.config.max_len - (2 + dataset.max_cds_tokens + 1)
    if max_target_tokens <= 0:
        raise ValueError("INDIGO checkpoint max_len is too short for the configured CDS prefix.")
    sampler = IndigoSampler(model, dataset, device, max_target_tokens=max_target_tokens)

    rows = []
    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Generating INDIGO samples"):
        rows.append(_gt_row(row))
        for sample_idx in range(args.samples_per_cds):
            generated = sampler.generate_from_cds(row["cds"], args.temperature, args.greedy)
            rows.append(
                {
                    "id": row["id"],
                    "sample": f"sample_{sample_idx}",
                    "utr5": generated["utr5"],
                    "cds": row["cds"],
                    "utr3": generated["utr3"],
                }
            )
    return rows


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate eval-ready mRNA samples from a trained model checkpoint."
    )
    parser.add_argument("model_type", choices=["transformer", "indigo", "loarm"])
    parser.add_argument("checkpoint_path", help="Path to the checkpoint/weights file.")
    parser.add_argument("input_csv", help="CSV with id, utr5, cds, utr3 columns.")
    parser.add_argument("output_csv", help="Where to save the eval-ready generated samples CSV.")

    parser.add_argument("--samples_per_cds", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20, help="Transformer top-k sampling.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy value sampling where supported.")

    parser.add_argument("--k", type=int, default=None, help="K-mer size for INDIGO/LO-ARM datasets.")
    parser.add_argument("--max_utr5_len", type=int, default=None)
    parser.add_argument("--max_cds_len", type=int, default=None)
    parser.add_argument("--max_utr3_len", type=int, default=None)

    parser.add_argument("--d_model", type=int, default=256, help="Transformer checkpoint d_model.")
    parser.add_argument("--n_heads", type=int, default=8, help="Transformer checkpoint n_heads.")
    parser.add_argument("--n_layers", type=int, default=6, help="Transformer checkpoint n_layers.")
    parser.add_argument("--max_len", type=int, default=904, help="Transformer maximum generation length.")
    return parser


def main() -> None:
    parser = _get_parser()
    args = parser.parse_args()

    device = _device()

    input_df = _load_input_csv(args.input_csv, args.max_samples)
    if args.model_type == "transformer":
        rows = _generate_transformer(args, input_df, device)
    elif args.model_type == "indigo":
        rows = _generate_indigo(args, input_df, device)
    elif args.model_type == "loarm":
        rows = _generate_loarm(args, device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    _write_samples(rows, args.output_csv)


if __name__ == "__main__":
    main()
