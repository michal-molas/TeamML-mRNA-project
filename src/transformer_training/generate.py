import argparse
import sys
from typing import List, Dict, Optional, Any
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import MRNATransformer, MRNATokenizer, MRNA_VOCAB
from utils import load_pretrained_weights


class MRNAInferenceSampler:
    def __init__(self, model, max_len=None, device=None, tokenizer=None):
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        if tokenizer is None:
            tokenizer = MRNATokenizer(only_utr5=False)

        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.device = device
        self.max_len = max_len if max_len is not None else model.pos_embedding.num_embeddings

        if model.seq_head.out_features != tokenizer.vocab_size:
            raise ValueError(
                "Tokenizer vocab size does not match the model output size: "
                f"{tokenizer.vocab_size} vs {model.seq_head.out_features}."
            )
        if self.max_len > model.pos_embedding.num_embeddings:
            raise ValueError(
                "Requested max_len exceeds the model positional embedding size: "
                f"{self.max_len} > {model.pos_embedding.num_embeddings}."
            )

        self.id_to_token = tokenizer.id_to_token
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.cds_id = tokenizer.cds_id
        self.utr5_id = tokenizer.utr5_id
        self.utr3_id = tokenizer.utr3_id
        self.nucleotide_ids = {
            tokenizer.vocab["A"],
            tokenizer.vocab["T"],
            tokenizer.vocab["C"],
            tokenizer.vocab["G"],
        }

    def tokenize_seq(self, seq: str) -> List[int]:
        seq = seq.upper()
        invalid_chars = sorted({ch for ch in seq if ch not in self.vocab})
        if invalid_chars:
            raise ValueError(f"Unknown nucleotide(s) {invalid_chars} in sequence: {seq}")
        return self.tokenizer.tokenize(seq)

    def detokenize_seq(self, token_ids: List[int]) -> str:
        return self.tokenizer.detokenize(token_ids)

    def build_prefix(self, cds: str) -> List[int]:
        cds_tokens = self.tokenize_seq(cds)
        return [self.bos_id, self.cds_id] + cds_tokens + [self.utr5_id]

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        """
        logits: [batch, vocab_size] for the next token
        returns: [batch]
        """
        if greedy:
            return torch.argmax(logits, dim=-1)

        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, k=top_k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(dim=-1, index=indices, src=values)
            logits = filtered

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def generate_from_cds(
        self,
        cds: str,
        k: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        greedy: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate k samples for a given CDS.
        Returns list of dicts with full generated text and parsed utr5/utr3.
        """
        prefix = self.build_prefix(cds)

        if len(prefix) >= self.max_len:
            raise ValueError(
                f"Prefix length {len(prefix)} >= max_len {self.max_len}. "
                "Increase max_len or shorten CDS."
            )

        # Repeat prefix k times for batched parallel generation
        batch_input = torch.tensor([prefix] * k, dtype=torch.long, device=self.device)
        finished = torch.zeros(k, dtype=torch.bool, device=self.device)

        for _ in range(self.max_len - len(prefix)):
            seq_len = batch_input.size(1)

            padding_mask = torch.zeros((k, seq_len), dtype=torch.bool, device=self.device)
            logits = self.model(batch_input, padding_mask)  # [k, seq_len, vocab]
            next_logits = logits[:, -1, :]                  # [k, vocab]

            next_token = self._sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                greedy=greedy,
            )

            # once finished, keep appending EOS so tensor shapes stay aligned
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.eos_id),
                next_token,
            )

            batch_input = torch.cat([batch_input, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == self.eos_id)

            if finished.all():
                break

        results = []
        for i in range(k):
            full_ids = batch_input[i].tolist()
            parsed = self.parse_generated_sequence(full_ids)
            results.append(parsed)

        return results

    def parse_generated_sequence(self, full_ids: List[int]) -> Dict[str, Any]:
        """
        Parse:
        <BOS> <CDS> cds... <UTR5> reversed_utr5 ... [<UTR3> utr3 ...] <EOS>

        Returns utr5 in natural orientation.
        """
        try:
            utr5_start = full_ids.index(self.utr5_id) + 1
        except ValueError:
            raise ValueError("Generated sequence does not contain <UTR5> token.")

        eos_pos = full_ids.index(self.eos_id) if self.eos_id in full_ids else len(full_ids)

        if self.utr3_id is not None and self.utr3_id in full_ids[utr5_start:eos_pos]:
            utr3_marker_pos = full_ids.index(self.utr3_id, utr5_start, eos_pos)
            utr5_rev_ids = full_ids[utr5_start:utr3_marker_pos]
            utr3_ids = full_ids[utr3_marker_pos + 1:eos_pos]
        else:
            utr5_rev_ids = full_ids[utr5_start:eos_pos]
            utr3_ids = []

        # keep only nucleotide ids in case the model emits weird specials
        utr5_rev_ids = [x for x in utr5_rev_ids if x in self.nucleotide_ids]
        utr3_ids = [x for x in utr3_ids if x in self.nucleotide_ids]

        utr5_reversed = self.detokenize_seq(utr5_rev_ids)
        utr5 = utr5_reversed[::-1]   # restore natural 5'UTR orientation
        utr3 = self.detokenize_seq(utr3_ids)

        return {
            "full_token_ids": full_ids,
            "utr5": utr5,
            "utr3": utr3,
            "utr5_reversed_generated": utr5_reversed,
        }


def generate_samples(
    model: MRNATransformer,
    dataset: pd.DataFrame,
    samples_per_cds: int = 3,
    max_samples: int = None,
) -> pd.DataFrame:
    """
    For each CDS in the dataset, generate n_samples UTR sequences using the
    provided model.
    """
    sampler = MRNAInferenceSampler(model)
    all_samples = []

    if max_samples is not None:
        dataset = dataset.head(max_samples)

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Generating samples"):
        seq_id = row["id"]
        cds = row["cds"]
        utr5_ref = row["utr5"]
        utr3_ref = row["utr3"]

        all_samples.append({
            "id": seq_id,
            "sample": "gt",
            "cds": cds,
            "utr5": utr5_ref,
            "utr3": utr3_ref,
        })

        samples = sampler.generate_from_cds(
            cds=cds,
            k=samples_per_cds,
            temperature=1.0,
            top_k=20,
            greedy=False,
        )

        for i, sample in enumerate(samples):
            all_samples.append({
                "id": seq_id,
                "sample": f"sample_{i}",
                "cds": cds,
                "utr5": sample["utr5"],
                "utr3": sample["utr3"],
            })

    samples_df = pd.DataFrame(all_samples)
    samples_df.fillna("", inplace=True)
    return samples_df


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate UTR sequences from a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--nheads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length for generation.")
    parser.add_argument("--dataset_csv", type=str, default="data/pretraining/small_test.csv", help="Path to the input dataset CSV file.")
    parser.add_argument("--output_csv", type=str, default="generated_samples.csv", help="Path to save the generated samples CSV file.")
    parser.add_argument("--samples_per_cds", type=int, default=3, help="Number of samples to generate per CDS.")
    parser.add_argument("--max_samples", type=int, help="Max number of total samples to generate")
    return parser


def main() -> None:
    parser = _get_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = pd.read_csv(args.dataset_csv)
    print(f"Loaded dataset with {len(dataset)} samples from {args.dataset_csv}")

    model = MRNATransformer(
        vocab_size=len(set(MRNA_VOCAB.values())),
        d_model=args.d_model,
        nhead=args.nheads,
        num_layers=args.n_layers,
        max_len=768,
    ).to(device)
    load_pretrained_weights(model, args.model_path, device)
    print(f"Loaded model from {args.model_path}")

    generated_samples_df = generate_samples(model, dataset, args.samples_per_cds, args.max_samples)
    generated_samples_df.to_csv(args.output_csv, index=False)
    print(f"Saved generated samples to {args.output_csv}")


if __name__ == "__main__":
    main()
