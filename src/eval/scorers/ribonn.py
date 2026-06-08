from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import torch
from tqdm import tqdm

from scorers.base import Scorer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "transformer_training"))
from finetune import (
    RIBONN_CONFIG,
    RIBONN_MAX_TX_LEN,
    RIBONN_MAX_UTR5_LEN,
    load_ribonn,
    ribonn_predict_using_nested_cross_validation_models,
)



def ribonn_input_from_string(
    utr5: str,
    cds: str,
    utr3: str,
    ribonn_max_len: int,
    label_codons: bool = True,
    ribonn_max_utr5_len: int = RIBONN_MAX_UTR5_LEN,
) -> torch.Tensor:
    """Convert sequence strings into the start-codon-aligned RiboNN input layout."""
    utr5 = str(utr5).strip().upper().replace("U", "T")
    cds = str(cds).strip().upper().replace("U", "T")
    utr3 = str(utr3).strip().upper().replace("U", "T")
    cds_utr3_len = len(cds) + len(utr3)

    if len(utr5) + cds_utr3_len > ribonn_max_len:
        raise ValueError(
            f"Sequence length {len(utr5) + cds_utr3_len} exceeds ribonn_max_len={ribonn_max_len}. "
            f"Lengths: utr5={len(utr5)}, cds={len(cds)}, utr3={len(utr3)}"
        )
    if len(utr5) > ribonn_max_utr5_len:
        raise ValueError(
            f"5' UTR length {len(utr5)} exceeds ribonn_max_utr5_len={ribonn_max_utr5_len}."
        )
    ribonn_max_cds_utr3_len = ribonn_max_len - ribonn_max_utr5_len
    if cds_utr3_len > ribonn_max_cds_utr3_len:
        raise ValueError(
            f"Combined CDS and 3' UTR length {cds_utr3_len} exceeds "
            f"ribonn_max_cds_utr3_len={ribonn_max_cds_utr3_len}."
        )
    if len(cds) % 3 != 0:
        raise ValueError("CDS length must be a multiple of 3.")
    if cds[-3:] not in ("TAA", "TGA", "TAG"):
        raise ValueError("CDS sequence must end with a stop codon.")

    num_channels = 5 if label_codons else 4
    out = torch.zeros(num_channels, ribonn_max_len, dtype=torch.float32)
    nt_to_idx = {"A": 0, "T": 1, "C": 2, "G": 3}

    utr5_start = ribonn_max_utr5_len - len(utr5)
    cds_start = ribonn_max_utr5_len
    utr3_start = ribonn_max_utr5_len + len(cds)
    for section_start, section_seq in (
        (utr5_start, utr5),
        (cds_start, cds),
        (utr3_start, utr3),
    ):
        for offset, nt in enumerate(section_seq):
            try:
                out[nt_to_idx[nt], section_start + offset] = 1.0
            except KeyError:
                raise ValueError(
                    f"Invalid nucleotide {nt!r} at position {section_start + offset}. "
                    "Allowed nucleotides: A, U, T, C, G."
                )

    if label_codons:
        for codon_pos in range(cds_start, cds_start + len(cds), 3):
            out[4, codon_pos] = 1.0

    return out



class RiboNNScorer(Scorer):
    def __init__(
        self,
        weights_path: str | None = None,
        weights_folder: str = "RiboNN/models/human",
        top_k_models_to_use: int = 5,
        batch_size: int = 32,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.name = "RiboNN"
        self.score_names = ("ribonn_te",)
        self.weights_path = weights_path
        self.weights_folder = weights_folder
        self.top_k_models_to_use = top_k_models_to_use
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._model = None

    def _predict_sequences(self, samples: pd.DataFrame, progress_bar: bool = False) -> torch.Tensor:
        if samples.empty:
            return torch.empty(0, RIBONN_CONFIG["num_targets"])

        ribonn_input = torch.stack(
            [
                ribonn_input_from_string(
                    utr5=str(row["utr5"]),
                    cds=str(row["cds"]),
                    utr3=str(row["utr3"]),
                    ribonn_max_len=RIBONN_MAX_TX_LEN,
                    label_codons=RIBONN_CONFIG["label_codons"],
                )
                for _, row in samples.iterrows()
            ],
            dim=0,
        )

        ribonn_input = ribonn_input.to(self.device)
        if self.weights_path is None:
            args = SimpleNamespace(
                ribonn_weights_folder=self.weights_folder,
                top_k_models_to_use=self.top_k_models_to_use,
            )
            with torch.no_grad():
                return ribonn_predict_using_nested_cross_validation_models(
                    args=args,
                    device=self.device,
                    ribonn_input=ribonn_input,
                    batch_width=len(samples),
                ).cpu()

        all_predictions = torch.zeros(
            (len(samples), RIBONN_CONFIG["num_targets"]),
            dtype=torch.float32,
            device=self.device,
        )
        print(f"Loading RiboNN model from {self.weights_path} on device {self.device}...")
        model, _ = load_ribonn(self.weights_path, self.device)
        with torch.no_grad():
            iterator = range(0, len(samples), self.batch_size)
            if progress_bar:
                iterator = tqdm(iterator, total=(len(samples) + self.batch_size - 1) // self.batch_size, desc="Predicting with RiboNN")
            for start in iterator:
                end = min(start + self.batch_size, len(samples))
                all_predictions[start:end] += model(ribonn_input[start:end])

        return all_predictions.cpu()

    def score(
        self,
        utr5: str,
        cds: str,
        utr3: str,
    ) -> dict[str, float]:
        predictions = self._predict_sequences(
            pd.DataFrame(
                [
                    {
                        "utr5": utr5,
                        "cds": cds,
                        "utr3": utr3,
                    }
                ]
            )
        )
        return {"ribonn_te": float(predictions.mean(dim=1).item())}

    def score_df(
        self,
        df: pd.DataFrame,
        index_cols: list[str] = ["id"],
        progress_bar: bool = False,
    ) -> pd.DataFrame:
        predictions = self._predict_sequences(df, progress_bar=progress_bar)
        scores = pd.DataFrame(
            {
                "ribonn_te": predictions.mean(dim=1).cpu().numpy(),
            }
        )
        return pd.concat([df[index_cols].reset_index(drop=True), scores], axis=1)
