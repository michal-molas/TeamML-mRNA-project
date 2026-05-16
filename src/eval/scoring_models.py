from types import SimpleNamespace

import pandas as pd
import torch

try:
    from .eval import ScoringModel
    from .schemas import SAMPLE_INDEX_COLS
except ImportError:
    from eval import ScoringModel
    from schemas import SAMPLE_INDEX_COLS

from transformer_training.ribonn_utils import (
    RIBONN_CONFIG,
    RIBONN_MAX_TX_LEN,
    load_ribonn,
    ribonn_predict_using_nested_cross_validation_models,
    ribonn_input_from_string,
)


class RiboNN(ScoringModel):
    def __init__(
        self,
        weights_path: str | None = None,
        weights_folder: str = "RiboNN/models/human",
        top_k_models_to_use: int = 5,
        batch_size: int = 32,
        device: str | torch.device | None = None,
        score_name: str = "ribonn_score",
    ):
        super().__init__()
        self.weights_path = weights_path
        self.weights_folder = weights_folder
        self.top_k_models_to_use = top_k_models_to_use
        self.batch_size = batch_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.score_name = score_name
        self._model = None

    def score_row(self, row) -> dict[str, float]:
        predictions = self._predict_sequences(
            pd.DataFrame(
                [
                    {
                        "utr5": row.get("utr5", ""),
                        "cds": row.get("cds", ""),
                        "utr3": row.get("utr3", ""),
                    }
                ]
            )
        )
        return {self.score_name: float(predictions.mean(dim=1).item())}

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        samples = features.fillna("")
        predictions = self._predict_sequences(samples)
        scores = pd.DataFrame(
            {
                self.score_name: predictions.mean(dim=1).cpu().numpy(),
            },
            index=samples.index,
        )
        return pd.concat([samples[SAMPLE_INDEX_COLS], scores], axis=1)

    def _predict_sequences(self, samples: pd.DataFrame) -> torch.Tensor:
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
        model, _ = load_ribonn(self.weights_path, self.device)
        with torch.no_grad():
            for start in range(0, len(samples), self.batch_size):
                end = start + self.batch_size
                all_predictions[start:end] += model(ribonn_input[start:end])

        return all_predictions.cpu()
