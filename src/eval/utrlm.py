from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Literal

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


Task = Literal["mrl", "te", "el"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UTRLM_ROOT = PROJECT_ROOT / "UTR-LM"
UTRLM_SCRIPTS = DEFAULT_UTRLM_ROOT / "Scripts"
if UTRLM_SCRIPTS.exists():
    sys.path.insert(0, str(UTRLM_SCRIPTS))

from esm import Alphabet  # noqa: E402
from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS  # noqa: E402
from esm.model.esm2_supervised import ESM2 as ESM2_SUPERVISED  # noqa: E402


LAYERS = 6
HEADS = 16
EMBED_DIM = 128
NODES = 40
MRL_DROPOUT3 = 0.5
TE_EL_DROPOUT3 = 0.2
FILTER_LEN = 8
NBR_FILTERS = 120
REPR_LAYERS = [0, LAYERS]
VALID_BASES = set("AGCT")

ALPHABET = Alphabet(standard_toks="AGCT", mask_prob=0.0)


@dataclass(frozen=True)
class UTRLMPredictions:
    mrl: float | None = None
    te: float | None = None
    el: float | None = None


def print_mps_memory_usage():
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return

    allocated = torch.mps.current_allocated_memory()
    driver = torch.mps.driver_allocated_memory()
    max_recommended = torch.mps.recommended_max_memory()

    print(f"Tensor allocated: {allocated / 1024**3:.2f} GB")
    print(f"Driver allocated: {driver / 1024**3:.2f} GB")
    print(f"Recommended max:  {max_recommended / 1024**3:.2f} GB")
    print(f"Tensor usage:     {100 * allocated / max_recommended:.1f}%")
    print(f"Driver usage:     {100 * driver / max_recommended:.1f}%")


class UTRLMHead(nn.Module):
    def __init__(self, *, backbone: Task, dropout3: float) -> None:
        super().__init__()
        if backbone == "mrl":
            self.esm2 = ESM2_SISS(
                num_layers=LAYERS,
                embed_dim=EMBED_DIM,
                attention_heads=HEADS,
                alphabet=ALPHABET,
            )
        elif backbone in {"te", "el"}:
            self.esm2 = ESM2_SUPERVISED(
                num_layers=LAYERS,
                embed_dim=EMBED_DIM,
                attention_heads=HEADS,
                alphabet=ALPHABET,
            )
        else:
            raise ValueError(f"Unsupported UTR-LM task: {backbone}")

        self.conv1 = nn.Conv1d(EMBED_DIM, NBR_FILTERS, kernel_size=FILTER_LEN, padding="same")
        self.conv2 = nn.Conv1d(NBR_FILTERS, NBR_FILTERS, kernel_size=FILTER_LEN, padding="same")
        self.dropout1 = nn.Dropout(0)
        self.dropout2 = nn.Dropout(0)
        self.dropout3 = nn.Dropout(dropout3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(EMBED_DIM, NODES)
        self.linear = nn.Linear(NBR_FILTERS, NODES)
        self.output = nn.Linear(NODES, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        esm_out = self.esm2(
            tokens,
            REPR_LAYERS,
            need_head_weights=True,
            return_contacts=False,
            return_representation=True,
        )
        bos_embedding = esm_out["representations"][LAYERS][:, 0].unsqueeze(2)
        x = self.flatten(bos_embedding)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout3(x)
        return self.output(x)


def choose_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is not None and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_sequence(sequence: str | None) -> str:
    sequence = "".join(str(sequence or "").split()).upper().replace("U", "T")
    invalid = sorted(set(sequence) - VALID_BASES)
    if invalid:
        raise ValueError(f"Sequence contains unsupported bases: {', '.join(invalid)}")
    return sequence


def normalize_cell_line(cell_line: str) -> str:
    aliases = {
        "hek": "HEK",
        "hek293": "HEK",
        "muscle": "Muscle",
        "pc3": "pc3",
        "pc-3": "pc3",
    }
    try:
        return aliases[cell_line.lower()]
    except KeyError as exc:
        raise ValueError("cell line must be one of: HEK, Muscle, pc3") from exc


def normalize_tasks(tasks: Iterable[str]) -> tuple[Task, ...]:
    normalized = tuple(task.lower() for task in tasks)
    invalid = sorted(set(normalized) - {"mrl", "te", "el"})
    if invalid:
        raise ValueError(f"Unsupported UTR-LM task(s): {', '.join(invalid)}")
    return normalized  # type: ignore[return-value]


def _checkpoint_dir(utrlm_root: Path) -> Path:
    return utrlm_root / "Model" / "Downstream"


def mrl_checkpoint(utrlm_root: Path = DEFAULT_UTRLM_ROOT) -> Path:
    return (
        _checkpoint_dir(utrlm_root)
        / "MRL"
        / "MJ3_seed1337_ESM2SISS_FS4.1.ep93.1e-2.dr5_unmod_1_utr_10folds_rl_"
        "LabelScalerFalse_LabelLog2False_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_"
        "nodes40_dropout30.5_finetuneTrue_huberlossTrue_lr0.01_fold0_epoch299.pt"
    )


def te_el_checkpoints(
    *,
    task: Literal["te", "el"],
    cell_line: str,
    fold: int | None,
    finetuned: bool,
    utrlm_root: Path = DEFAULT_UTRLM_ROOT,
) -> list[Path]:
    normalized_cell_line = normalize_cell_line(cell_line)
    label_type = "te_log" if task == "te" else "rnaseq_log"
    pattern = (
        f"MJ4_seed1337_{task.upper()}_ESM2SI_3.1.1e-2.*.dropout2_"
        f"{normalized_cell_line}_{label_type}_utr_seqlen100_*dropout30.2_"
        f"finetune{finetuned}_huberlossTrue_magicFalse_lr0.01"
    )
    if fold is None:
        pattern += "_fold*_epoch*.pt"
    else:
        pattern += f"_fold{fold}_epoch*.pt"

    paths = sorted((_checkpoint_dir(utrlm_root) / "TE_EL").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No UTR-LM checkpoint matched: {pattern}")
    return paths


class UTRLMPredictor:
    """Lazy-loading UTR-LM predictor for MRL, TE, and EL.

    The released UTR-LM downstream checkpoints all consume the 5' UTR. The
    public methods accept utr5/cds/utr3 triples so callers can pass eval rows
    directly, but cds and utr3 are not used by these checkpoints.
    """

    def __init__(
        self,
        *,
        utrlm_root: str | Path = DEFAULT_UTRLM_ROOT,
        device: str | torch.device | None = None,
        te_cell_line: str = "HEK",
        el_cell_line: str | None = None,
        fold: int | None = None,
        finetuned: bool = True,
        trim_te_el_to_last_100: bool = True,
        batch_size: int = 32,
        mrl_model_path: str | Path | None = None,
        te_model_paths: Iterable[str | Path] | None = None,
        el_model_paths: Iterable[str | Path] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("UTR-LM batch_size must be greater than 0.")

        self.utrlm_root = Path(utrlm_root)
        self.device = choose_device(device)
        self.te_cell_line = normalize_cell_line(te_cell_line)
        self.el_cell_line = normalize_cell_line(el_cell_line or te_cell_line)
        self.fold = fold
        self.finetuned = finetuned
        self.trim_te_el_to_last_100 = trim_te_el_to_last_100
        self.batch_size = batch_size

        self.mrl_model_path = Path(mrl_model_path) if mrl_model_path is not None else mrl_checkpoint(self.utrlm_root)
        self._te_model_paths_override = (
            [Path(path) for path in te_model_paths]
            if te_model_paths is not None
            else None
        )
        self._el_model_paths_override = (
            [Path(path) for path in el_model_paths]
            if el_model_paths is not None
            else None
        )
        self._models: dict[tuple[Task, Path], UTRLMHead] = {}

    def predict(
        self,
        utr5: str,
        cds: str = "",
        utr3: str = "",
        *,
        tasks: Iterable[Task] = ("mrl", "te", "el"),
    ) -> UTRLMPredictions:
        predictions = self.predict_many(
            [{"utr5": utr5, "cds": cds, "utr3": utr3}],
            tasks=tasks,
        )
        row = predictions.iloc[0]
        return UTRLMPredictions(
            mrl=float(row["utrlm_mrl"]) if "utrlm_mrl" in row else None,
            te=float(row["utrlm_te"]) if "utrlm_te" in row else None,
            el=float(row["utrlm_el"]) if "utrlm_el" in row else None,
        )

    def predict_dict(
        self,
        utr5: str,
        cds: str = "",
        utr3: str = "",
        *,
        tasks: Iterable[Task] = ("mrl", "te", "el"),
    ) -> dict[str, float]:
        prediction = self.predict(utr5=utr5, cds=cds, utr3=utr3, tasks=tasks)
        return {
            key: value
            for key, value in {
                "utrlm_mrl": prediction.mrl,
                "utrlm_te": prediction.te,
                "utrlm_el": prediction.el,
            }.items()
            if value is not None
        }

    def predict_many(
        self,
        rows: pd.DataFrame | Iterable[dict[str, str]],
        *,
        tasks: Iterable[Task] = ("mrl", "te", "el"),
        progress_bar: bool = True,
    ) -> pd.DataFrame:
        samples = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        samples = samples.fillna("")
        utr5_sequences = [clean_sequence(seq) for seq in samples.get("utr5", [])]
        output: dict[str, list[float]] = {}

        if "utr5" not in samples:
            raise ValueError("UTR-LM predictions require a 'utr5' column or key.")

        requested_tasks = normalize_tasks(tasks)
        if "mrl" in requested_tasks:
            output["utrlm_mrl"] = self._predict_sequences(
                sequences=utr5_sequences,
                task="mrl",
                model_paths=[self.mrl_model_path],
                progress_bar=progress_bar,
            )
        if "te" in requested_tasks:
            output["utrlm_te"] = self._predict_sequences(
                sequences=self._prepare_te_el_sequences(utr5_sequences),
                task="te",
                model_paths=self._model_paths_for_task("te"),
                progress_bar=progress_bar,
            )
        if "el" in requested_tasks:
            output["utrlm_el"] = self._predict_sequences(
                sequences=self._prepare_te_el_sequences(utr5_sequences),
                task="el",
                model_paths=self._model_paths_for_task("el"),
                progress_bar=progress_bar,
            )

        return pd.DataFrame(output, index=samples.index)

    def _model_paths_for_task(self, task: Literal["te", "el"]) -> list[Path]:
        if task == "te":
            if self._te_model_paths_override is not None:
                return self._te_model_paths_override
            return te_el_checkpoints(
                task="te",
                cell_line=self.te_cell_line,
                fold=self.fold,
                finetuned=self.finetuned,
                utrlm_root=self.utrlm_root,
            )
        if self._el_model_paths_override is not None:
            return self._el_model_paths_override
        return te_el_checkpoints(
            task="el",
            cell_line=self.el_cell_line,
            fold=self.fold,
            finetuned=self.finetuned,
            utrlm_root=self.utrlm_root,
        )

    def _prepare_te_el_sequences(self, sequences: list[str]) -> list[str]:
        if not self.trim_te_el_to_last_100:
            return sequences
        return [seq[-100:] for seq in sequences]

    def _predict_sequences(
        self,
        *,
        sequences: list[str],
        task: Task,
        model_paths: list[Path],
        progress_bar: bool = True,
    ) -> list[float]:
        if not sequences:
            return []

        predictions = torch.zeros(len(sequences), dtype=torch.float32)
        models = [self._load_model(task, model_path) for model_path in model_paths]

        print(f"Predicting UTR-LM {task.upper()} with {len(models)} model(s) on {self.device}...")
        print(f"Batch size: {self.batch_size}")

        iterator = range(0, len(sequences), self.batch_size)
        if progress_bar:
            iterator = tqdm(
                iterator,
                desc=f"Predicting UTR-LM {task.upper()}",
                unit="batch",
                total=(len(sequences) + self.batch_size - 1) // self.batch_size,
            )


        with torch.no_grad():
            for start in iterator:
                end = min(start + self.batch_size, len(sequences))
                tokens = self._tokenize_many(sequences[start:end]).to(self.device)
                for model in models:
                    predictions[start:end] += model(tokens).reshape(-1).detach().cpu()
                print_mps_memory_usage()

        predictions /= len(models)
        return predictions.tolist()

    def _load_model(self, task: Task, model_path: Path) -> UTRLMHead:
        model_path = model_path.resolve()
        key = (task, model_path)
        if key in self._models:
            return self._models[key]

        dropout3 = MRL_DROPOUT3 if task == "mrl" else TE_EL_DROPOUT3
        model = UTRLMHead(backbone=task, dropout3=dropout3).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict({key.replace("module.", ""): value for key, value in state_dict.items()})
        model.eval()
        self._models[key] = model
        return model

    @staticmethod
    def _tokenize_many(sequences: list[str]) -> torch.Tensor:
        encoded = [[ALPHABET.cls_idx] + [ALPHABET.get_idx(base) for base in seq] + [ALPHABET.eos_idx] for seq in sequences]
        max_len = max(len(seq) for seq in encoded)
        tokens = torch.full((len(encoded), max_len), ALPHABET.padding_idx, dtype=torch.long)
        for index, seq in enumerate(encoded):
            tokens[index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return tokens
