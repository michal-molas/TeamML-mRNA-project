"""Shared loading and ensemble inference for pretrained RiboNN models."""

import importlib
import sys
from pathlib import Path

import pandas as pd
import torch


RIBONN_MAX_UTR5_LEN = 1_381
RIBONN_MAX_CDS_UTR3_LEN = 11_937
RIBONN_MAX_TX_LEN = RIBONN_MAX_UTR5_LEN + RIBONN_MAX_CDS_UTR3_LEN
RIBONN_LEN_AFTER_CONV = 9

RIBONN_CONFIG = {
    "with_NAs": False,
    "split_utr5_cds_utr3_channels": False,
    "label_codons": True,
    "label_utr5": False,
    "label_utr3": False,
    "label_splice_sites": False,
    "label_up_probs": False,
    "filters": 64,
    "conv_stride": 1,
    "conv_padding": 0,
    "ln_epsilon": 0.007,
    "dropout": 0.3,
    "residual": False,
    "activation": "relu",
    "kernel_size": 5,
    "num_conv_layers": 10,
    "len_after_conv": RIBONN_LEN_AFTER_CONV,
    "num_targets": 78,
    "max_shift": 0,
    "symmetric_shift": True,
}


def _import_ribonn_model():
    """Import RiboNN while supporting the submodule's legacy package layout."""
    utils = importlib.import_module("RiboNN.src.utils")
    helpers = importlib.import_module("RiboNN.src.utils.helpers")
    sys.modules.setdefault("src.utils", utils)
    sys.modules.setdefault("src.utils.helpers", helpers)

    module = importlib.import_module("RiboNN.src.model")
    return module.RiboNN


def load_ribonn(weights_path, device, verbose=False):
    """Load one frozen RiboNN model from a state-dict checkpoint."""
    model_class = _import_ribonn_model()
    model = model_class(**dict(RIBONN_CONFIG))
    state_dict = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if verbose:
        print(
            f"[ribonn] loaded {weights_path} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
        )

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _select_model_paths(weights_folder, top_k):
    weights_folder = Path(weights_folder)
    run_df = pd.read_csv(weights_folder / "runs.csv")
    fold_keys = run_df["params.test_fold"].astype(str)

    model_paths = []
    for test_fold in sorted(fold_keys.unique()):
        fold_runs = run_df.loc[fold_keys == test_fold]
        fold_runs = fold_runs.sort_values("metrics.val_r2", ascending=False).head(top_k)
        model_paths.extend(
            weights_folder / str(run_id) / "state_dict.pth"
            for run_id in fold_runs["run_id"]
        )
    return model_paths


class RiboNNEnsemble(torch.nn.Module):
    """Preload and average the top-k RiboNN models from every test fold."""

    def __init__(self, weights_folder, device, top_k=5, verbose=False):
        super().__init__()
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        model_paths = _select_model_paths(weights_folder, top_k)
        if not model_paths:
            raise ValueError(f"No RiboNN models found in {weights_folder}")

        if verbose:
            print(f"[ribonn] ensemble size={len(model_paths)}", file=sys.stderr)

        self.models = torch.nn.ModuleList(
            [load_ribonn(path, device, verbose=verbose) for path in model_paths]
        )
        self.n = len(self.models)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        """Keep the frozen ensemble in evaluation mode."""
        return super().train(False)

    def forward(self, ribonn_input):
        prediction_sum = None
        for model in self.models:
            prediction = model(ribonn_input)
            prediction_sum = (
                prediction
                if prediction_sum is None
                else prediction_sum + prediction
            )
        return prediction_sum / self.n
