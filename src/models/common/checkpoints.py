from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
import os
import tempfile

import torch


CHECKPOINT_FORMAT_VERSION = 1
MODEL_TYPES = frozenset({"transformer", "indigo", "lo_arm"})
_MODEL_TYPE_ALIASES = {"loarm": "lo_arm", "lo-arm": "lo_arm"}
_V1_KEYS = {
    "format_version",
    "model_type",
    "model_config",
    "model_state_dict",
    "tokenizer_config",
    "data_config",
    "training",
}
_SHAPE_CONFIG_KEYS = frozenset({"vocab_size", "d_model", "max_len", "num_layers", "n_layers"})
_REQUIRED_MODEL_CONFIG = {
    "transformer": {"vocab_size", "d_model", "num_layers", "max_len"},
    "indigo": {"vocab_size", "d_model", "num_heads", "num_layers", "max_len"},
    "lo_arm": {
        "vocab_size",
        "d_model",
        "n_heads",
        "n_layers",
        "max_len",
        "prefix_len",
        "target_len",
    },
}


@dataclass(frozen=True)
class LoadedCheckpoint:
    format_version: int
    model_type: str
    model_config: dict
    model_state_dict: dict[str, torch.Tensor]
    tokenizer_config: dict
    data_config: dict
    training: dict
    is_legacy: bool = False

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "model_type": self.model_type,
            "model_config": dict(self.model_config),
            "model_state_dict": dict(self.model_state_dict),
            "tokenizer_config": dict(self.tokenizer_config),
            "data_config": dict(self.data_config),
            "training": dict(self.training),
        }


def _canonical_model_type(model_type: str) -> str:
    canonical = _MODEL_TYPE_ALIASES.get(str(model_type), str(model_type))
    if canonical not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type={model_type!r}; expected one of {sorted(MODEL_TYPES)}"
        )
    return canonical


def _as_mapping(value, field_name: str) -> dict:
    if value is None:
        return {}
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return vars(value).copy()
    raise TypeError(f"{field_name} must be a mapping or dataclass, got {type(value).__name__}")


def _validate_model_config(model_type: str, model_config: Mapping) -> None:
    missing = _REQUIRED_MODEL_CONFIG[model_type] - set(model_config)
    if model_type == "transformer" and not ({"n_heads", "nhead"} & set(model_config)):
        missing.add("n_heads")
    if missing:
        raise ValueError(
            f"{model_type} model_config is missing fields: {sorted(missing)}"
        )


def normalize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("model_state_dict must be a non-empty mapping")

    normalized = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("model_state_dict keys must be strings")
        if not torch.is_tensor(value):
            raise TypeError(f"model_state_dict[{key!r}] must be a tensor")
        while key.startswith("module."):
            key = key[len("module.") :]
        if key in normalized:
            raise ValueError(f"Duplicate state-dict key after prefix normalization: {key!r}")
        normalized[key] = value
    return normalized


def build_checkpoint(
    *,
    model_type: str,
    model_config,
    model_state_dict: Mapping[str, torch.Tensor],
    tokenizer_config=None,
    data_config=None,
    training=None,
) -> dict:
    canonical_model_type = _canonical_model_type(model_type)
    normalized_model_config = _as_mapping(model_config, "model_config")
    _validate_model_config(canonical_model_type, normalized_model_config)
    checkpoint = LoadedCheckpoint(
        format_version=CHECKPOINT_FORMAT_VERSION,
        model_type=canonical_model_type,
        model_config=normalized_model_config,
        model_state_dict=normalize_state_dict(model_state_dict),
        tokenizer_config=_as_mapping(tokenizer_config, "tokenizer_config"),
        data_config=_as_mapping(data_config, "data_config"),
        training=_as_mapping(training, "training"),
    )
    _validate_v1(checkpoint.to_dict())
    return checkpoint.to_dict()


def save_checkpoint(
    path,
    *,
    model_type: str,
    model_config,
    model_state_dict: Mapping[str, torch.Tensor],
    tokenizer_config=None,
    data_config=None,
    training=None,
) -> None:
    payload = build_checkpoint(
        model_type=model_type,
        model_config=model_config,
        model_state_dict=model_state_dict,
        tokenizer_config=tokenizer_config,
        data_config=data_config,
        training=training,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path,
    *,
    map_location="cpu",
    expected_model_type: str | None = None,
    legacy_config=None,
) -> LoadedCheckpoint:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    return load_checkpoint_payload(
        payload,
        expected_model_type=expected_model_type,
        legacy_config=legacy_config,
    )


def load_checkpoint_payload(
    payload,
    *,
    expected_model_type: str | None = None,
    legacy_config=None,
) -> LoadedCheckpoint:
    expected = (
        _canonical_model_type(expected_model_type)
        if expected_model_type is not None
        else None
    )
    if isinstance(payload, Mapping) and "format_version" in payload:
        return _load_v1(payload, expected)
    return _load_legacy(payload, expected, legacy_config)


def _validate_v1(payload: Mapping) -> None:
    missing = _V1_KEYS - set(payload)
    if missing:
        raise ValueError(f"Checkpoint v1 is missing fields: {sorted(missing)}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version={payload['format_version']!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    model_type = _canonical_model_type(payload["model_type"])
    for field_name in ("model_config", "tokenizer_config", "data_config", "training"):
        if not isinstance(payload[field_name], Mapping):
            raise TypeError(f"{field_name} must be a mapping")
    _validate_model_config(model_type, payload["model_config"])
    normalize_state_dict(payload["model_state_dict"])


def _load_v1(payload: Mapping, expected_model_type: str | None) -> LoadedCheckpoint:
    _validate_v1(payload)
    model_type = _canonical_model_type(payload["model_type"])
    if expected_model_type is not None and model_type != expected_model_type:
        raise ValueError(
            f"Checkpoint contains model_type={model_type!r}, "
            f"expected {expected_model_type!r}"
        )
    return LoadedCheckpoint(
        format_version=CHECKPOINT_FORMAT_VERSION,
        model_type=model_type,
        model_config=dict(payload["model_config"]),
        model_state_dict=normalize_state_dict(payload["model_state_dict"]),
        tokenizer_config=dict(payload["tokenizer_config"]),
        data_config=dict(payload["data_config"]),
        training=dict(payload["training"]),
    )


def _extract_legacy_state_dict(payload) -> tuple[dict[str, torch.Tensor], dict]:
    if not isinstance(payload, Mapping):
        raise TypeError("Legacy checkpoint must be a state dict or checkpoint mapping")

    if "model_state_dict" in payload:
        return normalize_state_dict(payload["model_state_dict"]), dict(payload)
    if "state_dict" in payload:
        return normalize_state_dict(payload["state_dict"]), dict(payload)
    return normalize_state_dict(payload), {}


def _infer_model_type(state_dict: Mapping[str, torch.Tensor]) -> str:
    keys = set(state_dict)
    if any(key.startswith("encoder.blocks.") for key in keys) or any(
        key.startswith("position_head_") for key in keys
    ):
        return "indigo"
    if {"value_head.weight", "order_head.weight", "posterior_head.weight"} & keys:
        return "lo_arm"
    if "seq_head.weight" in keys or any(key.startswith("transformer.layers.") for key in keys):
        return "transformer"
    raise ValueError("Could not infer model type from legacy state-dict keys")


def _layer_count(state_dict: Mapping[str, torch.Tensor], prefix: str) -> int:
    indices = set()
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        index = remainder.split(".", 1)[0]
        if index.isdigit():
            indices.add(int(index))
    if not indices:
        raise ValueError(f"Could not infer layer count from state-dict prefix {prefix!r}")
    return max(indices) + 1


def _infer_legacy_config(model_type: str, state_dict: Mapping[str, torch.Tensor]) -> dict:
    if model_type == "transformer":
        embedding = state_dict["embedding.weight"]
        positions = state_dict["pos_embedding.weight"]
        return {
            "vocab_size": embedding.shape[0],
            "d_model": embedding.shape[1],
            "num_layers": _layer_count(state_dict, "transformer.layers."),
            "max_len": positions.shape[0],
        }

    if model_type == "indigo":
        embedding = state_dict["encoder.embedding_layer.token_embedding.weight"]
        positions = state_dict["encoder.embedding_layer.position_embedding.weight"]
        relative = state_dict[
            "encoder.blocks.0.attention_layer.relative_positional_embedding.weight"
        ]
        return {
            "vocab_size": embedding.shape[0],
            "d_model": embedding.shape[1],
            "num_heads": embedding.shape[1] // relative.shape[1],
            "num_layers": _layer_count(state_dict, "encoder.blocks."),
            "max_len": positions.shape[0],
        }

    embedding = state_dict["embedding.weight"]
    positions = state_dict["pos_embedding.weight"]
    return {
        "vocab_size": embedding.shape[0],
        "d_model": embedding.shape[1],
        "n_layers": _layer_count(state_dict, "encoder.layers."),
        "max_len": positions.shape[0],
    }


def _merge_legacy_config(inferred: dict, embedded: dict, overrides: dict) -> dict:
    for source_name, source in (("embedded", embedded), ("override", overrides)):
        for key in _SHAPE_CONFIG_KEYS & source.keys() & inferred.keys():
            if source[key] != inferred[key]:
                raise ValueError(
                    f"Legacy {source_name} config {key}={source[key]!r} conflicts "
                    f"with state-dict value {inferred[key]!r}"
                )
    return {**inferred, **embedded, **overrides}


def _load_legacy(
    payload,
    expected_model_type: str | None,
    legacy_config,
) -> LoadedCheckpoint:
    state_dict, envelope = _extract_legacy_state_dict(payload)
    inferred_type = _infer_model_type(state_dict)
    if expected_model_type is not None and inferred_type != expected_model_type:
        raise ValueError(
            f"Legacy checkpoint appears to be {inferred_type!r}, "
            f"expected {expected_model_type!r}"
        )
    model_type = expected_model_type or inferred_type
    inferred_config = _infer_legacy_config(model_type, state_dict)
    embedded_config = _as_mapping(envelope.get("config"), "config")
    overrides = _as_mapping(legacy_config, "legacy_config")
    model_config = _merge_legacy_config(inferred_config, embedded_config, overrides)

    if model_type == "transformer" and not ({"n_heads", "nhead"} & model_config.keys()):
        raise ValueError(
            "Transformer legacy checkpoints do not encode the attention-head count; "
            "pass legacy_config={'n_heads': ...}"
        )
    if model_type == "lo_arm":
        missing = {"n_heads", "prefix_len", "target_len"} - model_config.keys()
        if missing:
            raise ValueError(
                "LO-ARM legacy checkpoint needs config values that cannot be inferred: "
                f"{sorted(missing)}"
            )
    _validate_model_config(model_type, model_config)

    training = {
        key: envelope[key]
        for key in ("epoch", "global_step", "best_loss", "best_val")
        if key in envelope
    }
    return LoadedCheckpoint(
        format_version=0,
        model_type=model_type,
        model_config=model_config,
        model_state_dict=state_dict,
        tokenizer_config=_as_mapping(envelope.get("tokenizer"), "tokenizer"),
        data_config=_as_mapping(envelope.get("data"), "data"),
        training=training,
        is_legacy=True,
    )
