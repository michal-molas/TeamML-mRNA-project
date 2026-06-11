from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class LoArmConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    max_len: int = 238
    prefix_len: int = 170
    target_len: int = 68
    num_regions: int = 5
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)


class LoArmTransformer(nn.Module):
    """Non-causal Transformer for LO-ARM masked target unrolling."""

    def __init__(self, config: LoArmConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_len, config.d_model)
        self.region_embedding = nn.Embedding(config.num_regions, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            batch_first=True,
            activation="gelu",
            dropout=config.dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.value_head = nn.Linear(config.d_model, config.vocab_size)
        self.order_head = nn.Linear(config.d_model, 1)
        self.posterior_head = nn.Linear(config.d_model, 1)

    def forward(self, input_ids, padding_mask=None, region_ids=None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_len={self.config.max_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos = pos.expand(batch_size, seq_len)
        if region_ids is None:
            region_ids = torch.zeros_like(input_ids)
        hidden = (
            self.embedding(input_ids)
            + self.pos_embedding(pos)
            + self.region_embedding(region_ids)
        )
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)

        start = self.config.prefix_len
        end = start + self.config.target_len
        target_hidden = hidden[:, start:end, :]
        return {
            "hidden": target_hidden,
            "value_logits": self.value_head(target_hidden),
            "order_logits": self.order_head(target_hidden).squeeze(-1),
            "posterior_logits": self.posterior_head(target_hidden).squeeze(-1),
        }
