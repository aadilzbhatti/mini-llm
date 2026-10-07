"""Tiny model configuration.

Deliberately small defaults: fast correctness experiments on a laptop, not
quality. Change the numbers here (or pass CLI flags) to scale up.
"""

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mini_llm.model import ModelCustomTransformer


@dataclass
class DynamicModelConfig:
    """Tensors derived from the config by the model (RoPE tables); never serialized."""

    cosine: torch.Tensor
    sine: torch.Tensor


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    use_rope_embeddings: bool = True
    # Filled in by ModelCustomTransformer.__init__; excluded from to_dict() so checkpoints
    # stay plain numbers and ModelConfig.from_dict(ckpt["config"]) keeps working.
    dynamic_model_config: DynamicModelConfig | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Rebuild from a checkpoint's saved config, ignoring keys the config no longer has
        (older checkpoints saved `use_cache`, which is no longer a setting). Checkpoints from before
        RoPE have no `use_rope_embeddings` key and were trained with learned absolute positions."""
        names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in names}
        d.setdefault("use_rope_embeddings", False)
        return cls(**d)

    def to_dict(self) -> dict[str, int | float | bool]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "dynamic_model_config"}


def build_model(cfg: ModelConfig) -> "ModelCustomTransformer":
    """Instantiate the custom Transformer from a config."""
    from mini_llm.model import ModelCustomTransformer  # deferred: model.py imports this module

    return ModelCustomTransformer(cfg)
