"""Tiny model configuration.

Deliberately small defaults: fast correctness experiments on a laptop, not
quality. Change the numbers here (or pass CLI flags) to scale up.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mini_llm.model import ModelCustomTransformer


@dataclass
class DynamicModelConfig:
    """Values derived from a ModelConfig (head size, RoPE frequencies). Built by the model and passed
    to its submodules next to the config; never part of ModelConfig, so never serialized."""

    head_size: int
    speeds: torch.Tensor

    @classmethod
    def from_config(cls, config: "ModelConfig") -> "DynamicModelConfig":
        head_size = config.n_embd // config.n_head
        pair_indices = torch.arange(0, head_size // 2, 1, dtype=torch.float64)
        # float64 so the angle tables built from these stay accurate at large positions (the rolling cache's
        # positions keep growing past block_size); the tables are cast to float32 once built.
        speeds = 10000 ** (-2 * pair_indices / head_size)
        return cls(head_size=head_size, speeds=speeds)


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    use_rope_embeddings: bool = True
    # model_fused.py (one q/k/v matmul + scaled_dot_product_attention) instead of model.py's per-head
    # attention. The same function; checkpoints record which one trained them. Absent = False.
    fused_attention: bool = False

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
        return {f.name: getattr(self, f.name) for f in fields(self)}


def build_model(cfg: ModelConfig) -> "ModelCustomTransformer":
    """Instantiate the custom Transformer from a config: model_fused.py's when cfg.fused_attention."""
    # deferred: the model modules import this one
    if cfg.fused_attention:
        from mini_llm.model_fused import ModelCustomTransformer as Fused

        return Fused(cfg)  # type: ignore[return-value]  # same interface
    from mini_llm.model import ModelCustomTransformer

    return ModelCustomTransformer(cfg)
