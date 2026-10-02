"""Tiny model configuration.

Deliberately small defaults: fast correctness experiments on a laptop, not
quality. Change the numbers here (or pass CLI flags) to scale up.
"""

from dataclasses import asdict, dataclass

from mini_llm.model import ModelCustomTransformer


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    use_cache: bool = False

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def build_model(cfg: ModelConfig) -> ModelCustomTransformer:
    """Instantiate the custom Transformer from a config.

    Note the positional order the original constructor uses:
    (vocab_size, n_embd, n_head, n_layer, block_size, dropout).
    """
    return ModelCustomTransformer(
        cfg.vocab_size,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_layer,
        cfg.block_size,
        cfg.dropout,
        cfg.use_cache
    )
