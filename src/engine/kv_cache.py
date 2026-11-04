"""Step 1: the KV cache, owned by the engine instead of the model.

Today (engine/models/mini_llm.py, copied from mini_llm/model_fused.py) each attention layer allocates
and owns its cache (self.k_cache, self.v_cache) and its own idea of the current position (self.pos).
That's what ties the model to one sequence at a time and makes it stateful. This class takes all of
that over; the model receives it as an argument and keeps nothing between calls.

The contract the tests rely on (change it if you prefer, and update the tests with it):

    cache = KVCache(num_layers=..., batch_size=..., num_kv_heads=..., head_dim=..., max_len=...,
                    dtype=torch.float32, device="cpu")
    logits = model(input_ids, positions=positions, kv_cache=cache)   # writes this call's k/v
    cache.reset()                                                    # reuse the storage

Everything else is yours to design. Questions worth deciding deliberately:
  - Layout: one (batch, heads, max_len, head_dim) tensor per layer, or one big tensor for all layers?
    (Later, paging replaces this layout with blocks: keep the interface narrow enough that it can.)
  - Who decides where a token's k/v goes: the cache (from `positions`), or the caller?
  - The rolling window: positions keep growing past max_len (RoPE only depends on offsets), so a token
    at position p lives in slot p % max_len. How does attention know which slots are valid, and in what
    order? (mini_llm handles this with min(pos, block_size) and a wrapped ring; tests cover the wrap.)
  - Per-sequence lengths: step 3 batches prompts of different lengths, so "how full is the cache" is
    one number per sequence, not one for the batch.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        raise NotImplementedError("engine step 1: the engine-owned KV cache")

    def reset(self) -> None:
        """Forget every cached token, keeping the storage for the next generation."""
        raise NotImplementedError("engine step 1")
