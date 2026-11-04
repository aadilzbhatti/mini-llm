"""Custom decoder-only Transformer, with fused attention.

Branched from model.py (see its git history): identical except MultiHeadAttention, which projects
q, k and v for all heads in one matmul, keeps one KV cache per layer, and calls
F.scaled_dot_product_attention instead of running a Head module per head. Selected by
ModelConfig.fused_attention (config.build_model); model.py's per-head checkpoints load into it.

model.py's docstring follows.

Custom decoder-only Transformer.

Copied from the original project's src/text_prediction/model.py
(commit 3ba9e94), then corrected. The attention math, residual structure,
dropout placement and sampling are unchanged; the fixes are marked inline
with `FIX (#n)` comments and cover:

  #1 per-head LayerNorm removed (Block.ln1 already pre-normalizes)
  #2 the unused `mask` / `attention_mask` plumbing removed
  #3 per-module init_weights no longer overwritten by the generic pass
  #4 blocks held in an nn.ModuleList rather than an uncallable nn.Sequential
  #6 no silent .to(device) coercion of inputs
  #7 unused self.step removed
  #8 generate() sets eval mode itself and restores the previous mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_llm.config import DynamicModelConfig, ModelConfig

RopeAngles = tuple[torch.Tensor, torch.Tensor]  # (cos, sin) rows for a call's positions, (T, head_size / 2) each


def rope_table(speeds: torch.Tensor, n: int) -> RopeAngles:
    """cos and sin of position * speed for positions 0 .. n - 1: (n, head_size / 2) each, on the CPU.
    Angles in float64 (position 50,000 is ~5e4 rad, where float32 keeps only ~4e-3 rad), stored as float32."""
    positions = torch.arange(n, dtype=torch.float64)
    angles = torch.outer(positions, speeds.double())
    return torch.cos(angles).float(), torch.sin(angles).float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (even, odd) pair of x's last dimension: row t by the angles in cos[t], sin[t].
    x is (B, T, head_size); cos and sin are (T, head_size / 2)."""
    ret = torch.empty_like(x)
    ret[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    ret[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    return ret


class MultiHeadAttention(nn.Module):
    """All heads at once: one fused q/k/v projection, one KV cache per layer, and
    F.scaled_dot_product_attention (a fused kernel where the backend has one) in place of model.py's
    per-head Head modules, each with its own projections, cache and explicit softmax(q k^T).

    The same function as model.py's attention: qkv.weight is model.py's per-head query, key and value
    weights stacked ([q heads | k heads | v heads], heads in order), and the output is the heads
    concatenated in order, as torch.cat did. Per-head checkpoints load into it (_load_from_state_dict).
    Not kept: attention_values (SDPA never materializes the attention matrix)."""

    mask: torch.Tensor

    def __init__(self, config: ModelConfig, dynamic_config: DynamicModelConfig):
        super().__init__()
        self.n_head, self.head_size = config.n_head, dynamic_config.head_size
        self.block_size = config.block_size
        self.qkv = nn.Linear(config.n_embd, 3 * self.n_head * self.head_size, bias=False)
        self.proj = nn.Linear(self.n_head * self.head_size, config.n_embd)
        self.dropout_p = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        self.use_rope_embeddings = config.use_rope_embeddings
        # Only needed for a multi-token write on top of an existing cache (see forward). Derived from the
        # config, so not saved (model.py saved its per-head tril masks; loading drops them).
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)), persistent=False
        )

        # (B, n_head, block_size, head_size), allocated once and written in place; min(pos, block_size)
        # says how many positions are filled. Not saved with the model.
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.pos = 0  # absolute position of the next token, only ever grows

        self.init_weights()

    def init_weights(self):
        # The same distribution as model.py: xavier_normal_ on each head's (head_size, n_embd) q, k and
        # v weights (not on the whole stacked matrix, whose fans differ). Not the same RNG draws.
        w = self.qkv.weight.data.view(3, self.n_head, self.head_size, -1)
        for h in range(self.n_head):
            for which in range(3):
                nn.init.xavier_normal_(w[which, h])
        nn.init.xavier_uniform_(self.proj.weight)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Per-head checkpoints (model.py) hold heads.{h}.{query,key,value}.weight and heads.{h}.tril:
        # stack the weights into qkv.weight in the order forward() splits it, drop the masks.
        if f"{prefix}heads.0.query.weight" in state_dict:
            state_dict[f"{prefix}qkv.weight"] = torch.cat(
                [
                    state_dict.pop(f"{prefix}heads.{h}.{name}.weight")
                    for name in ("query", "key", "value")
                    for h in range(self.n_head)
                ]
            )
            for h in range(self.n_head):
                state_dict.pop(f"{prefix}heads.{h}.tril", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, use_cache: bool = False, rope: "RopeAngles | None" = None):
        """rope: the (cos, sin) rows for this call's positions, sliced once per forward by the model
        (ModelCustomTransformer.rope_angles); required when RoPE is on."""
        B, T, C = x.shape
        # (B, T, 3 * n_head * hs) -> three (B, n_head, T, hs)
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_size).permute(2, 0, 3, 1, 4)
        if self.use_rope_embeddings:
            assert rope is not None, "RoPE attention needs the model's angle rows for this call's positions"
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)  # (T, hs/2) broadcasts over batch and heads

        if use_cache:
            if self.k_cache is None or self.k_cache.size(0) != B or self.k_cache.dtype != k.dtype:
                # cache is uninitialized or batch size changed; allocate it
                self.k_cache = k.new_empty(B, self.n_head, self.block_size, self.head_size)
                self.v_cache = torch.empty_like(self.k_cache)
            start, end = self.pos % self.block_size, (self.pos % self.block_size) + T
            assert end <= self.block_size, f"Cache overflow: pos={self.pos}, T={T}, block_size={self.block_size}"
            # once the buffer has wrapped, slots are out of position order, so only single tokens may be written
            assert (
                T == 1 or self.pos < self.block_size
            ), f"Multi-token write after the cache wrapped: pos={self.pos}, T={T}, block_size={self.block_size}"
            self.k_cache[:, :, start:end] = k
            self.v_cache[:, :, start:end] = v
            self.pos += T
            filled = min(self.pos, self.block_size)
            k, v = self.k_cache[:, :, :filled], self.v_cache[:, :, :filled]

        T_total = k.size(2)  # != T when attending over the cache
        # One new token sees the whole cache (no mask). A full causal pass uses SDPA's is_causal path. Only
        # a multi-token write on top of an existing cache needs the explicit offset mask.
        mask = self.mask[T_total - T : T_total, :T_total] if 1 < T < T_total else None
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=1 < T == T_total,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # (B, n_head, T, hs); default scale hs**-0.5, as before
        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.head_size)  # heads in order, as torch.cat
        return self.dropout(self.proj(out))

    def clear_cache(self):
        self.pos = 0  # keep the buffers: the next generation reuses them


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.gelu_activation = None  # Store GELU activation here

    def forward(self, x: torch.Tensor):
        x = self.net[0](x)
        x = self.net[1](x)  # GELU activation
        self.gelu_activation = x.detach()  # store the activation, detaching to avoid gradient issues.
        x = self.net[2](x)
        x = self.net[3](x)
        return x


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config: ModelConfig, dynamic_config: DynamicModelConfig):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(config, dynamic_config)
        self.ffwd = FeedForward(config.n_embd, config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor, use_cache: bool = False, rope: "RopeAngles | None" = None):
        x = x + self.sa(self.ln1(x), use_cache, rope)
        x = x + self.ffwd(self.ln2(x))
        return x

    def clear_cache(self):
        self.sa.clear_cache()


class ModelCustomTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_rope_embeddings = config.use_rope_embeddings
        self.block_size = config.block_size
        dynamic_config = DynamicModelConfig.from_config(config)  # derived values, not saved in checkpoints

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        if not self.use_rope_embeddings:
            self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, dynamic_config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        if self.use_rope_embeddings:
            # One (positions, head_size / 2) cos/sin table for the whole model: every head in every layer
            # rotates the same positions, so they share it instead of each recomputing its angles per call.
            # Not saved with the model (derived from the config); grown when positions pass its end.
            self.rope_speeds = dynamic_config.speeds  # float64 on the CPU (MPS has no float64), so not a buffer
            cos, sin = rope_table(self.rope_speeds, config.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.init_weights()

        self.lm_head.weight = self.token_embedding_table.weight  # weight tying

    def init_weights(self):
        nn.init.xavier_uniform_(self.token_embedding_table.weight)
        if not self.use_rope_embeddings:
            nn.init.xavier_uniform_(self.position_embedding_table.weight)
        # Generic default for every Linear in the tree.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(self.lm_head.bias, 0)  # no need to init bias, it's not used in lm_head
        for module in self.modules():
            if module is not self and hasattr(module, "init_weights"):
                module.init_weights()

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        last_only: bool = False,
        use_cache: bool = False,
    ):
        """Plain forward is stateless. With use_cache=True the heads append this call's keys/values to
        their KV cache and attend over everything cached so far; only generation does that."""
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C), or (batch_size, block_size, n_embd)
        offset = self.cache_len() if use_cache else 0
        if not self.use_rope_embeddings:
            pos_emb = self.position_embedding_table(torch.arange(offset, offset + T, device=idx.device))  # (T,C)

        # Log embedding values and gradients
        if self.training:
            self.tok_embedding_values = tok_emb.detach()
            if not self.use_rope_embeddings:
                self.pos_embedding_values = pos_emb.detach()
                assert pos_emb.shape == (
                    T,
                    tok_emb.size(-1),
                ), f"Expected pos_emb shape {(T, tok_emb.size(-1))}, but got {pos_emb.shape}"

        # Add assertions to check tensor shapes
        assert tok_emb.shape == (
            B,
            T,
            tok_emb.size(-1),
        ), f"Expected tok_emb shape {(B, T, tok_emb.size(-1))}, but got {tok_emb.shape}"

        x = tok_emb  # (B, T, C)
        if not self.use_rope_embeddings:
            x = tok_emb + pos_emb
        rope = self.rope_angles(offset, T) if self.use_rope_embeddings else None
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, use_cache, rope)
        if last_only and targets is None:
            x = x[:, -1:]  # generation only needs the next-token distribution
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # pyright: ignore[reportConstantRedefinition]
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int, greedy: bool = False) -> torch.Tensor:
        was_training = self.training
        self.eval()
        self.clear_cache()  # Clear cache before generation
        try:
            return self._generate(idx, max_new_tokens, block_size, greedy)
        finally:
            self.train(was_training)

    def next_token_logits(self, idx: torch.Tensor, block_size: int) -> torch.Tensor:
        """(B, vocab) logits for the token after `idx`, decoding through the KV cache. The single
        place that decides what the cache sees, shared by every generation path so they cannot differ.

        Only the newest token is fed once the cache holds the context. When the window is full:
        a RoPE model run at its full block_size keeps feeding single tokens through the wrapped cache
        (RoPE makes Q·K depend only on relative offsets, so cached K/V stay valid as the window rolls;
        with more than one layer this drifts slightly from a windowed recompute). Every other case clears the cache and re-runs the cropped window,
        which is exact: absolute-PE models (older checkpoints) have no position past block_size, and
        a window smaller than the model's would not match where the cache ring wraps. Callers clear
        the cache before the first call."""
        assert block_size <= self.block_size, f"window {block_size} exceeds the model's block_size {self.block_size}"
        cache_len = self.cache_len()
        rolls = self.use_rope_embeddings and block_size == self.block_size
        if 0 < cache_len < block_size or (rolls and cache_len >= block_size):
            idx_cond = idx[:, -1:]
        else:
            self.clear_cache()  # nothing cached yet, or the window is full and this model can't roll
            idx_cond = idx[:, -block_size:]
        logits, _ = self(idx_cond, last_only=True, use_cache=True)
        return logits[:, -1, :]

    def _generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int, greedy: bool) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self.next_token_logits(idx, block_size)  # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            if greedy:
                # take the argmax
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)
            else:
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def rope_angles(self, start: int, T: int) -> "RopeAngles":
        """The (cos, sin) rows for positions start .. start + T - 1, each (T, head_size / 2). The rolling
        cache's positions keep growing past block_size, so the table doubles when they run past its end."""
        if start + T > self.rope_cos.size(0):
            n = max(2 * self.rope_cos.size(0), start + T)
            cos, sin = rope_table(self.rope_speeds, n)
            self.rope_cos, self.rope_sin = cos.to(self.rope_cos.device), sin.to(self.rope_sin.device)
        return self.rope_cos[start : start + T], self.rope_sin[start : start + T]

    def cache_len(self) -> int:
        # every layer holds the same number of positions
        return self.blocks[0].sa.pos

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()
