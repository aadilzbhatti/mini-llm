"""Custom decoder-only Transformer.

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


class Head(nn.Module):
    """one head of self-attention"""

    tril: torch.Tensor

    def __init__(self, config: ModelConfig, dynamic_config: DynamicModelConfig):
        super().__init__()
        head_size = dynamic_config.head_size
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.scale = head_size**-0.5
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

        # Allocated once at (B, block_size, head_size) and written in place; pos % block_size says how
        # many positions are filled. Not saved with the model.
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

        # absolute position of next token, only ever grows
        self.pos = 0

        self.head_size = head_size
        self.block_size = config.block_size
        self.use_rope_embeddings = config.use_rope_embeddings
        if self.use_rope_embeddings:
            self.register_buffer("speeds", dynamic_config.speeds, persistent=False)

        # Initialize linear layers
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.key.weight)
        nn.init.xavier_normal_(self.query.weight)
        nn.init.xavier_normal_(self.value.weight)

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        B, T, C = x.shape  # pyright: ignore[reportUnusedVariable]
        k_new = self.key(x)
        v_new = self.value(x)
        q = self.query(x)

        start = self.pos if use_cache else 0  # uncached passes (training, loss evals) always start at 0
        if self.use_rope_embeddings:
            k_new, q = self.rotate(k_new, start), self.rotate(q, start)

        if use_cache:
            if self.k_cache is None or self.k_cache.size(0) != B or self.k_cache.dtype != k_new.dtype:
                # cache is uninitialized or batch size changed; allocate it
                self.k_cache = k_new.new_empty(B, self.tril.size(0), k_new.size(-1))
                self.v_cache = torch.empty_like(self.k_cache)
            start, end = self.pos % self.block_size, (self.pos % self.block_size) + T
            assert end <= self.block_size, f"Cache overflow: pos={self.pos}, T={T}, block_size={self.block_size}"
            # once the buffer has wrapped, slots are out of position order, so only single tokens may be written
            assert (
                T == 1 or self.pos < self.block_size
            ), f"Multi-token write after the cache wrapped: pos={self.pos}, T={T}, block_size={self.block_size}"
            self.k_cache[:, start:end] = k_new
            self.v_cache[:, start:end] = v_new
            self.pos += T

            filled = min(self.pos, self.block_size)
            k = self.k_cache[:, :filled]
            v = self.v_cache[:, :filled]
        else:
            k = k_new
            v = v_new

        T_total = k.shape[1]  # k.shape != q.shape if we are using the cache
        wei = (q @ k.transpose(-2, -1)) * self.scale
        if T > 1:  # a single new token attends to the whole cache: its mask row is all ones
            wei = wei.masked_fill(self.tril[T_total - T : T_total, :T_total] == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Log wei values
        if self.training:
            self.attention_values = wei.detach()

        out = wei @ v

        return out

    def clear_cache(self):
        self.pos = 0  # keep the buffers: the next generation reuses them

    def rotate(self, x: torch.Tensor, start: int) -> torch.Tensor:
        T = x.shape[1]
        assert isinstance(self.speeds, torch.Tensor)
        positions = torch.arange(start, start + T, device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, self.speeds)
        cosines = torch.cos(angles)
        sines = torch.sin(angles)
        ret = torch.empty_like(x)
        # for each row, rotate each pair
        ret[..., 0::2] = x[..., 0::2] * cosines - x[..., 1::2] * sines
        ret[..., 1::2] = x[..., 0::2] * sines + x[..., 1::2] * cosines
        return ret


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, config: ModelConfig, dynamic_config: DynamicModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, dynamic_config) for _ in range(config.n_head)])
        self.proj = nn.Linear(dynamic_config.head_size * config.n_head, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        # FIX (#2): no mask to pass along any more -- see Head.forward.
        out = torch.cat([h(x, use_cache) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    def clear_cache(self):
        for m in self.modules():
            if isinstance(m, Head):
                m.clear_cache()


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

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        x = x + self.sa(self.ln1(x), use_cache)
        x = x + self.ffwd(self.ln2(x))
        return x

    def clear_cache(self):
        self.sa.clear_cache()


class ModelCustomTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_rope_embeddings = config.use_rope_embeddings
        dynamic_config = DynamicModelConfig.from_config(config)  # derived values, not saved in checkpoints

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        if not self.use_rope_embeddings:
            self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, dynamic_config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

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
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, use_cache)
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
        RoPE keeps feeding single tokens through the wrapped cache (positions are relative, so the
        window rolls; with more than one layer this drifts slightly from a windowed recompute).
        Absolute-PE models (older checkpoints) have no position past block_size, so they clear the
        cache and re-run the cropped window, which is exact. Callers clear the cache before the
        first call."""
        cache_len = self.cache_len()
        if 0 < cache_len < block_size or (self.use_rope_embeddings and cache_len >= block_size):
            idx_cond = idx[:, -1:]
        else:
            self.clear_cache()  # nothing cached yet, or the window is full for absolute PE
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

    def cache_len(self) -> int:
        # every head in every layer holds the same number of positions
        return self.blocks[0].sa.heads[0].pos

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()
