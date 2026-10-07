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

    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.scale = head_size**-0.5
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

        # Allocated once at (B, block_size, head_size) and written in place; cache_len says how
        # many positions are filled. Not saved with the model.
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.cache_len = 0

        self.head_size = head_size
        self.use_rope_embeddings = config.use_rope_embeddings
        if self.use_rope_embeddings:
            self.block_size = config.block_size
            assert config.dynamic_model_config is not None
            self.register_buffer("cosine", config.dynamic_model_config.cosine, persistent=False)
            self.register_buffer("sine", config.dynamic_model_config.sine, persistent=False)

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

        start = self.cache_len if use_cache else 0
        if self.use_rope_embeddings:
            k_new, q = self.rotate(k_new, start), self.rotate(q, start)

        if use_cache:
            if self.k_cache is None or self.k_cache.size(0) != B or self.k_cache.dtype != k_new.dtype:
                # cache is uninitialized or batch size changed; allocate it
                self.k_cache = k_new.new_empty(B, self.tril.size(0), k_new.size(-1))
                self.v_cache = torch.empty_like(self.k_cache)
            start, self.cache_len = self.cache_len, self.cache_len + T
            self.k_cache[:, start : self.cache_len] = k_new
            self.v_cache[:, start : self.cache_len] = v_new
            k = self.k_cache[:, : self.cache_len]
            v = self.v_cache[:, : self.cache_len]
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
        self.cache_len = 0  # keep the buffers: the next generation reuses them

    def rotate(self, x: torch.Tensor, start: int) -> torch.Tensor:
        T = x.shape[1]
        assert isinstance(self.cosine, torch.Tensor)
        assert isinstance(self.sine, torch.Tensor)
        ret = torch.empty_like(x)
        # for each row, rotate each pair
        cosines = self.cosine[start : start + T]
        sines = self.sine[start : start + T]
        ret[..., 0::2] = x[..., 0::2] * cosines - x[..., 1::2] * sines
        ret[..., 1::2] = x[..., 0::2] * sines + x[..., 1::2] * cosines
        return ret


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(head_size * config.n_head, config.n_embd)
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

    def __init__(self, config: ModelConfig):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
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
        head_size = config.n_embd // config.n_head
        pair_indices = torch.arange(0, head_size // 2, 1)
        speeds = 10000 ** (-2 * pair_indices / head_size)
        angles = torch.outer(torch.arange(0, config.block_size, 1, dtype=torch.float32), speeds)

        config.dynamic_model_config = DynamicModelConfig(
            cosine=torch.cos(angles),
            sine=torch.sin(angles),
        )

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        if not self.use_rope_embeddings:
            self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
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
            x += pos_emb
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

    def _generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int, greedy: bool) -> torch.Tensor:
        for _ in range(max_new_tokens):
            if 0 < self.cache_len() < block_size:
                idx_cond = idx[:, -1:]
            else:
                self.clear_cache()  # nothing cached yet, or the window is full: re-run the cropped window
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond, last_only=True, use_cache=True)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
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
        return self.blocks[0].sa.heads[0].cache_len

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()
