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

class Head(nn.Module):
    """ one head of self-attention """

    tril: torch.Tensor

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        # FIX (#1): the per-head LayerNorm that used to live here is gone.
        # Block.forward already normalizes with ln1 before calling the
        # attention sublayer, so every head was re-normalizing an
        # already-normalized input -- and with its own parameters, so an
        # n_head=4 model carried 4 redundant LayerNorms per block. Pre-norm
        # means one norm per sublayer, shared by all heads.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

        # Initialize linear layers
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.key.weight)
        nn.init.xavier_normal_(self.query.weight)
        nn.init.xavier_normal_(self.value.weight)

    # FIX (#2): the `mask` parameter is gone. It was threaded through Block ->
    # MultiHeadAttention -> Head and then never applied (the masked_fill that
    # would have used it was commented out), so it read as if padding were
    # being handled when it wasn't. Batches here are fixed-length crops with
    # no padding, so the causal tril below is the only mask needed. If padded
    # batches come back later, add a key-padding mask then -- deliberately,
    # and with a test.
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape  # pyright: ignore[reportUnusedVariable]
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32, device=k.device))
        # wei = q @ k.transpose(-2, -1)  / torch.sqrt(torch.tensor(C, dtype=torch.float32) + 1e-6)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        # wei = wei * self.tril[:T, :T]
        wei = self.dropout(wei)
        # Log wei values 
        if self.training:
            self.attention_values = wei.detach()
            
        v = self.value(x)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd: int, num_heads: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor):
        # FIX (#2): no mask to pass along any more -- see Head.forward.
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
        self.gelu_activation = x.detach() # store the activation, detaching to avoid gradient issues.
        x = self.net[2](x)
        x = self.net[3](x)
        return x
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor):
        # FIX (#2): mask parameter dropped. ln1/ln2 stay exactly where they
        # were: this is the only LayerNorm on the attention path now that the
        # per-head one is gone (#1).
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class ModelCustomTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int, dropout: float = 0.2):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # FIX (#4): nn.ModuleList, not nn.Sequential. Sequential advertises
        # that you can call self.blocks(x), and the loop in forward is what
        # actually runs. ModuleList says "a list of submodules, called by
        # hand" -- which is the truth. Parameter names are unchanged
        # ("blocks.0.sa..."), so old checkpoints still load.
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)
        # FIX (#7): self.step was set and never read. Step counting belongs to
        # the training loop, not the model.

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.token_embedding_table.weight)
        nn.init.xavier_uniform_(self.position_embedding_table.weight)
        # Generic default for every Linear in the tree.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(self.lm_head.bias, 0)
        # FIX (#3): re-run the submodules' own init AFTER the generic pass.
        # Head.init_weights sets q/k/v with xavier_normal_, but the loop above
        # used to run last and silently overwrote all three with
        # xavier_uniform_, so the normal init never reached the weights. Now
        # the specific choice wins over the generic default, which is what the
        # per-module init_weights methods were written to express.
        for module in self.modules():
            if module is not self and hasattr(module, "init_weights"):
                module.init_weights()

    # FIX (#2): attention_mask parameter dropped -- nothing ever applied it.
    # FIX (#6): inputs are no longer silently moved onto the weights' device.
    # The old `idx = idx.to(...)` made a device mismatch invisible (and copied
    # every batch, every step). Callers move the batch; a mismatch now raises.
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C), or (batch_size, block_size, vocab_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)

        # Log embedding values and gradients
        if self.training:
            self.tok_embedding_values = tok_emb.detach()
            self.pos_embedding_values = pos_emb.detach()

        # Add assertions to check tensor shapes
        assert tok_emb.shape == (B, T, tok_emb.size(-1)), f"Expected tok_emb shape {(B, T, tok_emb.size(-1))}, but got {tok_emb.shape}"
        assert pos_emb.shape == (T, tok_emb.size(-1)), f"Expected pos_emb shape {(T, tok_emb.size(-1))}, but got {pos_emb.shape}"

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # NOTE (bootstrap): a loop that normalized each Head's stored attention
        # matrix and wrote it to TensorBoard as an image used to run here on
        # every forward pass. Removed with the rest of the TensorBoard
        # infrastructure. See BOOTSTRAP_NOTES.md.

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # pyright: ignore[reportConstantRedefinition]
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int, greedy: bool = False) -> torch.Tensor:
        # FIX (#8): put the model in eval mode here (dropout off -- sampling
        # through dropout is not what inference should do) and restore the
        # previous mode on the way out. generate_text() used to call .eval()
        # and never switch back, so generating mid-training silently left the
        # model in eval mode for the rest of the run.
        # FIX (#6): idx is no longer moved onto the weights' device for you;
        # the caller owns placement, as in forward().
        was_training = self.training
        self.eval()
        try:
            return self._generate(idx, max_new_tokens, block_size, greedy)
        finally:
            self.train(was_training)

    def _generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int, greedy: bool) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            if greedy:
                # take the argmax
                idx_next = torch.argmax(probs, dim=-1, keepdim=True) # (B, 1)
            else:
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
