import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
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

    def forward(self, x, mask):
        B, T, C = x.shape
        x = self.ln(x)
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32, device=k.device))
        # wei = q @ k.transpose(-2, -1)  / torch.sqrt(torch.tensor(C, dtype=torch.float32) + 1e-6)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # if mask is not None:
        #     wei = wei.masked_fill(mask[:, :T, :T] == 0, float('-inf'))

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

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x, mask): #add mask
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1) #pass the mask
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.gelu_activation = None  # Store GELU activation here

    def forward(self, x):
        x = self.net[0](x)
        x = self.net[1](x)  # GELU activation
        self.gelu_activation = x.detach() # store the activation, detaching to avoid gradient issues.
        x = self.net[2](x)
        x = self.net[3](x)
        return x
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class ModelCustomTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.2):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.step = 0

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.token_embedding_table.weight)
        nn.init.xavier_uniform_(self.position_embedding_table.weight)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(self.lm_head.bias, 0)

    def forward(self, idx, targets=None, attention_mask=None, writer=None, step=None):
        idx = idx.to(self.token_embedding_table.weight.device)
        if targets is not None:
            targets = targets.to(self.token_embedding_table.weight.device)
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
            x = block(x, attention_mask) #pass the attention mask
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        for name, module in self.named_modules():
            if isinstance(module, Head) and hasattr(module, "attention_values"):
                attention_matrix = module.attention_values
                B, T, _ = attention_matrix.shape

                # Normalize attention matrix to [0, 1]
                attention_image = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min() + 1e-8)

                # Log each attention matrix in the batch as a separate image
                # for b in range(B):
                # Reshape to (C, H, W)
                attention_image_single = attention_image[0].view(1, T, T)

                # Log image to TensorBoard
                if writer is not None and step is not None:
                    writer.add_image(f"attention_patterns/{name}/batch_{0}", attention_image_single, step)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, block_size):
        idx = idx.to(self.token_embedding_table.weight.device)  # Ensure idx is on the correct device
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond, attention_mask=None)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
