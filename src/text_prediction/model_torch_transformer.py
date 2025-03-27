import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class ModelTorchTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # Define TransformerDecoder
        decoder_layer = TransformerDecoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=n_layer)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None, attention_mask=None, writer=None, step=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding(idx)  # (B, T, n_embd)
        position_embeddings = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = token_embeddings + position_embeddings.unsqueeze(0)  # (B, T, n_embd)

        # Process attention_mask to match the expected shape
        if attention_mask is not None:
            # Ensure attention_mask is 2D (T, T) or 3D (B, T, T)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension
            elif attention_mask.dim() == 3 and attention_mask.size(0) != B:
                raise ValueError(f"Expected attention_mask batch size {B}, but got {attention_mask.size(0)}")

        # Use x as memory if no external memory is provided
        memory = x  # (B, T, n_embd)

        # Apply TransformerDecoder
        x = self.decoder(x, memory=memory, tgt_mask=attention_mask)  # (B, T, n_embd)
        x = self.dropout(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Optionally log to TensorBoard
        if writer is not None and step is not None:
            writer.add_scalar("logits_mean", logits.mean().item(), step)

        if targets is None:
            return logits, None

        # Compute loss
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, block_size):
        idx = idx.to(self.token_embedding.weight.device)  # Ensure idx is on the correct device
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond, attention_mask=None)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
