import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from data import WikipediaDataset

# Define the model hyperparameters
n_embd = 768
n_head = 12
n_layer = 12
max_len = 1024
dropout = 0.2
block_size = 128
batch_size = 32
eval_iters = 100
max_iters = 10000
max_epochs = 10
eval_interval = 100
learning_rate = 1e-4
checkpoint_dir = "./checkpoints"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size

dataset = WikipediaDataset(tokenizer, max_len, block_size,  regenerate=False, num_samples=1000)
train_dataloader = dataset.get_test_train_dataloaders("train", batch_size)
val_dataloader = dataset.get_test_train_dataloaders("val", batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print("Using device:", device)

def get_batch(split):
    if split == 'train':
        next_item = next(iter(train_dataloader))
    elif split == 'val':
        next_item = next(iter(val_dataloader))
    else:
        raise ValueError(f"Unknown split: {split}")
    
    X, Y = next_item['input_ids'], next_item['labels']
    return X.to(device), Y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class WikiCompleteModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.2):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C), or (batch_size, block_size, vocab_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # print(logits.shape)
            logits = logits.view(B * T, C)
            # print(logits.shape)
            # print(targets.shape)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {checkpoint_path}")
    return epoch, loss

def get_completion(tokenizer, model, max_tokens):
    model.eval()
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context = tokenizer.encode("Pakistan", return_tensors="pt").to(device)
    out = model.generate(context, max_tokens)
    # print(out)
    decoded_text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    return decoded_text

def get_memory_usage():
    if torch.cuda.is_available():
        return f"GPU Memory: {torch.cuda.memory_allocated() / 1e6:.2f}MB"
    return "Using CPU"

# torch.manual_seed(1337)
model = WikiCompleteModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_epoch = 0
if os.path.exists(os.path.join(checkpoint_dir, "model_epoch_0.pt")):
    start_epoch, _ = load_checkpoint(model, optimizer, os.path.join(checkpoint_dir, "model_epoch_0.pt"))

for epoch in range(start_epoch, max_epochs):
    print(f"Epoch {epoch}/{max_epochs}")
    
    model.train()
    epoch_loss = 0
    start_time = time.time()

    # Use tqdm for progress tracking
    progress_bar = tqdm(range(max_iters), desc=f"Epoch {epoch}", unit="batch")

    for i in progress_bar:
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Accumulate loss for better monitoring
        epoch_loss += loss.item()

        # Show active feedback every few steps
        if i % (eval_interval // 10) == 0:  # More frequent updates
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", mem=get_memory_usage())

        # Periodic evaluation
        if i % eval_interval == 0:
            losses = estimate_loss(model)
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (i + 1) * (max_iters - i)  # Estimated time remaining
            
            print(f"\n[Epoch {epoch} | Step {i}] Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f} | ETA: {eta:.2f}s")
            save_checkpoint(model, optimizer, epoch, losses['val'], checkpoint_dir)

            # Generate a short sample to see model quality
            print("Sample Output:")
            print(get_completion(tokenizer, model, 100))  # Shorter text sample for quick feedback
    
    print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s | Avg Loss: {epoch_loss / max_iters:.4f}")


print(get_completion(tokenizer, model, max_len))