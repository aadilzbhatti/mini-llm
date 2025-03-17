import os
import time
import torch
from tqdm import tqdm

from text_completer import TextCompleter

class Trainer:
    def __init__(self, model, dataset, device, tokenizer, optimizer, checkpoint_dir, batch_size, max_epochs, max_iters, eval_interval, learning_rate):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate

        self.train_dataloader = dataset.get_test_train_dataloaders("train", batch_size)
        self.val_dataloader = dataset.get_test_train_dataloaders("val", batch_size)
        self.text_completer = TextCompleter(self.model, self.tokenizer, self.device)

    def get_batch(self, split):
        if split == 'train':
            next_item = next(iter(self.train_dataloader))
        elif split == 'val':
            next_item = next(iter(self.val_dataloader))
        else:
            raise ValueError(f"Unknown split: {split}")
        
        X, Y = next_item['input_ids'], next_item['labels']
        return X.to(self.device), Y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def save_checkpoint(self, epoch, loss):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: {checkpoint_path}")
        return epoch, loss

    def get_memory_usage(self):
        if torch.cuda.is_available():
            return f"GPU Memory: {torch.cuda.memory_allocated() / 1e6:.2f}MB"
        return "Using CPU"

    def train(self):
        start_epoch = 0
        if os.path.exists(os.path.join(self.checkpoint_dir, "model_epoch_0.pt")):
            start_epoch, _ = self.load_checkpoint(os.path.join(self.checkpoint_dir, "model_epoch_0.pt"))

        for epoch in range(start_epoch, self.max_epochs):
            print(f"Epoch {epoch}/{self.max_epochs}")
            
            self.model.train()
            epoch_loss = 0
            start_time = time.time()

            progress_bar = tqdm(range(self.max_iters), desc=f"Epoch {epoch}", unit="batch")

            for i in progress_bar:
                xb, yb = self.get_batch('train')

                _, loss = self.model(xb, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if i % (self.eval_interval // 10) == 0:
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", mem=self.get_memory_usage())

                if i % self.eval_interval == 0:
                    losses = self.estimate_loss()
                    elapsed_time = time.time() - start_time
                    eta = elapsed_time / (i + 1) * (self.max_iters - i)
                    
                    print(f"\n[Epoch {epoch} | Step {i}] Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f} | ETA: {eta:.2f}s")
                    self.save_checkpoint(epoch, losses['val'])

                    print("Sample Output:")
                    print(self.text_completer.get_text_completions(100, "The capital of France is"))
            
            print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s | Avg Loss: {epoch_loss / self.max_iters:.4f}")

# Example usage:
# trainer = Trainer(model, tokenizer, optimizer, checkpoint_dir, max_epochs, max_iters, eval_interval, learning_rate)
# trainer.train()