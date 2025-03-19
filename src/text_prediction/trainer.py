import os
import time
import torch
import logging
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from .data import WikipediaDataset
from .data_pipeline import DataPipeline
from .model import ModelCustomTransformer
from .text_completer import TextCompleter
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import argparse
import signal
from contextlib import nullcontext
import json

from .utils import RankFilter
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR  # Add this import

class Trainer:
    def __init__(self, model, dataset, device, tokenizer, optimizer, checkpoint_dir, batch_size, max_epochs, max_iters, eval_iters, eval_interval, learning_rate, rank, world_size, verbose=True, use_tqdm=True, grad_accum_steps=1, enable_profiling=False, weight_decay=0.01, save_checkpoints=True):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_iters = max_iters
        self.eval_iters = eval_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.grad_accum_steps = grad_accum_steps
        self.enable_profiling = enable_profiling
        self.weight_decay = weight_decay
        self.profiler = None
        self.train_losses = []
        self.val_losses = []
        self.metadata_file = os.path.join(checkpoint_dir, "training_metadata.json")
        self.save_checkpoints = save_checkpoints
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)  # Initialize the scheduler

        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.addFilter(RankFilter(rank))
        for handler in self.logger.handlers:
            handler.addFilter(RankFilter(rank))


        self.train_dataloader = dataset.get_dataloader(batch_size, split="train")
        self.val_dataloader = dataset.get_dataloader(batch_size, split="val", shuffle=False)

        underlying_model = self.model.module if isinstance(self.model, DDP) else self.model
        self.text_completer = TextCompleter(underlying_model, self.tokenizer, self.device, block_size=128)

    def log(self, level, msg):
        if self.verbose:
            self.logger.log(level, msg)

    def get_batch(self, split):
        start_time = time.time()
        self.log(logging.DEBUG, f"Starting get_batch for split: {split}")

        if split == 'train':
            next_item = next(iter(self.train_dataloader))
        elif split == 'val':
            next_item = next(iter(self.val_dataloader))
        else:
            raise ValueError(f"Unknown split: {split}")

        self.log(logging.DEBUG, f"DataLoader iteration time: {time.time() - start_time:.2f}s")

        X, Y, attention_mask = next_item #get the attention mask
        self.log(logging.DEBUG, f"Batch loading time: {time.time() - start_time:.2f}s")
        return X.to(self.device), Y.to(self.device), attention_mask.to(self.device) #return the attention mask

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters).to(self.device)
            for k in range(self.eval_iters):
                X, Y, attention_mask = self.get_batch(split)
                _, loss = self.model(X, Y, attention_mask) #pass the attention mask
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save_checkpoint(self, epoch, loss):
        if not self.save_checkpoints:
            return
        if self.rank == 0:  # Save checkpoint only on rank 0
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            self.log(logging.INFO, f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        if not self.save_checkpoints:
            return 0, None
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.log(logging.INFO, f"Checkpoint loaded: {checkpoint_path}")
        return epoch, loss

    def get_latest_checkpoint(self):
        if not self.save_checkpoints:
            return None
        if self.rank == 0:  # Only rank 0 needs to find the latest checkpoint
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("model_epoch_") and f.endswith(".pt")]
            if not checkpoint_files:
                return None
            latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[2].split('.')[0]))
            return os.path.join(self.checkpoint_dir, latest_checkpoint)
        return None

    def get_memory_usage(self):
        if torch.cuda.is_available():
            return f"GPU Memory: {torch.cuda.memory_allocated() / 1e6:.2f}MB"
        return "Using CPU"

    def save_metadata(self, grad_norms):
        metadata = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "grad_norms": grad_norms,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "max_iters": self.max_iters,
            "eval_iters": self.eval_iters,
            "eval_interval": self.eval_interval,
            "learning_rate": self.learning_rate,
            "grad_accum_steps": self.grad_accum_steps,
            "weight_decay": self.weight_decay,
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)

    def train(self):
        start_epoch = 0
        if self.save_checkpoints:
            latest_checkpoint_path = self.get_latest_checkpoint()
            if latest_checkpoint_path:
                self.log(logging.INFO, "Checkpoint found, resuming training...")
                start_epoch, _ = self.load_checkpoint(latest_checkpoint_path)

        if self.enable_profiling:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=tensorboard_trace_handler('./log_dir'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )

        for epoch in range(start_epoch, self.max_epochs):
            self.log(logging.INFO, f"Epoch {epoch}/{self.max_epochs}")

            torch.manual_seed(epoch)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(epoch)

            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        start_time = time.time()

        if self.use_tqdm:
            progress_bar = tqdm(range(self.max_iters), desc=f"Epoch {epoch}", unit="batch")
        else:
            progress_bar = range(self.max_iters)

        current_val_loss = None
        grad_norms = []

        with self.profiler if self.enable_profiling else nullcontext() as prof:
            for i in progress_bar:
                self.log(logging.DEBUG, f"Starting iteration {i} of epoch {epoch}")

                with record_function("get_batch"):
                    xb, yb, attention_mask = self.get_batch('train')

                with record_function("forward_pass"):
                    _, loss = self.model(xb, yb, attention_mask)
                    loss = loss / self.grad_accum_steps
                    loss.backward()

                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    if (i + 1) % self.grad_accum_steps == 0:
                        with record_function("optimizer_step"):
                            self.optimizer.step()
                            self.scheduler.step()  # Step the scheduler
                            self.optimizer.zero_grad(set_to_none=True)

                        epoch_loss += loss.item() * self.grad_accum_steps

                if i % self.eval_interval == 0:
                    self.log(logging.DEBUG, f"Evaluating at iteration {i} of epoch {epoch}")
                    losses = self.estimate_loss()
                    grad_norms.append(grad_norm.item())
                    current_val_loss = losses['val']
                    elapsed_time = time.time() - start_time
                    eta = elapsed_time / (i + 1) * (self.max_iters - i)

                    self.log(logging.INFO, f"\n[Epoch {epoch} | Step {i}] Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f} | ETA: {eta:.2f}s")
                    self.save_checkpoint(epoch, losses['val'])
                    self.train_losses.append(losses['train'])
                    self.val_losses.append(losses['val'])
                    self.save_metadata(grad_norms)  # Save metadata including grad_norms

                if self.use_tqdm and self.verbose and i % (self.eval_interval // 10) == 0: #only set postfix if tqdm is being used.
                    progress_bar.set_postfix(loss=f"{loss.item() * self.grad_accum_steps:.4f}", val_loss=f"{current_val_loss:.4f}" if current_val_loss else "N/A", mem=self.get_memory_usage())

            self.log(logging.INFO, f"Epoch {epoch} completed in {time.time() - start_time:.2f}s | Avg Loss: {epoch_loss / self.max_iters:.4f}")
            self.log(logging.INFO, self.text_completer.get_text_completions(max_tokens=100))

            if self.enable_profiling:
                self.log(logging.INFO, self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        # Save train losses and gradient norms to metadata at the end of the epoch
        self.save_metadata(grad_norms)

def distributed_training(rank, num_gpus, max_epochs=3, max_iters=1000, eval_iters=100, eval_interval=100, num_samples=10000, verbose=False, enable_profiling=False, enable_tqdm=False, save_checkpoints=True):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(num_gpus)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group(backend="nccl")  # Initialize distributed training

    def signal_handler(sig, frame):
        print(f"Rank {rank}: Received signal {sig}. Cleaning up...")
        dist.destroy_process_group()
        os._exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    try:
        device = torch.device(f"cuda:{rank}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dp = DataPipeline(tokenizer, max_len=1024, block_size=128, regenerate=False, num_samples=10000, verbose=True, augment_data=False, parent_path=".")

        vocab_size = tokenizer.vocab_size + 2  # Add 2 for special tokens
        model = ModelCustomTransformer(vocab_size, 768, 6, 6, 128, 0.1).to(device)
        if num_gpus > 1:
            model = DDP(model, device_ids=[rank], output_device=rank)
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

        trainer = Trainer(
            model=model,
            dataset=dp,
            device=device,
            tokenizer=tokenizer,
            optimizer=optimizer,
            checkpoint_dir="models/checkpoints",
            batch_size=64,
            max_epochs=max_epochs,
            max_iters=max_iters,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            learning_rate=1e-4,
            rank=rank,
            world_size=num_gpus,
            verbose=verbose,
            use_tqdm=enable_tqdm,
            grad_accum_steps=4,
            enable_profiling=enable_profiling,
            weight_decay=0.01,
            save_checkpoints=save_checkpoints
        )

        if trainer.verbose:
            trainer.logger.info("Starting training...")
        trainer.train()
    finally:
        if trainer.verbose:
            trainer.logger.info("Destroying process group...")
        dist.destroy_process_group()

def single_thread_train(num_samples, batch_size, max_epochs, max_iters, eval_interval, eval_iters, verbose, enable_profiling, enable_tqdm, save_checkpoints):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dp = DataPipeline(tokenizer, max_len=1024, block_size=128, regenerate=False, num_samples=num_samples, verbose=True, augment_data=False, parent_path=".")
    model = ModelCustomTransformer(tokenizer.vocab_size + 2, 1024, 16, 24, 128, 0.2).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    trainer = Trainer(
        model=model,
        dataset=dp,
        device=device,
        tokenizer=tokenizer,
        optimizer=optimizer,
        checkpoint_dir="models/checkpoints",
        batch_size=batch_size,
        max_epochs=max_epochs,
        max_iters=max_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        learning_rate=1e-3,
        rank=0,
        world_size=0,
        verbose=verbose,
        use_tqdm=enable_tqdm,
        grad_accum_steps=4,
        enable_profiling=enable_profiling,
        weight_decay=0.01,
        save_checkpoints=save_checkpoints
    )

    if trainer.verbose:
        trainer.logger.info("Starting training...")
    trainer.train()

def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for training")
    parser.add_argument("--save_checkpoints", action="store_true", help="Enable saving checkpoints")
    parser.add_argument("--distributed", action="store_true", default=True, help="Enable distributed training")
    args = parser.parse_args()

    if args.num_gpus > torch.cuda.device_count():
        raise ValueError(f"Requested {args.num_gpus} GPUs, but only {torch.cuda.device_count()} are available.")

    if args.distributed:
        if torch.cuda.is_available():
            print(f"CUDA is available. Starting distributed training with {args.num_gpus} GPUs...")
            torch.multiprocessing.spawn(distributed_training, args=(args.num_gpus, args.save_checkpoints), nprocs=args.num_gpus, join=True)
        else:
            print("CUDA is not available. Starting single-threaded training...")
            single_thread_train(save_checkpoints=args.save_checkpoints)
    else:
        print("Starting single-threaded training...")
        single_thread_train(save_checkpoints=args.save_checkpoints)

if __name__ == "__main__":
    main()