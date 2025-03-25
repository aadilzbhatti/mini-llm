import argparse
import json
import logging
import os
import signal
import time
from contextlib import nullcontext
import hashlib

from numpy import isin
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from text_prediction.streaming_data_pipeline import StreamingDataPipeline
from text_prediction.data_pipeline import DataPipeline
from text_prediction.model import FeedForward, Head, ModelCustomTransformer
from text_prediction.text_completer import TextCompleter
from text_prediction.utils import RankFilter

if torch.cuda.is_available():
    from torch.amp import GradScaler, autocast
else:
    GradScaler = None
    autocast = nullcontext

from torch.utils.data import DistributedSampler

def get_checkpoint_dir(hyperparams, checkpoint_dir):
    hyperparams_str = json.dumps(hyperparams, sort_keys=True)
    hyperparams_hash = hashlib.md5(hyperparams_str.encode()).hexdigest()
    checkpoint_dir = os.path.join(checkpoint_dir, hyperparams_hash)
    return checkpoint_dir

class Trainer:
    def __init__(self, model, dataset, device, tokenizer, optimizer, rank, world_size, hyperparams, training_params):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.training_params = training_params
        self.checkpoint_dir = get_checkpoint_dir(hyperparams, training_params["checkpoint_dir"])
        self.rank = rank
        self.world_size = world_size
        self.hyperparams = hyperparams
        self.grad_accum_steps = hyperparams["grad_accum_steps"]
        self.grad_norm_clip_value = hyperparams["grad_norm_clip_value"]

        self.batch_size = training_params["batch_size"]
        self.num_samples = training_params["num_samples"]

        self.max_epochs = training_params["max_epochs"]
        self.max_iters = training_params["max_iters"]
        self.eval_iters = training_params["eval_iters"]
        self.eval_interval = training_params["eval_interval"]
        self.verbose = training_params["verbose"]
        self.use_tqdm = training_params["use_tqdm"]
        self.save_checkpoints = training_params["save_checkpoints"]

        self.early_stopping_patience = training_params.get("early_stopping_patience", 5)
        self.early_stopping_min_delta = training_params.get("early_stopping_min_delta", 0.001)
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

        self.profiler = None
        self.train_losses = []
        self.val_losses = []
        self.grad_norms = []
        self.metadata_file = os.path.join(self.checkpoint_dir, "training_metadata.json")
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(0.1 * self.max_iters),  # 10% of total iterations for warmup
            num_training_steps=self.max_iters
        )
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        logging.basicConfig(level=logging.DEBUG if self.verbose else logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.addFilter(RankFilter(rank))
        for handler in self.logger.handlers:
            handler.addFilter(RankFilter(rank))

        if isinstance(dataset, DataPipeline):
            self.train_sampler = DistributedSampler(dataset.get_dataset(split="train"), num_replicas=world_size, rank=rank) if world_size > 1 else None
            self.val_sampler = DistributedSampler(dataset.get_dataset(split="val"), num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
            self.train_dataloader = dataset.get_dataloader(self.training_params["batch_size"], split="train", sampler=self.train_sampler)
            self.val_dataloader = dataset.get_dataloader(self.training_params["batch_size"], split="val", sampler=self.val_sampler)
        elif isinstance(dataset, StreamingDataPipeline):
            self.train_dataloader_generator = dataset.get_dataloader_generator(self.training_params["batch_size"], split="train")
            self.val_dataloader_generator = dataset.get_dataloader_generator(self.training_params["batch_size"], split="val", shuffle=False)

        underlying_model = self.model.module if isinstance(self.model, DDP) else self.model
        self.text_completer = TextCompleter(underlying_model, self.tokenizer, self.device, block_size=self.hyperparams["block_size"])

        # Initialize TensorBoard writer
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, 'tensorboard'))
        else:
            self.writer = None

    def log(self, level, msg):
        if self.verbose:
            self.logger.log(level, msg)

    def get_batch(self, split):
        start_time = time.time()
        self.log(logging.DEBUG, f"Starting get_batch for split: {split}")

        if split == 'train':
            if isinstance(self.dataset, DataPipeline):
                next_item = next(iter(self.train_dataloader))
            else:
                if not hasattr(self, 'train_batch_generator'):
                    self.train_batch_generator = self.example_from_dataloader_generator(self.train_dataloader_generator)
                next_item = next(self.train_batch_generator)
        elif split == 'val':
            if isinstance(self.dataset, DataPipeline):
                next_item = next(iter(self.val_dataloader))
            else:
                if not hasattr(self, 'val_batch_generator'):
                    self.val_batch_generator = self.example_from_dataloader_generator(self.val_dataloader_generator)
                next_item = next(self.val_batch_generator)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.log(logging.DEBUG, f"DataLoader iteration time: {time.time() - start_time:.2f}s")

        X, Y, attention_mask = next_item
        self.log(logging.DEBUG, f"Batch loading time: {time.time() - start_time:.2f}s")
        return X.to(self.device), Y.to(self.device), attention_mask.to(self.device)

    def example_from_dataloader_generator(self, dataloader_generator):
        for dataloader in dataloader_generator:
            for X, Y, attention_mask in dataloader:
                yield X, Y, attention_mask

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters).to(self.device)
            for k in range(self.eval_iters):
                X, Y, attention_mask = self.get_batch(split)
                _, loss = self.model(X, Y, attention_mask)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save_checkpoint(self, epoch, loss):
        if not self.save_checkpoints:
            return
        if self.rank == 0:
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
        if self.rank == 0:
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

    def save_metadata(self):
        metadata = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "grad_norms": self.grad_norms,
            "hyperparams": self.hyperparams,
            "training_params": self.training_params,
            "start_time": self.start_time
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

    def train(self):
        start_epoch = 0
        if self.save_checkpoints:
            latest_checkpoint_path = self.get_latest_checkpoint()
            if latest_checkpoint_path:
                self.log(logging.INFO, "Checkpoint found, resuming training...")
                start_epoch, _ = self.load_checkpoint(latest_checkpoint_path)

        for epoch in range(start_epoch, self.max_epochs):
            self.log(logging.INFO, f"Epoch {epoch}/{self.max_epochs}")

            torch.manual_seed(epoch)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(epoch)

            if isinstance(self.dataset, DataPipeline):
                self.train_sampler.set_epoch(epoch)

            self.train_one_epoch(epoch)

            # Early stopping check
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.log(logging.INFO, "Early stopping triggered. Stopping training.")
                break

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        start_time = time.time()

        if self.use_tqdm:
            progress_bar = tqdm(range(self.max_iters), desc=f"Epoch {epoch}", unit="batch")
        else:
            progress_bar = range(self.max_iters)

        current_val_loss = None

        for i in progress_bar:
            self.log(logging.DEBUG, f"Starting iteration {i} of epoch {epoch}")
            xb, yb, attention_mask = self.get_batch('train')
            
            with autocast(device_type='cuda') if torch.cuda.is_available() else nullcontext():
                _, loss = self.model(xb, yb, attention_mask, self.writer, i)
                loss = loss / self.grad_accum_steps
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip_value)
                self.grad_norms.append(grad_norm.item())
                self.writer.add_scalar("grad_norm", grad_norm.item(), i)
                
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                epoch_loss += loss.item() * self.grad_accum_steps

            if i % self.eval_interval == 0:
                self.log(logging.DEBUG, f"Evaluating at iteration {i} of epoch {epoch}")
                losses = self.estimate_loss()
                current_val_loss = losses['val']
                elapsed_time = time.time() - start_time
                eta = elapsed_time / (i + 1) * (self.max_iters - i)
                self.log(logging.INFO, f"\n[Epoch {epoch} | Step {i}] Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f} | ETA: {eta:.2f}s")
                self.save_checkpoint(epoch, losses['val'])
                self.train_losses.append(losses['train'])
                self.val_losses.append(losses['val'])
                self.save_metadata()

                # profiling information
                self.writer.add_scalar('Loss/train', losses['train'], epoch * self.max_iters + i)
                self.writer.add_scalar('Loss/val', losses['val'], epoch * self.max_iters + i)
                self.writer.add_scalar("learning_rate", self.scheduler.get_last_lr()[0], i)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'gradients/{name}', param.grad, epoch * self.max_iters + i)
                        self.writer.add_scalar(f"gradients/{name}_max", param.grad.max().item(), i)
                        self.writer.add_scalar(f"gradients/{name}_min", param.grad.min().item(), i)
                        self.writer.add_scalar(f"gradients/{name}_mean", param.grad.mean().item(), i)
                    self.writer.add_histogram(f'weights/{name}', param, epoch * self.max_iters + i)

                # Log activations
                for name, module in self.model.named_modules():
                    if isinstance(module, FeedForward) and module.gelu_activation is not None:
                        activation = module.gelu_activation
                        self.writer.add_histogram(f"activations/{name}", activation, i)
                        self.writer.add_scalar(f"activations_mean/{name}", activation.mean().item(), i)
                        self.writer.add_scalar(f"activations_std/{name}", activation.std().item(), i)
                    if isinstance(module, ModelCustomTransformer):
                        if hasattr(module, 'tok_embedding_values'):
                            self.writer.add_histogram("tok_embedding_values", module.tok_embedding_values, i)
                        if hasattr(module, 'pos_embedding_values'):
                            self.writer.add_histogram("pos_embedding_values", module.pos_embedding_values, i)
                    if isinstance(module, Head):
                        if hasattr(module, 'attention_values'):
                            self.writer.add_histogram("attention_values", module.attention_values, i)
                            self.writer.add_scalar("attention_mean", module.attention_values.mean().item(), i)
                            self.writer.add_scalar("attention_std", module.attention_values.std().item(), i)

                # Early stopping logic
                if current_val_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = current_val_loss
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
            if self.use_tqdm and self.verbose and i % (self.eval_interval // 10) == 0:
                progress_bar.set_postfix(loss=f"{loss.item() * self.grad_accum_steps:.4f}", val_loss=f"{current_val_loss:.4f}" if current_val_loss else "N/A", mem=self.get_memory_usage())

        self.log(logging.INFO, f"Epoch {epoch} completed in {time.time() - start_time:.2f}s | Avg Loss: {epoch_loss / self.max_iters:.4f}")
        self.log(logging.INFO, self.text_completer.get_text_completions(max_tokens=100, context="<ARTICLE_START>"))

        self.save_metadata()
        self.writer.close()

def initialize_training_params(num_samples, batch_size, max_epochs, max_iters, eval_iters, eval_interval, verbose, enable_tqdm, save_checkpoints, early_stopping_patience=5, early_stopping_min_delta=0.001, grad_norm_clip_value=1.0):
    hyperparams = {
        "max_len": 1024,
        "block_size": 256,
        "vocab_size": AutoTokenizer.from_pretrained("gpt2").vocab_size + 2,
        "hidden_size": 768,
        "num_layers": 8,
        "num_heads": 12,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0001,
        "grad_accum_steps": 4,
        "grad_norm_clip_value": grad_norm_clip_value 
    }

    training_params = {
        "max_epochs": max_epochs,
        "max_iters": max_iters,
        "eval_iters": eval_iters,
        "eval_interval": eval_interval,
        "verbose": verbose,
        "use_tqdm": enable_tqdm,
        "save_checkpoints": save_checkpoints,
        "checkpoint_dir": "models/wiki-llm/checkpoints",
        "num_samples": num_samples,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "batch_size": batch_size,
    }

    return hyperparams, training_params

def initialize_trainer(rank, num_gpus, device, tokenizer, hyperparams, training_params):
    if training_params["num_samples"] is not None:
        dp = DataPipeline(tokenizer, max_len=hyperparams["max_len"], block_size=hyperparams["block_size"], regenerate=False, num_samples=training_params["num_samples"], verbose=True, augment_data=False, parent_path=".")
    else:
        dp = StreamingDataPipeline(tokenizer, max_len=hyperparams["max_len"], block_size=hyperparams["block_size"], verbose=True, augment_data=False, parent_path=".")
    
    model = ModelCustomTransformer(hyperparams["vocab_size"], hyperparams["hidden_size"], hyperparams["num_heads"], hyperparams["num_layers"], hyperparams["block_size"], hyperparams["dropout"]).to(device)
    if num_gpus > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])

    trainer = Trainer(
        model=model,
        dataset=dp,
        device=device,
        tokenizer=tokenizer,
        optimizer=optimizer,
        rank=rank,
        world_size=num_gpus,
        hyperparams=hyperparams,
        training_params=training_params
    )

    return trainer

def distributed_training(rank, num_gpus, batch_size, max_epochs=3, max_iters=1000, eval_iters=100, eval_interval=100, num_samples=None, verbose=False, enable_tqdm=False, save_checkpoints=True, early_stopping_patience=5, early_stopping_min_delta=0.001, grad_norm_clip_value=1.0):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(num_gpus)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group(backend="nccl")

    def signal_handler(sig, frame):
        print(f"Rank {rank}: Received signal {sig}. Cleaning up...")
        dist.destroy_process_group()
        os._exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    hyperparams, training_params = initialize_training_params(num_samples, batch_size, max_epochs, max_iters, eval_iters, eval_interval, verbose, enable_tqdm, save_checkpoints, early_stopping_patience, early_stopping_min_delta, grad_norm_clip_value)
    trainer = initialize_trainer(rank, num_gpus, device, tokenizer, hyperparams, training_params)

    try:
        if trainer.verbose:
            trainer.logger.info("Starting training...")
        trainer.train()
    finally:
        if trainer.verbose:
            trainer.logger.info("Destroying process group...")
        dist.destroy_process_group()

def single_thread_train(num_samples, batch_size, max_epochs, max_iters, eval_interval, eval_iters, verbose, enable_tqdm, save_checkpoints, early_stopping_patience=5, early_stopping_min_delta=0.001, grad_norm_clip_value=1.0):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    hyperparams, training_params = initialize_training_params(num_samples, batch_size, max_epochs, max_iters, eval_iters, eval_interval, verbose, enable_tqdm, save_checkpoints, early_stopping_patience, early_stopping_min_delta)
    trainer = initialize_trainer(0, 1, device, tokenizer, hyperparams, training_params)

    if trainer.verbose:
        trainer.logger.info("Starting training...")
    trainer.train()

def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for training")
    parser.add_argument("--save_checkpoints", action="store_true", help="Enable saving checkpoints")
    parser.add_argument("--distributed", action="store_true", default=True, help="Enable distributed training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, help="Minimum change in the monitored quantity to qualify as an improvement")
    args = parser.parse_args()

    if args.num_gpus > torch.cuda.device_count():
        raise ValueError(f"Requested {args.num_gpus} GPUs, but only {torch.cuda.device_count()} are available.")

    if args.distributed:
        if torch.cuda.is_available():
            print(f"CUDA is available. Starting distributed training with {args.num_gpus} GPUs...")
            torch.multiprocessing.spawn(distributed_training, args=(args.num_gpus, args.batch_size, args.max_epochs, args.max_iters, args.eval_interval, args.eval_iters, args.verbose, args.enable_profiling, args.enable_tqdm, args.save_checkpoints, args.early_stopping_patience, args.early_stopping_min_delta), nprocs=args.num_gpus, join=True)
        else:
            print("CUDA is not available. Starting single-threaded training...")
            single_thread_train(args.num_samples, args.batch_size, args.max_epochs, args.max_iters, args.eval_interval, args.eval_iters, args.verbose, args.enable_profiling, args.enable_tqdm, args.save_checkpoints, args.early_stopping_patience, args.early_stopping_min_delta)
    else:
        print("Starting single-threaded training...")
        single_thread_train(args.num_samples, args.batch_size, args.max_epochs, args.max_iters, args.eval_interval, args.eval_iters, args.verbose, args.enable_profiling, args.enable_tqdm, args.save_checkpoints, args.early_stopping_patience, args.early_stopping_min_delta)

if __name__ == "__main__":
    main()