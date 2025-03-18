import os
import optuna
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from .data import WikipediaDataset
from .model import ModelCustomTransformer
from .trainer import Trainer
import argparse
import signal
import random
import numpy as np

# Fixed hyperparameters
max_len = 1024
block_size = 128
batch_size = 32
eval_iters = 100
eval_interval = 100

optuna.logging.set_verbosity(optuna.logging.DEBUG)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class HyperparameterOptimizer:
    def __init__(self, dataset, device, tokenizer, rank, world_size, checkpoint_dir="models/optuna_checkpoints", n_trials=50, max_epochs=3, max_iters=1000):
        self.dataset = dataset
        self.device = device
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.checkpoint_dir = checkpoint_dir
        self.n_trials = n_trials
        self.max_epochs = max_epochs
        self.max_iters = max_iters

    def objective(self, trial):
        # Set seed for reproducibility
        seed = 42
        set_seed(seed)

        # Hyperparameters to optimize
        n_embd_n_head_options = [
            (384, 12),
            (512, 8),
            (768, 12),
            (1024, 16)
        ]

        n_embd_n_head = trial.suggest_categorical("n_embd_n_head", n_embd_n_head_options)
        n_embd, n_head = n_embd_n_head
        n_layer = trial.suggest_categorical("n_layer", [4, 8, 12, 16])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        grad_accum_steps = trial.suggest_int("grad_accum_steps", 1, 8)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

        # Broadcast trial parameters to all processes
        trial_params = {
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'grad_accum_steps': grad_accum_steps,
            'weight_decay': weight_decay
        }
        if self.rank == 0:
            trial_params_tensor = torch.tensor([n_embd, n_head, n_layer, dropout, learning_rate, grad_accum_steps, weight_decay], dtype=torch.float32).to(self.device)
        else:
            trial_params_tensor = torch.empty(7, dtype=torch.float32).to(self.device)
        dist.broadcast(trial_params_tensor, src=0)
        trial_params = {
            'n_embd': int(trial_params_tensor[0].item()),
            'n_head': int(trial_params_tensor[1].item()),
            'n_layer': int(trial_params_tensor[2].item()),
            'dropout': trial_params_tensor[3].item(),
            'learning_rate': trial_params_tensor[4].item(),
            'grad_accum_steps': int(trial_params_tensor[5].item()),
            'weight_decay': trial_params_tensor[6].item()
        }

        vocab_size = self.tokenizer.vocab_size

        model = ModelCustomTransformer(vocab_size, trial_params['n_embd'], trial_params['n_head'], trial_params['n_layer'], block_size, trial_params['dropout']).to(self.device)
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=trial_params['learning_rate'], weight_decay=trial_params['weight_decay'])

        # Create a trial-specific checkpoint directory
        trial_checkpoint_dir = os.path.join(self.checkpoint_dir, f"trial_{trial.number}")
        os.makedirs(trial_checkpoint_dir, exist_ok=True)

        trainer = Trainer(
            model=model,
            dataset=self.dataset,
            device=self.device,
            tokenizer=self.tokenizer,
            optimizer=optimizer,
            checkpoint_dir=trial_checkpoint_dir,
            batch_size=batch_size,
            max_epochs=self.max_epochs,
            max_iters=self.max_iters,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            rank=self.rank,
            world_size=self.world_size,
            verbose=False,
            use_tqdm=False,
            grad_accum_steps=trial_params['grad_accum_steps'],
            weight_decay=trial_params['weight_decay']
        )

        trainer.train()

        return trainer.estimate_loss()['val'].item()

    def optimize(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

def distributed_tuning(rank, num_gpus, n_trials=5, num_samples=10000, verbose=False):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Uncomment for debugging
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(num_gpus)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12361'
    dist.init_process_group(backend="nccl")

    def signal_handler(sig, frame):
        print(f"Rank {rank}: Received signal {sig}. Cleaning up...")
        dist.destroy_process_group()
        os._exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    try:
        device = torch.device(f"cuda:{rank}")  # Ensure each rank gets a unique GPU
        print(f"Rank {rank}: Using device {device}")  # Log the CUDA device for each rank
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = WikipediaDataset(tokenizer, 1024, 128, regenerate=False, num_samples=num_samples, verbose=verbose)

        optimizer = HyperparameterOptimizer(
            dataset=dataset,
            device=device,
            tokenizer=tokenizer,
            rank=rank,
            world_size=num_gpus,
            n_trials=n_trials
        )
        print(f"Rank {rank}: Starting hyperparameter optimization")
        optimizer.optimize()
    finally:
        print(f"Rank {rank}: Destroying process group...")
        dist.destroy_process_group()

def main(rank, num_gpus):
    distributed_tuning(rank, num_gpus, n_trials=5, num_samples=10000, verbose=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Hyperparameter Tuning")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for tuning")
    args = parser.parse_args()

    if args.num_gpus > torch.cuda.device_count():
        raise ValueError(f"Requested {args.num_gpus} GPUs, but only {torch.cuda.device_count()} are available.")

    if torch.cuda.is_available():
        print(f"CUDA is available. Starting tuning with {args.num_gpus} GPUs...")
    else:
        print("CUDA is not available. Tuning on CPU...")
        args.num_gpus = 1  # Force to 1 GPU if CUDA is not available

    torch.multiprocessing.spawn(main, args=(args.num_gpus,), nprocs=args.num_gpus, join=True)