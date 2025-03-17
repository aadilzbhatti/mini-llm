import os
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from data import WikipediaDataset
from model import ModelCustomTransformer
from trainer import Trainer

# Fixed hyperparameters
max_len = 1024
block_size = 128
batch_size = 32
eval_iters = 100
eval_interval = 100

optuna.logging.set_verbosity(optuna.logging.INFO)

class HyperparameterOptimizer:
    def __init__(self, dataset, device, tokenizer, checkpoint_dir="./optuna_checkpoints", n_trials=1, max_epochs=3, max_iters=1000):
        self.dataset = dataset
        self.device = device
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir
        self.n_trials = n_trials
        self.max_epochs = max_epochs
        self.max_iters = max_iters

    def objective(self, trial):
        # Hyperparameters to optimize
        n_embd = trial.suggest_categorical("n_embd", [384, 512, 768, 1024])
        valid_n_head_options = {
            384: [4, 8, 12],
            512: [4, 8, 16],
            768: [8, 12, 16],
            1024: [8, 12, 16]
        }
        n_head = trial.suggest_categorical("n_head", valid_n_head_options[n_embd])
        n_layer = trial.suggest_categorical("n_layer", [4, 8, 12, 16])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

        vocab_size = self.tokenizer.vocab_size

        model = ModelCustomTransformer(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
            verbose=False
        )

        trainer.train()

        return trainer.estimate_loss()['val'].item()

    def optimize(self):
        def before_trial(study, trial):
            print(f"Starting trial {trial.number}")
            print("  Params:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=-1, callbacks=[before_trial])

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = WikipediaDataset(tokenizer, 1024, 128, regenerate=False, num_samples=10000)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print("Using device", device)

    optimizer = HyperparameterOptimizer(
        dataset=dataset,
        device=device,
        tokenizer=tokenizer)
    print("Starting hyperparameter optimization")
    optimizer.optimize()