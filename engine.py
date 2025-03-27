import argparse
import torch
import json
from text_prediction.model_custom_transformer import ModelCustomTransformer
from text_prediction.trainer import distributed_training, single_thread_train
from text_prediction.hyperparam_tuning import HyperparameterOptimizer, distributed_tuning
from text_prediction.text_completer import TextCompleter
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Train, tune hyperparameters, or generate text predictions for the WikiCompleteModel.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'tune', 'inference'], help="Mode to run: 'train', 'tune', or 'inference'")
    parser.add_argument('--regenerate_dataset', action='store_true', help="Regenerate the dataset")
    parser.add_argument('--num_samples', type=int, help="Number of samples for the dataset (default: 10000)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--grad_norm_clip_value', type=float, default=1.0, help="Gradient norm clip value (default: 1.0)")
    parser.add_argument('--max_epochs', type=int, default=10, help="Maximum number of epochs (default: 10)")
    parser.add_argument('--max_iters', type=int, default=1000, help="Maximum number of iterations (default: 1000)")
    parser.add_argument('--eval_interval', type=int, default=100, help="Evaluation interval (default: 100)")
    parser.add_argument('--eval_iters', type=int, default=100, help="Evaluation iterations (default: 100)")
    parser.add_argument('--distributed', action='store_true', help="Use distributed training or tuning")
    parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose logging")
    parser.add_argument('--n_trials', type=int, help="Number of trials for hyperparameter tuning (required if mode is 'tune')")
    parser.add_argument('--enable_tqdm', action='store_true', default=False, help="Enable tqdm progress bar")
    parser.add_argument('--max_new_tokens', type=int, help="Maximum number of new tokens to generate (required if mode is 'inference')")
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help="Number of GPUs to use (default: 1)")
    parser.add_argument('--save_checkpoints', action='store_true', default=False, help="Enable saving checkpoints")
    parser.add_argument('--early_stopping_patience', type=int, default=5, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help="Minimum change in the monitored quantity to qualify as an improvement")
    args = parser.parse_args()

    if args.mode == 'tune' and args.n_trials is None:
        parser.error("--n_trials is required when mode is 'tune'")
    
    if args.mode == 'inference' and args.max_new_tokens is None:
        parser.error("--max_new_tokens is required when mode is 'inference'")

    if args.mode == 'train':
        if args.num_gpus > torch.cuda.device_count():
            raise ValueError(f"Requested {args.num_gpus} GPUs, but only {torch.cuda.device_count()} are available.")

        if args.distributed and torch.cuda.is_available():
            print(f"CUDA is available. Starting distributed training with {args.num_gpus} GPUs...")
            torch.multiprocessing.spawn(distributed_training, args=(
                args.num_gpus,
                args.batch_size,
                args.max_epochs,
                args.max_iters,
                args.eval_iters,
                args.eval_interval,
                args.num_samples,
                args.verbose,
                args.enable_tqdm,
                args.save_checkpoints,
                args.early_stopping_patience,
                args.early_stopping_min_delta,
                args.grad_norm_clip_value
            ), nprocs=args.num_gpus, join=True)
        else:
            print("Distributed training not enabled. Starting single-threaded training...")
            single_thread_train(
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                max_iters=args.max_iters,
                eval_interval=args.eval_interval,
                eval_iters=args.eval_iters,
                verbose=args.verbose,
                enable_tqdm=args.enable_tqdm,
                save_checkpoints=args.save_checkpoints,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                grad_norm_clip_value=args.grad_norm_clip_value
            )
        
    elif args.mode == 'tune':
        if args.distributed:
            if args.num_gpus > torch.cuda.device_count():
                raise ValueError(f"Requested {args.num_gpus} GPUs, but only {torch.cuda.device_count()} are available.")

            if torch.cuda.is_available():
                print(f"CUDA is available. Starting tuning with {args.num_gpus} GPUs...")
            else:
                print("CUDA is not available. Tuning on CPU...")
                args.num_gpus = 1  # Force to 1 GPU if CUDA is not available

            torch.multiprocessing.spawn(distributed_tuning, args=(
                args.num_gpus,
                args.n_trials,
                args.num_samples,
                args.verbose,
            ), nprocs=args.num_gpus, join=True)
        else:
            optimizer = HyperparameterOptimizer(
                tokenizer=args.tokenizer,
                regenerate_dataset=args.regenerate_dataset,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                max_iters=args.max_iters,
                eval_interval=args.eval_interval
            )
            optimizer.optimize()
    
    elif args.mode == 'inference':
        print("Starting inference...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print("Using device:", device)
        
        # Load hyperparameters from JSON file
        with open("models/wiki-llm/best/curr_best_params.json", "r") as f:
            params = json.load(f)["hyperparams"]

        model = ModelCustomTransformer(
            block_size=params["block_size"],
            vocab_size=params["vocab_size"],
            n_embd=params["hidden_size"],
            n_layer=params["num_layers"],
            n_head=params["num_heads"],
            dropout=params["dropout"]
        )
        checkpoint = torch.load("models/wiki-llm/best/curr_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        text_completer = TextCompleter(model, tokenizer, device, params["block_size"])

        while True:
            context = input("Enter context (or 'exit' to quit): ")
            if context.lower() == 'exit':
                break
            completion = text_completer.get_text_completions(max_tokens=args.max_new_tokens, context=context)
            print(f"Completion: {completion}")

if __name__ == "__main__":
    main()