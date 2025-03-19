import argparse
import torch
from text_prediction.trainer import distributed_training, single_thread_train
from text_prediction.hyperparam_tuning import HyperparameterOptimizer, distributed_tuning
from text_prediction.text_completer import TextCompleter
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Train, tune hyperparameters, or generate text predictions for the WikiCompleteModel.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'tune', 'predict'], help="Mode to run: 'train', 'tune', or 'predict'")
    parser.add_argument('--regenerate_dataset', action='store_true', help="Regenerate the dataset")
    parser.add_argument('--num_samples', type=int, default=10000, help="Number of samples for the dataset (default: 10000)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--max_epochs', type=int, default=10, help="Maximum number of epochs (default: 10)")
    parser.add_argument('--max_iters', type=int, default=1000, help="Maximum number of iterations (default: 1000)")
    parser.add_argument('--eval_interval', type=int, default=100, help="Evaluation interval (default: 100)")
    parser.add_argument('--eval_iters', type=int, default=100, help="Evaluation iterations (default: 100)")
    parser.add_argument('--distributed', action='store_true', help="Use distributed training or tuning")
    parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose logging")
    parser.add_argument('--n_trials', type=int, help="Number of trials for hyperparameter tuning (required if mode is 'tune')")
    parser.add_argument('--enable_tqdm', action='store_true', default=False, help="Enable tqdm progress bar")
    parser.add_argument('--enable_profiling', action='store_true', default=False, help="Enable profiling during training")
    parser.add_argument('--max_new_tokens', type=int, help="Maximum number of new tokens to generate (required if mode is 'predict')")
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help="Number of GPUs to use (default: 1)")
    parser.add_argument('--save_checkpoints', action='store_true', default=False, help="Enable saving checkpoints")
    # parser.add_argument('--scheduler_step_size', type=int, default=10, help="Step size for learning rate scheduler")
    # parser.add_argument('--scheduler_gamma', type=float, default=0.1, help="Gamma for learning rate scheduler")
    args = parser.parse_args()

    if args.mode == 'tune' and args.n_trials is None:
        parser.error("--n_trials is required when mode is 'tune'")
    
    if args.mode == 'predict' and args.max_new_tokens is None:
        parser.error("--max_new_tokens is required when mode is 'predict'")

    if args.mode == 'train':
        if args.num_gpus > torch.cuda.device_count():
            raise ValueError(f"Requested {args.num_gpus} GPUs, but only {torch.cuda.device_count()} are available.")

        if args.distributed and torch.cuda.is_available():
            print(f"CUDA is available. Starting distributed training with {args.num_gpus} GPUs...")
            torch.multiprocessing.spawn(distributed_training, args=(
                args.num_gpus,
                args.max_epochs,
                args.max_iters,
                args.eval_iters,
                args.eval_interval,
                args.num_samples,
                args.verbose,
                args.enable_profiling,
                args.enable_tqdm,
                # args.scheduler_step_size,
                # args.scheduler_gamma
            ), nprocs=args.num_gpus, join=True)
        else:
            print("Disributed training not enabled. Starting single-threaded training...")
            single_thread_train(
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                max_iters=args.max_iters,
                eval_interval=args.eval_interval,
                eval_iters=args.eval_iters,
                verbose=args.verbose,
                enable_profiling=args.enable_profiling,
                enable_tqdm=args.enable_tqdm,
                save_checkpoints=args.save_checkpoints,
                # scheduler_step_size=args.scheduler_step_size,
                # scheduler_gamma=args.scheduler_gamma
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
    
    elif args.mode == 'predict':
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda" if torch.cuda.is_available() else "cpu")
        text_completer = TextCompleter(model, tokenizer, "cuda" if torch.cuda.is_available() else "cpu", args.max_new_tokens)

        while True:
            context = input("Enter context (or 'exit' to quit): ")
            if context.lower() == 'exit':
                break
            completion = text_completer.get_text_completions(max_tokens=args.max_new_tokens, context=context)
            print(f"Completion: {completion}")

if __name__ == "__main__":
    main()