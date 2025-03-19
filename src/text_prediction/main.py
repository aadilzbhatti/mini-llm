import argparse
from regex import F
import torch
from transformers import AutoTokenizer
from data import WikipediaDataset
from engine import TextPredictor

def main():
    parser = argparse.ArgumentParser(description="Wiki-LLM Project")
    
    parser.add_argument('--generate_new_dataset', action='store_true', 
                        help='Erase the data in the data/ dir and regenerate it', required=False)
    parser.add_argument('--num_samples', type=int, 
                        help='Number of samples to generate if generate_new_dataset is true', required=False)
    parser.add_argument('--train_model', action='store_true', 
                        help='Create the model and run a training loop', required=False)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

    if args.generate_new_dataset and args.num_samples is None:
        parser.error("--num_samples is required when --generate_new_dataset is set")
    
    if args.generate_new_dataset:
        print(f"Generating new dataset with {args.num_samples} samples...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        WikipediaDataset(tokenizer, 512, args.block_size, args.num_samples)
    
    if args.train_model:
        print("Training model...")
        # Add code to train the model here

if __name__ == "__main__":
    main()