import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, Dataset
import os

class WikipediaDataset:

    def __init__(self, tokenizer, max_len, block_size, num_samples=10000):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.block_size = block_size
        self.num_samples = num_samples
        self.train_dataloader = None
        self.val_dataloader = None

    def _get_max_length(self, dataset):
        def token_length(example):
            return {"length": len(self.tokenizer(example["text"])["input_ids"])}

        dataset_with_lengths = dataset.map(token_length)
        max_length = max(dataset_with_lengths["length"])
        return max_length

    def _tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_len
        )

    def _get_tokenized_dataset(self):
        tokenized_dataset_path = "data/wiki_data_tokenized"

        if not os.path.exists(tokenized_dataset_path):
            print("Local cache of dataset not found, downloading and tokenizing dataset...")
            # Load dataset (small subset of num_samples samples)
            ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
            ds = ds.shuffle(seed=42).select(range(self.num_samples))  # Select only a few samples
            # Select only the 'text' column
            ds = ds.remove_columns([col for col in ds.column_names if col != "text"])
            # Tokenize the dataset
            ds = ds.map(self._tokenize_function, batched=True)
            ds.save_to_disk(tokenized_dataset_path)
        else:
            print("Local cache of dataset found, loading tokenized dataset...")
            ds = load_from_disk(tokenized_dataset_path)
        return ds

    def _create_sequences(self, examples, block_size=128):
        # Get the tokenized input ids (should be a list of integers)
        input_ids = examples["input_ids"]
        
        # Initialize lists to store the new sequences and labels
        input_sequences = []
        label_sequences = []
        
        # Ensure input_ids is a list of integers
        if isinstance(input_ids, list):
            # Loop through the input sequences and split them into blocks
            for i in range(len(input_ids) - block_size):
                input_seq = input_ids[i:i + block_size]
                label_seq = input_ids[i + 1:i + block_size + 1]  # The next token is the label
                
                input_sequences.append(input_seq)
                label_sequences.append(label_seq)
        
        # Return the input sequences and labels as dictionaries
        return {"input_ids": input_sequences, "labels": label_sequences}
    
    def _create_sliding_window_dataset(self, block_size=128):
        tds = self._get_tokenized_dataset()

        data = [{"input_ids": row["input_ids"]} for row in tds]

        # Apply the function to create sequences for all rows
        new_data = [self._create_sequences(ex, block_size) for ex in data]

        ds = Dataset.from_dict({
            "input_ids": [seq for ex in new_data for seq in ex["input_ids"]],
            "labels": [seq for ex in new_data for seq in ex["labels"]],
        })
        return ds

    def _get_sliding_window_dataset(self, block_size=128):
        dataset_path = "data/wiki_data_tokenized_sliding_window"

        if not os.path.exists(dataset_path):
            print("Local cache of sliding window dataset not found, creating dataset...")
            ds = self._create_sliding_window_dataset(block_size)
            ds.save_to_disk(dataset_path)
        else:
            print("Local cache of sliding window dataset found, loading dataset...")
            ds = load_from_disk(dataset_path)
        return ds

    def _create_train_test_dataloaders(self, batch_size):
        ds = self._get_sliding_window_dataset(self.block_size)

        # Split the dataset into train and validation sets
        train_size = 0.8
        train_ds, val_ds = ds.train_test_split(test_size=1-train_size, seed=42).values()

        # Create the dataloaders
        train_dataloader = DataLoader(
            train_ds.with_format('torch'),
            batch_size=batch_size,
            collate_fn=lambda x: {
                "input_ids": torch.stack([d['input_ids'] for d in x]),
                "labels": torch.stack([d['labels'] for d in x])
            }
        )

        val_dataloader = DataLoader(
            val_ds.with_format('torch'),
            batch_size=batch_size,
            collate_fn=lambda x: {
                "input_ids": torch.stack([d['input_ids'] for d in x]),
                "labels": torch.stack([d['labels'] for d in x])
            }
        )

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
    
    def get_test_train_dataloaders(self, split, batch_size=32):
        if self.train_dataloader is None:
            self._create_train_test_dataloaders(batch_size)
        if split == "train":
            return self.train_dataloader
        elif split == "val":
            return self.val_dataloader
        else:
            raise ValueError("Invalid split value. Must be either 'train' or 'val'.")
