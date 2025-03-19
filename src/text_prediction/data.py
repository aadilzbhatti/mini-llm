import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, load_from_disk, Dataset
import os
import pyarrow as pa
from torch.utils.data import Dataset as TorchDataset
from torch.profiler import profile, record_function, ProfilerActivity
import logging
import random
from filelock import FileLock

from .utils import RankFilter

class WikipediaDataset:

    def __init__(self, tokenizer, max_len, block_size, regenerate=False, num_samples=10000, verbose=False, augment_data=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.block_size = block_size
        self.regenerate = regenerate
        self.num_samples = num_samples
        self.train_dataloader = None
        self.val_dataloader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.augment_data = augment_data

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')  # Update logging format
        self.logger = logging.getLogger(__name__)
        self.logger.addFilter(RankFilter(0))

        # # Mute filelock debug messages
        # logging.getLogger("filelock").setLevel(logging.WARNING)
        # # Mute datasets debug messages
        # logging.getLogger("datasets").setLevel(logging.WARNING)

    def log(self, message, level=logging.INFO):
        if self.verbose:
            self.logger.log(level, message)

    def _get_max_length(self, dataset):
        def token_length(example):
            return {"length": len(self.tokenizer(example["text"])["input_ids"])}

        dataset_with_lengths = dataset.map(token_length)
        max_length = max(dataset_with_lengths["length"])
        return max_length

    def _tokenize_function(self, examples):
        tokenized = self.tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_len
        )
        if self.augment_data:
            # Add data augmentation logic here
            tokenized["input_ids"] = self._augment(tokenized["input_ids"])
        return tokenized

    def _augment(self, input_ids):
        # Implement data augmentation logic
        # For example, randomly mask some tokens
        augmented = []
        for ids in input_ids:
            if random.random() < 0.1:  # 10% chance to mask a token
                ids[random.randint(0, len(ids) - 1)] = self.tokenizer.mask_token_id
            augmented.append(ids)
        return augmented

    def _get_tokenized_dataset(self):
        tokenized_dataset_path = f"data/{self.tokenizer.name_or_path}/wiki_data_tokenized_augmented" if self.augment_data else f"../data/{self.tokenizer.name_or_path}/wiki_data_tokenized"

        if not os.path.exists(tokenized_dataset_path) or self.regenerate:
            self.log("Local cache of dataset not found, downloading and tokenizing dataset...")
            # Load dataset (small subset of num_samples samples)
            ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
            ds = ds.select(range(self.num_samples))
            # Select only the 'text' column
            ds = ds.remove_columns([col for col in ds.column_names if col != "text"])
            # Tokenize the dataset
            ds = ds.map(self._tokenize_function, batched=True)
            ds.save_to_disk(tokenized_dataset_path)
        else:
            self.log("Local cache of dataset found, loading tokenized dataset...")
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
                label_seq = input_ids[i + 1:i + block_size + 1]  # The next token + block_size - 1 is the "label" sequence
                
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

    def _is_dataset_valid(self, dataset_path):
        try:
            ds = load_from_disk(dataset_path)
            # Attempt to read a small portion of the dataset to ensure it's valid
            _ = ds[:1]
            return True
        except (pa.lib.ArrowInvalid, FileNotFoundError):
            return False

    def _get_sliding_window_dataset(self, block_size=128):
        dataset_path = f"data/{self.tokenizer.name_or_path}/wiki_data_tokenized_sliding_window_augmented" if self.augment_data else f"../data/{self.tokenizer.name_or_path}/wiki_data_tokenized_sliding_window"
        lock_path = f"{dataset_path}.lock"

        with FileLock(lock_path):
            if not os.path.exists(dataset_path) or self.regenerate or not self._is_dataset_valid(dataset_path):
                self.log("Local cache of sliding window dataset not found or invalid, creating dataset...")
                ds = self._create_sliding_window_dataset(block_size)
                ds.save_to_disk(dataset_path)
            else:
                self.log("Local cache of sliding window dataset found, loading dataset...")
                ds = load_from_disk(dataset_path)
        return ds

    @staticmethod
    def collate_fn(batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = torch.stack([d['input_ids'] for d in batch]).pin_memory()
        labels = torch.stack([d['labels'] for d in batch]).pin_memory()
        return {
            "input_ids": input_ids.to(device, non_blocking=True),
            "labels": labels.to(device, non_blocking=True)
        }

    def _create_train_test_dataloaders(self, batch_size, rank=0, world_size=1):
        self.log("Creating train and validation dataloaders...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            ds = self._get_sliding_window_dataset(self.block_size)

            # Split dataset into train and validation sets
            train_size = 0.8
            train_ds, val_ds = ds.train_test_split(test_size=1 - train_size, seed=42).values()

            # Convert datasets to PyTorch format
            train_ds.set_format(type='torch', columns=['input_ids', 'labels'])
            val_ds.set_format(type='torch', columns=['input_ids', 'labels'])

            # Use DistributedSampler for multi-GPU training
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=False)
            val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

            # Determine pin_memory based on device
            pin_memory = self.device.type == "cuda"

            # Create the dataloaders with distributed sampling
            train_dataloader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=train_sampler,
                collate_fn=WikipediaDataset.collate_fn,
                pin_memory=False  # Enable pin_memory for faster data transfer to GPU
            )

            val_dataloader = DataLoader(
                val_ds,
                batch_size=batch_size,
                sampler=val_sampler,
                collate_fn=WikipediaDataset.collate_fn,
                pin_memory=False  # Enable pin_memory for faster data transfer to GPU
            )

            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader

            self.log(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    def get_test_train_dataloaders(self, split, batch_size=32, rank=0, world_size=1):
        if self.train_dataloader is None:
            self._create_train_test_dataloaders(batch_size, rank, world_size)

        if split == "train":
            return self.train_dataloader
        elif split == "val":
            return self.val_dataloader
        else:
            raise ValueError("Invalid split value. Must be either 'train' or 'val'.")
