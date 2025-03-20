import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_from_disk
import os
import pyarrow as pa
import logging
import random
import re

from text_prediction.tokenized_dataset import TokenizedDataset
from text_prediction.utils import RankFilter, sanitize_text

class DataPipeline:

    def __init__(self, tokenizer, max_len, block_size, regenerate=False, num_samples=10000, verbose=False, augment_data=False, parent_path="."):
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
        self.parent_path = parent_path

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')  # Update logging format
        self.logger = logging.getLogger(__name__)
        self.logger.addFilter(RankFilter(0))
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<ARTICLE_START>", "<ARTICLE_END>"]})

    def log(self, message, level=logging.INFO):
        if self.verbose:
            self.logger.log(level, message)

    def _tokenize_function(self, examples):
        """Tokenizes the examples, adds special tokens, and optionally augments data."""
        
        # Sanitize text data
        sanitized_texts = [sanitize_text(text) for text in examples["text"]]

        # Add special tokens to the beginning and end of each text
        texts_with_special_tokens = [
            "<ARTICLE_START>" + text + "<ARTICLE_END>" for text in sanitized_texts
        ]

        tokenized = self.tokenizer(
            texts_with_special_tokens,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=False,
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
        tokenized_dataset_path = f"{self.parent_path}/data/{self.tokenizer.name_or_path}/wiki/tokenized_augmented" if self.augment_data else f"{self.parent_path}/data/{self.tokenizer.name_or_path}/wiki/tokenized"

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

    def _is_dataset_valid(self, dataset_path):
        try:
            ds = load_from_disk(dataset_path)
            # Attempt to read a small portion of the dataset to ensure it's valid
            _ = ds[:1]
            return True
        except (pa.lib.ArrowInvalid, FileNotFoundError):
            return False

    @staticmethod
    def custom_collate(batch, tokenizer):
        inputs, labels = zip(*batch)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

        # Create attention masks
        B, T = inputs.shape
        tril_mask = torch.tril(torch.ones(T, T)).bool().to(inputs.device) #move tril to the same device as inputs
        attention_mask = (inputs != pad_token_id).unsqueeze(1).repeat(1, T, 1) & tril_mask #create a causal mask.
        return inputs, labels, attention_mask.to(inputs.device)

    def get_dataloader(self, batch_size, split="train", shuffle=True):
        # Ensure split is either "train" or "val"
        if split not in ["train", "val"]:
            raise ValueError("split must be either 'train' or 'val'")
        
        # DataLoader with random sampling
        tds = self._get_tokenized_dataset()
        
        # Split the dataset into train and validation sets
        split_datasets = tds.train_test_split(test_size=0.2, seed=42)
        selected_dataset = split_datasets["train" if split == "train" else "test"]
        
        dataset = TokenizedDataset(selected_dataset, self.block_size, self.device)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: DataPipeline.custom_collate(batch, self.tokenizer),
        )