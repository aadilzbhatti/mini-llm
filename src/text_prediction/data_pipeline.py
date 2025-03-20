import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset as TDataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_from_disk, Dataset
import os
import pyarrow as pa
from torch.utils.data import Dataset as TorchDataset
import logging
import random
import re

from text_prediction.utils import RankFilter

class TokenizedDataset(TorchDataset):
    def __init__(self, tokenized_dataset, block_size, device="cpu"):
        """
        tokenized_examples: List of tokenized sequences (each a list of token IDs).
        block_size: Length of each input sequence.
        """
        self.device = device
        self.block_size = block_size

        self._data = [torch.tensor(example['input_ids'], dtype=torch.long) for example in tokenized_dataset if len(example['input_ids']) > block_size]
        
    def __len__(self):
        return len(self._data) - self.block_size  # Max index to sample from

    def __getitem__(self, idx):
        """
        Return a single sample (input and target sequences)
        """
        example = self._data[idx]
        ix = torch.randint(0, len(example) - self.block_size, (1,)).item()
        input_seq = example[ix:ix + self.block_size]
        label_seq = example[ix + 1:ix + self.block_size + 1]
        return input_seq.to(self.device), label_seq.to(self.device)

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

    def _sanitize_text(self, text):
        """Sanitizes the text while preserving meaningful dashes, numerical values with units, and hyperlink text."""
        
        # Preserve Markdown-style and HTML links by keeping only the text
        text = re.sub(r'\[([^\]]+)\]\(http\S+\)', r'\1', text)  # Handles Markdown links
        text = re.sub(r'<a\s+href=["\']http\S+["\']>(.*?)</a>', r'\1', text, flags=re.IGNORECASE)  # Handles HTML links

        # Remove standalone URLs
        text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)

        # Ensure we preserve dashes in hyphenated place names (e.g., Indianapolis–Carmel–Anderson)
        text = re.sub(r'(\w)\s*[-–]\s*(\w)', r'\1–\2', text)  # Normalize hyphens and remove unwanted spaces around them

        # Keep valid numerical values with units (e.g., "3.0 square miles (7.8 km2)")
        text = re.sub(r'(\d+(\.\d+)?)\s*([a-zA-Z²]+)', r'\1 \3', text)  # Ensures numbers and units stay together
        text = re.sub(r'\((\d+(\.\d+)?\s*[a-zA-Z²]+)\)', r'(\1)', text)  # Ensures parenthetical units remain intact

        # Preserve valid year ranges (e.g., 1992-2002)
        text = re.sub(r'(?<!\d)(\d{4})-(\d{4})(?!\d)', r'\1-\2', text)  # Ensure valid formatting

        # Remove unwanted characters but keep punctuation, parentheses, and percentage signs
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"()%-²]', '', text)

        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text


    def _tokenize_function(self, examples):
        """Tokenizes the examples, adds special tokens, and optionally augments data."""
        
        # Sanitize text data
        sanitized_texts = [self._sanitize_text(text) for text in examples["text"]]

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