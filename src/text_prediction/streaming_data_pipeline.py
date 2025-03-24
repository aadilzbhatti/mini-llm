import tokenize
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import logging
import random
import hashlib

from text_prediction.tokenized_dataset import TokenizedDataset
from text_prediction.utils import RankFilter, sanitize_text

class StreamingDataPipeline:
    def __init__(self, tokenizer, max_len, block_size, verbose=False, augment_data=False, parent_path="."):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.block_size = block_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.augment_data = augment_data
        self.parent_path = parent_path

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.addFilter(RankFilter(0))
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<ARTICLE_START>", "<ARTICLE_END>"]})
        random.seed(42)

    def log(self, message, level=logging.INFO):
        if self.verbose:
            self.logger.log(level, message)

    def _tokenize_function(self, examples):
        sanitized_texts = [sanitize_text(text) for text in examples["text"]]
        texts_with_special_tokens = ["<ARTICLE_START>" + text + "<ARTICLE_END>" for text in sanitized_texts]
        tokenized = self.tokenizer(texts_with_special_tokens, truncation=True, max_length=self.max_len, add_special_tokens=False)
        if self.augment_data:
            tokenized["input_ids"] = self._augment(tokenized["input_ids"])
        return tokenized

    def _augment(self, input_ids):
        augmented = []
        for ids in input_ids:
            if random.random() < 0.1:
                ids[random.randint(0, len(ids) - 1)] = self.tokenizer.mask_token_id
            augmented.append(ids)
        return augmented

    @staticmethod
    def custom_collate(batch, tokenizer):
        inputs, labels = zip(*batch)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
        B, T = inputs.shape
        tril_mask = torch.tril(torch.ones(T, T)).bool().to(inputs.device)
        attention_mask = (inputs != pad_token_id).unsqueeze(1).repeat(1, T, 1) & tril_mask
        return inputs, labels, attention_mask.to(inputs.device)

    def is_train(self, example, test_size=0.2):
        example_str = str(example)
        hash_val = int(hashlib.md5(example_str.encode()).hexdigest(), 16)
        random_val = random.random()
        return random_val > test_size * (hash_val % 1000 / 1000)

    def get_dataloader_generator(self, batch_size, split="train", shuffle=True):
        if split not in ["train", "val"]:
            raise ValueError("split must be either 'train' or 'val'")

        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        ds = ds.map(self._tokenize_function, batched=True)

        tokenized_data = []
        for example in ds:
            if split == "train" and self.is_train(example):
                tokenized_data.append(example)
            elif split == "val" and not self.is_train(example):
                tokenized_data.append(example)
            if len(tokenized_data) > 10000: #load 10,000 samples at a time.
                dataset = TokenizedDataset(tokenized_data, self.block_size, self.device)
                yield DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: StreamingDataPipeline.custom_collate(batch, self.tokenizer))
                tokenized_data = []
        if len(tokenized_data) > 0:
            dataset = TokenizedDataset(tokenized_data, self.block_size, self.device)
            yield DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: StreamingDataPipeline.custom_collate(batch, self.tokenizer))