import torch
from torch.utils.data import Dataset as TorchDataset

class TokenizedDataset(TorchDataset):
    def __init__(self, tokenized_dataset, block_size, device="cpu"):
        """
        tokenized_examples: List of tokenized sequences (each a list of token IDs).
        block_size: Length of each input sequence.
        """
        self.device = device
        self.block_size = block_size

        # for example in tokenized_dataset:
        #     print("printing one example")
        #     print(example)
        #     break
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