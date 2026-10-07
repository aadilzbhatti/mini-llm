"""Tiny local text -> tokens -> (input, target) batches.

No HF datasets, no DataLoader, no caching. One text file becomes one long
1-D tensor of token ids, and batches are random crops of it.

The (x, y) slicing convention is carried over verbatim from the original
project's src/text_prediction/tokenized_dataset.py:

    x = tokens[i     : i + block_size]
    y = tokens[i + 1 : i + block_size + 1]
"""

from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TEXT_PATH = Path("data/tiny.txt")

# Fallback so the package works even if the data file is missing.
FALLBACK_TEXT = (
    "the quick brown fox jumps over the lazy dog. " "the lazy dog sleeps while the quick brown fox runs on. "
)


def get_tokenizer(name: str = "gpt2") -> PreTrainedTokenizerBase:
    """GPT-2 tokenizer, as in the original project.

    The original also registered <ARTICLE_START> / <ARTICLE_END> special
    tokens to mark Wikipedia article boundaries. That is dropped here, so
    vocab_size is just len(tokenizer).
    """
    return AutoTokenizer.from_pretrained(name)


def load_text(path: str | Path | None = None) -> str:
    """Read a local text file, or fall back to a built-in snippet."""
    if path is None:
        path = DEFAULT_TEXT_PATH
    path = Path(path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return FALLBACK_TEXT


def encode(text: str, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Text -> 1-D LongTensor of token ids."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def decode(ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> str:
    """Token ids (1-D tensor or list) -> text."""
    text = tokenizer.decode(ids)
    assert isinstance(text, str)
    return text


def load_tokens(path: str | Path, mmap: bool = True) -> torch.Tensor:
    """Load a pre-tokenized 1-D token tensor from disk.

    Agnostic to whatever dataset or text produced it — this just reads
    tensors written by `torch.save`, e.g. by `mini_llm.prepare_dataset`.

    By default the file is memory-mapped: the OS pages tokens in as random
    crops touch them and can drop them again under memory pressure, so a
    multi-GB train file needn't fit in RAM, and DDP ranks on one machine
    share one copy via the page cache. Everything downstream only slices
    and stacks short crops, so nothing ever forces the whole file in.
    Pass mmap=False to read it all up front instead -- better when the file
    sits on network storage, where each cold page would be a remote read.
    """
    tokens = torch.load(Path(path), weights_only=True, mmap=mmap)
    assert isinstance(tokens, torch.Tensor)
    return tokens


def make_batch(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random crops and return (x, y), each (B, T).

    Pass a seeded `generator` to get the same batch every time — that is how
    you build the one fixed batch for an overfit run.
    """
    if tokens.numel() < block_size + 1:
        raise ValueError(f"need at least block_size + 1 = {block_size + 1} tokens, got {tokens.numel()}")
    high = tokens.numel() - block_size
    ix = torch.randint(0, high, (batch_size,), generator=generator)
    x = torch.stack([tokens[i : i + block_size] for i in ix])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in ix])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y


def fixed_batch(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One reproducible batch: same seed in, same (x, y) out."""
    g = torch.Generator().manual_seed(seed)
    return make_batch(tokens, batch_size, block_size, device=device, generator=g)
