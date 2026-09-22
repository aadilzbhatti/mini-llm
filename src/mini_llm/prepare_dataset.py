"""Pull a Hugging Face text dataset and turn it into fixed local token files.

    HF dataset (e.g. HuggingFaceFW/fineweb-edu)
        -> deterministic subset (seeded shuffle + take, streamed)
        -> extract text field
        -> tokenize
        -> fixed train tokens / fixed val tokens
        -> save locally (.pt files)

This is the only module that knows about Hugging Face `datasets` or the
shape of a particular dataset's records. `data.py` and training only ever
see the resulting token tensors, via `data.load_tokens`, so they stay
agnostic to which dataset (or text) produced them.
"""

import argparse
from pathlib import Path
from typing import cast

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from mini_llm.data import get_tokenizer

DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-10BT"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_OUT_DIR = Path("data")


def load_subset(
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    num_examples: int = 2000,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Stream `dataset` and deterministically take `num_examples` rows.

    Streaming avoids downloading the whole dataset (FineWeb-Edu is
    multi-terabyte). Shuffling a streaming dataset shuffles within a
    fixed-size buffer, so a fixed seed + buffer_size reproduces the same
    subset every run without materializing the full dataset.
    """
    ds = load_dataset(dataset, config, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    return list(ds.take(num_examples))


def extract_text(rows: list[dict[str, object]], text_field: str = DEFAULT_TEXT_FIELD) -> list[str]:
    """Pull the text field out of each dataset row."""
    return [cast(str, row[text_field]) for row in rows]


def tokenize_texts(texts: list[str], tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Concatenate all texts into one 1-D EOS-separated token stream."""
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
        if eos is not None:
            ids.append(eos)
    return torch.tensor(ids, dtype=torch.long)


def split_train_val(tokens: torch.Tensor, val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one token stream into a fixed train prefix and val suffix."""
    n_val = int(tokens.numel() * val_fraction)
    return tokens[:-n_val], tokens[-n_val:]


def save_tokens(tokens: torch.Tensor, path: str | Path) -> None:
    """Save a 1-D token tensor, loadable later with `data.load_tokens`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, path)


def prepare(
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    text_field: str = DEFAULT_TEXT_FIELD,
    num_examples: int = 2000,
    val_fraction: float = 0.1,
    seed: int = 0,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    tokenizer_name: str = "gpt2",
) -> tuple[Path, Path]:
    """Run the full pipeline and write train/val token tensors to `out_dir`.

    Returns (train_path, val_path).
    """
    tokenizer = get_tokenizer(tokenizer_name)
    rows = load_subset(dataset, config, split, num_examples, seed)
    texts = extract_text(rows, text_field)
    tokens = tokenize_texts(texts, tokenizer)
    train_tokens, val_tokens = split_train_val(tokens, val_fraction)

    out_dir = Path(out_dir)
    train_path = out_dir / "train.pt"
    val_path = out_dir / "val.pt"
    save_tokens(train_tokens, train_path)
    save_tokens(val_tokens, val_path)
    return train_path, val_path


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Pull a HF dataset, tokenize a deterministic subset, and save fixed train/val token files."
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset repo id.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset config/subset name.")
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--text-field", default=DEFAULT_TEXT_FIELD, help="Column name holding the raw text.")
    p.add_argument("--num-examples", type=int, default=2000, help="How many rows to pull from the stream.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed; fixes the subset deterministically.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tokenizer", default="gpt2")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_path, val_path = prepare(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        text_field=args.text_field,
        num_examples=args.num_examples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        out_dir=args.out_dir,
        tokenizer_name=args.tokenizer,
    )
    print(f"Saved train tokens to {train_path}")
    print(f"Saved val tokens to {val_path}")


if __name__ == "__main__":
    main()
