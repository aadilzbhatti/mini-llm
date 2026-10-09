"""Pull a Hugging Face text dataset and turn it into fixed local token files.

    HF dataset (e.g. HuggingFaceFW/fineweb-edu)
        -> deterministic subset (seeded shuffle, streamed)
        -> bucket each row into train or val by a content hash
        -> extract text field
        -> tokenize (in batches, as the stream is scanned)
        -> fixed train tokens / fixed val tokens
        -> save locally (.pt files)

This is the only module that knows about Hugging Face `datasets` or the
shape of a particular dataset's records. `data.py` and training only ever
see the resulting token tensors, via `data.load_tokens`, so they stay
agnostic to which dataset (or text) produced them.

Growing the dataset later (a bigger --num-examples or --val-examples) must
never let a row switch sides, in either direction. A row's side is decided
by hashing its own text (see `_val_pool_score`) against a threshold, never
by its position in the stream or by how many rows either side has asked
for. That's what makes it safe: raising --num-examples can only add new
rows to train, and raising --val-examples can only add new rows to val --
neither can ever steal from, or reclassify, a row already claimed by the
other side, because a row's hash never changes.

An earlier version split by *position* instead (val = the first N rows,
train = the rows after it). That made train safely growable, but growing
val shifted the val/train boundary forward and silently stole rows that
were already part of train -- exactly the leak this hashing scheme rules
out for both sides at once.

`val_pool_fraction` controls what fraction of the stream's rows are even
eligible for the val bucket -- it must stay fixed across runs (like
--seed) for the guarantee above to hold, the same way changing --seed
would change which rows you see at all. It's a ratio for scan efficiency,
not a size knob -- --val-examples controls the actual count.
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from mini_llm.data import get_tokenizer

DEFAULT_DATASET = "HuggingFaceTB/smollm-corpus"
DEFAULT_CONFIG = "fineweb-edu-dedup"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_OUT_DIR = Path("data")
DEFAULT_VAL_EXAMPLES = 200
DEFAULT_VAL_POOL_FRACTION = 0.1
TOKENIZE_BATCH = 1000


def _val_pool_score(text: str) -> float:
    """Deterministic pseudo-random value in [0, 1) from a row's content.

    Content-addressed, not position-addressed: whether a row belongs to
    the val pool depends only on its own text, never on stream/shuffle
    order or on how many rows either side has requested.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def iter_split(
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    text_field: str = DEFAULT_TEXT_FIELD,
    num_examples: int = 2000,
    val_examples: int = DEFAULT_VAL_EXAMPLES,
    val_pool_fraction: float = DEFAULT_VAL_POOL_FRACTION,
    seed: int = 0,
) -> Iterator[tuple[Literal["train", "val"], dict[str, object]]]:
    """Stream `dataset`, yielding (side, row) as each row is claimed by train or val.

    Scans until both quotas are filled (or the stream runs out). See the
    module docstring for why this makes both sides independently growable
    without cross-contamination. Yielding rather than collecting keeps
    memory flat however many rows are asked for.
    """
    ds = load_dataset(dataset, config, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    n_train = n_val = 0
    for row in ds:
        text = cast(str, row[text_field])
        if _val_pool_score(text) < val_pool_fraction:
            if n_val < val_examples:
                n_val += 1
                yield "val", row
        elif n_train < num_examples:
            n_train += 1
            yield "train", row

        if n_train >= num_examples and n_val >= val_examples:
            return


def load_subset(
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    text_field: str = DEFAULT_TEXT_FIELD,
    num_examples: int = 2000,
    val_examples: int = DEFAULT_VAL_EXAMPLES,
    val_pool_fraction: float = DEFAULT_VAL_POOL_FRACTION,
    seed: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collect `iter_split` into (train_rows, val_rows). Holds every row in memory."""
    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    for side, row in iter_split(
        dataset, config, split, text_field, num_examples, val_examples, val_pool_fraction, seed
    ):
        (train_rows if side == "train" else val_rows).append(row)
    return train_rows, val_rows


def extract_text(rows: list[dict[str, object]], text_field: str = DEFAULT_TEXT_FIELD) -> list[str]:
    """Pull the text field out of each dataset row."""
    return [cast(str, row[text_field]) for row in rows]


class TokenStream:
    """Accumulates an EOS-separated token stream without holding Python ints.

    Texts are buffered and tokenized `batch_size` at a time (the fast
    tokenizer batches in Rust), and each batch is packed into an int32
    chunk -- 4 bytes/token, versus ~36 for a Python `list[int]`. `tensor()`
    copies the chunks into one preallocated int64 tensor, releasing each
    chunk as it goes, so peak memory is ~1.5x the final tensor rather than
    several times it.

    Tokenizing a batch gives exactly what per-text `tokenizer.encode` would:
    neither adds special tokens for GPT-2, and neither truncates.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int = TOKENIZE_BATCH):
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        self.batch_size = batch_size
        self.pending: list[str] = []
        self.chunks: list[torch.Tensor] = []
        self.num_docs = 0
        self.num_tokens = 0

    def add(self, text: str) -> None:
        self.pending.append(text)
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        flat: list[int] = []
        for ids in self.tokenizer(self.pending, add_special_tokens=False)["input_ids"]:
            flat.extend(ids)
            if self.eos is not None:
                flat.append(self.eos)
        chunk = torch.tensor(flat, dtype=torch.int32)
        self.chunks.append(chunk)
        self.num_docs += len(self.pending)
        self.num_tokens += len(chunk)
        self.pending.clear()

    def tensor(self) -> torch.Tensor:
        """Return the whole stream as one 1-D int64 tensor, consuming the chunks."""
        self.flush()
        out = torch.empty(self.num_tokens, dtype=torch.long)
        pos = 0
        self.chunks.reverse()
        while self.chunks:
            chunk = self.chunks.pop()
            out[pos : pos + len(chunk)] = chunk
            pos += len(chunk)
        return out


def tokenize_texts(texts: Iterable[str], tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Concatenate all texts into one 1-D EOS-separated token stream."""
    stream = TokenStream(tokenizer)
    for text in texts:
        stream.add(text)
    return stream.tensor()


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
    val_examples: int = DEFAULT_VAL_EXAMPLES,
    val_pool_fraction: float = DEFAULT_VAL_POOL_FRACTION,
    seed: int = 0,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    tokenizer_name: str = "gpt2",
    progress_every: int = 20_000,
) -> tuple[Path, Path]:
    """Run the full pipeline and write train/val token tensors to `out_dir`.

    Rows are tokenized as the stream is scanned, so raw text is never all
    held at once. Returns (train_path, val_path).
    """
    tokenizer = get_tokenizer(tokenizer_name)
    streams = {"train": TokenStream(tokenizer), "val": TokenStream(tokenizer)}
    seen = {"train": 0, "val": 0}
    for side, row in iter_split(
        dataset, config, split, text_field, num_examples, val_examples, val_pool_fraction, seed
    ):
        streams[side].add(cast(str, row[text_field]))
        seen[side] += 1
        if progress_every and side == "train" and seen["train"] % progress_every == 0:
            print(
                f"train {seen['train']:,}/{num_examples:,} docs, " f"{streams['train'].num_tokens:,} tokens so far",
                flush=True,
            )

    out_dir = Path(out_dir)
    paths = {"train": out_dir / "train.pt", "val": out_dir / "val.pt"}
    for side in ("val", "train"):
        tokens = streams.pop(side).tensor()
        save_tokens(tokens, paths[side])
        del tokens
    return paths["train"], paths["val"]


# --- mixtures: several sources in fixed token proportions ----------------------


@dataclass
class MixSource:
    """One dataset in a mixture: `--mix DATASET[:CONFIG]=FRACTION`."""

    dataset: str
    config: str | None
    fraction: float

    @property
    def label(self) -> str:
        return f"{self.dataset}:{self.config}" if self.config else self.dataset

    @classmethod
    def parse(cls, spec: str) -> "MixSource":
        name, sep, frac = spec.rpartition("=")
        if not sep or not name:
            raise ValueError(f"--mix wants DATASET[:CONFIG]=FRACTION, got {spec!r}")
        dataset, _, config = name.partition(":")
        try:
            fraction = float(frac)
        except ValueError:
            raise ValueError(f"--mix fraction must be a number, got {frac!r} in {spec!r}") from None
        if not 0 < fraction <= 1:
            raise ValueError(f"--mix fraction must be in (0, 1], got {fraction} in {spec!r}")
        return cls(dataset, config or None, fraction)


def token_quotas(total: int, fractions: list[float]) -> list[int]:
    """Split `total` tokens by `fractions` (which must sum to 1); the rounding remainder goes to the last."""
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError(f"--mix fractions must sum to 1, got {sum(fractions)}")
    quotas = [int(total * f) for f in fractions[:-1]]
    return quotas + [total - sum(quotas)]


class _MixedSide:
    """One side (train or val) of a mixture: one EOS-separated token stream, filled from several sources,
    with each source's tokens counted. The stream is flushed whenever the source changes, so every
    tokenized batch belongs to one source and the per-source counts are exact (at flush granularity)."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, quotas: list[int], batch_size: int):
        self.stream = TokenStream(tokenizer, batch_size)
        self.quotas = quotas
        self.tokens = [0] * len(quotas)
        self.docs = [0] * len(quotas)
        self.current: int | None = None

    def _count(self, before: int) -> None:
        if self.current is not None:
            self.tokens[self.current] += self.stream.num_tokens - before

    def add(self, source: int, text: str) -> None:
        if source != self.current:
            before = self.stream.num_tokens
            self.stream.flush()
            self._count(before)
            self.current = source
        before = self.stream.num_tokens
        self.stream.add(text)
        self._count(before)
        self.docs[source] += 1

    def finish(self) -> torch.Tensor:
        before = self.stream.num_tokens
        self.stream.flush()
        self._count(before)
        return self.stream.tensor()

    def full(self, source: int) -> bool:
        return self.tokens[source] >= self.quotas[source]

    def progress(self, source: int) -> float:
        return self.tokens[source] / self.quotas[source] if self.quotas[source] else 1.0


def prepare_mix(
    sources: list[MixSource],
    num_tokens: int,
    val_num_tokens: int,
    split: str = DEFAULT_SPLIT,
    text_field: str = DEFAULT_TEXT_FIELD,
    val_pool_fraction: float = DEFAULT_VAL_POOL_FRACTION,
    seed: int = 0,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    tokenizer_name: str = "gpt2",
    progress_every: int = 50_000,
) -> tuple[Path, Path, dict]:
    """Like prepare(), from several datasets in fixed TOKEN proportions (documents differ in length between
    sources, so document proportions would not hold the token budget).

    Each source is streamed exactly as prepare() streams a single one (same seed, shuffle and content-hash
    train/val rule), so its rows are a prefix of what prepare() would take from it alone. Rows are drawn
    from whichever source is furthest behind its train quota, so the token file interleaves the sources
    in ~batch-sized runs and any prefix of it holds roughly the mixture. A source stops contributing to a
    side once that side's quota is met; each side may overshoot its quota by up to one tokenized batch
    (train) or one document (val, tokenized one at a time so its small quota is met exactly).

    Writes train.pt, val.pt (this mixture's own val) and mix.json (per-source docs and tokens) to out_dir.
    """
    tokenizer = get_tokenizer(tokenizer_name)
    fractions = [src.fraction for src in sources]
    train = _MixedSide(tokenizer, token_quotas(num_tokens, fractions), TOKENIZE_BATCH)
    val = _MixedSide(tokenizer, token_quotas(val_num_tokens, fractions), 1)
    streams = [
        iter_split(
            src.dataset,
            src.config,  # type: ignore[arg-type]
            split,
            text_field,
            num_examples=10**12,  # unbounded: quotas here are in tokens, checked below
            val_examples=10**12,
            val_pool_fraction=val_pool_fraction,
            seed=seed,
        )
        for src in sources
    ]
    next_progress = progress_every
    while True:
        open_ = [i for i in range(len(sources)) if not (train.full(i) and val.full(i))]
        if not open_:
            break
        # the source furthest behind on train (val fills early: it's a tiny fraction of the stream)
        i = min(open_, key=lambda j: (train.progress(j), j))
        try:
            side, row = next(streams[i])
        except StopIteration:
            raise RuntimeError(f"{sources[i].label} ran out before its token quotas were met") from None
        target = train if side == "train" else val
        if not target.full(i):
            target.add(i, cast(str, row[text_field]))
        if progress_every and sum(train.docs) >= next_progress:
            next_progress += progress_every
            done = ", ".join(f"{s.label} {train.tokens[j]:,}/{train.quotas[j]:,}" for j, s in enumerate(sources))
            print(f"train {sum(train.docs):,} docs; tokens {done}", flush=True)

    out_dir = Path(out_dir)
    paths = {"train": out_dir / "train.pt", "val": out_dir / "val.pt"}
    stats = {
        "sources": [
            {
                "dataset": s.dataset,
                "config": s.config,
                "fraction": s.fraction,
                **{
                    f"{name}_{k}": getattr(side, attr)[j]
                    for name, side in (("train", train), ("val", val))
                    for k, attr in (("quota", "quotas"), ("tokens", "tokens"), ("docs", "docs"))
                },
            }
            for j, s in enumerate(sources)
        ],
        "seed": seed,
        "split": split,
        "val_pool_fraction": val_pool_fraction,
        "tokenizer": tokenizer_name,
    }
    for name, side in (("val", val), ("train", train)):
        tokens = side.finish()
        stats[f"{name}_tokens"] = int(tokens.numel())
        save_tokens(tokens, paths[name])
        del tokens
    for src in stats["sources"]:
        src["train_share"] = src["train_tokens"] / stats["train_tokens"]
    (out_dir / "mix.json").write_text(json.dumps(stats, indent=2) + "\n")
    return paths["train"], paths["val"], stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pull a HF dataset, tokenize a deterministic subset, and save fixed train/val token files."
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset repo id.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset config/subset name.")
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--text-field", default=DEFAULT_TEXT_FIELD, help="Column name holding the raw text.")
    p.add_argument(
        "--num-examples",
        type=int,
        default=2000,
        help="How many train rows to pull. Raising this later only appends new rows to "
        "train -- it never touches val.",
    )
    p.add_argument(
        "--val-examples",
        type=int,
        default=DEFAULT_VAL_EXAMPLES,
        help="How many rows to pull into validation. Raising this later only appends new "
        "rows to val -- it never touches train.",
    )
    p.add_argument(
        "--val-pool-fraction",
        type=float,
        default=DEFAULT_VAL_POOL_FRACTION,
        help="Fraction of rows eligible for the val bucket, by content hash (default "
        f"{DEFAULT_VAL_POOL_FRACTION}). Must stay fixed across runs, like --seed, for the "
        "growability guarantee to hold -- it's a scan-efficiency ratio, not a size knob.",
    )
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed; fixes the scan order deterministically.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument(
        "--mix",
        action="append",
        metavar="DATASET[:CONFIG]=FRACTION",
        help="Build a mixture instead of one dataset: repeat per source, fractions of TOKENS summing to 1, "
        "e.g. --mix HuggingFaceFW/fineweb:sample-10BT=0.7 --mix HuggingFaceTB/smollm-corpus:fineweb-edu-dedup=0.3. "
        "Sized by --num-tokens / --val-num-tokens; --dataset, --config, --num-examples and --val-examples are "
        "ignored.",
    )
    p.add_argument("--num-tokens", type=int, default=None, help="With --mix: train tokens in total.")
    p.add_argument("--val-num-tokens", type=int, default=None, help="With --mix: val tokens in total.")
    return p


def parse_args(argv: list[str] | None = None):
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mix:
        if args.num_tokens is None or args.val_num_tokens is None:
            raise SystemExit("--mix needs --num-tokens and --val-num-tokens")
        if len(args.mix) < 2:
            raise SystemExit("--mix needs at least two sources")
        train_path, val_path, stats = prepare_mix(
            [MixSource.parse(spec) for spec in args.mix],
            num_tokens=args.num_tokens,
            val_num_tokens=args.val_num_tokens,
            split=args.split,
            text_field=args.text_field,
            val_pool_fraction=args.val_pool_fraction,
            seed=args.seed,
            out_dir=args.out_dir,
            tokenizer_name=args.tokenizer,
        )
        for src in stats["sources"]:
            print(
                f"{src['dataset']}:{src['config']}: train {src['train_docs']:,} docs, {src['train_tokens']:,} tokens "
                f"({src['train_share']:.1%}); val {src['val_docs']:,} docs, {src['val_tokens']:,} tokens"
            )
        print(
            f"Saved train tokens to {train_path}\nSaved val tokens to {val_path}\nWrote {Path(args.out_dir) / 'mix.json'}"
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # see below: the streaming backend can leave a thread that blocks normal exit
    train_path, val_path = prepare(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        text_field=args.text_field,
        num_examples=args.num_examples,
        val_examples=args.val_examples,
        val_pool_fraction=args.val_pool_fraction,
        seed=args.seed,
        out_dir=args.out_dir,
        tokenizer_name=args.tokenizer,
    )
    print(f"Saved train tokens to {train_path}")
    print(f"Saved val tokens to {val_path}")

    # Force-exit here rather than returning normally. load_subset()
    # deliberately `break`s out of the streaming iterator early, once both
    # quotas are filled, rather than exhausting it -- and `datasets`'
    # streaming backend can leave a non-daemon background thread alive from
    # that (observed directly: the process sat at 0% CPU for 15+ minutes
    # after both "Saved ... tokens" lines above had already printed and the
    # files were fully written to disk). Normal interpreter shutdown waits
    # for every non-daemon thread to finish, so the process never returns
    # control to the shell. This has to live in main() itself, not behind
    # `if __name__ == "__main__"` -- the installed console script imports
    # main and calls it directly, so that guard never runs. All real work
    # (the token files) is already durably written by this point via
    # torch.save, so there's nothing left to lose by skipping Python's
    # normal cleanup.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
