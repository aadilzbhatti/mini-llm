"""Measure and remove document overlap between token files.

Why this exists: `prepare_dataset` guarantees train/val disjointness *for
files it produced*, by hashing each row's text (see `_val_pool_score`).
That guarantee says nothing about a token file produced by an *older*
version of the splitter, and `data/data10k` is exactly that -- a
positional split, made before the content-hash scheme landed. Its val
set therefore has no hash property at all: ~90% of its documents are
train-eligible under the current scheme, so they appear in
`data/data20k/train.pt`. Trusting the code's invariant instead of
measuring the files on disk is how a 90%-contaminated yardstick survives
unnoticed.

So: measure, don't assume. A document here is a maximal run of tokens
between EOS separators, identified by the SHA-256 of its token ids --
comparing ids rather than text means no tokenizer is needed and no
decode can go wrong.

    # report only
    python -m mini_llm.token_overlap data/data20k/train.pt data/data10k/val.pt

    # write a val file clean against both of data10k's splits
    python -m mini_llm.token_overlap data/data20k/val.pt \
        --exclude data/data10k/train.pt data/data10k/val.pt \
        --out data/val_clean.pt
"""

import argparse
import hashlib
from pathlib import Path

import torch

EOS_TOKEN_ID = 50256


def _load(path: str | Path) -> torch.Tensor:
    """Memory-mapped load: pages come in as they're read, so a multi-GB train
    file doesn't have to fit in RAM alongside everything else."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True).reshape(-1)


def document_spans(tokens: torch.Tensor) -> list[tuple[int, int]]:
    """(start, end) of each document in a 1-D EOS-separated stream, EOS excluded.

    Mirrors `prepare_dataset.tokenize_texts`, which appends EOS after each
    document, so document N is the tokens between separator N-1 and N.
    Empty runs (back-to-back separators) are skipped.
    """
    cuts = (tokens == EOS_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
    if not cuts or cuts[-1] != len(tokens) - 1:
        cuts.append(len(tokens))
    spans: list[tuple[int, int]] = []
    start = 0
    for cut in cuts:
        if cut > start:
            spans.append((start, cut))
        start = cut + 1
    return spans


def split_documents(tokens: torch.Tensor) -> list[torch.Tensor]:
    """Split a 1-D EOS-separated token stream into documents (views), EOS dropped."""
    flat = tokens.reshape(-1)
    return [flat[s:e] for s, e in document_spans(flat)]


def doc_hash(doc: torch.Tensor) -> str:
    """Content address for a document: SHA-256 over its token ids as int32."""
    return hashlib.sha256(doc.to(torch.int32).contiguous().numpy().tobytes()).hexdigest()


def _span_hashes(tokens: torch.Tensor, spans: list[tuple[int, int]]) -> list[str]:
    """`doc_hash` for every span, with one int32 conversion for the whole file
    instead of one per document."""
    buf = memoryview(tokens.to(torch.int32).contiguous().numpy()).cast("B")
    return [hashlib.sha256(buf[4 * s : 4 * e]).hexdigest() for s, e in spans]


def file_hashes(path: str | Path) -> set[str]:
    """Hashes of every document in a token file, without keeping the documents."""
    tokens = _load(path)
    return set(_span_hashes(tokens, document_spans(tokens)))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Report or remove document overlap between token files.")
    p.add_argument("tokens", help="Token file to inspect (and filter, with --out).")
    p.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Token files whose documents must not appear in the result.",
    )
    p.add_argument("--out", default=None, help="Write the filtered token stream here.")
    p.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Keep at most this many surviving documents, in file order. Use it to hold "
        "full-val cost fixed when the clean set comes out larger than the yardstick it replaces.",
    )
    args = p.parse_args(argv)

    tokens = _load(args.tokens)
    spans = document_spans(tokens)
    hashes = _span_hashes(tokens, spans)
    hash_set = set(hashes)
    n_tokens = sum(e - s for s, e in spans)
    print(f"{args.tokens}: {len(spans)} docs, {n_tokens:,} tokens (excluding separators)")

    excluded: set[str] = set()
    for path in args.exclude:
        other = file_hashes(path)
        shared = hash_set & other
        excluded |= shared
        pct = 100 * len(shared) / len(spans) if spans else 0.0
        print(f"  overlap with {path}: {len(shared)} docs ({pct:.1f}% of {Path(args.tokens).parent.name}/{Path(args.tokens).name})")

    keep = [span for span, h in zip(spans, hashes) if h not in excluded]
    removed = len(spans) - len(keep)
    if args.max_docs is not None and len(keep) > args.max_docs:
        print(f"  capping {len(keep)} clean docs to --max-docs {args.max_docs}")
        keep = keep[: args.max_docs]
    kept_tokens = sum(e - s for s, e in keep)
    print(f"  clean: {len(keep)} docs, {kept_tokens:,} tokens (removed {removed})")

    if args.out is None:
        return
    if not keep:
        raise SystemExit("refusing to write an empty token file")

    # Re-emit in the same shape prepare_dataset writes: documents joined by
    # a single EOS separator after each one.
    out_tokens = torch.empty(kept_tokens + len(keep), dtype=torch.long)
    pos = 0
    for s, e in keep:
        out_tokens[pos : pos + e - s] = tokens[s:e]
        out_tokens[pos + e - s] = EOS_TOKEN_ID
        pos += e - s + 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_tokens, out_path)
    print(f"Wrote {out_path}: {len(out_tokens):,} tokens")


if __name__ == "__main__":
    main()
