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


def split_documents(tokens: torch.Tensor) -> list[torch.Tensor]:
    """Split a 1-D EOS-separated token stream into documents, EOS dropped.

    Mirrors `prepare_dataset.tokenize_texts`, which appends EOS after each
    document, so document N is the tokens between separator N-1 and N.
    """
    flat = tokens.reshape(-1)
    cuts = (flat == EOS_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()
    docs: list[torch.Tensor] = []
    start = 0
    for cut in cuts + ([len(flat)] if (len(cuts) == 0 or cuts[-1] != len(flat) - 1) else []):
        doc = flat[start:cut]
        if len(doc) > 0:
            docs.append(doc)
        start = cut + 1
    return docs


def doc_hash(doc: torch.Tensor) -> str:
    """Content address for a document: SHA-256 over its token ids as int32."""
    return hashlib.sha256(doc.to(torch.int32).contiguous().numpy().tobytes()).hexdigest()


def hashed_docs(path: str | Path) -> tuple[list[torch.Tensor], list[str]]:
    tokens = torch.load(path, map_location="cpu", weights_only=True)
    docs = split_documents(tokens)
    return docs, [doc_hash(d) for d in docs]


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

    docs, hashes = hashed_docs(args.tokens)
    n_tokens = sum(len(d) for d in docs)
    print(f"{args.tokens}: {len(docs)} docs, {n_tokens:,} tokens (excluding separators)")

    excluded: set[str] = set()
    for path in args.exclude:
        _, other = hashed_docs(path)
        shared = set(hashes) & set(other)
        excluded |= set(other)
        pct = 100 * len(shared) / len(docs) if docs else 0.0
        print(f"  overlap with {path}: {len(shared)} docs ({pct:.1f}% of {Path(args.tokens).parent.name}/{Path(args.tokens).name})")

    keep = [d for d, h in zip(docs, hashes) if h not in excluded]
    removed = len(docs) - len(keep)
    if args.max_docs is not None and len(keep) > args.max_docs:
        print(f"  capping {len(keep)} clean docs to --max-docs {args.max_docs}")
        keep = keep[: args.max_docs]
    kept_tokens = sum(len(d) for d in keep)
    print(f"  clean: {len(keep)} docs, {kept_tokens:,} tokens (removed {removed})")

    if args.out is None:
        return
    if not keep:
        raise SystemExit("refusing to write an empty token file")

    # Re-emit in the same shape prepare_dataset writes: documents joined by
    # a single EOS separator after each one.
    pieces: list[torch.Tensor] = []
    for doc in keep:
        pieces.append(doc)
        pieces.append(torch.tensor([EOS_TOKEN_ID], dtype=torch.long))
    out_tokens = torch.cat(pieces).to(torch.long)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_tokens, out_path)
    print(f"Wrote {out_path}: {len(out_tokens):,} tokens")


if __name__ == "__main__":
    main()
