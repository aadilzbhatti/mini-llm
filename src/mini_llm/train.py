"""Minimal training loop: forward -> loss -> zero_grad -> backward -> step.

No scheduler, no AMP, no grad accumulation, no clipping, no eval loop, no
checkpoint resume. Add those back deliberately when you want them.
"""

import argparse

import torch
from torch.optim import AdamW

from mini_llm.config import ModelConfig, build_model
from mini_llm.data import encode, fixed_batch, get_tokenizer, load_text, make_batch
from mini_llm.device import select_device
from mini_llm.generate import generate_text


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the tiny custom Transformer.")
    # data
    p.add_argument("--text", default=None, help="Path to a local .txt file (default: data/tiny.txt).")
    # model
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    # optimization
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fixed-batch",
        action="store_true",
        help="Train on ONE batch, resampled never. For overfit experiments.",
    )
    # reporting / output
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--sample-tokens", type=int, default=0, help="Generate N tokens after training.")
    p.add_argument("--save", default=None, help="Optional path to write model weights + config.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = select_device()
    print(f"Using device: {device}")

    tokenizer = get_tokenizer()
    tokens = encode(load_text(args.text), tokenizer)
    print(f"Tokens: {tokens.numel()}")

    cfg = ModelConfig(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    batch = None
    if args.fixed_batch:
        batch = fixed_batch(tokens, args.batch_size, cfg.block_size, device=device, seed=args.seed)
        print("Training on one fixed batch.")

    model.train()
    for step in range(args.steps):
        x, y = batch if batch is not None else make_batch(
            tokens, args.batch_size, cfg.block_size, device=device
        )

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps - 1:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    if args.save:
        torch.save({"config": cfg.to_dict(), "model_state_dict": model.state_dict()}, args.save)
        print(f"Saved to {args.save}")

    if args.sample_tokens:
        print(generate_text(model, tokenizer, "\n", args.sample_tokens, cfg.block_size, device))


if __name__ == "__main__":
    main()
