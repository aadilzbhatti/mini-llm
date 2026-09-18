"""Autoregressive generation, kept as thin as the original TextCompleter."""

import argparse

import torch

from mini_llm.config import ModelConfig, build_model
from mini_llm.data import get_tokenizer
from mini_llm.device import select_device


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    device: torch.device | str,
) -> str:
    """Encode prompt -> model.generate() -> decode.

    Same three steps as src/text_prediction/text_completer.py in the original
    project. model.generate() itself is untouched: it crops to the last
    block_size tokens, softmaxes the final position and samples with
    torch.multinomial.
    """
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(idx, max_new_tokens, block_size)
    return tokenizer.decode(out[0].tolist())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate text from a tiny model.")
    p.add_argument("--prompt", default="\n", help="Prompt text.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional .pt written by `mini-llm-train --save`. Without it, "
        "generation runs from a freshly initialized (untrained) model.",
    )
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = select_device()
    print(f"Using device: {device}")

    tokenizer = get_tokenizer()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = ModelConfig(**ckpt["config"])
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        cfg = ModelConfig(
            vocab_size=len(tokenizer),
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
        )
        model = build_model(cfg).to(device)

    print(generate_text(model, tokenizer, args.prompt, args.max_new_tokens, cfg.block_size, device))


if __name__ == "__main__":
    main()
