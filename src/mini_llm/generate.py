"""Autoregressive generation, kept as thin as the original TextCompleter."""

import argparse

import torch
from transformers import PreTrainedTokenizerBase

from mini_llm.config import ModelConfig, build_model
from mini_llm.data import decode, encode, get_tokenizer
from mini_llm.device import select_device
from mini_llm.model import ModelCustomTransformer


@torch.no_grad()
def generate_text(
    model: ModelCustomTransformer,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    device: torch.device | str,
    greedy: bool = False,
) -> str:
    """Encode prompt -> model.generate() -> decode.

    Same three steps as src/text_prediction/text_completer.py in the original
    project. model.generate() itself is untouched: it crops to the last
    block_size tokens, softmaxes the final position and samples with
    torch.multinomial.
    """
    idx = encode(prompt, tokenizer).unsqueeze(0).to(device)
    out = model.generate(idx, max_new_tokens, block_size, greedy=greedy)
    return decode(out[0], tokenizer)


def parse_args(argv: list[str] | None = None):
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
    p.add_argument("--interactive", action="store_true", help="Run in interactive mode.")
    p.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    return p.parse_args(argv)



def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device()
    print(f"Using device: {device}")

    tokenizer = get_tokenizer()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = ModelConfig(**ckpt["config"])
        cfg.use_cache = True
        print(f"Config: {cfg.to_dict()}")
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
            use_cache=True,
        )
        print(f"Config: {cfg.to_dict()}")
        model = build_model(cfg).to(device)

    if args.interactive:
        while True:
            prompt = input("Enter a prompt (or 'quit' to exit): ")
            if not prompt:
                prompt = "\n"
            if prompt == "exit":
                break
            if prompt == "quit":
                break
            print(generate_text(model, tokenizer, prompt, args.max_new_tokens, cfg.block_size, device, greedy=args.greedy))
    else:
        print(generate_text(model, tokenizer, args.prompt, args.max_new_tokens, cfg.block_size, device, greedy=args.greedy))


if __name__ == "__main__":
    main()
