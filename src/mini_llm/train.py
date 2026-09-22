"""Minimal training loop: forward -> loss -> zero_grad -> backward -> step.

No scheduler, no AMP, no grad accumulation, no clipping. Add those back
deliberately when you want them.

Reproducibility: --seed drives (a) model init, via torch.manual_seed before
the model is built, and (b) the training batch sequence, via a dedicated
Generator so it doesn't depend on how many random draws init happened to
consume. Validation always runs over the *entire* val set (every
non-overlapping window), so it needs no seed of its own and is identical
across runs for a given val-tokens file. The train/val split itself is
decided once, upstream, by `mini_llm.prepare_dataset --seed`.

Checkpoint resume: --resume loads model + optimizer + batch-generator state
and the running loss history, then continues from the saved step count.
Model hyperparams (--block-size, --n-embd, etc.) must match what the
checkpoint was trained with -- resume checks this and refuses to load a
mismatched config rather than silently producing shape errors partway
through.
"""

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from mini_llm.config import ModelConfig, build_model
from mini_llm.data import encode, fixed_batch, get_tokenizer, load_text, load_tokens, make_batch
from mini_llm.device import select_device
from mini_llm.generate import generate_text

PLOTS_DIR = Path("plots")
CHECKPOINTS_DIR = Path("checkpoints")


@torch.no_grad()
def evaluate_full(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str,
) -> float:
    """Average loss over every non-overlapping (x, y) window in `tokens`.

    Exhaustive rather than sampled, so it's the same every time for a given
    val-tokens file, model, and block_size -- no seed needed.
    """
    n_windows = (tokens.numel() - 1) // block_size
    if n_windows == 0:
        raise ValueError(
            f"need at least block_size + 1 = {block_size + 1} tokens, got {tokens.numel()}"
        )

    model.eval()
    total_loss, total_windows = 0.0, 0
    for start in range(0, n_windows, batch_size):
        offsets = [i * block_size for i in range(start, min(start + batch_size, n_windows))]
        x = torch.stack([tokens[o : o + block_size] for o in offsets]).to(device)
        y = torch.stack([tokens[o + 1 : o + block_size + 1] for o in offsets]).to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * len(offsets)
        total_windows += len(offsets)
    model.train()
    return total_loss / total_windows


def hyperparam_slug(cfg: ModelConfig, optim_cfg: dict[str, object], total_steps: int) -> str:
    """Compact identifier for a run's model + optimization hyperparams.

    Uses `total_steps` (cumulative across any --resume chain) rather than
    optim_cfg["steps"] (just this run's contribution), so a checkpoint or
    plot's filename always names how much training it actually represents.
    """
    return "_".join(
        [
            f"blk{cfg.block_size}",
            f"emb{cfg.n_embd}",
            f"head{cfg.n_head}",
            f"layer{cfg.n_layer}",
            f"bs{optim_cfg['batch_size']}",
            f"steps{total_steps}",
            f"lr{optim_cfg['lr']:g}",
            f"seed{optim_cfg['seed']}",
        ]
    )


def default_plot_name(cfg: ModelConfig, optim_cfg: dict[str, object], total_steps: int) -> str:
    return f"loss_{hyperparam_slug(cfg, optim_cfg, total_steps)}.png"


def default_checkpoint_name(cfg: ModelConfig, optim_cfg: dict[str, object], total_steps: int) -> str:
    return f"ckpt_{hyperparam_slug(cfg, optim_cfg, total_steps)}.pt"


def plot_loss(
    train_history: list[tuple[int, float]],
    val_history: list[tuple[int, float]],
    path: str | Path,
    hyperparams: dict[str, object] | None = None,
) -> None:
    """Save a PNG of train (and optional val) loss vs. step.

    Each line's legend entry reports its best (lowest) and last loss. The
    run's hyperparams are printed under the plot so a saved PNG can be
    matched back to the run that produced it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot_series(history: list[tuple[int, float]], name: str) -> None:
        steps, losses = zip(*history)
        best, last = min(losses), losses[-1]
        plt.plot(steps, losses, label=f"{name} (best {best:.4f}, last {last:.4f})")

    plt.figure()
    plot_series(train_history, "train")
    if val_history:
        plot_series(val_history, "val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.legend()

    if hyperparams:
        text = ", ".join(f"{k}={v}" for k, v in hyperparams.items())
        plt.figtext(0.5, -0.05, text, ha="center", va="top", fontsize=7, wrap=True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Train the tiny custom Transformer.")
    # data
    p.add_argument("--text", default=None, help="Path to a local .txt file (default: data/tiny.txt).")
    p.add_argument(
        "--tokens",
        default=None,
        help="Path to a pre-tokenized token tensor (.pt), e.g. from `mini-llm-prepare-data`. "
        "Overrides --text.",
    )
    p.add_argument(
        "--val-tokens",
        default=None,
        help="Path to pre-tokenized validation tokens (.pt). Enables val loss logging.",
    )
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
    p.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint (.pt) to continue training from. The current model "
        "flags (--block-size, --n-embd, --n-head, --n-layer, --dropout) must match what "
        "the checkpoint was trained with.",
    )
    # reporting / output
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=100, help="How often to compute val loss.")
    p.add_argument("--sample-tokens", type=int, default=0, help="Generate N tokens after training.")
    p.add_argument("--save", action="store_true", help="Save a checkpoint after training.")
    p.add_argument(
        "--save-name",
        default=None,
        help="Filename for the checkpoint when --save is set, saved under checkpoints/. "
        "Defaults to a name generated from the run's hyperparams and total step count.",
    )
    p.add_argument(
        "--plot-loss",
        action="store_true",
        help="Save a PNG plot of train/val loss over the run.",
    )
    p.add_argument(
        "--plot-name",
        default=None,
        help="Filename for the loss plot when --plot-loss is set, saved under plots/. "
        "Defaults to a name generated from the run's hyperparams.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = select_device()
    print(f"Using device: {device}")

    tokenizer = get_tokenizer()
    tokens = load_tokens(args.tokens) if args.tokens else encode(load_text(args.text), tokenizer)
    print(f"Tokens: {tokens.numel()}")

    cfg = ModelConfig(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    print(f"Config: {cfg.to_dict()}")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    batch = None
    if args.fixed_batch:
        batch = fixed_batch(tokens, args.batch_size, cfg.block_size, device=device, seed=args.seed)
        print("Training on one fixed batch.")

    # Dedicated generator for the resampled-batch case: the batch sequence
    # then depends only on --seed, not on how many random draws model init
    # (or anything else before the loop) happened to consume.
    batch_rng = torch.Generator().manual_seed(args.seed)

    start_step = 0
    train_history: list[tuple[int, float]] = []
    val_history: list[tuple[int, float]] = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_cfg = ModelConfig(**ckpt["config"])
        if ckpt_cfg.to_dict() != cfg.to_dict():
            raise ValueError(
                f"--resume checkpoint was trained with config {ckpt_cfg.to_dict()}, but "
                f"the current flags produce {cfg.to_dict()}. Pass the same model flags "
                "(--block-size, --n-embd, --n-head, --n-layer, --dropout) used for the "
                "original run to resume it."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        batch_rng.set_state(ckpt["batch_rng_state"])
        start_step = ckpt["step"]
        train_history = ckpt["train_history"]
        val_history = ckpt["val_history"]
        print(f"Resumed from {args.resume} at step {start_step}")

    total_steps = start_step + args.steps
    tokens_processed = args.steps * args.batch_size * cfg.block_size
    optim_cfg = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "total_steps": total_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "fixed_batch": args.fixed_batch,
        "tokens_processed": tokens_processed,
    }
    print(f"Optimization: {optim_cfg}")

    val_tokens = load_tokens(args.val_tokens) if args.val_tokens else None

    model.train()
    for local_step in range(args.steps):
        step = start_step + local_step
        x, y = batch if batch is not None else make_batch(
            tokens, args.batch_size, cfg.block_size, device=device, generator=batch_rng
        )

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        is_last_step = local_step == args.steps - 1
        log_now = step % args.log_interval == 0 or is_last_step
        eval_now = val_tokens is not None and (step % args.eval_interval == 0 or is_last_step)

        val_loss = None
        if eval_now:
            val_loss = evaluate_full(model, val_tokens, args.batch_size, cfg.block_size, device)
            val_history.append((step, val_loss))

        if log_now:
            train_history.append((step, loss.item()))
            msg = f"step {step:5d} | loss {loss.item():.4f}"
            if val_loss is not None:
                msg += f" | val_loss {val_loss:.4f}"
            print(msg)

    if args.plot_loss:
        plot_path = PLOTS_DIR / (args.plot_name or default_plot_name(cfg, optim_cfg, total_steps))
        plot_loss(train_history, val_history, plot_path, hyperparams={**cfg.to_dict(), **optim_cfg})
        print(f"Saved loss plot to {plot_path}")

    if args.save:
        save_path = CHECKPOINTS_DIR / (args.save_name or default_checkpoint_name(cfg, optim_cfg, total_steps))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": cfg.to_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "batch_rng_state": batch_rng.get_state(),
                "step": total_steps,
                "train_history": train_history,
                "val_history": val_history,
            },
            save_path,
        )
        print(f"Saved to {save_path}")

    if args.sample_tokens:
        print(generate_text(model, tokenizer, "\n", args.sample_tokens, cfg.block_size, device))


if __name__ == "__main__":
    main()
