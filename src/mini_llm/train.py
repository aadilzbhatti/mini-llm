"""Minimal training loop: forward -> loss -> zero_grad -> backward -> step.

No AMP, no grad accumulation, no clipping. Add those back deliberately when
you want them.

LR schedule: linear warmup (--warmup-steps, default 500) then cosine decay
from --lr (default 1e-3) down to --min-lr (default 2e-6), over this run's
total step horizon (start_step + --steps, so a --resume decays across its
own new horizon rather than the original run's). --min-lr is the actual
decay knob -- set it equal to --lr to disable decay and train at a constant
rate; --warmup-steps 0 disables warmup.

Reproducibility: --seed drives (a) model init, via torch.manual_seed before
the model is built, and (b) the training batch sequence, via a dedicated
Generator so it doesn't depend on how many random draws init happened to
consume. Train/val split itself is decided once, upstream, by
`mini_llm.prepare_dataset --seed`.

Eval: the per-step "loss" printed during the loop is one noisy training
batch mid-backprop -- not comparable to a val loss. So at every
--eval-interval checkpoint we instead compute eval_train_loss and
eval_val_loss the *same* way: both are the average loss over
--eval-batches batches, both in eval() mode, under no_grad. Those batches
are sampled once, at init, and then reused unchanged at every checkpoint --
step 500, 1000, 1500, ... all evaluate the exact same tokens. Resampling a
fresh eval set each time would confound "the model changed" with "the
measuring stick changed"; a fixed one isolates the former.

--eval-seed picks that fixed set and is deliberately separate from --seed:
--seed is for the *model* (init + training batch order), so sweeping it
across experiments compares different models on the *same* measuring stick.
--eval-seed only needs to change if you deliberately want a different eval
sample. (Resuming with the same --batch-size/--eval-batches/--eval-seed
regenerates the identical set, since it's a pure function of those plus the
token file.)

full_val_loss is a third, separate metric: the exhaustive loss over *every*
window in the validation set (not a --eval-batches sample of it). It's the
true number, but expensive, so it only runs every --full-eval-interval
steps and always once at the end of training -- not at --eval-interval
cadence like eval_val_loss.

Checkpoint resume: --resume loads model + optimizer + batch-generator state
and the running loss history, then continues from the saved step count.
Model hyperparams (--block-size, --n-embd, etc.) must match what the
checkpoint was trained with -- resume checks this and refuses to load a
mismatched config rather than silently producing shape errors partway
through.
"""

import argparse
import math
from pathlib import Path

import torch
from torch.optim import AdamW

from mini_llm.config import ModelConfig, build_model
from mini_llm.data import encode, fixed_batch, get_tokenizer, load_text, load_tokens, make_batch
from mini_llm.device import select_device
from mini_llm.generate import generate_text
from mini_llm.report import write_sample_report

PLOTS_DIR = Path("plots")
CHECKPOINTS_DIR = Path("checkpoints")
DEFAULT_EVAL_SEED = 1234


def sample_eval_batches(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str,
    num_batches: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Sample a fixed set of batches, once, to reuse at every eval checkpoint.

    Its own throwaway generator, seeded independently of the training batch
    sequence, so building this set never perturbs training.
    """
    generator = torch.Generator().manual_seed(seed)
    return [
        make_batch(tokens, batch_size, block_size, device=device, generator=generator)
        for _ in range(num_batches)
    ]


@torch.no_grad()
def evaluate_fixed(model: torch.nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Average loss over a fixed, pre-sampled set of batches.

    Reusing the same batches at every checkpoint means successive
    evaluations measure how the model changed, not how the sample did.
    """
    model.eval()
    total_loss = 0.0
    for x, y in batches:
        _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / len(batches)


@torch.no_grad()
def evaluate_full(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str,
) -> float:
    """Average loss over every non-overlapping window in `tokens`.

    Exhaustive, not sampled -- the true loss over the whole set, at the
    cost of a full pass. Meant to run infrequently (--full-eval-interval),
    as a periodic sanity check against the cheaper fixed-sample estimate.
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


def lr_at_step(step: int, total_steps: int, lr: float, min_lr: float, warmup_steps: int) -> float:
    """Linear warmup for `warmup_steps`, then cosine decay from `lr` to `min_lr`.

    `total_steps` is the horizon the cosine curve spans; `min_lr == lr`
    collapses this to a constant rate regardless of warmup. `min_lr` is a
    genuine floor -- the warmup ramp is clamped to it, so no step ever runs
    below it.
    """
    if warmup_steps > 0 and step < warmup_steps:
        # Clamp to min_lr: without this the ramp starts at lr/warmup_steps,
        # which for a small peak sits BELOW the floor (peak 3e-4 over 500
        # warmup steps starts at 6e-7 against a 2e-6 floor). It went unnoticed
        # for a long time because peak 1e-3 over 500 steps starts at exactly
        # 2e-6 -- numerically identical to the usual floor, by coincidence.
        return max(min_lr, lr * (step + 1) / warmup_steps)
    if total_steps <= warmup_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


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
            f"minlr{optim_cfg['min_lr']:g}",
            f"seed{optim_cfg['seed']}",
        ]
    )


def default_plot_name(cfg: ModelConfig, optim_cfg: dict[str, object], total_steps: int, suffix: str = "") -> str:
    suffix_part = f"_{suffix}" if suffix else ""
    return f"loss_{hyperparam_slug(cfg, optim_cfg, total_steps)}{suffix_part}.png"


def default_checkpoint_name(cfg: ModelConfig, optim_cfg: dict[str, object], total_steps: int) -> str:
    return f"ckpt_{hyperparam_slug(cfg, optim_cfg, total_steps)}.pt"


def print_dataset_stats(
    train_tokens: int,
    val_tokens: int | None,
    batch_tokens: int,
    steps: int,
    tokens_processed: int,
    effective_epochs: float,
) -> None:
    """Startup summary: dataset size vs. how much of it this run will touch.

    effective_epochs = tokens_processed / train_tokens -- how many times, on
    average, this run sweeps the training set (batches are random crops, not
    an epoch iterator, so this is an expectation, not an exact pass count).
    """
    rows = [
        ("Train tokens", f"{train_tokens:,}"),
        ("Validation tokens", f"{val_tokens:,}" if val_tokens is not None else "n/a"),
        ("Batch tokens", f"{batch_tokens:,}"),
        ("Training steps", f"{steps:,}"),
        ("Tokens processed", f"{tokens_processed:,}"),
        ("Effective epochs", f"{effective_epochs:.2f}"),
    ]
    label_width = max(len(label) for label, _ in rows) + 1
    value_width = max(len(value) for _, value in rows)
    for label, value in rows:
        print(f"{label + ':':<{label_width}} {value:>{value_width}}")


def plot_loss(
    train_history: list[tuple[int, float]],
    val_history: list[tuple[int, float]],
    full_val_history: list[tuple[int, float]] | None,
    path: str | Path,
    hyperparams: dict[str, object] | None = None,
    lr_history: list[tuple[int, float]] | None = None,
) -> None:
    """Save a PNG of eval_train_loss, eval_val_loss, and full_val_loss vs. step.

    eval_train_loss/eval_val_loss are the same metric -- average loss over N
    sampled batches, in eval() mode -- so they're directly comparable, not
    train's noisy per-step loss against val's averaged one. full_val_loss is
    the exhaustive loss over the entire validation set, plotted sparsely
    (markers, dashed) since it only runs every --full-eval-interval steps --
    a periodic ground-truth check against the cheaper sampled estimate. Its
    exact values are also rendered as a side table, since reading them off
    a handful of sparse markers is imprecise. Each line's legend entry
    reports its best (lowest) and last loss. The run's hyperparams are
    printed under the plot so a saved PNG can be matched back to the run
    that produced it.

    lr_history, when given, is drawn on a secondary log-scale y-axis --
    loss and LR live on completely different scales, so sharing an axis
    would flatten the LR line to near-invisible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if full_val_history:
        fig, (ax1, ax_table) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]})
        ax_table.axis("off")
        ax_table.set_title("full_val_loss", fontsize=9)
        table = ax_table.table(
            cellText=[[f"{step:,}", f"{loss:.4f}"] for step, loss in full_val_history],
            colLabels=["step", "loss"],
            cellLoc="center",
            loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
    else:
        fig, ax1 = plt.subplots()

    def plot_series(ax, history: list[tuple[int, float]], name: str, **kwargs: object):
        steps, values = zip(*history)
        best, last = min(values), values[-1]
        return ax.plot(steps, values, label=f"{name} (best {best:.4f}, last {last:.4f})", **kwargs)[0]

    lines = [plot_series(ax1, train_history, "eval_train_loss")]
    if val_history:
        lines.append(plot_series(ax1, val_history, "eval_val_loss"))
    if full_val_history:
        lines.append(plot_series(ax1, full_val_history, "full_val_loss", marker="o", linestyle="--"))
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")

    if lr_history:
        ax2 = ax1.twinx()
        steps, lrs = zip(*lr_history)
        # Label the CONFIGURED floor, not min(lrs): the smallest value on the
        # curve is the first warmup step, not the annealing floor, and showing
        # it under the name "min" reads as the run having annealed somewhere
        # it did not.
        floor = (hyperparams or {}).get("min_lr")
        floor_txt = f"floor {float(floor):.2e}" if floor is not None else f"min {min(lrs):.2e}"
        (line_lr,) = ax2.plot(
            steps, lrs, color="gray", alpha=0.6, linestyle=":",
            label=f"lr (peak {max(lrs):.2e}, {floor_txt})",
        )
        ax2.set_ylabel("learning rate")
        ax2.set_yscale("log")
        lines.append(line_lr)

    ax1.set_title("Eval loss (sampled) vs. full validation loss (exhaustive)")
    ax1.legend(handles=lines, loc="best")

    if hyperparams:
        text = ", ".join(f"{k}={v}" for k, v in hyperparams.items())
        fig.text(0.5, -0.05, text, ha="center", va="top", fontsize=7, wrap=True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    # optimization
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--restart-lr",
        type=float,
        default=None,
        help="Only with --resume. The learning rate the continuation STARTS at. The "
        "continuation then cosine-decays from it to --min-lr over --steps, as its own "
        "schedule, with no warmup. Without this flag a resumed run continues the "
        "ORIGINAL cosine over [0, start_step + --steps], so it picks up partway down "
        "and --lr is not the rate it begins at.",
    )
    p.add_argument(
        "--min-lr",
        type=float,
        default=2e-6,
        help="Cosine decay target for the LR schedule. Set equal to --lr to disable decay.",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Linear warmup length before the cosine decay begins. 0 = no warmup.",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument("--eval-interval", type=int, default=100, help="How often to compute eval loss.")
    p.add_argument(
        "--eval-batches",
        type=int,
        default=20,
        help="Batches to average over for eval_train_loss/eval_val_loss at each --eval-interval.",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help="Seed for the fixed eval batches (train + val), independent of --seed. Fixed by "
        "default so sweeping --seed across experiments compares models on the same measuring "
        "stick; only change this if you deliberately want a different eval sample.",
    )
    p.add_argument(
        "--full-eval-interval",
        type=int,
        default=5000,
        help="How often to run an exhaustive full_val_loss over the entire validation set "
        "(in addition to always running one at the end of training). Only runs if "
        "--val-tokens is set; 0 disables the periodic runs (the end-of-training one still "
        "runs).",
    )
    p.add_argument("--sample-tokens", type=int, default=0, help="Generate N tokens after training.")
    p.add_argument("--save", action="store_true", help="Save a checkpoint after training.")
    p.add_argument(
        "--sample-report",
        action="store_true",
        help="After training, generate from a fixed prompt battery under fixed decoding "
        "settings and write the result to checkpoints/<checkpoint stem>.md. Uses the same "
        "name as --save-name, so a checkpoint and its report stay paired.",
    )
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
    p.add_argument(
        "--plot-suffix",
        default=None,
        help="Extra suffix appended to the generated plot filename (ignored if --plot-name "
        "is set), e.g. --plot-suffix notes gives loss_..._notes.png.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    if args.restart_lr is not None and not args.resume:
        raise SystemExit("--restart-lr only means something with --resume; there is no run to restart.")

    device = select_device()
    print(f"Using device: {device}")

    tokenizer = get_tokenizer()
    tokens = load_tokens(args.tokens) if args.tokens else encode(load_text(args.text), tokenizer)
    val_tokens = load_tokens(args.val_tokens) if args.val_tokens else None

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

    batch_tokens = args.batch_size * cfg.block_size
    tokens_processed = args.steps * batch_tokens
    effective_epochs = tokens_processed / tokens.numel()
    print_dataset_stats(
        tokens.numel(),
        val_tokens.numel() if val_tokens is not None else None,
        batch_tokens,
        args.steps,
        tokens_processed,
        effective_epochs,
    )

    min_lr = args.min_lr

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    batch = None
    if args.fixed_batch:
        batch = fixed_batch(tokens, args.batch_size, cfg.block_size, device=device, seed=args.seed)
        print("Training on one fixed batch.")

    # Dedicated generator for the resampled-batch case: the batch sequence
    # then depends only on --seed, not on how many random draws model init
    # (or anything else before the loop) happened to consume.
    batch_rng = torch.Generator().manual_seed(args.seed)

    # Fixed once, at init, and reused unchanged at every eval checkpoint --
    # see the module docstring on why these aren't resampled per checkpoint.
    train_eval_batches = sample_eval_batches(
        tokens, args.batch_size, cfg.block_size, device, args.eval_batches, args.eval_seed
    )
    val_eval_batches = (
        sample_eval_batches(val_tokens, args.batch_size, cfg.block_size, device, args.eval_batches, args.eval_seed)
        if val_tokens is not None
        else None
    )

    start_step = 0
    train_history: list[tuple[int, float]] = []
    val_history: list[tuple[int, float]] = []
    full_val_history: list[tuple[int, float]] = []
    lr_history: list[tuple[int, float]] = []

    if args.resume:
        # Load to CPU regardless of `device`: batch_rng is a CPU generator
        # and set_state() requires a CPU ByteTensor. model/optimizer
        # load_state_dict both copy onto the existing (already-on-device)
        # tensors, so this doesn't block GPU/MPS training.
        ckpt = torch.load(args.resume, map_location="cpu")
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
        full_val_history = ckpt.get("full_val_history", [])
        lr_history = ckpt.get("lr_history", [])
        print(f"Resumed from {args.resume} at step {start_step}")

    total_steps = start_step + args.steps
    optim_cfg = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "total_steps": total_steps,
        "lr": args.lr,
        "min_lr": min_lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "fixed_batch": args.fixed_batch,
        "eval_batches": args.eval_batches,
        "eval_seed": args.eval_seed,
        "full_eval_interval": args.full_eval_interval,
        "tokens_processed": tokens_processed,
    }
    if args.resume:
        # A resumed run's raw --lr is a schedule parameter, not a rate the run
        # ever uses: the cosine spans [0, total_steps], so a continuation picks
        # it up partway down and starts well below --lr. Recording only `lr`
        # makes a restart's plot footer describe the arithmetic used to set the
        # run up rather than the run itself -- and, worse, look like an
        # ordinary high-LR run. Record what actually happened alongside it.
        optim_cfg["resumed_from"] = Path(args.resume).name
        optim_cfg["resumed_at_step"] = start_step
        if lr_history:  # loaded from the checkpoint, so this is the original peak
            optim_cfg["initial_max_lr"] = max(v for _, v in lr_history)
        optim_cfg["continuation_restart_lr"] = (
            args.restart_lr
            if args.restart_lr is not None
            else lr_at_step(start_step, total_steps, args.lr, min_lr, args.warmup_steps)
        )
        optim_cfg["restart_schedule"] = "own cosine" if args.restart_lr is not None else "original cosine tail"
        optim_cfg["continuation_steps"] = args.steps
    print(f"Optimization: {optim_cfg}")

    model.train()
    for local_step in range(args.steps):
        step = start_step + local_step

        if args.restart_lr is not None:
            # Own schedule for this leg: a full cosine from --restart-lr down to
            # --min-lr across --steps, so the first step runs at exactly
            # --restart-lr. No warmup -- the model is already trained, and a
            # measured 50x restart jump produced no loss spike at all.
            current_lr = lr_at_step(step - start_step, args.steps, args.restart_lr, min_lr, 0)
        else:
            current_lr = lr_at_step(step, total_steps, args.lr, min_lr, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        lr_history.append((step, current_lr))

        x, y = batch if batch is not None else make_batch(
            tokens, args.batch_size, cfg.block_size, device=device, generator=batch_rng
        )

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        is_last_step = local_step == args.steps - 1
        log_now = step % args.log_interval == 0 or is_last_step
        eval_now = step % args.eval_interval == 0 or is_last_step
        full_eval_now = val_tokens is not None and (
            is_last_step or (args.full_eval_interval > 0 and step % args.full_eval_interval == 0)
        )

        if log_now:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {current_lr:.2e}")

        if eval_now:
            eval_train_loss = evaluate_fixed(model, train_eval_batches)
            train_history.append((step, eval_train_loss))
            msg = f"step {step:5d} | eval_train_loss {eval_train_loss:.4f}"
            if val_eval_batches is not None:
                eval_val_loss = evaluate_fixed(model, val_eval_batches)
                val_history.append((step, eval_val_loss))
                msg += f" | eval_val_loss {eval_val_loss:.4f}"
            print(msg)

        if full_eval_now:
            full_val_loss = evaluate_full(model, val_tokens, args.batch_size, cfg.block_size, device)
            full_val_history.append((step, full_val_loss))
            print(f"step {step:5d} | full_val_loss {full_val_loss:.4f} (all {val_tokens.numel():,} val tokens)")

    if args.plot_loss:
        decay_active = min_lr != args.lr or args.warmup_steps > 0
        plot_path = PLOTS_DIR / (
            args.plot_name or default_plot_name(cfg, optim_cfg, total_steps, args.plot_suffix or "")
        )
        plot_loss(
            train_history,
            val_history,
            full_val_history,
            plot_path,
            hyperparams={**cfg.to_dict(), **optim_cfg},
            lr_history=lr_history if decay_active else None,
        )
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
                "full_val_history": full_val_history,
                "lr_history": lr_history,
            },
            save_path,
        )
        print(f"Saved to {save_path}")

    if args.sample_report:
        # Same name as the checkpoint so the two stay paired, whether or not
        # --save was passed (without it, the report is still named for the run).
        report_target = CHECKPOINTS_DIR / (
            args.save_name or default_checkpoint_name(cfg, optim_cfg, total_steps)
        )
        report_file = write_sample_report(
            model,
            tokenizer,
            cfg.block_size,
            device,
            report_target,
            meta={
                "checkpoint": str(report_target) if args.save else "(not saved)",
                "step": total_steps,
                "params": f"{n_params:,}",
                "config": cfg.to_dict(),
                "eval_train_loss": train_history[-1][1] if train_history else None,
                "eval_val_loss": val_history[-1][1] if val_history else None,
                "full_val_loss": full_val_history[-1][1] if full_val_history else None,
            },
        )
        print(f"Saved sample report to {report_file}")

    if args.sample_tokens:
        print(generate_text(model, tokenizer, "\n", args.sample_tokens, cfg.block_size, device))


if __name__ == "__main__":
    main()
