"""Minimal training loop: forward -> loss -> zero_grad -> backward -> step.

No AMP (unless --bf16 on CUDA), no grad accumulation, no clipping. Add those
back deliberately when you want them.

LR schedule: linear warmup (--warmup-steps, default 500) then cosine decay
from --lr (default 1e-3) down to --min-lr (default 2e-6), over this run's
total step horizon (start_step + --steps, so a --resume decays across its
own new horizon rather than the original run's). --min-lr is the actual
decay knob -- set it equal to --lr to disable decay and train at a constant
rate; --warmup-steps 0 disables warmup.

--warmup-tokens N sets the warmup in tokens instead: it becomes
ceil(N / (batch_size * block_size)) steps. Use it when comparing batch
sizes, so every run warms up over the same data. At an equal token budget
that is also the same fraction of the run, whereas a fixed --warmup-steps
covers 16x the tokens at 16x the batch, and a bigger share of a run that
now has fewer steps.

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

--baseline upserts this run into baselines.md (see mini_llm.baselines),
keyed by run name (--save-name/--plot-name, or the generated hyperparam
name if neither is set) so a --resume continuation replaces its own
earlier row rather than duplicating it. Re-sorted by best loss on every
write, so the table always reads best-run-first.

Live control and TensorBoard (see mini_llm.control): every run gets a run id
(MINI_LLM_RUN_ID from the queue runner, else a timestamp + hyperparam slug).
Scalars go to TensorBoard under --tensorboard-dir/<run id>, and every
--control-poll steps the loop drains runs/<run id>.commands.jsonl, which can
scale the LR, change eval/log cadence, pause, force an eval or checkpoint, or
stop early. Commands only ever apply between steps, and each one is logged
with its step to runs/<run id>.events.jsonl and to TensorBoard, so a steered
run is still reproducible from its launch args plus its events.

Multi-GPU (DDP, see mini_llm.distributed): launched under torchrun, each
process trains a full model copy on its own shard of --tokens, with
--batch-size meaning the GLOBAL batch (each rank takes batch_size /
world_size of it) so the same flags mean the same optimization on 1 or N
devices. Eval uses the same fixed batches as a single device, split across
ranks and summed back, so eval numbers stay comparable with local runs. Only
rank 0 prints, writes TensorBoard, checkpoints, plots and reports. Live
control is off under DDP. --bf16 turns on bf16 autocast for the training
forward pass (CUDA only). Without torchrun none of this applies and the run
is exactly the single-device loop.
"""

import argparse
import contextlib
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from mini_llm.baselines import update_baselines
from mini_llm.config import ModelConfig, build_model
from mini_llm.control import ControlState, RunControl
from mini_llm.data import encode, fixed_batch, get_tokenizer, load_text, load_tokens, make_batch
from mini_llm.device import select_device
from mini_llm.distributed import (
    DistInfo,
    all_reduce_mean,
    all_reduce_sum,
    cleanup_distributed,
    gather_objects,
    setup_distributed,
    shard_tokens,
)
from mini_llm.generate import generate_text
from mini_llm.report import write_sample_report
from mini_llm.systems import SystemsMeter

PLOTS_DIR = Path("plots")
CHECKPOINTS_DIR = Path("checkpoints")
RUNS_DIR = Path("runs")
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
def evaluate_fixed(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    info: DistInfo = DistInfo(),
) -> float:
    """Average loss over a fixed, pre-sampled set of batches.

    Reusing the same batches at every checkpoint means successive
    evaluations measure how the model changed, not how the sample did.

    Under DDP this is a collective: every rank must call it.
    """
    model.eval()
    total_loss = 0.0
    for x, y in batches:
        weight = 1.0
        if info.enabled:
            # DDP: every rank holds the SAME fixed batches (same --eval-seed,
            # same global --batch-size) and evaluates only its own contiguous
            # block of rows of each. A batch's loss is a mean over its rows,
            # so weighting this rank's mean by its share of the rows, then
            # summing across ranks below, rebuilds exactly the full-batch
            # mean a single device would get. Same measuring stick, split N
            # ways. (A contiguous block, not rows rank::N: the model .view()s
            # its targets, which fails on a strided slice.)
            n_rows = x.shape[0]
            x = torch.tensor_split(x, info.world_size)[info.rank]
            y = torch.tensor_split(y, info.world_size)[info.rank]
            weight = x.shape[0] / n_rows
            if weight == 0:  # fewer rows than ranks: nothing for this rank
                continue
        _, loss = model(x, y)
        total_loss += loss.item() * weight
    (total_loss,) = all_reduce_sum([total_loss], info)
    model.train()
    return total_loss / len(batches)


@torch.no_grad()
def evaluate_full(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str,
    info: DistInfo = DistInfo(),
) -> float:
    """Average loss over every non-overlapping window in `tokens`.

    Exhaustive, not sampled -- the true loss over the whole set, at the
    cost of a full pass. Meant to run infrequently (--full-eval-interval),
    as a periodic sanity check against the cheaper fixed-sample estimate.

    Under DDP this is a collective: every rank must call it.
    """
    n_windows = (tokens.numel() - 1) // block_size
    if n_windows == 0:
        raise ValueError(
            f"need at least block_size + 1 = {block_size + 1} tokens, got {tokens.numel()}"
        )

    model.eval()
    total_loss, total_windows = 0.0, 0
    # DDP: chunks are dealt round-robin (rank r takes chunks r, r+N, r+2N, ...)
    # so the N ranks cover every window exactly once between them. The
    # (loss sum, window count) pairs are summed across ranks below, so the
    # result is the same exhaustive average as on one device.
    for start in range(info.rank * batch_size, n_windows, batch_size * info.world_size):
        offsets = [i * block_size for i in range(start, min(start + batch_size, n_windows))]
        x = torch.stack([tokens[o : o + block_size] for o in offsets]).to(device)
        y = torch.stack([tokens[o + 1 : o + block_size + 1] for o in offsets]).to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * len(offsets)
        total_windows += len(offsets)
    total_loss, total_windows = all_reduce_sum([total_loss, total_windows], info)
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


def save_checkpoint(
    path: Path,
    cfg: ModelConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_rng: torch.Generator,
    step: int,
    train_history: list[tuple[int, float]],
    val_history: list[tuple[int, float]],
    full_val_history: list[tuple[int, float]],
    lr_history: list[tuple[int, float]],
    batch_rng_states: list[object] | None = None,
    systems: dict | None = None,
) -> Path:
    """`model` must be the bare module, never the DDP wrapper: DDP's own
    state_dict prefixes every key with "module.", which a plain
    ModelCustomTransformer (e.g. on the Mac) then refuses to load.

    `batch_rng_states` (DDP only) holds every rank's batch generator, in rank
    order, so a resume at the same world size continues each rank's own crop
    sequence. `batch_rng_state` stays rank 0's, so a single device can still
    resume from a DDP checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"batch_rng_states": batch_rng_states} if batch_rng_states is not None else {}
    if systems is not None:
        extra["systems"] = systems
    torch.save(
        {
            "config": cfg.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "batch_rng_state": batch_rng.get_state(),
            "step": step,
            "train_history": train_history,
            "val_history": val_history,
            "full_val_history": full_val_history,
            "lr_history": lr_history,
            **extra,
        },
        path,
    )
    return path


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


def build_parser() -> argparse.ArgumentParser:
    """The CLI definition, separate from parsing so the control API can read
    every flag's default and help text to build its job form."""
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
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Global batch size per optimizer step. Under torchrun each of the N ranks "
        "takes batch_size / N of it, so it must divide evenly.",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="bf16 autocast for the training forward pass. CUDA only; ignored (with a "
        "notice) on MPS/CPU. Eval stays fp32 so eval losses remain comparable across runs.",
    )
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
    p.add_argument(
        "--warmup-tokens",
        type=int,
        default=None,
        help="Warmup length in tokens instead of steps: ceil(N / (batch-size * block-size)) "
        "steps. Keeps warmup comparable across batch sizes. Can't be combined with "
        "--warmup-steps.",
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
        "--sample-report-tokens",
        type=int,
        default=None,
        help="Override the sample report's token budget. Default is 2 x --block-size.",
    )
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
    p.add_argument(
        "--tensorboard-dir",
        default="runs/tb",
        help="TensorBoard log root; this run logs to <dir>/<run id>. Point `tensorboard "
        "--logdir` at the root to compare runs.",
    )
    p.add_argument("--no-tensorboard", action="store_true", help="Skip TensorBoard logging.")
    p.add_argument(
        "--control-poll",
        type=int,
        default=25,
        help="Check runs/<run id>.commands.jsonl for live commands every N steps "
        "(see mini_llm.control). 0 disables live control.",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Upsert this run into baselines.md (and baselines.json next to it) at the end "
        "of training, keyed by --save-name/--plot-name (or the generated hyperparam name if "
        "neither is set) and re-sorted by best loss.",
    )
    return p


def parse_args(argv: list[str] | None = None):
    return build_parser().parse_args(argv)


class _SilentControl(RunControl):
    """RunControl for DDP ranks other than 0: the same state and interface, so
    the loop needs no rank checks, but it writes no files, heartbeats or
    TensorBoard events (its _tb stays None, so scalar/text/close are no-ops)."""

    def __post_init__(self) -> None:
        self.paths = {}

    def heartbeat(self, *args: object, **kwargs: object) -> None:
        pass


def resolve_warmup(args: argparse.Namespace, argv: list[str] | None = None) -> argparse.Namespace:
    """Turn --warmup-tokens into args.warmup_steps (in place). The schedule
    itself only ever sees steps; this is the one place tokens get converted."""
    if args.warmup_tokens is None:
        return args
    raw = sys.argv[1:] if argv is None else argv
    if any(a == "--warmup-steps" or a.startswith("--warmup-steps=") for a in raw):
        raise SystemExit("Pass --warmup-steps or --warmup-tokens, not both.")
    if args.warmup_tokens < 0:
        raise SystemExit("--warmup-tokens must be >= 0.")
    # Global batch: under DDP every rank takes a step together, so one step is
    # batch_size * block_size tokens however many ranks share it.
    args.warmup_steps = math.ceil(args.warmup_tokens / (args.batch_size * args.block_size))
    return args


def main(argv: list[str] | None = None) -> None:
    args = resolve_warmup(parse_args(argv), argv)
    # Under torchrun: join the process group. Otherwise a no-op, and
    # dist_info.enabled is False everywhere below.
    dist_info = setup_distributed()
    try:
        run_training(args, dist_info)
    finally:
        # Always leave the process group, even on an exception, so NCCL
        # doesn't warn about a leaked group at exit.
        cleanup_distributed(dist_info)


def run_training(args: argparse.Namespace, dist_info: DistInfo) -> None:
    torch.manual_seed(args.seed)

    if args.restart_lr is not None and not args.resume:
        raise SystemExit("--restart-lr only means something with --resume; there is no run to restart.")

    # DDP: the device is fixed by LOCAL_RANK (cuda:N, or cpu under gloo), not
    # picked. Single device: the usual MPS -> CUDA -> CPU choice.
    device = dist_info.device if dist_info.enabled else select_device()
    print(f"Using device: {device}")
    if dist_info.enabled:
        print(f"DDP: world_size={dist_info.world_size}, backend={torch.distributed.get_backend()}")
        if args.batch_size % dist_info.world_size:
            raise SystemExit(
                f"--batch-size {args.batch_size} is the global batch and must divide evenly "
                f"across {dist_info.world_size} ranks."
            )
    # --batch-size is the global batch. DDP averages gradients across ranks,
    # so N ranks x (batch_size / N) rows gives the same gradient as one device
    # with batch_size rows: same LR, same schedule, just faster.
    per_rank_batch = args.batch_size // dist_info.world_size

    use_bf16 = args.bf16 and device.type == "cuda"
    if args.bf16 and not use_bf16:
        print(f"--bf16 ignored: bf16 autocast is only enabled on CUDA, and this run is on {device.type}.")
    # bf16 has fp32's exponent range, so unlike fp16 it needs no GradScaler.
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext

    tokenizer = get_tokenizer()
    tokens = load_tokens(args.tokens) if args.tokens else encode(load_text(args.text), tokenizer)
    val_tokens = load_tokens(args.val_tokens) if args.val_tokens else None
    if val_tokens is not None and tokens.numel() == val_tokens.numel() and torch.equal(tokens, val_tokens):
        # Training on the val set makes every val loss a memorization score
        # (it has happened: a run passed val.pt as --tokens and "scored" 0.35).
        # Caught by content, not path, so a copy under another name is refused too.
        raise SystemExit("--tokens and --val-tokens contain identical data: refusing to train on the validation set.")
    # DDP: each rank trains only on its own contiguous 1/N of the stream, so
    # no two ranks ever see the same training tokens. `tokens` itself stays
    # whole: the fixed eval batches are sampled from it identically on every
    # rank. Single device: train_tokens is just tokens.
    train_tokens = shard_tokens(tokens, dist_info)

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
    if dist_info.enabled:
        # Every rank seeded the same above, so every rank built the same
        # initial weights (DDP would broadcast rank 0's anyway). From here on,
        # give each rank its own RNG stream, or all ranks would draw identical
        # dropout masks for their different rows.
        torch.manual_seed(args.seed + dist_info.rank)
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
        if dist_info.enabled:
            # DDP: the fixed batch is the global one; each rank trains on its
            # own contiguous block of per_rank_batch rows of it.
            lo = dist_info.rank * per_rank_batch
            batch = (batch[0][lo : lo + per_rank_batch], batch[1][lo : lo + per_rank_batch])
        print("Training on one fixed batch.")

    # Dedicated generator for the resampled-batch case: the batch sequence
    # then depends only on --seed, not on how many random draws model init
    # (or anything else before the loop) happened to consume.
    # DDP: seed + rank, so each rank draws its own crop positions. Rank 0
    # keeps plain --seed, which is also the single-device stream.
    batch_rng = torch.Generator().manual_seed(args.seed + dist_info.rank)

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
        rng_states = ckpt.get("batch_rng_states")
        if dist_info.enabled and rng_states is not None and len(rng_states) == dist_info.world_size:
            # DDP resume at the same world size: each rank picks up its own stream.
            batch_rng.set_state(rng_states[dist_info.rank])
        elif dist_info.rank == 0:
            batch_rng.set_state(ckpt["batch_rng_state"])
        else:
            # The checkpoint came from a different world size (e.g. a Mac run),
            # so there's no saved stream for this rank: it keeps its fresh
            # seed + rank one.
            print("resume: no saved batch generator for this rank; using a fresh one", force=True)  # pyright: ignore[reportCallIssue]
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
    if args.warmup_tokens is not None:
        # What was asked for; warmup_steps above is what it became.
        optim_cfg["warmup_tokens"] = args.warmup_tokens
    # Recorded only when they apply, so single-device runs log exactly as before.
    if dist_info.enabled:
        optim_cfg["world_size"] = dist_info.world_size
        optim_cfg["per_rank_batch_size"] = per_rank_batch
    if use_bf16:
        optim_cfg["bf16"] = True
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

    run_id = os.environ.get("MINI_LLM_RUN_ID") or (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{hyperparam_slug(cfg, optim_cfg, total_steps)}"
    )
    print(f"Run id: {run_id}")
    control_poll = args.control_poll
    if dist_info.enabled and control_poll > 0:
        # Live control is single-process only. A command that rank 0 alone
        # read (stop, pause, a new eval_interval) would send the ranks down
        # different code paths, and the next collective would hang forever.
        # Making it safe means broadcasting ControlState from rank 0 at every
        # poll. Not worth it for remote runs nobody can reach mid-flight.
        print("Live control disabled under DDP.")
        control_poll = 0
    # Only rank 0 writes heartbeat/events files and TensorBoard.
    control = (RunControl if dist_info.is_main else _SilentControl)(
        run_id=run_id,
        runs_dir=RUNS_DIR,
        state=ControlState(
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            full_eval_interval=args.full_eval_interval,
        ),
        poll_every=max(args.control_poll, 1),
        tensorboard_dir=None if args.no_tensorboard else Path(args.tensorboard_dir) / run_id,
    )
    ctl = control.state
    control.text(
        "config",
        "\n".join(f"    {k}: {v}" for k, v in {**cfg.to_dict(), **optim_cfg, "run_id": run_id}.items()),
        start_step,
    )
    control.heartbeat(start_step, total_steps, force=True)
    stopped_at: int | None = None

    # DDP wraps the model for the TRAINING forward/backward only. Wrapping
    # broadcasts rank 0's weights so every replica starts identical, then
    # hooks backward() to all-reduce (average) gradients across ranks, so
    # every rank's optimizer.step() makes the same update. `model` stays the
    # bare module (it is train_model.module): eval, generation and checkpoints
    # use it directly, so they trigger no DDP collectives, and saved
    # state_dicts have no "module." key prefix.
    train_model = (
        DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
        if dist_info.enabled
        else model
    )

    # Throughput / memory / wall-clock for this run (see mini_llm.systems).
    meter = SystemsMeter(device)
    model.train()
    for local_step in range(args.steps):
        step = start_step + local_step

        if control_poll > 0:
            control.poll(step)
            if ctl.paused:
                print(f"step {step:5d} | paused", flush=True)
                control.wait_while_paused(step)
                print(f"step {step:5d} | resumed", flush=True)

        if args.restart_lr is not None:
            # Own schedule for this leg: a full cosine from --restart-lr down to
            # --min-lr across --steps, so the first step runs at exactly
            # --restart-lr. No warmup -- the model is already trained, and a
            # measured 50x restart jump produced no loss spike at all.
            current_lr = lr_at_step(step - start_step, args.steps, args.restart_lr, min_lr, 0)
        else:
            current_lr = lr_at_step(step, total_steps, args.lr, min_lr, args.warmup_steps)
        # Live LR scaling (control knob). 1.0 unless someone changed it mid-run;
        # recorded in lr_history, so the plot shows the LR that actually ran.
        current_lr *= ctl.lr_scale
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        lr_history.append((step, current_lr))

        x, y = batch if batch is not None else make_batch(
            train_tokens, per_rank_batch, cfg.block_size, device=device, generator=batch_rng
        )

        # Autocast wraps the forward pass only. backward() then runs each op
        # in whatever dtype its forward used.
        with autocast():
            _, loss = train_model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # DDP: gradients are all-reduced across ranks in here
        optimizer.step()

        # A stop command ends the run *here*, as if this had been the last
        # step, so the final eval/plot/save/report all still happen.
        is_last_step = local_step == args.steps - 1 or ctl.stop
        forced_eval = ctl.eval_now
        ctl.eval_now = False
        log_now = step % ctl.log_interval == 0 or is_last_step
        eval_now = step % ctl.eval_interval == 0 or is_last_step or forced_eval
        full_eval_now = val_tokens is not None and (
            is_last_step
            or forced_eval
            or (ctl.full_eval_interval > 0 and step % ctl.full_eval_interval == 0)
        )

        if log_now:
            # DDP: each rank's loss covers only its rows of the batch, so
            # average across ranks to log the global-batch loss. This is a
            # collective, which is safe because log_now comes out the same on
            # every rank (same step, same intervals, live control off).
            loss_value = all_reduce_mean(loss.item(), dist_info)
            print(f"step {step:5d} | loss {loss_value:.4f} | lr {current_lr:.2e}")
            meter.sample_memory()
            control.scalar("train/batch_loss", loss_value, step)
            control.scalar("train/lr", current_lr, step)

        # Evaluation time is excluded from training throughput. Only paused on
        # eval steps: pause() synchronizes CUDA, which would stall every step.
        with meter.pause() if (eval_now or full_eval_now) else contextlib.nullcontext():
            if eval_now:
                eval_train_loss = evaluate_fixed(model, train_eval_batches, dist_info)
                train_history.append((step, eval_train_loss))
                msg = f"step {step:5d} | eval_train_loss {eval_train_loss:.4f}"
                if val_eval_batches is not None:
                    eval_val_loss = evaluate_fixed(model, val_eval_batches, dist_info)
                    val_history.append((step, eval_val_loss))
                    msg += f" | eval_val_loss {eval_val_loss:.4f}"
                    control.scalar("eval/val_loss", eval_val_loss, step)
                control.scalar("eval/train_loss", eval_train_loss, step)
                print(msg)

            if full_eval_now:
                # Chunk size doesn't change the (exhaustive) result, only memory,
                # so each rank uses its training batch size. Single device:
                # per_rank_batch == --batch-size.
                full_val_loss = evaluate_full(model, val_tokens, per_rank_batch, cfg.block_size, device, dist_info)
                full_val_history.append((step, full_val_loss))
                print(f"step {step:5d} | full_val_loss {full_val_loss:.4f} (all {val_tokens.numel():,} val tokens)")
                control.scalar("eval/full_val_loss", full_val_loss, step)

        if ctl.checkpoint_now:
            ctl.checkpoint_now = False
            stem = Path(args.save_name).stem if args.save_name else f"ckpt_{run_id}"
            mid_path = save_checkpoint(
                CHECKPOINTS_DIR / f"{stem}.step{step + 1}.pt", cfg, model, optimizer, batch_rng,
                step + 1, train_history, val_history, full_val_history, lr_history,
            )
            print(f"step {step:5d} | saved mid-run checkpoint to {mid_path}", flush=True)
            control.text("control/events", f"step {step}: checkpoint -> {mid_path}", step)

        if log_now or eval_now or step % control.poll_every == 0:
            control.heartbeat(
                step + 1,
                total_steps,
                extra={
                    "lr": current_lr,
                    "eval_train_loss": train_history[-1][1] if train_history else None,
                    "eval_val_loss": val_history[-1][1] if val_history else None,
                    "full_val_loss": full_val_history[-1][1] if full_val_history else None,
                },
            )

        if ctl.stop:
            stopped_at = step
            print(f"step {step:5d} | stopped early by control command", flush=True)
            break

    steps_done = (stopped_at - start_step + 1) if stopped_at is not None else args.steps
    systems = meter.finish(steps_done, args.batch_size * cfg.block_size, dist_info.world_size)

    if stopped_at is not None:
        # Name and record the run by the training it actually got, not the
        # horizon it was launched with.
        total_steps = stopped_at + 1
        optim_cfg["total_steps"] = total_steps
        optim_cfg["stopped_early_at"] = total_steps

    # DDP: everything from here on writes files, so it's rank 0 only. The
    # one thing rank 0 needs from the others is their batch generator states
    # for the checkpoint, and gathering them is a collective, so every rank
    # takes part before the others leave.
    batch_rng_states = (
        gather_objects(batch_rng.get_state(), dist_info) if args.save and dist_info.enabled else None
    )
    if not dist_info.is_main:
        return
    print(
        f"Systems: {systems['train_tokens_per_sec']:,.0f} train tokens/s on {systems['world_size']}x "
        f"{systems['device']} | peak memory {systems['peak_mem_gb']:.2f} GB per process "
        f"({systems['peak_mem_kind']}) | wall {systems['wall_sec'] / 60:.1f} min "
        f"(train {systems['train_sec'] / 60:.1f}, eval {systems['eval_sec'] / 60:.1f})"
    )

    plot_path: Path | None = None
    save_path: Path | None = None

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
        save_path = save_checkpoint(
            CHECKPOINTS_DIR / (args.save_name or default_checkpoint_name(cfg, optim_cfg, total_steps)),
            cfg, model, optimizer, batch_rng, total_steps,
            train_history, val_history, full_val_history, lr_history,
            batch_rng_states=batch_rng_states,
            systems=systems,
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
            max_new_tokens=args.sample_report_tokens,
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
        try:
            control.text("sample_report", Path(report_file).read_text(), total_steps)
        except OSError:
            pass

    if args.baseline:
        run_name = args.save_name or args.plot_name or default_checkpoint_name(cfg, optim_cfg, total_steps)
        baselines_path = update_baselines(
            {
                "run": run_name,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "steps": total_steps,
                "params": n_params,
                "n_layer": cfg.n_layer,
                "n_embd": cfg.n_embd,
                "n_head": cfg.n_head,
                "block_size": cfg.block_size,
                "batch_size": optim_cfg["batch_size"],
                "lr": optim_cfg["lr"],
                "min_lr": optim_cfg["min_lr"],
                "warmup_steps": optim_cfg["warmup_steps"],
                "seed": optim_cfg["seed"],
                "eval_train_loss": train_history[-1][1] if train_history else None,
                "eval_val_loss": val_history[-1][1] if val_history else None,
                "full_val_loss": full_val_history[-1][1] if full_val_history else None,
                "checkpoint": str(save_path) if save_path is not None else None,
                "plot": str(plot_path) if plot_path is not None else None,
            }
        )
        print(f"Updated {baselines_path}")

    if args.sample_tokens:
        print(generate_text(model, tokenizer, "\n", args.sample_tokens, cfg.block_size, device))

    if control.tb is not None:
        hparams = {
            k: v for k, v in {**cfg.to_dict(), **optim_cfg}.items()
            if isinstance(v, (int, float, str, bool))
        }
        metrics = {
            "hparam/eval_train_loss": train_history[-1][1] if train_history else float("nan"),
            "hparam/eval_val_loss": val_history[-1][1] if val_history else float("nan"),
            "hparam/full_val_loss": full_val_history[-1][1] if full_val_history else float("nan"),
        }
        control.tb.add_hparams(hparams, metrics, run_name=".")
    control.heartbeat(total_steps, total_steps, extra={"finished": True}, force=True)
    control.close()


if __name__ == "__main__":
    main()
