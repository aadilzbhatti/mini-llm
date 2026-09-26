"""Multi-process (DDP) plumbing, kept out of train.py so the loop stays readable.

Launch under torchrun and every process gets RANK / LOCAL_RANK / WORLD_SIZE in
its environment:

    torchrun --nproc_per_node=2 -m mini_llm.train --batch-size 64 ...

Without those variables (a plain `uv run mini-llm-train`) nothing here does
anything: setup_distributed() returns a DistInfo with world_size=1 and
enabled=False, and every helper below is the identity. That is what keeps the
single-device MPS/CPU path exactly as it was.

The mental model of DDP this module assumes:
  * N identical processes, one per GPU, each holding a full copy of the model.
  * Each process trains on a DIFFERENT slice of the data (shard_tokens).
  * backward() on the DDP-wrapped model all-reduces (averages) gradients
    across processes, so after backward every rank holds the same gradient --
    the gradient of the mean loss over the whole global batch -- and every
    rank's optimizer.step() makes the same update. The copies never drift.
  * Anything that is NOT part of forward/backward (eval, logging, saving) is
    ours to coordinate, which is what the helpers below are for.
"""

import builtins
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def is_main(self) -> bool:
        """Rank 0 is the only process that writes files or logs."""
        return self.rank == 0


def setup_distributed() -> DistInfo:
    """Join the process group if launched by torchrun; otherwise a no-op."""
    if "WORLD_SIZE" not in os.environ:
        return DistInfo()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])  # index of this process's GPU on this machine
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        # One GPU per process. set_device first, so NCCL (and any bare
        # .cuda() call) uses this rank's GPU rather than piling onto cuda:0.
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        # CPU fallback (tests, a Mac). gloo is the CPU backend; MPS has no
        # collective backend, so DDP on a Mac means CPU.
        device = torch.device("cpu")
        backend = "gloo"
    # torchrun also sets MASTER_ADDR/MASTER_PORT, which the default env://
    # init method reads to find rank 0.
    dist.init_process_group(backend=backend)

    if rank != 0:
        # N processes printing the same log lines is just noise. Same trick as
        # torchvision's reference scripts: silence print() on non-zero ranks,
        # but let print(..., force=True) through for per-rank debugging.
        builtin_print = builtins.print

        def rank_print(*args, force: bool = False, **kwargs):
            if force:
                builtin_print(f"[rank {rank}]", *args, **kwargs)

        builtins.print = rank_print

    return DistInfo(rank, local_rank, world_size, True, device)


def cleanup_distributed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def shard_tokens(tokens: torch.Tensor, info: DistInfo) -> torch.Tensor:
    """This rank's contiguous, non-overlapping slice of the training stream.

    The data here is one long token stream sampled by random crops, not an
    epoch-based Dataset, so there is no DistributedSampler / set_epoch to use.
    The equivalent guarantee (ranks never train on the same tokens) comes from
    giving each rank its own 1/world_size of the stream and letting it crop
    only inside that. Any tail that doesn't divide evenly is dropped
    (< world_size tokens).
    """
    if not info.enabled:
        return tokens
    per_rank = tokens.numel() // info.world_size
    return tokens[info.rank * per_rank : (info.rank + 1) * per_rank]


def all_reduce_sum(values: list[float], info: DistInfo) -> list[float]:
    """Sum a few Python floats across ranks (a collective: every rank must call it).

    float64 so summing per-rank partial losses doesn't lose precision.
    """
    if not info.enabled:
        return values
    t = torch.tensor(values, dtype=torch.float64, device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def all_reduce_mean(value: float, info: DistInfo) -> float:
    """Average one float across ranks, e.g. each rank's batch loss.

    Every rank has the same per-rank batch size, so the mean of per-rank mean
    losses IS the mean loss over the global batch -- the number a single
    device would have printed for the same global batch.
    """
    if not info.enabled:
        return value
    return all_reduce_sum([value], info)[0] / info.world_size


def gather_objects(obj: object, info: DistInfo) -> list[object]:
    """Every rank's `obj`, in rank order (a collective). [obj] when not distributed."""
    if not info.enabled:
        return [obj]
    out: list[object] = [None] * info.world_size
    dist.all_gather_object(out, obj)
    return out
