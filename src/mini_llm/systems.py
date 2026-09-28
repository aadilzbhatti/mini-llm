"""Training systems metrics: throughput, peak memory, wall-clock.

Loss says how good a model is; these say what it cost. A run records them in
its checkpoint (`systems`) so every row in baselines/evals can be read on the
quality/compute axes together.

    train_tokens_per_sec   tokens trained / seconds spent training. Time inside
                           evaluation (fixed-batch evals, full val passes) is
                           excluded, so a run with a dense eval schedule isn't
                           reported as slower training.
    peak_mem_gb            per process. CUDA: torch.cuda.max_memory_allocated,
                           exact. MPS has no peak counter, so it is the max of
                           samples taken at every log step (a lower bound).
                           CPU: the process's peak resident set size.

GPU work is asynchronous, so eval spans start and stop behind a CUDA
synchronize: otherwise queued training kernels would be billed to eval.
"""

from __future__ import annotations

import resource
import sys
import time
from contextlib import contextmanager

import torch


class SystemsMeter:
    def __init__(self, device: torch.device):
        self.device = device
        self.eval_sec = 0.0
        self.peak_bytes = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self.t0 = time.perf_counter()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def pause(self):
        """Time spent inside this block counts as eval, not training."""
        self._sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.eval_sec += time.perf_counter() - t

    def sample_memory(self) -> None:
        if self.device.type == "mps":
            self.peak_bytes = max(self.peak_bytes, torch.mps.current_allocated_memory())

    def _peak(self) -> tuple[int, str]:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device), "cuda max_memory_allocated"
        if self.device.type == "mps":
            self.sample_memory()
            return self.peak_bytes, "mps allocated, sampled at log steps"
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss if sys.platform == "darwin" else rss * 1024, "cpu peak RSS"  # macOS: bytes, Linux: KiB

    def finish(self, steps: int, tokens_per_step: int, world_size: int = 1) -> dict:
        self._sync()
        wall = time.perf_counter() - self.t0
        train_sec = max(wall - self.eval_sec, 1e-9)
        peak, kind = self._peak()
        name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else self.device.type
        return {
            "device": name,
            "world_size": world_size,
            "steps": steps,
            "tokens": steps * tokens_per_step,
            "wall_sec": round(wall, 1),
            "train_sec": round(train_sec, 1),
            "eval_sec": round(self.eval_sec, 1),
            "train_tokens_per_sec": round(steps * tokens_per_step / train_sec, 1),
            "peak_mem_gb": round(peak / 2**30, 3),
            "peak_mem_kind": kind,
        }
