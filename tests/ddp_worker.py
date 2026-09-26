"""Run under torchrun by tests/test_ddp.py -- not a test module itself.

    python -m torch.distributed.run --standalone --nproc_per_node=2 tests/ddp_worker.py <mode> <workdir>

Modes:
  math   DDP's gradient and reduced loss equal a single device's on the
         concatenated global batch, and sharded eval equals unsharded eval.
         Asserts in-process; a failure exits non-zero.
  train  A real (tiny) train.main run. Each rank records every token id its
         training batches touched to <workdir>/seen_rank<r>.json, for the
         test to check the ranks' data is disjoint.
"""

import copy
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import mini_llm.train as train
from mini_llm.config import ModelConfig, build_model
from mini_llm.data import make_batch
from mini_llm.distributed import all_reduce_mean, cleanup_distributed, setup_distributed

VOCAB = 4096  # token ids double as stream positions in `train` mode, so it must cover them


def check_math() -> None:
    info = setup_distributed()
    try:
        torch.manual_seed(0)
        model = build_model(ModelConfig(vocab_size=64, block_size=8, n_embd=16, n_head=2, n_layer=1))
        reference = copy.deepcopy(model)  # a plain single-device copy of the same weights
        ddp = DDP(model)

        tokens = torch.randint(0, 64, (2000,), generator=torch.Generator().manual_seed(123))
        # Same global batch on every rank; each rank trains on its own rows.
        global_x, global_y = make_batch(tokens, 8, 8, generator=torch.Generator().manual_seed(1))
        per = 8 // info.world_size
        rows = slice(info.rank * per, (info.rank + 1) * per)

        _, loss = ddp(global_x[rows], global_y[rows])
        loss.backward()  # all-reduces grads across ranks
        _, ref_loss = reference(global_x, global_y)
        ref_loss.backward()

        reduced = all_reduce_mean(loss.item(), info)
        assert math.isclose(reduced, ref_loss.item(), rel_tol=1e-5), (reduced, ref_loss.item())
        for (name, p), q in zip(model.named_parameters(), reference.parameters()):
            assert p.grad is not None and q.grad is not None, name
            assert torch.allclose(p.grad, q.grad, atol=1e-6), f"grad mismatch in {name}"

        # Sharded eval (collective) == unsharded eval (each rank alone).
        batches = [make_batch(tokens, 8, 8, generator=torch.Generator().manual_seed(s)) for s in range(3)]
        sharded, alone = train.evaluate_fixed(model, batches, info), train.evaluate_fixed(model, batches)
        assert math.isclose(sharded, alone, rel_tol=1e-6), (sharded, alone)
        sharded = train.evaluate_full(model, tokens[:500], 4, 8, "cpu", info)
        alone = train.evaluate_full(model, tokens[:500], 4, 8, "cpu")
        assert math.isclose(sharded, alone, rel_tol=1e-6), (sharded, alone)

        # Why checkpoints save the bare module rather than the wrapper.
        assert all(k.startswith("module.") for k in ddp.state_dict())
        assert not any(k.startswith("module.") for k in model.state_dict())
    finally:
        cleanup_distributed(info)


class StubTokenizer:
    def __len__(self) -> int:
        return VOCAB


def run_train(workdir: Path) -> None:
    os.chdir(workdir)
    rank = int(os.environ["RANK"])
    train.get_tokenizer = lambda: StubTokenizer()

    seen: set[int] = set()
    real_make_batch = train.make_batch

    def recording_make_batch(tokens, batch_size, block_size, device=None, generator=None):
        x, y = real_make_batch(tokens, batch_size, block_size, device=device, generator=generator)
        # Eval batches (global size 8) are sampled from the whole stream on
        # purpose; only the training batches (per-rank size 4) must be disjoint.
        if batch_size == 4:
            seen.update(x.flatten().tolist())
            seen.update(y.flatten().tolist())
        return x, y

    train.make_batch = recording_make_batch
    train.main(sys.argv[3:])
    (workdir / f"seen_rank{rank}.json").write_text(json.dumps(sorted(seen)))


if __name__ == "__main__":
    mode, workdir = sys.argv[1], Path(sys.argv[2])
    if mode == "math":
        check_math()
    elif mode == "train":
        run_train(workdir)
    else:
        raise SystemExit(f"unknown mode {mode!r}")
