"""PROTECTED: train.main (single process) is exactly the plain training loop, bit for bit.

Same intent as tests/test_ddp.py::test_single_process_matches_plain_loop (the DDP plumbing must
not change single-process training), but the reference loop uses the program's own evolvable
pieces -- build_optimizer, before_optimizer_step and lr_at_step -- instead of hard-coding plain
AdamW. The evaluation cascade runs this and deselects the original, which would otherwise reject
every optimizer change the search is allowed to make ([evolve] cpu_test_deselect).
"""

import pytest
import torch
import torch.distributed as dist

import mini_llm.train as train
from mini_llm.config import ModelConfig, build_model
from mini_llm.data import make_batch


class StubTokenizer:
    def __len__(self):
        return 64


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(train, "get_tokenizer", lambda: StubTokenizer())
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setenv("MINI_LLM_RUN_ID", "t1")
    torch.save(torch.randint(0, 64, (4000,), generator=torch.Generator().manual_seed(0)), tmp_path / "train.pt")
    return tmp_path


def test_single_process_matches_plain_loop_with_program_optimizer(workdir):
    steps, lr, min_lr, warmup = 30, 1e-3, 1e-4, 5
    argv = ["--tokens", "train.pt", "--block-size", "8", "--n-embd", "16", "--n-head", "2", "--n-layer", "1",
            "--dropout", "0.1", "--batch-size", "4", "--steps", str(steps), "--lr", str(lr), "--min-lr", str(min_lr),
            "--warmup-steps", str(warmup), "--weight-decay", "0.01", "--eval-interval", "10", "--eval-batches", "2",
            "--no-tensorboard", "--save", "--save-name", "m.pt"]
    train.main(argv)
    assert not dist.is_initialized()
    ckpt = torch.load(workdir / "checkpoints" / "m.pt", weights_only=False)

    args = train.parse_args(argv)
    tokens = torch.load(workdir / "train.pt")
    torch.manual_seed(42)
    model = build_model(ModelConfig(vocab_size=64, block_size=8, n_embd=16, n_head=2, n_layer=1, dropout=0.1))
    optimizer = train.build_optimizer(model, args)
    rng = torch.Generator().manual_seed(42)
    model.train()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = train.lr_at_step(step, steps, lr, min_lr, warmup)
        x, y = make_batch(tokens, 4, 8, generator=rng)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        train.before_optimizer_step(model, optimizer, step)
        optimizer.step()

    for k, v in model.state_dict().items():
        assert torch.equal(ckpt["model_state_dict"][k], v), k
