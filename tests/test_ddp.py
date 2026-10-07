"""DDP path: the single-process run is untouched, and a 2-rank gloo run on CPU
splits data, gradients and eval the way it should."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.optim import AdamW

import mini_llm.train as train
from mini_llm.config import ModelConfig, build_model
from mini_llm.data import make_batch

WORKER = Path(__file__).with_name("ddp_worker.py")


class StubTokenizer:
    def __len__(self):
        return 64


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(train, "get_tokenizer", lambda: StubTokenizer())
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))  # deterministic
    monkeypatch.setenv("MINI_LLM_RUN_ID", "t1")
    torch.save(torch.randint(0, 64, (4000,), generator=torch.Generator().manual_seed(0)), tmp_path / "train.pt")
    return tmp_path


def test_single_process_matches_plain_loop(workdir):
    """Without torchrun, train.main must be exactly the plain loop:
    same seed -> same batches -> same weights, bit for bit, dropout included."""
    steps, lr = 30, 1e-3
    train.main(
        [
            "--tokens",
            "train.pt",
            "--block-size",
            "8",
            "--n-embd",
            "16",
            "--n-head",
            "2",
            "--n-layer",
            "1",
            "--dropout",
            "0.1",
            "--batch-size",
            "4",
            "--steps",
            str(steps),
            "--lr",
            str(lr),
            "--min-lr",
            str(lr),
            "--warmup-steps",
            "0",
            "--eval-interval",
            "10",
            "--eval-batches",
            "2",
            "--no-tensorboard",
            "--save",
            "--save-name",
            "m.pt",
        ]
    )
    assert not dist.is_initialized()
    ckpt = torch.load(workdir / "checkpoints" / "m.pt", weights_only=False)
    assert "batch_rng_states" not in ckpt  # DDP-only key

    # The loop as it was before DDP: nothing but forward/backward/step.
    tokens = torch.load(workdir / "train.pt")
    torch.manual_seed(42)
    model = build_model(ModelConfig(vocab_size=64, block_size=8, n_embd=16, n_head=2, n_layer=1, dropout=0.1))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = torch.Generator().manual_seed(42)
    model.train()
    for _ in range(steps):
        x, y = make_batch(tokens, 4, 8, generator=rng)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for k, v in model.state_dict().items():
        assert torch.equal(ckpt["model_state_dict"][k], v), k


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def torchrun(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}}
    env["MINI_LLM_RUN_ID"] = "ddp"
    # Explicit loopback rendezvous rather than --standalone: on macOS,
    # --standalone resolves the hostname to an ip6.arpa name getaddrinfo
    # can't resolve, and the ranks hang waiting for each other.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master-addr",
            "127.0.0.1",
            "--master-port",
            str(free_port()),
            str(WORKER),
            *args,
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"torchrun failed\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr}"
    return proc


def test_ddp_gradients_loss_and_eval_match_single_device(tmp_path):
    torchrun("math", str(tmp_path), cwd=tmp_path)


def test_ddp_training_run(tmp_path):
    n_tokens = 4000
    torch.save(torch.arange(n_tokens), tmp_path / "train.pt")  # token id == position in the stream
    torch.save(torch.arange(n_tokens, 4096), tmp_path / "val.pt")  # disjoint from train, within the vocab
    proc = torchrun(
        "train",
        str(tmp_path),
        "--tokens",
        "train.pt",
        "--val-tokens",
        "val.pt",
        "--block-size",
        "8",
        "--n-embd",
        "16",
        "--n-head",
        "2",
        "--n-layer",
        "1",
        "--batch-size",
        "8",
        "--steps",
        "20",
        "--warmup-steps",
        "0",
        "--eval-interval",
        "10",
        "--eval-batches",
        "2",
        "--full-eval-interval",
        "0",
        "--save",
        "--save-name",
        "m.pt",
        "--plot-loss",
        "--plot-name",
        "p.png",
        cwd=tmp_path,
    )

    # Disjoint data: rank r only ever touched tokens from its own half.
    seen = [set(json.loads((tmp_path / f"seen_rank{r}.json").read_text())) for r in range(2)]
    assert seen[0] and seen[1]
    assert seen[0].isdisjoint(seen[1])
    assert max(seen[0]) < n_tokens // 2 <= min(seen[1])

    # Only rank 0 logs: each line once, not twice.
    assert proc.stdout.count("Saved to checkpoints/m.pt") == 1
    assert "world_size=2" in proc.stdout

    # Checkpoint loads into a plain (non-DDP) model, e.g. on the Mac.
    ckpt = torch.load(tmp_path / "checkpoints" / "m.pt", weights_only=False)
    assert not any(k.startswith("module.") for k in ckpt["model_state_dict"])
    build_model(ModelConfig.from_dict(ckpt["config"])).load_state_dict(ckpt["model_state_dict"])
    assert len(ckpt["batch_rng_states"]) == 2
    assert ckpt["step"] == 20
    assert [s for s, _ in ckpt["train_history"]] == [0, 10, 19]
    assert [s for s, _ in ckpt["full_val_history"]] == [19]  # end-of-training full eval
    assert (tmp_path / "plots" / "p.png").exists()
