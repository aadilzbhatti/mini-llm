"""The tiny random checkpoint and the harness runner shared by the infer and decode tests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


# --- the real harness, on a tiny random checkpoint on the CPU ---------------------------------------------------


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from labs.infer.reference import ModelConfig, ModelCustomTransformer

    d = tmp_path_factory.mktemp("tiny")
    cfg = ModelConfig(vocab_size=50257, block_size=1024, n_embd=32, n_head=2, n_layer=2, use_rope_embeddings=True,
                      fused_attention=True)
    torch.manual_seed(0)
    torch.save({"config": cfg.to_dict(), "model_state_dict": ModelCustomTransformer(cfg).state_dict()}, d / "c.pt")
    v = torch.randint(0, 50000, (20_000,))
    v[::400] = 50256  # document boundaries for labs.decode
    torch.save(v, d / "v.pt")
    return d


def run(tiny, module, *args):
    out = tiny / f"{args[0]}-{abs(hash(args))}.json"
    subprocess.run([sys.executable, "-m", module, *args, "--out", str(out)], check=True,
                   env={"LABS_DEVICE": "cpu", "PATH": "", "HF_HUB_OFFLINE": "1", "HOME": str(Path.home())})
    return json.loads(out.read_text())


