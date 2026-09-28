"""Control API: checkpoint listing, inference, eval reports."""

import torch
from fastapi.testclient import TestClient

from mini_llm.config import ModelConfig, build_model
from mini_llm.server import create_app
from test_server import repo  # noqa: F401 - fixture


def with_checkpoint(repo):  # noqa: F811
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50257, block_size=16, n_embd=16, n_head=2, n_layer=1)
    (repo / "checkpoints").mkdir()
    torch.save({"config": cfg.to_dict(), "model_state_dict": build_model(cfg).state_dict(), "step": 7,
                "full_val_history": [(6, 9.87)]}, repo / "checkpoints" / "tiny.pt")
    return TestClient(create_app(repo=repo, uv="uv"))


def test_lists_checkpoints_with_config(repo):  # noqa: F811
    [c] = with_checkpoint(repo).get("/api/checkpoints").json()
    assert c["name"] == "tiny.pt" and c["step"] == 7 and c["full_val_loss"] == 9.87 and c["config"]["block_size"] == 16


def test_generate_each_method_and_seeded_reproducibility(repo):  # noqa: F811
    c = with_checkpoint(repo)
    base = {"checkpoint": "tiny.pt", "prompt": "The capital of France is", "max_new_tokens": 12, "stop_at_eos": False}
    for method in ("topk", "sample", "argmax"):
        r = c.post("/api/generate", json={**base, "method": method, "seed": 3})
        assert r.status_code == 200, r.text
        assert r.json()["new_tokens"] == 12 and r.json()["prompt_truncated"] is False
    a = c.post("/api/generate", json={**base, "method": "topk", "seed": 11}).json()["completion"]
    b = c.post("/api/generate", json={**base, "method": "topk", "seed": 11}).json()["completion"]
    assert a == b  # same seed, same text
    long = c.post("/api/generate", json={**base, "prompt": "word " * 40, "method": "argmax"}).json()
    assert long["prompt_truncated"] is True  # 40+ tokens > block_size 16


def test_generate_rejects_bad_input(repo):  # noqa: F811
    c = with_checkpoint(repo)
    for body in ({"checkpoint": "../server.py"}, {"checkpoint": "nope.pt"}, {"checkpoint": "tiny.pt", "method": "beam"},
                 {"checkpoint": "tiny.pt", "max_new_tokens": 5000}, {"checkpoint": "tiny.pt", "temperature": 0}):
        assert c.post("/api/generate", json=body).status_code in (404, 422), body


def test_evals_endpoints(repo):  # noqa: F811
    c = TestClient(create_app(repo=repo, uv="uv"))
    assert c.get("/api/evals").json() == {"summary": None, "reports": []}
    (repo / "evals").mkdir()
    (repo / "evals" / "summary.md").write_text("# Evals summary\n")
    (repo / "evals" / "m1.md").write_text("# Evals: m1\n")
    assert c.get("/api/evals").json() == {"summary": "# Evals summary\n", "reports": ["m1"]}
    assert c.get("/api/evals/m1").text.startswith("# Evals: m1")
    assert c.get("/api/evals/..%2Fsecret").status_code == 404
