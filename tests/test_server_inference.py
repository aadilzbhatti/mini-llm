"""Control API: checkpoint listing, inference, eval reports."""

import json

import torch
from fastapi.testclient import TestClient

from mini_llm.config import ModelConfig, build_model
from mini_llm.server import create_app
from test_server import repo  # noqa: F401 - fixture


def with_checkpoint(repo):  # noqa: F811
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50257, block_size=16, n_embd=16, n_head=2, n_layer=1)
    (repo / "checkpoints").mkdir()
    torch.save(
        {
            "config": cfg.to_dict(),
            "model_state_dict": build_model(cfg).state_dict(),
            "step": 7,
            "full_val_history": [(6, 9.87)],
        },
        repo / "checkpoints" / "tiny.pt",
    )
    return TestClient(create_app(repo=repo, uv="uv"))


def test_lists_checkpoints_with_config(repo):  # noqa: F811
    [c] = with_checkpoint(repo).get("/api/checkpoints").json()
    assert c["name"] == "tiny.pt" and c["step"] == 7 and c["full_val_loss"] == 9.87 and c["config"]["block_size"] == 16


def test_generate_settings_and_seeded_reproducibility(repo):  # noqa: F811
    c = with_checkpoint(repo)
    base = {"checkpoint": "tiny.pt", "prompt": "The capital of France is", "max_new_tokens": 12, "stop_at_eos": False}
    for knobs in ({}, {"temperature": 0}, {"top_k": 0}, {"top_p": 0.9, "top_k": 0}, {"temperature": 1.5, "top_k": 200}):
        r = c.post("/api/generate", json={**base, **knobs, "seed": 3})
        assert r.status_code == 200, (knobs, r.text)
        assert r.json()["new_tokens"] == 12 and r.json()["prompt_truncated"] is False
    greedy = c.post("/api/generate", json={**base, "temperature": 0}).json()
    assert greedy["greedy"] and greedy["top_k"] is None and greedy["top_p"] is None
    a = c.post("/api/generate", json={**base, "seed": 11}).json()
    b = c.post("/api/generate", json={**base, "seed": 11}).json()
    assert a["completion"] == b["completion"] and (a["temperature"], a["top_k"], a["top_p"]) == (0.7, 40, None)
    long = c.post("/api/generate", json={**base, "prompt": "word " * 40, "temperature": 0}).json()
    assert long["prompt_truncated"] is True  # 40+ tokens > block_size 16


def test_generate_rejects_bad_input(repo):  # noqa: F811
    c = with_checkpoint(repo)
    t = {"checkpoint": "tiny.pt"}
    for body in (
        {"checkpoint": "../server.py"},
        {"checkpoint": "nope.pt"},
        {**t, "max_new_tokens": 5000},
        {**t, "max_new_tokens": 2.5},
        {**t, "temperature": 2},
        {**t, "temperature": -0.1},
        {**t, "top_k": 201},
        {**t, "top_k": 4.5},
        {**t, "top_p": 0},
        {**t, "top_p": 1.2},
        {**t, "top_p": "x"},
    ):
        assert c.post("/api/generate", json=body).status_code in (404, 422), body


def test_next_token_returns_all_logits(repo):  # noqa: F811
    import base64
    import numpy as np

    c = with_checkpoint(repo)
    r = c.post("/api/next_token", json={"checkpoint": "tiny.pt", "prompt": "The capital of France is"}).json()
    z = np.frombuffer(base64.b64decode(r["logits"]), dtype="<f4")
    assert z.size == 50257 and len(r["top"]) == 200 and r["block_size"] == 16
    assert r["top"][0]["id"] == int(z.argmax()) and isinstance(r["top"][0]["text"], str)
    assert c.get("/api/meta").json()["sampling"]["top_p"] == {"min": 0.05, "max": 1.0, "default": 1.0}


def test_nucleus_keeps_smallest_set_reaching_p():
    from mini_llm.report import nucleus

    p = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    assert torch.allclose(nucleus(p, 0.8), torch.tensor([[0.5 / 0.8, 0.3 / 0.8, 0, 0]]))
    assert torch.allclose(nucleus(p, 0.81), torch.tensor([[0.5, 0.3, 0.15, 0]]) / 0.95)
    assert torch.allclose(nucleus(p, 0.05), torch.tensor([[1.0, 0, 0, 0]]))  # always keeps the top token


def test_evals_endpoints(repo):  # noqa: F811
    c = TestClient(create_app(repo=repo, uv="uv"))
    assert c.get("/api/evals").json() == {"summary": None, "guide": None, "shared": [], "reports": []}
    (repo / "evals").mkdir()
    for n, t in (
        ("summary", "# Evals summary\n"),
        ("GUIDE", "# Guide\n"),
        ("samples", "# Samples\n"),
        ("m1", "# Evals: m1\n"),
    ):
        (repo / "evals" / f"{n}.md").write_text(t)
    (repo / "evals" / "m2_data80k.md").write_text("# Evals: m2\n")
    (repo / "evals" / "m2_data80k.json").write_text(
        json.dumps(
            {
                "checkpoint": "checkpoints/m2_data80k.pt",
                "config": {"block_size": 256, "n_embd": 256, "n_layer": 4},
                "params": 16_000_000,
                "step": 15000,
                "quality": {"full_val@256": 4.5},
            }
        )
    )
    j = c.get("/api/evals").json()
    assert j["summary"] == "# Evals summary\n" and j["guide"] == "# Guide\n" and j["shared"] == ["samples"]
    rows = {r["name"]: r for r in j["reports"]}
    assert set(rows) == {"m1", "m2_data80k"} and rows["m1"]["val"] is None
    assert (
        rows["m2_data80k"]["val"] == 4.5
        and rows["m2_data80k"]["label"] == "data80k · d256-L4 · 16.0M · T256 · 15K steps"
    )
    assert c.get("/api/evals/m1").text.startswith("# Evals: m1")
    assert c.get("/api/evals/..%2Fsecret").status_code == 404
