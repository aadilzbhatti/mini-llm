"""Training on the validation set is refused: by path at submit time, by content at startup."""

import shutil

import pytest
import torch

import mini_llm.train as train
from test_modal_launch import JOB, post, repo, spawned  # noqa: F401 - fixtures


def test_submit_rejects_same_file_for_train_and_val(repo, spawned):  # noqa: F811
    args = {**JOB["args"], "tokens": "data/d1/train.pt", "val-tokens": "data/d1/train.pt"}
    for target in ("modal", "local"):
        r = post(repo, {**JOB, "args": args, "target": target})
        assert r.status_code == 422 and "trains on the validation set" in r.text
    assert spawned == []


class StubTokenizer:
    def __len__(self):
        return 64


def test_train_refuses_identical_data_under_another_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "get_tokenizer", lambda: StubTokenizer())
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    torch.save(torch.randint(0, 64, (600,), generator=torch.Generator().manual_seed(0)), tmp_path / "val.pt")
    shutil.copy(tmp_path / "val.pt", tmp_path / "train.pt")  # same data, different path
    with pytest.raises(SystemExit, match="identical data"):
        train.main(
            [
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
                "--steps",
                "2",
                "--no-tensorboard",
            ]
        )
