"""--warmup-tokens: warmup measured in tokens, converted to steps per batch size."""

import pytest
import torch

import mini_llm.train as train


def resolved(argv):
    return train.resolve_warmup(train.parse_args(argv), argv)


@pytest.mark.parametrize("batch, tokens, steps", [
    (4, 256_000, 500),      # the batch-4 baseline's 500-step warmup, in tokens
    (64, 256_000, 32),      # same tokens at batch 64: 31.25 -> rounds up
    (64, 819_200, 100),     # the batch-64 sweep's 100-step warmup
    (128, 819_200, 50),     # ...held fixed in tokens at batch 128
    (64, 0, 0),             # 0 tokens = no warmup, as with --warmup-steps 0
])
def test_converts_tokens_to_steps_with_the_global_batch(batch, tokens, steps):
    args = resolved(["--batch-size", str(batch), "--block-size", "128", "--warmup-tokens", str(tokens)])
    assert args.warmup_steps == steps


def test_unset_leaves_warmup_steps_alone():
    assert resolved([]).warmup_steps == 500
    assert resolved(["--warmup-steps", "100"]).warmup_steps == 100


@pytest.mark.parametrize("flag", [["--warmup-steps", "500"], ["--warmup-steps=100"]])
def test_refuses_both_flags(flag):
    with pytest.raises(SystemExit, match="not both"):
        resolved(["--warmup-tokens", "1000", *flag])


class StubTokenizer:
    def __len__(self):
        return 64


def test_lr_ramp_has_the_converted_length(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(train, "get_tokenizer", lambda: StubTokenizer())
    monkeypatch.setattr(train, "select_device", lambda: torch.device("cpu"))
    torch.save(torch.randint(0, 64, (4000,), generator=torch.Generator().manual_seed(0)), tmp_path / "t.pt")
    # batch 4 x block 8 = 32 tokens/step, so 96 warmup tokens = 3 warmup steps.
    train.main(["--tokens", "t.pt", "--block-size", "8", "--n-embd", "16", "--n-head", "2", "--n-layer", "1",
                "--batch-size", "4", "--steps", "10", "--lr", "1e-3", "--warmup-tokens", "96",
                "--eval-interval", "100", "--eval-batches", "1", "--no-tensorboard", "--save", "--save-name", "m.pt"])
    lrs = [lr for _, lr in torch.load(tmp_path / "checkpoints" / "m.pt", weights_only=False)["lr_history"]]
    assert lrs[:3] == pytest.approx([1e-3 / 3, 2e-3 / 3, 1e-3])      # linear ramp over 3 steps
    assert lrs[3] == pytest.approx(1e-3) and lrs[4] < lrs[3]          # then cosine from the peak
