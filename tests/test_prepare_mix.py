"""Mixtures in prepare_dataset: several sources in fixed token proportions (no network: fake streams)."""

import json
from unittest.mock import patch

import pytest
import torch
from datasets import IterableDataset

import mini_llm.prepare_dataset as prep
from mini_llm.data import get_tokenizer
from mini_llm.prepare_dataset import MixSource, load_subset, prepare_mix, token_quotas


def _rows(prefix: str, words: int, n: int = 4000):
    """Source `prefix`: documents of ~`words` words, each unique."""
    for i in range(n):
        yield {"text": f"{prefix} doc {i} " + " ".join(f"{prefix}{i}w{k}" for k in range(words))}


SOURCES = {"short": 8, "long": 40}  # different document lengths: token shares != document shares


def _fake_load(dataset, config=None, split=None, streaming=True):
    return IterableDataset.from_generator(lambda: _rows(dataset, SOURCES[dataset]))


@pytest.fixture
def mixed(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "TOKENIZE_BATCH", 10)  # small batches: tight quotas in a small test
    with patch("mini_llm.prepare_dataset.load_dataset", side_effect=_fake_load):
        _, _, stats = prepare_mix(
            [MixSource("short", None, 0.7), MixSource("long", None, 0.3)],
            num_tokens=60_000,
            val_num_tokens=3_000,
            out_dir=tmp_path,
            progress_every=0,
        )
    return tmp_path, stats


def test_parse_and_quotas():
    s = MixSource.parse("HuggingFaceFW/fineweb:sample-10BT=0.7")
    assert (s.dataset, s.config, s.fraction) == ("HuggingFaceFW/fineweb", "sample-10BT", 0.7)
    assert MixSource.parse("a/b=0.3").config is None
    for bad in ("a/b", "a/b=x", "a/b=0", "a/b=1.5", "=0.5"):
        with pytest.raises(ValueError):
            MixSource.parse(bad)
    assert token_quotas(1001, [0.7, 0.3]) == [700, 301]
    with pytest.raises(ValueError, match="sum to 1"):
        token_quotas(100, [0.7, 0.2])


def test_token_shares_hold_despite_different_document_lengths(mixed):
    out, stats = mixed
    short, long_ = stats["sources"]
    for src in stats["sources"]:
        assert src["train_tokens"] >= src["train_quota"] and src["val_tokens"] >= src["val_quota"]
    assert abs(short["train_share"] - 0.7) < 0.02 and abs(long_["train_share"] - 0.3) < 0.02
    assert short["train_docs"] > 3 * long_["train_docs"]  # many more of the short documents
    assert (
        torch.load(out / "train.pt").numel() == stats["train_tokens"] == short["train_tokens"] + long_["train_tokens"]
    )
    assert json.loads((out / "mix.json").read_text())["sources"][0]["dataset"] == "short"


def test_each_source_is_a_prefix_of_its_single_source_scan(mixed):
    """The mixture takes the same rows, in the same order, that prepare() would take from each source alone."""
    out, stats = mixed
    tok = get_tokenizer("gpt2")
    train = torch.load(out / "train.pt").tolist()
    eos = tok.eos_token_id
    docs, cur = [], []
    for t in train:
        if t == eos:
            docs.append(tok.decode(cur)), cur.clear()
        else:
            cur.append(t)
    for src in stats["sources"]:
        mine = [d for d in docs if d.startswith(src["dataset"] + " ")]
        with patch("mini_llm.prepare_dataset.load_dataset", side_effect=_fake_load):
            alone, _ = load_subset(src["dataset"], None, num_examples=len(mine), val_examples=0)
        assert mine == [r["text"] for r in alone]


def test_sources_are_interleaved_and_val_is_disjoint(mixed):
    out, stats = mixed
    tok = get_tokenizer("gpt2")
    first_quarter = tok.decode(torch.load(out / "train.pt")[: stats["train_tokens"] // 4].tolist())
    assert "short doc" in first_quarter and "long doc" in first_quarter  # any prefix holds the mixture
    val = set(tok.decode(torch.load(out / "val.pt").tolist()).split("<|endoftext|>")) - {""}
    train = set(tok.decode(torch.load(out / "train.pt").tolist()).split("<|endoftext|>")) - {""}
    assert val and val.isdisjoint(train)
