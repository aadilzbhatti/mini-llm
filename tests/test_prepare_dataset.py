"""Growability guarantees for the train/val split in prepare_dataset.py.

Uses a synthetic in-memory streaming dataset (no network) so these run
fast and deterministically.
"""

from unittest.mock import patch

from datasets import IterableDataset

from mini_llm.prepare_dataset import load_subset


def _fake_rows(n: int):
    for i in range(n):
        yield {"text": f"document number {i} has some unique content xyz{i}qrs"}


def _load_subset(num_examples: int, val_examples: int, pool_size: int = 5000, seed: int = 0):
    with patch(
        "mini_llm.prepare_dataset.load_dataset",
        return_value=IterableDataset.from_generator(lambda: _fake_rows(pool_size)),
    ):
        return load_subset(num_examples=num_examples, val_examples=val_examples, seed=seed)


def _texts(rows):
    return [row["text"] for row in rows]


def test_train_and_val_never_overlap():
    train_rows, val_rows = _load_subset(num_examples=100, val_examples=20)
    assert len(train_rows) == 100
    assert len(val_rows) == 20
    assert set(_texts(train_rows)).isdisjoint(_texts(val_rows))


def test_growing_num_examples_only_extends_train():
    train_small, val_small = _load_subset(num_examples=50, val_examples=20)
    train_big, val_big = _load_subset(num_examples=150, val_examples=20)

    # val is completely unchanged
    assert _texts(val_small) == _texts(val_big)
    # old train is an exact prefix of new train -- only appended to, never reshuffled
    assert _texts(train_big)[: len(train_small)] == _texts(train_small)
    # and the new rows are genuinely new, not stolen from val
    new_rows = set(_texts(train_big)) - set(_texts(train_small))
    assert new_rows.isdisjoint(_texts(val_big))


def test_growing_val_examples_only_extends_val():
    train_small, val_small = _load_subset(num_examples=50, val_examples=20)
    train_big, val_big = _load_subset(num_examples=50, val_examples=60)

    # train is completely unchanged
    assert _texts(train_small) == _texts(train_big)
    # old val is an exact prefix of new val
    assert _texts(val_big)[: len(val_small)] == _texts(val_small)
    # and the new val rows were never part of train
    new_rows = set(_texts(val_big)) - set(_texts(val_small))
    assert new_rows.isdisjoint(_texts(train_big))


def test_token_stream_matches_per_text_encode():
    """Batched, chunked tokenization must give byte-identical files to the old
    per-text `encode` + EOS loop -- otherwise growing a dataset would break the
    prefix property against files built before."""
    import torch

    from mini_llm.data import get_tokenizer
    from mini_llm.prepare_dataset import TokenStream

    tok = get_tokenizer("gpt2")
    texts = [f"doc {i}: héllo wörld\n\n  spaces  and <|endoftext|> marker {i * 7}" for i in range(25)]
    texts.append("")
    expected: list[int] = []
    for text in texts:
        expected.extend(tok.encode(text))
        expected.append(tok.eos_token_id)

    stream = TokenStream(tok, batch_size=4)  # forces several chunks + a partial one
    for text in texts:
        stream.add(text)
    out = stream.tensor()

    assert out.dtype == torch.long
    assert out.tolist() == expected
    assert stream.num_docs == len(texts)
