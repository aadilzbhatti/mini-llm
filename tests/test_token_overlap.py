import torch

from mini_llm.token_overlap import EOS_TOKEN_ID, doc_hash, file_hashes, main, split_documents


def _stream(*docs: list[int]) -> torch.Tensor:
    out: list[int] = []
    for d in docs:
        out.extend(d)
        out.append(EOS_TOKEN_ID)
    return torch.tensor(out, dtype=torch.long)


def test_split_documents_drops_separators_and_empty_runs():
    tokens = torch.tensor([1, 2, EOS_TOKEN_ID, EOS_TOKEN_ID, 3, EOS_TOKEN_ID, 4, 5])
    assert [d.tolist() for d in split_documents(tokens)] == [[1, 2], [3], [4, 5]]


def test_file_hashes_match_doc_hash(tmp_path):
    tokens = _stream([1, 2, 3], [4], [5, 6])
    path = tmp_path / "a.pt"
    torch.save(tokens, path)
    assert file_hashes(path) == {doc_hash(d) for d in split_documents(tokens)}


def test_main_filters_excluded_docs(tmp_path, capsys):
    a, b, c, d = [10, 11], [20], [30, 31, 32], [40]
    torch.save(_stream(a, b, c, d), tmp_path / "val.pt")
    torch.save(_stream(b, [99], d), tmp_path / "train.pt")

    main([str(tmp_path / "val.pt"), "--exclude", str(tmp_path / "train.pt"), "--out", str(tmp_path / "clean.pt")])

    out = torch.load(tmp_path / "clean.pt", weights_only=True)
    assert out.dtype == torch.long
    assert out.tolist() == _stream(a, c).tolist()
    assert "overlap with" in capsys.readouterr().out


def test_load_tokens_mmap_matches_full_load(tmp_path):
    from mini_llm.data import load_tokens, make_batch

    tokens = torch.arange(10_000, dtype=torch.long)
    path = tmp_path / "t.pt"
    torch.save(tokens, path)

    mapped, full = load_tokens(path), load_tokens(path, mmap=False)
    assert torch.equal(mapped, tokens) and torch.equal(full, tokens)
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    x1, y1 = make_batch(mapped, 4, 16, generator=g1)
    x2, y2 = make_batch(full, 4, 16, generator=g2)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
