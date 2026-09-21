from pathlib import Path

from src.bpe_tokenizer import (
    bpe_tokenize,
    initialize_tokenizer,
)


def test_tokenizer_initialization_fast(tmp_path: Path) -> None:
    """Verify initialize_tokenizer loads vocab and merges from local files."""
    tokenizer_path = tmp_path / "tokenizer.json"
    merges_path = tmp_path / "merges.txt"
    tokenizer_path.write_text(
        '{"model": {"vocab": {"<|endoftext|>": 0, "a": 1}}}',
        encoding="utf-8",
    )
    merges_path.write_text("#version: 0.2\n", encoding="utf-8")

    vocab, merge_ranks = initialize_tokenizer(tokenizer_path, merges_path)

    assert vocab["<|endoftext|>"] == 0
    assert vocab["a"] == 1
    assert merge_ranks == {}


def test_bpe_tokenize_unknown_tokens_mapped_to_unk() -> None:
    # Setup vocabulary containing <unk> (ID 0) and a few known characters
    vocab = {
        "<unk>": 0,
        "c": 1,
        "a": 2,
        "t": 3,
    }
    merge_ranks: dict[tuple[str, str], int] = {}

    # "cat" consists of known tokens [1, 2, 3]
    # "dog" consists of unknown characters ('d', 'o', 'g')
    input_text = "cat dog"

    token_ids = bpe_tokenize(input_text, vocab, merge_ranks)

    # 1. Verify sequence length is preserved (unknown tokens are not dropped)
    # Expected characters in sequence: 'c', 'a', 't', ' ', 'd', 'o', 'g'
    # (7 tokens)
    assert len(token_ids) == len(input_text)

    # 2. Verify 'cat' maps to [1, 2, 3]
    assert token_ids[:3] == [1, 2, 3]

    # 3. Verify ' ' and 'dog' (unknown characters) mapped to <unk> ID (0)
    assert token_ids[3:] == [0, 0, 0, 0]


def test_bpe_tokenize_without_unk_uses_endoftext_fallback() -> None:
    """
    When <unk> is absent, unknown pieces map to <|endoftext|> (never drop).
    """
    vocab = {
        "<|endoftext|>": 151643,
        "c": 1,
        "a": 2,
        "t": 3,
    }
    merge_ranks: dict[tuple[str, str], int] = {}

    input_text = "cat dog"
    token_ids = bpe_tokenize(input_text, vocab, merge_ranks)

    assert len(token_ids) == len(input_text)
    assert token_ids[:3] == [1, 2, 3]
    assert token_ids[3:] == [151643, 151643, 151643, 151643]
