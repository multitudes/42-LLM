from pathlib import Path

from llm_sdk import Small_LLM_Model
from src.bpe_tokenizer import (
    initialize_tokenizer,
    bpe_tokenize,
)
from src.schemas import SelectedFunction
from src.utils import extract_json_from_response


def test_tokenizer_initialization() -> None:
    """Verify tokenizer loads without crashing."""

    llm = Small_LLM_Model()
    tokenizer_path = Path(llm.get_path_to_tokenizer_file())
    print("Tokenizer file path:", tokenizer_path)
    merges_path = Path(llm.get_path_to_merges_file())

    vocab, merge_ranks = initialize_tokenizer(tokenizer_path, merges_path)
    assert vocab is not None
    assert merge_ranks is not None


def test_extract_json_valid() -> None:
    """Verify JSON extraction handles valid tool calls."""
    prompt = "Multiply 5 and 3"
    response = ('Here is the call: {"name": "multiply", '
                '"parameters": {"a": 5, "b": 3}}'
                )

    result = extract_json_from_response(prompt, response)

    assert isinstance(result, SelectedFunction)
    assert result.name == "multiply"
    assert result.parameters["a"] == 5


def test_extract_json_missing() -> None:
    """Verify JSON extraction handles malformed output gracefully."""
    prompt = "Hello"
    response = "I cannot help with that."

    result = extract_json_from_response(prompt, response)

    assert result.name == ""


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


def test_bpe_tokenize_without_unk_in_vocab() -> None:
    # Fallback sanity check when <unk> is missing from vocab
    vocab = {"c": 1, "a": 2, "t": 3}
    merge_ranks: dict[tuple[str, str], int] = {}

    input_text = "cat dog"
    token_ids = bpe_tokenize(input_text, vocab, merge_ranks)

    # Should safely drop missing tokens if <unk> is absent in vocab
    assert token_ids == [1, 2, 3]
