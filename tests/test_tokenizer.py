from src.schemas import (
    FunctionDefinition,
    SelectedFunction,
)
import pytest
from pathlib import Path

from llm_sdk import Small_LLM_Model
from src.bpe_tokenizer import (
    initialize_tokenizer,
    bpe_tokenize,
)
from src.utils import extract_json_from_response


from typing import Any


class DummyModel:
    """Stub replacing AutoModelForCausalLM without loading weights."""

    def to(self, device: Any) -> "DummyModel":
        return self

    def eval(self) -> "DummyModel":
        return self

    def parameters(self) -> list[Any]:
        return []


def test_tokenizer_initialization_fast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify tokenizer initialization using pure pytest monkeypatching."""

    # 1. Bypass heavy model loading with the dummy class
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: DummyModel(),
    )

    # 2. Setup dummy local files
    dummy_tokenizer = tmp_path / "tokenizer.json"
    dummy_merges = tmp_path / "merges.txt"
    dummy_tokenizer.write_text('{"model": {"vocab": {"<|endoftext|>": 0}}}')
    dummy_merges.write_text("#version: 0.2\n")

    # 3. Patch hf_hub_download to return local paths
    monkeypatch.setattr(
        "llm_sdk.hf_hub_download",
        lambda repo_id, filename, **kwargs: str(tmp_path / filename),
    )

    # 4. Execute test
    llm = Small_LLM_Model()
    tokenizer_path = Path(llm.get_path_to_tokenizer_file())
    merges_path = Path(llm.get_path_to_merges_file())

    vocab, merge_ranks = initialize_tokenizer(tokenizer_path, merges_path)
    assert vocab is not None
    assert merge_ranks is not None


def test_extract_json_valid() -> None:
    """Verify JSON extraction handles valid tool calls."""
    prompt = "Add 5 and 3"
    response = ('Here is the call: {"name": "fn_add_numbers", '
                '"parameters": {"a": 5.0, "b": 3.0}}'
                )

    result = extract_json_from_response(prompt, response)

    assert isinstance(result, SelectedFunction)
    assert result.name == "fn_add_numbers"
    assert result.parameters["a"] == 5.0


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


def test_extract_json_missing_required_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that extract_json_from_response returns an empty SelectedFunction
    when a required parameter is missing.
    """

    # 1. Mock tool schema requiring both 'a' and 'b'
    mock_functions = [
        FunctionDefinition.model_validate({
            "name": "fn_add",
            "description": "Add two numbers",
            "parameters": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "returns": {"type": "integer"},
        })
    ]

    # 2. Patch get_functions where defined
    monkeypatch.setattr(
        "src.utils.get_functions",
        lambda *args, **kwargs: mock_functions,
    )

    # 3. Model output providing 'a' but omitting required 'b'
    prompt = "Add 5 and 10"
    response = '{"name": "fn_add_numbers", "parameters": {"a": 5}}'

    # 4. Execute
    result = extract_json_from_response(prompt, response)

    # 5. Assert fallback to empty SelectedFunction
    assert result.name == ""
    assert result.parameters == {}
    assert result.prompt == prompt
