from pathlib import Path

from llm_sdk import Small_LLM_Model
from src.bpe_tokenizer import initialize_tokenizer
from src.schemas import SelectedFunction
from src.utils import extract_json_from_response


def test_tokenizer_initialization() -> None:
    """Verify tokenizer loads without crashing."""

    llm = Small_LLM_Model()
    vocab_path = Path(llm.get_path_to_tokenizer_file())

    tokenizer = initialize_tokenizer(vocab_path=vocab_path)
    assert tokenizer is not None


def test_extract_json_valid() -> None:
    """Verify JSON extraction handles valid tool calls."""
    prompt = "Multiply 5 and 3"
    response = ('Here is the call: {"fn_name": "multiply", '
                '"args": {"a": 5, "b": 3}}'
                )

    result = extract_json_from_response(prompt, response)

    assert isinstance(result, SelectedFunction)
    assert result.fn_name == "multiply"
    assert result.args["a"] == 5


def test_extract_json_missing() -> None:
    """Verify JSON extraction handles malformed output gracefully."""
    prompt = "Hello"
    response = "I cannot help with that."

    result = extract_json_from_response(prompt, response)

    assert result.fn_name == ""
