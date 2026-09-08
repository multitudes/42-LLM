from pathlib import Path
import pytest
from pydantic import ValidationError
from src.schemas import FunctionDefinition, SelectedFunction
from src.utils import (
    enforce_arg_types,
    extract_json_from_response,
    get_functions,
    get_input_prompts,
    get_tool_list,
    write_output_to_file,
)


def test_get_functions_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_functions successfully parses a valid tools definition file."""
    d = tmp_path / "exercise_input"
    d.mkdir()
    f = d / "functions_definition.json"
    f.write_text(
        '[{"fn_name": "fn_add", "args_names": ["a"], "args_types": {"a": "float"}, "return_type": "float"}]',
        encoding="utf-8",
    )
    monkeypatch.setattr("src.utils.TOOLS_DEFINITION_FILE", f)

    funcs = get_functions()
    assert len(funcs) == 1
    assert funcs[0].fn_name == "fn_add"


def test_get_functions_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_functions raises RuntimeError when the file is missing."""
    monkeypatch.setattr("src.utils.TOOLS_DEFINITION_FILE",
                        "nonexistent_file.json")
    with pytest.raises(RuntimeError):
        get_functions()


def test_get_input_prompts(tmp_path: Path) -> None:
    """Verify get_input_prompts extracts prompt strings correctly."""
    f = tmp_path / "prompts.json"
    f.write_text(
        '[{"prompt": "Hello"}, {"prompt": "World"}]',
        encoding="utf-8",
    )
    prompts = get_input_prompts(f)
    assert prompts == ["Hello", "World"]


def test_enforce_arg_types() -> None:
    """
    Verify type coercion correctly converts string/int inputs to expected types.
    """
    functions_def = [
        {
            "fn_name": "fn_multiply",
            "args_names": ["a", "b", "name"],
            "args_types": {"a": "float", "b": "int", "name": "str"},
            "return_type": "float",
        }
    ]
    raw_args = {"a": "10", "b": "5", "name": 123}
    cleaned = enforce_arg_types("fn_multiply", raw_args, functions_def)

    assert cleaned["a"] == 10.0
    assert isinstance(cleaned["a"], float)
    assert cleaned["b"] == 5.0
    assert isinstance(cleaned["b"], int)
    assert cleaned["name"] == "123"


def test_extract_json_from_response_with_think(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify JSON extraction successfully strips think blocks and parses valid response."""
    monkeypatch.setattr(
        "src.utils.get_functions",
        lambda: [
            FunctionDefinition(
                fn_name="multiply",
                args_names=["a", "b"],
                args_types={"a": "float", "b": "float"},
                return_type="float",
            )
        ],
    )

    prompt = "Multiply numbers"
    response = (
        '</think>\n'
        'Here is your answer: {"fn_name": "multiply", "args": {"a": 2, "b": 4}}'
    )

    result = extract_json_from_response(prompt, response)
    assert isinstance(result, SelectedFunction)
    assert result.fn_name == "multiply"
    assert result.args["a"] == 2.0


def test_extract_json_from_response_invalid() -> None:
    """Verify extraction falls back to empty SelectedFunction on malformed text."""
    prompt = "Hello"
    response = "I cannot fulfill this request."
    result = extract_json_from_response(prompt, response)
    assert result.fn_name == ""
    assert result.args == {}


def test_write_output_to_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify write_output_to_file successfully serializes models to JSON."""
    out_file = tmp_path / "output" / "results.json"
    monkeypatch.setattr("src.utils.OUTPUT_FILE", out_file)

    model = SelectedFunction(prompt="Test", fn_name="func", args={"x": 1})
    write_output_to_file([model])

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "func" in content
