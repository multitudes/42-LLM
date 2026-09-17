from pathlib import Path

import pytest

from src.schemas import (
    FunctionDefinition,
    SelectedFunction,
)
from src.utils import (
    enforce_arg_types,
    extract_json_from_response,
    get_functions,
    get_input_prompts,
    write_output_to_file,
)


def test_get_functions_success(tmp_path: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify get_functions successfully parses a valid tools definition file.
    """
    d = tmp_path / "exercise_input"
    d.mkdir()
    f = d / "functions_definition.json"
    f.write_text(
        '[{"name": "fn_add", "description": "Add numbers",'
        '"parameters": {"a": {"type": "number"}},'
        '"returns": {"type": "number"}}]',
        encoding="utf-8",
    )
    monkeypatch.setattr("src.utils.TOOLS_DEFINITION_FILE", f)

    funcs = get_functions()
    assert len(funcs) == 1
    assert funcs[0].name == "fn_add"


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
            "name": "fn_multiply",
            "description": "Multiply numbers",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "integer"},
                "name": {"type": "string"},
            },
            "returns": {"type": "number"},
        },
    ]
    raw_parameters = {"a": "10", "b": "5", "name": 123}
    cleaned = enforce_arg_types("fn_multiply", raw_parameters, functions_def)

    assert cleaned["a"] == 10.0
    assert isinstance(cleaned["a"], float)
    assert cleaned["b"] == 5
    assert isinstance(cleaned["b"], int)
    assert cleaned["name"] == "123"


def test_extract_json_from_response_with_think(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify JSON extraction successfully strips think blocks and parses valid
    response.
    """
    monkeypatch.setattr(
        "src.utils.get_functions",
        lambda *args, **kwargs: [
            FunctionDefinition.model_validate({
                "name": "multiply",
                "description": "Multiply numbers",
                "parameters": {"a": {"type": "number"},
                               "b": {"type": "number"}},
                "returns": {"type": "number"},
            })
        ],
    )

    prompt = "Multiply numbers"
    response = (
        '</think>\n'
        'Here is your answer: {"name": "multiply", '
        '"parameters": {"a": 2, "b": 4}}'
    )

    result = extract_json_from_response(prompt, response)
    assert isinstance(result, SelectedFunction)
    assert result.name == "multiply"
    assert result.parameters["a"] == 2.0


def test_extract_json_from_response_invalid() -> None:
    """
    Verify extraction falls back to empty SelectedFunction on malformed text.
    """
    prompt = "Hello"
    response = "I cannot fulfill this request."
    result = extract_json_from_response(prompt, response)
    assert result.name == ""
    assert result.parameters == {}


def test_write_output_to_file(tmp_path: Path,
                              monkeypatch: pytest.MonkeyPatch,
                              ) -> None:
    """Verify write_output_to_file successfully serializes models to JSON."""
    out_file = tmp_path / "output" / "results.json"
    monkeypatch.setattr("src.utils.OUTPUT_FILE", out_file)

    model = SelectedFunction(prompt="Test", name="func", parameters={"x": 1})
    write_output_to_file([model], output_file=out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "func" in content
