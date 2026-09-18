from pathlib import Path
from typing import Any

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


def test_get_functions_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify get_functions successfully parses a valid tools def file."""
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
    """Verify type coercion correctly converts string/int to expected types."""
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
    """Verify JSON extraction successfully strips think blocks"""
    monkeypatch.setattr(
        "src.utils.get_functions",
        lambda *args, **kwargs: [
            FunctionDefinition.model_validate({
                "name": "multiply",
                "description": "Multiply numbers",
                "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
                "returns": {"type": "number"},
            })
        ],
    )

    prompt = "Multiply numbers"
    response = (
        "</think>\n"
        'Here is your answer: {"name": "multiply", "parameters": {"a": 2, "b": 4}}'
    )

    result = extract_json_from_response(prompt, response)
    assert isinstance(result, SelectedFunction)
    assert result.name == "multiply"
    assert result.parameters["a"] == 2.0


def test_extract_json_from_response_invalid() -> None:
    """Verify extraction falls back to empty SelectedFunction on malformed text."""
    prompt = "Hello"
    response = "I cannot fulfill this request."
    result = extract_json_from_response(prompt, response)
    assert result.name == ""
    assert result.parameters == {}


def test_write_output_to_file(
    tmp_path: Path,
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


@pytest.fixture
def sample_functions_def() -> list[dict[str, Any]]:
    """Sample function definition matching OpenAI/Pydantic schema layout."""
    return [
        {
            "name": "calculate_metrics",
            "parameters": {
                "item_count": {"type": "integer"},
                "threshold": {"type": "number"},
                "category_name": {"type": "string"},
                "is_enabled": {"type": "boolean"},
            },
        }
    ]


# --- INTEGER COERCION TESTS ---


def test_integer_from_float_string(
    sample_functions_def: list[dict[str, Any]],
) -> None:
    params = {"item_count": "11.0"}
    res = enforce_arg_types("calculate_metrics", params, sample_functions_def)
    assert res["item_count"] == 11


def test_integer_rounding_behavior(
    sample_functions_def: list[dict[str, Any]],
) -> None:
    """Verifies that floats/strings with decimals round correctly."""
    res1 = enforce_arg_types(
        "calculate_metrics", {"item_count": "11.9"}, sample_functions_def
    )
    assert res1["item_count"] == 12

    res2 = enforce_arg_types(
        "calculate_metrics", {"item_count": 11.9}, sample_functions_def
    )
    assert res2["item_count"] == 12

    res3 = enforce_arg_types(
        "calculate_metrics", {"item_count": 11.2}, sample_functions_def
    )
    assert res3["item_count"] == 11


# --- NUMBER (FLOAT) COERCION TESTS ---


def test_number_from_string_and_int(
    sample_functions_def: list[dict[str, Any]],
) -> None:
    """Verifies string floats and integers convert to float."""
    res1 = enforce_arg_types(
        "calculate_metrics", {"threshold": "45.5"}, sample_functions_def
    )
    assert res1["threshold"] == 45.5
    assert isinstance(res1["threshold"], float)

    res2 = enforce_arg_types(
        "calculate_metrics", {"threshold": "10"}, sample_functions_def
    )
    assert res2["threshold"] == 10.0
    assert isinstance(res2["threshold"], float)


# --- BOOLEAN COERCION TESTS ---


@pytest.mark.parametrize("truthy_value", ["true", "True", "1", "yes", True])
def test_boolean_truthy_coercion(
    sample_functions_def: list[dict[str, Any]],
    truthy_value: str | bool,
) -> None:
    params = {"is_enabled": truthy_value}
    res = enforce_arg_types("calculate_metrics", params, sample_functions_def)
    assert res["is_enabled"] is True


@pytest.mark.parametrize("falsy_value", ["false", "False", "0", "no", False])
def test_boolean_falsy_coercion(
    sample_functions_def: list[dict[str, Any]],
    falsy_value: str | bool,
) -> None:
    params = {"is_enabled": falsy_value}
    res = enforce_arg_types("calculate_metrics", params, sample_functions_def)
    assert res["is_enabled"] is False


# --- STRING COERCION TESTS ---


def test_string_coercion(
    sample_functions_def: list[dict[str, Any]],
) -> None:
    """Verifies non-string primitives cast to string."""
    params = {"category_name": 12345}
    res = enforce_arg_types("calculate_metrics", params, sample_functions_def)
    assert res["category_name"] == "12345"
    assert isinstance(res["category_name"], str)


# --- FALLBACK & FAILURE EDGE CASES ---


def test_unparseable_value_retains_original(
    sample_functions_def: list[dict[str, Any]],
) -> None:
    """Verifies that completely invalid types fall back gracefully."""
    params = {"item_count": "invalid_integer_string"}
    res = enforce_arg_types("calculate_metrics", params, sample_functions_def)
    assert res["item_count"] == "invalid_integer_string"


def test_unknown_function_name_returns_unmodified_dict(
    sample_functions_def: list[dict[str, Any]],
) -> None:
    """Verifies unlisted functions pass parameters through untouched."""
    params = {"item_count": "11.0"}
    res = enforce_arg_types("unknown_function", params, sample_functions_def)
    assert res["item_count"] == "11.0"


def test_missing_parameter_key_handled_safely(
    sample_functions_def: list[dict[str, Any]],
) -> None:
    """Verifies missing keys in parameters dict don't cause KeyErrors."""
    params = {"threshold": "1.5"}
    res = enforce_arg_types("calculate_metrics", params, sample_functions_def)
    assert "item_count" not in res
    assert res["threshold"] == 1.5
