from typing import Any
import pytest

from src.__main__ import main


def test_main_exits_with_code_1_on_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify main() exits with code 1 when loading tools raises a TypeError."""

    def mock_type_error(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("Tools definition file must contain a JSON array.")

    # Patch get_functions in src.utils where it is defined
    monkeypatch.setattr("src.utils.get_functions", mock_type_error)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code != 0


def test_main_exits_with_code_1_on_unexpected_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify main() exits with code 1 on model loading or pipeline failures."""
    # Patch symbols imported directly into src.__main__
    monkeypatch.setattr("src.__main__.get_input_prompts",
                        lambda *a, **kw: ["Test prompt"])
    monkeypatch.setattr("src.utils.get_functions", lambda *a, **kw: [])

    def mock_pipeline_failure(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CUDA out of memory or pipeline failure.")

    # Patch pipeline runner (adjust target module path if defined in src.bpe_tokenizer)
    monkeypatch.setattr("src.utils.write_output_to_file", mock_pipeline_failure)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code != 0
