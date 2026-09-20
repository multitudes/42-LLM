# Static Analysis, Linting & Type Enforcement

I enforce strict static analysis and runtime type validation across the entire codebase to catch bugs early and guarantee schema compliance during LLM tool calling.

---

## Modern Python Type Hinting (Python 3.10+)

I target Python 3.10+ and utilize native built-in generics (PEP 585) and standard union syntax (PEP 604). I avoid deprecated imports from `typing` like `List`, `Dict`, or `Optional`, relying instead on lower-case built-ins and the `|` operator.

| Deprecated (`typing`) | Modern Built-in | Purpose |
| --- | --- | --- |
| `List[str]` | `list[str]` | Typed list collections |
| `Dict[str, Any]` | `dict[str, Any]` | Dictionary mappings |
| `Tuple[int, str]` | `tuple[int, str]` | Fixed-length tuple signatures |
| `Optional[str]` | `str | None` |

*Note: `from typing import Any` is retained because `Any` is a static type construct rather than a runtime class object.*

---

## Automated Linting & Import Sorting

I configure **Ruff** to enforce PEP 8 guidelines and handle `isort` import ordering (`Ruff(I001)`). Imports are automatically grouped into standard library, third-party, and local module blocks.

To automatically format code and reorder imports across the project:

```zsh
uv run ruff check --fix .
uv run ruff format .

```

---

## Runtime Argument Type Coercion

LLMs frequently output raw JSON values that mismatch expected function signatures (e.g., passing an integer `1` instead of a float `1.0`).

I implemented `enforce_arg_types` to sanitize arguments against tool definitions at runtime. The function includes Google-style PEP 257 docstrings and type guards (`isinstance`, `.get()`) to pass `mypy` checks without warnings:

```python
from typing import Any


def enforce_arg_types(
    fn_name: str,
    args: dict[str, Any],
    functions_def: list[dict[str, Any]],
) -> dict[str, Any]:
    """Converts argument values to their defined types based on function definitions.

    Args:
        fn_name: The name of the target function to look up.
        args: Dictionary of extracted argument names and raw LLM outputs.
        functions_def: List of tool definition dictionaries.

    Returns:
        Dictionary with argument values converted to their expected types.
    """
    fn_def = next((f for f in functions_def if f.get("fn_name") == fn_name), None)
    if not fn_def or "args_types" not in fn_def:
        return args

    args_types = fn_def["args_types"]
    if not isinstance(args_types, dict):
        return args

    for arg_name, arg_type in args_types.items():
        if arg_name in args:
            try:
                if arg_type == "float":
                    args[arg_name] = float(args[arg_name])
                elif arg_type == "int":
                    args[arg_name] = int(args[arg_name])
                elif arg_type == "str":
                    args[arg_name] = str(args[arg_name])
            except (ValueError, TypeError):
                pass  # Keep original value if conversion fails

    return args

```

---

## Static Type Checking (`mypy`) Configuration

I manage `mypy` flags centrally inside `pyproject.toml` rather than passing long CLI flag strings. This ensures identical static analysis enforcement across IDEs, terminal runs, and CI checks.

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true
ignore_missing_imports = true
disallow_untyped_defs = true
check_untyped_defs = true

```

### Core Type-Checking Rules

1. **Explicit Return Annotations:** Every constructor ends with `-> None:`.
2. **Explicit Null Checks:** `Optional` or `| None` variables must undergo explicit `if val is not None:` validation before usage.
3. **Empty Collections:** Empty lists or dicts are explicitly typed at declaration (e.g., `results: list[str] = []`).

---

## Automated Verification via `Makefile`

I centralize static analysis execution in the project `Makefile`:

```makefile
lint:
	uv run ruff check .
	uv run mypy .

```

To run the complete quality check suite:

```zsh
make lint

```
