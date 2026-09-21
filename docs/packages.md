# Python Package Structure & Import Resolution

I structure my application source code as an explicit Python package inside the `src/` directory so it can be executed directly as a module via `uv run python -m src`.

## Modules vs. Packages

* **Module:** A single Python file containing code (e.g., `src/bpe_tokenizer.py`).
* **Package:** A directory containing an `__init__.py` file (e.g., `src/` or `llm_sdk/`).

By adding `__init__.py` to a directory, I tell Python and static analysis tools to treat that directory as an explicit package. This enables structured imports across the codebase:

```python
# Absolute import from the embedded SDK package
from llm_sdk import Small_LLM_Model

# Relative import within the local src package
from .schemas import ToolParameter

```

## Resolving Static Analysis (`mypy`) Import Errors

When running static type checks (`uv run mypy .`), `mypy` analyzes how modules resolve relative import statements.

### The Problem

Inside `src/utils.py`, relative imports use dot notation (e.g., `from .schemas import ...`) where `.` represents the current package. Without `src/__init__.py`, Python and `mypy` treated `src/` as a plain folder rather than a package. As a result, `mypy` failed to establish a parent package anchor for `utils.py`, looking for `schemas.py` in the root directory and raising an `[import-not-found]` error.

### The Fix

Placing `__init__.py` inside `src/` resolved this issue by:

1. **Defining the Package Anchor:** Explicitly declaring `src/` as a top-level package to the Python runtime and `mypy`.
2. **Establishing Namespace Hierarchy:** Formally registering `src/utils.py` as `src.utils` and `src/schemas.py` as `src.schemas`.
3. **Validating Relative Imports:** Allowing `mypy` to resolve `from .schemas` relative to the established `src` package context without error.