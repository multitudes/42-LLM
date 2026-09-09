## Python packages

Just as a recap for me... since we use uv and the code to run the program is

```
	@uv run python -m src
```
so the src has to have a package structure.


In Python, a **folder containing an `__init__.py` file is a package**.

- A **module** is a single `.py` file (e.g., `ollama.py`).
- A **package** is a directory with an `__init__.py` file (e.g., llm_sdk).

The `__init__.py` file can be empty or contain code.  
It tells Python that the folder should be treated as a package, allowing you to import from it:

```python
from llm_sdk import call_ollama_api
```

You can also have submodules (other `.py` files) inside the package and import them:

```python
from llm_sdk.ollama import call_ollama_api
```

**Summary:**  
- llm_sdk with `__init__.py` = package  
- `llm_sdk/ollama.py` = module inside the package

Adding `src/__init__.py` fixed the issue because it transformed `src` from a plain folder into an **explicit Python package**.

### The Core Problem: How `mypy` Resolves Relative Imports

When you run `mypy .` from your project root:

1. `mypy` discovers `src/utils.py` and inspects its imports.
2. Inside `utils.py`, it sees a relative import: `from .schemas import ...`.
3. In Python, the dot (`.`) in `from . module` means **"the current package"**.

Without an `__init__.py` file inside `src`, Python and `mypy` treated `src` as a plain folder rather than a package. As a result, when `mypy` tried to resolve `from .schemas`, it had no parent package context to anchor to—it looked at the top-level root directory for `schemas.py`, failed to find it, and raised `[import-not-found]`.

---

### What `__init__.py` Changed

By placing an `__init__.py` inside `src/`:

1. **Defines a Package:** You explicitly declared to Python and static analysis tools that `src` is a top-level package.
2. **Establishes Hierarchy:** Now, `src/utils.py` is formally recognized as `src.utils`.
3. **Resolves Relative Imports:** When `utils.py` says `from .schemas import ...`, `mypy` understands that `.` refers to the `src` package, allowing it to correctly resolve `src/schemas.py`.