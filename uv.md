# Using `uv` with Python Projects

`uv` is a fast Python package manager and virtual environment tool. In this project, you are required to use `uv` for dependency management and running your code.

## Setting Up Your Environment

1. **Install `uv`** (if not already installed):
```zsh
pipx install uv
# or
pip install uv
```

2. **Create a virtual environment and install dependencies:**
   ```zsh
   uv venv .venv
   uv pip install numpy pydantic
   # If you have a requirements.txt:
   uv pip install -r requirements.txt
   ```

3. **Sync dependencies (recommended for reproducibility):**
   ```zsh
   uv sync
   ```
   This will install all dependencies listed in `requirements.txt` or `pyproject.toml`.

## Running Your Project

To run your main script as required by the project:

```zsh
uv run python -m src
```

This command will:
- Use the Python interpreter from your virtual environment (if activated).
- Run the `src` module as the entry point.

## Common `uv` Commands

- **Install a package:**
  ```zsh
  uv pip install <package>
  ```
- **List installed packages:**
  ```zsh
  uv pip list
  ```
- **Remove a package:**
  ```zsh
  uv pip uninstall <package>
  ```
- **Run scripts:**
  ```zsh
  uv run python <script.py>
  ```

## Notes
- Always activate your virtual environment before running commands:
  ```zsh
  source .venv/bin/activate
  ```
- The `uv run` command ensures your code runs in the correct environment.
- For this project, all classes must use `pydantic` for validation, and you may use `numpy` and `json`.
- Do **not** use forbidden packages (see README for details).

## References
- [uv documentation](https://github.com/astral-sh/uv)
- [pipx documentation](https://pipx.pypa.io/)

---
This guide explains how to use `uv` for Python projects as required by the 42-LLM-test project.
