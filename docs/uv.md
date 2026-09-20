# Package Management with `uv`

I use `uv` as the fast package manager and virtual environment manager for this project to guarantee fast, deterministic, and reproducible builds.

## Setup & Environment Initialization

1. **Install `uv**`:

```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh

```

2. **Initialize the project structure**:

```zsh
uv init

```

3. **Add production and development dependencies**:

```zsh
uv add torch transformers huggingface-hub pydantic numpy
uv add --dev ruff mypy pytest

```

4. **Sync the virtual environment**:

```zsh
uv sync

```

This builds my local `.venv/` and generates an updated `uv.lock` file to lock all transitive dependencies.

## Execution

To run my project's main entry point without needing to manually activate the virtual environment:

```zsh
uv run python -m src

```

Using `uv run` ensures the script uses the project's dedicated isolated interpreter and all pinned packages automatically.

## Core Workflow Commands

* **Add a runtime package:**

```zsh
uv add <package>

```

* **Add a dev tool:**

```zsh
uv add --dev <package>

```

* **Remove a dependency:**

```zsh
uv remove <package>

```

* **Inspect installed dependency tree:**

```zsh
uv tree

```

* **Run project tools and tests:**

```zsh
uv run pytest
uv run mypy .
uv run ruff check --fix .

```

* **Synchronize state:**

```zsh
uv sync

```

## Project File Structure

My project relies on modern Python packaging tools rather than a legacy `requirements.txt`:

* `pyproject.toml` — Standard project configuration, metadata, and dependency definitions.
* `uv.lock` — Cross-platform lockfile enforcing exact version parity across machines.
* `.venv/` — Automatically managed virtual environment directory.

## Implementation Notes

* I never manually activate `.venv/` — `uv run` handles virtual environment isolation and context execution dynamically.
* The `uv.lock` file ensures my project builds identically across different environments.
* All data schemas and tool parameters in my pipeline use `pydantic` for runtime validation.

## References

* [uv Documentation](https://docs.astral.sh/uv/)
* [pyproject.toml Specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)