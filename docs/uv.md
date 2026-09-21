# Package Management with `uv`

This project uses `uv` as the package and virtual-environment manager for fast, deterministic builds.

## Setup & Environment Initialization

1. **Install `uv`** using the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/). On macOS you can also run:

```zsh
brew install uv
```

Avoid piping remote install scripts into a shell unless you trust the source. The project Makefile expects `uv` to already be on `PATH` (`make install` will fail with a clear message if it is missing).

2. **Synchronize the project environment** from the repository root:

```zsh
uv sync
# or: make install
```

This creates/updates `.venv/` and honors `uv.lock` for transitive dependency versions.

## Execution

Run the main entry point without manually activating the virtual environment:

```zsh
uv run python -m src
```

`uv run` selects the project interpreter and pinned packages automatically.

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

* **Inspect the dependency tree:**

```zsh
uv tree
```

* **Run project tools and tests:**

```zsh
uv run pytest
uv run mypy .
uv run flake8 .
uv run ruff check .
```

* **Synchronize state:**

```zsh
uv sync
```

## Project File Structure

Packaging uses modern Python metadata rather than a legacy `requirements.txt`:

* `pyproject.toml` — Project configuration, metadata, and dependencies.
* `uv.lock` — Cross-platform lockfile for exact version parity.
* `.venv/` — Managed virtual environment directory.

## Implementation Notes

* Prefer `uv run …` over manually activating `.venv/`.
* Keep `uv.lock` committed so environments stay reproducible.
* Runtime schemas and tool parameters use `pydantic` for validation.

## References

* [uv Documentation](https://docs.astral.sh/uv/)
* [uv Installation](https://docs.astral.sh/uv/getting-started/installation/)
* [pyproject.toml Specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
