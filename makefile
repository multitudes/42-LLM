.PHONY: install run debug clean lint lint-strict

install:
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv not found. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "uv version: $$(uv --version)"
	@if [ ! -f pyproject.toml ]; then \
		uv init; \
		echo "uv project initialized. Edit pyproject.toml if needed"; \
	else \
		echo "uv project already initialized"; \
	fi
	uv sync

run:
	uv run python -m src

test:
	uv run pytest -v

debug:
	uv run python -m pdb -m src

clean:
	@echo "Cleaning temporary cache files and virtual environment..."
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .venv
	rm -rf output
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict