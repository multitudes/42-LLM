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

run:
	@uv run python -m src

debug:
	@uv run python -m pdb -m src

clean:
	@echo "Removing .venv"
	@rm -rf .venv
	@echo "Removing __pycache__"
	@rm -rf src/__pycache__
	@rm -rf llm_sdk/__pycache__

lint:
	uv run flake8 src
	uv run mypy .

lint-strict: 
	flake8 . mypy . --strict

PHONY: install run debug clean lint