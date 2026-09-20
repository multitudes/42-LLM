# Variables
UV := uv
PYTHON := $(UV) run python
MYPY := $(UV) run mypy
FLAKE8 := $(UV) run flake8
PYTEST := $(UV) run pytest

.PHONY: all install run test debug clean lint lint-strict

all: install lint test

install:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "Error: 'uv' is required but not installed."; \
		echo "Please install uv before running setup (see README.md)."; \
		exit 1; \
	}
	@echo "uv version: $$($(UV) --version)"
	@if [ ! -f pyproject.toml ]; then \
		echo "Initializing new uv project..."; \
		$(UV) init; \
	fi
	$(UV) sync

run:
	$(PYTHON) -m src

test:
	$(PYTEST) -v

debug:
	$(PYTHON) -m pdb src/__main__.py

clean:
	@echo "Cleaning temporary cache files and virtual environment..."
	rm -rf .venv .mypy_cache .pytest_cache tests/.ruff_cache output build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(FLAKE8) .
	$(MYPY) .

lint-strict:
	$(FLAKE8) .
	$(MYPY) . --strict