install:
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv not found. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "uv version: $$(uv --version)"
	

run:
	@echo "Running"
debug:
	@echo "Debugging"
clean:
	@echo "Cleaning"
lint:
	@echo "Linting"

PHONY: install run debug clear lint