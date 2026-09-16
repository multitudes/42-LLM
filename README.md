# 42-LLM: Local Tool-Calling LLM System

I built this project for **42-Global** to run a local Large Language Model (LLM) that autonomously selects and executes Python function tools based on user prompts.

Rather than relying on external API services or standalone inference engines, my setup executes the model **directly inside the Python process** using local runtime libraries.

* **Embedded Execution:** The model weights and token generation run natively within my script's memory space via `llm_sdk`.
* **Zero Network Overhead:** Everything operates completely offline without external cloud calls or server dependencies.
* **No Background Daemons:** Unlike setups requiring Ollama or `llama.cpp` background servers, my application requires no separate process window.

---

## Documentation Index

For deep dives into specific sub-components of my project, check the dedicated guides in the `docs/` directory:

| Topic / Module | Description & Link |
| --- | --- |
| **BPE Tokenization** | Custom Byte Pair Encoding implementation and subword splitting → [`docs/bpe.md`](docs/bpe.md) |
| **BPE Pair Merges** | Step-by-step token pair rank evaluation and merging → [`docs/merge.md`](docs/merge.md) |
| **Prompt Engineering** | ChatML control tokens, JSON output formatting, and `/no_think` usage → [`docs/prompting_json.md`](docs/prompting_json.md) |
| **Prompt Examples** | Concrete tool-calling input/output execution samples → [`docs/example-prompts.md`](docs/example-prompts.md) |
| **LLM SDK** | Integration guidelines and constraints for the `llm_sdk` runtime → [`docs/llm_sdk.md`](docs/llm_sdk.md) |
| **Ollama vs Local SDK** | Architectural comparison between standalone daemons and embedded runtime → [`docs/ollama.md`](docs/ollama.md) |
| **Pydantic Validation** | Data models, type checking, and schema enforcement → [`docs/pydantic.md`](docs/pydantic.md) |
| **Environment Management** | Virtual environment isolation and synchronization using `uv` → [`docs/uv.md`](docs/uv.md) |
| **Dependency Management** | Approved Python packages (`numpy`, `pydantic`) and forbidden tools → [`docs/packages.md`](docs/packages.md) |
| **Generation Logs** | Understanding HuggingFace model startup output and token IDs → [`docs/hugginface.md`](docs/hugginface.md) |
| **Debugging** | Interactive troubleshooting using `breakpoint()` and Python's `pdb` → [`docs/python-debugger.md`](docs/python-debugger.md) |
| **Developer Hints** | Useful tips for exception handling, linting, and resource management → [`docs/hints.md`](docs/hints.md) |

---

## System Requirements & Guidelines

I designed the codebase to strictly adhere to 42 school specifications:

* **Python Version:** Python 3.11 or later.
* **Coding Standards:** PEP 8 compliance checked strictly via `flake8`.
* **Error Handling:** All routines use `try-except` blocks to manage exceptions gracefully and prevent unexpected crashes.
* **Resource Safety:** File handles and process memory are properly released to avoid leaks.
* **Validation:** All data structures and class inputs are validated using [`pydantic`](docs/pydantic.md).
* **Tool Selection:** Tool calls are decided exclusively through LLM reasoning (no rigid heuristics or manual string matching).

### Package & Dependency Constraints

* **Allowed Dependencies:** `numpy`, `pydantic`, and `llm_sdk` (placed at root alongside `src/`).
* **Forbidden Libraries:** `dspy`, standard `transformers`, `torch` imports, or high-level agent frameworks.
* **Target Model:** `ollama_chat/qwen3:0.6b` (default engine provided via `llm_sdk`).

---

## Installation & Execution

I use [`uv`](docs/uv.md) for ultra-fast, deterministic virtual environment setup and dependency synchronization.

### 1. Install `uv` (Linux / macOS)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

### 2. Install Project Dependencies

```bash
uv sync

```

### 3. Run the Main Application

Per project requirements, the main entry point is executed as a module:

```bash
uv run python -m src

```

*Note: Running `python -m src` executes `src/__main__.py`, ensuring standard package-level execution rather than running loose scripts.*

---

## Makefile Automation

My `Makefile` provides standardized targets for building, running, and auditing the repository:

| Rule | Command | Purpose |
| --- | --- | --- |
| `make install` | `uv sync` | Installs all required dependencies into `.venv` |
| `make run` | `uv run python -m src` | Runs the main tool-calling LLM application |
| `make debug` | `uv run python -m pdb -c continue src/__main__.py` | Executes the application in Python's interactive debugger |
| `make clean` | `rm -rf __pycache__ .venv .pytest_cache` | Cleans temporary cache files and virtual environments |
| `make lint` | `uv run flake8 src` | Audits source code against PEP 8 coding standards |

---

## Understanding Model Initialization Logs

When initializing the local LLM runtime, the system outputs configuration details similar to this:

```text
d81485cdf75e47ca/generation_config.json
Generate config GenerationConfig {
  "bos_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "pad_token_id": 151643,
}

```

This informational logging originates from the underlying generation configuration loaded by the SDK:

* `bos_token_id`: **Beginning of Sequence** token ID (`151643`).
* `eos_token_id`: **End of Sequence** token IDs (`151645`, `151643`). Output generation stops when the model emits one of these IDs.
* `pad_token_id`: **Padding** token ID (`151643`), used to equalize batch array dimensions.
* `do_sample`: Set to `True` to enable probabilistic sampling during generation.

For further analysis of generation flags, refer to [`docs/hugginface.md`](docs/hugginface.md).

---

## External References

* [Flake8 User Guide](https://flake8.pycqa.org/en/latest/index.html)
* [Pydantic Documentation](https://docs.pydantic.dev/)
* [Astral `uv` Project Guide](https://docs.astral.sh/uv/guides/projects/)