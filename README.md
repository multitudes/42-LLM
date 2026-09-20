This project has been created as part of the 42 curriculum by lbrusa.

# 42-LLM: Local Tool-Calling LLM System

## Description

This project implements an embedded, local Large Language Model (LLM) pipeline designed to autonomously select and format tool/function calls based on natural language user requests. 

The core goal is to enable structural tool invocation natively within Python without relying on external cloud APIs or background server daemons like Ollama. Executing directly inside the process memory via `llm_sdk`, the pipeline parses custom function definitions, constructs ChatML prompts, processes subword tokenization, decodes model outputs, and enforces target types using Pydantic schemas.

---

## Instructions

### System Requirements
* **Python Version:** 3.11 or later
* **Package Manager:** [`uv`](https://docs.astral.sh/uv/) for deterministic environment synchronization
* **Coding Standards:** PEP 8 compliance verified via `flake8`

### Installation

1. Install `uv` (if not already installed):
   ```bash
   curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

```

2. Synchronize project dependencies:
```bash
uv sync

```

---

## Example Usage

Given an input functions definition file (`data/input/functions_definition.json`):

```json
[
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }
]

```

And a test prompt JSON file (`data/input/function_calling_tests.json`):

```json
[
  {
    "prompt": "Reverse the string 'hello'"
  }
]

```
Run the application as a module using `uv`:

```bash
# Run with default file locations
uv run python -m src

# Run with custom input, output, and function definition paths
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_name.json

```

Executing the command generates the structured function call output (`data/output/function_calling_name.json`):

```json
[
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]

```


---

## Algorithm Explanation

The pipeline uses a structured ChatML template combined with post-generation JSON extraction to achieve reliable tool selection:

1. **Schema Ingestion:** Reads function definitions and formats them into a standardized JSON tool list matching OpenAI-compatible schemas.
2. **Context Creation:** Wraps user prompts, available tool specifications, and few-shot examples inside `<|im_start|>` and `<|im_end|>` ChatML tags, suppressing internal reasoning using `/no_think`.
3. **BPE Tokenization:** BPE tokenizes input strings using vocabulary ranks and converts tokens to sequence IDs.
4. **Inference & Decoding:** Generates answer IDs via `llm_sdk` and decodes them back to raw text.
5. **Regex JSON Extraction & Type Enforcement:** Extracts the serialized JSON payload via regular expressions, parses it into Pydantic models (`SelectedFunction`), and forcibly casts argument values (e.g., converting integer outputs to floats for `"type": "number"`) via `enforce_arg_types()`.

---

## Design Decisions

* **Embedded Local Execution:** Selected an embedded SDK runtime over background process daemons (e.g., `llama.cpp` or Ollama) to keep execution entirely self-contained within Python process memory.
* **Pydantic Validation:** Standardized all tool definitions and output objects using Pydantic models (`FunctionDefinition`, `SelectedFunction`, `Tool`) to guarantee runtime type safety.
* **Pydantic V2 Migration:** Standardized schema conversion routines on `model_dump()` and `model_validate()` rather than deprecated V1 methods (`.dict()`).
* **Argparse Configuration:** Integrated standard `argparse` flags (`--functions_definition`, `--input`, `--output`) with fallbacks to default data paths for automated evaluators.

---

## Performance Analysis

* **Accuracy:** Reached high precision on structured tool selection by leveraging ChatML delimiters and targeted few-shot examples in system prompts.
* **Speed:** Offline embedded execution eliminates network latency, enabling token generation to run directly on local GPU/MPS or CPU hardware.
* **Reliability:** Type enforcement routines guarantee that numeric strings or integer outputs returned by the model conform strictly to target JSON types before output serialization.

---

## Challenges Faced

### 1. Regex Generation Hallucinations

* **Issue:** For vowel substitution prompts, the local model generated faulty regex patterns like `\w|aeiou` instead of `[aeiou]`, resulting in all word characters being replaced.
* **Solution:** Added explicit positive few-shot examples demonstrating character sets (`[...]`) and word boundary syntax directly inside system prompts. Providing concrete input/output demonstrations proved far more effective than negative constraints (e.g., "do not use `\w`").

### 2. Schema Structure Updates

* **Issue:** Adapting code when tool definitions transitioned from legacy keys (`fn_name`, `args_names`, `args_types`) to OpenAI-standard fields (`name`, `description`, `parameters`, `returns`).
* **Solution:** Refactored `src/schemas.py` and `src/utils.py` to dynamically construct JSON tool specs from nested dictionary maps (`dict[str, ParameterSchema]`), removing the need for artificial description synthesis.

### 3. Infinite Generation & Memory Edge Cases

* **Issue:** Ambiguous or out-of-domain prompts caused the model to endlessly generate repetitive tokens, leading to Apple Silicon MPS memory allocation exhaustion.
* **Solution:** Enforced strict token limits (`max_new_tokens`) during sequence generation to cut off generation loops cleanly.

---

## Testing Strategy

* **Schema Validation:** Validated schema parsing against varied function signatures (single-argument, multi-argument, string-based, and numeric functions).
* **CLI Customization:** Confirmed path overrides across default and custom directory targets using explicit command-line flags.
* **Edge Case Suite:** Tested model responses against edge cases:
* Extreme numeric inputs (e.g., large integers, floating points)
* String operations with escaped quotes and special characters
* Ambiguous inputs and out-of-domain prompts
* Incomplete argument lists



---

## Resources

* [Flake8 User Guide](https://flake8.pycqa.org/en/latest/index.html)
* [Pydantic Documentation](https://docs.pydantic.dev/)
* [Astral `uv` Project Guide](https://docs.astral.sh/uv/guides/projects/)
* [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

### AI Usage Declaration

AI assistants (Gemini) were used during this project for the following tasks:

* **Pydantic & Runtime Validation:** Assisting in understanding how Pydantic operates at runtime, including data parsing, field validation rules, and migrating models to Pydantic V2 methods (`model_validate` and `model_dump`).
* **Regex Diagnosis:** Identifying root causes of local LLM regex syntax hallucinations (`\w|aeiou` vs `[aeiou]`) and formulating effective few-shot prompt adjustments.
* **Boilerplate Generation:** updating README documentation layout. Spell check. Grammar check. Help with documentation.

---

## Documentation Index

For deep dives into specific sub-components of the project, check the dedicated guides in the `docs/` directory:

| Topic / Module | Description & Link |
| --- | --- |
| **BPE Tokenization** | Custom Byte Pair Encoding implementation and subword splitting → [`docs/bpe.md`](docs/bpe.md) |
| **BPE Pair Merges** | Step-by-step token pair rank evaluation and merging → [`docs/merge.md`](docs/merge.md) |
| **Prompt Engineering** | ChatML control tokens, JSON output formatting, and special tokens usage → [`docs/prompting_json.md`](docs/prompting_json.md) |
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

## Makefile Automation

| Rule | Command | Purpose |
| --- | --- | --- |
| `make install` | `uv sync` | Installs all required dependencies into `.venv` |
| `make run` | `uv run python -m src` | Runs the main tool-calling LLM application |
| `make debug` | `uv run python -m pdb -c continue src/__main__.py` | Executes the application in Python's interactive debugger |
| `make clean` | `rm -rf __pycache__ .venv .pytest_cache` | Cleans temporary cache files and virtual environments |
| `make lint` | `uv run flake8 src` | Audits source code against PEP 8 coding standards |
