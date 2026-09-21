*This project has been created as part of the 42 curriculum by lbrusa.*

# 42-LLM: Local Tool-Calling LLM System

## Description

This project implements an embedded, local Large Language Model (LLM) pipeline that selects and formats tool/function calls from natural language prompts.

The goal is structural tool invocation inside a single Python process—without cloud APIs or background daemons such as Ollama. Using `llm_sdk` (Hugging Face Transformers), the pipeline loads function definitions, builds ChatML prompts with few-shot examples, runs custom BPE tokenization, greedily decodes with stop tokens, extracts the first balanced JSON object, validates the chosen tool against the schema, and coerces argument types with Pydantic models before writing JSON output.

---

## Instructions

### System Requirements

* **Python:** 3.10 or later (`pyproject.toml` requires `>=3.10`; development target is 3.12)
* **Package manager:** [`uv`](https://docs.astral.sh/uv/) for locked, reproducible environments
* **Lint / types:** `flake8` and `mypy` via the Makefile; `ruff` is also available as a dev dependency

### Installation

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available (for example on macOS: `brew install uv`). Prefer the official installer documentation over piping remote scripts into a shell.
2. From the repository root, synchronize dependencies:

```bash
uv sync
# or: make install
```

3. (Optional) Run checks:

```bash
make lint
make test
```

### Execution

```bash
# Defaults: data/input/functions_definition.json, data/input/function_calling_tests.json,
#           data/output/function_calls.json
uv run python -m src
# or: make run

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_name.json
```

The first run downloads the Qwen3-0.6B tokenizer/model assets from the Hugging Face Hub into the local cache.

---

## Example Usage

Given `data/input/functions_definition.json`:

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

And `data/input/function_calling_tests.json`:

```json
[
  {
    "prompt": "Reverse the string 'hello'"
  }
]
```

Run:

```bash
uv run python -m src
```

Example structured output (`data/output/function_calls.json`):

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

The pipeline uses **constrained decoding via prompt structure and post-generation validation**, not logit masking:

1. **Schema ingestion:** Load function definitions into Pydantic `FunctionDefinition` models and serialize an OpenAI-style tools JSON string for the system prompt (`get_tool_list`).
2. **Constrained prompt construction:** Build a ChatML conversation (`<|im_start|>` / `<|im_end|>`) with:
   * tool schemas and few-shot `<tool_call>…</tool_call>` examples in the system turn;
   * sanitized user text (control tokens stripped) in the user turn;
   * a **prefilled assistant prefix** `<|im_start|>assistant\n<tool_call>\n` so generation continues inside a tool-call JSON payload and skips chain-of-thought (`<think>`) turns.
3. **Custom BPE tokenization:** Encode the full prompt with Hub `tokenizer.json` vocabulary and Hub `merges.txt` ranks (`initialize_tokenizer` + `bpe_tokenize`).
4. **Greedy generation with stops:** At each step, take `argmax` over next-token logits from `llm_sdk`, append the token, and stop on `</tool_call>`, `<|im_end|>`, or `<|endoftext|>`, or after a hard `MAX_TOKENS` budget.
5. **Balanced JSON extraction:** Parse the first complete JSON object with `json.JSONDecoder.raw_decode` (not a greedy `\{.*\}` regex), validate against `SelectedFunction`, reject unknown tool names and missing required parameters, then coerce types with `enforce_arg_types`.
6. **Serialize:** Write the list of validated calls to the output JSON file (write failures propagate so the process exits non-zero).

---

## Design Decisions

* **Embedded local runtime:** Prefer in-process `llm_sdk` over Ollama/`llama.cpp` daemons so evaluation stays self-contained.
* **Prompt-level constraints over logit masks:** Prefill `<tool_call>` and few-shot schemas instead of vocabulary masking—compatible with the provided SDK surface (`get_logits_from_input_ids`).
* **Pydantic V2 schemas:** `FunctionDefinition`, `SelectedFunction`, and OpenAI-style `Tool*` models with `model_validate` / `model_dump`.
* **Custom BPE parity path:** Own tokenizer for the assignment, initialized from Hub tokenizer/merges files used at runtime.
* **CLI via argparse:** `--functions_definition`, `--input`, `--output` with defaults under `data/` for automated graders.
* **Fail closed on bad tool calls:** Unknown function names or missing required args become empty `SelectedFunction` entries rather than invented calls.

---

## Performance Analysis

* **Accuracy:** ChatML delimiters, few-shot tool-call examples (including regex character-set demos), and schema validation keep structured selection reliable on the exercise prompts.
* **Speed:** No network round-trips at inference time, but each generated token re-forwards the full growing sequence (no KV cache) and materializes ~152k vocab logits as a Python list before `max(enumerate(...))`. These SDK-level bottlenecks dominate latency; see [`docs/performance.md`](docs/performance.md).
* **Reliability:** Type coercion, required-parameter checks, and re-raised output write errors reduce silent grader failures. Hub downloads and `trust_remote_code=True` remain supply-chain assumptions for the fixed Qwen model id.

---

## Challenges Faced

### 1. Regex generation hallucinations

* **Issue:** For vowel substitution, the model emitted patterns like `\w|aeiou` instead of `[aeiou]`.
* **Solution:** Positive few-shot examples in the system prompt showing character sets and `\d+` / `\b` usage.

### 2. Schema structure updates

* **Issue:** Tool definitions moved from legacy keys (`fn_name`, `args_names`, …) to OpenAI-style `name` / `parameters` / `returns`.
* **Solution:** Refactored `src/schemas.py` and `src/utils.py` around nested `ParameterSchema` maps.

### 3. Infinite generation and memory pressure

* **Issue:** Out-of-domain prompts caused long repetitive generations and MPS memory pressure.
* **Solution:** Hard `MAX_TOKENS` cap plus stop-token IDs for `</tool_call>` and ChatML end markers.

### 4. Suppressing chain-of-thought without `/no_think`

* **Issue:** Prefacing with reasoning tags wasted tokens and broke JSON extraction.
* **Solution:** Prefill the assistant turn with `<tool_call>\n` so decoding starts inside the required JSON envelope.

---

## Testing Strategy

* **Unit tests (`pytest`):** Schema loading, JSON extraction (balanced objects, missing required args, invalid payloads), type coercion edge cases, prompt-injection sanitization, and BPE unknown-token behavior.
* **Tokenizer parity:** Compare custom encode/decode against Hugging Face on representative strings (`tests/test_tokenizer_parity.py`).
* **CLI / pipeline stubs:** Monkeypatch heavy model loads and I/O in `tests/test_main.py` / `tests/test_tokenizer.py`.
* **Manual runs:** Execute `uv run python -m src` against `data/input/function_calling_tests.json` and inspect `data/output/`.

```bash
uv run pytest -v
# or: make test
```

---

## Resources

* [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
* [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
* [Pydantic Documentation](https://docs.pydantic.dev/)
* [Astral `uv` Project Guide](https://docs.astral.sh/uv/guides/projects/)
* [Flake8 User Guide](https://flake8.pycqa.org/en/latest/index.html)
* [Qwen ChatML / tokenizer notes](https://huggingface.co/Qwen)

### AI Usage Declaration

AI assistants (Gemini, Cursor) were used during this project for:

* **Pydantic & runtime validation:** Understanding parsing, field rules, and V2 `model_validate` / `model_dump` migration.
* **Regex diagnosis:** Root-causing hallucinated patterns (`\w|aeiou` vs `[aeiou]`) and drafting few-shot prompt fixes.
* **Documentation:** README/layout polish, spell check, grammar, and aligning docs with the implemented pipeline.
* **Refactor assistance:** Code review triage and test updates around tokenizer/utils changes.

---

## Documentation Index

| Topic / Module | Description & Link |
| --- | --- |
| **BPE Tokenization** | Custom Byte Pair Encoding → [`docs/bpe.md`](docs/bpe.md) |
| **BPE Pair Merges** | Token pair rank merging → [`docs/merge.md`](docs/merge.md) |
| **Prompt Engineering** | ChatML and JSON formatting → [`docs/prompting_json.md`](docs/prompting_json.md) |
| **Prompt Examples** | Tool-calling prompt samples → [`docs/example-prompts.md`](docs/example-prompts.md) |
| **LLM SDK** | Embedded runtime constraints → [`docs/llm_sdk.md`](docs/llm_sdk.md) |
| **Ollama vs Local SDK** | Daemon vs in-process → [`docs/ollama.md`](docs/ollama.md) |
| **Pydantic Validation** | Schemas and type checks → [`docs/pydantic.md`](docs/pydantic.md) |
| **Environment Management** | `uv` workflow → [`docs/uv.md`](docs/uv.md) |
| **Package Structure** | Modules, packages, imports → [`docs/packages.md`](docs/packages.md) |
| **Performance** | KV cache / logits bottlenecks → [`docs/performance.md`](docs/performance.md) |
| **Generation Logs** | Hugging Face startup output → [`docs/hugginface.md`](docs/hugginface.md) |
| **Debugging** | `pdb` / breakpoints → [`docs/python-debugger.md`](docs/python-debugger.md) |
| **Developer Hints** | Exceptions, linting tips → [`docs/hints.md`](docs/hints.md) |

---

## Makefile Automation

| Rule | Command | Purpose |
| --- | --- | --- |
| `make install` | `uv sync` | Install dependencies into `.venv` (requires `uv` already installed) |
| `make run` | `uv run python -m src` | Run the tool-calling pipeline |
| `make test` | `uv run pytest -v` | Run the test suite |
| `make debug` | `uv run python -m pdb src/__main__.py` | Run under `pdb` |
| `make lint` | `uv run flake8 .` and `uv run mypy .` | Style and type checks |
| `make clean` | remove caches / `.venv` | Clean temporary build artifacts |
| `make fclean` | `make clean` + wipe HF Hub cache | Also deletes `~/.cache/huggingface/hub` (models re-download on next run) |
