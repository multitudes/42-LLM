# 42-LLM-beta code review

Static review of the local tool-calling pipeline (`src/`, `llm_sdk/`, `tests/`, `docs/`). Source: repository files as of 18 Sep 2026. Counts are distinct findings, not automated scanner hits.

| Findings | High | Medium | Low |
| --- | --- | --- | --- |
| 30 | 5 | 11 | 14 |

**Highest-impact cluster:** the production tokenizer is not the one the parity tests exercise. Hub merges are downloaded and ignored, local `merges.txt` is CWD-relative, and unknown tokens are dropped. That can change model inputs without failing tests.

## Findings by category

| Category | Count |
| --- | --- |
| Bugs / logic | 6 |
| Edge cases | 5 |
| Error handling | 2 |
| Dead code / unused | 6 |
| Security | 3 |
| Performance | 5 |
| Submission polish | 3 |

## Index

| Sev | Category | Finding | Location |
| --- | --- | --- | --- |
| high | Bugs / logic | Tokenizer ignores SDK merge path | `src/bpe_tokenizer.py`, `src/__main__.py` |
| high | Bugs / logic | Unknown BPE tokens are dropped | `src/bpe_tokenizer.py` · `bpe_tokenize` |
| high | Bugs / logic | Greedy regex can swallow extra braces | `src/utils.py` · `extract_json_from_response` |
| high | Error handling | Output write failures are swallowed | `src/utils.py` · `write_output_to_file` |
| high | Edge cases | Integer coercion silently keeps bad values | `src/utils.py` · `enforce_arg_types` |
| medium | Bugs / logic | Broken few-shot example in the system prompt | `src/bpe_tokenizer.py` · `create_prompt` |
| medium | Error handling | `main()` only catches `RuntimeError` | `src/__main__.py` |
| medium | Edge cases | Selected tool name is never validated | `src/utils.py` · `extract_json_from_response` |
| medium | Edge cases | Every parameter is marked required | `src/utils.py` · `get_tool_list` |
| medium | Security | User text is interpolated into ChatML raw | `src/bpe_tokenizer.py` · `create_prompt` |
| medium | Security | `trust_remote_code` defaults to `True` | `llm_sdk/__init__.py` |
| medium | Security | Makefile and README pipe curl into sh | `makefile`, `README.md` |
| medium | Performance | Generation re-runs the full sequence every token | `src/bpe_tokenizer.py` · `get_answer_ids`, `llm_sdk` |
| medium | Performance | Full vocab logits copied into Python each step | `llm_sdk/__init__.py` · `get_logits_from_input_ids` |
| medium | Submission polish | README is wrapped in a markdown code fence | `README.md` |
| medium | Bugs / logic | Parity tests do not exercise production tokenize path | `tests/test_tokenizer_parity.py` vs `src/__main__.py` |
| low | Performance | Function definitions re-read per prompt | `src/utils.py` · `extract_json_from_response` |
| low | Dead code / unused | `THINK_TAG` is unused | `src/utils.py` |
| low | Dead code / unused | Stale end-token constants | `src/bpe_tokenizer.py` |
| low | Dead code / unused | Unused imports in `llm_sdk` | `llm_sdk/__init__.py` |
| low | Dead code / unused | Downloaded vocab/merges paths are print-only | `src/__main__.py` |
| low | Dead code / unused | `SPECIAL_TOKENS` empty-branch is dead | `src/bpe_tokenizer.py` · `bpe_tokenize` |
| low | Bugs / logic | `custom_decode` docstring disagrees with code | `src/bpe_tokenizer.py` · `custom_decode` |
| low | Submission polish | Python version and linters disagree | `README.md`, `pyproject.toml`, `makefile` |
| low | Dead code / unused | Leftover data and a spaced filename | `old-exercise_input/`, `data/output/` |
| low | Submission polish | Docs show a module that does not exist | `docs/packages.md` |
| low | Performance | Unit tests load the full 0.6B model | `tests/test_tokenizer.py`, `tests/test_tokenizer_parity.py` |
| low | Performance | CUDA path may double-move the model | `llm_sdk/__init__.py` |
| low | Edge cases | `get_answer_ids` mutates `input_ids` in place | `src/bpe_tokenizer.py` · `get_answer_ids` |
| low | Edge cases | Empty token list is unhandled in the SDK | `llm_sdk/__init__.py` · `get_logits_from_input_ids` |

## What to fix first

### Before a school submit

Wire `initialize_tokenizer` to the downloaded merges path (or document why local `merges.txt` is required). Stop dropping OOV tokens. Parse JSON from the first balanced object, not a greedy regex. Re-raise write errors.

Unwrap `README.md`, replace `<login1>`, and point docs links at real files. Remove `function_calling_name copy.json` and `old-exercise_input/`.

### Nice for the write-up

Mention greedy decoding without a KV cache, and logits copied to a Python list, as the main speed limits. `trust_remote_code=True` and unpinned Hub revisions are the honest security notes — this is a local assignment, not a networked app.

Align parity tests with `initialize_tokenizer()` so they actually cover the submitted tokenizer.

## Finding notes

### Tokenizer ignores SDK merge path

- **Severity:** high
- **Category:** Bugs / logic
- **Where:** `src/bpe_tokenizer.py`, `src/__main__.py`

`main()` downloads the Hub merges file, then never uses it. `initialize_tokenizer()` always reads `MERGES_PATH = "merges.txt"` from the current working directory. Running from any other directory, or drifting from the Hub merges, silently tokenizes with the wrong rules.

### Unknown BPE tokens are dropped

- **Severity:** high
- **Category:** Bugs / logic
- **Where:** `src/bpe_tokenizer.py` · `bpe_tokenize`

The final map is `[vocab[token] for token in tokens if token in vocab]`. Characters or merged pieces missing from vocab vanish instead of becoming `<unk>` or a byte fallback. The model then sees a mutated prompt.

### Greedy regex can swallow extra braces

- **Severity:** high
- **Category:** Bugs / logic
- **Where:** `src/utils.py` · `extract_json_from_response`

Pattern `r'\{.*\}'` with `re.DOTALL` takes from the first `{` to the last `}`. Nested objects, trailing junk, or a second JSON blob make `json.loads` fail and the pipeline returns an empty `SelectedFunction`.

### Output write failures are swallowed

- **Severity:** high
- **Category:** Error handling
- **Where:** `src/utils.py` · `write_output_to_file`

`except Exception` prints and returns. `main()` still finishes with exit code 0, so a grader can see a successful run and a missing or stale output file.

### Integer coercion silently keeps bad values

- **Severity:** high
- **Category:** Edge cases
- **Where:** `src/utils.py` · `enforce_arg_types`

`int("11.0")` and `int("11.9")` raise and are ignored, leaving strings in the output. A float `11.9` becomes `11` via truncation. The assignment's type-enforcement story is weaker than the README claims.

### Broken few-shot example in the system prompt

- **Severity:** medium
- **Category:** Bugs / logic
- **Where:** `src/bpe_tokenizer.py` · `create_prompt`

The digit-substitution example inserts a stray `</tool_call>` before the Assistant turn. That teaches the model a malformed ChatML / tool-call pattern on the hardest prompt class.

### `main()` only catches `RuntimeError`

- **Severity:** medium
- **Category:** Error handling
- **Where:** `src/__main__.py`

`get_functions()` raises `TypeError` when the tools file is not a list. Model load, CUDA/MPS, and Hub download errors are also uncaught. Some failures exit 1 with a short message; others dump a traceback.

### Selected tool name is never validated

- **Severity:** medium
- **Category:** Edge cases
- **Where:** `src/utils.py` · `extract_json_from_response`

If the model invents a function name, the code still serializes it. `enforce_arg_types()` no-ops when the name is missing from the definitions, so hallucinated tools look like successful calls.

### Every parameter is marked required

- **Severity:** medium
- **Category:** Edge cases
- **Where:** `src/utils.py` · `get_tool_list`

`required=list(fn.parameters.keys())` even when a definition might omit optional args. Combined with no post-check that required keys are present, incomplete calls still write out.

### User text is interpolated into ChatML raw

- **Severity:** medium
- **Category:** Security
- **Where:** `src/bpe_tokenizer.py` · `create_prompt`

`user_input` is f-string inserted between `<|im_start|>user` and `<|im_end|>`. A prompt containing those tags can close the user turn early. Fine for the given test set; worth knowing for a security write-up.

### `trust_remote_code` defaults to `True`

- **Severity:** medium
- **Category:** Security
- **Where:** `llm_sdk/__init__.py`

`AutoTokenizer` / `AutoModelForCausalLM` will execute Hub-hosted Python for the model. Acceptable for Qwen, but it is a supply-chain footgun if `model_name` is ever changed. Hub files are also unpinned by revision.

### Makefile and README pipe curl into sh

- **Severity:** medium
- **Category:** Security
- **Where:** `makefile`, `README.md`

`uv` is installed with `curl | sh`. Common, but a reviewer looking for school-assignment security notes will flag it. Prefer a documented installer or a versioned binary.

### Generation re-runs the full sequence every token

- **Severity:** medium
- **Category:** Performance
- **Where:** `src/bpe_tokenizer.py` · `get_answer_ids`, `llm_sdk`

Each step calls `get_logits_from_input_ids` on the growing list with no `past_key_values`. Cost is roughly O(prompt_len × MAX_TOKENS). The few-shot system prompt makes this especially expensive.

### Full vocab logits copied into Python each step

- **Severity:** medium
- **Category:** Performance
- **Where:** `llm_sdk/__init__.py` · `get_logits_from_input_ids`

Qwen3-0.6B vocab is ~152k. Every token does `.tolist()` then `max(enumerate(logits))`. That is 152k Python floats × up to 92 tokens × 11 prompts, plus a Python argmax. Keep logits on device and use `torch.argmax`.

### README is wrapped in a markdown code fence

- **Severity:** medium
- **Category:** Submission polish
- **Where:** `README.md`

The file starts with prose then a ` ```markdown ` block containing the real README, plus `<login1>` and docs links that point at Google searches. That is the first file a 42 reviewer opens.

### Parity tests do not exercise production tokenize path

- **Severity:** medium
- **Category:** Bugs / logic
- **Where:** `tests/test_tokenizer_parity.py` vs `src/__main__.py`

Parity loads `vocab.json` + Hub `merges.txt`. Production uses `tokenizer.json` + local `merges.txt` via `initialize_tokenizer()`. Tests can pass while the submitted pipeline tokenizes differently.

### Function definitions re-read per prompt

- **Severity:** low
- **Category:** Performance
- **Where:** `src/utils.py` · `extract_json_from_response`

`get_functions()` opens and validates the tools JSON on every extraction, after `get_tool_list()` already loaded it. Cheap on disk, wasteful and a place for the file to change mid-run.

### `THINK_TAG` is unused

- **Severity:** low
- **Category:** Dead code / unused
- **Where:** `src/utils.py`

`THINK_TAG = "</think>"` is never referenced. `test_extract_json_from_response_with_think` still passes because the greedy JSON regex happens to find the object after the tag.

### Stale end-token constants

- **Severity:** low
- **Category:** Dead code / unused
- **Where:** `src/bpe_tokenizer.py`

`END_TOKEN_ID1 = 3417` and `END_TOKEN_ID2 = 30975` are unused leftovers. Stopping uses `STOP_TOKEN_IDS` instead.

### Unused imports in `llm_sdk`

- **Severity:** low
- **Category:** Dead code / unused
- **Where:** `llm_sdk/__init__.py`

`os`, `time`, and `typing.Tuple` are imported and never used (ruff F401). Also `typing.Tuple` is the old spelling under a 3.10+ codebase.

### Downloaded vocab/merges paths are print-only

- **Severity:** low
- **Category:** Dead code / unused
- **Where:** `src/__main__.py`

`merge_path` and `vocab_path` are fetched and printed. Tokenizer init uses `tokenizer.json` plus local `merges.txt`. The extra Hub downloads cost time and look like they are wired up when they are not.

### `SPECIAL_TOKENS` empty-branch is dead

- **Severity:** low
- **Category:** Dead code / unused
- **Where:** `src/bpe_tokenizer.py` · `bpe_tokenize`

`if SPECIAL_TOKENS:` is always true for the module-level dict. The `else: parts = [text]` path never runs.

### `custom_decode` docstring disagrees with code

- **Severity:** low
- **Category:** Bugs / logic
- **Where:** `src/bpe_tokenizer.py` · `custom_decode`

The docstring says special token IDs are filtered out. The implementation keeps them, and tests assert that. Graders reading comments will score this as sloppy.

### Python version and linters disagree

- **Severity:** low
- **Category:** Submission polish
- **Where:** `README.md`, `pyproject.toml`, `makefile`

README requires 3.11 and flake8; pyproject requires `>=3.10` and configures ruff + mypy; makefile runs flake8 and mypy. Easy inconsistency points on a rubric.

### Leftover data and a spaced filename

- **Severity:** low
- **Category:** Dead code / unused
- **Where:** `old-exercise_input/`, `data/output/`

`old-exercise_input/` and `data/output/function_calling_name copy.json` look like local scratch, not submission artifacts.

### Docs show a module that does not exist

- **Severity:** low
- **Category:** Submission polish
- **Where:** `docs/packages.md`

Example import `from llm_sdk.ollama import call_ollama_api` has no matching code. README algorithm text still mentions `/no_think`, which `create_prompt` does not use.

### Unit tests load the full 0.6B model

- **Severity:** low
- **Category:** Performance
- **Where:** `tests/test_tokenizer.py`, `tests/test_tokenizer_parity.py`

`test_tokenizer_initialization` and parity fixtures instantiate `Small_LLM_Model()`. That is slow, needs Hub cache, and is more integration than unit. `extract_json` tests are also duplicated in `test_utils.py`.

### CUDA path may double-move the model

- **Severity:** low
- **Category:** Performance
- **Where:** `llm_sdk/__init__.py`

`from_pretrained(..., device_map="auto")` then `self._model.to(self._device)`. On CUDA that can conflict with accelerate's map. MPS/CPU skip `device_map`, which is fine.

### `get_answer_ids` mutates `input_ids` in place

- **Severity:** low
- **Category:** Edge cases
- **Where:** `src/bpe_tokenizer.py` · `get_answer_ids`

Documented, and currently each prompt builds a fresh list. Reusing a list later would mix prompt and generated tokens.

### Empty token list is unhandled in the SDK

- **Severity:** low
- **Category:** Edge cases
- **Where:** `llm_sdk/__init__.py` · `get_logits_from_input_ids`

`torch.tensor([[]])` / `logits[0, -1]` is not guarded. Unlikely on the happy path, but a tokenize-everything-dropped prompt could reach it after the OOV drop bug.

## Method notes

Unused-import check: ruff F401 on `llm_sdk/__init__.py` (`os`, `time`, `Tuple`). No F401 in `src/`. Review did not execute the model.
