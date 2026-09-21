# 42-LLM-beta code review (status)

Re-review of `dev` @ `e8a4688` (21 Sep 2026). Original static review dated 18 Sep 2026; this document tracks what remains open after the recent tokenizer / utils / docs refactor.

| Status | High | Medium | Low | Total |
| --- | --- | --- | --- | --- |
| **Open** | 0 (2 medium residuals of former highs) | 7 | 4 | **11** |
| **Resolved** | 5 | 7 | 9 | **21** |

**Highest remaining cluster:** parity tests still load `vocab.json` instead of the production `initialize_tokenizer(tokenizer.json, Hub merges)` path, and unknown BPE tokens can still be dropped when `"<unk>"` is absent from the vocab (common for Qwen byte-level tokenizers).

## Open findings

| Sev | Category | Finding | Location |
| --- | --- | --- | --- |
| medium | Bugs / logic | Parity tests do not exercise production tokenize path | `tests/test_tokenizer_parity.py` vs `src/__main__.py` |
| medium (residual) | Bugs / logic | OOV tokens still dropped if `<unk>` missing | `src/bpe_tokenizer.py` · `bpe_tokenize` |
| medium (residual) | Edge cases | Unparseable values kept via `except: pass` | `src/utils.py` · `enforce_arg_types` |
| medium | Edge cases | Every parameter is marked required | `src/utils.py` · `get_tool_list` |
| medium | Security | `trust_remote_code` defaults to `True` | `llm_sdk/__init__.py` |
| medium | Performance | Generation re-runs the full sequence every token | `src/bpe_tokenizer.py` · `get_answer_ids`, `llm_sdk` |
| medium | Performance | Full vocab logits copied into Python each step | `llm_sdk/__init__.py` · `get_logits_from_input_ids` |
| low | Performance | CUDA path may double-move the model | `llm_sdk/__init__.py` |
| low | Edge cases | Empty token list is unhandled in the SDK | `llm_sdk/__init__.py` · `get_logits_from_input_ids` |
| low | Dead code / unused | Commented blocks and unused imports | `src/bpe_tokenizer.py`, `src/__main__.py`, `llm_sdk/__init__.py` |
| low | Performance | Parity / init tests still hit Hub tokenizer; weak duplicate test | `tests/test_tokenizer_parity.py`, `tests/test_tokenizer.py` |

## What to fix next

### Before a school submit

Align parity fixtures with `initialize_tokenizer()` so they cover the submitted path. Fix or document the `<unk>`-missing OOV fallback.

### Nice for the write-up

KV cache absence and logits `.tolist()` + Python `max` remain the main speed limits (`docs/performance.md` already covers them). `trust_remote_code=True` and unpinned Hub revisions are honest security notes for a local assignment. Optional: default `trust_remote_code=False`, keep logits on device with `torch.argmax`, and strip dead commented code / unused `llm_sdk` imports.

---

## Open finding notes

### Parity tests do not exercise production tokenize path

- **Severity:** medium
- **Category:** Bugs / logic
- **Where:** `tests/test_tokenizer_parity.py` vs `src/__main__.py`

Production calls `initialize_tokenizer(tokenizer.json, Hub merges.txt)`. Parity loads `vocab.json` via `get_path_to_vocab_file()` and a local `load_merges_txt`, never `initialize_tokenizer()`. Tests can pass while the submitted pipeline tokenizes from a different vocab source.

### OOV tokens still dropped if `<unk>` missing

- **Severity:** medium (residual of original high)
- **Category:** Bugs / logic
- **Where:** `src/bpe_tokenizer.py` · `bpe_tokenize` (~201–207)

When `"<unk>"` is in vocab, unknown pieces map to that ID. When it is absent, the code falls back to `[vocab[token] for token in tokens if token in vocab]`, which still drops OOVs. Qwen byte-level vocabs often lack an `"<unk>"` key, so this path can fire in production.

### Unparseable values kept via `except: pass`

- **Severity:** medium (residual of original high)
- **Category:** Edge cases
- **Where:** `src/utils.py` · `enforce_arg_types`

Integer coercion via `int(round(float(val)))` fixed the `"11.0"` / truncation cases. Coercion failures still swallow `ValueError` / `TypeError` and leave the original value (e.g. `"invalid_integer_string"`). Tests treat this as intended; either document it as policy or fail/clear the tool call.

### Every parameter is marked required

- **Severity:** medium (reduced impact)
- **Category:** Edge cases
- **Where:** `src/utils.py` · `get_tool_list`

`required=list(fn.parameters.keys())` remains. Runtime now rejects missing keys before serialize. Still open only if optional parameters are in scope; `FunctionDefinition` has no optional-args field.

### `trust_remote_code` defaults to `True`

- **Severity:** medium
- **Category:** Security
- **Where:** `llm_sdk/__init__.py`

`AutoTokenizer` / `AutoModelForCausalLM` will execute Hub-hosted Python. Acceptable for Qwen; supply-chain footgun if `model_name` changes. Hub files remain unpinned by revision. Speed notes live in `docs/performance.md`; README Performance Analysis now mentions `trust_remote_code=True` as a supply-chain assumption.

### Generation re-runs the full sequence every token

- **Severity:** medium
- **Category:** Performance
- **Where:** `src/bpe_tokenizer.py` · `get_answer_ids`, `llm_sdk`

Each step calls `get_logits_from_input_ids` on the growing list with no `past_key_values`. Documented in `docs/performance.md`; code unchanged.

### Full vocab logits copied into Python each step

- **Severity:** medium
- **Category:** Performance
- **Where:** `llm_sdk/__init__.py` · `get_logits_from_input_ids`; `src/bpe_tokenizer.py` · `get_answer_ids`

Qwen3-0.6B vocab is ~152k. Every token does `.tolist()` then `max(enumerate(logits))`. Prefer keeping logits on device and using `torch.argmax`.

### CUDA path may double-move the model

- **Severity:** low
- **Category:** Performance
- **Where:** `llm_sdk/__init__.py`

`from_pretrained(..., device_map="auto")` then `self._model.to(self._device)`. On CUDA that can conflict with accelerate's map. MPS/CPU skip `device_map`.

### Empty token list is unhandled in the SDK

- **Severity:** low
- **Category:** Edge cases
- **Where:** `llm_sdk/__init__.py` · `get_logits_from_input_ids`

`torch.tensor([[]])` / `logits[0, -1]` is not guarded. More likely if the OOV-drop path empties the prompt.

### Commented blocks and unused imports

- **Severity:** low
- **Category:** Dead code / unused
- **Where:** `src/bpe_tokenizer.py`, `src/__main__.py`, `llm_sdk/__init__.py`

Large commented blocks remain in `bpe_tokenizer.py` (`gpt2_bytes_to_unicode`, old `custom_decode`, `load_control_tokens`). `__main__.py` has a commented `vocab_path` download. `llm_sdk` still imports unused `os`, `time`, and `typing.Tuple`.

### Parity / init tests still hit Hub tokenizer; weak duplicate test

- **Severity:** low
- **Category:** Performance
- **Where:** `tests/test_tokenizer_parity.py`, `tests/test_tokenizer.py`

`DummyModel` avoids weight load, but `Small_LLM_Model()` still downloads the Hub tokenizer. Duplicate `test_extract_json_missing_required_parameter` in `test_tokenizer.py` uses `functions=[]` and mismatched names (`fn_add` vs `fn_add_numbers`); the stronger copy lives in `test_utils.py`.

---

## Resolved findings

| Sev | Finding | Evidence |
| --- | --- | --- |
| high | Tokenizer ignored SDK merge path | `main()` passes `llm.get_path_to_merges_file()` into `initialize_tokenizer()` |
| high | Unknown BPE tokens dropped (main case) | `bpe_tokenize` maps OOVs to `vocab["<unk>"]` when present; covered by `tests/test_tokenizer.py` |
| high | Greedy regex swallowed extra braces | Replaced by `extract_first_json_string()` using `json.JSONDecoder.raw_decode` |
| high | Output write failures swallowed | `write_output_to_file` re-raises; `main()` exits `1` on expected and unexpected errors |
| high | Integer coercion kept bad values / truncated | Integer path is `int(round(float(val)))`; tests in `tests/test_utils.py` |
| medium | Broken few-shot `</tool_call>` in system prompt | Digit-substitution example in `create_prompt` is well-formed ChatML |
| medium | Selected tool name never validated | Unknown names become empty `SelectedFunction` |
| medium | Missing required args still serialized | `required_keys.issubset(provided_keys)` before write |
| medium | User text interpolated into ChatML raw | `sanitize_input()` + `tests/test_prompt_injection.py` |
| medium | Makefile piped curl into sh | `makefile` requires preinstalled `uv` and points at README |
| medium | README wrapped in fence / `<login1>` / fake links | README is real markdown with local `docs/` links |
| medium | README install snippet broken + `curl \| sh` | Official `uv` install docs / `brew install uv`; no pipe-to-sh in README or `docs/uv.md` |
| low | README claimed `/no_think` and regex JSON | Algorithm describes `<tool_call>` prefill + `JSONDecoder`; `docs/example-prompts.md` updated |
| low | Python version and linters disagreed | README: Python `>=3.10`, flake8 + mypy (+ optional ruff) aligned with `pyproject.toml` / makefile |
| low | Docs showed nonexistent `llm_sdk.ollama` | `docs/packages.md` imports `Small_LLM_Model` |
| low | `THINK_TAG` unused | Removed |
| low | Stale `END_TOKEN_ID*` constants | Removed; stopping uses `STOP_TOKEN_IDS` |
| low | Function defs re-read per prompt | `get_functions()` loaded once in `main()` and passed into extraction |
| low | Leftover `old-exercise_input/` / spaced output filename | Removed |
| low | `custom_decode` docstring disagreed with code | Docstring matches implementation |
| low | `get_answer_ids` mutated caller `input_ids` | Copies into `working_ids` before generation |

## Method notes

Re-review inspected `src/`, `llm_sdk/`, `tests/`, `README.md`, `makefile`, `pyproject.toml`, and `docs/` against the prior finding list. Model inference was not re-executed. Counts are distinct findings, not automated scanner hits.
