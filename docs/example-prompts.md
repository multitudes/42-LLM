# Prompt Engineering & ChatML Formatting

Prompts use ChatML control tokens (`<|im_start|>` and `<|im_end|>`) to enforce role boundaries between `system`, `user`, and `assistant`. Structural constraints—few-shot `<tool_call>` examples and a prefilled assistant prefix—steer the model into JSON tool calls without relying on a `/no_think` directive.

---

## ChatML Template & Special Control Tokens

Model architectures like Qwen rely on reserved control token IDs to segment conversation turns:

| Token String | Token ID | Purpose |
| --- | --- | --- |
| `<\|im_start\|>` | 151644 | Begin a ChatML role block |
| `<\|im_end\|>` | 151645 | End a ChatML role block |
| `<tool_call>` | 151657 | Begin a structured tool-call payload |
| `</tool_call>` | 151658 | End a structured tool-call payload (also a stop ID) |

The production template in `create_prompt` looks like:

```python
safe_user_input = sanitize_input(user_input)

final_prompt = (
    f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    f"<|im_start|>user\n{safe_user_input}<|im_end|>\n"
    f"<|im_start|>assistant\n<tool_call>\n"
)
```

Prefilling `<tool_call>\n` after the assistant marker constrains decoding to continue inside the JSON envelope and bypasses the model's tendency to open a `<think>` block.

User input is sanitized first (`sanitize_input`) so injected ChatML or tool tags cannot close the user turn early.

---

## Tool Injection & Context Setup

Tool definitions from `data/input/functions_definition.json` are converted to an OpenAI-style tools JSON string and injected into the system role together with few-shot examples:

```text
<|im_start|>system
You are a helpful assistant that uses tools. Based on the user's request, you must call the appropriate tool with the correct arguments by wrapping the JSON in <tool_call> tags.

You have access to the following tools:
[
  {
    "type": "function",
    "function": {
      "name": "fn_reverse_string",
      "description": "Reverse a string and return the reversed result.",
      "parameters": {
        "type": "object",
        "properties": {"s": {"type": "string"}},
        "required": ["s"]
      }
    }
  }
]
---
Here are some examples:

User: can you reverse the word 'banana'?
Assistant: <tool_call>
{"name": "fn_reverse_string", "parameters": {"s": "banana"}}

</tool_call>
---
<|im_end|>
<|im_start|>user
Reverse the string 'hello'<|im_end|>
<|im_start|>assistant
<tool_call>

```

---

## Why Not `/no_think`?

Earlier experiments appended `/no_think` inside the user block. The current approach instead **preforces the tool-call opening tag** in the assistant turn. That is a form of constrained decoding at the prompt level: the model is already inside `<tool_call>` when generation starts, so empty `<think>…</think>` preambles are avoided without a special directive token.

---

## Generation Loop Safeguards & Stopping Criteria

To keep generation from drifting after a valid tool call:

1. **Stop token IDs:** Halt when the next token is `</tool_call>` (151658), `<|im_end|>` (151645), or `<|endoftext|>` (151643).
2. **Maximum token budget:** `MAX_TOKENS` caps runaway loops if the model never emits a stop ID.
3. **Post-parse validation:** The first balanced JSON object is extracted with `json.JSONDecoder.raw_decode`; unknown tool names and missing required parameters become empty `SelectedFunction` results.
