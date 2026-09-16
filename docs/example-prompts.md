# Prompt Engineering & ChatML Formatting

I format all prompt inputs using ChatML control tokens (`<|im_start|>` and `<|im_end|>`) to enforce role boundaries between `system`, `user`, and `assistant`. Explicitly structuring prompts with control tokens prevents model hallucinations, ensures strict JSON formatting, and guarantees that control directives like `/no_think` are obeyed during inference.

---

## ChatML Template & Special Control Tokens

Model architectures like Qwen rely on reserved control token IDs to segment turn conversations:

| Token String | Token ID | Purpose |
| --- | --- | --- |
| `< | im_start | >` |
| `< | im_end | >` |

I construct prompt strings dynamically using the following template:

```python
user_msg = f"{prompt}"
final_prompt = (
    f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    f"<|im_start|>user\n{user_msg}/no_think<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

```

Adding the `/no_think` directive inside the ChatML user block instructs reasoning models to bypass internal chain-of-thought token generation and immediately emit structured output.

---

## Tool Injection & Context Setup

I load tool definitions directly from `exercise_input/functions_definition.json` and inject them into the `system` role block alongside explicit output instructions and few-shot formatting examples:

```text
<|im_start|>system
You are a helpful assistant that uses tools. Based on the user's request, you must call the appropriate tool with the correct arguments. You have access to the following tools:
[
  {
    "type": "function",
    "function": {
      "name": "fn_reverse_string",
      "description": "reverse string function",
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

User: Multiply 45 by 11
Assistant: {"fn_name": "fn_multiply_numbers", "args": {"a": 45, "b": 11}}

User: can you reverse the word 'banana'?
Assistant: {"fn_name": "fn_reverse_string", "args": {"s": "banana"}}
---

Now, answer the following request. Only provide the JSON for the tool call.<|im_end|>
<|im_start|>user
Reverse the string 'hello'/no_think<|im_end|>
<|im_start|>assistant

```

---

## Token Logit Inspection & Reasoning Directives

By inspecting raw logit generation using the model's `_decode` utility, I identified how the model processes the `/no_think` directive at the token level:

```text
Next token ID: 151667, Token: <think>
Next token ID: 271,    Token: ĊĊ
Next token ID: 151668, Token: </think>
Next token ID: 271,    Token: ĊĊ
Next token ID: 4913,   Token: {"
Next token ID: 8822,   Token: fn
...
Next token ID: 30975,  Token: "}}

```

* **Special Thinking Tokens:** Token IDs `151667` (`<think>`) and `151668` (`</think>`) wrap the reasoning stage.
* **Bypassing Deliberation:** When supplied with `/no_think` inside valid ChatML tags, the model immediately opens and closes an empty thinking block (emitting only newline tokens `ĊĊ`) before outputting the standard JSON payload.

---

## Generation Loop Safeguards & Stopping Criteria

To prevent the generation loop from repeating output or drifting into infinite token loops after emitting valid JSON, I implement dual stopping constraints:

1. **Sentinel Token Matching:** The generation loop monitors output tokens for target JSON terminators (such as `}}` or `"` followed by `}}`). Once a complete JSON payload structure is detected, generation halts immediately.
2. **Maximum Token Budget:** A hard upper limit on `max_new_tokens` acts as a fail-safe against runaway generations if the model hallucinates or fails to produce a closing brace.
