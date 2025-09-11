# Prompting

to encode the prompt I used a custom BPE encoder. 
There are two special token that the LLM expects:
```
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
```
Those token are not in the vocab but appears when I use the `_encode` function which is a private API. We are not supposed to use that function but it was helpful to debug. Using the special tokens the LLM understands better my prompt and doesnt allucinate as much, also the responce is straightaway in JSON.
As in the huggingface documentation there is a flag I pass at the end of the prompt to avoid thinking:  `/no_think`
This `/no_think` directive however is not being observed if I dont uyse the special tokens `<|im_start|>` and `<|im_end|>`. Lesson learned!

So this is the scheme I used for the prompt:
```python
user_msg = f"{prompt}"
final_prompt = (
    f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    f"<|im_start|>user\n{user_msg}/no_think<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
```

In the system prompt I parse the json I get from the `functions_definition.json` file.
```
system
You are a helpful assistant that uses tools. Based on the user's request, you must call the appropriate tool with the correct arguments. You have access to the following tools:
[
  {
    "type": "function",
    "function": {
      "name": "fn_add_numbers",
      "description": "add numbers function",
      "parameters": {
        "type": "object",
        "properties": {
          "a": {
            "type": "number"
          },
          "b": {
            "type": "number"
          }
        },
        "required": [
          "a",
          "b"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "fn_get_square_root",
      "description": "get square root function",
      "parameters": {
        "type": "object",
        "properties": {
          "a": {
            "type": "number"
          }
        },
        "required": [
          "a"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "fn_greet",
      "description": "greet function",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          }
        },
        "required": [
          "name"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "fn_is_even",
      "description": "is even function",
      "parameters": {
        "type": "object",
        "properties": {
          "n": {
            "type": "integer"
          }
        },
        "required": [
          "n"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "fn_multiply_numbers",
      "description": "multiply numbers function",
      "parameters": {
        "type": "object",
        "properties": {
          "a": {
            "type": "number"
          },
          "b": {
            "type": "number"
          }
        },
        "required": [
          "a",
          "b"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "fn_reverse_string",
      "description": "reverse string function",
      "parameters": {
        "type": "object",
        "properties": {
          "s": {
            "type": "string"
          }
        },
        "required": [
          "s"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "fn_substitute_string_with_regex",
      "description": "substitute string with regex function",
      "parameters": {
        "type": "object",
        "properties": {
          "source_string": {
            "type": "string"
          },
          "regex": {
            "type": "string"
          },
          "replacement": {
            "type": "string"
          }
        },
        "required": [
          "source_string",
          "regex",
          "replacement"
        ]
      }
    }
  }
]
---
Here are some examples:

User: Multiply 45 by 11
Assistant: {{"fn_name": "fn_multiply_numbers", "args": {{"a": 45, "b": 11}}}}

User: can you reverse the word 'banana'?
Assistant: {{"fn_name": "fn_reverse_string", "args": {{"s": "banana"}}}}
---

Now, answer the following request. Only provide the JSON for the tool call.

user
```
The prompt here is `Reverse the string 'hello'`. 
```
Reverse the string 'hello'/no_think
assistant
```


The response looks like this. At first I did not know what the tokens ID 151667 and 151668 represented because they were not in my dictionary but using the private _decode API given in the `Small_LLM_Model` class I found out that they were just the `<think>` and `<\think>` tokens with a blank token in between because it is not thinking but still letting us know that it is skipping the thinking part!  

```
.Next token ID: 151667, Token: <unk>
.Next token ID: 271, Token: ĊĊ
.Next token ID: 151668, Token: <unk>
.Next token ID: 271, Token: ĊĊ
.Next token ID: 4913, Token: {"
.Next token ID: 8822, Token: fn
.Next token ID: 1269, Token: _name
.Next token ID: 788, Token: ":
.Next token ID: 330, Token: Ġ"
.Next token ID: 8822, Token: fn
.Next token ID: 43277, Token: _reverse
.Next token ID: 3904, Token: _string
.Next token ID: 497, Token: ",
.Next token ID: 330, Token: Ġ"
.Next token ID: 2116, Token: args
.Next token ID: 788, Token: ":
.Next token ID: 5212, Token: Ġ{"
.Next token ID: 82, Token: s
.Next token ID: 788, Token: ":
.Next token ID: 330, Token: Ġ"
.Next token ID: 14990, Token: hello
.Next token ID: 30975, Token: "}}
```

So I use the `\"}}` or `}}` as a sentinel token to stop. The model is supposed to output just the json and in a certain format so this would actually tell me that it is done with the output. Because, I also noticed that if I continue with the request it would keep on repeating the answer I already have... Therefore I also set up a token limit in case the model allucinates and starts to output random thoughts, even if it is not supposed to think!



