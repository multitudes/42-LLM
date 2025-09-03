# src/__main__.py
import requests
import json
import os

from .utils import get_prompts, get_tools
from .ollama_request import OllamaRequest, Message
from .output_classes import NameFunctionCall
from llm_sdk import Small_LLM_Model

# OLLAMA_URL_API = "http://localhost:11434/api/chat"


# def call_ollama_api(data):
#     response = requests.post(OLLAMA_URL_API, json=data)
#     result = json.loads(response.content.decode())
#     return result.get("response", result)


def main():
    llm = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")  # or your preferred model

    prompt = "Is 4 an even number?"
    input_ids = llm._tokenizer.encode(prompt, add_special_tokens=False)
    logits = llm.get_logits_from_input_ids(input_ids)
    print("Logits for next token:", logits)

    vocab_path = llm.get_path_to_vocabulary_json()
    print("Vocabulary file path:", vocab_path)
    
    # outputs = []
    # prompts = get_prompts()
    # for prompt in prompts:
    #     print(f"Prompt: {prompt}")
    #     messages = [
    #         Message(
    #             role="user",
    #             content=f"{prompt}.  "
    #             "Reply ONLY in JSON with this exact format: "
    #             '{"name": <function_name or null>, "arguments": <dict of arguments>}. '
    #             "If no function is called, set 'name' to null and 'arguments' to {}. Do not include any other fields or text."
    #         )
    #     ]
    #     tools = get_tools()  # Should return a list of dicts
    #     data = OllamaRequest(
    #         model="qwen3:0.6b",
    #         messages=messages,
    #         tools=tools,
    #         stream=False,
    #         think=False
    #     )

    #     result = call_ollama_api(data.dict())

    #     print(result)
    #     tool_calls = result["message"].get("tool_calls")
    #     if tool_calls and len(tool_calls) > 0:
    #         tool_call = tool_calls[0]
    #         fn_name = tool_call["function"]["name"]
    #         args = tool_call["function"]["arguments"]
    #         print(fn_name)
    #         print(args)
    #     else:
    #         content = result["message"].get("content")
    #         if content:
    #             try:
    #                 parsed = json.loads(content)
    #                 fn_name = parsed["name"]
    #                 args = parsed["arguments"]
    #                 print(fn_name)
    #                 print(args)
    #             except (TypeError, json.JSONDecodeError):
    #                 print("Plain text reply from LLM:")
    #                 print(content)
    #                 fn_name = None
    #                 args = {}

    #     outputs.append(NameFunctionCall(
    #         prompt=prompt,
    #         fn_name=fn_name,
    #         args=args
    #     ))

    # os.makedirs("output", exist_ok=True)
    # with open("output/name_function_calls.json", "w") as f:
    #     json.dump([o.dict() for o in outputs], f)


if __name__ == "__main__":
    main()
