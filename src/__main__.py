# src/__main__.py
import json
import string
import re

from time import sleep
from .utils import get_prompts, get_functions, convert_functions_to_tools
from llm_sdk import Small_LLM_Model


def main():
    """
    from the function_calling_tests.json i get the prompts    
    using the get_prompts() function
    using the get_functions() function i get the functions definitions
    from the functions_definition.json file which I convert to a FunctionDef objects list
    using the convert_functions_to_tools() function I convert the FunctionDef list to a JSON string
    which is what I will use in the prompt.
    I initialize the Small_LLM_Model from llm_sdk with the Qwen3-0.6B model
    I load the vocabulary from the model to get the token IDs
    for each prompt I build the input_ids list with the token IDs for the prompt
    with my own tokenizer
    """
    input_ids = []
    llm = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    vocab_path = llm.get_path_to_vocabulary_json()
    with open(vocab_path, "r") as f:
        vocabs = json.load(f)
    prompts = get_prompts()
    functions = get_functions()
    tools = convert_functions_to_tools(functions)
    # Reverse the vocab dict for ID to token lookup
    # id_to_token = {v: k for k, v in vocabs.items()}

    for prompt in prompts:
        logits = []
        system_msg = f"You are a helpful assistant that uses tools. "
        system_msg += "Based on the user's request, you must call the "
        system_msg += "appropriate tool with the correct arguments. "
        system_msg += "You have access to the following tools:\n"
        system_msg += f"{tools}"
        system_msg += """
        ---
        Here are some examples:

        User: Multiply 45 by 11
        Assistant: {{"fn_name": "fn_multiply_numbers", "args": {{"a": 45, "b": 11}}}}

        User: can you reverse the word 'banana'?
        Assistant: {{"fn_name": "fn_reverse_string", "args": {{"s": "banana"}}}}
        ---

        Now, answer the following request. Only provide the JSON for the tool call.
        """
        user_msg = f"{prompt}"
        final_prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}/no_think<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids = llm._encode(final_prompt).tolist()[0]
        answer_ids = []
        for _ in range(40):  # Generate 150 tokens
            print(".", end="", flush=True)
            logits = llm.get_logits_from_input_ids(input_ids)
            next_token_id = max(enumerate(logits), key=lambda x: x[1])[0]
            input_ids.append(next_token_id)
            answer_ids.append(next_token_id)
            
        output = llm._decode(answer_ids)
        print("\nDecoded output:", output, end="\n")


    # # outputs = []
    # # prompts = get_prompts()
    # # for prompt in prompts:
    # #     print(f"Prompt: {prompt}")
    # #     messages = [
    # #         Message(
    # #             role="user",
    # #             content=f"{prompt}.  "
    # #             "Reply ONLY in JSON with this exact format: "
    # #             '{"name": <function_name or null>, "arguments": <dict of arguments>}. '
    # #             "If no function is called, set 'name' to null and 'arguments' to {}. Do not include any other fields or text."
    # #         )
    # #     ]
    # #     tools = get_tools()  # Should return a list of dicts
    # #     data = OllamaRequest(
    # #         model="qwen3:0.6b",
    # #         messages=messages,
    # #         tools=tools,
    # #         stream=False,
    # #         think=False
    # #     )

    # #     result = call_ollama_api(data.dict())

    # #     print(result)
    # #     tool_calls = result["message"].get("tool_calls")
    # #     if tool_calls and len(tool_calls) > 0:
    # #         tool_call = tool_calls[0]
    # #         fn_name = tool_call["function"]["name"]
    # #         args = tool_call["function"]["arguments"]
    # #         print(fn_name)
    # #         print(args)
    # #     else:
    # #         content = result["message"].get("content")
    # #         if content:
    # #             try:
    # #                 parsed = json.loads(content)
    # #                 fn_name = parsed["name"]
    # #                 args = parsed["arguments"]
    # #                 print(fn_name)
    # #                 print(args)
    # #             except (TypeError, json.JSONDecodeError):
    # #                 print("Plain text reply from LLM:")
    # #                 print(content)
    # #                 fn_name = None
    # #                 args = {}

    # #     outputs.append(NameFunctionCall(
    # #         prompt=prompt,
    # #         fn_name=fn_name,
    # #         args=args
    # #     ))

    # # os.makedirs("output", exist_ok=True)
    # # with open("output/name_function_calls.json", "w") as f:
    # #     json.dump([o.dict() for o in outputs], f)


if __name__ == "__main__":
    main()
