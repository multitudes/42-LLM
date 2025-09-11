# src/__main__.py
import json
import os

from .utils import get_prompts, get_functions, convert_functions_to_tools
from .utils import extract_json_from_response
from llm_sdk import Small_LLM_Model
from .bpe_tokenizer import bpe_tokenize, custom_decode

MAX_TOKENS = 150


def create_prompt(prompt: str, tools: str) -> str:
    system_msg = "You are a helpful assistant that uses tools. "
    system_msg += "Based on the user's request, you must call the "
    system_msg += "appropriate tool with the correct arguments. "
    system_msg += "You have access to the following tools:\n"
    system_msg += f"{tools}"
    system_msg += """
---
Here are some examples:

User: Multiply 45 by 11
Assistant: {"fn_name": "fn_multiply_numbers", "args": {"a": 45, "b": 11}}

User: can you reverse the word 'banana'?
Assistant: {"fn_name": "fn_reverse_string", "args": {"s": "banana"}}
---

Now, answer the following request. Only provide the JSON for the tool call.
"""
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}/no_think<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    """
    I initialize the Small_LLM_Model from llm_sdk with the Qwen3-0.6B model
    which is also the default model for that class.
    I load the vocabulary from the model to get the token IDs.
    From the input file I get the prompts.
    The functions_definition.json file contains the list of functions that the
    llm will choose from to answer the prompt.
    They will be converted to a tool objects list using the
    convert_functions_to_tools() function.
    For each prompt I build the input_ids list with my own tokenizer.
    """
    llm = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")

    vocab_path = llm.get_path_to_vocabulary_json()
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    # Add special tokens if missing
    special_tokens = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<think>": 151667,
        "</think>": 151668,
    }
    for tok, tid in special_tokens.items():
        if tok not in vocab:
            vocab[tok] = tid
    # Load merges
    with open("merges.txt", "r") as f:
        merges = [line.strip().split()
                  for line in f if not line.startswith("#")]

    # Build merge ranks for fast lookup
    merge_ranks = {tuple(merge): i for i, merge in enumerate(merges)}
    input_ids = []
    output_to_write_to_file = []

    prompts = get_prompts()
    functions = get_functions()
    tools = convert_functions_to_tools(functions)
    # Reverse the vocab dict for ID to token lookup
    id_to_token = {v: k for k, v in vocab.items()}

    for prompt in prompts:
        logits = []
        answer_ids = []
        prompt = create_prompt(prompt, tools)
        # input_ids = llm._encode(final_prompt).tolist()[0]
        input_ids = bpe_tokenize(
            prompt, vocab=vocab, merge_ranks=merge_ranks)
        for _ in range(MAX_TOKENS):
            print(".", end="", flush=True)
            logits = llm.get_logits_from_input_ids(input_ids)
            next_token_id = max(enumerate(logits), key=lambda x: x[1])[0]
            input_ids.append(next_token_id)
            answer_ids.append(next_token_id)
            if (next_token_id == 3417 or next_token_id == 30975):
                break

        # llm_output = llm._decode(answer_ids)
        llm_output = custom_decode(answer_ids, id_to_token)
        print("\nllm output:", llm_output)
        result = extract_json_from_response(
            prompt, llm_output)
        output_to_write_to_file.append(result)

    os.makedirs("output", exist_ok=True)
    with open("output/function_calling_name.json", "w") as f:
        json.dump([o.dict() for o in output_to_write_to_file], f)


if __name__ == "__main__":
    main()
