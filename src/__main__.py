# src/__main__.py
import json
import string
import re
import os

from time import sleep
from .output_classes import FunctionCallingName
from .utils import get_prompts, get_functions, convert_functions_to_tools, extract_json_from_response
from llm_sdk import Small_LLM_Model


def has_consecutive_token(ids, token=92):
    return any(ids[i] == token and ids[i+1] == token for i in range(len(ids)-1))


def get_pairs(tokens):
    """Return set of adjacent token pairs."""
    return {(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)}


def preprocess_for_bpe(text):
    # Add a special marker for spaces (e.g., "Ġ")
    text = text.replace(" ", "Ġ")
    text = text.replace("\n", "Ċ")
    text = text.replace("\t", "ĉ")
    # text = text.replace("?", "")
    return text


def bpe_tokenize(text, vocab, merge_ranks):
    # Start with characters (or initial tokens)
    tokens = list(preprocess_for_bpe(text))
    while True:
        pairs = get_pairs(tokens)
        # Find the best pair to merge
        min_rank = float('inf')
        best_pair = None
        for pair in pairs:
            if pair in merge_ranks and merge_ranks[pair] < min_rank:
                min_rank = merge_ranks[pair]
                best_pair = pair
        if best_pair is None:
            break
        # Merge all occurrences of the best pair
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    #debug
    print("Final tokens before vocab mapping:", tokens)
    # Map tokens to IDs
    return [vocab[token] for token in tokens if token in vocab]


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
    llm = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")

    vocab_path = llm.get_path_to_vocabulary_json()
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # Load merges
    with open("merges.txt", "r") as f:
        merges = [line.strip().split() for line in f if not line.startswith("#")]

    # Build merge ranks for fast lookup
    merge_ranks = {tuple(merge): i for i, merge in enumerate(merges)}
    # ids = bpe_tokenize("\"root of 16\"}}")
    # print(ids)
    input_ids = []
    output_to_write_to_file = []

    prompts = get_prompts()
    functions = get_functions()
    tools = convert_functions_to_tools(functions)
    # Reverse the vocab dict for ID to token lookup
    id_to_token = {v: k for k, v in vocab.items()}

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
Assistant: {"fn_name": "fn_multiply_numbers", "args": {"a": 45, "b": 11}}

User: can you reverse the word 'banana'?
Assistant: {"fn_name": "fn_reverse_string", "args": {"s": "banana"}}
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
        # input_ids = bpe_tokenize(final_prompt, vocab=vocab, merge_ranks=merge_ranks)
        # print("Input IDs:", input_ids)
        print("Decoded Input:", llm._decode(input_ids))
        answer_ids = []
        for _ in range(150):  # Generate 150 tokens
            print(".", end="", flush=True)
            logits = llm.get_logits_from_input_ids(input_ids)
            next_token_id = max(enumerate(logits), key=lambda x: x[1])[0]
            input_ids.append(next_token_id)
            answer_ids.append(next_token_id)
            # print(f"Next token ID: {next_token_id}, Token: {id_to_token.get(next_token_id, '<unk>')}")
            if (next_token_id == 3417 or next_token_id == 30975):
                print("Token }} appears! End of json!")
                break

        llm_output = llm._decode(answer_ids)
        print("\nDecoded output:", llm_output, end="\n")
        result = extract_json_from_response(
            prompt, llm_output)
        output_to_write_to_file.append(result)

    os.makedirs("output", exist_ok=True)
    with open("output/function_calling_name.json", "w") as f:
        json.dump([o.dict() for o in output_to_write_to_file], f)


if __name__ == "__main__":
    main()
