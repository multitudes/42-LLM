# src/__main__.py
import json
import os

from .utils import get_prompts, get_tool_list
from .utils import extract_json_from_response
from llm_sdk import Small_LLM_Model
from .bpe_tokenizer import initialize_tokenizer, bpe_tokenize, custom_decode

INPUT_FILE = "exercise_input/function_calling_tests.json"
OUTPUT_FILE = "output/function_calling_name.json"
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


def get_answer_ids(llm, input_ids):
    """
    The llmm takes a list of input token ids and generates
    a list of logits for the next token at each step.
    The next token is chosen as the one with the highest logit,
    and appended to the input_ids for the next generation.
    At the same time I am interested in collecting the
    generated token ids to decode later in answer_ids.
    """
    answer_ids = []
    for _ in range(MAX_TOKENS):
        print(".", end="", flush=True)
        logits = llm.get_logits_from_input_ids(input_ids)
        next_token_id = max(enumerate(logits), key=lambda x: x[1])[0]
        input_ids.append(next_token_id)
        answer_ids.append(next_token_id)
        if (next_token_id == 3417 or next_token_id == 30975):
            break
    return answer_ids


def write_output_to_file(output_to_write_to_file):
    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump([o.dict() for o in output_to_write_to_file], f)


def main(input_file: str = INPUT_FILE):
    """
    Main entry point for function-calling LLM pipeline.

    Args:
        input_file (str, optional): Path to the prompts JSON file. Defaults to
            "exercise_input/function_calling_tests.json".
    """
    llm = Small_LLM_Model()
    vocab_path = llm.get_path_to_vocabulary_json()
    vocab, merge_ranks = initialize_tokenizer(vocab_path)
    outputs = []

    prompts = get_prompts(input_file)
    tools = get_tool_list()
    # Reverse the vocab dict for ID to token lookup
    id_to_token = {v: k for k, v in vocab.items()}

    for prompt in prompts:
        user_prompt = create_prompt(prompt, tools)
        # input_ids = llm._encode(final_prompt).tolist()[0]
        input_ids = bpe_tokenize(
            user_prompt, vocab=vocab, merge_ranks=merge_ranks)
        answer_ids = get_answer_ids(llm, input_ids)
        # llm_output = llm._decode(answer_ids)
        llm_output = custom_decode(answer_ids, id_to_token)
        result = extract_json_from_response(
            prompt, llm_output)
        outputs.append(result)

    write_output_to_file(outputs)


if __name__ == "__main__":
    main()
