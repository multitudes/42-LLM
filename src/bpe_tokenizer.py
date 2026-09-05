# src/bpe_tokenizer.py
import json
import re
from typing import Any  # Replace with Small_LLM_Model if imported

MAX_TOKENS = 150
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
END_TOKEN_ID1 = 3417
END_TOKEN_ID2 = 30975
MERGES_PATH = "merges.txt"
SPECIAL_TOKENS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<think>": 151667,
    "</think>": 151668,
}
# These token IDs to signal end of json generation '}}' and '\"}}"


def initialize_tokenizer(vocab_path):
    """
    Initialize the BPE tokenizer by loading the vocabulary and merge ranks
    from the respective files. The merge_ranks is a dictionary mapping
    token pairs to their rank (lower rank means higher priority for merging).
    The special tokens are added to the vocabulary if not already present.
    The merge.txt file is expected and eventually needs to be downloaded
    from the same source as the vocab.json file. It contains the rules for
    merging tokens during the BPE tokenization process.
    Returns: vocab (dict): Mapping of tokens to their IDs.
        merge_ranks (dict): Mapping of token pairs to their merge ranks.
    Raises: RuntimeError: If there is an error loading the vocabulary
        or merges file.
    """
    try:
        with open(vocab_path) as f:
            vocab = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading vocabulary: {e}")
    for tok, tid in SPECIAL_TOKENS.items():
        if tok not in vocab:
            vocab[tok] = tid
    try:
        with open(MERGES_PATH) as f:
            merges = [line.strip().split()
                      for line in f if not line.startswith("#")]
        merge_ranks = {tuple(merge): i for i, merge in enumerate(merges)}
    except Exception as e:
        raise RuntimeError(
            f"Error loading merges.txt file needed for the tokenizer: {e}")
    return vocab, merge_ranks


def get_pairs(tokens):
    """
    Return set of adjacent token pairs.
    """
    return {(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)}


def preprocess_for_bpe(text):
    """
    Preprocess text for BPE tokenization by replacing spaces,
    newlines, and tabs for consistent tokenization.
    """
    text = text.replace(" ", "Ġ")
    text = text.replace("\n", "Ċ")
    text = text.replace("\t", "ĉ")
    return text


def bpe_tokenize(text, vocab, merge_ranks):
    """
    A simple BPE tokenizer implementation.
    This function tokenizes the input text using Byte Pair Encoding (BPE)
    based on the provided vocabulary and merge ranks.
    There are some special tokens that are needed for the prompt structure
    that should be treated as single tokens and not split further like:
    151644: <|im_start|>
    151645: <|im_end|>
    """
    pattern = "(" + "|".join(
        re.escape(tok) for tok in SPECIAL_TOKENS.keys()) + ")"
    parts = re.split(pattern, text)
    tokens = []
    for part in parts:
        if part in SPECIAL_TOKENS:
            tokens.append(part)
        else:
            tokens.extend(list(preprocess_for_bpe(part)))
    while True:
        pairs = get_pairs(tokens)
        # Find the best pair to merge
        min_rank = float("inf")
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
    return [vocab[token] for token in tokens if token in vocab]


def custom_decode(
    ids: list[int],
    id_to_token: dict[int, str],
) -> str:
    """
    Converts a sequence of token IDs back into a decoded text string.

    Filters out special token IDs defined in `SPECIAL_TOKENS` (such as think tags),
    replaces missing tokens with `<unk>`, and converts byte-level BPE whitespace
    markers back to standard spaces and line breaks.

    Args:
        ids: List of token IDs to decode.
        id_to_token: Mapping of token IDs to their corresponding token strings.

    Returns:
        The decoded string representation.

    """
    skip_ids = set(SPECIAL_TOKENS.values())
    tokens = [id_to_token.get(i, "<unk>") for i in ids if i not in skip_ids]

    text = "".join(tokens)
    text = text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")

    print(f"\n\nllm output: {text}", end="")
    return text


def create_prompt(user_input: str, tools: str) -> str:
    """
    Creates the prompt for the LLM based on user input and available tools.

    Args:
        user_input: The natural language request from the user.
        tools: JSON string describing available tools.

    Returns:
        Formatted chat prompt string ready for inference.

    """
    system_msg = (
        "You are a helpful assistant that uses tools. "
        "Based on the user's request, you must call the "
        "appropriate tool with the correct arguments. "
        f"You have access to the following tools:\n{tools}\n\n"
        "---\n"
        "Here are some examples:\n\n"
        "User: Multiply 45 by 11\n"
        'Assistant: {"fn_name": "fn_multiply_numbers", "args": {"a": 45.0, "b": 11.0}}\n\n'
        "User: can you reverse the word 'banana'?\n"
        'Assistant: {"fn_name": "fn_reverse_string", "args": {"s": "banana"}}\n\n'
        "User: Substitute the digits in the string\n"
        "'Hello 34 I'm 233 years old' with 'NUMBERS'\n"
        'Assistant: {"fn_name": "fn_substitute_string_with_regex", '
        '"args": {"source_string": "Hello 34 I\'m 233 years old", '
        '"regex": "\\\\d+", "replacement": "NUMBERS"}}\n'
        "---\n\n"
        "Now, answer the following request. Only provide the JSON for the tool call."
    )

    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}/no_think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def get_answer_ids(
    llm: Any,
    input_ids: list[int],
) -> list[int]:
    """
    Generates token IDs sequentially using the language model's logits.

    The model generates logits for the next token at each step. The token with
    the highest logit value is selected and appended to `input_ids` for subsequent
    generation. Generation stops when reaching `MAX_TOKENS` or an end token.

    Args:
        llm: Instance of the language model class.
        input_ids: List of input token IDs as integers. This list is modified
            in-place during generation.

    Returns:
        List of generated answer token IDs.

    """
    answer_ids: list[int] = []
    stop_tokens = {END_TOKEN_ID1, END_TOKEN_ID2}

    for _ in range(MAX_TOKENS):
        print(".", end="", flush=True)
        logits = llm.get_logits_from_input_ids(input_ids)

        # Get token index with the highest logit
        next_token_id = max(enumerate(logits), key=lambda x: x[1])[0]

        input_ids.append(next_token_id)
        answer_ids.append(next_token_id)

        # Fixes PLR1714
        if next_token_id in stop_tokens:
            break

    return answer_ids
