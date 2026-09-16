# src/bpe_tokenizer.py
import json
import re
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

MAX_TOKENS = 150
# SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
END_TOKEN_ID1 = 3417
END_TOKEN_ID2 = 30975
EXPECTED_MERGE_TOKENS = 2
MERGES_PATH = "merges.txt"
SPECIAL_TOKENS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<think>": 151667,
    "</think>": 151668,
}
# These token IDs to signal end of json generation '}}' and '\"}}"


def initialize_tokenizer(
    vocab_path: str | Path,
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """
    Initializes the BPE tokenizer by loading vocabulary and merge ranks.

    Reads the vocabulary JSON file and the BPE merges text file, adds any
    missing special tokens to the vocabulary, and builds the merge priority
    mapping.

    Args:
        vocab_path: Path to the vocabulary JSON file.

    Returns:
        A tuple containing:
            - Vocabulary mapping token strings to integer IDs.
            - Merge ranks mapping token pair tuples to their integer ranks.

    Raises:
        RuntimeError: If loading or parsing the vocabulary or merges file
        fails.

    """
    path = Path(vocab_path)
    merges_path = Path(MERGES_PATH)

# 1. Load vocabulary JSON
    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

            # Check if it's a nested HuggingFace tokenizer.json format
            if "model" in raw_data and "vocab" in raw_data["model"]:
                vocab = cast("dict[str, int]", raw_data["model"]["vocab"])
            else:
                # Fallback assuming it's already a flat dictionary
                vocab = cast("dict[str, int]", raw_data)

    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        msg = f"Error loading vocabulary from {path}: {e}"
        raise RuntimeError(msg) from e

    # Ensure special tokens are included
    for tok, tid in SPECIAL_TOKENS.items():
        if tok not in vocab:
            vocab[tok] = tid

    # 2. Load merge rules and build rank map
    try:
        with merges_path.open("r", encoding="utf-8") as f:
            merges = [
                line.strip().split()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        merge_ranks: dict[tuple[str, str], int] = {
            (m[0], m[1]): i for i, m in enumerate(merges)
            if len(m) == EXPECTED_MERGE_TOKENS
        }
    except (FileNotFoundError, OSError, IndexError) as e:
        msg = f"Error loading merges file from {merges_path}: {e}"
        raise RuntimeError(msg) from e

    return vocab, merge_ranks


def get_pairs(tokens: list[str]) -> set[tuple[str, str]]:
    """
    Returns a set of adjacent token pairs from a list of tokens.

    Args:
        tokens: List of string tokens.

    Returns:
        Set of tuples containing adjacent token pairs.

    """
    return set(pairwise(tokens))


def preprocess_for_bpe(text: str) -> str:
    """
    Preprocesses text for BPE tokenization by mapping whitespace characters.

    Replaces spaces, newlines, and tabs with special byte-level BPE sequence
    markers (`Ġ`, `Ċ`, `ĉ`).

    Args:
        text: Raw input text string to format.

    Returns:
        Text string with whitespace replaced by BPE sequence markers.

    """
    return text.replace(" ", "Ġ").replace("\n", "Ċ").replace("\t", "ĉ")


def bpe_tokenize(
    text: str,
    vocab: dict[str, int],
    merge_ranks: dict[tuple[str, str], int],
) -> list[int]:
    """
    Tokenizes input text using Byte Pair Encoding (BPE).

    Splits text while preserving special tokens (e.g., prompt control tags),
    iteratively merges token pairs according to `merge_ranks`, and maps
    the final tokens to their vocabulary IDs.

    Args:
        text: The raw input text string to tokenize.
        vocab: Mapping from token strings to integer vocabulary IDs.
        merge_ranks: Mapping from token pair tuples to their merge rank.

    Returns:
        List of mapped integer token IDs.

    """
    if SPECIAL_TOKENS:
        pattern = f"({'|'.join(re.escape(tok) for tok in SPECIAL_TOKENS)})"
        parts = [p for p in re.split(pattern, text) if p]
    else:
        parts = [text]

    tokens: list[str] = []
    for part in parts:
        if part in SPECIAL_TOKENS:
            tokens.append(part)
        else:
            tokens.extend(preprocess_for_bpe(part))

    while True:
        pairs = get_pairs(tokens)
        min_rank = float("inf")
        best_pair: tuple[str, str] | None = None

        for pair in pairs:
            if pair in merge_ranks and merge_ranks[pair] < min_rank:
                min_rank = merge_ranks[pair]
                best_pair = pair

        if best_pair is None:
            break

        new_tokens: list[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_tokens.append(tokens[i] + tokens[i + 1])
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

    Filters out special token IDs defined in `SPECIAL_TOKENS`
    (such as think tags), replaces missing tokens with `<unk>`,
    and converts byte-level BPE whitespace markers back to standard
    spaces and line breaks.

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
        "When constructing regular expressions, use character sets like [aeiou] "
        "for matching specific characters, \\d+ for digits, and \\b for word boundaries.\n\n"
        f"You have access to the following tools:\n{tools}\n\n"
        "---\n"
        "Here are some examples:\n\n"
        "User: Multiply 45 by 11\n"
        'Assistant: {"fn_name": "fn_multiply_numbers", '
        '"args": {"a": 45.0, "b": 11.0}}\n\n'
        "User: can you reverse the word 'banana'?\n"
        'Assistant: {"fn_name": "fn_reverse_string", '
        '"args": {"s": "banana"}}\n\n'
        "User: Substitute the digits in the string "
        "'Hello 34 I'm 233 years old' with 'NUMBERS'\n"
        'Assistant: {"fn_name": "fn_substitute_string_with_regex", '
        '"args": {"source_string": "Hello 34 I\'m 233 years old", '
        '"regex": "\\\\d+", "replacement": "NUMBERS"}}\n\n'
        "User: Replace vowels in 'hello' with '*'\n"
        'Assistant: {"fn_name": "fn_substitute_string_with_regex", '
        '"args": {"source_string": "hello", '
        '"regex": "[aeiouAEIOU]", "replacement": "*"}}\n'
        "---\n\n"
        "Now, answer the following request. Only provide the JSON for "
        "the tool call."
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
    the highest logit value is selected and appended to `input_ids` for
    subsequent generation. Generation stops when reaching `MAX_TOKENS` or an
    end token.

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
