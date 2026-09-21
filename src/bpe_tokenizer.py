# src/bpe_tokenizer.py
import json
import re
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

MAX_TOKENS = 92
EXPECTED_MERGE_TOKENS = 2

SPECIAL_TOKENS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|object_ref_start|>": 151646,
    "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648,
    "<|box_end|>": 151649,
    "<|quad_start|>": 151650,
    "<|quad_end|>": 151651,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
    "<tool_call>": 151657,
    "</tool_call>": 151658,
    "<|fim_prefix|>": 151659,
    "<|fim_middle|>": 151660,
    "<|fim_suffix|>": 151661,
    "<|fim_pad|>": 151662,
    "<|repo_name|>": 151663,
    "<|file_sep|>": 151664,
    "<tool_response>": 151665,
    "</tool_response>": 151666,
    "<think>": 151667,
    "</think>": 151668,
}

STOP_TOKEN_IDS = {
    151658,  # </tool_call>
    151645,  # <|im_end|>
    151643,  # <|endoftext|>
}


def initialize_tokenizer(
    tokenizer_path: Path,
    merges_path: Path,
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """
    Initializes the BPE tokenizer by loading vocabulary and merge ranks.

    Reads the vocabulary JSON file and the BPE merges text file, adds any
    missing special tokens to the vocabulary, and builds the merge priority
    mapping.

    Args:
        tokenizer_path: Path to the vocabulary JSON file.
        merges_path: Path to the BPE merges text file.

    Returns:
        A tuple containing:
            - Vocabulary mapping token strings to integer IDs.
            - Merge ranks mapping token pair tuples to their integer ranks.

    Raises:
        RuntimeError: If loading or parsing the vocabulary or merges file
            fails.

    """
    # Load vocabulary JSON
    try:
        with tokenizer_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        vocab = cast("dict[str, int]", raw_data["model"]["vocab"])

    except (FileNotFoundError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            OSError) as e:
        msg = f"Error loading vocabulary from {tokenizer_path}: {e}"
        raise RuntimeError(msg) from e

    # Ensure special tokens are included in the vocab
    for tok, tid in SPECIAL_TOKENS.items():
        if tok not in vocab:
            vocab[tok] = tid

    # Load merge rules and build rank map
    try:
        with merges_path.open("r", encoding="utf-8") as f:
            merges = [
                line.strip().split()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        merge_ranks: dict[tuple[str, str], int] = {
            (m[0], m[1]): i
            for i, m in enumerate(merges)
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
    pattern = f"({'|'.join(re.escape(tok) for tok in SPECIAL_TOKENS)})"
    parts = [p for p in re.split(pattern, text) if p]

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
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == best_pair
            ):
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    # Retrieves the ID for unknown tokens as fallback
    unk_id = vocab.get("<unk>")
    # Avoid dropping unknown tokens
    if unk_id is not None:
        return [vocab.get(token, unk_id) for token in tokens]

    return [vocab[token] for token in tokens if token in vocab]


def custom_decode(
    ids: list[int],
    id_to_token: dict[int, str],
) -> str:
    """Converts a sequence of token IDs back into a decoded text string.

    Replaces byte-level BPE whitespace markers (`Ġ`, `Ċ`, `ĉ`) back to
    standard spaces, newlines, and tabs.

    Args:
        ids: List of integer token IDs to decode.
        id_to_token: Mapping of token IDs to their corresponding strings.

    Returns:
        The decoded text string representation.

    """
    tokens = [id_to_token.get(i, "<unk>") for i in ids]
    text = "".join(tokens)

    # Revert BPE whitespace markers to standard formatting characters
    return text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


# def gpt2_bytes_to_unicode() -> dict[int, str]:
#     """
#     Returns the standard GPT-2 / Qwen byte-to-unicode character map.

#     Returns:
#         Dictionary mapping byte values to Unicode character strings.

#     """
#     bs = (
#         list(range(ord("!"), ord("~") + 1))
#         + list(range(ord("¡"), ord("¬") + 1))
#         + list(range(ord("®"), ord("ÿ") + 1))
#     )
#     cs = bs[:]
#     n = 0
#     for b in range(2**8):
#         if b not in bs:
#             bs.append(b)
#             cs.append(2**8 + n)
#             n += 1
#     return dict(zip(bs, [chr(n) for n in cs]))


# # Inverted mapping: Unicode Symbol -> Raw Byte (0-255)
# UNICODE_TO_BYTE: dict[str, int] = {
#     v: k for k, v in gpt2_bytes_to_unicode().items()
# }


# def custom_decode(
#     ids: list[int],
#     id_to_token: dict[int, str],
# ) -> str:
#     """
#     Decodes a list of token IDs back into a UTF-8 text string.

#     Args:
#         ids: List of integer token IDs to decode.
#         id_to_token: Mapping of token IDs to string tokens.

#     Returns:
#         The decoded UTF-8 string representation.

#     """
#     tokens = [id_to_token.get(i, "") for i in ids]
#     text_symbolic = "".join(tokens)

#     # Convert characters back to raw byte array
#     raw_bytes = bytearray(
#         UNICODE_TO_BYTE[char]
#         for char in text_symbolic
#         if char in UNICODE_TO_BYTE
#     )

#     # Decode bytes as UTF-8
#     return raw_bytes.decode("utf-8", errors="replace")


# @lru_cache(maxsize=1)
# def load_control_tokens(tokenizer_path: str | Path) -> tuple[str, ...]:
#     """
#     Loads special control tokens dynamically from a tokenizer.json file.

#     Args:
#         tokenizer_path: Path to the tokenizer JSON file.

#     Returns:
#         Tuple of control token strings loaded from the file.

#     """
#     path = Path(tokenizer_path)
#     if not path.is_file():
#         print(f"Warning: Tokenizer file not found at {path}")
#         return ()

#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         added_tokens = data.get("added_tokens", [])

#         # Extract tokens marked as special (e.g. <|im_start|>, <|im_end|>)
#         # or custom structural formatting markers
#         control_tokens = [
#             token["content"]
#             for token in added_tokens
#             if isinstance(token, dict)
#             and (
#                 token.get("special", False)
#                 or token["content"].startswith("<|")
#             )
#         ]
#         return tuple(control_tokens)

#     except (json.JSONDecodeError, OSError) as e:
#         print(f"Warning: Failed to parse tokenizer file at {path}: {e}")
#         return ()


def sanitize_input(text: str) -> str:
    """
    Strips control tokens from user input to prevent prompt injection.

    Args:
        text: Raw input text string.

    Returns:
        Sanitized text string with special control tokens removed.

    """
    sanitized = text
    for token in SPECIAL_TOKENS:
        sanitized = sanitized.replace(token, "")
    return sanitized


def create_prompt(user_input: str, tools: str) -> str:
    """
    Creates the prompt for the LLM based on user input and available tools.

    Args:
        user_input: The natural language request from the user.
        tools: JSON string describing available tools.

    Returns:
        Formatted chat prompt string ready for inference.

    """
    safe_user_input = sanitize_input(user_input)

    # in the system_msg I pass the tools it can use
    # and give some few shots examples
    system_msg = (
        "You are a helpful assistant that uses tools. "
        "Based on the user's request, you must call the "
        "appropriate tool with the correct arguments by wrapping the JSON "
        "in <tool_call> tags.\n\n"
        "When constructing regular expressions, use character sets like "
        "[aeiou] for matching specific characters, \\d+ for digits, "
        "and \\b for word boundaries.\n\n"
        f"You have access to the following tools:\n{tools}\n\n"
        "---\n"
        "Here are some examples:\n\n"
        "User: Multiply 45 by 11\n"
        "Assistant: <tool_call>\n"
        '{"name": "fn_multiply_numbers", '
        '"parameters": {"a": 45.0, "b": 11.0}}\n\n'
        "</tool_call>\n"
        "User: can you reverse the word 'banana'?\n"
        "Assistant: <tool_call>\n"
        '{"name": "fn_reverse_string", '
        '"parameters": {"s": "banana"}}\n\n'
        "</tool_call>\n"
        "User: Substitute the digits in the string "
        "'Hello 34 I'm 233 years old' with 'NUMBERS'\n"
        "Assistant: <tool_call>\n"
        '{"name": "fn_substitute_string_with_regex", '
        '"parameters": {"source_string": "Hello 34 I\'m 233 years old", '
        '"regex": "\\\\d+", "replacement": "NUMBERS"}}\n\n'
        "</tool_call>\n"
        "User: Replace vowels in 'hello' with '*'\n"
        "Assistant: <tool_call>\n"
        '{"name": "fn_substitute_string_with_regex", '
        '"parameters": {"source_string": "hello", '
        '"regex": "[aeiouAEIOU]", "replacement": "*"}}\n'
        "</tool_call>\n"
        "---\n\n"
    )

    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{safe_user_input}<|im_end|>\n"
        # Notice we end by pre-filling the exact start of the tool call
        # This completely bypasses the model's desire to output <think>
        f"<|im_start|>assistant\n<tool_call>\n"
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
    # Isolate context growth to a local list to prevent caller side-effects
    # lists are mutable references. So I make a copy. Also,
    # working_ids is the full input context
    # the entire previous prompt plus all newly generated tokens
    working_ids: list[int] = list(input_ids)

    # Collects only the new tokens generated by the model
    answer_ids: list[int] = []

    for _ in range(MAX_TOKENS):
        print(".", end="", flush=True)

        # The logits vector returned by the model is a
        # list equal to the size of the model's vocabulary
        logits = llm.get_logits_from_input_ids(working_ids)

        # Get token index with the highest logit
        # could use numpy argmax but the llm class returns a python list,
        # so we use max with enumerate
        next_token_id = max(enumerate(logits), key=lambda x: x[1])[0]

        working_ids.append(next_token_id)
        answer_ids.append(next_token_id)

        if next_token_id in STOP_TOKEN_IDS:
            break

    return answer_ids
