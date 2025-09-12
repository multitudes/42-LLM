# src/bpe_tokenizer.py
import re
import json

MAX_TOKENS = 150
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
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading vocabulary: {e}")
    for tok, tid in SPECIAL_TOKENS.items():
        if tok not in vocab:
            vocab[tok] = tid
    try:
        with open(MERGES_PATH, "r") as f:
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
    return [vocab[token] for token in tokens if token in vocab]


def custom_decode(ids, id_to_token):
    """
    Custom decode function to convert token IDs back to text.
    There are some special token IDS that we want to skip in the response:
    151667: <think>
    151668: </think>
    See SPECIAL_TOKENS dictionary above for reference.
    Those tokens are not included in the vocab dictionary we use for decoding.
    """
    skip_ids = set(SPECIAL_TOKENS.values())
    tokens = [id_to_token.get(i, "<unk>") for i in ids if i not in skip_ids]
    text = "".join(tokens)
    text = text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
    print("\n\nllm output:", text, end="")
    return text


def create_prompt(user_input: str, tools: str) -> str:
    """
    Create the prompt for the LLM based on user input and available tools.
    """
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
        f"<|im_start|>user\n{user_input}/no_think<|im_end|>\n"
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
    args:
        llm: instance of Small_LLM_Model class
        input_ids: list of input token ids (integers)
    returns: list of generated token ids (integers)
    The generation stops when either the maximum number of tokens
    is reached or when the end token is generated.
    """
    answer_ids = []
    for _ in range(MAX_TOKENS):
        print(".", end="", flush=True)
        logits = llm.get_logits_from_input_ids(input_ids)
        next_token_id = max(enumerate(logits), key=lambda x: x[1])[0]
        input_ids.append(next_token_id)
        answer_ids.append(next_token_id)
        if (next_token_id == END_TOKEN_ID1 or next_token_id == END_TOKEN_ID2):
            break
    return answer_ids
