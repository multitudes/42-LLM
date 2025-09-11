# src/bpe_tokenizer.py
import re

SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def get_pairs(tokens):
    """Return set of adjacent token pairs."""
    return {(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)}


def preprocess_for_bpe(text):
    # Add a special marker for spaces (e.g., "Ġ")
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
    that should be treated as single tokens and not split further.
    151644: <|im_start|>
    151645: <|im_end|>
    """
    pattern = "(" + "|".join(re.escape(tok) for tok in SPECIAL_TOKENS) + ")"
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
    # debug
    # print("Final tokens before vocab mapping:", tokens)
    # Map tokens to IDs
    return [vocab[token] for token in tokens if token in vocab]


def custom_decode(ids, id_to_token, skip_ids={151644, 151645, 151667, 151668}):
    """
    Custom decode function to convert token IDs back to text.
    There are some special token IDS that we want to skip in the response:
    151667: <think>
    151668: </think>
    Those tokens are not included in the vocab dictionary we use for decoding.
    """
    tokens = [id_to_token.get(i, "<unk>") for i in ids if i not in skip_ids]
    text = "".join(tokens)
    text = text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
    return text
