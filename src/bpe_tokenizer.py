# src/bpe_tokenizer.py
import re
import json

SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
MERGES_PATH = "merges.txt"
SPECIAL_TOKENS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<think>": 151667,
    "</think>": 151668,
}


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
    """Preprocess text for BPE tokenization by replacing spaces,
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
