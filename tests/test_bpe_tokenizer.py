from src.bpe_tokenizer import (
    bpe_tokenize,
    custom_decode,
    create_prompt,
    get_pairs,
    preprocess_for_bpe,
)


def test_preprocess_for_bpe() -> None:
    """Verify whitespace characters are replaced by BPE sequence markers."""
    raw = "Hello world\nTab\tcheck"
    processed = preprocess_for_bpe(raw)
    assert processed == "HelloĠworldĊTabĉcheck"


def test_get_pairs() -> None:
    """Verify adjacent token pairs are correctly generated from a list."""
    tokens = ["h", "e", "l", "l", "o"]
    pairs = get_pairs(tokens)
    expected = {("h", "e"), ("e", "l"), ("l", "l"), ("l", "o")}
    assert pairs == expected


def test_bpe_tokenize_with_special_tokens() -> None:
    """Verify tokenization preserves special control tokens and maps IDs."""
    vocab = {"<|im_start|>": 151644, "H": 1, "ello": 2, "<|im_end|>": 151645}
    merge_ranks = {("H", "ello"): 0}

    text = "<|im_start|>Hello<|im_end|>"
    ids = bpe_tokenize(text, vocab, merge_ranks)

    # Should contain special tokens plus mapped IDs
    assert 151644 in ids
    assert 151645 in ids


def test_custom_decode() -> None:
    """
    Verify decoding filters special tokens and restores whitespace
    formatting.
    """
    # Setup dummy vocabulary inverse lookup (id_to_token)
    id_to_token = {
        151644: "<|im_start|>",
        1: "HelloĠworld",
        2: "ĊNewlines",
        151645: "<|im_end|>",
    }

    ids = [151644, 1, 2, 151645]
    decoded = custom_decode(ids, id_to_token)

    # Special tokens should be stripped, BPE markers replaced
    assert "<|im_start|>" not in decoded
    assert "<|im_end|>" not in decoded
    assert decoded == "Hello world\nNewlines"


def test_create_prompt_format() -> None:
    """Verify prompt builder structures chat template correctly."""
    user_input = "Add 5 and 10"
    tools_json = '{"name": "add"}'

    prompt = create_prompt(user_input, tools_json)

    assert "<|im_start|>system" in prompt
    assert tools_json in prompt
    assert "<|im_start|>user" in prompt
    assert user_input in prompt
    assert "<|im_start|>assistant\n" in prompt
