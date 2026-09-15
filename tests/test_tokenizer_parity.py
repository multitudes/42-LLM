import json
from pathlib import Path
from typing import cast

import pytest

from llm_sdk import Small_LLM_Model
from src.bpe_tokenizer import bpe_tokenize, custom_decode


def load_vocab_json(path: str) -> dict[str, int]:
    vocab_path = Path(path)
    with vocab_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        return cast(dict[str, int], data)


def load_merges_txt(path: str) -> dict[tuple[str, str], int]:
    merges = {}
    merge_path = Path(path)
    with merge_path.open("r", encoding="utf-8") as f:
        rank = 0
        for line in f:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue
            parts = line_stripped.split()
            if len(parts) == 2:
                merges[(parts[0], parts[1])] = rank
                rank += 1
    return merges


@pytest.fixture(scope="module")
def tokenizer_assets() -> tuple[
    Small_LLM_Model,
    dict[str, int],
    dict[tuple[str, str], int],
    dict[int, str],
]:
    """
    Module-scoped fixture to load the model assets and custom tokenizer data
    once.
    """
    model = Small_LLM_Model()

    # Retrieve paths to model assets using SDK helpers
    vocab_path = model.get_path_to_vocab_file()
    merges_path = model.get_path_to_merges_file()

    # Load vocabulary and merges
    vocab = load_vocab_json(vocab_path)
    merges = load_merges_txt(merges_path)
    id_to_token = {v: k for k, v in vocab.items()}

    return model, vocab, merges, id_to_token


TEST_STRINGS = [
    "Hello World",
    "Python 3.10 with PyTorch inference",
    "Line one\nLine two\tIndented",
    "Punctuation check: (1 + 2) * 3 = 9!",
    "Subword tokenization test for unexpected vocabulary items.",
]


@pytest.mark.parametrize("sample_text", TEST_STRINGS)
def test_encode_parity(tokenizer_assets, sample_text: str):
    """
    Verify that custom bpe_tokenize produces identical token IDs
    to Small_LLM_Model.encode (Hugging Face reference).
    """
    model, vocab, merges, _ = tokenizer_assets

    # Reference HF Encoding (2D Tensor -> 1D list)
    ref_tensor = model.encode(sample_text)
    ref_ids = ref_tensor.squeeze(0).tolist()

    # Custom BPE Encoding
    custom_ids = bpe_tokenize(sample_text, vocab, merges)

    assert custom_ids == ref_ids, (
        f"\n[ENCODE MISMATCH] for input: '{sample_text}'"
        f"\nCustom Output:    {custom_ids}"
        f"\nReference Output: {ref_ids}"
    )


@pytest.mark.parametrize("sample_text", TEST_STRINGS)
def test_decode_parity(tokenizer_assets, sample_text: str):
    """
    Verify that custom_decode produces identical plain text output
    to Small_LLM_Model.decode (Hugging Face reference).
    """
    model, _, _, id_to_token = tokenizer_assets

    # Get baseline token IDs from reference encoder
    token_ids = model.encode(sample_text).squeeze(0).tolist()

    # Reference HF Decoding
    ref_decoded = model.decode(token_ids)

    # Custom Decoding
    custom_decoded = custom_decode(token_ids, id_to_token)

    assert custom_decoded == ref_decoded, (
        f"\n[DECODE MISMATCH]"
        f"\nCustom Output:    '{custom_decoded}'"
        f"\nReference Output: '{ref_decoded}'"
    )
