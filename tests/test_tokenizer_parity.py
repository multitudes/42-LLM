from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from llm_sdk import Small_LLM_Model
from src.bpe_tokenizer import bpe_tokenize, custom_decode, initialize_tokenizer


class DummyModel:
    """Stub replacing AutoModelForCausalLM without loading weights."""

    def to(self, device: Any) -> "DummyModel":
        return self

    def eval(self) -> "DummyModel":
        return self

    def parameters(self) -> list[Any]:
        return []


# Type definition for fixture return tuple
TokenizerAssets = tuple[
    Small_LLM_Model,
    dict[str, int],
    dict[tuple[str, str], int],
    dict[int, str],
]


@pytest.fixture(scope="module")
def tokenizer_assets() -> Iterator[TokenizerAssets]:
    """Load production tokenizer assets (tokenizer.json + Hub merges)."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: DummyModel(),
    )

    model = Small_LLM_Model()
    # Same paths as src/__main__.py — not vocab.json + ad-hoc merge parsing
    tokenizer_path = Path(model.get_path_to_tokenizer_file())
    merges_path = Path(model.get_path_to_merges_file())
    vocab, merges = initialize_tokenizer(tokenizer_path, merges_path)
    id_to_token = {v: k for k, v in vocab.items()}

    yield model, vocab, merges, id_to_token
    monkeypatch.undo()


TEST_STRINGS = [
    "Hello World",
    "Python 3.10 with PyTorch inference",
    "Line one\nLine two\tIndented",
    "Punctuation check: (1 + 2) * 3 = 9!",
    "Subword tokenization test for unexpected vocabulary items.",
]


@pytest.mark.parametrize("sample_text", TEST_STRINGS)
def test_encode_parity(
    tokenizer_assets: TokenizerAssets,
    sample_text: str,
) -> None:
    """Verify custom bpe_tokenize produces identical IDs to Hugging Face."""
    model, vocab, merges, _ = tokenizer_assets

    ref_tensor = model.encode(sample_text)
    ref_ids = ref_tensor.squeeze(0).tolist()

    custom_ids = bpe_tokenize(sample_text, vocab, merges)

    assert custom_ids == ref_ids, (
        f"\n[ENCODE MISMATCH] for input: '{sample_text}'"
        f"\nCustom Output:    {custom_ids}"
        f"\nReference Output: {ref_ids}"
    )


@pytest.mark.parametrize("sample_text", TEST_STRINGS)
def test_decode_parity(
    tokenizer_assets: TokenizerAssets,
    sample_text: str,
) -> None:
    """Verify custom_decode produces identical output to Hugging Face."""
    model, _, _, id_to_token = tokenizer_assets

    token_ids = model.encode(sample_text).squeeze(0).tolist()

    ref_decoded = model.decode(token_ids)
    custom_decoded = custom_decode(token_ids, id_to_token)

    assert custom_decoded == ref_decoded, (
        f"\n[DECODE MISMATCH]"
        f"\nCustom Output:    '{custom_decoded}'"
        f"\nReference Output: '{ref_decoded}'"
    )
