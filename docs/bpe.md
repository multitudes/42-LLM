A comprehensive overview of Byte Pair Encoding (BPE) tokenization, designed to document your custom implementation in your repository's `README.md`.

---

# Byte Pair Encoding (BPE) Tokenizer

This repository features a custom **Byte Pair Encoding (BPE) Tokenizer** built from scratch in Python to handle model vocabulary formatting, tokenization, and decoding without external high-level tokenizer libraries.

---

## What is BPE?

Byte Pair Encoding (BPE) is a subword tokenization algorithm that builds a vocabulary dynamically by iteratively merging the most frequently adjacent pairs of characters or character sequences in a corpus.

It balances the compact representations of word-level tokenization with the out-of-vocabulary (OOV) handling of character-level tokenization.

---

## Tokenizer Pipeline & Core Components

```
 Raw Text Input ──> Preprocessing ──> Special Token Split ──> BPE Pair Merging ──> Vocab Mapping (IDs)

```

### 1. Preprocessing & Byte-Level Space Encoding

Standard white space, newlines, and tabs are mapped to explicit byte-level BPE sequence markers before subword merging occurs:

* ` ` (space) $\rightarrow$ `Ġ`
* `\n` (newline) $\rightarrow$ `Ċ`
* `\t` (tab) $\rightarrow$ `ĉ`

This preserves structural layout within single subword tokens across multiple lines or indented blocks.

### 2. Special Token Handling

Special prompt markers (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`) are protected during tokenization. The text is pre-segmented around these tokens using regex patterns, preventing special tags from being broken into arbitrary character fragments.

### 3. Iterative BPE Pair Merging

During encoding:

1. The sequence is split into individual base character tokens.
2. The tokenizer evaluates adjacent token pairs $(t_i, t_{i+1})$.
3. The pair with the lowest rank in the pre-computed `merge_ranks` mapping is merged into a single combined token string.
4. Step 2–3 repeats recursively until no further pairs exist in `merge_ranks`.

```python
# Example Pair Extraction
tokens = ["H", "e", "l", "l", "o"]
pairs = {("H", "e"), ("e", "l"), ("l", "l"), ("l", "o")}

```

### 4. Vocabulary Mapping

Once all merges are complete, each subword string is converted into its corresponding integer ID via the vocabulary dictionary (`dict[str, int]`). Missing tokens default to an unexpected or unknown representation.

---

## Decoding Strategy

Decoding reverses the encoding pipeline:

1. Filters out system/control token IDs (e.g., `<think>`, `<|im_start|>`).
2. Maps remaining integer IDs back to string subwords.
3. Replaces BPE markers (`Ġ`, `Ċ`, `ĉ`) back to standard whitespace characters (` `, `\n`, `\t`).

---

**Step-by-Step BPE Tokenization Walkthrough: `"Hello World"**`

**1. Raw Input**

```text
"Hello World"

```

**2. Preprocessing (`preprocess_for_bpe`)**
Whitespace is mapped to the byte-level BPE marker (`Ġ`):

```text
"HelloĠWorld"

```

**3. Initial Character Split**
The string is split into individual base-level character tokens:

```python
tokens = ["H", "e", "l", "l", "o", "Ġ", "W", "o", "r", "l", "d"]

```

**4. Iterative Pair Merging**
The algorithm evaluates adjacent pairs `(t_i, t_{i+1})` against `merge_ranks` iteratively:

* **Iteration 1:**
* Pairs found: `('H', 'e')`, `('e', 'l')`, `('l', 'l')`, etc.
* Lowest rank match: `('H', 'e')` $\rightarrow$ merged into `"He"`.
* **Current tokens:** `["He", "l", "l", "o", "Ġ", "W", "o", "r", "l", "d"]`


* **Iteration 2:**
* Lowest rank match: `('l', 'l')` $\rightarrow$ merged into `"ll"`.
* **Current tokens:** `["He", "ll", "o", "Ġ", "W", "o", "r", "l", "d"]`


* **Iteration 3:**
* Lowest rank match: `('He', 'll')` $\rightarrow$ merged into `"Hell"`.
* **Current tokens:** `["Hell", "o", "Ġ", "W", "o", "r", "l", "d"]`


* **Iteration 4:**
* Lowest rank match: `('Hell', 'o')` $\rightarrow$ merged into `"Hello"`.
* **Current tokens:** `["Hello", "Ġ", "W", "o", "r", "l", "d"]`


* **Final Merges:**
* Subsequent merges combine `("W", "o", "r", "l", "d")` into `"ĠWorld"`.
* **Final token list:** `["Hello", "ĠWorld"]`



**5. Vocabulary ID Mapping**
Each final token string is mapped to its integer ID from the vocabulary dictionary:

* `"Hello"` $\rightarrow$ `15496`
* `"ĠWorld"` $\rightarrow$ `2159`
* **Output IDs:** `[15496, 2159]`

**6. Decoding (`custom_decode`)**
To reverse the process:

1. Lookup IDs in `id_to_token`: `["Hello", "ĠWorld"]`
2. Join strings: `"HelloĠWorld"`
3. Replace BPE markers (`Ġ` $\rightarrow$ space): `"Hello World"`
