# BPE Tokenization Mechanics & `merges.txt`

I implement a custom Byte-Pair Encoding (BPE) tokenizer that relies on two core asset files: `vocab.json` and `merges.txt`. Together, these files allow my tokenizer to break arbitrary, out-of-vocabulary strings into meaningful sub-word tokens.

---

## The Role of `vocab.json` vs. `merges.txt`

* **`vocab.json` (The Token Dictionary):** Maps every recognized sub-word token to a unique integer ID.
* **`merges.txt` (The Merge Rulebook):** Contains an ordered list of rank-prioritized character pair merges. It teaches the tokenizer how to construct sub-words from individual characters step-by-step.

Without `merges.txt`, my tokenizer would fail to parse unseen words and fall back to unknown (`<unk>`) tokens. With `merges.txt`, it iteratively combines character pairs until every segment maps to a valid entry in `vocab.json`.

---

## How My Tokenizer Processes Text

When my custom `bpe_tokenize` function encounters an unseen word like `"tokenization"`, it executes the following sequence:

1. **Initial Character Splitting:**
The raw string is split into individual character tokens:
`['t', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n']`
2. **Sequential Merge Application:**
The tokenizer reads `merges.txt` and applies matching rules in strict order of their priority rank:
* Rule `#1` (`t` + `o` $\rightarrow$ `to`): `['to', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n']`
* Rule `#50` (`a` + `t` $\rightarrow$ `at`): `['to', 'k', 'e', 'n', 'i', 'z', 'at', 'i', 'o', 'n']`
* Higher-ranked rules merge remaining pairs (`i` + `on` $\rightarrow$ `ion`, `at` + `ion` $\rightarrow$ `ation`).


3. **Final Sub-word Output:**
The process stops when no further merges from `merges.txt` can be applied, yielding known sub-word tokens:
`['token', 'ization']`
4. **Vocabulary Lookup:**
My implementation maps these final sub-words directly to integer IDs via `vocab.json` for model inference.

---

## Why Both Files Are Required

* **`vocab.json`** defines the valid terminal states (the final set of available IDs).
* **`merges.txt`** defines the deterministic path used to reach those terminal states from raw character inputs.