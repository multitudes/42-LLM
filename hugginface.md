# Using Hugginface

So we are not allowed to use transformers package but because it is already in the 
llm_sdk...
I need to instantiate the model with the `Small_LLM_Model` class
```
llm = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")  # or your preferred model

prompt = "Is 4 an even number?"
input_ids = llm._tokenizer.encode(prompt, add_special_tokens=False)
logits = llm.get_logits_from_input_ids(input_ids)
print("Logits for next token:", logits)

vocab_path = llm.get_path_to_vocabulary_json()
print("Vocabulary file path:", vocab_path)
```
```
tokenizer_config.json: 9.73kB [00:00, 5.50MB/s]
vocab.json: 2.78MB [00:00, 19.6MB/s]
merges.txt: 1.67MB [00:00, 25.0MB/s]
tokenizer.json: 100%|█████████████████| 11.4M/11.4M [00:01<00:00, 8.01MB/s]
config.json: 100%|████████████████████████| 726/726 [00:00<00:00, 7.41MB/s]
model.safetensors: 100%|██████████████| 1.50G/1.50G [02:00<00:00, 12.5MB/s]
```

This output shows that the Hugging Face `transformers` library is downloading the model and its associated files from the Hugging Face Hub for the first time.

Here’s what’s happening:
- `tokenizer_config.json`, `vocab.json`, `merges.txt`, `tokenizer.json`: These files are needed for the tokenizer, which splits text into tokens the model understands.
- `config.json`: Configuration for the model architecture.
- `model.safetensors`: The actual neural network weights (the largest file, often several GB).

The progress bars and file sizes indicate download status.  
Once downloaded, these files are cached locally and will not be downloaded again unless you clear the cache or use a different model.

**Summary:**  
- Your code is instantiating `Small_LLM_Model`, which triggers Hugging Face to download the model and tokenizer files.
- This is normal and only happens the first time you use a new model.
- After download, you can use the model for inference as shown in your code.

Great question — you’re exactly right that when you load a Hugging Face model, it pulls down a *bundle* of files that together make the model usable. Let’s break them down and explain how they fit together, with some intuition about what’s going on under the hood:

---

### 1. **Tokenizer files**

These define *how raw text (words, sentences) get turned into tokens (numbers)* that the model can actually process.

* **`vocab.json`** (for BPE/WordPiece models):
  This is a dictionary mapping each token string → an integer ID.
  Example: `"hello" → 12345`.
  Tokens aren’t always full words — depending on the tokenizer, they might be wordpieces (`"ing"`), subwords, or even characters.

* **`merges.txt`** (specific to BPE tokenizers):
  Byte-Pair Encoding works by merging frequent pairs of characters/subwords into bigger tokens. This file lists the merge rules the tokenizer applies to build tokens from raw text.
  Example: `"h" + "ello" → "hello"`.

* **`tokenizer.json`**:
  A self-contained file that bundles the vocabulary, merges, and tokenizer config in a more portable format. Sometimes it’s used instead of separate `vocab.json` + `merges.txt`.

* **`tokenizer_config.json`**:
  Metadata about the tokenizer — like special tokens (`[PAD]`, `[CLS]`, `<eos>`), casing rules, whether to strip accents, etc.

👉 Together, these files answer the question:
*"Given raw text, how do I turn it into a sequence of numbers (token IDs) the model understands?"*
And also the reverse: *"Given output IDs, how do I turn them back into readable text?"*

---

### 2. **Model configuration**

* **`config.json`**:
  This describes the **architecture and hyperparameters** of the model, so the library knows how to rebuild the neural network before loading the weights.

  Example entries:

  * number of layers (`n_layer: 12`)
  * hidden size (`hidden_size: 768`)
  * number of attention heads
  * dropout rates
  * type of model (`bert`, `gpt2`, etc.)

👉 Without this, the weights are just a big blob of numbers. The config tells the library how to wire them into an actual neural network.

---

### 3. **Model weights**

* **`model.safetensors`** (sometimes `.bin` instead):
  This is the *actual trained neural network parameters* — millions or billions of floating-point numbers (weights and biases for every layer).

  * **What’s inside:** Arrays (tensors) that represent things like attention matrices, feedforward layer weights, embeddings, etc.
  * **Why `.safetensors`:** It’s a safer and faster alternative to pickle-based `.bin` files. It avoids arbitrary code execution and loads faster.
  * **Size:** This is why the download is so huge (gigabytes). The sheer number of parameters is what makes LLMs powerful.

👉 Think of this as the "memory" of the model: the knowledge it has absorbed during training.

---

### ⚙️ How they work together

Let’s walk through an example pipeline:

1. You write: `"The cat sat on the mat."`
2. **Tokenizer** (using `vocab.json` + `merges.txt` + config) → `[101, 2003, 3948, 102]` (IDs).
3. **Config.json** tells Transformers how to build, say, a 12-layer GPT or BERT model.
4. **Model.safetensors** loads all the trained weights into that architecture.
5. The model processes the token IDs → produces hidden states or logits.
6. Those outputs (IDs) get passed back through the tokenizer → `"The cat is sleeping."`


Perfect 👍 let’s peek inside each file type so you can *see* what’s really there. (I’ll use a small GPT-2 model as an example, since it’s lightweight and the files are readable.)

---

### 🔤 `vocab.json`

This is a giant dictionary of tokens → IDs. For GPT-2 (which uses BPE), tokens look weird because they include special byte encodings:

```json
{
  "!": 0,
  "\"": 1,
  "#": 2,
  "$": 3,
  "%": 4,
  "hello": 31373,
  " world": 995,
  "Ġcat": 5876
}
```

* Keys = token strings (can be whole words, subwords, or byte sequences).
* Values = integer IDs used inside the model.

👉 Notice `"Ġcat"`: the `Ġ` means “a space before this token.” So `" cat"` is one token, not `" "` + `"cat"`.

---

### 🔗 `merges.txt`

The BPE merge rules, one per line:

```
#version: 0.2
Ġ c
a t
Ġ ca
cat </w>
```

* These rules tell the tokenizer which smaller pieces to merge into bigger tokens.
* Example: it might first split into chars `c a t`, then merge `"c"+"a"="ca"`, then `"ca"+"t"="cat"`.

---

### 🛠️ `tokenizer_config.json`

General settings for how the tokenizer should behave:

```json
{
  "do_lower_case": false,
  "bos_token": "",
  "eos_token": "",
  "unk_token": "<|unk|>",
  "padding_side": "right",
  "model_max_length": 1024
}
```

* Special tokens (beginning/end of sentence, unknown, etc.).
* Whether to lowercase text before tokenization.
* Max sequence length the model can handle.

---

### 🧩 `config.json`

Defines the **neural net architecture**:

```json
{
  "activation_function": "gelu_new",
  "architectures": ["GPT2LMHeadModel"],
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "vocab_size": 50257,
  "n_positions": 1024,
  "n_ctx": 1024
}
```

* `n_layer=12`: number of Transformer blocks.
* `n_head=12`: attention heads per layer.
* `n_embd=768`: hidden dimension size.
* `vocab_size=50257`: must match `vocab.json`.

👉 The library uses this to reconstruct the model architecture in code.

---

### 🧠 `model.safetensors`

This one is binary, but we can look at it with Hugging Face tools:

Example of what you’d see if you list its tensors:

```
wte.weight         (50257, 768)   # word embedding table
wpe.weight         (1024, 768)    # position embeddings
h.0.attn.c_attn.weight   (768, 2304)
h.0.attn.c_attn.bias     (2304,)
h.0.attn.c_proj.weight   (768, 768)
...
lm_head.weight     (50257, 768)
```

* Each entry = a tensor (array of floats).
* Shapes correspond to layers (embeddings, attention, feedforward, etc.).
* These are the *trained parameters* — billions of numbers like `-0.01423, 0.2837, ...`

👉 Without the config, the weights are just raw arrays. With config + code, they become the Transformer model you can run.

Great! Let’s walk through what happens in code when you do something like:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

---

### 🔍 Step 1: Download files

When you call `.from_pretrained("gpt2")`, Transformers:

* Looks on your disk cache first.
* If missing, pulls the files from the Hugging Face Hub (`config.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, `model.safetensors`).

These get stored in `~/.cache/huggingface/transformers/` by default.

---

### 📖 Step 2: Load the tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

1. Reads `tokenizer_config.json` → decides which tokenizer class to use (BPE, WordPiece, SentencePiece, etc.).
2. Loads `vocab.json` (map from token → ID).
3. Loads `merges.txt` (rules for combining byte pairs).
4. Wraps them in a Python `PreTrainedTokenizerFast` object.

👉 Now you can do:

```python
ids = tokenizer("The cat sat on the mat")
print(ids.input_ids)  # e.g. [464, 2019, 4005, ...]
```

---

### 🏗️ Step 3: Load the model architecture

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

1. Reads `config.json`.

   * Constructs the Python class `GPT2LMHeadModel` with `12` layers, `12` heads, hidden size `768`, etc.
   * At this point, the model exists in memory but with random/uninitialized weights.

---

### 🧠 Step 4: Load the weights

1. Opens `model.safetensors`.
2. Matches tensor names (`h.0.attn.c_attn.weight`, etc.) with the model’s layers.
3. Loads the arrays of numbers into the correct spots in the network.

👉 Now the model is identical to the one that was trained originally.

---

### 🔁 Step 5: Inference loop

Example:

```python
inputs = tokenizer("The cat sat", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0]))
```

* `tokenizer` turns text → IDs.
* `model` processes IDs through the Transformer layers (using the weights).
* Output IDs → back through `tokenizer.decode()` → text.

---

⚡ **In summary**:

* **Tokenizer files** = rules for converting text to IDs.
* **Config** = blueprint for the neural net.
* **Safetensors** = the learned knowledge (weights).
* Hugging Face just automates: *download → rebuild model → inject weights → ready to run*.

---
