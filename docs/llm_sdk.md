**Initialization (`__init__`)**

* **Device Selection:** Evaluates the runtime environment to prioritize high-performance accelerators, choosing Apple Silicon (`mps`) or NVIDIA GPUs (`cuda`) when available, and falling back to `cpu`.
* **Precision Control:** Sets numerical precision to `float16` on GPUs and MPS to conserve memory bandwidth, while defaulting to `float32` on CPUs for standard compatibility.
**Data Type Definition**
**`dtype`** stands for **data type**. In PyTorch and machine learning, it defines the exact amount of computer memory (measured in bits) allocated to store numerical values, such as model weights, biases, and internal activations.

**Why It Matters for LLMs**

* **`float32` (32-bit floating-point / Single Precision):** Uses 32 bits per number. It offers maximum mathematical accuracy and is the standard fallback for CPUs, but consumes more memory and bandwidth.
* **`float16` (16-bit floating-point / Half Precision):** Uses 16 bits per number. It cuts the model's memory footprint in half, allowing larger models to fit comfortably into GPU or Apple Silicon (`mps`) VRAM while significantly speeding up calculation speeds, with a negligible impact on output accuracy.

* **Model & Tokenizer Loading:** Instantiates the Hugging Face `AutoTokenizer` and `AutoModelForCausalLM`. It assigns a fallback padding token if none is specified, moves the model to the target device, activates evaluation mode (`self._model.eval()`), and freezes weights (`requires_grad = False`) to prevent gradient tracking.

**`AutoTokenizer`** and **`AutoModelForCausalLM`** are Hugging Face factory classes designed to load text tokenizers and neural network models dynamically without needing to hardcode specific model architectures.

* **`AutoTokenizer`**: Automatically inspects a model's configuration file to select and instantiate the correct tokenizer class (e.g., handling specific vocabularies, special control tokens, and text-to-ID mappings for models like Qwen or Llama).
* **`AutoModelForCausalLM`**: Automatically loads the correct neural network architecture for **Causal Language Modeling** (autoregressive text generation), where the model is trained to predict the next token in a sequence based strictly on preceding tokens.

**Why Use "Auto" Classes?**
They provide a unified interface that abstracts away underlying architectural differences. Whether you load a Qwen, Llama, or GPT model, your code uses the exact same instantiation pattern.

**Text Encoding & Decoding**

* **`encode(text)`:** Converts a raw string into a 2-D PyTorch tensor of token IDs (`[[id1, id2, ...]]`) mapped directly to the active hardware device.
Transformer models are built to process **batches** of data simultaneously, meaning they always expect a 2-D input tensor structured as `(batch_size, sequence_length)`.

* **Batch Dimension Requirement**: Even when you are only sending a single prompt (where `batch_size = 1`), the model's internal tensor operations, embedding lookups, and attention layers are hardcoded to expect a multi-dimensional structure. A 2-D tensor like `[[id1, id2, ...]]` satisfies this requirement.
* **Preventing Shape Mismatch Errors**: If you passed a 1-D tensor (`[id1, id2, ...]`), PyTorch would interpret the first token ID as the batch size rather than part of the sequence, causing shape mismatch or dimension out-of-bounds errors during matrix multiplication.
* **Batch Processing Readiness**: This structure allows the inference code to seamlessly scale to process multiple independent prompts simultaneously (e.g., `[[prompt_1_tokens], [prompt_2_tokens]]`) without needing to rewrite the underlying neural network layers.
ex
```python
[
  [15496, 2159, 0],       # Row 1 (Prompt A): 2 real tokens + 1 padding token = 3 elements
  [42, 88, 99]             # Row 2 (Prompt B): 3 real tokens = 3 elements
]
```

* **`decode(ids)`:** Reverses tokenization by converting a PyTorch tensor or a list of integers back into a plain text string, automatically stripping out special control tokens.

**Inference & Logits Extraction**

* **`get_logits_from_input_ids(input_ids)`:** Takes an active sequence of token IDs, transforms them into a tensor, and performs a forward inference pass inside a `torch.no_grad()` context to disable tracking. It isolates the prediction vector (`out.logits[0, -1]`) for the **very last token** in the sequence, returning raw, unnormalized prediction scores (logits) as a standard float list for custom greedy or sampling generation loops.

**File Resource Fetchers**

* **`get_path_to_vocab_file()`:** Queries the Hugging Face Hub API and local cache to locate or download the model's native `vocab.json` file path.
* **`get_path_to_merges_file()`:** Retrieves the physical file path for the model's subword merge rules (`merges.txt`).
* **`get_path_to_tokenizer_file()`:** Fetches the full Hugging Face `tokenizer.json` structure. These asset helper methods allow external custom components—like your scratch-built BPE tokenizer—to hook directly into the exact vocabulary configurations of the downloaded model.