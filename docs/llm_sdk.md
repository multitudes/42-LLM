# Model Initialization & Inference Architecture

I encapsulate model execution within a dedicated wrapper class that manages hardware acceleration, precision allocation, tensor shape constraints, and raw logit extraction for custom text generation loops.

---

## Hardware Initialization & Precision Control

When my model wrapper initializes, it automatically configures the runtime environment for optimal hardware performance:

* **Dynamic Device Selection:** The wrapper evaluates active system hardware, prioritizing NVIDIA GPUs (`cuda`) or Apple Silicon (`mps`), and falling back to `cpu` when no dedicated accelerator is available.
* **Precision Allocation (`dtype`):** On GPU and MPS runtimes, I set numerical precision to half-precision (`float16`). This cuts the model's VRAM footprint in half and accelerates matrix operations with negligible impact on generation accuracy. On CPUs, the wrapper defaults to single-precision (`float32`).
* **Model & Tokenizer Loading:** I use Hugging Face factory classes (`AutoTokenizer` and `AutoModelForCausalLM`) to dynamically load the model weights and token definitions without hardcoding specific underlying architectures.
* **Evaluation Mode & Gradient Freezing:** During initialization, I explicitly set the model to evaluation mode (`self._model.eval()`) and freeze all parameter gradients (`requires_grad = False`) to prevent unnecessary memory allocations during inference.

---

## Tensor Shape Requirements & Text Processing

### Encoding (`encode`)

My `encode` method converts raw input strings into PyTorch tensors mapped directly to the target execution device.

Transformer models expect inputs in a 2D batch format structured as `(batch_size, sequence_length)`—even for single-prompt inference:

```python
# 2D tensor format expected by transformer attention layers:
[
  [15496, 2159, 0],   # Row 1 (Prompt A): 2 content tokens + 1 padding token
  [42, 88, 99]        # Row 2 (Prompt B): 3 content tokens
]

```

Passing a 1D tensor (`[id1, id2, ...]`) would cause PyTorch to misinterpret token IDs as the batch dimension, resulting in matrix shape mismatch errors. Structuring my encoded output as a 2D tensor (`[[id1, id2, ...]]`) satisfies transformer layer constraints while allowing the pipeline to scale to multi-prompt batching seamlessly.

### Decoding (`decode`)

My `decode` method converts output token ID lists or PyTorch tensors back into human-readable strings while automatically filtering out special control tokens.

---

## Logit Extraction for Custom Generation

To support custom greedy or sampling generation algorithms, my model wrapper provides `get_logits_from_input_ids(input_ids)`:

1. Takes an active sequence of token IDs and converts them into a 2D device tensor.
2. Runs a forward pass inside a `torch.no_grad()` context block to disable gradient tracking.
3. Isolates the prediction vector for the **final token position** in the sequence (`out.logits[0, -1]`).
4. Returns these unnormalized logit scores as a raw list of floats, allowing downstream generation loops to apply top-$k$, top-$p$, or temperature sampling.

---

## SDK Asset Retrieval Helpers

To verify that my custom BPE tokenizer produces identical tokens to the reference Hugging Face model, my wrapper provides asset path helper methods:

* `get_path_to_vocab_file()`: Resolves the local filesystem path to the model's native `vocab.json`.
* `get_path_to_merges_file()`: Locates the physical `merges.txt` sub-word rulebook.
* `get_path_to_tokenizer_file()`: Returns the complete `tokenizer.json` configuration file.

These helpers supply the exact vocabulary files required by my custom unit test suite (`test_tokenizer_parity.py`) to confirm parity against the Hugging Face reference SDK.
