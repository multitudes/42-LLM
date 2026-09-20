## Performance Analysis & Architectural Bottlenecks

In this implementation, inference latency is heavily constrained by two foundational bottlenecks in the low-level model generation loop. Because the underlying SDK framework isolates execution away from raw framework primitives (e.g., PyTorch KV caching and native tensor ops), these bottlenecks operate at the runtime layer and cannot be refactored from higher-level application code.

---

### 1. Autoregressive Sequence Re-Evaluation (Missing KV Cache)

In transformer-based language models, generating text is an autoregressive process where each new token depends on all previous tokens.

#### The Mechanism & Bottleneck

When generating a token sequence, the model relies on **Key ($K$)** and **Value ($V$)** matrix projections for every attention head across every layer.

In an un-cached setup:

1. **Token 1:** Input $[t_1 \dots t_N]$ $\longrightarrow$ Compute $K, V$ for $N$ tokens $\longrightarrow$ Predict $t_{N+1}$
2. **Token 2:** Input $[t_1 \dots t_{N+1}]$ $\longrightarrow$ Compute $K, V$ for $N+1$ tokens $\longrightarrow$ Predict $t_{N+2}$
3. **Token 3:** Input $[t_1 \dots t_{N+2}]$ $\longrightarrow$ Compute $K, V$ for $N+2$ tokens $\longrightarrow$ Predict $t_{N+3}$

Because key-value projections for static context (such as the system prompt and few-shot tool definitions) are non-mutable, re-computing them at every single step introduces severe computational redundancy.

#### Computational Impact

* **Time Complexity:** $O(N \cdot T + T^2)$ where $N$ is prompt length and $T$ is the number of generated tokens.
* **Impact:** For long system prompts containing multiple JSON tool schemas, $N \gg T$. Evaluating the static prompt repeatedly dominates total generation time, causing step latency to scale linearly with output length rather than remaining constant.

---

### 2. Full-Vocabulary Logit Materialization in Python

The output layer of Qwen3-0.6B projects hidden states onto a vocabulary dimension of $V \approx 151,646$ logits per token step.

#### The Mechanism & Bottleneck

When token selection is performed by converting raw model output tensors into native Python primitives via `.tolist()`, the model incurs two massive execution overheads:

1. **CPython Heap Allocations:** Converting a 151,646-element vector into a Python list instantiates ~151,646 discrete `PyFloatObject` heap instances in CPython memory per generated token.
2. **Interpreted Loop Overhead:** Running `max(enumerate(logits))` in Python executes an $O(V)$ loop inside the interpreted CPython runtime rather than utilizing compiled vector operations.

#### Quantitative Impact

For a test suite generating $T = 1,012$ total tokens across $P = 11$ prompts:

$$\text{Heap Allocations} = 1,012 \text{ tokens} \times 151,646 \text{ floats} \approx 153,465,752 \text{ objects}$$

Over **153 million Python float objects** are dynamically allocated, inspected, and garbage-collected, making the Python interpreter loop the primary bottleneck rather than matrix arithmetic.

---

### Architectural Summary & Comparison

| Metric / Feature | Current Framework Behavior | Standard Production Architecture |
| --- | --- | --- |
| **KV Cache State** | Disabled / Absent | Enabled (`past_key_values`) |
| **Generation Step Complexity** | $O(N + i)$ (Grows per token) | $O(1)$ (Constant decode step) |
| **Total Sequence Complexity** | $O(N \cdot T + T^2)$ | $O(N + T)$ |
| **Logit Transfer Boundary** | $151,646 \text{ floats} \rightarrow$ Python | $1 \text{ scalar int} \rightarrow$ Python |
| **Token Selection Runtime** | Python C interpreter loop (`max()`) | Native C++/CUDA kernel (`argmax`) |
| **Memory Allocation Overhead** | $O(V)$ Python objects per token | $O(1)$ scalar return |
