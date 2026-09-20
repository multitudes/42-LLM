## Testing with Pytest `monkeypatch`

The `monkeypatch` fixture is a built-in Pytest tool that allows you to safely modify classes, functions, environment variables, or dictionaries during a test run. Once the test completes, Pytest automatically restores everything to its original state, ensuring tests remain isolated and side-effect free.

---

### Why Use `monkeypatch`?

In LLM pipelines and automated CLI tools, running tests against live models, actual GPUs, or hardware file systems is slow, expensive, and fragile. `monkeypatch` allows you to:

* **Isolate Unit Tests:** Mock heavy dependencies like CUDA device allocation, HuggingFace downloads, or API calls.
* **Simulate Error States:** Force specific exceptions (e.g., `TypeError`, `RuntimeError`, `OOM`) to verify error handling and exit codes.
* **Prevent State Bleed:** Changes automatically revert after each test function finishes.

---

### Key Methods Overview

| Method | Syntax | Purpose |
| --- | --- | --- |
| `setattr` | `monkeypatch.setattr(target, name, value)` | Replaces a function, method, or attribute with a mock or substitute value. |
| `delattr` | `monkeypatch.delattr(target, name)` | Deletes an attribute from a module or class for the test duration. |
| `setenv` | `monkeypatch.setenv(name, value)` | Sets or updates an environment variable. |
| `delenv` | `monkeypatch.delenv(name)` | Removes an environment variable. |
| `setitem` | `monkeypatch.setitem(dict, key, value)` | Sets a dictionary key/value pair. |

---

### Practical Examples

#### 1. Mocking Function Return Values

To test a CLI command without actually invoking a real model or reading disk files, patch the utility functions to return static data:

```python
from typing import Any
import pytest
from src.__main__ import main

def test_main_runs_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main execution loop with mock inputs."""
    # Substitute file loaders with fast, static return values
    monkeypatch.setattr("src.utils.get_functions", lambda: [])
    monkeypatch.setattr("src.__main__.get_input_prompts", lambda: ["Test prompt"])
    
    # Mock the pipeline runner to avoid running actual model inference
    monkeypatch.setattr(
        "src.bpe_tokenizer.run_pipeline", 
        lambda prompts, functions_def: []
    )
    
    # Run main loop without throwing exceptions
    main()

```

#### 2. Simulating Exceptions and Failures

Verify that your error handling correctly catches failures and exits with a non-zero exit code:

```python
import pytest
from src.__main__ import main

def test_main_exits_on_file_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a invalid tools file triggers a non-zero exit status."""
    
    def mock_broken_get_functions(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("Invalid schema in functions file.")

    # Patch the function to raise an exception when called
    monkeypatch.setattr("src.utils.get_functions", mock_broken_get_functions)

    # Intercept the system exit call
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code != 0

```

#### 3. Overriding Environment Variables

Override environment configurations (like device type or log levels) without modifying system settings:

```python
import os
import pytest

def test_device_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify system forces CPU mode when CUDA is explicitly disabled."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    
    # Code under test will see CUDA_VISIBLE_DEVICES as empty
    assert os.getenv("CUDA_VISIBLE_DEVICES") == ""

```

---

### Critical Rule: Where to Patch

The target string passed to `monkeypatch.setattr("path.to.target", value)` depends on **how the symbol is imported**:

1. **When imported directly (`from src.utils import get_functions`):**
Patch the symbol where it is **used** (imported into):
```python
monkeypatch.setattr("src.__main__.get_functions", mock_func)

```


2. **When imported as a module (`import src.utils` or `.utils`):**
Patch the symbol where it is **defined**:
```python
monkeypatch.setattr("src.utils.get_functions", mock_func)

```