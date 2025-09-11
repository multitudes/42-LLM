## Python packages

Just as a recap for me... since we use uv and the code to run the program is 
```
	@uv run python -m src
```
so the src has to have a package structure.


In Python, a **folder containing an `__init__.py` file is a package**.

- A **module** is a single `.py` file (e.g., `ollama.py`).
- A **package** is a directory with an `__init__.py` file (e.g., llm_sdk).

The `__init__.py` file can be empty or contain code.  
It tells Python that the folder should be treated as a package, allowing you to import from it:

```python
from llm_sdk import call_ollama_api
```

You can also have submodules (other `.py` files) inside the package and import them:

```python
from llm_sdk.ollama import call_ollama_api
```

**Summary:**  
- llm_sdk with `__init__.py` = package  
- `llm_sdk/ollama.py` = module inside the package