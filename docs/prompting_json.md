# Function Calling via JSON Prompt Injection

I implement function selection (tool calling) by injecting tool definitions from `exercise_input/functions_definition.json` directly into the LLM system prompt. This allows my pipeline to select functions dynamically and extract target arguments from user queries.

## Pipeline Workflow

1. **Load Tool Schemas:** I read the function definitions dynamically from the JSON configuration file using `pathlib`:

```python
import json
from pathlib import Path

functions_path = Path("exercise_input/functions_definition.json")
with functions_path.open("r", encoding="utf-8") as f:
    functions = json.load(f)

```

2. **Construct System Context:** I inject the serialized tool definitions into my system prompt template along with the incoming user prompt:

```text
Available functions:
{functions_json}

User query: 2 + 2

Select the appropriate function from the list above to answer the query. Return the result strictly as a JSON object matching the required schema.

```

3. **Structured Model Output:** Instead of outputting a plain-text answer directly, the model returns a JSON payload containing the function name (`fn_name`, e.g., `"fn_add_numbers"`) and required arguments (`args`).
4. **Parsing & Execution:** My `extract_json_from_response` utility captures the JSON block, validates it against my `SelectedFunction` Pydantic model, enforces strict type coercion on parameters, and routes execution to the targeted Python function.