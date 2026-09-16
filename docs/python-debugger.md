# Interactive Debugging with `pdb`

I use Python's built-in interactive debugger (`pdb`) to step through execution, inspect runtime state, and troubleshoot complex tokenizer operations or JSON tool-calling logic.

## Launching the Debugger

I launch the debugger via `uv` to ensure execution runs within the project's virtual environment:

```zsh
# Run the main entry point under pdb
uv run python -m pdb -m src

# Run a specific script under pdb
uv run python -m pdb src/main.py

```

## Essential Command Workflow

When paused inside a debugging session, I rely on the following core commands:

* **`b <location>`** — Set a breakpoint by line number or function name (e.g., `b extract_json_from_response` or `b src/bpe_tokenizer.py:42`).
* **`c`** — **Continue** execution at full speed until the next breakpoint or exception.
* **`n`** — **Next**: execute the current line and advance to the next line in the current function.
* **`s`** — **Step**: step inside a function call on the current line.
* **`p <expr>` / `pp <expr>**` — **Print** or **pretty-print** the runtime value of a variable or data structure.
* **`l` / `ll**` — **List** source code surrounding the current line or view the full function (`ll`).
* **`w`** — **Where**: print the current call stack trace.
* **`q`** — **Quit** and abort execution immediately.

## Inline Breakpoints

Instead of running the entire script from line 1 under `pdb`, I insert targeted inline breakpoints directly into my source code during active development:

```python
# Modern Python 3.7+ built-in breakpoint
breakpoint()

```

When execution hits `breakpoint()`, Python automatically drops into an interactive `pdb` session in the active terminal window.

## References

* [Python pdb Documentation](https://docs.python.org/3/library/pdb.html)