# Python Debugger (pdb) Basics

The Python Debugger (`pdb`) is a built-in tool for interactive debugging of Python programs. It allows you to pause execution, inspect variables, step through code, and set breakpoints.

## How to Start pdb

### From the Command Line
Run your script with pdb:

```zsh
python -m pdb src/main.py
```
Or, if using `uv`:
```zsh
uv run python -m pdb src/main.py
```

## Common pdb Commands

- `l` (list): Show source code around the current line.
- `n` (next): Execute the next line of code.
- `s` (step): Step into a function call.
- `c` (continue): Continue execution until the next breakpoint.
- `b <line>`: Set a breakpoint at the specified line number.
- `b <file>:<line>`: Set a breakpoint in a specific file and line.
- `p <expression>`: Print the value of an expression.
- `q` (quit): Exit the debugger.

## Example Debugging Session

1. Start the debugger:
   ```zsh
   python -m pdb src/main.py
   ```
2. Use `n` to step through lines, `l` to list code, and `p` to print variable values.
3. Set breakpoints with `b` and continue with `c`.

## Setting Breakpoints in Code
You can also set breakpoints directly in your code:
```python
import pdb; pdb.set_trace()
```
When execution reaches this line, the debugger will start.

## Tips
- Use `help` in the debugger for a list of commands.
- You can inspect and modify variables while paused.
- Combine with `uv run` for virtual environment support.

## References
- [Python pdb documentation](https://docs.python.org/3/library/pdb.html)

---
This guide covers the basics of using the Python debugger (`pdb`) for interactive debugging.
