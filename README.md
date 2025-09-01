# 42-LLM-test
A project for 42-Global with python and alms


## Common Instructions
General Rules
- Your project must be written in Python 3.11 or later.
- Your project must adhere to the flake8 coding standard. Bonus files are also subject to
this standard.
- Your functions should handle exceptions gracefully to avoid crashes. Use try-except
blocks to manage potential errors. If your program crashes due to unhandled exceptions
during the review, it will be considered non-functional.
- All resources (e.g., file handles, network connections) must be properly managed to prevent
leaks.

## Makefile
Include a Makefile in your project to automate common tasks. It must contain the following

rules:
- install: Install project dependencies using pip, uv, pipx, or any other package manager
of your choice.
- run: Execute the main script of your project.
- debug: Run the main script in debug mode using Python’s built-in debugger.
- clean: Remove temporary files or caches to keep the project environment clean.
- lint: Lint your code using flake8 to ensure it meets coding standards.

## Additional Guidelines

- Create test programs to verify project functionality (not submitted or graded).
- Submit your work to the assigned Git repository. Only the content in this repository will
be graded.

If any additional project-specific requirements apply, they will be stated immediately below this
section.