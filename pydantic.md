# pydantic

**Pydantic v1 vs v2 vs v3:**

- **Pydantic v1**: The original version. Uses Python type hints for data validation and parsing. Models are based on standard Python classes. Validation is performed at runtime. Widely used and stable, but slower and less flexible for advanced use cases.

- **Pydantic v2**: Major rewrite for performance and flexibility. Uses a new core written in Rust for much faster validation. The API is more strict and explicit. Some features and syntax have changed (e.g., `model_validate` instead of `parse_obj`). Migration from v1 to v2 may require code changes.

- **Pydantic v3**: Latest version (still in development as of mid-2025). Builds on v2, with further improvements, stricter typing, and new features. May introduce breaking changes and new APIs.

**Summary:**  
- All versions provide data validation using Python type hints.
- v2 and v3 are faster and more strict than v1.
- If your project requires stability and compatibility, use v1.  
- If you want better performance and modern features, use v2 or v3 (but check your code for compatibility).

**Docs:**  
- [Pydantic v1](https://docs.pydantic.dev/1.10/)
- [Pydantic v2](https://docs.pydantic.dev/latest/)
- [Pydantic v3](https://docs.pydantic.dev/dev/)

## resources

https://pypi.org/project/pydantic/