# AGENTS.md

## Coding Conventions

- Write all code comments in English.
- Add type annotations to all function and method parameters and return values. Use concrete, parameterized types whenever possible.
- Add explicit type annotations to complex local variables. This includes non-obvious composite types such as lists of structured records, nested dictionaries, and tuples of tensors.
- For complex dictionaries, prefer named data structures over untyped dictionaries. Use `pydantic.BaseModel` when practical; otherwise, use `TypedDict` or `dataclass`. This keeps data schemas documented and enables static type checking.
- Organize distinct functional blocks using `# region` and `# endregion` comments. Each region should include a concise description of its purpose.
- Use the module-level docstring at the top of every Python source file as persistent implementation memory. It should be the file's first statement. Organize entries under dated headings (`# YYYY-MM-DD`), update them when behavior, configuration, design decisions, outputs, or usage changes, and preserve still-relevant history. Document supported workflows and, when applicable, runnable Bash commands for entry points. Keep lines reasonably within Ruff's configured line-length in pyproject.toml (currently 119 characters), without repeatedly adjusting line breaks to match it exactly.

  ````python
  """Future agents: maintain this file's implementation memory in this region.

  # YYYY-MM-DD

  Describe the file's behavior, configuration, design decisions, and outputs. Wrap every physical line to Ruff's
  configured `line-length`.

  (Optional) Run from the repository root:

  ```bash
  conda run -n deep_learning_2026 python path/to/script.py
  ```
  """
  ````
- Structure executable Python modules around a `main()` function and invoke it through an `if __name__ == "__main__":` guard. Do not execute application logic or declare mutable variables directly at the module level. Module-level declarations should be limited to constants, type aliases, class definitions, and function definitions, with constants named using `UPPER_SNAKE_CASE`. When command-line arguments are required, do not use `argparse`; instead, define a configuration class using `pydantic_settings.BaseSettings` with `cli_parse_args=True` and parse it using `CliApp.run()`:

  ```python
  from pydantic_settings import BaseSettings, CliApp


  class Config(BaseSettings, cli_parse_args=True):
      ...


  def main() -> None:
      config: Config = CliApp.run(Config)
      ...


  if __name__ == "__main__":
      main()
  ```
- After adding or modifying Python code, apply both commands to every changed Python file, in this order:
  ```bash
  ruff check --fix <changed-python-files>
  ruff format <changed-python-files>
  ```
  Run these commands after the final code edit and before reporting completion.
- During writing Markdown, keep each paragraph on a single line. Do not insert manual line breaks within a paragraph.


## Documentation

### Docstrings

Docstrings must follow the format below. Do **not** convert docstrings to other styles (Google, NumPy, etc.).

Rules:

- Types appear in backticks inside parentheses: (`str`)
- Optional parameters are marked with `*optional*`
 Defaults are written as: `defaults to <value>`
- When the default is `None`, prefer ```(`str`, *optional*)``` instead of ```(`str` or `None`, *optional*, defaults to `None`)```
- Union types use `or`: `str` or `None`
- References to classes use the format: [`~transformers.PreTrainedModel`]
- Class docstrings may group parameters using headers such as: `> Parameters for X:`

Example:

````python
def method(self, param1: str, param2: int = 1, param3: float | None = None):
    """
    Brief one-line description of what this does.

    Args:
        param1 (`str`):
            Description of required param.
        param2 (`int`, *optional*, defaults to `1`):
            Description of optional param with default.
        param3 (`float`, *optional*):
            Description of optional param without explicit default.

    Returns:
        `tuple` with:
            - `list[InferenceResult]`: the resulting inference results.
            - `datetime`: the time inference started.
            - `float`: the number of seconds inference took.

    Examples:

    ```python
    >>> my_func("hello")
    ```
    """
````

## Environment Setup

I use conda to manage Python environments. The environment for this project is named `deep_learning_2026`. You can use `pip install` to install missing packages.
