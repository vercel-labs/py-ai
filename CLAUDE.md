1. use `uv` to manage the project; `uv add` and `uv remove` to manage dependencies, `uv run` to run
2. after making changes run lint and typecheck: `uv run ruff check --fix src tests` and `uv run mypy src tests`
3. import by module (except `typing`) to improve readability via namespacing
4. treat `stream_step` and `stream_loop` as user code. they are convenience functions that could be reimplemented by the user, they *must* stay clean.
