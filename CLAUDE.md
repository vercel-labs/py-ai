# ai

## development guidelines

1. use `uv` to manage the project; `uv add` and `uv remove` to manage dependencies, `uv run` to run
2. after making changes run lint and typecheck: `uv run ruff check --fix src tests` and `uv run mypy src tests`
3. import by module (except `typing`) to improve readability via namespacing
4. treat `stream_step` and `stream_loop` as user code. they are convenience functions that could be reimplemented by the user, they *must* stay clean.

## design principles

### 1. maximize composability

provide simple lego bricks that the user can build their feature with. each block should do one thing and be reasonably decoupled from the rest.
expose correct primitives to make it easy to modify behavior without rewriting it from scratch.

- *example*: `agents` module provides `@ai.stream`, `@ai.tool` and `@ai.hook` that can be combined into an arbitrarily complex agent graph using plain python.
- *can the user rewrite this feature in plain python using the existing primitives?*

### 2. minimize dsl-ness and frameworkiness

express features in a way that doesn't require the user to read documentation and learn the framework. glue things together using python.
handle complexity inside the framework instead of delegating it to users.

- *example*: `Runtime` does the heavy lifting so that multi-agent graphs can be expressed using python `asyncio`.
- *does this require the user to learn a framework-specific concept that has a direct python equivalent?*

### 3. keep data model simple

ensure state is easy to serialize and deserialize, modify, and compose at any level of granularity.
move normalization and translation complexity inside the framework and keep the public data model minimal.

- *example*: public data model consists of a single unified `Message` type. the framework does not expose events and other intermediate steps unless the user is writing a custom adapter.


