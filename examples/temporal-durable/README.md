# Durable Agent Execution with Temporal

Reproduces the `custom_loop.py` sample with Temporal durable execution.
LLM calls and tool calls are Temporal activities; the agent loop is a
Temporal workflow. The whole thing survives crashes and restarts.

## How it works

```
workflow.py          durable.py             activities.py
┌──────────┐    ┌───────────────────┐    ┌──────────────────┐
│ ai.run() │───>│TemporalLanguage   │───>│llm_stream_activity│
│ agent()  │    │Model.stream()     │    │ (real LLM call)   │
│          │    │  execute_activity()│    │                   │
│          │    ├───────────────────┤    ├──────────────────┤
│ execute_ │───>│temporal_tool().fn  │───>│tool_call_activity │
│ tool()   │    │  execute_activity()│    │ (real tool call)  │
└──────────┘    └───────────────────┘    └──────────────────┘
  WORKFLOW           WRAPPERS              ACTIVITIES (I/O)
  (deterministic)    (routing)             (non-deterministic)
```

- **Workflow** (`workflow.py`): The agent loop runs inside `ai.run()`, unchanged
  from the non-durable version. Uses `asyncio.TaskGroup`, `Queue`, `Future` — all
  fine inside Temporal's custom event loop.
- **Wrappers** (`durable.py`): `TemporalLanguageModel` and `temporal_tool()` replace
  real I/O with `workflow.execute_activity()` calls. On replay, Temporal returns
  cached results.
- **Activities** (`activities.py`): Real I/O happens here — LLM API calls and tool
  function execution. Activities are retried automatically on failure.

## Setup

### 1. Install Temporal CLI

```bash
brew install temporal    # macOS
# or: curl -sSf https://temporal.download/cli.sh | sh
```

### 2. Start Temporal dev server

```bash
temporal server start-dev
```

This starts a local server on `localhost:7233` with a UI at `http://localhost:8233`.

### 3. Install dependencies

```bash
cd examples/temporal-durable
uv sync
```

### 4. Set environment variables

```bash
export AI_GATEWAY_API_KEY=your-key-here
```

### 5. Run

```bash
uv run python main.py
# or with a custom query:
uv run python main.py "What is the weather in Tokyo?"
```

## Architecture notes

### Streaming limitation

Temporal activities are request/response — they can't stream. The
`TemporalLanguageModel` buffers the full LLM response inside the activity
and returns it as a single result. The workflow sees a "stream" that
completes in one shot.

### Tool registry hack

`temporal_tool()` replaces the global tool registry entry with the
activity-calling wrapper, and stashes the original function for the
activity to use. This is the "dumbest version" — a proper framework
primitive will replace this once the overall shape of durable execution
settles.

### Sandbox passthrough

Temporal's workflow sandbox re-imports modules on each replay. We use
`workflow.unsafe.imports_passed_through()` to pass `vercel_ai_sdk` and
our local modules through unchanged, since they don't do non-deterministic
things at import time.
