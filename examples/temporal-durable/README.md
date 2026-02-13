# Durable Agent Execution with Temporal

Two implementations of the same agent (weather + population tools) as a
Temporal workflow. Both survive crashes and restarts — every LLM call and
tool call is a durable activity that Temporal replays from history.

## `with_sdk/` — Using vercel-ai-sdk

The framework and Temporal compose via plain async/await:

- **Tools**: `@ai.tool` with `execute_activity()` in the body — each tool
  is its own activity with a matching signature, no dispatch table needed
- **LLM**: `buffered_model()` wraps an activity call into a `LanguageModel`;
  the activity uses `llm.buffer()` and `ToolSchema` (no dummy `fn`, no drain loop)
- **Loop**: `ai.stream_loop()` runs the agent loop unchanged
- **Bus**: `ai.run()` provides the unified message bus for streaming

The agent function is identical to the non-Temporal version.
Temporal doesn't know about the framework; the framework doesn't know
about Temporal.

**3 files:** `activities.py` (I/O), `workflow.py` (agent + wrappers), `main.py`

## `raw/` — No framework

The same agent as plain Python + Temporal + anthropic SDK. No framework.
The entire agent loop is ~30 lines of dict manipulation.

**3 files:** `activities.py` (tools + I/O), `workflow.py` (loop), `main.py`

## Setup

```bash
# 1. Install & start Temporal
brew install temporal
temporal server start-dev

# 2. Install deps
cd examples/temporal-durable
uv sync

# 3. Set API key (both examples use AI Gateway)
export AI_GATEWAY_API_KEY=...

# 4. Run
uv run python with_sdk/main.py
uv run python raw/main.py
```

## How it works

```
Workflow (deterministic)              Activities (real I/O)
┌─────────────────────────┐          ┌──────────────────────┐
│ while True:             │          │                      │
│   response = activity───┼─────────>│  llm_call(messages)  │
│                         │<─────────┼  → Anthropic API     │
│   if no tool_calls:     │          │                      │
│     return text         │          │                      │
│                         │          │                      │
│   gather(               │          │                      │
│     activity(tool1) ────┼─────────>│  get_weather(city)   │
│     activity(tool2) ────┼─────────>│  get_population(city)│
│   )                     │<─────────┼  → plain functions   │
└─────────────────────────┘          └──────────────────────┘
```

On crash/restart, Temporal replays activity results from its event history.
The workflow re-executes deterministically — each `execute_activity()` call
returns the cached result instead of re-running the I/O.

## Framework primitives used in `with_sdk/`

| Pattern | What it replaces |
|---|---|
| `@ai.tool` with activity body | Manual `TOOL_SCHEMAS` dicts + `TOOL_FNS` dispatch table |
| `ai.ToolSchema` | `Tool(fn=_noop)` hacks for passing schemas across boundaries |
| `llm.buffer(messages, tools)` | Manual stream drain loop |
| `ai.stream_loop(llm, msgs, tools)` | Manual while/gather/append loop |
| `ai.make_messages(system=..., user=...)` | Raw dict construction |
| `ai.Message` (typed, Pydantic) | `dict[str, Any]` blobs |
