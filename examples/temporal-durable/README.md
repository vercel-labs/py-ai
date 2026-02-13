# Durable Agent Execution with Temporal

Two implementations of the same agent (weather + population tools) as a
Temporal workflow. Both survive crashes and restarts — every LLM call and
tool call is a durable activity that Temporal replays from history.

## `with_sdk/` — Using vercel-ai-sdk

Same agent loop as `examples/samples/custom_loop.py`, but with:
- `TemporalLanguageModel` — wraps `llm.stream()` in an activity
- Tool calls routed through activities via `execute_tool_via_activity()`
- `ai.run()`, `ai.stream_step()`, `ai.make_messages()` all work unchanged

**3 files:** `activities.py` (tools + I/O), `workflow.py` (loop + wrappers), `main.py`

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
│     activity(tool1) ────┼─────────>│  tool_call(name,args)│
│     activity(tool2) ────┼─────────>│  tool_call(name,args)│
│   )                     │<─────────┼  → plain functions   │
└─────────────────────────┘          └──────────────────────┘
```

On crash/restart, Temporal replays activity results from its event history.
The workflow re-executes deterministically — each `execute_activity()` call
returns the cached result instead of re-running the I/O.
