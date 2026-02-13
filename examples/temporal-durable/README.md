# Durable Agent Execution with Temporal

An agent (weather + population tools) running as a Temporal workflow.
Every LLM call and tool call is a durable activity — the agent survives
crashes and restarts because Temporal replays activity results from its
event history.

## How it works

The framework and Temporal compose via plain async/await:

- **Tools**: `@ai.tool` with `execute_activity()` in the body — each tool
  is its own activity with a matching signature
- **LLM**: `DurableModel` wraps an activity call into a `LanguageModel`;
  the activity uses `llm.buffer()` and `ToolSchema`
- **Loop**: `ai.stream_loop()` runs the agent loop unchanged
- **Bus**: `ai.run()` provides the unified message bus for streaming

The agent function is identical to the non-Temporal version.
Temporal doesn't know about the framework; the framework doesn't know
about Temporal.

**3 files:** `activities.py` (I/O), `workflow.py` (agent + wrappers), `main.py`

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

## Setup

```bash
# 1. Install & start Temporal
brew install temporal
temporal server start-dev

# 2. Install deps
cd examples/temporal-durable
uv sync

# 3. Set API key
export AI_GATEWAY_API_KEY=...

# 4. Run
uv run python main.py
```
