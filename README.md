# vercel-ai-sdk

A Python version of the [AI SDK](https://ai-sdk.dev/).

## Quick Start

```bash
uv add vercel-ai-sdk
```

```python
import os
import vercel_ai_sdk as ai

@ai.tool
async def talk_to_mothership(question: str) -> str:
    """Contact the mothership for important decisions."""
    return "Soon."

async def agent(llm, query):
    return await ai.stream_loop(
        llm,
        messages=ai.make_messages(
            system="You are a robot assistant.",
            user=query,
        ),
        tools=[talk_to_mothership],
    )

llm = ai.openai.OpenAIModel(
    model="anthropic/claude-opus-4.6",
    base_url="https://ai-gateway.vercel.sh/v1",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
)

async for msg in ai.run(agent, llm, "When will the robots take over?"):
    print(msg.text_delta, end="")
```

## Reference

### Core Primitives

#### `ai.run(root, *args, checkpoint=None, cancel_on_hooks=False)`

Entry point. Starts `root` as a background task, processes the step/hook queue, yields `Message` objects. Returns a `RunResult`.

```python
result = ai.run(my_agent, llm, "hello")
async for msg in result:
    print(msg.text_delta, end="")

result.checkpoint      # Checkpoint with all completed work
result.pending_hooks   # dict of unresolved hooks (empty if run completed)
```

If `root` declares a `runtime: ai.Runtime` parameter, it's auto-injected.

#### `@ai.tool`

Decorator that turns an async function into a `Tool`. Parameters extracted from type hints, docstring becomes description.

```python
@ai.tool
async def search(query: str, limit: int = 10) -> list[str]:
    """Search the database."""
    ...
```

If a tool declares a `runtime: ai.Runtime` parameter, it's auto-injected (not passed by the LLM):

```python
@ai.tool
async def long_task(input: str, runtime: ai.Runtime) -> str:
    """Runtime is auto-injected, not passed by LLM."""
    await runtime.put_message(ai.Message(...))  # stream intermediate results
    ...
```

#### `@ai.stream`

Decorator that wires an async generator into the `Runtime`. Use this to make any streaming operation (like an LLM call) work with `ai.run()`.

```python
@ai.stream
async def my_custom_step(llm, messages):
    async for msg in llm.stream(messages):
        yield msg

result = await my_custom_step(llm, messages)  # returns StreamResult
```

Must be called within `ai.run()` (needs a Runtime context).

#### `@ai.hook`

Decorator that creates a suspension point from a pydantic model. The model defines the resolution schema.

```python
@ai.hook
class Approval(pydantic.BaseModel):
    granted: bool
    reason: str
```

Inside your agent — blocks until resolved:

```python
approval = await Approval.create("approve_send_email", metadata={"tool": "send_email"})
if approval.granted:
    ...
```

From outside (API handler, websocket, iterator loop, etc.):

```python
Approval.resolve("approve_send_email", {"granted": True, "reason": "User approved"})
Approval.cancel("approve_send_email")  # or cancel it
```

**Long-running mode** (`cancel_on_hooks=False`, the default): the `await` in `create()` blocks until `resolve()` or `cancel()` is called from external code.

**Serverless mode** (`cancel_on_hooks=True`): if no resolution is available, the hook's future is cancelled and the branch dies. Inspect `result.pending_hooks` and `result.checkpoint` to resume later:

```python
result = ai.run(my_agent, llm, query, cancel_on_hooks=True)
async for msg in result:
    ...

if result.pending_hooks:
    # Save result.checkpoint, collect resolutions, then re-enter:
    Approval.resolve("approve_send_email", {"granted": True, "reason": "User approved"})
    result = ai.run(my_agent, llm, query, checkpoint=result.checkpoint)
    async for msg in result:
        ...
```

### Convenience Functions

#### `ai.stream_step(llm, messages, tools=None, label=None)`

Single LLM call. Built on `@ai.stream`. Returns `StreamResult`.

```python
result = await ai.stream_step(llm, messages, tools=[search])
# result.text, result.tool_calls, result.last_message
```

#### `ai.stream_loop(llm, messages, tools, label=None)`

Full agent loop: calls LLM, executes tools, repeats until no more tool calls. Returns final `StreamResult`.

```python
result = await ai.stream_loop(llm, messages, tools=[search, get_weather])
```

#### `ai.execute_tool(tool_call, message=None)`

Execute a single tool call. Looks up the tool from the global registry (populated by `@ai.tool`). Updates the `ToolPart` with the result. If `message` is provided, emits it to the Runtime queue so the UI sees the status change.

```python
await asyncio.gather(*(ai.execute_tool(tc, message=last_msg) for tc in result.tool_calls))
```

Supports checkpoint replay — returns the cached result without re-executing if one exists.

#### `ai.make_messages(*, system=None, user)`

Build a message list from system + user strings.

```python
messages = ai.make_messages(system="You are helpful.", user="Hello!")
```

#### `ai.get_checkpoint()`

Get the current `Checkpoint` from the active Runtime context. Call this from within `ai.run()`.

```python
checkpoint = ai.get_checkpoint()
```

### Checkpoints

`Checkpoint` records completed work (LLM steps, tool executions, hook resolutions) so a run can be replayed without re-executing already-finished operations.

```python
# After a run completes or suspends
checkpoint = result.checkpoint
data = checkpoint.serialize()   # dict, JSON-safe

# Later: restore and resume
checkpoint = ai.Checkpoint.deserialize(data)
result = ai.run(my_agent, llm, query, checkpoint=checkpoint)
```

Three event types are tracked:
- **Steps** — LLM call results (replayed without calling the model)
- **Tools** — tool execution results (replayed without re-executing)
- **Hooks** — hook resolutions (replayed without re-suspending)

### Adapters

#### LLM Providers

```python
# OpenAI-compatible (including Vercel AI Gateway)
llm = ai.openai.OpenAIModel(
    model="anthropic/claude-opus-4.6",
    base_url="https://ai-gateway.vercel.sh/v1",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    thinking=True,           # enable reasoning output
    budget_tokens=10000,     # or reasoning_effort="medium"
)

# Anthropic (native client)
llm = ai.anthropic.AnthropicModel(
    model="claude-opus-4.6-20250916",
    thinking=True,
    budget_tokens=10000,
)
```

#### MCP

```python
# HTTP transport
tools = await ai.mcp.get_http_tools(
    "https://mcp.example.com/mcp",
    headers={"Authorization": "Bearer ..."},
    tool_prefix="docs",
)

# Stdio transport (subprocess)
tools = await ai.mcp.get_stdio_tools(
    "npx", "-y", "@anthropic/mcp-server-filesystem", "/tmp",
    tool_prefix="fs",
)
```

MCP connections are pooled per `ai.run()` and cleaned up automatically.

#### AI SDK UI

For streaming to AI SDK frontend (`useChat`, etc.):

```python
from vercel_ai_sdk.ai_sdk_ui import to_sse_stream, to_messages, UI_MESSAGE_STREAM_HEADERS

# Convert incoming UI messages
messages = to_messages(request.messages)

# Stream response as SSE
async def stream_response():
    async for chunk in to_sse_stream(ai.run(agent, llm, query)):
        yield chunk

return StreamingResponse(stream_response(), headers=UI_MESSAGE_STREAM_HEADERS)
```

### Types

| Type | Description |
|------|-------------|
| `Message` | Universal message with `role`, `parts`, `label`. Properties: `text`, `text_delta`, `reasoning_delta`, `tool_deltas`, `tool_calls`, `is_done` |
| `TextPart` | Text content with streaming `state` and `delta` |
| `ToolPart` | Tool call with `tool_call_id`, `tool_name`, `tool_args`, `status`, `result`. Has `.set_result()` |
| `ToolDelta` | Tool argument streaming delta (`tool_call_id`, `tool_name`, `args_delta`) |
| `ReasoningPart` | Model reasoning/thinking with optional `signature` (Anthropic) |
| `HookPart` | Hook suspension with `hook_id`, `hook_type`, `status` (`pending`/`resolved`/`cancelled`), `metadata`, `resolution` |
| `Part` | Union: `TextPart \| ToolPart \| ReasoningPart \| HookPart` |
| `PartState` | Literal: `"streaming"` \| `"done"` |
| `StreamResult` | Result of a stream step: `messages`, `tool_calls`, `text`, `last_message` |
| `Tool` | Tool definition: `name`, `description`, `schema`, `fn` |
| `ToolSchema` | Serializable tool description: `name`, `description`, `tool_schema` (no `fn`) |
| `Runtime` | Central coordinator for the agent loop. Step queue, message queue, checkpoint replay/record |
| `RunResult` | Return type of `run()`. Async-iterable for messages, then `.checkpoint` and `.pending_hooks` |
| `HookInfo` | Pending hook info: `label`, `hook_type`, `metadata` |
| `Hook` | Generic hook base with `.create()`, `.resolve()`, `.cancel()` class methods |
| `Checkpoint` | Serializable snapshot of completed work: `steps[]`, `tools[]`, `hooks[]`. Has `.serialize()` / `.deserialize()` |
| `LanguageModel` | Abstract base class for LLM providers |

## Examples

See the `examples/` directory:

**Samples** (`examples/samples/`):

- `simple.py` — Basic agent with tools and `stream_loop`
- `agent.py` — Coding agent with local filesystem tools
- `hooks.py` — Human-in-the-loop approval flow
- `streaming_tool.py` — Tool that streams progress via Runtime
- `multiagent.py` — Parallel agents with labels, then summarization
- `custom_loop.py` — Custom step with `@ai.stream`
- `mcp.py` — MCP integration (Context7)

**Projects**:

- `examples/fastapi-vite/` — Full-stack chat app (FastAPI + Vite + AI SDK UI)
- `examples/temporal-durable/` — Durable execution with Temporal workflows
- `examples/multiagent-textual/` — Multi-agent TUI with Textual
