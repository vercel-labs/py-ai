---
name: vercel-ai-sdk
description: Vercel AI SDK (Python) - current `ai` module patterns for models, agents, hooks, MCP, and structured output
---

# Vercel AI SDK (Python)

```bash
uv add vercel-ai-sdk
```

```python
import ai
```

## Core workflow

The framework is split into a few modules:

- `ai.models` — model catalog, streaming, generation, explicit clients
- `ai.agents` — loops, tools, hooks, durability, nested agents
- `ai.types` — immutable `Message` datamodel plus builders
- `ai.adapters` / `ai.telemetry` — protocol adapters and observability

The current default path is:

1. pick a model with `ai.model(provider, model_id)`
2. build messages with `ai.system_message(...)` / `ai.user_message(...)`
3. either call `ai.stream(...)` directly, or create an `ai.agent(...)`
4. stream messages from `agent.run(model, messages)`

```python
@ai.tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 72F in {city}"

model = ai.model("ai-gateway", "anthropic/claude-sonnet-4")
agent = ai.agent(tools=[get_weather])

messages = [
    ai.system_message("You are a helpful weather assistant."),
    ai.user_message("What's the weather in Tokyo?"),
]

async for msg in agent.run(model, messages):
    print(msg.text_delta, end="")
```

**`@ai.tool`** turns an async function into a `Tool`. Schema is extracted from type hints + docstring and validated with Pydantic.

**`ai.model(provider, model_id)`** looks up a `Model` from the built-in catalog. Current built-in providers include `ai-gateway`, `anthropic`, and `openai`.

**`ai.stream(model, messages, ...)`** streams model output directly and returns a `StreamResult` with `.text`, `.tool_calls`, `.output`, and `.usage` after iteration completes.

**`ai.agent(tools=[...])`** creates an `Agent`. `agent.run(model, messages)` uses the default loop: stream -> resolve tool calls -> execute them -> append tool-result messages -> repeat.

### Multi-agent

Use `asyncio.gather` with `ai.yield_from(...)` and labels to run agents in parallel while forwarding their streamed output:

```python
async def multi(model: ai.Model, query: str) -> str:
    researcher = ai.agent(tools=[t1])
    analyst = ai.agent(tools=[t2])

    r1, r2 = await asyncio.gather(
        ai.yield_from(researcher.run(model, msgs1, label="researcher")),
        ai.yield_from(analyst.run(model, msgs2, label="analyst")),
    )
    return f"{r1}\n{r2}"
```

The `label` field on messages lets the consumer distinguish which agent produced output (e.g. `msg.label == "researcher"`).

### Messages

The datamodel is immutable. Use builders instead of hand-assembling parts when possible:

```python
messages = [
    ai.system_message("Be concise."),
    ai.user_message("Describe this image:", ai.file_part(url)),
]
```

`Message` is a Pydantic model with `role`, `parts`, optional `label`, and optional `usage`. Serialize with `msg.model_dump()`, restore with `ai.Message.model_validate(data)`.

Key properties for consuming streamed output:
- `msg.text_delta` -- current text chunk (use for live streaming display)
- `msg.text` -- full accumulated text
- `msg.tool_calls` -- list of `ToolCallPart` objects on assistant messages
- `msg.output` -- validated Pydantic instance (when using `output_type`)
- `msg.get_hook_part()` -- find a hook suspension part (for human-in-the-loop)

Tool results are separate `role="tool"` messages. Use:

- `ai.tool_result(...)` to create a single `ToolResultPart`
- `ai.tool_message(...)` to build or merge tool-result messages

## Customization

### Custom loop

When the default loop doesn't fit (approval gates, conditional routing, batching), define a loop with `@agent.loop`:

```python
my_agent = ai.agent(tools=[get_weather, get_population])

@my_agent.loop
async def custom(context: ai.Context):
    while True:
        s = await ai.models.stream(
            context.model, context.messages, tools=context.tools
        )
        async for msg in s:
            yield msg

        tool_calls = context.resolve(s.tool_calls)
        if not tool_calls:
            return

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(tc()) for tc in tool_calls]

        yield ai.tool_message(*(t.result() for t in tasks))
```

The most important loop helpers are:

- `context.model`, `context.messages`, `context.tools`
- `context.resolve(s.tool_calls)` to turn tool-call parts into callable `ToolCall` objects
- `await tc()` to execute a tool call and get a tool-result message back
- `ai.tool_message(...)` to merge multiple tool-result messages into one history entry

## Hooks

Hooks are typed suspension points for human-in-the-loop. The current API is function-based:

```python
class Approval(pydantic.BaseModel):
    granted: bool
    reason: str
```

Inside agent code -- blocks until resolved:

```python
approval = await ai.hook(
    "approve_send_email",
    payload=Approval,
    metadata={"tool": "send_email"},
)
if approval.granted:
    await tc()
else:
    ...
```

From outside (API handler, iterator loop):

```python
ai.resolve_hook("approve_send_email", {"granted": True, "reason": "User approved"})
await ai.cancel_hook("approve_send_email")
```

Hook messages are emitted with `role="signal"` and contain a `HookPart`.

**Long-running mode** (`interrupt_loop=False`, default): the await blocks until `resolve_hook()` or `cancel_hook()` is called externally. Use for websocket/interactive UIs.

**Serverless mode** (`interrupt_loop=True`): unresolved hooks are cancelled, the run ends. Store the checkpoint, pre-register a resolution with `resolve_hook(...)`, then rerun with `checkpoint=...`.

Consuming hooks in the iterator:

```python
async for msg in my_agent.run(model, messages):
    if msg.role == "signal" and (hook := msg.get_hook_part()):
        answer = input(f"Approve {hook.hook_id}? [y/n] ")
        ai.resolve_hook(
            hook.hook_id,
            Approval(granted=answer == "y", reason="operator"),
        )
        continue
    print(msg.text_delta, end="")
```

### Checkpoints

`Checkpoint` records completed steps (LLM calls), tool executions, and hook resolutions. On replay, cached results are returned without re-executing.

```python
data = result.checkpoint.model_dump()  # serialize (JSON-safe dict)
checkpoint = ai.Checkpoint.model_validate(data)  # restore
result = my_agent.run(model, messages, checkpoint=checkpoint)  # replay completed work
```

Primary use case is serverless hook re-entry.

## Adapters

### Models and clients

```python
# Catalog lookup
model = ai.model("ai-gateway", "anthropic/claude-sonnet-4")
model = ai.model("anthropic", "claude-sonnet-4-20250514")
model = ai.model("openai", "gpt-5.4")

# Explicit client when you do not want provider defaults
client = ai.Client(base_url="https://custom.example.com/v1", api_key="sk-...")
stream = await ai.stream(model, messages, client=client)
```

Default client config comes from provider-specific env vars:

- `AI_GATEWAY_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

Use `ai.check_connection(model)` to verify credentials and model availability:

```python
ok = await ai.check_connection(model)
```

### AI SDK UI

For streaming to AI SDK frontend (`useChat`, etc.):

```python
from ai.ai_sdk_ui import UI_MESSAGE_STREAM_HEADERS, to_messages, to_sse_stream

messages = to_messages(request.messages)
return StreamingResponse(
    to_sse_stream(agent.run(model, messages)),
    headers=UI_MESSAGE_STREAM_HEADERS,
)
```

## Other features

### Structured output

Pass a Pydantic model as `output_type`:

```python
class Forecast(pydantic.BaseModel):
    city: str
    temperature: float

stream = await ai.stream(model, messages, output_type=Forecast)
async for msg in stream:
    ...
stream.output.city
```

### MCP

```python
tools = await ai.mcp.get_http_tools(
    "https://mcp.example.com/mcp",
    headers={...},
    tool_prefix="docs",
)
tools = await ai.mcp.get_stdio_tools(
    "npx",
    "-y",
    "@anthropic/mcp-server-filesystem",
    "/tmp",
    tool_prefix="fs",
)
```

Returns `Tool` objects usable in `ai.stream(...)` or `ai.agent(...)`.

### Telemetry

```python
ai.telemetry.enable()  # OTel-based, emits run/step/tool events
```

## Example map

- `examples/samples/` — focused API examples and copy-paste starting points
- `examples/fastapi-vite/` — FastAPI backend + React frontend with hook-driven tool approval
- `examples/multiagent-textual/` — parallel sub-agents with interactive hook resolution in a TUI
- `examples/temporal-durable/` — durability integrations
