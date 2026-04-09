---
name: ai
description: Python `ai` module — models, agents, hooks, MCP, structured output
---

# ai

```bash
uv add vercel-ai-sdk
```

```python
import ai
```

## Quick reference

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

`ai.model(provider, model_id)` — built-in providers: `ai-gateway`, `anthropic`, `openai`. Default clients auto-created from env vars (`AI_GATEWAY_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). Pass `ai.Client(base_url=, api_key=)` for custom endpoints.

`ai.stream(model, messages, ...)` — streaming without an agent loop. Returns `StreamResult` with `.text`, `.tool_calls`, `.output`, `.usage` after iteration.

`ai.generate(model, messages)` — non-streaming / image / video generation.

`ai.check_connection(model)` — verify credentials and model availability.

## Messages

Immutable Pydantic models. Use builders:

```python
ai.system_message("Be concise.")
ai.user_message("Describe this image:", ai.file_part(url))
ai.assistant_message(...)
ai.tool_message(...)       # merge one or more tool-result messages/parts
ai.tool_result(...)        # single ToolResultPart
ai.thinking(...)
```

Key properties on streamed messages:

- `msg.text_delta` — current text chunk (for live display)
- `msg.text` — full accumulated text
- `msg.tool_calls` — list of `ToolCallPart` on assistant messages
- `msg.output` — validated Pydantic instance (when using `output_type`)
- `msg.get_hook_part()` — find a hook suspension part
- `msg.label` — which agent produced the message (for multi-agent)

Serialize: `msg.model_dump()`. Restore: `ai.Message.model_validate(data)`.

## Custom agent loops

Override the default loop when you need approval gates, routing, or batching:

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

Loop helpers: `context.model`, `context.messages`, `context.tools`, `context.resolve(s.tool_calls)`. `await tc()` executes a tool call and returns a tool-result message.

## Multi-agent

Use `asyncio.gather` with `ai.yield_from(...)` and labels to run agents in parallel:

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

`msg.label` lets the consumer distinguish which agent produced output.

## Hooks

Typed suspension points for human-in-the-loop:

```python
class Approval(pydantic.BaseModel):
    granted: bool
    reason: str
```

Inside agent code (blocks until resolved):

```python
approval = await ai.hook(
    "approve_send_email",
    payload=Approval,
    metadata={"tool": "send_email"},
)
if approval.granted:
    await tc()
```

From outside:

```python
ai.resolve_hook("approve_send_email", {"granted": True, "reason": "User approved"})
await ai.cancel_hook("approve_send_email")
```

Hook messages have `role="signal"` with a `HookPart`.

**Long-running mode** (`interrupt_loop=False`, default): await blocks until resolved. Use for websocket/interactive UIs.

**Serverless mode** (`interrupt_loop=True`): unresolved hooks cancel the run. Store the checkpoint, pre-register a resolution, rerun with `checkpoint=`.

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

## Checkpoints

`Checkpoint` records completed LLM calls, tool executions, and hook resolutions. On replay, cached results are returned without re-executing.

```python
data = result.checkpoint.model_dump()
checkpoint = ai.Checkpoint.model_validate(data)
result = my_agent.run(model, messages, checkpoint=checkpoint)
```

Primary use case: serverless hook re-entry.

## Structured output

```python
class Forecast(pydantic.BaseModel):
    city: str
    temperature: float

stream = await ai.stream(model, messages, output_type=Forecast)
async for msg in stream:
    ...
stream.output.city
```

## MCP

```python
tools = await ai.mcp.get_http_tools(
    "https://mcp.example.com/mcp",
    headers={...},
    tool_prefix="docs",
)
tools = await ai.mcp.get_stdio_tools(
    "npx", "-y", "@anthropic/mcp-server-filesystem", "/tmp",
    tool_prefix="fs",
)
```

Returns `Tool` objects usable in `ai.stream(...)` or `ai.agent(...)`.

## AI SDK UI adapter

```python
from ai.ai_sdk_ui import UI_MESSAGE_STREAM_HEADERS, to_messages, to_sse_stream

messages = to_messages(request.messages)
return StreamingResponse(
    to_sse_stream(agent.run(model, messages)),
    headers=UI_MESSAGE_STREAM_HEADERS,
)
```

## Telemetry

```python
ai.telemetry.enable()   # OTel-based, emits run/step/tool events
ai.telemetry.disable()
```
