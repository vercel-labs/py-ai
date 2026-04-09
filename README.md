# vercel-ai-sdk

> [!WARNING]
> This SDK is **experimental**. It is not stable and is not guaranteed to be maintained in the future. For evaluation purposes only.

Python toolkit for building model-powered apps and agent loops with the `ai` module.

## Install

```bash
uv add vercel-ai-sdk
```

```python
import ai
```

## Quick Start

```python
import asyncio

import ai


@ai.tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 72F in {city}"


async def main() -> None:
    model = ai.model("ai-gateway", "anthropic/claude-sonnet-4")
    agent = ai.agent(tools=[get_weather])

    messages = [
        ai.system_message("You are a helpful weather assistant."),
        ai.user_message("What's the weather in Tokyo?"),
    ]

    async for msg in agent.run(model, messages):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
```

## Framework Layout

- `ai.models`: provider adapters, model catalog, streaming, generation, explicit clients
- `ai.agents`: agent loops, tools, hooks, durability, nested agents
- `ai.types`: immutable `Message` data model and builder helpers
- `ai.telemetry`: OpenTelemetry-style event hooks across the stack
- `ai.adapters`: protocol adapters such as AI SDK UI streaming

The main split is intentional:

- Use `ai.models` on its own when you just need model access.
- Add `ai.agents` when you need looping, tools, hooks, or durability.

## Core Workflow

### 1. Pick a model from the catalog

```python
model = ai.model("ai-gateway", "anthropic/claude-sonnet-4")
```

Built-in providers currently include:

- `ai-gateway`
- `anthropic`
- `openai`

`ai.model(provider, model_id)` returns a `Model` metadata object. Provider defaults are used to auto-create a client from environment variables when you do not pass one explicitly.

### 2. Build messages with helpers

The data model is immutable. Prefer builder helpers over manually assembling parts:

```python
messages = [
    ai.system_message("You are concise."),
    ai.user_message("Explain why the sky is blue."),
]
```

Useful builders:

- `ai.system_message(...)`
- `ai.user_message(...)`
- `ai.assistant_message(...)`
- `ai.tool_message(...)`
- `ai.tool_result(...)`
- `ai.file_part(...)`
- `ai.thinking(...)`

### 3. Run a simple model stream

If you only need model access, use `ai.stream(...)` directly:

```python
async for msg in await ai.stream(model, messages):
    if msg.text_delta:
        print(msg.text_delta, end="", flush=True)
```

After iteration, the returned stream object exposes final state such as:

- `.text`
- `.tool_calls`
- `.usage`
- `.output`

### 4. Add an agent loop when you need tools

```python
@ai.tool
async def search(query: str) -> str:
    """Search the docs."""
    return "..."


agent = ai.agent(tools=[search])

async for msg in agent.run(model, messages):
    ...
```

The default agent loop does:

1. stream the model response
2. resolve tool calls against registered tools
3. execute tools
4. append tool-result messages
5. repeat until there are no tool calls

## Agents

### `ai.agent(...)`

Creates an `Agent` with tools and an overridable loop. The default loop is enough for common tool-calling agents.

```python
agent = ai.agent(tools=[get_weather])
async for msg in agent.run(model, messages):
    ...
```

### `@ai.tool`

Wraps an async function with:

- schema generation from type hints
- argument validation via Pydantic
- execution through the agent runtime

```python
@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"
```

Resolved tool calls become `ToolCall` objects. Calling `await tc()` executes the tool and returns a `role="tool"` message.

### Custom loops with `@agent.loop`

Use a custom loop when you need approval gates, special routing, or custom tool orchestration.

```python
@agent.loop
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

        results = [await tc() for tc in tool_calls]
        yield ai.tool_message(*results)
```

`ai.tool_message(...)` can build a single tool-result message from:

- one or more tool-result messages
- one or more `ToolResultPart` values
- keyword fields such as `tool_call_id=...`, `result=...`

### Hooks

Hooks are function-based suspension points:

```python
import pydantic


class Approval(pydantic.BaseModel):
    granted: bool
    reason: str = ""


approval = await ai.hook(
    "approve_send_email",
    payload=Approval,
    metadata={"tool": "send_email"},
)
```

Resolve from outside the loop:

```python
ai.resolve_hook("approve_send_email", Approval(granted=True, reason="operator"))
```

Hook messages are emitted with `role="signal"` and carry a `HookPart`.

For serverless replay flows, pass `interrupt_loop=True`, store the checkpoint from the run, pre-resolve the hook, and re-enter with the checkpoint.

### Nested agents and streaming tools

Async-generator tools can stream intermediate messages through the runtime and return their final text as the tool result. For nested agents and sub-agent fan-out, use `ai.yield_from(...)` to forward streamed output while collecting the final text.

See:

- `examples/samples/streaming_tool.py`
- `examples/samples/agent_nested.py`
- `examples/multiagent-textual/`

## Models

### Streaming

```python
stream = await ai.stream(model, messages, tools=[get_weather])
async for msg in stream:
    ...
print(stream.text)
```

### Structured output

```python
class Forecast(pydantic.BaseModel):
    city: str
    temperature_c: float


stream = await ai.stream(model, messages, output_type=Forecast)
async for msg in stream:
    ...

forecast = stream.output
```

### Image and video generation

Use `ai.generate(...)` with a model that supports image or video generation:

```python
image_model = ai.model("ai-gateway", "google/imagen-4.0-generate-001")
result = await ai.generate(image_model, [ai.user_message("A sunset over Tokyo")])
```

See:

- `examples/samples/image_generation.py`
- `examples/samples/image_edit.py`
- `examples/samples/video_generation.py`

### Explicit clients

Auto-client creation uses provider defaults and env vars:

- `AI_GATEWAY_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

Pass an explicit client when you need a custom base URL or key:

```python
client = ai.Client(base_url="https://custom.example.com/v1", api_key="sk-...")
stream = await ai.stream(model, messages, client=client)
```

### Connection checks

Use `ai.check_connection(model)` to verify credentials and model availability before running a workflow:

```python
ok = await ai.check_connection(ai.model("openai", "gpt-5.4-mini"))
```

See `examples/samples/check_connection.py`.

## MCP

MCP servers can be exposed as tools:

```python
tools = await ai.mcp.get_http_tools(
    "https://mcp.context7.com/mcp",
    headers={"CONTEXT7_API_KEY": "..."},
    tool_prefix="context7",
)
```

See `examples/samples/mcp_tools.py`.

## Telemetry

```python
ai.telemetry.enable()
ai.telemetry.disable()
```

Events include run, step, and tool-call lifecycle events. The telemetry layer is shared across models and agents.

## Examples

### Samples

Small, focused examples live in `examples/samples/`:

- `agent_simple.py`
- `agent_custom_loop.py`
- `agent_hooks.py`
- `agent_hooks_serverless.py`
- `agent_nested.py`
- `stream.py`
- `structured_output.py`
- `streaming_tool.py`
- `check_connection.py`
- `mcp_tools.py`
- `image_generation.py`
- `image_edit.py`
- `video_generation.py`

### End-to-end demos

- `examples/fastapi-vite/`: FastAPI backend + Vite frontend with hook-based tool approval
- `examples/multiagent-textual/`: Textual TUI with parallel agents and interactive hook resolution
- `examples/temporal-durable/`: durable execution patterns with Temporal

## Reference Surface

Top-level exports are organized around the reworked framework shape:

- model layer: `ai.model`, `ai.stream`, `ai.generate`, `ai.check_connection`, `ai.Client`
- agent layer: `ai.agent`, `ai.tool`, `ai.hook`, `ai.resolve_hook`, `ai.cancel_hook`, `ai.yield_from`
- types/builders: `ai.Message`, `ai.FilePart`, `ai.TextPart`, `ai.system_message`, `ai.user_message`, `ai.tool_message`, `ai.tool_result`
- integrations: `ai.mcp`, `ai.ai_sdk_ui`, `ai.telemetry`
