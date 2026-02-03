# vercel-ai-sdk

A Python version of the [AI SDK](https://ai-sdk.dev/).

## Quick Start

```bash
uv add vercel-ai-sdk
```

```python
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
    model="anthropic/claude-sonnet-4",
    base_url="https://ai-gateway.vercel.sh/v1",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
)

async for msg in ai.run(agent, llm, "When will the robots take over?"):
    print(msg.text_delta, end="")
```

## Reference

### Core Primitives

#### `ai.run(root, *args)`

Entry point. Executes an async function, yields all `Message` objects from nested streams.

```python
async for msg in ai.run(my_agent, llm, "hello"):
    print(msg.text_delta, end="")
```

#### `@ai.tool`

Decorator that turns an async function into a `Tool`. Parameters extracted from type hints, docstring becomes description.

```python
@ai.tool
async def search(query: str, limit: int = 10) -> list[str]:
    """Search the database."""
    ...
```

If a tool declares a `runtime: ai.Runtime` parameter, it's auto-injected:

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

#### `@ai.hook`

Decorator that creates a suspension point from a pydantic model. The model defines the resolution schema.

```python
@ai.hook
class Approval(pydantic.BaseModel):
    granted: bool
    reason: str

# In your agent - blocks until resolved
approval = await Approval.create(metadata={"tool": "send_email"})
if approval.granted:
    ...

# From outside (API handler, websocket, etc.)
Approval.resolve(hook_id, {"granted": True, "reason": "User approved"})
```

For serverless (raises `HookPending` if resolution not provided):

```python
approval = Approval.create_or_raise(f"approval_{tool_call_id}", resolutions=saved_resolutions)
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

#### `ToolPart.execute()`

Execute a tool call. Tools are looked up from the global registry (populated by `@ai.tool`).

```python
await asyncio.gather(*(tc.execute() for tc in result.tool_calls))
```

#### `ai.make_messages(*, system=None, user)`

Build a message list from system + user strings.

```python
messages = ai.make_messages(system="You are helpful.", user="Hello!")
```

### Adapters

#### LLM Providers

```python
# OpenAI-compatible (including Vercel AI Gateway)
llm = ai.openai.OpenAIModel(
    model="anthropic/claude-sonnet-4",
    base_url="https://ai-gateway.vercel.sh/v1",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    thinking=True,           # enable reasoning output
    budget_tokens=10000,     # or reasoning_effort="medium"
)

# Anthropic (native client)
llm = ai.anthropic.AnthropicModel(
    model="claude-sonnet-4-5-20250929",
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
| `Message` | Universal message with `role`, `parts`, `label`. Properties: `text`, `text_delta`, `reasoning_delta`, `tool_deltas`, `is_done` |
| `TextPart` | Text content with streaming `state` and `delta` |
| `ToolPart` | Tool call with `tool_call_id`, `tool_name`, `tool_args`, `status`, `result`. Has `.execute()` method |
| `ReasoningPart` | Model reasoning/thinking with optional `signature` (Anthropic) |
| `HookPart` | Hook suspension with `hook_id`, `hook_type`, `status`, `metadata`, `resolution` |
| `PartState` | Literal type: `"streaming"` or `"done"` |
| `StreamResult` | Result of a stream: `messages`, `tool_calls`, `text`, `last_message` |
| `Tool` | Tool definition: `name`, `description`, `schema`, `fn` |
| `Runtime` | Step queue with `put_message()`, `get_all_hooks()` |
| `LanguageModel` | Abstract base class for LLM providers |
| `HookPending` | Exception raised by `Hook.create_or_raise()` when resolution needed |

## Examples

See the `examples/` directory:

- `run_agent.py` - Basic agent with tools
- `run_multiagent.py` - Parallel agents with live display
- `run_hooks.py` - Human-in-the-loop approval flow
- `run_streaming_tool.py` - Tool that streams progress via Runtime
- `run_custom_loop.py` - Custom step with `@ai.stream`
- `run_mcp.py` - MCP integration
- `run_fake_serverless.py` - Suspend/resume with `HookPending`
