# vercel-ai-sdk

A Python version of the [AI SDK](https://ai-sdk.dev/).

## Quick Start

```bash
uv add vercel-ai-sdk
```

```python
import vercel_ai_sdk as ai

llm = ai.openai.OpenAIModel(model="gpt-4", api_key="...")

async for msg in ai.execute(my_agent, llm, "Hello"):
    print(msg.text)
```

## API Reference

### `@ai.tool`

Decorator that turns an async function into a tool. Parameters are auto-extracted from type hints and docstrings become the tool description.

```python
@ai.tool
async def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city."""
    return f"72°F in {city}"
```

### `ai.stream_text(llm, messages, label=None)`

Streams text from the LLM without tool support. Returns a `Stream` that can be awaited or async-iterated.

```python
result = await ai.stream_text(llm, messages)
# or iterate for real-time updates
async for msg in ai.stream_text(llm, messages):
    print(msg.text_delta, end="")
```

### `ai.stream_loop(llm, messages, tools, label=None)`

Streams LLM responses and automatically executes tool calls in a loop until complete. This is the main function for agentic workflows.

```python
result = await ai.stream_loop(llm, messages, tools=[get_weather])
```

### `ai.execute(root_fn, *args)`

Runs an agent function and yields all messages from nested streams. Use this as the top-level entry point for any agent workflow.

```python
async def my_agent(llm, query):
    return await ai.stream_loop(llm, messages, tools)

async for msg in ai.execute(my_agent, llm, "What's the weather?"):
    print(msg)
```

### `ai.Message`

Universal message type with `role` ("user", "assistant", "system") and `parts`. Access text via `msg.text`. The `label` field tags messages for multi-agent routing.

### `ai.TextPart`, `ai.ToolPart`, `ai.ReasoningPart`

Message parts. `TextPart` holds text content. `ToolPart` contains tool invocation details and results. `ReasoningPart` holds model reasoning/thinking output.

## MCP Integration

### `ai.mcp.get_http_tools(url, headers={}, tool_prefix="")`

Connects to an MCP server over HTTP and returns tools. Optional `tool_prefix` namespaces tool names.

```python
tools = await ai.mcp.get_http_tools(
    "https://mcp.example.com/mcp",
    headers={"API_KEY": "..."},
    tool_prefix="docs"
)
```

### `ai.mcp.get_stdio_tools(cmd, *args, tool_prefix="")`

Spawns an MCP server process via stdio. Useful for local MCP servers like npx packages.

```python
tools = await ai.mcp.get_stdio_tools(
    "npx", "-y", "@upstash/context7-mcp",
    "--api-key", os.environ["CONTEXT7_API_KEY"],
    tool_prefix="context7"
)
```

## Multi-Agent Example

```python
async def multiagent(llm, query):
    # Run two agents in parallel
    stream1, stream2 = await asyncio.gather(
        ai.stream_loop(llm, msgs1, tools=[add_one], label="agent1"),
        ai.stream_loop(llm, msgs2, tools=[multiply], label="agent2"),
    )

    # Combine results and summarize
    combined = stream1[-1].text + stream2[-1].text
    return await ai.stream_text(llm, make_messages(combined), label="summarizer")

async for msg in ai.execute(multiagent, llm, "10"):
    print(f"[{msg.label}] {msg.text_delta}", end="")
```