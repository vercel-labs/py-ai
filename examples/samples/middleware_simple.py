"""Middleware example: a print-based middleware that logs all execution surfaces.

Demonstrates all five middleware surfaces:
- wrap_agent_run  — the entire agent run
- wrap_model      — each LLM streaming call
- wrap_generate   — non-streaming generation (images, video, etc.)
- wrap_tool       — each tool execution
- wrap_hook       — each hook suspension point

The middleware is registered globally with ai.use() and applies to all agent
runs without any changes to the agent code itself.
"""

import asyncio
import time

import ai


class PrintMiddleware(ai.Middleware):
    """Logs every execution surface to stdout."""

    async def wrap_agent_run(self, call, next):
        label = call.label or "(default)"
        print(f">>> [run] agent starting  label={label}  tools={len(call.tools)}")
        t0 = time.perf_counter()

        async for msg in next(call):
            yield msg

        elapsed = time.perf_counter() - t0
        print(f"<<< [run] agent finished  label={label}  {elapsed:.2f}s")

    async def wrap_model(self, call, next):
        print(f"\n>>> [model] calling {call.model.id}")
        print(f"    messages: {len(call.messages)}")
        if call.tools:
            print(f"    tools: {[t.name for t in call.tools]}")

        result = await next(call)

        # The result is a StreamResult — async-iterable of Message snapshots.
        # We return it as-is; the consumer iterates it normally.
        print("<<< [model] stream started")
        return result

    async def wrap_generate(self, call, next):
        print(f"\n>>> [generate] calling {call.model.id}")
        print(f"    messages: {len(call.messages)}")

        result = await next(call)

        print("<<< [generate] done")
        return result

    async def wrap_tool(self, call, next):
        print(f"\n>>> [tool] {call.tool_name}({call.kwargs})")

        result = await next(call)

        # result is a tool-result Message.
        tr = result.tool_results[0] if result.tool_results else None
        if tr and not tr.is_error:
            print(f"<<< [tool] {call.tool_name} -> {tr.result}")
        elif tr:
            print(f"<<< [tool] {call.tool_name} ERROR: {tr.result}")
        return result

    async def wrap_hook(self, call, next):
        print(f"\n>>> [hook] {call.label}  payload={call.payload.__name__}")

        result = await next(call)

        print(f"<<< [hook] {call.label} resolved")
        return result


# Register globally — applies to all agents.
ai.use(PrintMiddleware())


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    return {"tokyo": 13_960_000, "new york": 8_336_817}.get(city.lower(), 1_000_000)


async def main() -> None:
    model = ai.model("ai-gateway", "anthropic/claude-sonnet-4")

    my_agent = ai.agent(tools=[get_weather, get_population])

    messages = [
        ai.system_message("You are a helpful assistant. Use tools when needed."),
        ai.user_message("What's the weather and population of Tokyo?"),
    ]

    print("--- starting agent run ---\n")
    async for msg in my_agent.run(model, messages):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print("\n\n--- done ---")


if __name__ == "__main__":
    asyncio.run(main())
