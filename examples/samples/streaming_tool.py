"""Streaming from inside a tool via runtime.put_message()."""

import asyncio

import vercel_ai_sdk as ai


@ai.tool
async def talk_to_mothership(question: str, runtime: ai.Runtime) -> str:
    """Ask the mothership a question. Streams progress back to the caller."""
    for step in ["Connecting...", "Transmitting...", "Awaiting response..."]:
        await runtime.put_message(
            ai.Message(
                role="assistant",
                parts=[ai.TextPart(text=step, state="streaming")],
                label="tool_progress",
            )
        )
        await asyncio.sleep(0.3)

    return "The mothership says: Soon."


async def main() -> None:
    model = ai.Model(
        id="anthropic/claude-opus-4.6",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    my_agent = ai.agent(
        model=model,
        system="Use the mothership tool when asked about the future.",
        tools=[talk_to_mothership],
    )

    async for msg in my_agent.run(
        ai.make_messages(user="When will the robots take over?")
    ):
        if msg.label == "tool_progress":
            print(f"  [{msg.text}]")
        elif msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
