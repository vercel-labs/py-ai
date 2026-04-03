"""Human-in-the-loop approval hooks."""

import asyncio

import pydantic

import vercel_ai_sdk as ai


@ai.tool
async def contact_mothership(query: str) -> str:
    """Contact the mothership for important decisions."""
    return "Soon."


@ai.hook
class CommunicationApproval(pydantic.BaseModel):
    granted: bool
    reason: str


async def main() -> None:
    model = ai.Model(
        id="anthropic/claude-opus-4.6",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    my_agent = ai.agent(
        model=model,
        system="Use the contact_mothership tool when asked about the future.",
        tools=[contact_mothership],
    )

    @my_agent.loop
    async def with_approval(
        agent: ai.Agent, messages: list[ai.Message]
    ) -> ai.StreamResult:
        local_messages = list(messages)

        while True:
            result = await ai.stream_step(agent.model, local_messages, agent.tools)

            if not result.tool_calls:
                break

            for tc in result.tool_calls:
                if tc.tool_name == "contact_mothership":
                    # Blocks until resolved (long-running) or cancelled (serverless)
                    # TODO: mypy doesn't support class decorators that change the
                    # class type — @ai.hook returns type[Hook[T]] but mypy still
                    # sees the original BaseModel.
                    approval = await CommunicationApproval.create(  # type: ignore[attr-defined]
                        f"approve_{tc.tool_call_id}",
                        metadata={"tool": tc.tool_name},
                    )
                    if approval.granted:
                        await ai.execute_tool(tc, message=result.last_message)
                    else:
                        tc.set_error(f"Rejected: {approval.reason}")
                else:
                    await ai.execute_tool(tc, message=result.last_message)

            if result.last_message is not None:
                local_messages.append(result.last_message)

        return result

    async for msg in my_agent.run(
        ai.make_messages(user="When will the robots take over?")
    ):
        # Hook parts arrive as pending, waiting for resolution
        if (hook := msg.get_hook_part()) and hook.status == "pending":
            answer = input(f"Approve {hook.hook_id}? [y/n] ")
            # TODO: mypy doesn't support class decorators that change the
            # class type — @ai.hook returns type[Hook[T]] but mypy still
            # sees the original BaseModel.
            CommunicationApproval.resolve(  # type: ignore[attr-defined]
                hook.hook_id,
                {
                    "granted": answer.strip().lower() in ("y", "yes"),
                    "reason": "operator decision",
                },
            )
            continue

        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
