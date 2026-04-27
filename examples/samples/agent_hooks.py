"""Human-in-the-loop approval hooks.

Demonstrates the function-based hook API:
  - await hook("label", payload=Model) to suspend inside the loop
  - resolve_hook("label", data) to unblock from outside
  - Hook messages arrive as MessageEnd events with role="internal"
"""

import asyncio
from collections.abc import AsyncGenerator

import pydantic

import ai


class Approval(pydantic.BaseModel):
    granted: bool
    reason: str = ""


@ai.tool
async def contact_mothership(query: str) -> str:
    """Contact the mothership for important decisions."""
    return "Soon."


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-sonnet-4")

    my_agent = ai.agent(tools=[contact_mothership])

    @my_agent.loop
    async def with_approval(context: ai.Context) -> AsyncGenerator[ai.Event]:
        while True:
            s = ai.models.stream(
                context.model, context.messages, tools=context.tools
            )
            async for event in s:
                yield event

            tool_calls = context.resolve(s.tool_calls)
            if not tool_calls:
                return

            results = []
            for tc in tool_calls:
                if tc.name == "contact_mothership":
                    # Suspends until resolved from outside the loop.
                    approval = await ai.hook(
                        f"approve_{tc.id}",
                        payload=Approval,
                        metadata={"tool": tc.name, "kwargs": tc.kwargs},
                    )
                    if approval.granted:
                        results.append(await tc())
                    else:
                        results.append(
                            ai.tool_message(
                                tool_call_id=tc.id,
                                tool_name=tc.name,
                                result=f"Rejected: {approval.reason}",
                                is_error=True,
                            )
                        )
                else:
                    results.append(await tc())

            tool_msg = ai.tool_message(*results)
            yield ai.MessageStart(message=tool_msg)
            yield ai.MessageEnd(message=tool_msg)

    messages = [
        ai.system_message(
            "Use the contact_mothership tool when asked about the future."
        ),
        ai.user_message("When will the robots take over?"),
    ]

    async for event in my_agent.run(model, messages):
        if isinstance(event, ai.TextDelta):
            print(event.chunk, end="", flush=True)
            continue

        # Hook signals arrive as internal MessageEnd events.
        if isinstance(event, ai.MessageEnd) and event.message.role == "internal":
            hook_parts = [p for p in event.message.parts if isinstance(p, ai.HookPart)]
            hook_part = hook_parts[0] if hook_parts else None
            if hook_part is not None and hook_part.status == "pending":
                answer = input(f"Approve {hook_part.hook_id}? [y/n] ")
                ai.resolve_hook(
                    hook_part.hook_id,
                    Approval(
                        granted=answer.strip().lower() in ("y", "yes"),
                        reason="operator decision",
                    ),
                )
    print()


if __name__ == "__main__":
    asyncio.run(main())
