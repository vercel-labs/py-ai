"""Serverless hook pattern: interrupt_loop=True.

Demonstrates the serverless/stateless pattern where the agent run suspends
cleanly when a hook has no resolution, then re-enters with a pre-registered
resolution.

Flow:
  1. First run: hook fires, interrupt_loop=True cancels the future,
     CancelledError is caught and the run ends.
  2. Second run: resolve_hook() pre-registers the answer, agent.run()
     replays from the same input, and hook finds the resolution immediately.

TODO: The implementation *works* but requires some hacks on the user
side. We need to come up with cleaner API support.
"""

import asyncio
from collections.abc import AsyncGenerator

import pydantic

import ai


class Confirmation(pydantic.BaseModel):
    approved: bool
    reason: str = ""


FILES_DELETED = set()


@ai.tool
async def delete_file(path: str) -> str:
    """Delete a file at the given path."""
    print("FILE DELETED:", path)
    FILES_DELETED.add(path)
    return f"Deleted {path}"


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-sonnet-4")

    my_agent = ai.agent(tools=[delete_file])

    @my_agent.loop
    async def with_confirmation(
        context: ai.Context,
    ) -> AsyncGenerator[ai.AgentEvent]:
        while True:
            # HACK: If there isn't anything to do, it's because we hit
            # a hook and bailed out. Skip running the stream in that
            # case, but process its tool calls (which should have
            # resolved now).
            if context.keep_running():
                s = ai.models.stream(
                    context.model, context.messages, tools=context.tools
                )
                async for event in s:
                    yield event

                context.add(s.message)

            tool_calls = context.resolve(context.messages[-1].tool_calls)
            if not tool_calls:
                return

            results: list[ai.ToolCallResult] = []
            for tc in tool_calls:
                try:
                    confirmation = await ai.hook(
                        f"confirm_{tc.id}",
                        payload=Confirmation,
                        metadata={"tool": tc.name, "kwargs": tc.kwargs},
                        interrupt_loop=True,  # serverless: cancel if unresolved
                    )
                except asyncio.CancelledError:
                    # No resolution available — bail out cleanly.
                    return

                if confirmation.approved:
                    results.append(await tc())
                else:
                    results.append(
                        ai.tool_result(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            result=f"Rejected: {confirmation.reason}",
                            is_error=True,
                        )
                    )

            context.add(ai.tool_message(*results))

    messages = [
        ai.system_message("Delete files when asked. Always use the delete_file tool."),
        ai.user_message("Delete /tmp/old_logs.txt"),
    ]

    # -- First run: no resolution, hook interrupts --
    print("--- Run 1: hook fires, no resolution, run suspends ---")
    pending_hook_labels: list[str] = []

    async for event in my_agent.run(model, messages):
        # HACK?: When we get a complete assistant message, add it to
        # messages so it can get replayed easily.
        if isinstance(event, ai.StreamEnd):
            messages.append(event.message)

        if isinstance(event, ai.TextDelta):
            print(event.chunk, end="", flush=True)
        elif isinstance(event, ai.HookEvent) and event.hook.status == "pending":
            hook_part = event.hook
            pending_hook_labels.append(hook_part.hook_id)
            print(
                f"  Hook pending: {hook_part.hook_id} (metadata={hook_part.metadata})"
            )

    print("\n  Run interrupted; approval will be pre-registered for re-entry.\n")

    # -- Second run: pre-register resolution, replay from checkpoint --
    print("--- Run 2: pre-register approval, resume from checkpoint ---")
    for label in pending_hook_labels:
        ai.resolve_hook(label, Confirmation(approved=True, reason="user approved"))

    async for event in my_agent.run(model, messages):
        if isinstance(event, ai.TextDelta):
            print(event.chunk, end="", flush=True)
        elif isinstance(event, ai.HookEvent):
            print(f"  Hook {event.hook.status}: {event.hook.hook_id}")
    print()

    assert {"/tmp/old_logs.txt"} == FILES_DELETED, (
        f"Wrong files deleted: {FILES_DELETED}"
    )


if __name__ == "__main__":
    asyncio.run(main())
