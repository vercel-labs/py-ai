"""
Example: Human-in-the-loop approval flow using hooks.

This demonstrates how to use hooks to suspend execution and wait for
external approval before executing sensitive tools.
"""

import asyncio
import os

import pydantic
import rich

import vercel_ai_sdk as ai


@ai.tool
async def contact_mothership(query: str) -> str:
    """Contact the mothership for important decisions."""
    return f"Mothership response to '{query}': The robots will take over soon."


@ai.hook
class CommunicationApproval(pydantic.BaseModel):
    """Approval required before contacting the mothership."""

    granted: bool
    reason: str


async def graph(llm: ai.LanguageModel, query: str):
    """
    Agent graph with human-in-the-loop approval.

    When a tool call requires approval, execution suspends until
    the hook is resolved from outside.
    """
    messages = ai.make_messages(
        system="You are a robot assistant. Use the contact_mothership tool when asked about the future.",
        user=query,
    )

    tools = [contact_mothership]

    while True:
        result = await ai.stream_step(llm, messages, tools)

        if not result.tool_calls:
            break

        # Process each tool call, potentially requiring approval
        for tc in result.tool_calls:
            if tc.tool_name == "contact_mothership":
                # Create hook - this emits a Message with HookPart(status="pending") and blocks
                approval = await CommunicationApproval.create(
                    metadata={
                        "tool_call_id": tc.tool_call_id,
                        "tool_name": tc.tool_name,
                        "tool_args": tc.tool_args,
                    }
                )

                if approval.granted:
                    # Execute the tool
                    await ai.execute_tool(tc, tools, result.last_message)
                else:
                    # Set rejection as tool result
                    tool_part = result.last_message.get_tool_part(tc.tool_call_id)
                    if tool_part:
                        tool_part.status = "result"
                        tool_part.result = {"error": f"Rejected: {approval.reason}"}
            else:
                # Non-sensitive tools execute directly
                await ai.execute_tool(tc, tools, result.last_message)

        messages.append(result.last_message)

    return result


def get_hook_part(msg: ai.Message) -> ai.HookPart | None:
    """Extract HookPart from a message if present."""
    for part in msg.parts:
        if isinstance(part, ai.HookPart):
            return part
    return None


async def main():
    llm = ai.openai.OpenAIModel(
        model="anthropic/claude-sonnet-4-20250514",
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=os.environ.get("AI_GATEWAY_API_KEY"),
    )

    # Run the graph and handle hook messages
    async for msg in ai.run(graph, llm, "When will the robots take over?"):
        hook_part = get_hook_part(msg)

        if hook_part:
            rich.print(
                f"[bold yellow]Hook:[/] {hook_part.status} - {hook_part.hook_type}"
            )
            rich.print(f"  ID: {hook_part.hook_id}")
            rich.print(f"  Metadata: {hook_part.metadata}")

            if hook_part.status == "pending":
                # In a real app, this would come from user input via API/websocket
                # For demo purposes, we auto-approve after a short delay
                rich.print("[bold green]Auto-approving in 1 second...[/]")
                await asyncio.sleep(1)

                # Resolve the hook
                CommunicationApproval.resolve(
                    hook_part.hook_id,
                    {"granted": True, "reason": "Auto-approved for demo"},
                )
        else:
            # Regular message
            if msg.is_done:
                # rich.print(f"[bold blue]Message:[/] {msg.text[:100]}...")
                pass
            elif msg.text_delta:
                print(msg.text_delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
