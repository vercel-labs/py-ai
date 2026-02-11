"""Multi-agent hooks example — client.

Connects to the server, streams agent output in real time, and prompts for
hook approvals. Resolved branches resume immediately — approving one hook
does not block the other.

    Terminal 1:  python examples/run_multiagent_hooks/server.py
    Terminal 2:  python examples/run_multiagent_hooks/client.py
"""

import asyncio
import json

import websockets
from rich.console import Console

import vercel_ai_sdk as ai

HOST = "localhost"
PORT = 8765

console = Console()

COLORS = {"mothership": "cyan", "data_centers": "magenta", "summary": "green"}


# ---------------------------------------------------------------------------
# Hook prompt (runs in thread so it doesn't block the event loop)
# ---------------------------------------------------------------------------


async def prompt_and_send(
    ws: websockets.ClientConnection,
    hook_part: ai.HookPart,
) -> None:
    """Prompt the user in a thread, then send the resolution over the websocket."""
    branch = hook_part.metadata.get("branch", "unknown")
    tool = hook_part.metadata.get("tool", "?")
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None, input, f"[{branch}] Approve {tool}? (y/n): "
    )
    granted = answer.strip().lower() == "y"
    reason = "approved by operator" if granted else "denied by operator"
    await ws.send(
        json.dumps({"hook_id": hook_part.hook_id, "granted": granted, "reason": reason})
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def get_hook_part(msg: ai.Message) -> ai.HookPart | None:
    for part in msg.parts:
        if isinstance(part, ai.HookPart):
            return part
    return None


async def main():
    prompt_tasks: list[asyncio.Task[None]] = []

    async with websockets.connect(f"ws://{HOST}:{PORT}") as ws:
        async for raw in ws:
            data = json.loads(raw)

            # Server sends {"type": "done"} when the run is complete
            if data.get("type") == "done":
                break

            msg = ai.Message.model_validate(data)
            hook_part = get_hook_part(msg)
            label = msg.label or "unknown"
            color = COLORS.get(label, "white")

            if hook_part and hook_part.status == "pending":
                branch = hook_part.metadata.get("branch", "unknown")
                console.print(
                    f"[bold yellow][{branch}] Hook pending: {hook_part.hook_id}[/]"
                )
                # Fire off the prompt as a background task — doesn't block
                # the message loop, so the other branch keeps streaming.
                prompt_tasks.append(asyncio.create_task(prompt_and_send(ws, hook_part)))

            elif hook_part and hook_part.status == "resolved":
                branch = hook_part.metadata.get("branch", "unknown")
                granted = hook_part.resolution and hook_part.resolution.get("granted")
                tag = "approved" if granted else "denied"
                console.print(f"[bold green][{branch}] Hook resolved ({tag})[/]")

            else:
                # Stream text deltas in real time with a label prefix
                if msg.text_delta:
                    console.print(
                        f"[{color}][{label}][/] {msg.text_delta}",
                        end="",
                    )

                # Show tool call results on completion
                if msg.is_done:
                    for part in msg.parts:
                        match part:
                            case ai.ToolPart(
                                status="result", tool_name=name, result=result
                            ):
                                console.print(
                                    f"\n[green][{label}] {name} = {result}[/]"
                                )

    await asyncio.gather(*prompt_tasks)
    console.print("\n[bold]Done.[/]")


if __name__ == "__main__":
    asyncio.run(main())
