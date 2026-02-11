"""Multi-agent hooks example — Textual TUI client.

Rich interactive terminal UI that streams agent output into separate panels
and handles hook approvals via an input widget at the bottom.

    Terminal 1:  python examples/run_multiagent_hooks/server.py
    Terminal 2:  python examples/run_multiagent_hooks/client_tui.py
"""

import asyncio
import json

import websockets
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, Static
from textual.worker import get_current_worker

import vercel_ai_sdk as ai

HOST = "localhost"
PORT = 8765


class AgentPanel(VerticalScroll):
    """A scrolling panel for one agent's output."""

    def __init__(self, agent_id: str, title: str, color: str) -> None:
        super().__init__(id=agent_id)
        self.border_title = title
        self.styles.border = ("solid", color)
        self._content = Text()
        self._color = color

    def compose(self) -> ComposeResult:
        yield Static(id=f"{self.id}-text")

    @property
    def text_widget(self) -> Static:
        return self.query_one(f"#{self.id}-text", Static)

    def append_text(self, delta: str, style: str = "") -> None:
        self._content.append(delta, style=style or self._color)
        self.text_widget.update(self._content)
        self.scroll_end(animate=False)

    def append_event(self, text: str, style: str = "bold") -> None:
        self._content.append(f"\n{text}", style=style)
        self.text_widget.update(self._content)
        self.scroll_end(animate=False)


class HookApprovalApp(App):
    """Textual app for the multi-agent hooks example."""

    CSS = """
    #top-row {
        height: 1fr;
    }
    #top-row AgentPanel {
        width: 1fr;
    }
    #summary {
        height: auto;
        max-height: 40%;
        min-height: 5;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    Input {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._hook_queue: asyncio.Queue[ai.HookPart] = asyncio.Queue()
        self._current_hook: ai.HookPart | None = None
        self._ws: websockets.ClientConnection | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-row"):
            yield AgentPanel("mothership", "Mothership Agent", "cyan")
            yield AgentPanel("data_centers", "Data Centers Agent", "magenta")
        yield AgentPanel("summary", "Summary Agent", "green")
        yield Input(
            placeholder="Waiting for agent to request approval...",
            disabled=True,
            id="input",
        )
        yield Static("Connecting...", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.run_websocket()

    # ------------------------------------------------------------------
    # Websocket reader (background worker)
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def run_websocket(self) -> None:
        """Connect to the server and stream messages into the UI.

        Async workers run on the app's event loop (thread=False by default),
        so widget methods can be called directly.
        """
        worker = get_current_worker()
        status = self.query_one("#status-bar", Static)

        try:
            async with websockets.connect(f"ws://{HOST}:{PORT}") as ws:
                self._ws = ws
                status.update(f"Connected to ws://{HOST}:{PORT}")

                async for raw in ws:
                    if worker.is_cancelled:
                        break

                    data = json.loads(raw)

                    if data.get("type") == "done":
                        self._on_run_complete()
                        break

                    msg = ai.Message.model_validate(data)
                    self._handle_message(msg)

        except (ConnectionRefusedError, OSError) as exc:
            status.update(f"Connection failed: {exc}  --  is the server running?")

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_message(self, msg: ai.Message) -> None:
        hook_part = _get_hook_part(msg)
        label = msg.label or "unknown"

        if hook_part and hook_part.status == "pending":
            self._on_hook_pending(hook_part)
            return

        if hook_part and hook_part.status == "resolved":
            self._on_hook_resolved(hook_part)
            return

        panel = self._get_panel(label)
        if panel is None:
            return

        # Stream text deltas
        if msg.text_delta:
            panel.append_text(msg.text_delta)

        if msg.reasoning_delta:
            panel.append_text(msg.reasoning_delta, style="dim")

        for delta in msg.tool_deltas:
            panel.append_text(delta.args_delta, style="yellow")

        # Show completed tool calls
        if msg.is_done:
            for part in msg.parts:
                match part:
                    case ai.ToolPart(status="pending", tool_name=name, tool_args=args):
                        panel.append_event(f"-> {name}({args})", style="yellow")
                    case ai.ToolPart(status="result", tool_name=name, result=result):
                        panel.append_event(f"ok {name} = {result}", style="green")

    def _on_hook_pending(self, hook_part: ai.HookPart) -> None:
        branch = hook_part.metadata.get("branch", "unknown")
        tool = hook_part.metadata.get("tool", "?")

        panel = self._get_panel(branch)
        if panel:
            panel.append_event(f"APPROVAL REQUIRED: {tool}", style="bold yellow")

        self._hook_queue.put_nowait(hook_part)
        self._maybe_activate_next_hook()

    def _on_hook_resolved(self, hook_part: ai.HookPart) -> None:
        branch = hook_part.metadata.get("branch", "unknown")
        granted = hook_part.resolution and hook_part.resolution.get("granted")
        tag = "APPROVED" if granted else "DENIED"
        style = "bold green" if granted else "bold red"

        panel = self._get_panel(branch)
        if panel:
            panel.append_event(tag, style=style)

        status = self.query_one("#status-bar", Static)
        status.update(f"Hook resolved ({tag.lower()}) for {branch}")

    def _on_run_complete(self) -> None:
        status = self.query_one("#status-bar", Static)
        status.update("Run complete. Press q to quit.")

        inp = self.query_one("#input", Input)
        inp.disabled = True
        inp.placeholder = "Done."

    # ------------------------------------------------------------------
    # Hook approval input
    # ------------------------------------------------------------------

    def _maybe_activate_next_hook(self) -> None:
        """Pop the next pending hook and activate the input prompt."""
        if self._current_hook is not None:
            return

        try:
            hook = self._hook_queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        self._current_hook = hook
        branch = hook.metadata.get("branch", "unknown")
        tool = hook.metadata.get("tool", "?")

        inp = self.query_one("#input", Input)
        inp.disabled = False
        inp.placeholder = f"[{branch}] Approve {tool}? (y/n)"
        inp.focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._current_hook is None:
            event.input.clear()
            return

        answer = event.value.strip().lower()
        if answer not in ("y", "n", "yes", "no"):
            event.input.clear()
            event.input.placeholder = "Please type y or n"
            return

        granted = answer in ("y", "yes")
        reason = "approved by operator" if granted else "denied by operator"

        hook = self._current_hook
        self._current_hook = None

        event.input.clear()
        event.input.disabled = True
        event.input.placeholder = "Waiting..."

        if self._ws is not None:
            await self._ws.send(
                json.dumps(
                    {
                        "hook_id": hook.hook_id,
                        "granted": granted,
                        "reason": reason,
                    }
                )
            )

        branch = hook.metadata.get("branch", "unknown")
        tag = "approved" if granted else "denied"
        status = self.query_one("#status-bar", Static)
        status.update(f"Sent {tag} for {branch}")

        self._maybe_activate_next_hook()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_panel(self, label: str) -> AgentPanel | None:
        try:
            return self.query_one(f"#{label}", AgentPanel)
        except Exception:
            return None


def _get_hook_part(msg: ai.Message) -> ai.HookPart | None:
    for part in msg.parts:
        if isinstance(part, ai.HookPart):
            return part
    return None


if __name__ == "__main__":
    app = HookApprovalApp()
    app.run()
