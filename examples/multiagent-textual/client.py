"""Multi-agent hooks demo — Textual TUI client.

Connects to the FastAPI server over WebSocket and renders three agent
panels with inline hook approval.  Monochrome styling with a dim
yellow accent for hook events.

    uv run python client.py
"""

from __future__ import annotations

import asyncio
import json

import rich.text
import pydantic
import textual
import textual.app
import textual.containers
import textual.widgets
import textual.worker
import websockets

import ai

WS_URL = "ws://localhost:8000/ws"

# ---------------------------------------------------------------------------
# Agent panel
# ---------------------------------------------------------------------------


class AgentPanel(textual.containers.VerticalScroll):
    """Scrolling panel for one agent's output stream."""

    DEFAULT_CSS = """
    AgentPanel {
        height: 1fr;
        border: solid $surface-lighten-2;
        padding: 0 1;
    }
    AgentPanel Static {
        width: 1fr;
    }
    """

    def __init__(self, agent_id: str, title: str) -> None:
        super().__init__(id=agent_id)
        self._title = title
        self._status = "idle"
        self._content = rich.text.Text()
        self._update_border_title()

    def compose(self) -> textual.app.ComposeResult:
        yield textual.widgets.Static(id=f"{self.id}-text")

    @property
    def text_widget(self) -> textual.widgets.Static:
        return self.query_one(f"#{self.id}-text", textual.widgets.Static)

    # -- status management -------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        self._status = value
        self._update_border_title()

    def _update_border_title(self) -> None:
        self.border_title = f" {self._title} "
        self.border_subtitle = f" {self._status} "

    # -- content helpers ---------------------------------------------------

    def append_text(self, delta: str, style: str = "") -> None:
        self._content.append(delta, style=style)
        self.text_widget.update(self._content)
        self.scroll_end(animate=False)

    def append_line(self, text: str, style: str = "dim") -> None:
        self._content.append(f"\n{text}", style=style)
        self.text_widget.update(self._content)
        self.scroll_end(animate=False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class MultiAgentApp(textual.app.App):
    """Textual app for the multi-agent hooks demo."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #input-bar {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }
    """

    BINDINGS = [("q", "quit", "quit")]

    def __init__(self) -> None:
        super().__init__()
        self._hook_queue: asyncio.Queue[ai.HookPart] = asyncio.Queue()
        self._current_hook: ai.HookPart | None = None
        self._ws: websockets.ClientConnection | None = None
        self._event_adapter = pydantic.TypeAdapter(ai.Event)
        self._current_label = "unknown"

    def compose(self) -> textual.app.ComposeResult:
        yield AgentPanel("mothership", "mothership")
        yield AgentPanel("data_centers", "data-centers")
        yield AgentPanel("summary", "summary")
        yield textual.widgets.Input(
            placeholder="waiting for agents...",
            disabled=True,
            id="input-bar",
        )

    def on_mount(self) -> None:
        self.run_websocket()

    # ------------------------------------------------------------------
    # WebSocket reader (background worker)
    # ------------------------------------------------------------------

    @textual.work(exclusive=True)
    async def run_websocket(self) -> None:
        worker = textual.worker.get_current_worker()

        try:
            async with websockets.connect(WS_URL) as ws:
                self._ws = ws
                self._set_input_placeholder("connected — waiting for agents...")

                async for raw in ws:
                    if worker.is_cancelled:
                        break

                    data = json.loads(raw)
                    if data.get("type") == "done":
                        self._on_run_complete()
                        break

                    event = self._event_adapter.validate_python(data)
                    self._handle_event(event)

        except (ConnectionRefusedError, OSError) as exc:
            self._set_input_placeholder(f"connection failed: {exc}")

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    def _handle_event(self, event: ai.Event) -> None:
        if isinstance(event, ai.MessageStart) and event.message is not None:
            self._current_label = event.message.source_label or "unknown"
            panel = self._get_panel(self._current_label)
            if panel is not None and panel.status == "idle":
                panel.status = "streaming..."
            return

        if isinstance(event, ai.TextDelta):
            panel = self._get_panel(self._current_label)
            if panel is not None:
                panel.append_text(event.chunk)
            return

        if isinstance(event, ai.ReasoningDelta | ai.ToolDelta):
            panel = self._get_panel(self._current_label)
            if panel is not None:
                panel.append_text(event.chunk, style="dim")
            return

        if not isinstance(event, ai.MessageEnd):
            return

        msg = event.message
        label = msg.source_label or self._current_label

        hook_parts = [p for p in msg.parts if isinstance(p, ai.HookPart)]
        if hook_parts:
            hook_part = hook_parts[0]
            if hook_part.status == "pending":
                self._on_hook_pending(hook_part)
                return
            if hook_part.status == "resolved":
                self._on_hook_resolved(hook_part)
                return

        panel = self._get_panel(label)
        if panel is None:
            return

        for part in msg.parts:
            match part:
                case ai.ToolCallPart(tool_name=name, tool_args=args):
                    panel.append_line(f"> {name}({args})")
                case ai.ToolResultPart(tool_name=name, result=result):
                    panel.append_line(f"< {name} = {result}")

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def _on_hook_pending(self, hook_part: ai.HookPart) -> None:
        branch = hook_part.metadata.get("branch", "unknown")
        tool = hook_part.metadata.get("tool", "?")

        panel = self._get_panel(branch)
        if panel:
            panel.append_line(f"!! approval required: {tool}", style="dim yellow")
            panel.status = "awaiting approval"

        self._hook_queue.put_nowait(hook_part)
        self._maybe_activate_next_hook()

    def _on_hook_resolved(self, hook_part: ai.HookPart) -> None:
        branch = hook_part.metadata.get("branch", "unknown")
        granted = hook_part.resolution and hook_part.resolution.get("granted")
        tag = "approved" if granted else "denied"
        style = "dim green" if granted else "dim red"

        panel = self._get_panel(branch)
        if panel:
            panel.append_line(f">> {tag}", style=style)
            panel.status = "streaming..."

    def _on_run_complete(self) -> None:
        for name in ("mothership", "data_centers", "summary"):
            panel = self._get_panel(name)
            if panel:
                panel.status = "complete"

        inp = self.query_one("#input-bar", textual.widgets.Input)
        inp.disabled = True
        inp.placeholder = "done — press q to quit"

    # ------------------------------------------------------------------
    # Hook approval input
    # ------------------------------------------------------------------

    def _maybe_activate_next_hook(self) -> None:
        if self._current_hook is not None:
            return

        try:
            hook = self._hook_queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        self._current_hook = hook
        branch = hook.metadata.get("branch", "unknown")
        tool = hook.metadata.get("tool", "?")

        inp = self.query_one("#input-bar", textual.widgets.Input)
        inp.disabled = False
        inp.placeholder = f"approve {branch}/{tool}? [y/n]"
        inp.focus()

    async def on_input_submitted(self, event: textual.widgets.Input.Submitted) -> None:
        if self._current_hook is None:
            event.input.clear()
            return

        answer = event.value.strip().lower()
        if answer not in ("y", "n", "yes", "no"):
            event.input.clear()
            event.input.placeholder = "type y or n"
            return

        granted = answer in ("y", "yes")
        reason = "approved by operator" if granted else "denied by operator"

        hook = self._current_hook
        self._current_hook = None

        event.input.clear()
        event.input.disabled = True
        event.input.placeholder = "waiting..."

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

        self._maybe_activate_next_hook()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_panel(self, label: str) -> AgentPanel | None:
        try:
            return self.query_one(f"#{label}", AgentPanel)
        except Exception:
            return None

    def _set_input_placeholder(self, text: str) -> None:
        inp = self.query_one("#input-bar", textual.widgets.Input)
        inp.placeholder = text


if __name__ == "__main__":
    app = MultiAgentApp()
    app.run()
