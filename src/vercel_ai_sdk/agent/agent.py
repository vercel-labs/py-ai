from __future__ import annotations

import asyncio
import dataclasses
import traceback

import pydantic

import vercel_ai_sdk as ai
from .tools import _filesystem, BUILTIN_TOOLS
from . import proto


@ai.hook
class ToolApproval(pydantic.BaseModel):
    granted: bool
    reason: str | None = None


@dataclasses.dataclass
class Agent:
    """
    Agent that runs an LLM loop with filesystem tools.

    Args:
        model: Language model to use.
        filesystem: Filesystem binding (local, sandbox, etc.).
        system: System prompt prepended to every run.
        tools: Extra tools to merge with the built-in filesystem tools.
        needs_approval: Which tools require human approval before execution.
            - True: all tools need approval
            - set of tool names: only those tools need approval
            - async callable(tool_name, tool_args) -> bool: dynamic check
    """

    model: ai.LanguageModel
    filesystem: proto.Filesystem
    system: str = ""
    tools: list[ai.Tool] = dataclasses.field(default_factory=list)

    async def _execute_tool(
        self, tc: ai.ToolPart, message: ai.Message | None = None
    ) -> None:
        """Execute a single tool call with approval check and error handling."""
        try:
            # TODO this should be tucked away into the framework
            # and done using Pydantic
            approval = await ToolApproval.create(
                f"approve_{tc.tool_call_id}",
                metadata={"tool_name": tc.tool_name, "tool_args": tc.tool_args},
            )

            if approval.granted:
                await ai.execute_tool(tc, message=message)
            else:
                tc.set_result({"error": "Tool call was denied by the user."})
                return

        except Exception as exc:
            tc.set_result(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    async def _loop(
        self,
        messages: list[ai.Message],
        tools: list[ai.Tool],
        label: str | None = None,
    ) -> ai.StreamResult:
        local_messages = list(messages)

        while True:
            result = await ai.stream_step(
                self.model, local_messages, tools, label=label
            )

            if not result.tool_calls:
                return result

            last_msg = result.last_message
            local_messages.append(last_msg)

            await asyncio.gather(
                *(self._execute_tool(tc, message=last_msg) for tc in result.tool_calls)
            )

    def run(
        self,
        messages: list[ai.Message],
        *,
        label: str | None = None,
        checkpoint: ai.Checkpoint | None = None,
    ) -> ai.RunResult:
        """
        Run the agent on the given messages.

        Returns a RunResult — async-iterate for messages, then check
        .checkpoint and .pending_hooks.
        """

        async def _root() -> None:
            fs_token = _filesystem.set(self.filesystem)
            try:
                all_tools = BUILTIN_TOOLS + self.tools

                system_messages: list[ai.Message] = []
                if self.system:
                    system_messages = ai.make_messages(system=self.system, user="")
                    # make_messages always adds a user msg — we only want system
                    system_messages = [m for m in system_messages if m.role == "system"]

                await self._loop(
                    messages=system_messages + messages,
                    tools=all_tools,
                    label=label,
                )
            finally:
                _filesystem.reset(fs_token)

        return ai.run(_root, checkpoint=checkpoint)
