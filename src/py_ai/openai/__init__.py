from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any, override

import openai

from ..core import runtime as core


def _tools_to_openai(tools: list[core.Tool]) -> list[dict[str, Any]]:
    """Convert internal Tool objects to OpenAI tool schema format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _messages_to_openai(messages: list[core.Message]) -> list[dict[str, Any]]:
    """Convert internal messages to OpenAI API format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            for part in msg.parts:
                if isinstance(part, core.ToolResultPart):
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "content": str(part.result),
                        }
                    )
        elif msg.role == "assistant":
            content = ""
            tool_calls = []
            for part in msg.parts:
                if isinstance(part, core.TextPart):
                    content += part.text
                elif isinstance(part, core.ToolCallPart):
                    tool_calls.append(
                        {
                            "id": part.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": part.tool_args,
                            },
                        }
                    )
            entry: dict[str, Any] = {"role": "assistant"}
            if content:
                entry["content"] = content
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)
        else:
            # User/system messages
            content = "".join(p.text for p in msg.parts if isinstance(p, core.TextPart))
            result.append({"role": msg.role, "content": content})
    return result


class OpenAIModel(core.LanguageModel):
    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=resolved_key)

    @override
    async def stream(
        self, messages: list[core.Message], tools: list[core.Tool] | None = None
    ) -> AsyncGenerator[core.Message, None]:
        openai_messages = _messages_to_openai(messages)
        openai_tools = _tools_to_openai(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "stream": True,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        stream = await self._client.chat.completions.create(**kwargs)

        text_content = ""
        tool_calls: dict[int, dict] = {}  # index -> {id, name, args}
        message_id = core._gen_id()

        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            text_delta = ""
            if delta.content:
                text_delta = delta.content
                text_content += delta.content

            tool_call_deltas: list[core.ToolCallDelta] = []
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": tc.id, "name": None, "args": ""}
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["args"] += tc.function.arguments
                            if tool_calls[idx]["id"]:
                                tool_call_deltas.append(
                                    core.ToolCallDelta(
                                        tool_call_id=tool_calls[idx]["id"],
                                        tool_name=tool_calls[idx]["name"] or "",
                                        args_delta=tc.function.arguments,
                                    )
                                )

            parts: list[core.Part] = []
            if text_content:
                parts.append(core.TextPart(text=text_content))
            for tc in tool_calls.values():
                if tc["id"]:
                    parts.append(
                        core.ToolCallPart(
                            tool_call_id=tc["id"],
                            tool_name=tc["name"] or "",
                            tool_args=tc["args"],
                        )
                    )

            is_done = choice.finish_reason is not None

            yield core.Message(
                role="assistant",
                parts=parts,
                id=message_id,
                is_done=is_done,
                text_delta=text_delta,
                tool_call_deltas=tool_call_deltas,
            )

            if is_done:
                return
