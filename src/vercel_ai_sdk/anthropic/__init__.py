from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

import anthropic
import pydantic

from .. import core


def _tools_to_anthropic(tools: Sequence[core.tools.ToolLike]) -> list[dict[str, Any]]:
    """Convert internal Tool objects to Anthropic tool schema format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.param_schema,
        }
        for tool in tools
    ]


def _file_part_to_anthropic(part: core.messages.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to an Anthropic content block.

    * ``image/*`` → ``{"type": "image", "source": ...}``
    * ``application/pdf`` → ``{"type": "document", "source": ...}``
    * ``text/plain`` → ``{"type": "document", "source": {"type": "text", ...}}``
    * anything else → ``ValueError``
    """
    mt = part.media_type

    if mt.startswith("image/"):
        media_type = "image/jpeg" if mt == "image/*" else mt
        if isinstance(part.data, str) and core.media.data.is_url(part.data):
            return {
                "type": "image",
                "source": {"type": "url", "url": part.data},
            }
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": core.media.data.data_to_base64(part.data),
            },
        }

    if mt == "application/pdf":
        if isinstance(part.data, str) and core.media.data.is_url(part.data):
            return {
                "type": "document",
                "source": {"type": "url", "url": part.data},
            }
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": core.media.data.data_to_base64(part.data),
            },
        }

    if mt == "text/plain":
        # Anthropic accepts text documents with source.type="text"
        if isinstance(part.data, bytes):
            text_data = part.data.decode("utf-8")
        elif core.media.data.is_url(part.data):
            return {
                "type": "document",
                "source": {"type": "url", "url": part.data},
            }
        else:
            import base64 as _b64

            text_data = _b64.b64decode(part.data).decode("utf-8")
        return {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": text_data,
            },
        }

    raise ValueError(f"Unsupported media type for Anthropic: {mt}")


async def _messages_to_anthropic(
    messages: list[core.messages.Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert internal messages to Anthropic API format.

    Returns (system_prompt, messages) tuple since Anthropic handles
    system prompts separately.

    Converts to the Anthropic wire format:

    - ``tool_use`` blocks in assistant messages
    - ``tool_result`` blocks in user messages (immediately after)

    A final merge pass ensures strictly alternating roles (Anthropic
    rejects consecutive same-role messages).
    """
    system_prompt: str | None = None
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            system_prompt = "".join(
                p.text for p in msg.parts if isinstance(p, core.messages.TextPart)
            )
        elif msg.role == "assistant":
            content: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for part in msg.parts:
                if isinstance(part, core.messages.ReasoningPart):
                    if part.signature:
                        content.append(
                            {
                                "type": "thinking",
                                "thinking": part.text,
                                "signature": part.signature,
                            }
                        )
                elif isinstance(part, core.messages.TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, core.messages.ToolPart):
                    tool_input = json.loads(part.tool_args) if part.tool_args else {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": part.tool_call_id,
                            "name": part.tool_name,
                            "input": tool_input,
                        }
                    )
                    if part.status in ("result", "error"):
                        entry: dict[str, Any] = {
                            "type": "tool_result",
                            "tool_use_id": part.tool_call_id,
                            "content": str(part.result)
                            if part.result is not None
                            else "",
                        }
                        if part.status == "error":
                            entry["is_error"] = True
                        tool_results.append(entry)

            if content:
                result.append({"role": "assistant", "content": content})
            if tool_results:
                result.append({"role": "user", "content": tool_results})
        elif msg.role == "user":
            has_files = any(isinstance(p, core.messages.FilePart) for p in msg.parts)
            if not has_files:
                content_text = "".join(
                    p.text for p in msg.parts if isinstance(p, core.messages.TextPart)
                )
                result.append({"role": "user", "content": content_text})
            else:
                user_content: list[dict[str, Any]] = []
                for p in msg.parts:
                    if isinstance(p, core.messages.TextPart):
                        user_content.append({"type": "text", "text": p.text})
                    elif isinstance(p, core.messages.FilePart):
                        user_content.append(_file_part_to_anthropic(p))
                result.append({"role": "user", "content": user_content})

    # Merge consecutive same-role messages (e.g. synthetic user(tool_result)
    # followed by a real user message).
    result = _merge_consecutive_roles(result)

    return system_prompt, result


def _merge_consecutive_roles(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge consecutive messages that share the same role.

    Anthropic requires strictly alternating user/assistant roles.  When
    our conversion emits a synthetic ``user`` message for ``tool_result``
    blocks followed by a real ``user`` message, they must be merged.

    Content is normalized to list-of-blocks so heterogeneous content
    (tool_result dicts + text strings) can coexist.
    """
    if not messages:
        return messages

    merged: list[dict[str, Any]] = [messages[0]]

    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev = _to_content_list(merged[-1]["content"])
            cur = _to_content_list(msg["content"])
            merged[-1]["content"] = prev + cur
        else:
            merged.append(msg)

    return merged


def _to_content_list(content: Any) -> list[dict[str, Any]]:
    """Normalize Anthropic message content to list-of-blocks format."""
    if isinstance(content, list):
        return list(content)
    return [{"type": "text", "text": content}]


class AnthropicModel(core.llm.LanguageModel):
    """Anthropic adapter with native extended thinking support."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool = False,
        budget_tokens: int = 10000,
    ) -> None:
        self._model = model
        self._thinking = thinking
        self._budget_tokens = budget_tokens
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        self._client = anthropic.AsyncAnthropic(base_url=base_url, api_key=resolved_key)

    async def stream_events(
        self,
        messages: list[core.messages.Message],
        tools: Sequence[core.tools.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[core.llm.StreamEvent]:
        """Yield raw stream events from Anthropic API."""
        system_prompt, anthropic_messages = await _messages_to_anthropic(messages)
        anthropic_tools = _tools_to_anthropic(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": 8192,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if self._thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._budget_tokens,
            }

        # Structured output: SDK handles schema transformation internally
        if output_type is not None:
            kwargs["output_format"] = output_type

        # Track block types by index to know what End event to emit
        block_types: dict[int, str] = {}  # index -> "text" | "thinking" | "tool_use"
        tool_ids: dict[int, str] = {}  # index -> tool_call_id
        signature_buffer: dict[int, str] = {}  # index -> accumulated signature

        stream_cm = self._client.messages.stream(**kwargs)

        async with stream_cm as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    idx = event.index
                    block_types[idx] = block.type

                    if block.type == "text":
                        yield core.llm.TextStart(block_id=str(idx))
                    elif block.type == "thinking":
                        yield core.llm.ReasoningStart(block_id=str(idx))
                    elif block.type == "tool_use":
                        tool_ids[idx] = block.id
                        yield core.llm.ToolStart(
                            tool_call_id=block.id, tool_name=block.name
                        )

                elif event.type == "content_block_delta":
                    delta = event.delta
                    idx = event.index

                    if delta.type == "text_delta":
                        yield core.llm.TextDelta(block_id=str(idx), delta=delta.text)
                    elif delta.type == "thinking_delta":
                        yield core.llm.ReasoningDelta(
                            block_id=str(idx), delta=delta.thinking
                        )
                    elif delta.type == "signature_delta":
                        # Accumulate signature for ReasoningEnd
                        signature_buffer[idx] = (
                            signature_buffer.get(idx, "") + delta.signature
                        )
                    elif delta.type == "input_json_delta":
                        tool_id = tool_ids.get(idx)
                        if tool_id:
                            yield core.llm.ToolArgsDelta(
                                tool_call_id=tool_id, delta=delta.partial_json
                            )

                elif event.type == "content_block_stop":
                    idx = event.index
                    block_type = block_types.get(idx)

                    if block_type == "text":
                        yield core.llm.TextEnd(block_id=str(idx))
                    elif block_type == "thinking":
                        yield core.llm.ReasoningEnd(
                            block_id=str(idx),
                            signature=signature_buffer.get(idx),
                        )
                    elif block_type == "tool_use":
                        tool_id = tool_ids.get(idx)
                        if tool_id:
                            yield core.llm.ToolEnd(tool_call_id=tool_id)

            # The Anthropic SDK accumulates usage across message_start and
            # message_delta events into current_message_snapshot.  Read it
            # once here instead of tracking state ourselves.
            snapshot = stream.current_message_snapshot
            sdk_usage = snapshot.usage
            usage = core.messages.Usage(
                input_tokens=sdk_usage.input_tokens or 0,
                output_tokens=sdk_usage.output_tokens or 0,
                cache_read_tokens=getattr(sdk_usage, "cache_read_input_tokens", None),
                cache_write_tokens=getattr(
                    sdk_usage, "cache_creation_input_tokens", None
                ),
                raw=sdk_usage.model_dump(exclude_none=True) or None,
            )
            yield core.llm.MessageDone(usage=usage)

    @override
    async def stream(
        self,
        messages: list[core.messages.Message],
        tools: Sequence[core.tools.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[core.messages.Message]:
        """Stream Messages (uses StreamHandler internally)."""
        handler = core.llm.StreamHandler()
        msg: core.messages.Message | None = None
        async for event in self.stream_events(messages, tools, output_type=output_type):
            msg = handler.handle_event(event)
            yield msg

        # After stream completes, validate and attach structured output part
        if output_type is not None and msg is not None and msg.text:
            data = json.loads(msg.text)
            output_type.model_validate(data)  # fail fast on bad data
            part = core.messages.StructuredOutputPart(
                data=data,
                output_type_name=f"{output_type.__module__}.{output_type.__qualname__}",
            )
            msg = msg.model_copy()
            msg.parts = [*msg.parts, part]
            yield msg
