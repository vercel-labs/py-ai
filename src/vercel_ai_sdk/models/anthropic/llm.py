from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

import anthropic
import pydantic

from ...types import messages as messages_
from ...types import tools as tools_
from ..core import llm as llm_
from ..core import media


def _tools_to_anthropic(tools: Sequence[tools_.ToolLike]) -> list[dict[str, Any]]:
    """Convert internal Tool objects to Anthropic tool schema format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.param_schema,
        }
        for tool in tools
    ]


def _file_part_to_anthropic(part: messages_.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to an Anthropic content block.

    * ``image/*`` → ``{"type": "image", "source": ...}``
    * ``application/pdf`` → ``{"type": "document", "source": ...}``
    * ``text/plain`` → ``{"type": "document", "source": {"type": "text", ...}}``
    * anything else → ``ValueError``
    """
    mt = part.media_type

    if mt.startswith("image/"):
        media_type = "image/jpeg" if mt == "image/*" else mt
        if isinstance(part.data, str) and media.data.is_url(part.data):
            return {
                "type": "image",
                "source": {"type": "url", "url": part.data},
            }
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": media.data.data_to_base64(part.data),
            },
        }

    if mt == "application/pdf":
        if isinstance(part.data, str) and media.data.is_url(part.data):
            return {
                "type": "document",
                "source": {"type": "url", "url": part.data},
            }
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": media.data.data_to_base64(part.data),
            },
        }

    if mt == "text/plain":
        # Anthropic accepts text documents with source.type="text"
        if isinstance(part.data, bytes):
            text_data = part.data.decode("utf-8")
        elif media.data.is_url(part.data):
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
    messages: list[messages_.Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert internal messages to Anthropic API format."""
    system_prompt: str | None = None
    result: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "system":
                system_prompt = "".join(
                    p.text for p in msg.parts if isinstance(p, messages_.TextPart)
                )
            case "assistant":
                content: list[dict[str, Any]] = []
                tool_results: list[dict[str, Any]] = []

                for part in msg.parts:
                    match part:
                        case messages_.ReasoningPart(text=text, signature=signature):
                            if signature:
                                content.append(
                                    {
                                        "type": "thinking",
                                        "thinking": text,
                                        "signature": signature,
                                    }
                                )
                        case messages_.TextPart(text=text):
                            content.append({"type": "text", "text": text})
                        case messages_.ToolPart():
                            tool_input = (
                                json.loads(part.tool_args) if part.tool_args else {}
                            )
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
            case "user":
                has_files = any(isinstance(p, messages_.FilePart) for p in msg.parts)
                if not has_files:
                    content_text = "".join(
                        p.text for p in msg.parts if isinstance(p, messages_.TextPart)
                    )
                    result.append({"role": "user", "content": content_text})
                else:
                    user_content: list[dict[str, Any]] = []
                    for p in msg.parts:
                        match p:
                            case messages_.TextPart(text=text):
                                user_content.append({"type": "text", "text": text})
                            case messages_.FilePart():
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


class AnthropicModel(llm_.LanguageModel):
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

    @override
    async def stream_events(
        self,
        messages: list[messages_.Message],
        tools: Sequence[tools_.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
    ) -> AsyncGenerator[llm_.StreamEvent]:
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
                match event.type:
                    case "content_block_start":
                        block = event.content_block
                        idx = event.index
                        block_types[idx] = block.type

                        match block.type:
                            case "text":
                                yield llm_.TextStart(block_id=str(idx))
                            case "thinking":
                                yield llm_.ReasoningStart(block_id=str(idx))
                            case "tool_use":
                                tool_ids[idx] = block.id
                                yield llm_.ToolStart(
                                    tool_call_id=block.id, tool_name=block.name
                                )

                    case "content_block_delta":
                        delta = event.delta
                        idx = event.index

                        match delta.type:
                            case "text_delta":
                                yield llm_.TextDelta(
                                    block_id=str(idx), delta=delta.text
                                )
                            case "thinking_delta":
                                yield llm_.ReasoningDelta(
                                    block_id=str(idx), delta=delta.thinking
                                )
                            case "signature_delta":
                                # Accumulate signature for ReasoningEnd
                                signature_buffer[idx] = (
                                    signature_buffer.get(idx, "") + delta.signature
                                )
                            case "input_json_delta":
                                tool_id = tool_ids.get(idx)
                                if tool_id:
                                    yield llm_.ToolArgsDelta(
                                        tool_call_id=tool_id,
                                        delta=delta.partial_json,
                                    )

                    case "content_block_stop":
                        idx = event.index
                        match block_types.get(idx):
                            case "text":
                                yield llm_.TextEnd(block_id=str(idx))
                            case "thinking":
                                yield llm_.ReasoningEnd(
                                    block_id=str(idx),
                                    signature=signature_buffer.get(idx),
                                )
                            case "tool_use":
                                tool_id = tool_ids.get(idx)
                                if tool_id:
                                    yield llm_.ToolEnd(tool_call_id=tool_id)

            # The Anthropic SDK accumulates usage across message_start and
            # message_delta events into current_message_snapshot.  Read it
            # once here instead of tracking state ourselves.
            snapshot = stream.current_message_snapshot
            sdk_usage = snapshot.usage
            usage = messages_.Usage(
                input_tokens=sdk_usage.input_tokens or 0,
                output_tokens=sdk_usage.output_tokens or 0,
                cache_read_tokens=getattr(sdk_usage, "cache_read_input_tokens", None),
                cache_write_tokens=getattr(
                    sdk_usage, "cache_creation_input_tokens", None
                ),
                raw=sdk_usage.model_dump(exclude_none=True) or None,
            )
            yield llm_.MessageDone(usage=usage)
