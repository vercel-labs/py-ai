"""Vercel AI Gateway v3 wire — streaming and generation.

Wire protocol for the AI Gateway's v3 endpoints:

* ``/language-model`` — streaming text/tool/reasoning responses.
* ``/image-model`` — dedicated image generation.
* ``/video-model`` — dedicated video generation (SSE response).
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import httpx
import pydantic

from ...types import messages as messages_
from ...types import tools as tools_
from ..core import client as client_
from ..core import model as model_
from ..core.helpers import media as media_
from ..core.helpers import streaming as streaming_

_PROTOCOL_VERSION = "0.0.1"

# ---------------------------------------------------------------------------
# Request building — Message list → v3 prompt
# ---------------------------------------------------------------------------


async def _file_part_to_v3(part: messages_.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to a v3 ``file`` content part."""
    data = part.data
    if isinstance(data, str) and media_.is_downloadable_url(data):
        downloaded, _ = await media_.download(data)
        data = downloaded

    entry: dict[str, Any] = {
        "type": "file",
        "mediaType": part.media_type,
        "data": media_.data_to_data_url(data, part.media_type),
    }
    if part.filename is not None:
        entry["filename"] = part.filename
    return entry


async def _messages_to_prompt(
    messages: list[messages_.Message],
) -> list[dict[str, Any]]:
    """Convert ``Message`` list to the v3 prompt wire format."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "system":
                text = "".join(
                    p.text for p in msg.parts if isinstance(p, messages_.TextPart)
                )
                result.append({"role": "system", "content": text})

            case "user":
                content: list[dict[str, Any]] = []
                for p in msg.parts:
                    if isinstance(p, messages_.TextPart):
                        content.append({"type": "text", "text": p.text})
                    elif isinstance(p, messages_.FilePart):
                        content.append(await _file_part_to_v3(p))
                result.append({"role": "user", "content": content})

            case "assistant":
                assistant_content: list[dict[str, Any]] = []
                tool_results: list[dict[str, Any]] = []

                for part in msg.parts:
                    match part:
                        case messages_.ReasoningPart(text=text):
                            assistant_content.append(
                                {"type": "reasoning", "text": text}
                            )

                        case messages_.TextPart(text=text):
                            assistant_content.append({"type": "text", "text": text})

                        case messages_.ToolPart() as tp:
                            tool_input: Any = (
                                json.loads(tp.tool_args) if tp.tool_args else {}
                            )
                            assistant_content.append(
                                {
                                    "type": "tool-call",
                                    "toolCallId": tp.tool_call_id,
                                    "toolName": tp.tool_name,
                                    "input": tool_input,
                                }
                            )
                            if tp.status in ("result", "error"):
                                output = (
                                    {
                                        "type": "error-text",
                                        "value": (
                                            str(tp.result)
                                            if tp.result is not None
                                            else ""
                                        ),
                                    }
                                    if tp.status == "error"
                                    else {
                                        "type": "json",
                                        "value": tp.result,
                                    }
                                )
                                tool_results.append(
                                    {
                                        "type": "tool-result",
                                        "toolCallId": tp.tool_call_id,
                                        "toolName": tp.tool_name,
                                        "output": output,
                                    }
                                )

                result.append({"role": "assistant", "content": assistant_content})
                if tool_results:
                    result.append({"role": "tool", "content": tool_results})

    return result


async def _build_request_body(
    messages: list[messages_.Message],
    tools: Sequence[tools_.ToolLike] | None = None,
    output_type: type[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the ``LanguageModelV3CallOptions`` request body."""
    body: dict[str, Any] = {
        "prompt": await _messages_to_prompt(messages),
    }
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.param_schema,
            }
            for tool in tools
        ]
    if output_type is not None and issubclass(output_type, pydantic.BaseModel):
        body["responseFormat"] = {
            "type": "json",
            "schema": output_type.model_json_schema(),
            "name": output_type.__name__,
        }
    if kwargs.get("provider_options"):
        body["providerOptions"] = kwargs["provider_options"]
    return body


# ---------------------------------------------------------------------------
# SSE response parsing — v3 stream parts → StreamEvent
# ---------------------------------------------------------------------------


def _expand_tool_call(data: dict[str, Any]) -> list[streaming_.StreamEvent]:
    """Expand a complete ``tool-call`` part into Start + ArgsDelta + End."""
    tc_id = data.get("toolCallId", "")
    tool_name = data.get("toolName", "")
    tool_input = data.get("input", "")
    args_str = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
    return [
        streaming_.ToolStart(tool_call_id=tc_id, tool_name=tool_name),
        streaming_.ToolArgsDelta(tool_call_id=tc_id, delta=args_str),
        streaming_.ToolEnd(tool_call_id=tc_id),
    ]


def _parse_usage(data: Any) -> messages_.Usage:
    """Parse v3 usage data into an internal ``Usage``."""
    if not isinstance(data, dict):
        return messages_.Usage()

    input_tokens_obj = data.get("inputTokens")
    output_tokens_obj = data.get("outputTokens")

    if isinstance(input_tokens_obj, dict) or isinstance(output_tokens_obj, dict):
        inp = input_tokens_obj if isinstance(input_tokens_obj, dict) else {}
        out = output_tokens_obj if isinstance(output_tokens_obj, dict) else {}
        return messages_.Usage(
            input_tokens=inp.get("total") or 0,
            output_tokens=out.get("total") or 0,
            reasoning_tokens=out.get("reasoning"),
            cache_read_tokens=inp.get("cacheRead"),
            cache_write_tokens=inp.get("cacheWrite"),
            raw=data,
        )

    return messages_.Usage(
        input_tokens=data.get("prompt_tokens") or data.get("inputTokens") or 0,
        output_tokens=(data.get("completion_tokens") or data.get("outputTokens") or 0),
        raw=data,
    )


def _parse_stream_part(data: dict[str, Any]) -> list[streaming_.StreamEvent]:
    """Convert a ``LanguageModelV3StreamPart`` to internal events."""
    match data.get("type", ""):
        case "text-start":
            return [streaming_.TextStart(block_id=data.get("id", "text"))]

        case "text-delta":
            return [
                streaming_.TextDelta(
                    block_id=data.get("id", "text"),
                    delta=data.get("textDelta", data.get("delta", "")),
                )
            ]

        case "text-end":
            return [streaming_.TextEnd(block_id=data.get("id", "text"))]

        case "reasoning-start":
            return [streaming_.ReasoningStart(block_id=data.get("id", "reasoning"))]

        case "reasoning-delta":
            return [
                streaming_.ReasoningDelta(
                    block_id=data.get("id", "reasoning"),
                    delta=data.get("delta", ""),
                )
            ]

        case "reasoning-end":
            return [streaming_.ReasoningEnd(block_id=data.get("id", "reasoning"))]

        case "tool-input-start":
            return [
                streaming_.ToolStart(
                    tool_call_id=data.get("id", ""),
                    tool_name=data.get("toolName", ""),
                )
            ]

        case "tool-input-delta":
            return [
                streaming_.ToolArgsDelta(
                    tool_call_id=data.get("id", ""),
                    delta=data.get("delta", ""),
                )
            ]

        case "tool-input-end":
            return [streaming_.ToolEnd(tool_call_id=data.get("id", ""))]

        case "tool-call":
            return _expand_tool_call(data)

        case "file":
            return [
                streaming_.FileEvent(
                    block_id=data.get("id", f"file-{len(data)}"),
                    media_type=data.get("mediaType", "application/octet-stream"),
                    data=data.get("data", ""),
                )
            ]

        case "finish":
            usage_data = data.get("usage")
            usage = _parse_usage(usage_data) if usage_data else None
            match data.get("finishReason"):
                case dict() as d:
                    finish_reason = d.get("unified", "stop")
                case str() as s:
                    finish_reason = s
                case _:
                    finish_reason = "stop"
            return [streaming_.MessageDone(finish_reason=finish_reason, usage=usage)]

        case _:
            return []


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


def _request_headers(
    client: client_.Client,
    model: model_.Model,
    *,
    streaming: bool,
) -> dict[str, str]:
    """Build gateway-specific request headers."""
    h: dict[str, str] = {
        "Content-Type": "application/json",
        "ai-gateway-protocol-version": _PROTOCOL_VERSION,
        "ai-language-model-specification-version": "3",
        "ai-language-model-id": model.id,
        "ai-language-model-streaming": str(streaming).lower(),
    }
    if client.api_key:
        h["Authorization"] = f"Bearer {client.api_key}"
        h["ai-gateway-auth-method"] = "api-key"
    return h


# ---------------------------------------------------------------------------
# Public wire functions
# ---------------------------------------------------------------------------


async def stream(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    *,
    tools: Sequence[tools_.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[messages_.Message]:
    """Stream an LLM response through the AI Gateway v3 protocol.

    Yields ``Message`` snapshots as the response streams in.  Each
    snapshot is a complete, self-contained message reflecting the
    accumulated state up to that point.
    """
    body = await _build_request_body(
        messages, tools=tools, output_type=output_type, **kwargs
    )
    headers = _request_headers(client, model, streaming=True)
    url = f"{client.base_url.rstrip('/')}/language-model"

    handler = streaming_.StreamHandler()

    async with client.http.stream(
        "POST",
        url,
        json=body,
        headers=headers,
    ) as response:
        if response.status_code >= 400:
            await response.aread()
            raise RuntimeError(
                f"AI Gateway returned HTTP {response.status_code}: {response.text}"
            )

        async for line in response.aiter_lines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            for event in _parse_stream_part(data):
                msg = handler.handle_event(event)
                yield msg


# ---------------------------------------------------------------------------
# Generate — image / video (non-streaming media generation)
# ---------------------------------------------------------------------------


def _file_part_to_wire(part: messages_.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to the gateway wire format for input files."""
    data = part.data
    if isinstance(data, str) and media_.is_url(data):
        return {"type": "url", "url": data}
    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    elif isinstance(data, str):
        b64 = data
    else:
        b64 = str(data)
    return {"type": "file", "data": b64, "mediaType": part.media_type}


def _extract_prompt(messages: list[messages_.Message]) -> str:
    """Concatenate all text from user/system messages."""
    parts: list[str] = []
    for msg in messages:
        if msg.role in ("user", "system"):
            for p in msg.parts:
                if isinstance(p, messages_.TextPart):
                    parts.append(p.text)
    return " ".join(parts)


def _extract_input_files(messages: list[messages_.Message]) -> list[messages_.FilePart]:
    """Collect all file parts from user messages."""
    files: list[messages_.FilePart] = []
    for msg in messages:
        if msg.role == "user":
            for p in msg.parts:
                if isinstance(p, messages_.FilePart):
                    files.append(p)
    return files


def _generate_headers(
    client: client_.Client,
    model: model_.Model,
    *,
    spec_version_header: str,
) -> dict[str, str]:
    """Build gateway request headers for generate endpoints."""
    h: dict[str, str] = {
        "Content-Type": "application/json",
        "ai-gateway-protocol-version": _PROTOCOL_VERSION,
        spec_version_header: "3",
        "ai-model-id": model.id,
    }
    if client.api_key:
        h["Authorization"] = f"Bearer {client.api_key}"
        h["ai-gateway-auth-method"] = "api-key"
    return h


async def _generate_image(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    **kwargs: Any,
) -> messages_.Message:
    """Hit ``/image-model`` and return a Message with FileParts."""
    prompt = _extract_prompt(messages)
    input_files = _extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        "n": kwargs.get("n", 1),
        "providerOptions": kwargs.get("provider_options", {}),
    }
    if kwargs.get("size") is not None:
        body["size"] = kwargs["size"]
    if kwargs.get("aspect_ratio") is not None:
        body["aspectRatio"] = kwargs["aspect_ratio"]
    if kwargs.get("seed") is not None:
        body["seed"] = kwargs["seed"]
    if input_files:
        body["files"] = [_file_part_to_wire(f) for f in input_files]

    url = f"{client.base_url.rstrip('/')}/image-model"
    headers = _generate_headers(
        client, model, spec_version_header="ai-image-model-specification-version"
    )

    response = await client.http.post(
        url,
        json=body,
        headers=headers,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"AI Gateway image-model returned HTTP {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()
    raw_images: list[str] = data.get("images", [])
    usage_data = data.get("usage")
    usage = None
    if usage_data:
        usage = messages_.Usage(
            input_tokens=usage_data.get("inputTokens") or 0,
            output_tokens=usage_data.get("outputTokens") or 0,
        )

    files: list[messages_.FilePart] = []
    for img_b64 in raw_images:
        media_type = media_.detect_image_media_type(img_b64) or "image/png"
        files.append(messages_.FilePart(data=img_b64, media_type=media_type))

    return messages_.Message(role="assistant", parts=files, usage=usage)


async def _generate_video(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    **kwargs: Any,
) -> messages_.Message:
    """Hit ``/video-model`` (SSE) and return a Message with FileParts."""
    prompt = _extract_prompt(messages)
    input_files = _extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        "n": kwargs.get("n", 1),
        "providerOptions": kwargs.get("provider_options", {}),
    }
    if kwargs.get("aspect_ratio") is not None:
        body["aspectRatio"] = kwargs["aspect_ratio"]
    if kwargs.get("resolution") is not None:
        body["resolution"] = kwargs["resolution"]
    if kwargs.get("duration") is not None:
        body["duration"] = kwargs["duration"]
    if kwargs.get("fps") is not None:
        body["fps"] = kwargs["fps"]
    if kwargs.get("seed") is not None:
        body["seed"] = kwargs["seed"]
    if input_files:
        body["image"] = _file_part_to_wire(input_files[0])

    url = f"{client.base_url.rstrip('/')}/video-model"
    headers = _generate_headers(
        client, model, spec_version_header="ai-video-model-specification-version"
    )
    headers["accept"] = "text/event-stream"

    async with client.http.stream(
        "POST",
        url,
        json=body,
        headers=headers,
        timeout=httpx.Timeout(timeout=600.0, connect=10.0),
    ) as response:
        if response.status_code >= 400:
            await response.aread()
            raise RuntimeError(
                f"AI Gateway video-model returned HTTP {response.status_code}: "
                f"{response.text}"
            )

        # Read first SSE data event — the gateway sends a single result event.
        event_data: dict[str, Any] = {}
        async for line in response.aiter_lines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            try:
                event_data = json.loads(payload)
                break
            except json.JSONDecodeError:
                continue

    if event_data.get("type") == "error":
        raise RuntimeError(
            f"AI Gateway video generation error: "
            f"{event_data.get('message', 'unknown error')}"
        )

    raw_videos: list[dict[str, Any]] = event_data.get("videos", [])
    files: list[messages_.FilePart] = []
    for video_data in raw_videos:
        vtype = video_data.get("type", "base64")
        media_type = video_data.get("mediaType", "video/mp4")

        if vtype == "url":
            downloaded_bytes, content_type = await media_.download(video_data["url"])
            if content_type:
                media_type = content_type
            files.append(
                messages_.FilePart(data=downloaded_bytes, media_type=media_type)
            )
        else:
            raw_data = video_data.get("data", "")
            files.append(messages_.FilePart(data=raw_data, media_type=media_type))

    return messages_.Message(role="assistant", parts=files)


async def generate(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    **kwargs: Any,
) -> messages_.Message:
    """Generate media (images or video) through the AI Gateway.

    Dispatches to ``/image-model`` or ``/video-model`` based on the
    model's capabilities.

    Keyword args are forwarded to the underlying endpoint and may include
    ``n``, ``size``, ``aspect_ratio``, ``seed``, ``duration``, ``fps``,
    ``resolution``, ``provider_options``.
    """
    caps = model.capabilities
    if "video" in caps:
        return await _generate_video(client, model, messages, **kwargs)
    # Default to image generation
    return await _generate_image(client, model, messages, **kwargs)
