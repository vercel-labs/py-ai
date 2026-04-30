"""AI Gateway v3 adapter.

Converts internal messages to AI Gateway wire payloads and maps gateway
responses back to public event/message types."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import httpx
import pydantic

from ...types import events as events_
from ...types import media
from ...types import messages as messages_
from ...types import proto as proto_
from ...types import usage as usage_
from ..core import client as client_
from ..core import model as model_
from ..core.helpers import files
from ..core.params import GenerateParams as GenerateParams
from ..core.params import ImageParams as ImageParams
from ..core.params import VideoParams as VideoParams
from . import errors, sdk

# ---------------------------------------------------------------------------
# Shared request helpers
# ---------------------------------------------------------------------------


def _extract_prompt(messages: list[messages_.Message]) -> str:
    """Concatenate all text from user/system messages into one prompt."""
    parts: list[str] = []
    for msg in messages:
        if msg.role in ("user", "system"):
            for p in msg.parts:
                if isinstance(p, messages_.TextPart):
                    parts.append(p.text)
    return " ".join(parts)


def _extract_input_files(
    messages: list[messages_.Message],
) -> list[messages_.FilePart]:
    """Collect all file parts from user messages."""
    files_: list[messages_.FilePart] = []
    for msg in messages:
        if msg.role == "user":
            for p in msg.parts:
                if isinstance(p, messages_.FilePart):
                    files_.append(p)
    return files_


def _file_part_to_wire(part: messages_.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to the gateway wire format for input files."""
    data = part.data
    if isinstance(data, str) and media.is_url(data):
        return {"type": "url", "url": data}
    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    elif isinstance(data, str):
        b64 = data
    else:
        b64 = str(data)
    return {"type": "file", "data": b64, "mediaType": part.media_type}


# ---------------------------------------------------------------------------
# Streaming request building — Message list → v3 prompt
# ---------------------------------------------------------------------------


async def _file_part_to_v3(part: messages_.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to a v3 ``file`` content part."""
    data = part.data
    if isinstance(data, str) and media.is_downloadable_url(data):
        downloaded, _ = await files.download(data)
        data = downloaded

    entry: dict[str, Any] = {
        "type": "file",
        "mediaType": part.media_type,
        "data": media.data_to_data_url(data, part.media_type),
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
                for part in msg.parts:
                    match part:
                        case messages_.ReasoningPart(text=text):
                            assistant_content.append(
                                {"type": "reasoning", "text": text}
                            )
                        case messages_.TextPart(text=text):
                            assistant_content.append({"type": "text", "text": text})
                        case messages_.ToolCallPart() as tp:
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
                result.append({"role": "assistant", "content": assistant_content})

            case "tool":
                tool_results: list[dict[str, Any]] = []
                for part in msg.parts:
                    if isinstance(part, messages_.ToolResultPart):
                        output = (
                            {
                                "type": "error-text",
                                "value": (
                                    str(part.result) if part.result is not None else ""
                                ),
                            }
                            if part.is_error
                            else {
                                "type": "json",
                                "value": part.result,
                            }
                        )
                        tool_results.append(
                            {
                                "type": "tool-result",
                                "toolCallId": part.tool_call_id,
                                "toolName": part.tool_name,
                                "output": output,
                            }
                        )
                if tool_results:
                    result.append({"role": "tool", "content": tool_results})

    return result


async def _build_request_body(
    messages: list[messages_.Message],
    tools: Sequence[proto_.ToolLike] | None = None,
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
# Streaming response parsing — v3 stream parts → public Event
# ---------------------------------------------------------------------------


def _expand_tool_call(
    data: dict[str, Any], streamed_tool_ids: set[str]
) -> list[events_.Event]:
    """Expand a complete ``tool-call`` part into Start + Delta + End.

    Returns empty when the tool was already streamed via ``tool-input-*``.
    """
    tc_id = data.get("toolCallId", "")
    if tc_id in streamed_tool_ids:
        return []
    tool_name = data.get("toolName", "")
    tool_input = data.get("input", "")
    args_str = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
    return [
        events_.ToolStart(tool_call_id=tc_id, tool_name=tool_name),
        events_.ToolDelta(tool_call_id=tc_id, chunk=args_str),
        events_.ToolEnd(tool_call_id=tc_id),
    ]


def _parse_usage(data: Any) -> usage_.Usage:
    """Parse v3 usage data into an internal ``Usage``."""
    if not isinstance(data, dict):
        return usage_.Usage()

    input_tokens_obj = data.get("inputTokens")
    output_tokens_obj = data.get("outputTokens")

    if isinstance(input_tokens_obj, dict) or isinstance(output_tokens_obj, dict):
        inp = input_tokens_obj if isinstance(input_tokens_obj, dict) else {}
        out = output_tokens_obj if isinstance(output_tokens_obj, dict) else {}
        return usage_.Usage(
            input_tokens=inp.get("total") or 0,
            output_tokens=out.get("total") or 0,
            reasoning_tokens=out.get("reasoning"),
            cache_read_tokens=inp.get("cacheRead"),
            cache_write_tokens=inp.get("cacheWrite"),
            raw=data,
        )

    return usage_.Usage(
        input_tokens=data.get("prompt_tokens") or data.get("inputTokens") or 0,
        output_tokens=(data.get("completion_tokens") or data.get("outputTokens") or 0),
        raw=data,
    )


def _parse_stream_part(
    data: dict[str, Any], streamed_tool_ids: set[str]
) -> list[events_.Event]:
    """Convert a ``LanguageModelV3StreamPart`` to public events."""
    match data.get("type", ""):
        case "text-start":
            return [events_.TextStart(block_id=data.get("id", "text"))]

        case "text-delta":
            return [
                events_.TextDelta(
                    block_id=data.get("id", "text"),
                    chunk=data.get("textDelta", data.get("delta", "")),
                )
            ]

        case "text-end":
            return [events_.TextEnd(block_id=data.get("id", "text"))]

        case "reasoning-start":
            return [events_.ReasoningStart(block_id=data.get("id", "reasoning"))]

        case "reasoning-delta":
            return [
                events_.ReasoningDelta(
                    block_id=data.get("id", "reasoning"),
                    chunk=data.get("delta", ""),
                )
            ]

        case "reasoning-end":
            return [events_.ReasoningEnd(block_id=data.get("id", "reasoning"))]

        case "tool-input-start":
            tcid = data.get("id", "")
            streamed_tool_ids.add(tcid)
            return [
                events_.ToolStart(
                    tool_call_id=tcid,
                    tool_name=data.get("toolName", ""),
                )
            ]

        case "tool-input-delta":
            return [
                events_.ToolDelta(
                    tool_call_id=data.get("id", ""),
                    chunk=data.get("delta", ""),
                )
            ]

        case "tool-input-end":
            return [events_.ToolEnd(tool_call_id=data.get("id", ""))]

        case "tool-call":
            return _expand_tool_call(data, streamed_tool_ids)

        case "file":
            return [
                events_.FileEvent(
                    block_id=data.get("id", ""),
                    media_type=data.get("mediaType", "application/octet-stream"),
                    data=data.get("data", ""),
                )
            ]

        case "finish":
            usage_data = data.get("usage")
            usage = _parse_usage(usage_data) if usage_data else None
            return [events_.StreamEnd(usage=usage)]

        case _:
            return []


async def stream(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    *,
    tools: Sequence[proto_.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[events_.Event]:
    """Stream an LLM response through the AI Gateway v3 protocol."""
    body = await _build_request_body(
        messages, tools=tools, output_type=output_type, **kwargs
    )
    gateway = sdk.GatewayClient(client, model)

    try:
        async with gateway.stream(
            "language-model",
            body,
            model_type="language",
            streaming=True,
        ) as response:
            yield events_.StreamStart()
            streamed_tool_ids: set[str] = set()
            async for data in gateway.iter_sse(response):
                for event in _parse_stream_part(data, streamed_tool_ids):
                    yield event
    except errors.GatewayError:
        raise
    except httpx.TimeoutException as exc:
        raise errors.GatewayTimeoutError(cause=exc) from exc
    except Exception as exc:
        raise errors.GatewayResponseError(
            message=f"Unexpected error during streaming: {exc}",
            cause=exc,
        ) from exc


# ---------------------------------------------------------------------------
# Media generation
# ---------------------------------------------------------------------------


async def _generate_image(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    params: ImageParams,
) -> messages_.Message:
    """Hit ``/image-model`` and return a Message with FileParts."""
    prompt = _extract_prompt(messages)
    input_files = _extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        **params.model_dump(by_alias=True, exclude_none=True),
    }
    if input_files:
        body["files"] = [_file_part_to_wire(f) for f in input_files]

    gateway = sdk.GatewayClient(client, model)
    response = await gateway.post_json("image-model", body, model_type="image")

    data = response.json()
    raw_images: list[str] = data.get("images", [])
    usage_data = data.get("usage")
    usage = None
    if usage_data:
        usage = messages_.Usage(
            input_tokens=usage_data.get("inputTokens") or 0,
            output_tokens=usage_data.get("outputTokens") or 0,
        )

    parts: list[messages_.Part] = []
    for img_b64 in raw_images:
        media_type = media.detect_image_media_type(img_b64) or "image/png"
        parts.append(messages_.FilePart(data=img_b64, media_type=media_type))

    return messages_.Message(role="assistant", parts=parts, usage=usage)


async def _generate_video(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    params: VideoParams,
) -> messages_.Message:
    """Hit ``/video-model`` (SSE) and return a Message with FileParts."""
    prompt = _extract_prompt(messages)
    input_files = _extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        **params.model_dump(by_alias=True, exclude_none=True),
    }
    if input_files:
        body["image"] = _file_part_to_wire(input_files[0])

    gateway = sdk.GatewayClient(client, model)

    async with gateway.stream(
        "video-model",
        body,
        model_type="video",
        accept="text/event-stream",
        timeout=httpx.Timeout(timeout=600.0, connect=10.0),
    ) as response:
        event_data: dict[str, Any] = {}
        async for parsed in gateway.iter_sse(response):
            event_data = parsed
            break

    if not event_data:
        raise errors.GatewayResponseError(
            "SSE stream ended without any data events",
        )

    if event_data.get("type") == "error":
        raise errors.GatewayInvalidRequestError(
            message=event_data.get("message", "unknown error"),
            status_code=event_data.get("statusCode", 400),
        )

    raw_videos: list[dict[str, Any]] = event_data.get("videos", [])
    parts: list[messages_.Part] = []
    for video_data in raw_videos:
        vtype = video_data.get("type", "base64")
        media_type = video_data.get("mediaType", "video/mp4")

        if vtype == "url":
            downloaded_bytes, content_type = await files.download(video_data["url"])
            if content_type:
                media_type = content_type
            parts.append(
                messages_.FilePart(data=downloaded_bytes, media_type=media_type)
            )
        else:
            raw_data = video_data.get("data", "")
            parts.append(messages_.FilePart(data=raw_data, media_type=media_type))

    return messages_.Message(role="assistant", parts=parts)


async def generate(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    params: GenerateParams,
) -> messages_.Message:
    """Generate media through the AI Gateway."""
    if isinstance(params, VideoParams):
        return await _generate_video(client, model, messages, params)
    return await _generate_image(client, model, messages, params)
