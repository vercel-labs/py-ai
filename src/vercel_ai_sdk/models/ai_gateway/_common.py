"""Shared helpers for the AI Gateway v3 adapter.

Contains utilities used by both the streaming (language-model) and generation
(image-model, video-model) endpoints.

.. note::

    Several helpers here are candidates for lifting to framework-level:

    - ``extract_prompt`` / ``extract_input_files`` → ``Message`` methods
    - ``parse_sse_lines`` → ``core/helpers/sse.py``
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from ...types import messages as messages_
from ..core import client as client_
from ..core import model as model_
from ..core.helpers import media as media_

_PROTOCOL_VERSION = "0.0.1"


# ---------------------------------------------------------------------------
# Message extraction helpers
# ---------------------------------------------------------------------------
# TODO: lift to Message methods — these are universally useful.


def extract_prompt(messages: list[messages_.Message]) -> str:
    """Concatenate all text from user/system messages into a single prompt string."""
    parts: list[str] = []
    for msg in messages:
        if msg.role in ("user", "system"):
            for p in msg.parts:
                if isinstance(p, messages_.TextPart):
                    parts.append(p.text)
    return " ".join(parts)


def extract_input_files(messages: list[messages_.Message]) -> list[messages_.FilePart]:
    """Collect all file parts from user messages."""
    files: list[messages_.FilePart] = []
    for msg in messages:
        if msg.role == "user":
            for p in msg.parts:
                if isinstance(p, messages_.FilePart):
                    files.append(p)
    return files


# ---------------------------------------------------------------------------
# Wire format helpers
# ---------------------------------------------------------------------------


def file_part_to_wire(part: messages_.FilePart) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Request headers
# ---------------------------------------------------------------------------


def request_headers(
    client: client_.Client,
    model: model_.Model,
    *,
    model_type: str = "language",
    streaming: bool = False,
) -> dict[str, str]:
    """Build gateway-specific request headers.

    Args:
        client: The HTTP client (provides api_key).
        model: The model (provides id).
        model_type: One of ``"language"``, ``"image"``, ``"video"``.
        streaming: Whether this is a streaming request (language-model only).
    """
    h: dict[str, str] = {
        "Content-Type": "application/json",
        "ai-gateway-protocol-version": _PROTOCOL_VERSION,
    }

    if model_type == "language":
        h["ai-language-model-specification-version"] = "3"
        h["ai-language-model-id"] = model.id
        h["ai-language-model-streaming"] = str(streaming).lower()
    elif model_type == "image":
        h["ai-image-model-specification-version"] = "3"
        h["ai-model-id"] = model.id
    elif model_type == "video":
        h["ai-video-model-specification-version"] = "3"
        h["ai-model-id"] = model.id

    if client.api_key:
        h["Authorization"] = f"Bearer {client.api_key}"
        h["ai-gateway-auth-method"] = "api-key"

    return h


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------
# TODO: lift to core/helpers/sse.py — any SSE-based adapter will need this.


async def parse_sse_lines(
    response: httpx.Response,
) -> AsyncGenerator[dict[str, Any]]:
    """Yield parsed JSON dicts from an SSE response stream.

    Handles the ``data: <json>`` / ``data: [DONE]`` protocol used by the
    AI Gateway's streaming endpoints.
    """
    async for line in response.aiter_lines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue
