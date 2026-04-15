"""Replay middleware for serverless re-entry across approval round-trips."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import ai
import pydantic

from replay_types import (
    PendingHookInfo,
    RecordedModelStep,
    RecordedToolResult,
    ReplayMetadata,
    ReplayState,
)

logger = logging.getLogger(__name__)

REPLAY_VERSION = 1


class ReplayMismatchError(RuntimeError):
    """Replay state does not match the live execution path."""


def serialize_message(message: ai.Message) -> dict[str, Any]:
    """Serialize an SDK message to JSON-safe data."""
    return message.model_dump(mode="json")


def deserialize_message(data: dict[str, Any]) -> ai.Message:
    """Deserialize a replayed SDK message."""
    return ai.Message.model_validate(data)


def serialize_messages(messages: list[ai.Message]) -> list[dict[str, Any]]:
    """Serialize a list of messages."""
    return [serialize_message(message) for message in messages]


def deserialize_messages(messages: list[dict[str, Any]]) -> list[ai.Message]:
    """Deserialize a list of messages."""
    return [deserialize_message(message) for message in messages]


def _canonicalize_part(part: ai.Part) -> dict[str, Any]:
    """Return a stable representation of a message part for fingerprinting."""
    data = part.model_dump(mode="json")
    data.pop("id", None)
    data.pop("state", None)
    data.pop("delta", None)
    data.pop("args_delta", None)
    return data


def _canonicalize_message(message: ai.Message) -> dict[str, Any]:
    """Return a stable representation of a message for fingerprinting."""
    return {
        "role": message.role,
        "parts": [_canonicalize_part(part) for part in message.parts],
    }


def compute_replay_fingerprint(
    *,
    session_id: str,
    system: ai.Message,
    messages: list[ai.Message],
    model: ai.Model,
    tools: list[ai.Tool[..., Any]],
) -> str:
    """Hash the normalized run input so stale replay state is never reused."""
    payload = {
        "version": REPLAY_VERSION,
        "session_id": session_id,
        "system": _canonicalize_message(system),
        "messages": [_canonicalize_message(message) for message in messages],
        "model": {
            "id": model.id,
            "adapter": model.adapter,
        },
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "param_schema": tool.param_schema,
            }
            for tool in tools
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
