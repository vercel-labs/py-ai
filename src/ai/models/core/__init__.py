"""Core types for models."""

from ...providers.base import Provider
from . import helpers
from .adapters import register_generate, register_stream
from .api import (
    Executor,
    GenerateExecutor,
    GenerateRequest,
    Stream,
    StreamExecutor,
    StreamRequest,
    check_connection,
    generate,
    stream,
)
from .model import Model, get_model
from .params import GenerateParams, ImageParams, VideoParams
from .proto import CheckConnFn, GenerateFn, StreamFn

__all__ = [
    "CheckConnFn",
    "Executor",
    "GenerateExecutor",
    "GenerateFn",
    "GenerateParams",
    "GenerateRequest",
    "ImageParams",
    "Model",
    "Provider",
    "Stream",
    "StreamExecutor",
    "StreamFn",
    "StreamRequest",
    "VideoParams",
    "check_connection",
    "generate",
    "get_model",
    "register_generate",
    "register_stream",
    "stream",
    "helpers",
]
