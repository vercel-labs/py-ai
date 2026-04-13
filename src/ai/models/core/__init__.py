"""Core types for models."""

from .adapters import register_generate, register_stream
from .api import check_connection, generate, stream
from .client import Client
from .model import Model
from .proto import CheckConnFn, GenerateFn, Provider, StreamFn
from .types import GenerateParams, ImageParams, StreamResult, VideoParams

__all__ = [
    "CheckConnFn",
    "Client",
    "GenerateFn",
    "GenerateParams",
    "ImageParams",
    "Model",
    "Provider",
    "StreamFn",
    "StreamResult",
    "VideoParams",
    "check_connection",
    "generate",
    "register_generate",
    "register_stream",
    "stream",
]
