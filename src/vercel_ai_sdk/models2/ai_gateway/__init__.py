"""AI Gateway provider — adapter for the Vercel AI Gateway v3 protocol."""

from .adapter import generate, stream

__all__ = ["generate", "stream"]
