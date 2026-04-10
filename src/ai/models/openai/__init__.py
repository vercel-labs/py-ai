"""OpenAI provider — adapter for the OpenAI chat completions API.

The adapter module is loaded lazily to avoid pulling in the ``openai``
SDK at import time.
"""

__all__: list[str] = []


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
