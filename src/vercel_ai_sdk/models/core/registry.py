"""Registry mapping ``api`` strings to execution functions.

Provider adapters call :func:`register_stream` / :func:`register_generate`
to make themselves available.  The module-level ``stream()`` and
``generate()`` functions in :mod:`vercel_ai_sdk.models` use
:func:`get_stream_fn` / :func:`get_generate_fn` to dispatch.
"""

from __future__ import annotations

from .protocol import GenerateFn, StreamFn

_stream_fns: dict[str, StreamFn] = {}
_generate_fns: dict[str, GenerateFn] = {}


def register_stream(api: str, fn: StreamFn) -> None:
    """Register a :class:`StreamFn` for the given wire-protocol ``api``."""
    _stream_fns[api] = fn


def register_generate(api: str, fn: GenerateFn) -> None:
    """Register a :class:`GenerateFn` for the given wire-protocol ``api``."""
    _generate_fns[api] = fn


def get_stream_fn(api: str) -> StreamFn:
    """Look up the registered :class:`StreamFn` for ``api``.

    Raises :class:`KeyError` with a descriptive message if no function
    has been registered for the given ``api``.
    """
    try:
        return _stream_fns[api]
    except KeyError:
        registered = ", ".join(sorted(_stream_fns)) or "(none)"
        raise KeyError(
            f"No StreamFn registered for api={api!r}. Registered: {registered}"
        ) from None


def get_generate_fn(api: str) -> GenerateFn:
    """Look up the registered :class:`GenerateFn` for ``api``.

    Raises :class:`KeyError` with a descriptive message if no function
    has been registered for the given ``api``.
    """
    try:
        return _generate_fns[api]
    except KeyError:
        registered = ", ".join(sorted(_generate_fns)) or "(none)"
        raise KeyError(
            f"No GenerateFn registered for api={api!r}. Registered: {registered}"
        ) from None
