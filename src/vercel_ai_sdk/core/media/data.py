"""Data-format helpers for multimodal content.

URL detection, ``data:`` URL parsing, base-64 encoding/decoding, and
media-type inference utilities used by :class:`~vercel_ai_sdk.core.messages.FilePart`
and the provider converters.
"""

from __future__ import annotations

import base64
import mimetypes

# -- URL helpers -----------------------------------------------------------


def is_url(data: str) -> bool:
    """Return True if *data* looks like a URL rather than raw base-64."""
    return data.startswith(("http://", "https://", "data:"))


def is_downloadable_url(data: str) -> bool:
    """Return True if *data* is an ``http(s)://`` URL that can be fetched."""
    return data.startswith(("http://", "https://"))


def split_data_url(url: str) -> tuple[str | None, str | None]:
    """Parse a ``data:`` URL into ``(media_type, base64_content)``.

    Returns ``(None, None)`` if the input is not a valid ``data:`` URL.

    Example::

        >>> split_data_url("data:image/png;base64,iVBOR...")
        ("image/png", "iVBOR...")
    """
    if not url.startswith("data:"):
        return None, None
    try:
        header, b64_content = url.split(",", 1)
        # header = "data:image/png;base64"
        mt = header.split(";")[0].split(":", 1)[1]
        return (mt or None), (b64_content or None)
    except (ValueError, IndexError):
        return None, None


# -- encoding helpers ------------------------------------------------------


def data_to_base64(data: str | bytes) -> str:
    """Ensure *data* is a base-64 encoded string.

    * ``bytes`` -> base-64 encoded.
    * ``str`` that is a ``data:`` URL -> base-64 content extracted.
    * ``str`` that is an ``http(s)://`` URL -> returned as-is (caller
      must handle).
    * ``str`` that is not a URL -> assumed to already be base-64.
    """
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("ascii")
    if data.startswith("data:"):
        _, b64 = split_data_url(data)
        if b64 is not None:
            return b64
    return data


def data_to_data_url(data: str | bytes, media_type: str) -> str:
    """Convert *data* to a ``data:`` URL.  Passes through existing URLs."""
    if isinstance(data, str) and is_url(data):
        return data
    b64 = data_to_base64(data)
    return f"data:{media_type};base64,{b64}"


# -- media-type inference --------------------------------------------------


def infer_media_type(url: str) -> str:
    """Infer IANA media type from a URL.

    * ``data:image/png;base64,...`` -> ``"image/png"``
    * ``https://example.com/cat.jpg`` -> ``"image/jpeg"`` (via :mod:`mimetypes`)
    * Unknown -> raises :class:`ValueError`
    """
    if url.startswith("data:"):
        # data:[<mediatype>][;base64],<data>
        rest = url[5:]  # strip "data:"
        sep = rest.find(",")
        meta = rest[:sep] if sep != -1 else rest
        mt = meta.split(";")[0]
        if mt:
            return mt
    else:
        guessed, _ = mimetypes.guess_type(url)
        if guessed:
            return guessed
    raise ValueError(
        f"Cannot infer media_type from URL: {url!r}. Provide media_type explicitly."
    )
