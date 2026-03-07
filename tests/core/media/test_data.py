"""Tests for media data-format helpers (URL detection, base-64, data URLs)."""

from vercel_ai_sdk.core.media.data import (
    data_to_base64,
    data_to_data_url,
    is_url,
    split_data_url,
)

# -- is_url ----------------------------------------------------------------


def test_is_url_http() -> None:
    assert is_url("https://example.com/img.png") is True
    assert is_url("http://example.com/img.png") is True


def test_is_url_data() -> None:
    assert is_url("data:image/png;base64,abc") is True


def test_is_url_base64() -> None:
    assert is_url("iVBORw0KGgo=") is False


# -- data_to_base64 -------------------------------------------------------


def test_data_to_base64_bytes() -> None:
    assert data_to_base64(b"\x01\x02\x03") == "AQID"


def test_data_to_base64_passthrough() -> None:
    assert data_to_base64("AQID") == "AQID"


def test_data_to_base64_extracts_from_data_url() -> None:
    """data: URLs must have the prefix stripped -- providers need raw base64."""
    result = data_to_base64("data:image/png;base64,AQID")
    assert result == "AQID"


def test_data_to_base64_passthrough_http_url() -> None:
    """HTTP URLs are passed through -- caller must handle."""
    url = "https://example.com/img.png"
    assert data_to_base64(url) == url


# -- data_to_data_url ------------------------------------------------------


def test_data_to_data_url_from_bytes() -> None:
    result = data_to_data_url(b"\x01\x02\x03", "image/png")
    assert result == "data:image/png;base64,AQID"


def test_data_to_data_url_passthrough_url() -> None:
    url = "https://example.com/img.png"
    assert data_to_data_url(url, "image/png") == url


# -- split_data_url --------------------------------------------------------


def test_split_data_url_valid() -> None:
    mt, b64 = split_data_url("data:image/png;base64,iVBOR")
    assert mt == "image/png"
    assert b64 == "iVBOR"


def test_split_data_url_non_data_url() -> None:
    mt, b64 = split_data_url("https://example.com/img.png")
    assert mt is None
    assert b64 is None


def test_split_data_url_malformed() -> None:
    mt, b64 = split_data_url("data:")
    assert mt is None
    assert b64 is None
