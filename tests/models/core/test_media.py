"""Tests for media data helpers and magic-byte media type detection.

Covers ``is_url``, ``data_to_base64``, ``data_to_data_url``,
``split_data_url``, ``detect_image_media_type``, ``detect_audio_media_type``,
and edge cases.
"""

from __future__ import annotations

import base64

from vercel_ai_sdk.models.core.helpers.media import (
    data_to_base64,
    data_to_data_url,
    detect_audio_media_type,
    detect_image_media_type,
    is_url,
    split_data_url,
)

# ---------------------------------------------------------------------------
# is_url
# ---------------------------------------------------------------------------


class TestIsUrl:
    def test_http(self) -> None:
        assert is_url("https://example.com/img.png") is True
        assert is_url("http://example.com/img.png") is True

    def test_data(self) -> None:
        assert is_url("data:image/png;base64,iVBOR") is True

    def test_base64(self) -> None:
        assert is_url("iVBORw0KGgo=") is False


# ---------------------------------------------------------------------------
# data_to_base64
# ---------------------------------------------------------------------------


class TestDataToBase64:
    def test_bytes(self) -> None:
        raw = b"\x89PNG"
        result = data_to_base64(raw)
        assert base64.b64decode(result) == raw

    def test_passthrough(self) -> None:
        b64 = base64.b64encode(b"hello").decode()
        assert data_to_base64(b64) == b64

    def test_extracts_from_data_url(self) -> None:
        payload = base64.b64encode(b"hello").decode()
        data_url = f"data:image/png;base64,{payload}"
        assert data_to_base64(data_url) == payload

    def test_passthrough_http_url(self) -> None:
        url = "https://example.com/image.png"
        assert data_to_base64(url) == url


# ---------------------------------------------------------------------------
# data_to_data_url
# ---------------------------------------------------------------------------


class TestDataToDataUrl:
    def test_from_bytes(self) -> None:
        raw = b"\x89PNG"
        result = data_to_data_url(raw, "image/png")
        assert result.startswith("data:image/png;base64,")
        assert base64.b64decode(result.split(",", 1)[1]) == raw

    def test_passthrough_url(self) -> None:
        url = "https://example.com/image.png"
        assert data_to_data_url(url, "image/png") == url


# ---------------------------------------------------------------------------
# split_data_url
# ---------------------------------------------------------------------------


class TestSplitDataUrl:
    def test_valid(self) -> None:
        media_type, content = split_data_url("data:image/png;base64,iVBOR")
        assert media_type == "image/png"
        assert content == "iVBOR"

    def test_non_data_url(self) -> None:
        assert split_data_url("https://example.com") == (None, None)

    def test_malformed(self) -> None:
        assert split_data_url("data:nope") == (None, None)


# ---------------------------------------------------------------------------
# Image detection
# ---------------------------------------------------------------------------


class TestGif:
    def test_from_bytes(self) -> None:
        assert detect_image_media_type(bytes([0x47, 0x49, 0x46])) == "image/gif"

    def test_from_base64(self) -> None:
        assert (
            detect_image_media_type(
                base64.b64encode(bytes([0x47, 0x49, 0x46])).decode()
            )
            == "image/gif"
        )


class TestPng:
    def test_from_bytes(self) -> None:
        assert detect_image_media_type(bytes([0x89, 0x50, 0x4E, 0x47])) == "image/png"

    def test_from_base64(self) -> None:
        assert (
            detect_image_media_type(
                base64.b64encode(bytes([0x89, 0x50, 0x4E, 0x47])).decode()
            )
            == "image/png"
        )


class TestJpeg:
    def test_from_bytes(self) -> None:
        assert detect_image_media_type(bytes([0xFF, 0xD8, 0xFF])) == "image/jpeg"

    def test_from_base64(self) -> None:
        assert (
            detect_image_media_type(
                base64.b64encode(bytes([0xFF, 0xD8, 0xFF])).decode()
            )
            == "image/jpeg"
        )


class TestWebp:
    _RIFF_WEBP = bytes(
        [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50]
    )

    def test_from_bytes(self) -> None:
        assert detect_image_media_type(self._RIFF_WEBP) == "image/webp"

    def test_from_base64(self) -> None:
        assert (
            detect_image_media_type(base64.b64encode(self._RIFF_WEBP).decode())
            == "image/webp"
        )

    def test_riff_wave_not_webp_bytes(self) -> None:
        riff_wave = bytes(
            [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45]
        )
        assert detect_image_media_type(riff_wave) is None

    def test_riff_wave_not_webp_base64(self) -> None:
        riff_wave = bytes(
            [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45]
        )
        assert detect_image_media_type(base64.b64encode(riff_wave).decode()) is None


class TestBmp:
    def test_from_bytes(self) -> None:
        assert detect_image_media_type(bytes([0x42, 0x4D])) == "image/bmp"

    def test_from_base64(self) -> None:
        assert (
            detect_image_media_type(base64.b64encode(bytes([0x42, 0x4D])).decode())
            == "image/bmp"
        )


class TestTiff:
    def test_little_endian_from_bytes(self) -> None:
        assert detect_image_media_type(bytes([0x49, 0x49, 0x2A, 0x00])) == "image/tiff"

    def test_big_endian_from_bytes(self) -> None:
        assert detect_image_media_type(bytes([0x4D, 0x4D, 0x00, 0x2A])) == "image/tiff"

    def test_little_endian_from_base64(self) -> None:
        assert (
            detect_image_media_type(
                base64.b64encode(bytes([0x49, 0x49, 0x2A, 0x00])).decode()
            )
            == "image/tiff"
        )

    def test_big_endian_from_base64(self) -> None:
        assert (
            detect_image_media_type(
                base64.b64encode(bytes([0x4D, 0x4D, 0x00, 0x2A])).decode()
            )
            == "image/tiff"
        )


class TestAvif:
    _AVIF = bytes(
        [0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x61, 0x76, 0x69, 0x66]
    )

    def test_from_bytes(self) -> None:
        assert detect_image_media_type(self._AVIF) == "image/avif"

    def test_from_base64(self) -> None:
        assert (
            detect_image_media_type(base64.b64encode(self._AVIF).decode())
            == "image/avif"
        )


class TestHeic:
    _HEIC = bytes(
        [0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x68, 0x65, 0x69, 0x63]
    )

    def test_from_bytes(self) -> None:
        assert detect_image_media_type(self._HEIC) == "image/heic"

    def test_from_base64(self) -> None:
        assert (
            detect_image_media_type(base64.b64encode(self._HEIC).decode())
            == "image/heic"
        )


# ---------------------------------------------------------------------------
# Audio detection
# ---------------------------------------------------------------------------


class TestMp3:
    def test_from_bytes(self) -> None:
        assert detect_audio_media_type(bytes([0xFF, 0xFB])) == "audio/mpeg"

    def test_from_base64(self) -> None:
        assert (
            detect_audio_media_type(base64.b64encode(bytes([0xFF, 0xFB])).decode())
            == "audio/mpeg"
        )

    def test_with_id3_tags_bytes(self) -> None:
        # ID3v2 header (10 bytes) + MP3 sync bytes
        id3_header = bytes([0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        mp3_data = id3_header + bytes([0xFF, 0xFB])
        assert detect_audio_media_type(mp3_data) == "audio/mpeg"

    def test_with_id3_tags_base64(self) -> None:
        id3_header = bytes([0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        mp3_data = id3_header + bytes([0xFF, 0xFB])
        assert (
            detect_audio_media_type(base64.b64encode(mp3_data).decode()) == "audio/mpeg"
        )


class TestWav:
    _RIFF_WAVE = bytes(
        [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45]
    )

    def test_from_bytes(self) -> None:
        assert detect_audio_media_type(self._RIFF_WAVE) == "audio/wav"

    def test_from_base64(self) -> None:
        assert (
            detect_audio_media_type(base64.b64encode(self._RIFF_WAVE).decode())
            == "audio/wav"
        )

    def test_riff_webp_not_wav_bytes(self) -> None:
        riff_webp = bytes(
            [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50]
        )
        assert detect_audio_media_type(riff_webp) is None

    def test_riff_webp_not_wav_base64(self) -> None:
        riff_webp = bytes(
            [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50]
        )
        assert detect_audio_media_type(base64.b64encode(riff_webp).decode()) is None


class TestOgg:
    def test_from_bytes(self) -> None:
        assert detect_audio_media_type(b"OggS") == "audio/ogg"

    def test_from_base64(self) -> None:
        assert (
            detect_audio_media_type(base64.b64encode(b"OggS").decode()) == "audio/ogg"
        )


class TestFlac:
    def test_from_bytes(self) -> None:
        assert detect_audio_media_type(b"fLaC") == "audio/flac"

    def test_from_base64(self) -> None:
        assert (
            detect_audio_media_type(base64.b64encode(b"fLaC").decode()) == "audio/flac"
        )


class TestAac:
    def test_from_bytes(self) -> None:
        assert detect_audio_media_type(bytes([0x40, 0x15, 0x00, 0x00])) == "audio/aac"

    def test_from_base64(self) -> None:
        assert (
            detect_audio_media_type(
                base64.b64encode(bytes([0x40, 0x15, 0x00, 0x00])).decode()
            )
            == "audio/aac"
        )


class TestMp4Audio:
    # The audio/mp4 signature starts at the `ftyp` atom directly (no box size prefix).
    _FTYP = bytes([0x66, 0x74, 0x79, 0x70])

    def test_from_bytes(self) -> None:
        assert detect_audio_media_type(self._FTYP) == "audio/mp4"

    def test_from_base64(self) -> None:
        assert (
            detect_audio_media_type(base64.b64encode(self._FTYP).decode())
            == "audio/mp4"
        )


class TestWebmAudio:
    _WEBM = bytes([0x1A, 0x45, 0xDF, 0xA3])

    def test_from_bytes(self) -> None:
        assert detect_audio_media_type(self._WEBM) == "audio/webm"

    def test_from_base64(self) -> None:
        assert (
            detect_audio_media_type(base64.b64encode(self._WEBM).decode())
            == "audio/webm"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_image_format(self) -> None:
        assert detect_image_media_type(bytes([0x00, 0x01, 0x02, 0x03])) is None

    def test_unknown_audio_format(self) -> None:
        assert detect_audio_media_type(bytes([0x00, 0x01, 0x02, 0x03])) is None

    def test_empty_bytes_image(self) -> None:
        assert detect_image_media_type(b"") is None

    def test_empty_bytes_audio(self) -> None:
        assert detect_audio_media_type(b"") is None

    def test_short_bytes_image(self) -> None:
        assert detect_image_media_type(bytes([0x89])) is None

    def test_short_bytes_audio(self) -> None:
        assert detect_audio_media_type(bytes([0xFF])) is None
