"""Tests for media data helpers and magic-byte media type detection.

Covers ``is_url``, ``data_to_base64``, ``data_to_data_url``,
``split_data_url``, ``detect_image_media_type``, ``detect_audio_media_type``,
and edge cases.
"""

from __future__ import annotations

import base64

from ai.types import media

# ---------------------------------------------------------------------------
# is_url
# ---------------------------------------------------------------------------


class TestIsUrl:
    def test_http(self) -> None:
        assert media.is_url("https://example.com/img.png") is True
        assert media.is_url("http://example.com/img.png") is True

    def test_data(self) -> None:
        assert media.is_url("data:image/png;base64,iVBOR") is True

    def test_base64(self) -> None:
        assert media.is_url("iVBORw0KGgo=") is False


# ---------------------------------------------------------------------------
# data_to_base64
# ---------------------------------------------------------------------------


class TestDataToBase64:
    def test_bytes(self) -> None:
        raw = b"\x89PNG"
        result = media.data_to_base64(raw)
        assert base64.b64decode(result) == raw

    def test_passthrough(self) -> None:
        b64 = base64.b64encode(b"hello").decode()
        assert media.data_to_base64(b64) == b64

    def test_extracts_from_data_url(self) -> None:
        payload = base64.b64encode(b"hello").decode()
        data_url = f"data:image/png;base64,{payload}"
        assert media.data_to_base64(data_url) == payload

    def test_passthrough_http_url(self) -> None:
        url = "https://example.com/image.png"
        assert media.data_to_base64(url) == url


# ---------------------------------------------------------------------------
# data_to_data_url
# ---------------------------------------------------------------------------


class TestDataToDataUrl:
    def test_from_bytes(self) -> None:
        raw = b"\x89PNG"
        result = media.data_to_data_url(raw, "image/png")
        assert result.startswith("data:image/png;base64,")
        assert base64.b64decode(result.split(",", 1)[1]) == raw

    def test_passthrough_url(self) -> None:
        url = "https://example.com/image.png"
        assert media.data_to_data_url(url, "image/png") == url


# ---------------------------------------------------------------------------
# split_data_url
# ---------------------------------------------------------------------------


class TestSplitDataUrl:
    def test_valid(self) -> None:
        mt, content = media.split_data_url("data:image/png;base64,iVBOR")
        assert mt == "image/png"
        assert content == "iVBOR"

    def test_non_data_url(self) -> None:
        assert media.split_data_url("https://example.com") == (None, None)

    def test_malformed(self) -> None:
        assert media.split_data_url("data:nope") == (None, None)


# ---------------------------------------------------------------------------
# Image detection -- bytes input (one per format)
# ---------------------------------------------------------------------------

_RIFF_WEBP = bytes(
    [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50]
)
_RIFF_WAVE = bytes(
    [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45]
)
_AVIF = bytes([0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x61, 0x76, 0x69, 0x66])
_HEIC = bytes([0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x68, 0x65, 0x69, 0x63])


class TestImageDetection:
    def test_gif(self) -> None:
        assert media.detect_image_media_type(bytes([0x47, 0x49, 0x46])) == "image/gif"

    def test_png(self) -> None:
        assert (
            media.detect_image_media_type(bytes([0x89, 0x50, 0x4E, 0x47]))
            == "image/png"
        )

    def test_jpeg(self) -> None:
        assert media.detect_image_media_type(bytes([0xFF, 0xD8, 0xFF])) == "image/jpeg"

    def test_webp(self) -> None:
        assert media.detect_image_media_type(_RIFF_WEBP) == "image/webp"

    def test_bmp(self) -> None:
        assert media.detect_image_media_type(bytes([0x42, 0x4D])) == "image/bmp"

    def test_tiff_little_endian(self) -> None:
        assert (
            media.detect_image_media_type(bytes([0x49, 0x49, 0x2A, 0x00]))
            == "image/tiff"
        )

    def test_tiff_big_endian(self) -> None:
        assert (
            media.detect_image_media_type(bytes([0x4D, 0x4D, 0x00, 0x2A]))
            == "image/tiff"
        )

    def test_avif(self) -> None:
        assert media.detect_image_media_type(_AVIF) == "image/avif"

    def test_heic(self) -> None:
        assert media.detect_image_media_type(_HEIC) == "image/heic"

    def test_riff_wave_not_webp(self) -> None:
        """RIFF+WAVE must NOT match as an image format."""
        assert media.detect_image_media_type(_RIFF_WAVE) is None


# ---------------------------------------------------------------------------
# Image detection -- base64 input (representative, not per-format)
# ---------------------------------------------------------------------------


class TestImageDetectionBase64:
    def test_png_from_base64(self) -> None:
        """Base64 path decodes correctly before matching."""
        b64 = base64.b64encode(bytes([0x89, 0x50, 0x4E, 0x47])).decode()
        assert media.detect_image_media_type(b64) == "image/png"

    def test_webp_from_base64(self) -> None:
        """RIFF container in base64 still detects correctly."""
        b64 = base64.b64encode(_RIFF_WEBP).decode()
        assert media.detect_image_media_type(b64) == "image/webp"


# ---------------------------------------------------------------------------
# Audio detection -- bytes input (one per format)
# ---------------------------------------------------------------------------


class TestAudioDetection:
    def test_mp3(self) -> None:
        assert media.detect_audio_media_type(bytes([0xFF, 0xFB])) == "audio/mpeg"

    def test_mp3_with_id3_tags(self) -> None:
        id3_header = bytes([0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        assert media.detect_audio_media_type(id3_header + bytes([0xFF, 0xFB])) == (
            "audio/mpeg"
        )

    def test_wav(self) -> None:
        assert media.detect_audio_media_type(_RIFF_WAVE) == "audio/wav"

    def test_ogg(self) -> None:
        assert media.detect_audio_media_type(b"OggS") == "audio/ogg"

    def test_flac(self) -> None:
        assert media.detect_audio_media_type(b"fLaC") == "audio/flac"

    def test_aac(self) -> None:
        assert (
            media.detect_audio_media_type(bytes([0x40, 0x15, 0x00, 0x00]))
            == "audio/aac"
        )

    def test_mp4(self) -> None:
        assert (
            media.detect_audio_media_type(bytes([0x66, 0x74, 0x79, 0x70]))
            == "audio/mp4"
        )

    def test_webm(self) -> None:
        assert (
            media.detect_audio_media_type(bytes([0x1A, 0x45, 0xDF, 0xA3]))
            == "audio/webm"
        )

    def test_riff_webp_not_wav(self) -> None:
        """RIFF+WEBP must NOT match as an audio format."""
        assert media.detect_audio_media_type(_RIFF_WEBP) is None


# ---------------------------------------------------------------------------
# Audio detection -- base64 input (representative)
# ---------------------------------------------------------------------------


class TestAudioDetectionBase64:
    def test_wav_from_base64(self) -> None:
        b64 = base64.b64encode(_RIFF_WAVE).decode()
        assert media.detect_audio_media_type(b64) == "audio/wav"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_image_format(self) -> None:
        assert media.detect_image_media_type(bytes([0x00, 0x01, 0x02, 0x03])) is None

    def test_unknown_audio_format(self) -> None:
        assert media.detect_audio_media_type(bytes([0x00, 0x01, 0x02, 0x03])) is None

    def test_empty_bytes_image(self) -> None:
        assert media.detect_image_media_type(b"") is None

    def test_empty_bytes_audio(self) -> None:
        assert media.detect_audio_media_type(b"") is None

    def test_short_bytes_image(self) -> None:
        assert media.detect_image_media_type(bytes([0x89])) is None

    def test_short_bytes_audio(self) -> None:
        assert media.detect_audio_media_type(bytes([0xFF])) is None
