"""Tests for magic-byte media type detection.

Ported from: .reference/ai/packages/ai/src/util/detect-media-type.test.ts
"""

from __future__ import annotations

import base64

from vercel_ai_sdk.models.core.media.detect import (
    AUDIO_SIGNATURES,
    IMAGE_SIGNATURES,
    detect_media_type,
)

# ---------------------------------------------------------------------------
# Image detection
# ---------------------------------------------------------------------------


class TestGif:
    def test_detect_gif_from_bytes(self) -> None:
        data = bytes([0x47, 0x49, 0x46, 0xFF, 0xFF])
        assert detect_media_type(data, IMAGE_SIGNATURES) == "image/gif"

    def test_detect_gif_from_base64(self) -> None:
        assert detect_media_type("R0lGabc123", IMAGE_SIGNATURES) == "image/gif"


class TestPng:
    def test_detect_png_from_bytes(self) -> None:
        data = bytes([0x89, 0x50, 0x4E, 0x47, 0xFF, 0xFF])
        assert detect_media_type(data, IMAGE_SIGNATURES) == "image/png"

    def test_detect_png_from_base64(self) -> None:
        assert detect_media_type("iVBORwabc123", IMAGE_SIGNATURES) == "image/png"


class TestJpeg:
    def test_detect_jpeg_from_bytes(self) -> None:
        data = bytes([0xFF, 0xD8, 0xFF, 0xFF])
        assert detect_media_type(data, IMAGE_SIGNATURES) == "image/jpeg"

    def test_detect_jpeg_from_base64(self) -> None:
        assert detect_media_type("/9j/abc123", IMAGE_SIGNATURES) == "image/jpeg"


class TestWebp:
    def test_detect_webp_from_bytes(self) -> None:
        # RIFF + 4 bytes (file size) + WEBP + VP8 data
        data = bytes(
            [
                0x52,
                0x49,
                0x46,
                0x46,  # "RIFF"
                0x24,
                0x00,
                0x00,
                0x00,  # file size (wildcard in sig)
                0x57,
                0x45,
                0x42,
                0x50,  # "WEBP"
                0x56,
                0x50,
                0x38,
                0x20,  # "VP8 " (trailing data)
            ]
        )
        assert detect_media_type(data, IMAGE_SIGNATURES) == "image/webp"

    def test_detect_webp_from_base64(self) -> None:
        data = bytes(
            [
                0x52,
                0x49,
                0x46,
                0x46,
                0x24,
                0x00,
                0x00,
                0x00,
                0x57,
                0x45,
                0x42,
                0x50,
                0x56,
                0x50,
                0x38,
                0x20,
            ]
        )
        b64 = base64.b64encode(data).decode()
        assert detect_media_type(b64, IMAGE_SIGNATURES) == "image/webp"

    def test_riff_audio_not_detected_as_webp_bytes(self) -> None:
        """RIFF + WAVE should NOT match WebP."""
        data = bytes(
            [
                0x52,
                0x49,
                0x46,
                0x46,
                0x24,
                0x00,
                0x00,
                0x00,
                0x57,
                0x41,
                0x56,
                0x45,  # "WAVE", not "WEBP"
            ]
        )
        assert detect_media_type(data, IMAGE_SIGNATURES) is None

    def test_riff_audio_not_detected_as_webp_base64(self) -> None:
        data = bytes(
            [
                0x52,
                0x49,
                0x46,
                0x46,
                0x24,
                0x00,
                0x00,
                0x00,
                0x57,
                0x41,
                0x56,
                0x45,
            ]
        )
        b64 = base64.b64encode(data).decode()
        assert detect_media_type(b64, IMAGE_SIGNATURES) is None


class TestBmp:
    def test_detect_bmp_from_bytes(self) -> None:
        data = bytes([0x42, 0x4D, 0xFF, 0xFF])
        assert detect_media_type(data, IMAGE_SIGNATURES) == "image/bmp"

    def test_detect_bmp_from_base64(self) -> None:
        data = bytes([0x42, 0x4D, 0xFF, 0xFF])
        b64 = base64.b64encode(data).decode()
        assert detect_media_type(b64, IMAGE_SIGNATURES) == "image/bmp"


class TestTiff:
    def test_detect_tiff_le_from_bytes(self) -> None:
        data = bytes([0x49, 0x49, 0x2A, 0x00, 0xFF])
        assert detect_media_type(data, IMAGE_SIGNATURES) == "image/tiff"

    def test_detect_tiff_le_from_base64(self) -> None:
        assert detect_media_type("SUkqAAabc123", IMAGE_SIGNATURES) == "image/tiff"

    def test_detect_tiff_be_from_bytes(self) -> None:
        data = bytes([0x4D, 0x4D, 0x00, 0x2A, 0xFF])
        assert detect_media_type(data, IMAGE_SIGNATURES) == "image/tiff"

    def test_detect_tiff_be_from_base64(self) -> None:
        assert detect_media_type("TU0AKgabc123", IMAGE_SIGNATURES) == "image/tiff"


class TestAvif:
    def test_detect_avif_from_bytes(self) -> None:
        data = bytes(
            [
                0x00,
                0x00,
                0x00,
                0x20,
                0x66,
                0x74,
                0x79,
                0x70,
                0x61,
                0x76,
                0x69,
                0x66,
                0xFF,
            ]
        )
        assert detect_media_type(data, IMAGE_SIGNATURES) == "image/avif"

    def test_detect_avif_from_base64(self) -> None:
        assert (
            detect_media_type("AAAAIGZ0eXBhdmlmabc123", IMAGE_SIGNATURES)
            == "image/avif"
        )


class TestHeic:
    def test_detect_heic_from_bytes(self) -> None:
        data = bytes(
            [
                0x00,
                0x00,
                0x00,
                0x20,
                0x66,
                0x74,
                0x79,
                0x70,
                0x68,
                0x65,
                0x69,
                0x63,
                0xFF,
            ]
        )
        assert detect_media_type(data, IMAGE_SIGNATURES) == "image/heic"

    def test_detect_heic_from_base64(self) -> None:
        assert (
            detect_media_type("AAAAIGZ0eXBoZWljabc123", IMAGE_SIGNATURES)
            == "image/heic"
        )


# ---------------------------------------------------------------------------
# Audio detection
# ---------------------------------------------------------------------------


class TestMp3:
    def test_detect_mp3_from_bytes(self) -> None:
        data = bytes([0xFF, 0xFB])
        assert detect_media_type(data, AUDIO_SIGNATURES) == "audio/mpeg"

    def test_detect_mp3_from_base64(self) -> None:
        assert detect_media_type("//s=", AUDIO_SIGNATURES) == "audio/mpeg"

    def test_detect_mp3_with_id3v2_tags_from_bytes(self) -> None:
        """ID3v2 header (10 bytes tag, size=4) followed by MP3 frame."""
        data = bytes(
            [
                0x49,
                0x44,
                0x33,  # "ID3"
                0x04,
                0x00,  # version
                0x00,  # flags
                0x00,
                0x00,
                0x00,
                0x04,  # size = 4 (syncsafe)
                0x00,
                0x00,
                0x00,
                0x00,  # 4 bytes of tag data
                0xFF,
                0xFB,  # MP3 frame sync
                0x90,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ]
        )
        assert detect_media_type(data, AUDIO_SIGNATURES) == "audio/mpeg"

    def test_detect_mp3_with_id3v2_tags_from_base64(self) -> None:
        data = bytes(
            [
                0x49,
                0x44,
                0x33,
                0x04,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x04,
                0x00,
                0x00,
                0x00,
                0x00,
                0xFF,
                0xFB,
                0x90,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ]
        )
        b64 = base64.b64encode(data).decode()
        assert detect_media_type(b64, AUDIO_SIGNATURES) == "audio/mpeg"


class TestWav:
    def test_detect_wav_from_bytes(self) -> None:
        data = bytes(
            [
                0x52,
                0x49,
                0x46,
                0x46,  # "RIFF"
                0x24,
                0x00,
                0x00,
                0x00,  # file size
                0x57,
                0x41,
                0x56,
                0x45,  # "WAVE"
            ]
        )
        assert detect_media_type(data, AUDIO_SIGNATURES) == "audio/wav"

    def test_detect_wav_from_base64(self) -> None:
        data = bytes(
            [
                0x52,
                0x49,
                0x46,
                0x46,
                0x24,
                0x00,
                0x00,
                0x00,
                0x57,
                0x41,
                0x56,
                0x45,
            ]
        )
        b64 = base64.b64encode(data).decode()
        assert detect_media_type(b64, AUDIO_SIGNATURES) == "audio/wav"

    def test_webp_not_detected_as_wav_bytes(self) -> None:
        """RIFF + WEBP should NOT match WAV."""
        data = bytes(
            [
                0x52,
                0x49,
                0x46,
                0x46,
                0x24,
                0x00,
                0x00,
                0x00,
                0x57,
                0x45,
                0x42,
                0x50,  # "WEBP", not "WAVE"
            ]
        )
        assert detect_media_type(data, AUDIO_SIGNATURES) is None

    def test_webp_not_detected_as_wav_base64(self) -> None:
        data = bytes(
            [
                0x52,
                0x49,
                0x46,
                0x46,
                0x24,
                0x00,
                0x00,
                0x00,
                0x57,
                0x45,
                0x42,
                0x50,
            ]
        )
        b64 = base64.b64encode(data).decode()
        assert detect_media_type(b64, AUDIO_SIGNATURES) is None


class TestOgg:
    def test_detect_ogg_from_bytes(self) -> None:
        data = bytes([0x4F, 0x67, 0x67, 0x53])
        assert detect_media_type(data, AUDIO_SIGNATURES) == "audio/ogg"

    def test_detect_ogg_from_base64(self) -> None:
        assert detect_media_type("T2dnUw", AUDIO_SIGNATURES) == "audio/ogg"


class TestFlac:
    def test_detect_flac_from_bytes(self) -> None:
        data = bytes([0x66, 0x4C, 0x61, 0x43])
        assert detect_media_type(data, AUDIO_SIGNATURES) == "audio/flac"

    def test_detect_flac_from_base64(self) -> None:
        assert detect_media_type("ZkxhQw", AUDIO_SIGNATURES) == "audio/flac"


class TestAac:
    def test_detect_aac_from_bytes(self) -> None:
        data = bytes([0x40, 0x15, 0x00, 0x00])
        assert detect_media_type(data, AUDIO_SIGNATURES) == "audio/aac"

    def test_detect_aac_from_base64(self) -> None:
        data = bytes([0x40, 0x15, 0x00, 0x00])
        b64 = base64.b64encode(data).decode()
        assert detect_media_type(b64, AUDIO_SIGNATURES) == "audio/aac"


class TestMp4Audio:
    def test_detect_mp4_from_bytes(self) -> None:
        data = bytes([0x66, 0x74, 0x79, 0x70])
        assert detect_media_type(data, AUDIO_SIGNATURES) == "audio/mp4"

    def test_detect_mp4_from_base64(self) -> None:
        assert detect_media_type("ZnR5cA", AUDIO_SIGNATURES) == "audio/mp4"


class TestWebmAudio:
    def test_detect_webm_from_bytes(self) -> None:
        data = bytes([0x1A, 0x45, 0xDF, 0xA3])
        assert detect_media_type(data, AUDIO_SIGNATURES) == "audio/webm"

    def test_detect_webm_from_base64(self) -> None:
        assert detect_media_type("GkXfow==", AUDIO_SIGNATURES) == "audio/webm"


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_image_format(self) -> None:
        data = bytes([0x00, 0x01, 0x02, 0x03])
        assert detect_media_type(data, IMAGE_SIGNATURES) is None

    def test_unknown_audio_format(self) -> None:
        data = bytes([0x00, 0x01, 0x02, 0x03])
        assert detect_media_type(data, AUDIO_SIGNATURES) is None

    def test_empty_bytes_image(self) -> None:
        assert detect_media_type(b"", IMAGE_SIGNATURES) is None

    def test_empty_bytes_audio(self) -> None:
        assert detect_media_type(b"", AUDIO_SIGNATURES) is None

    def test_short_bytes_image(self) -> None:
        """Bytes shorter than longest signature should not crash."""
        data = bytes([0x89, 0x50])  # incomplete PNG
        assert detect_media_type(data, IMAGE_SIGNATURES) is None

    def test_short_bytes_audio(self) -> None:
        data = bytes([0x4F, 0x67])  # incomplete OGG
        assert detect_media_type(data, AUDIO_SIGNATURES) is None

    def test_invalid_base64_image(self) -> None:
        assert detect_media_type("invalid123", IMAGE_SIGNATURES) is None

    def test_invalid_base64_audio(self) -> None:
        assert detect_media_type("invalid123", AUDIO_SIGNATURES) is None
