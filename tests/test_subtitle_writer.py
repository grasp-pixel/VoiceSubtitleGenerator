"""Tests for subtitle writer module."""

import pytest
from pathlib import Path

from src.subtitle_writer import SubtitleWriter
from src.models import Segment


class TestSubtitleWriter:
    """Tests for SubtitleWriter."""

    def test_writer_initialization(self):
        """Test SubtitleWriter initialization."""
        writer = SubtitleWriter()

        assert writer is not None
        assert writer.include_speaker is True
        assert writer.include_original is False

    def test_writer_custom_options(self):
        """Test SubtitleWriter with custom options."""
        writer = SubtitleWriter(
            include_speaker=False,
            include_original=True,
        )

        assert writer.include_speaker is False
        assert writer.include_original is True

    def test_supported_formats(self):
        """Test supported subtitle formats."""
        formats = SubtitleWriter.supported_formats()

        assert isinstance(formats, list)
        assert "srt" in formats
        assert "ass" in formats
        assert "vtt" in formats

    def test_write_srt(self, sample_segments, temp_dir):
        """Test writing SRT file."""
        writer = SubtitleWriter()
        output_path = temp_dir / "output.srt"

        writer.write(sample_segments, str(output_path), format="srt")

        assert output_path.exists()

        content = output_path.read_text(encoding="utf-8")
        assert "너, 뭐 하는 거야!" in content
        assert "잠깐만 기다려줘" in content
        assert "00:00:00" in content

    def test_write_vtt(self, sample_segments, temp_dir):
        """Test writing VTT file."""
        writer = SubtitleWriter()
        output_path = temp_dir / "output.vtt"

        writer.write(sample_segments, str(output_path), format="vtt")

        assert output_path.exists()

        content = output_path.read_text(encoding="utf-8")
        assert "WEBVTT" in content

    def test_write_ass(self, sample_segments, temp_dir):
        """Test writing ASS file."""
        writer = SubtitleWriter()
        output_path = temp_dir / "output.ass"

        writer.write(sample_segments, str(output_path), format="ass")

        assert output_path.exists()

        content = output_path.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content

    def test_write_with_speaker(self, sample_segments, temp_dir):
        """Test writing with speaker names included."""
        writer = SubtitleWriter(include_speaker=True)
        output_path = temp_dir / "output.srt"

        writer.write(sample_segments, str(output_path), format="srt")

        content = output_path.read_text(encoding="utf-8")
        assert "[히로인]" in content or "히로인" in content

    def test_write_without_speaker(self, sample_segments, temp_dir):
        """Test writing without speaker names."""
        writer = SubtitleWriter(include_speaker=False)
        output_path = temp_dir / "output.srt"

        writer.write(sample_segments, str(output_path), format="srt")

        content = output_path.read_text(encoding="utf-8")
        # Should still have the translated text
        assert "너, 뭐 하는 거야!" in content

    def test_write_with_original(self, sample_segments, temp_dir):
        """Test writing with original text included."""
        writer = SubtitleWriter(include_original=True, include_speaker=False)
        output_path = temp_dir / "output.srt"

        writer.write(sample_segments, str(output_path), format="srt")

        content = output_path.read_text(encoding="utf-8")
        # Should have both original and translated
        assert "너, 뭐 하는 거야!" in content

    def test_write_empty_segments(self, temp_dir):
        """Test writing with empty segments."""
        writer = SubtitleWriter()
        output_path = temp_dir / "output.srt"

        writer.write([], str(output_path), format="srt")

        assert output_path.exists()

    def test_auto_detect_format(self, sample_segments, temp_dir):
        """Test auto-detecting format from extension."""
        writer = SubtitleWriter()

        # SRT
        srt_path = temp_dir / "output.srt"
        writer.write(sample_segments, str(srt_path))
        assert srt_path.exists()

        # ASS
        ass_path = temp_dir / "output.ass"
        writer.write(sample_segments, str(ass_path))
        assert ass_path.exists()

        # VTT
        vtt_path = temp_dir / "output.vtt"
        writer.write(sample_segments, str(vtt_path))
        assert vtt_path.exists()
