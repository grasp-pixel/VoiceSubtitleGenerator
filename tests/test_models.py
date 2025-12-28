"""Tests for data models."""

import pytest

from src.models import (
    Segment,
    Word,
    ProcessResult,
    AudioInfo,
)


class TestWord:
    """Tests for Word model."""

    def test_word_creation(self):
        """Test creating a Word."""
        word = Word(word="テスト", start=0.0, end=1.0)

        assert word.word == "テスト"
        assert word.start == 0.0
        assert word.end == 1.0
        assert word.confidence is None

    def test_word_with_confidence(self):
        """Test creating a Word with confidence."""
        word = Word(word="テスト", start=0.0, end=1.0, confidence=0.95)

        assert word.confidence == 0.95

    def test_word_duration(self):
        """Test word duration property."""
        word = Word(word="テスト", start=1.0, end=2.5)

        assert word.end - word.start == 1.5


class TestSegment:
    """Tests for Segment model."""

    def test_segment_creation(self):
        """Test creating a Segment."""
        segment = Segment(
            start=0.0,
            end=2.5,
            original_text="こんにちは",
        )

        assert segment.start == 0.0
        assert segment.end == 2.5
        assert segment.original_text == "こんにちは"
        assert segment.translated_text == ""

    def test_segment_with_all_fields(self):
        """Test creating a Segment with all fields."""
        words = [
            Word(word="こん", start=0.0, end=0.5),
            Word(word="にち", start=0.5, end=1.0),
            Word(word="は", start=1.0, end=1.5),
        ]

        segment = Segment(
            start=0.0,
            end=2.5,
            original_text="こんにちは",
            translated_text="안녕하세요",
            words=words,
        )

        assert segment.translated_text == "안녕하세요"
        assert len(segment.words) == 3

    def test_segment_duration(self):
        """Test segment duration calculation."""
        segment = Segment(
            start=1.5,
            end=4.0,
            original_text="テスト",
        )

        assert segment.end - segment.start == 2.5


class TestProcessResult:
    """Tests for ProcessResult model."""

    def test_successful_result(self, sample_segments):
        """Test creating a successful ProcessResult."""
        result = ProcessResult(
            success=True,
            segments=sample_segments,
            output_path="output/test.srt",
        )

        assert result.success is True
        assert len(result.segments) == 3
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed ProcessResult."""
        result = ProcessResult(
            success=False,
            segments=[],
            error="Audio file not found",
        )

        assert result.success is False
        assert len(result.segments) == 0
        assert result.error == "Audio file not found"

    def test_result_duration(self, sample_segments):
        """Test calculating result duration."""
        result = ProcessResult(
            success=True,
            segments=sample_segments,
        )

        # Duration should be from first segment start to last segment end
        if result.segments:
            duration = result.segments[-1].end - result.segments[0].start
            assert duration == 9.0  # 9.0 - 0.0


class TestAudioInfo:
    """Tests for AudioInfo model."""

    def test_audio_info_creation(self):
        """Test creating AudioInfo."""
        info = AudioInfo(
            path="/path/to/audio.mp3",
            duration=120.5,
            sample_rate=44100,
            channels=2,
            format="mp3",
        )

        assert info.path == "/path/to/audio.mp3"
        assert info.duration == 120.5
        assert info.sample_rate == 44100
        assert info.channels == 2
        assert info.format == "mp3"

    def test_audio_info_defaults(self):
        """Test AudioInfo with minimal fields."""
        info = AudioInfo(
            path="/path/to/audio.wav",
            duration=60.0,
            sample_rate=16000,
            channels=1,
        )

        assert info.format is None
