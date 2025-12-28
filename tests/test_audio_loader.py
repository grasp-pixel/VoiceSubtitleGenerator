"""Tests for audio loader module."""

import pytest

from src.audio_loader import AudioLoader


class TestAudioLoader:
    """Tests for AudioLoader."""

    def test_supported_formats(self):
        """Test getting supported formats."""
        formats = AudioLoader.supported_formats()

        assert isinstance(formats, list)
        assert "mp3" in formats
        assert "wav" in formats
        assert "flac" in formats
        assert "m4a" in formats
        assert "ogg" in formats

    def test_is_supported_valid_formats(self):
        """Test checking valid audio formats."""
        assert AudioLoader.is_supported("audio.mp3") is True
        assert AudioLoader.is_supported("audio.wav") is True
        assert AudioLoader.is_supported("audio.flac") is True
        assert AudioLoader.is_supported("audio.m4a") is True
        assert AudioLoader.is_supported("audio.ogg") is True

    def test_is_supported_invalid_formats(self):
        """Test checking invalid audio formats."""
        assert AudioLoader.is_supported("document.pdf") is False
        assert AudioLoader.is_supported("image.png") is False
        assert AudioLoader.is_supported("video.mp4") is False
        assert AudioLoader.is_supported("text.txt") is False

    def test_is_supported_case_insensitive(self):
        """Test that format checking is case insensitive."""
        assert AudioLoader.is_supported("audio.MP3") is True
        assert AudioLoader.is_supported("audio.WAV") is True
        assert AudioLoader.is_supported("audio.Flac") is True

    def test_load_audio_file(self, sample_audio_file):
        """Test loading an audio file."""
        loader = AudioLoader()
        audio, sample_rate = loader.load(str(sample_audio_file))

        assert audio is not None
        assert sample_rate == 16000

    def test_get_audio_info(self, sample_audio_file):
        """Test getting audio file info."""
        loader = AudioLoader()
        info = loader.get_info(str(sample_audio_file))

        assert info is not None
        assert info.path == str(sample_audio_file)
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert info.duration > 0

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading a non-existent file."""
        loader = AudioLoader()
        non_existent = temp_dir / "nonexistent.wav"

        with pytest.raises(Exception):
            loader.load(str(non_existent))

    def test_loader_initialization(self):
        """Test AudioLoader initialization."""
        loader = AudioLoader(normalize=True, target_sample_rate=16000)

        assert loader.normalize is True
        assert loader.target_sample_rate == 16000

    def test_loader_default_values(self):
        """Test AudioLoader default values."""
        loader = AudioLoader()

        assert loader.normalize is False
        assert loader.target_sample_rate is None
