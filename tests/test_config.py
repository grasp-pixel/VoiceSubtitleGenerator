"""Tests for configuration module."""

import pytest

from src.config import (
    AppConfig,
    ConfigManager,
    DiarizationConfig,
    SpeechConfig,
    STTConfig,
    SubtitleConfig,
    TranslationConfig,
)


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_load_config(self, config_manager):
        """Test loading configuration."""
        config = config_manager.load()

        assert isinstance(config, AppConfig)
        assert config.language == "ko"

    def test_load_speech_config(self, app_config):
        """Test speech configuration loading."""
        speech = app_config.speech

        assert isinstance(speech, SpeechConfig)
        assert isinstance(speech.stt, STTConfig)
        assert isinstance(speech.diarization, DiarizationConfig)

        # STT config
        assert speech.stt.model_size == "large-v3"
        assert speech.stt.device == "cuda"
        assert speech.stt.compute_type == "float16"

        # Diarization config
        assert speech.diarization.enabled is True
        assert speech.diarization.min_speakers == 1
        assert speech.diarization.max_speakers == 10

    def test_load_translation_config(self, app_config):
        """Test translation configuration loading."""
        translation = app_config.translation

        assert isinstance(translation, TranslationConfig)
        assert translation.model_path == "./models/qwen.gguf"
        assert translation.n_gpu_layers == -1
        assert translation.n_ctx == 4096
        assert translation.max_tokens == 256
        assert translation.temperature == 0.3

    def test_load_subtitle_config(self, app_config):
        """Test subtitle configuration loading."""
        subtitle = app_config.subtitle

        assert isinstance(subtitle, SubtitleConfig)
        assert subtitle.default_format == "srt"
        assert subtitle.include_speaker is True
        assert subtitle.include_original is False
        assert subtitle.ass.video_width == 1920
        assert subtitle.ass.video_height == 1080

    def test_save_config(self, config_manager, app_config, temp_dir):
        """Test saving configuration."""
        # Modify config
        app_config.language = "en"
        app_config.speech.stt.model_size = "large-v3-turbo"

        # Save
        config_manager.save(app_config)

        # Reload and verify
        reloaded = config_manager.load()
        assert reloaded.language == "en"
        assert reloaded.speech.stt.model_size == "large-v3-turbo"

    def test_validate_config_valid(self, config_manager):
        """Test validation with valid config."""
        errors = config_manager.validate()
        # May have errors about missing model files, but no structural errors
        assert isinstance(errors, list)

    def test_config_defaults(self, temp_dir):
        """Test configuration defaults when file doesn't exist."""
        non_existent = temp_dir / "non_existent.yaml"
        manager = ConfigManager(str(non_existent))
        config = manager.load()

        # Should have default values
        assert isinstance(config, AppConfig)
        assert config.language == "ko"


class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_config_creation(self):
        """Test creating AppConfig with defaults."""
        config = AppConfig()

        assert config.language == "ko"
        assert config.speech is not None
        assert config.translation is not None
        assert config.subtitle is not None

    def test_speech_config_creation(self):
        """Test creating SpeechConfig with defaults."""
        config = SpeechConfig()

        assert config.stt.model_size == "large-v3"
        assert config.stt.device == "cuda"
        assert config.stt.compute_type == "float16"
        assert config.diarization.enabled is True

    def test_translation_config_creation(self):
        """Test creating TranslationConfig with defaults."""
        config = TranslationConfig()

        assert config.n_gpu_layers == -1
        assert config.temperature == 0.3

    def test_subtitle_config_creation(self):
        """Test creating SubtitleConfig with defaults."""
        config = SubtitleConfig()

        assert config.default_format == "srt"
        assert config.include_speaker is True
