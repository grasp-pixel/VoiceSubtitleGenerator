"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest

from src.config import AppConfig, ConfigManager
from src.models import Segment, SpeakerMapping, Word


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict() -> dict:
    """Sample configuration dictionary."""
    return {
        "language": "ko",
        "models": {
            "path": "./models",
            "auto_download": True,
        },
        "whisperx": {
            "model_size": "large-v3",
            "device": "cuda",
            "compute_type": "float16",
            "batch_size": 16,
            "hf_token": "",
            "diarization": {
                "enabled": True,
                "min_speakers": 1,
                "max_speakers": 10,
            },
        },
        "translation": {
            "model_path": "./models/qwen.gguf",
            "n_gpu_layers": -1,
            "n_ctx": 4096,
            "max_tokens": 256,
            "temperature": 0.3,
        },
        "subtitle": {
            "default_format": "srt",
            "include_speaker": True,
            "include_original": False,
            "ass": {
                "video_width": 1920,
                "video_height": 1080,
                "default_font": "Malgun Gothic",
                "default_size": 48,
            },
        },
        "processing": {
            "workers": 1,
            "normalize_audio": True,
            "target_sample_rate": 16000,
        },
        "ui": {
            "width": 1200,
            "height": 900,
            "font_size": 14,
            "remember_position": True,
        },
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_dict) -> Path:
    """Create a sample configuration file."""
    import yaml

    config_path = temp_dir / "settings.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(sample_config_dict, f, allow_unicode=True)

    return config_path


@pytest.fixture
def config_manager(sample_config_file) -> ConfigManager:
    """Create a ConfigManager with sample config."""
    return ConfigManager(str(sample_config_file))


@pytest.fixture
def app_config(config_manager) -> AppConfig:
    """Load application config."""
    return config_manager.load()


@pytest.fixture
def sample_segments() -> list[Segment]:
    """Sample transcript segments."""
    return [
        Segment(
            start=0.0,
            end=2.5,
            original_text="お前、何やってんだよ！",
            translated_text="너, 뭐 하는 거야!",
            speaker_id="SPEAKER_00",
            speaker_name="히로인",
            words=[
                Word(word="お前", start=0.0, end=0.5),
                Word(word="何", start=0.6, end=0.8),
                Word(word="やってんだよ", start=0.9, end=2.0),
            ],
        ),
        Segment(
            start=3.0,
            end=5.5,
            original_text="ちょっと待ってくれ",
            translated_text="잠깐만 기다려줘",
            speaker_id="SPEAKER_01",
            speaker_name="주인공",
            words=[
                Word(word="ちょっと", start=3.0, end=3.5),
                Word(word="待ってくれ", start=3.6, end=5.0),
            ],
        ),
        Segment(
            start=6.0,
            end=9.0,
            original_text="わかった、行こう",
            translated_text="알았어, 가자",
            speaker_id="SPEAKER_00",
            speaker_name="히로인",
            words=[
                Word(word="わかった", start=6.0, end=7.0),
                Word(word="行こう", start=7.5, end=8.5),
            ],
        ),
    ]


@pytest.fixture
def sample_speaker_mapping() -> dict[str, SpeakerMapping]:
    """Sample speaker mapping."""
    return {
        "SPEAKER_00": SpeakerMapping(
            speaker_id="SPEAKER_00",
            name="히로인",
            color="FF69B4",
        ),
        "SPEAKER_01": SpeakerMapping(
            speaker_id="SPEAKER_01",
            name="주인공",
            color="00BFFF",
        ),
    }


@pytest.fixture
def sample_audio_file(temp_dir) -> Path:
    """Create a sample audio file (silent WAV)."""
    import wave
    import struct

    audio_path = temp_dir / "sample.wav"

    # Create a 1-second silent WAV file
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)

    with wave.open(str(audio_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        # Write silence (zeros)
        for _ in range(num_samples):
            wav_file.writeframes(struct.pack("h", 0))

    return audio_path
