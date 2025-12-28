"""Configuration management for Voice Subtitle Generator."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Supported Whisper models (large-v3 and turbo only)
SUPPORTED_WHISPER_MODELS = ["large-v3", "large-v3-turbo"]

# Supported diarization backends
SUPPORTED_DIARIZATION_BACKENDS = ["pyannote", "speechbrain"]


@dataclass
class STTConfig:
    """STT (Speech-to-Text) settings."""

    model_size: str = "large-v3-turbo"  # "large-v3" | "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ja"
    beam_size: int = 5
    vad_filter: bool = True


@dataclass
class DiarizationConfig:
    """Speaker diarization settings."""

    enabled: bool = True
    backend: str = "pyannote"  # "pyannote" | "speechbrain"
    model_name: str = "pyannote/speaker-diarization-3.1"
    device: str = "cuda"
    min_speakers: int = 1
    max_speakers: int = 10
    hf_token: str = ""

    def get_hf_token(self) -> str | None:
        """Get HF token from config or environment."""
        token = self.hf_token or os.environ.get("HF_TOKEN", "")
        return token if token else None


@dataclass
class SpeechConfig:
    """Combined speech processing settings (STT + Diarization)."""

    stt: STTConfig = field(default_factory=STTConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)


# Legacy aliases for backward compatibility
WhisperXDiarizationConfig = DiarizationConfig
WhisperXConfig = STTConfig


@dataclass
class TranslationModelPreset:
    """Translation model preset definition."""

    name: str  # Display name
    filename: str  # Expected filename in models folder
    hf_repo: str  # HuggingFace repo for download
    hf_file: str  # HuggingFace filename
    description: str  # Short description
    vram_gb: float  # Estimated VRAM usage
    size_mb: int = 0  # File size in MB (for download progress)

    @property
    def download_url(self) -> str | None:
        """Generate HuggingFace download URL."""
        if not self.hf_repo or not self.hf_file:
            return None
        return f"https://huggingface.co/{self.hf_repo}/resolve/main/{self.hf_file}"


# Translation model preset (Qwen3-8B only)
TRANSLATION_MODEL_PRESETS: dict[str, TranslationModelPreset] = {
    "qwen3-8b": TranslationModelPreset(
        name="Qwen3-8B",
        filename="Qwen_Qwen3-8B-Q4_K_M.gguf",
        hf_repo="bartowski/Qwen_Qwen3-8B-GGUF",
        hf_file="Qwen_Qwen3-8B-Q4_K_M.gguf",
        description="번역 전용 모델",
        vram_gb=5.0,
        size_mb=5130,
    ),
}


@dataclass
class TranslationConfig:
    """LLM translation settings."""

    model_preset: str = "qwen3-8b"
    model_path: str = "./models/Qwen_Qwen3-8B-Q4_K_M.gguf"
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    max_tokens: int = 256
    temperature: float = 0.3
    prompt_template: str = "config/prompts/ja_to_ko.txt"

    # Qwen3 thinking mode
    enable_thinking: bool = True  # Allow model to think before translating

    # Review settings
    enable_review: bool = False  # Enable translation review/refinement
    review_prompt_template: str = "config/prompts/review.txt"

    def get_model_path(self, models_dir: str = "./models") -> str:
        """Get full model path based on preset."""
        if self.model_preset in TRANSLATION_MODEL_PRESETS:
            preset = TRANSLATION_MODEL_PRESETS[self.model_preset]
            return str(Path(models_dir) / preset.filename)
        return self.model_path

    def get_prompt_template(self) -> str:
        """Load prompt template from file."""
        path = Path(self.prompt_template)
        if path.exists():
            return path.read_text(encoding="utf-8")
        # Default fallback prompt
        return (
            "Translate the following Japanese text to Korean.\n"
            "Japanese: {text}\n"
            "Korean:"
        )

    def get_review_prompt_template(self) -> str:
        """Load review prompt template from file."""
        path = Path(self.review_prompt_template)
        if path.exists():
            return path.read_text(encoding="utf-8")
        # Default fallback
        return (
            "Review and fix the Korean translation.\n"
            "Original: {original}\n"
            "Translation: {translation}\n"
            "Corrected:"
        )


@dataclass
class SubtitleASSConfig:
    """ASS-specific subtitle settings."""

    video_width: int = 1920
    video_height: int = 1080
    default_font: str = "Malgun Gothic"
    default_size: int = 48
    original_font: str = "Noto Sans JP"  # Font for original Japanese text
    original_size: int = 36  # Smaller size for original text
    outline_width: float = 2.0
    shadow_depth: float = 1.0


@dataclass
class SubtitleConfig:
    """Subtitle output settings."""

    default_format: str = "srt"
    include_original: bool = False
    include_speaker: bool = False
    ass: SubtitleASSConfig = field(default_factory=SubtitleASSConfig)


@dataclass
class ProcessingConfig:
    """Processing settings."""

    workers: int = 1
    normalize_audio: bool = True
    target_sample_rate: int = 16000


@dataclass
class SegmentConfig:
    """Segment splitting and merging settings."""

    enabled: bool = False  # Enable segment splitting
    max_chars: int = 80  # Maximum characters per segment
    max_duration: float = 4.0  # Maximum duration in seconds
    split_punctuation: str = "。！？!?…"  # Sentence-ending punctuation

    # Merge settings for short segments
    min_chars: int = 5  # Minimum characters per segment (merge if below)
    min_duration: float = 0.5  # Minimum duration in seconds (merge if below)
    merge_gap: float = 0.3  # Maximum gap between segments to allow merging
    merge_incomplete: bool = True  # Merge segments that don't end with punctuation


@dataclass
class UIConfig:
    """UI settings."""

    width: int = 1200
    height: int = 900
    font_size: int = 14
    remember_position: bool = True
    hide_warnings: bool = False  # Hide warning logs from console


@dataclass
class ModelsConfig:
    """Model paths and settings."""

    path: str = "./models"
    auto_download: bool = True

    def get_models_dir(self) -> Path:
        """Get models directory path."""
        path = Path(self.path)
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class AppConfig:
    """Main application configuration."""

    language: str = "ko"
    theme: str = "dark"
    models: ModelsConfig = field(default_factory=ModelsConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    speaker_styles: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Legacy property for backward compatibility
    @property
    def whisperx(self) -> STTConfig:
        """Deprecated: use speech.stt instead."""
        return self.speech.stt


class ConfigManager:
    """Configuration file manager."""

    DEFAULT_CONFIG_PATH = "config/settings.yaml"

    def __init__(self, config_path: str | None = None):
        """
        Initialize config manager.

        Args:
            config_path: Path to config file. Uses default if not specified.
        """
        self.config_path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        self._config: AppConfig | None = None

    def load(self) -> AppConfig:
        """Load configuration from file."""
        if not self.config_path.exists():
            self._config = self.get_default()
            return self._config

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._config = self._parse_config(data)
        return self._config

    def save(self, config: AppConfig | None = None) -> None:
        """Save configuration to file."""
        if config is not None:
            self._config = config

        if self._config is None:
            self._config = self.get_default()

        data = self._to_dict(self._config)

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def get_default(self) -> AppConfig:
        """Get default configuration."""
        return AppConfig()

    @property
    def config(self) -> AppConfig:
        """Get current config, loading if necessary."""
        if self._config is None:
            self.load()
        return self._config  # type: ignore

    def _parse_config(self, data: dict) -> AppConfig:
        """Parse config dictionary into AppConfig."""
        app_data = data.get("app", {})

        # Parse speech config (new structure)
        speech_data = data.get("speech", {})
        if speech_data:
            # New config format
            stt_data = speech_data.get("stt", {})
            diarization_data = speech_data.get("diarization", {})
            speech_config = SpeechConfig(
                stt=STTConfig(**stt_data),
                diarization=DiarizationConfig(**diarization_data),
            )
        else:
            # Legacy whisperx config format - migrate
            whisperx_data = data.get("whisperx", {})
            if whisperx_data:
                diarization_data = whisperx_data.pop("diarization", {})
                # Map old fields to new structure
                stt_config = STTConfig(
                    model_size=whisperx_data.get("model_size", "large-v3-turbo"),
                    device=whisperx_data.get("device", "cuda"),
                    compute_type=whisperx_data.get("compute_type", "float16"),
                    language=whisperx_data.get("language", "ja"),
                    beam_size=5,
                    vad_filter=True,
                )
                diar_config = DiarizationConfig(
                    enabled=diarization_data.get("enabled", True),
                    backend="pyannote",
                    device=whisperx_data.get("device", "cuda"),
                    min_speakers=diarization_data.get("min_speakers", 1),
                    max_speakers=diarization_data.get("max_speakers", 10),
                    hf_token=whisperx_data.get("hf_token", ""),
                )
                speech_config = SpeechConfig(stt=stt_config, diarization=diar_config)
            else:
                speech_config = SpeechConfig()

        translation_config = TranslationConfig(**data.get("translation", {}))

        subtitle_data = data.get("subtitle", {})
        ass_data = subtitle_data.pop("ass", {})
        subtitle_config = SubtitleConfig(
            **subtitle_data,
            ass=SubtitleASSConfig(**ass_data),
        )

        processing_config = ProcessingConfig(**data.get("processing", {}))
        segment_config = SegmentConfig(**data.get("segment", {}))
        ui_config = UIConfig(**data.get("ui", {}))
        models_config = ModelsConfig(**data.get("models", {}))

        return AppConfig(
            language=app_data.get("language", "ko"),
            theme=app_data.get("theme", "dark"),
            models=models_config,
            speech=speech_config,
            translation=translation_config,
            subtitle=subtitle_config,
            processing=processing_config,
            segment=segment_config,
            ui=ui_config,
            speaker_styles=data.get("speaker_styles", {}),
        )

    def _to_dict(self, config: AppConfig) -> dict:
        """Convert AppConfig to dictionary for YAML."""
        return {
            "app": {
                "language": config.language,
                "theme": config.theme,
            },
            "models": {
                "path": config.models.path,
                "auto_download": config.models.auto_download,
            },
            "speech": {
                "stt": {
                    "model_size": config.speech.stt.model_size,
                    "device": config.speech.stt.device,
                    "compute_type": config.speech.stt.compute_type,
                    "language": config.speech.stt.language,
                    "beam_size": config.speech.stt.beam_size,
                    "vad_filter": config.speech.stt.vad_filter,
                },
                "diarization": {
                    "enabled": config.speech.diarization.enabled,
                    "backend": config.speech.diarization.backend,
                    "model_name": config.speech.diarization.model_name,
                    "device": config.speech.diarization.device,
                    "min_speakers": config.speech.diarization.min_speakers,
                    "max_speakers": config.speech.diarization.max_speakers,
                    "hf_token": config.speech.diarization.hf_token,
                },
            },
            "translation": {
                "model_preset": config.translation.model_preset,
                "model_path": config.translation.model_path,
                "n_gpu_layers": config.translation.n_gpu_layers,
                "n_ctx": config.translation.n_ctx,
                "max_tokens": config.translation.max_tokens,
                "temperature": config.translation.temperature,
                "prompt_template": config.translation.prompt_template,
                "enable_review": config.translation.enable_review,
                "review_prompt_template": config.translation.review_prompt_template,
            },
            "subtitle": {
                "default_format": config.subtitle.default_format,
                "include_original": config.subtitle.include_original,
                "include_speaker": config.subtitle.include_speaker,
                "ass": {
                    "video_width": config.subtitle.ass.video_width,
                    "video_height": config.subtitle.ass.video_height,
                    "default_font": config.subtitle.ass.default_font,
                    "default_size": config.subtitle.ass.default_size,
                    "original_font": config.subtitle.ass.original_font,
                    "original_size": config.subtitle.ass.original_size,
                    "outline_width": config.subtitle.ass.outline_width,
                    "shadow_depth": config.subtitle.ass.shadow_depth,
                },
            },
            "processing": {
                "workers": config.processing.workers,
                "normalize_audio": config.processing.normalize_audio,
                "target_sample_rate": config.processing.target_sample_rate,
            },
            "segment": {
                "enabled": config.segment.enabled,
                "max_chars": config.segment.max_chars,
                "max_duration": config.segment.max_duration,
                "split_punctuation": config.segment.split_punctuation,
                "min_chars": config.segment.min_chars,
                "min_duration": config.segment.min_duration,
                "merge_gap": config.segment.merge_gap,
                "merge_incomplete": config.segment.merge_incomplete,
            },
            "ui": {
                "width": config.ui.width,
                "height": config.ui.height,
                "font_size": config.ui.font_size,
                "remember_position": config.ui.remember_position,
                "hide_warnings": config.ui.hide_warnings,
            },
            "speaker_styles": config.speaker_styles,
        }

    def validate(self) -> list[str]:
        """
        Validate current configuration.

        Returns:
            List of error messages. Empty if valid.
        """
        errors = []
        config = self.config

        # Check model path
        if config.translation.model_path:
            model_path = Path(config.translation.model_path)
            if not model_path.exists() and not config.models.auto_download:
                errors.append(f"Translation model not found: {model_path}")

        # Check STT settings
        if config.speech.stt.model_size not in SUPPORTED_WHISPER_MODELS:
            errors.append(
                f"Invalid Whisper model size: {config.speech.stt.model_size}. "
                f"Supported: {SUPPORTED_WHISPER_MODELS}"
            )

        valid_devices = ["cuda", "cpu"]
        if config.speech.stt.device not in valid_devices:
            errors.append(f"Invalid STT device: {config.speech.stt.device}")

        # Check diarization settings
        if config.speech.diarization.enabled:
            if config.speech.diarization.backend not in SUPPORTED_DIARIZATION_BACKENDS:
                errors.append(
                    f"Invalid diarization backend: {config.speech.diarization.backend}. "
                    f"Supported: {SUPPORTED_DIARIZATION_BACKENDS}"
                )

            # PyAnnote requires HF token
            if config.speech.diarization.backend == "pyannote":
                if not config.speech.diarization.get_hf_token():
                    errors.append("HF token required for pyannote speaker diarization")

            if config.speech.diarization.device not in valid_devices:
                errors.append(f"Invalid diarization device: {config.speech.diarization.device}")

        # Check subtitle format
        valid_formats = ["srt", "ass", "vtt"]
        if config.subtitle.default_format not in valid_formats:
            errors.append(f"Invalid subtitle format: {config.subtitle.default_format}")

        return errors


# Global config instance
_config_manager: ConfigManager | None = None


def get_config() -> AppConfig:
    """Get global configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config


def get_config_manager() -> ConfigManager:
    """Get global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
