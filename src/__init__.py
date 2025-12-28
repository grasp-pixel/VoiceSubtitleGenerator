"""Voice Subtitle Generator - Core Package.

A tool for generating Korean subtitles from Japanese audio files
with speaker diarization support.
"""

# Apply compatibility patches before any other imports
from . import torch_compat  # noqa: F401

from .ass_styler import (
    ASSColor,
    ASSStyle,
    ASSStyler,
    KeywordStyle,
    SpeakerStyle,
    StylePreset,
    TextAlignment,
)
from .audio_analyzer import (
    AudioPosition,
    AudioPositionAnalyzer,
    PositionResult,
    SegmentPosition,
    analyze_audio_positions,
)
from .audio_loader import AudioFormatError, AudioLoader
from .config import (
    AppConfig,
    ConfigManager,
    DiarizationConfig,
    SpeechConfig,
    STTConfig,
    SubtitleConfig,
    TranslationConfig,
    get_config,
    get_config_manager,
)
from .exceptions import (
    VoiceSubtitleError,
    ConfigurationError,
    ConfigNotFoundError,
    ConfigValidationError,
    AudioError,
    AudioNotFoundError,
    AudioLoadError,
    ModelError,
    ModelNotFoundError as ModelNotFoundExc,
    ModelLoadError,
    ModelInitializationError,
    TranscriptionError,
    DiarizationError,
    HFTokenError,
    TranslationError as TranslationExc,
    TranslatorNotInitializedError,
    TranslationFailedError,
    SubtitleError,
    SubtitleWriteError,
    UnsupportedSubtitleFormatError,
    PipelineError as PipelineExc,
    PipelineNotInitializedError,
    ProcessingCancelledError,
    ProcessingError,
    SpeakerError,
    SpeakerPresetNotFoundError,
    SpeakerMappingError,
)
from .logging_config import (
    setup_logging,
    get_logger,
    set_log_level,
    LoggingConfig,
    ProgressLogger,
)
from .models import (
    AudioInfo,
    BatchCallbacks,
    PipelineCallbacks,
    ProcessingStage,
    ProcessResult,
    Segment,
    SpeakerMapping,
    SubtitleFormat,
    SubtitleStyle,
    Word,
)
from .pipeline import PipelineError, SubtitlePipeline
from .speaker_manager import SpeakerManager
from .speech_engine import SpeechEngine, SpeechEngineError
from .subtitle_writer import SubtitleWriter, SubtitleWriterError
from .translator import ModelNotFoundError, TranslationError, Translator
from .model_manager import ModelInfo, ModelManager, ModelStatus
from .font_manager import FontInfo, FontManager, get_font_manager

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Pipeline
    "SubtitlePipeline",
    "PipelineError",
    # Audio
    "AudioLoader",
    "AudioFormatError",
    "AudioInfo",
    # Speech Engine
    "SpeechEngine",
    "SpeechEngineError",
    # Translation
    "Translator",
    "TranslationError",
    "ModelNotFoundError",
    # Speaker
    "SpeakerManager",
    # Subtitle
    "SubtitleWriter",
    "SubtitleWriterError",
    # Models
    "Segment",
    "Word",
    "SpeakerMapping",
    "ProcessResult",
    "SubtitleStyle",
    "SubtitleFormat",
    "ProcessingStage",
    # Callbacks
    "PipelineCallbacks",
    "BatchCallbacks",
    # Config
    "AppConfig",
    "ConfigManager",
    "DiarizationConfig",
    "SpeechConfig",
    "STTConfig",
    "TranslationConfig",
    "SubtitleConfig",
    "get_config",
    "get_config_manager",
    # Exceptions
    "VoiceSubtitleError",
    "ConfigurationError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    "AudioError",
    "AudioNotFoundError",
    "AudioLoadError",
    "ModelError",
    "ModelNotFoundExc",
    "ModelLoadError",
    "ModelInitializationError",
    "TranscriptionError",
    "DiarizationError",
    "HFTokenError",
    "TranslationExc",
    "TranslatorNotInitializedError",
    "TranslationFailedError",
    "SubtitleError",
    "SubtitleWriteError",
    "UnsupportedSubtitleFormatError",
    "PipelineExc",
    "PipelineNotInitializedError",
    "ProcessingCancelledError",
    "ProcessingError",
    "SpeakerError",
    "SpeakerPresetNotFoundError",
    "SpeakerMappingError",
    # Logging
    "setup_logging",
    "get_logger",
    "set_log_level",
    "LoggingConfig",
    "ProgressLogger",
    # ASS Styling
    "ASSColor",
    "ASSStyle",
    "ASSStyler",
    "KeywordStyle",
    "SpeakerStyle",
    "StylePreset",
    "TextAlignment",
    # Audio Analysis
    "AudioPosition",
    "AudioPositionAnalyzer",
    "PositionResult",
    "SegmentPosition",
    "analyze_audio_positions",
    # Model Management
    "ModelInfo",
    "ModelManager",
    "ModelStatus",
    # Font Management
    "FontInfo",
    "FontManager",
    "get_font_manager",
]
