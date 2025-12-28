"""Voice Subtitle Generator - Core Package.

A tool for generating Korean subtitles from Japanese audio files.
"""

# Apply compatibility patches before any other imports
from . import torch_compat  # noqa: F401

from .ass_styler import (
    ASSColor,
    ASSStyle,
    ASSStyler,
    KeywordStyle,
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
    SubtitleFormat,
    SubtitleStyle,
    Word,
)
from .pipeline import PipelineError, SubtitlePipeline
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
    # Subtitle
    "SubtitleWriter",
    "SubtitleWriterError",
    # Models
    "Segment",
    "Word",
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
