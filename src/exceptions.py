"""Custom exceptions for Voice Subtitle Generator."""


class VoiceSubtitleError(Exception):
    """Base exception for Voice Subtitle Generator."""

    def __init__(self, message: str, details: str | None = None):
        """
        Initialize exception.

        Args:
            message: Error message.
            details: Additional details.
        """
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


# Configuration errors


class ConfigurationError(VoiceSubtitleError):
    """Configuration related errors."""

    pass


class ConfigNotFoundError(ConfigurationError):
    """Configuration file not found."""

    def __init__(self, config_path: str):
        super().__init__(
            "Configuration file not found",
            f"Path: {config_path}",
        )
        self.config_path = config_path


class ConfigValidationError(ConfigurationError):
    """Configuration validation failed."""

    def __init__(self, errors: list[str]):
        super().__init__(
            "Configuration validation failed",
            "; ".join(errors),
        )
        self.errors = errors


# Audio errors


class AudioError(VoiceSubtitleError):
    """Audio processing related errors."""

    pass


class AudioNotFoundError(AudioError):
    """Audio file not found."""

    def __init__(self, audio_path: str):
        super().__init__(
            "Audio file not found",
            f"Path: {audio_path}",
        )
        self.audio_path = audio_path


class AudioFormatError(AudioError):
    """Unsupported audio format."""

    def __init__(self, audio_path: str, format: str | None = None):
        details = f"Path: {audio_path}"
        if format:
            details += f", Format: {format}"
        super().__init__(
            "Unsupported audio format",
            details,
        )
        self.audio_path = audio_path
        self.format = format


class AudioLoadError(AudioError):
    """Failed to load audio file."""

    def __init__(self, audio_path: str, reason: str | None = None):
        details = f"Path: {audio_path}"
        if reason:
            details += f", Reason: {reason}"
        super().__init__(
            "Failed to load audio file",
            details,
        )
        self.audio_path = audio_path
        self.reason = reason


# Model errors


class ModelError(VoiceSubtitleError):
    """Model related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Model file not found."""

    def __init__(self, model_path: str, model_type: str = "model"):
        super().__init__(
            f"{model_type.capitalize()} not found",
            f"Path: {model_path}",
        )
        self.model_path = model_path
        self.model_type = model_type


class ModelLoadError(ModelError):
    """Failed to load model."""

    def __init__(self, model_name: str, reason: str | None = None):
        details = f"Model: {model_name}"
        if reason:
            details += f", Reason: {reason}"
        super().__init__(
            "Failed to load model",
            details,
        )
        self.model_name = model_name
        self.reason = reason


class ModelInitializationError(ModelError):
    """Failed to initialize model."""

    def __init__(self, model_name: str, reason: str | None = None):
        details = f"Model: {model_name}"
        if reason:
            details += f", Reason: {reason}"
        super().__init__(
            "Failed to initialize model",
            details,
        )
        self.model_name = model_name
        self.reason = reason


# Transcription errors


class TranscriptionError(VoiceSubtitleError):
    """Transcription related errors."""

    pass


class STTError(TranscriptionError):
    """Speech-to-text processing error."""

    def __init__(self, stage: str, reason: str | None = None):
        details = f"Stage: {stage}"
        if reason:
            details += f", Reason: {reason}"
        super().__init__(
            "STT processing failed",
            details,
        )
        self.stage = stage
        self.reason = reason


class HFTokenError(TranscriptionError):
    """HuggingFace token error."""

    def __init__(self, reason: str = "Token not provided or invalid"):
        super().__init__(
            "HuggingFace token error",
            reason,
        )


# Translation errors


class TranslationError(VoiceSubtitleError):
    """Translation related errors."""

    pass


class TranslatorNotInitializedError(TranslationError):
    """Translator not initialized."""

    def __init__(self):
        super().__init__(
            "Translator not initialized",
            "Call initialize() before translating",
        )


class TranslationFailedError(TranslationError):
    """Translation failed."""

    def __init__(self, text: str | None = None, reason: str | None = None):
        details = None
        if text:
            preview = text[:50] + "..." if len(text) > 50 else text
            details = f"Text: {preview}"
        if reason:
            details = f"{details}, Reason: {reason}" if details else f"Reason: {reason}"
        super().__init__(
            "Translation failed",
            details,
        )
        self.text = text
        self.reason = reason


# Subtitle errors


class SubtitleError(VoiceSubtitleError):
    """Subtitle related errors."""

    pass


class SubtitleWriteError(SubtitleError):
    """Failed to write subtitle file."""

    def __init__(self, output_path: str, reason: str | None = None):
        details = f"Path: {output_path}"
        if reason:
            details += f", Reason: {reason}"
        super().__init__(
            "Failed to write subtitle file",
            details,
        )
        self.output_path = output_path
        self.reason = reason


class UnsupportedSubtitleFormatError(SubtitleError):
    """Unsupported subtitle format."""

    def __init__(self, format: str, supported_formats: list[str] | None = None):
        details = f"Format: {format}"
        if supported_formats:
            details += f", Supported: {', '.join(supported_formats)}"
        super().__init__(
            "Unsupported subtitle format",
            details,
        )
        self.format = format
        self.supported_formats = supported_formats


# Pipeline errors


class PipelineError(VoiceSubtitleError):
    """Pipeline related errors."""

    pass


class PipelineNotInitializedError(PipelineError):
    """Pipeline not initialized."""

    def __init__(self):
        super().__init__(
            "Pipeline not initialized",
            "Call initialize() before processing",
        )


class ProcessingCancelledError(PipelineError):
    """Processing was cancelled."""

    def __init__(self, file_path: str | None = None):
        super().__init__(
            "Processing cancelled by user",
            f"File: {file_path}" if file_path else None,
        )
        self.file_path = file_path


class ProcessingError(PipelineError):
    """General processing error."""

    def __init__(self, stage: str, file_path: str | None = None, reason: str | None = None):
        details = f"Stage: {stage}"
        if file_path:
            details += f", File: {file_path}"
        if reason:
            details += f", Reason: {reason}"
        super().__init__(
            "Processing failed",
            details,
        )
        self.stage = stage
        self.file_path = file_path
        self.reason = reason


