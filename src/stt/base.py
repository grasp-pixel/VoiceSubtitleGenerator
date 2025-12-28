"""Base classes and interfaces for STT engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TranscriptionWord:
    """Word-level transcription result."""

    word: str
    start: float
    end: float
    probability: float = 1.0

    @property
    def duration(self) -> float:
        """Get word duration in seconds."""
        return self.end - self.start


@dataclass
class TranscriptionSegment:
    """Segment-level transcription result."""

    start: float
    end: float
    text: str
    words: list[TranscriptionWord] = field(default_factory=list)
    language: str = ""

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Full transcription result."""

    segments: list[TranscriptionSegment]
    language: str
    duration: float


# Type alias for progress callback
ProgressCallback = Callable[[float], None]


class STTEngine(ABC):
    """Abstract base class for STT engines."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the STT model."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free resources."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: str = "ja",
        progress_callback: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file.
            language: Language code (e.g., "ja", "en").
            progress_callback: Progress callback (0.0 to 1.0).

        Returns:
            TranscriptionResult with segments and words.
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        pass
