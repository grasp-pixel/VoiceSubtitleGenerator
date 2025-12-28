"""Base classes and interfaces for speaker diarization engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DiarizationTurn:
    """Single speaker turn."""

    start: float
    end: float
    speaker_id: str

    @property
    def duration(self) -> float:
        """Get turn duration in seconds."""
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Full diarization result."""

    turns: list[DiarizationTurn] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)

    @property
    def num_speakers(self) -> int:
        """Get number of speakers."""
        return len(self.speakers)


# Type alias for progress callback
ProgressCallback = Callable[[float], None]


class DiarizationEngine(ABC):
    """Abstract base class for speaker diarization engines."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the diarization model."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free resources."""
        pass

    @abstractmethod
    def diarize(
        self,
        audio_path: str,
        min_speakers: int = 1,
        max_speakers: int = 10,
        progress_callback: ProgressCallback | None = None,
    ) -> DiarizationResult:
        """
        Run speaker diarization on audio file.

        Args:
            audio_path: Path to audio file.
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            progress_callback: Progress callback (0.0 to 1.0).

        Returns:
            DiarizationResult with speaker turns.
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @property
    @abstractmethod
    def requires_auth_token(self) -> bool:
        """Check if this engine requires an authentication token."""
        pass
