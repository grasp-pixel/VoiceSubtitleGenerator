"""STT (Speech-to-Text) module."""

from .base import (
    STTEngine,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)
from .faster_whisper_engine import FasterWhisperEngine

__all__ = [
    "STTEngine",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionWord",
    "FasterWhisperEngine",
]
