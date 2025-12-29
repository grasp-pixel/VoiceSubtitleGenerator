"""Faster-Whisper STT engine implementation."""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import torch
from faster_whisper import WhisperModel

from .base import (
    ProgressCallback,
    STTEngine,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)

if TYPE_CHECKING:
    from ..config import STTConfig

logger = logging.getLogger(__name__)

# Supported Whisper models (large-v3 and turbo only)
SUPPORTED_MODELS = ["large-v3", "large-v3-turbo"]


class FasterWhisperEngine(STTEngine):
    """
    Faster-Whisper based STT engine.

    Uses CTranslate2 for efficient inference.
    Provides native word-level timestamps.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        """
        Initialize Faster-Whisper engine.

        Args:
            model_size: Model size ("large-v3" or "large-v3-turbo").
            device: Compute device ("cuda" or "cpu").
            compute_type: Compute type ("float16", "int8", "float32").
            beam_size: Beam size for decoding.
            vad_filter: Enable VAD filter to skip silent sections.
        """
        if model_size not in SUPPORTED_MODELS:
            logger.warning(
                f"Model {model_size} not in supported list {SUPPORTED_MODELS}, "
                f"defaulting to large-v3-turbo"
            )
            model_size = "large-v3-turbo"

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        self._model: WhisperModel | None = None

    @classmethod
    def from_config(cls, config: STTConfig) -> FasterWhisperEngine:
        """Create engine from config."""
        device = config.device
        compute_type = config.compute_type

        # Auto-detect: fallback to CPU if CUDA is requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
            # float16 is not supported on CPU, use int8 instead
            if compute_type == "float16":
                compute_type = "int8"

        return cls(
            model_size=config.model_size,
            device=device,
            compute_type=compute_type,
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
        )

    def load_model(self) -> None:
        """Load the Whisper model."""
        if self._model is not None:
            return

        logger.info(f"Loading faster-whisper model: {self.model_size}")

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

        logger.info("Faster-whisper model loaded successfully")

    def unload_model(self) -> None:
        """Unload the model and free memory."""
        if self._model is None:
            return

        self._model = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Faster-whisper model unloaded")

    def transcribe(
        self,
        audio_path: str,
        language: str = "ja",
        progress_callback: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file with word-level timestamps.

        Args:
            audio_path: Path to audio file.
            language: Language code (e.g., "ja", "en").
            progress_callback: Progress callback (0.0 to 1.0).

        Returns:
            TranscriptionResult with segments and words.
        """
        if self._model is None:
            self.load_model()

        logger.info(f"Transcribing: {audio_path}")

        # Transcribe with word timestamps
        segments_gen, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=self.vad_filter,
        )

        # Convert generator to list with progress
        segments: list[TranscriptionSegment] = []
        total_duration = info.duration

        for segment in segments_gen:
            words: list[TranscriptionWord] = []

            if segment.words:
                for w in segment.words:
                    words.append(
                        TranscriptionWord(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )

            segments.append(
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    words=words,
                    language=language,
                )
            )

            # Report progress based on segment end time
            if progress_callback and total_duration > 0:
                progress = min(segment.end / total_duration, 1.0)
                progress_callback(progress)

        logger.info(f"Transcribed {len(segments)} segments")

        return TranscriptionResult(
            segments=segments,
            language=info.language or language,
            duration=total_duration,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        return [
            "ja",  # Japanese
            "en",  # English
            "zh",  # Chinese
            "ko",  # Korean
            "de",  # German
            "fr",  # French
            "es",  # Spanish
            "it",  # Italian
            "pt",  # Portuguese
            "ru",  # Russian
        ]

    def estimate_vram_usage(self) -> dict[str, float]:
        """
        Estimate VRAM usage in GB.

        Returns:
            dict: Estimated VRAM for model.
        """
        model_vram = {
            "large-v3": 4.0,
            "large-v3-turbo": 3.5,
        }

        return {
            "whisper": model_vram.get(self.model_size, 4.0),
            "total": model_vram.get(self.model_size, 4.0),
        }
