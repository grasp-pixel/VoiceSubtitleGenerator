"""Speech processing engine - STT only."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .models import ProgressCallback, Segment, Word
from .stt import FasterWhisperEngine, STTEngine, TranscriptionResult

if TYPE_CHECKING:
    from .config import SpeechConfig

logger = logging.getLogger(__name__)


class SpeechEngineError(Exception):
    """Raised when speech processing fails."""

    pass


class SpeechEngine:
    """
    Speech processing engine using faster-whisper for STT.
    """

    def __init__(self, stt_engine: STTEngine):
        """
        Initialize speech engine.

        Args:
            stt_engine: STT engine for transcription.
        """
        self.stt = stt_engine

    @classmethod
    def from_config(cls, config: SpeechConfig) -> SpeechEngine:
        """
        Create engine from config.

        Args:
            config: Speech configuration.

        Returns:
            Configured SpeechEngine instance.
        """
        stt = FasterWhisperEngine.from_config(config.stt)
        return cls(stt)

    def load_models(self, language: str = "ja") -> None:
        """
        Load STT model.

        Args:
            language: Language code for STT.
        """
        logger.info("Loading STT model...")
        self.stt.load_model()
        logger.info("STT model loaded")

    def unload_models(self) -> None:
        """Unload models and free memory."""
        logger.info("Unloading STT model...")
        self.stt.unload_model()
        logger.info("STT model unloaded")

    def process(
        self,
        audio_path: str,
        language: str = "ja",
        progress_callback: ProgressCallback | None = None,
    ) -> list[Segment]:
        """
        Process audio: STT transcription.

        Progress breakdown:
        - 0-10%: Loading/initialization
        - 10-100%: Transcription

        Args:
            audio_path: Path to audio file.
            language: Source language code.
            progress_callback: Progress callback (0.0 to 1.0).

        Returns:
            List of Segment objects.

        Raises:
            SpeechEngineError: If processing fails.
        """
        try:
            self._report_progress(progress_callback, 0.0)

            # Ensure model is loaded
            if not self.stt.is_loaded:
                self.stt.load_model()
            self._report_progress(progress_callback, 0.1)

            # Transcription (10-100%)
            def stt_progress(p: float) -> None:
                self._report_progress(progress_callback, 0.1 + p * 0.9)

            logger.info(f"Starting transcription: {audio_path}")
            transcription = self.stt.transcribe(
                audio_path,
                language=language,
                progress_callback=stt_progress,
            )

            # Convert to segments
            segments = self._convert_transcription(transcription)
            self._report_progress(progress_callback, 1.0)

            logger.info(f"Processed {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Speech processing failed: {e}")
            raise SpeechEngineError(f"Processing failed: {e}") from e

    def _convert_transcription(self, transcription: TranscriptionResult) -> list[Segment]:
        """
        Convert transcription result to Segment list.

        Args:
            transcription: Transcription result from STT.

        Returns:
            List of Segment objects.
        """
        segments: list[Segment] = []

        for trans_seg in transcription.segments:
            words = [
                Word(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
                for w in trans_seg.words
            ]

            segment = Segment(
                start=trans_seg.start,
                end=trans_seg.end,
                original_text=trans_seg.text,
                words=words,
            )
            segments.append(segment)

        return segments

    def _report_progress(
        self,
        callback: ProgressCallback | None,
        progress: float,
    ) -> None:
        """Report progress if callback provided."""
        if callback is not None:
            callback(progress)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.stt.is_loaded

    @property
    def device(self) -> str:
        """Get the device used by STT engine."""
        return self.stt.device

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        return self.stt.get_supported_languages()

    def estimate_vram_usage(self) -> dict[str, float]:
        """
        Estimate total VRAM usage in GB.

        Returns:
            dict: Estimated VRAM for each component and total.
        """
        result: dict[str, float] = {}

        if hasattr(self.stt, "estimate_vram_usage"):
            stt_vram = self.stt.estimate_vram_usage()
            result["stt"] = stt_vram.get("total", 4.0)
        else:
            result["stt"] = 4.0

        result["total"] = result["stt"]

        return result
