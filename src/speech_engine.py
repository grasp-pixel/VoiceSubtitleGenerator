"""Speech processing engine - combines STT and diarization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .diarization import DiarizationEngine, DiarizationResult, PyAnnoteEngine, SpeechBrainEngine
from .models import ProgressCallback, Segment, Word
from .stt import FasterWhisperEngine, STTEngine, TranscriptionResult

if TYPE_CHECKING:
    from .config import DiarizationConfig, SpeechConfig, STTConfig

logger = logging.getLogger(__name__)


class SpeechEngineError(Exception):
    """Raised when speech processing fails."""

    pass


class SpeechEngine:
    """
    Unified speech processing engine.

    Combines STT (faster-whisper) and speaker diarization (pyannote/speechbrain)
    to produce segments with speaker information.
    """

    def __init__(
        self,
        stt_engine: STTEngine,
        diarization_engine: DiarizationEngine | None = None,
    ):
        """
        Initialize speech engine.

        Args:
            stt_engine: STT engine for transcription.
            diarization_engine: Optional diarization engine for speaker separation.
        """
        self.stt = stt_engine
        self.diarization = diarization_engine

    @classmethod
    def from_config(cls, config: SpeechConfig) -> SpeechEngine:
        """
        Create engine from config.

        Args:
            config: Speech configuration.

        Returns:
            Configured SpeechEngine instance.
        """
        # Create STT engine
        stt = FasterWhisperEngine.from_config(config.stt)

        # Create diarization engine if enabled
        diarization: DiarizationEngine | None = None
        if config.diarization.enabled:
            if config.diarization.backend == "pyannote":
                diarization = PyAnnoteEngine.from_config(config.diarization)
            elif config.diarization.backend == "speechbrain":
                diarization = SpeechBrainEngine.from_config(config.diarization)
            else:
                logger.warning(
                    f"Unknown diarization backend: {config.diarization.backend}, "
                    "defaulting to pyannote"
                )
                diarization = PyAnnoteEngine.from_config(config.diarization)

        return cls(stt, diarization)

    def load_models(self, language: str = "ja") -> None:
        """
        Load all required models.

        Args:
            language: Language code for STT.
        """
        logger.info("Loading speech processing models...")

        self.stt.load_model()

        if self.diarization:
            self.diarization.load_model()

        logger.info("All models loaded")

    def unload_models(self) -> None:
        """Unload all models and free memory."""
        logger.info("Unloading speech processing models...")

        self.stt.unload_model()

        if self.diarization:
            self.diarization.unload_model()

        logger.info("All models unloaded")

    def process(
        self,
        audio_path: str,
        language: str = "ja",
        min_speakers: int = 1,
        max_speakers: int = 10,
        enable_diarization: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> list[Segment]:
        """
        Process audio: STT + optional diarization.

        Progress breakdown:
        - 0-10%: Loading/initialization
        - 10-60%: Transcription
        - 60-90%: Diarization (if enabled)
        - 90-100%: Merging results

        Args:
            audio_path: Path to audio file.
            language: Source language code.
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            enable_diarization: Whether to run diarization.
            progress_callback: Progress callback (0.0 to 1.0).

        Returns:
            List of Segment objects with speaker info.

        Raises:
            SpeechEngineError: If processing fails.
        """
        try:
            self._report_progress(progress_callback, 0.0)

            # Ensure models are loaded
            if not self.stt.is_loaded:
                self.stt.load_model()
            self._report_progress(progress_callback, 0.1)

            # Transcription (10-60%)
            def stt_progress(p: float) -> None:
                self._report_progress(progress_callback, 0.1 + p * 0.5)

            logger.info(f"Starting transcription: {audio_path}")
            transcription = self.stt.transcribe(
                audio_path,
                language=language,
                progress_callback=stt_progress,
            )
            self._report_progress(progress_callback, 0.6)

            # Diarization (60-90%)
            diarization_result: DiarizationResult | None = None

            if enable_diarization and self.diarization:
                if not self.diarization.is_loaded:
                    self.diarization.load_model()

                def diar_progress(p: float) -> None:
                    self._report_progress(progress_callback, 0.6 + p * 0.3)

                logger.info("Starting speaker diarization")
                diarization_result = self.diarization.diarize(
                    audio_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    progress_callback=diar_progress,
                )

            self._report_progress(progress_callback, 0.9)

            # Merge results (90-100%)
            logger.info("Merging transcription and diarization results")
            if diarization_result:
                logger.info(
                    f"Diarization: {len(diarization_result.speakers)} speakers, "
                    f"{len(diarization_result.turns)} turns"
                )
            segments = self._merge_results(transcription, diarization_result)
            self._report_progress(progress_callback, 1.0)

            logger.info(f"Processed {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Speech processing failed: {e}")
            raise SpeechEngineError(f"Processing failed: {e}") from e

    def transcribe_only(
        self,
        audio_path: str,
        language: str = "ja",
        progress_callback: ProgressCallback | None = None,
    ) -> list[Segment]:
        """
        Transcribe without diarization.

        Args:
            audio_path: Path to audio file.
            language: Source language code.
            progress_callback: Progress callback.

        Returns:
            List of Segment objects without speaker info.
        """
        return self.process(
            audio_path=audio_path,
            language=language,
            enable_diarization=False,
            progress_callback=progress_callback,
        )

    def _merge_results(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult | None,
    ) -> list[Segment]:
        """
        Merge transcription and diarization results.

        Assigns speaker IDs to segments based on time overlap.

        Args:
            transcription: Transcription result from STT.
            diarization: Optional diarization result.

        Returns:
            List of Segment objects.
        """
        segments: list[Segment] = []

        for trans_seg in transcription.segments:
            # Convert words
            words = [
                Word(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
                for w in trans_seg.words
            ]

            # Find speaker for this segment
            speaker_id = "UNKNOWN"
            if diarization:
                speaker_id = self._find_speaker(
                    trans_seg.start,
                    trans_seg.end,
                    diarization,
                )

            segment = Segment(
                start=trans_seg.start,
                end=trans_seg.end,
                original_text=trans_seg.text,
                speaker_id=speaker_id,
                words=words,
            )
            segments.append(segment)

        # Log speaker assignment summary
        speaker_counts: dict[str, int] = {}
        for seg in segments:
            speaker_counts[seg.speaker_id] = speaker_counts.get(seg.speaker_id, 0) + 1
        logger.debug(f"Speaker assignment: {speaker_counts}")

        return segments

    def _find_speaker(
        self,
        start: float,
        end: float,
        diarization: DiarizationResult,
    ) -> str:
        """
        Find the dominant speaker for a time range.

        Uses overlap duration to determine the most likely speaker.

        Args:
            start: Segment start time.
            end: Segment end time.
            diarization: Diarization result.

        Returns:
            Speaker ID with most overlap, or "UNKNOWN".
        """
        if not diarization.turns:
            logger.warning("No diarization turns available")
            return "UNKNOWN"

        speaker_durations: dict[str, float] = {}

        for turn in diarization.turns:
            # Calculate overlap
            overlap_start = max(start, turn.start)
            overlap_end = min(end, turn.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > 0:
                speaker_durations[turn.speaker_id] = (
                    speaker_durations.get(turn.speaker_id, 0.0) + overlap
                )

        if not speaker_durations:
            # Log first few turns for debugging
            if diarization.turns:
                sample_turns = diarization.turns[:3]
                logger.debug(
                    f"No overlap found for segment [{start:.2f}-{end:.2f}], "
                    f"sample turns: {[(t.start, t.end, t.speaker_id) for t in sample_turns]}"
                )
            return "UNKNOWN"

        # Return speaker with most overlap
        return max(speaker_durations, key=lambda k: speaker_durations[k])

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
        """Check if all models are loaded."""
        stt_loaded = self.stt.is_loaded
        diar_loaded = self.diarization.is_loaded if self.diarization else True
        return stt_loaded and diar_loaded

    @property
    def device(self) -> str:
        """Get the device used by STT engine."""
        return self.stt.device

    @property
    def diarization_available(self) -> bool:
        """Check if diarization is available and loaded."""
        return self.diarization is not None and self.diarization.is_loaded

    @property
    def diarization_enabled(self) -> bool:
        """Check if diarization engine is configured."""
        return self.diarization is not None

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

        # Get STT VRAM
        if hasattr(self.stt, "estimate_vram_usage"):
            stt_vram = self.stt.estimate_vram_usage()
            result["stt"] = stt_vram.get("total", 4.0)
        else:
            result["stt"] = 4.0

        # Get diarization VRAM
        if self.diarization and hasattr(self.diarization, "estimate_vram_usage"):
            diar_vram = self.diarization.estimate_vram_usage()
            result["diarization"] = diar_vram.get("total", 1.0)
        else:
            result["diarization"] = 0.0

        result["total"] = result["stt"] + result["diarization"]

        return result
