"""WhisperX engine for STT and speaker diarization."""

# Ensure compatibility patches are applied
import src.torch_compat  # noqa: F401

import gc
import logging
from typing import Any

import torch
import whisperx

from .config import WhisperXConfig
from .models import ProgressCallback, Segment, Word

logger = logging.getLogger(__name__)


class WhisperXError(Exception):
    """Raised when WhisperX processing fails."""

    pass


class WhisperXEngine:
    """
    WhisperX wrapper for STT + word alignment + speaker diarization.

    Combines faster-whisper for STT, wav2vec for alignment,
    and pyannote for speaker diarization.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        hf_token: str | None = None,
        batch_size: int = 16,
    ):
        """
        Initialize WhisperX engine.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
            device: Compute device ("cuda" or "cpu").
            compute_type: Compute type ("float16", "int8", "float32").
            hf_token: HuggingFace token for pyannote (speaker diarization).
            batch_size: Batch size for processing.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.hf_token = hf_token
        self.batch_size = batch_size

        self._model: Any = None
        self._diarize_model: Any = None
        self._align_model: Any = None
        self._align_metadata: Any = None

    @classmethod
    def from_config(cls, config: WhisperXConfig) -> "WhisperXEngine":
        """Create engine from config."""
        return cls(
            model_size=config.model_size,
            device=config.device,
            compute_type=config.compute_type,
            hf_token=config.get_hf_token(),
            batch_size=config.batch_size,
        )

    def load_models(self, language: str = "ja") -> None:
        """
        Load all required models.

        Args:
            language: Language code for alignment model.
        """
        logger.info(f"Loading WhisperX model: {self.model_size}")

        # Load STT model
        self._model = whisperx.load_model(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

        # Load alignment model
        logger.info(f"Loading alignment model for: {language}")
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )

        # Load diarization model if token available
        if self.hf_token:
            logger.info("Loading diarization model (pyannote)")
            from whisperx.diarize import DiarizationPipeline

            self._diarize_model = DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device,
            )
        else:
            logger.warning("HF token not provided, speaker diarization disabled")

    def unload_models(self) -> None:
        """Unload all models and free memory."""
        self._model = None
        self._diarize_model = None
        self._align_model = None
        self._align_metadata = None

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("WhisperX models unloaded")

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
        Process audio file: STT + alignment + diarization.

        Args:
            audio_path: Path to audio file.
            language: Source language code.
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            enable_diarization: Whether to run speaker diarization.
            progress_callback: Progress callback (0.0 to 1.0).

        Returns:
            list[Segment]: Processed segments with speaker info.

        Raises:
            WhisperXError: If processing fails.
        """
        if self._model is None:
            self.load_models(language)

        try:
            # Stage 1: Load audio (10%)
            self._report_progress(progress_callback, 0.0)
            logger.info(f"Loading audio: {audio_path}")
            audio = whisperx.load_audio(audio_path)
            self._report_progress(progress_callback, 0.1)

            # Stage 2: Transcribe (40%)
            logger.info("Transcribing...")
            result = self._model.transcribe(
                audio,
                language=language,
                batch_size=self.batch_size,
            )
            self._report_progress(progress_callback, 0.5)

            # Stage 3: Align (20%)
            logger.info("Aligning words...")
            result = whisperx.align(
                result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            self._report_progress(progress_callback, 0.7)

            # Stage 4: Diarize (20%)
            if enable_diarization and self._diarize_model is not None:
                logger.info("Running speaker diarization...")
                diarize_result = self._diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                result = whisperx.assign_word_speakers(diarize_result, result)
            self._report_progress(progress_callback, 0.9)

            # Stage 5: Convert to Segments (10%)
            segments = self._convert_to_segments(result)
            self._report_progress(progress_callback, 1.0)

            logger.info(f"Processed {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"WhisperX processing failed: {e}")
            raise WhisperXError(f"Processing failed: {e}") from e

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
            list[Segment]: Transcribed segments.
        """
        return self.process(
            audio_path=audio_path,
            language=language,
            enable_diarization=False,
            progress_callback=progress_callback,
        )

    def _convert_to_segments(self, result: dict) -> list[Segment]:
        """Convert WhisperX result to Segment objects."""
        segments = []

        for seg in result.get("segments", []):
            # Extract words if available
            words = []
            for w in seg.get("words", []):
                if "start" in w and "end" in w:
                    words.append(
                        Word(
                            word=w.get("word", ""),
                            start=w.get("start", 0.0),
                            end=w.get("end", 0.0),
                            probability=w.get("score", 1.0),
                        )
                    )

            segment = Segment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                original_text=seg.get("text", "").strip(),
                speaker_id=seg.get("speaker", "UNKNOWN"),
                words=words,
            )
            segments.append(segment)

        return segments

    def _report_progress(
        self, callback: ProgressCallback | None, progress: float
    ) -> None:
        """Report progress if callback provided."""
        if callback is not None:
            callback(progress)

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._model is not None

    @property
    def diarization_available(self) -> bool:
        """Check if diarization is available."""
        return self._diarize_model is not None

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        # Common languages supported by Whisper
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
            dict: Estimated VRAM for each component.
        """
        model_vram = {
            "tiny": 1.0,
            "base": 1.5,
            "small": 2.0,
            "medium": 3.5,
            "large-v2": 4.0,
            "large-v3": 4.0,
        }

        return {
            "whisper": model_vram.get(self.model_size, 4.0),
            "alignment": 0.5,
            "diarization": 1.0 if self.hf_token else 0.0,
            "total": (
                model_vram.get(self.model_size, 4.0)
                + 0.5
                + (1.0 if self.hf_token else 0.0)
            ),
        }
