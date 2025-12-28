"""PyAnnote speaker diarization implementation."""

from __future__ import annotations

import gc
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from .base import (
    DiarizationEngine,
    DiarizationResult,
    DiarizationTurn,
    ProgressCallback,
)

if TYPE_CHECKING:
    from pyannote.audio import Pipeline

    from ..config import DiarizationConfig

logger = logging.getLogger(__name__)


class PyAnnoteEngine(DiarizationEngine):
    """
    PyAnnote-based speaker diarization.

    Uses pyannote.audio directly for speaker diarization.
    Requires HuggingFace token for model access.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "cuda",
        hf_token: str | None = None,
    ):
        """
        Initialize PyAnnote engine.

        Args:
            model_name: HuggingFace model name for diarization.
            device: Compute device ("cuda" or "cpu").
            hf_token: HuggingFace token for model access.
        """
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")

        self._pipeline: Pipeline | None = None

    @classmethod
    def from_config(cls, config: DiarizationConfig) -> PyAnnoteEngine:
        """Create engine from config."""
        return cls(
            model_name=config.model_name,
            device=config.device,
            hf_token=config.get_hf_token(),
        )

    def load_model(self) -> None:
        """Load the diarization pipeline."""
        if self._pipeline is not None:
            return

        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required for pyannote models. "
                "Set HF_TOKEN environment variable or provide hf_token in config."
            )

        logger.info(f"Loading pyannote pipeline: {self.model_name}")

        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token,
        )

        # Move to GPU if available
        if self.device == "cuda" and torch.cuda.is_available():
            self._pipeline.to(torch.device("cuda"))

        logger.info("PyAnnote pipeline loaded successfully")

    def unload_model(self) -> None:
        """Unload the pipeline and free memory."""
        if self._pipeline is None:
            return

        self._pipeline = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("PyAnnote pipeline unloaded")

    def diarize(
        self,
        audio_path: str,
        min_speakers: int = 1,
        max_speakers: int = 10,
        progress_callback: ProgressCallback | None = None,
    ) -> DiarizationResult:
        """
        Run speaker diarization.

        Args:
            audio_path: Path to audio file.
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            progress_callback: Progress callback.

        Returns:
            DiarizationResult with speaker turns.
        """
        if self._pipeline is None:
            self.load_model()

        logger.info(f"Running pyannote diarization: {audio_path}")

        # Create progress hook if callback provided
        hook = None
        if progress_callback:
            try:
                from pyannote.audio.pipelines.utils.hook import ProgressHook

                hook = ProgressHook()
            except ImportError:
                pass

        # Run diarization
        with hook if hook else nullcontext():
            diarization = self._pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        # Report completion
        if progress_callback:
            progress_callback(1.0)

        # Convert to our format
        turns: list[DiarizationTurn] = []
        speakers: set[str] = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(
                    start=turn.start,
                    end=turn.end,
                    speaker_id=speaker,
                )
            )
            speakers.add(speaker)

        logger.info(f"Found {len(speakers)} speakers, {len(turns)} turns")

        return DiarizationResult(
            turns=turns,
            speakers=sorted(speakers),
        )

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self._pipeline is not None

    @property
    def requires_auth_token(self) -> bool:
        """PyAnnote requires HuggingFace token."""
        return True

    def estimate_vram_usage(self) -> dict[str, float]:
        """
        Estimate VRAM usage in GB.

        Returns:
            dict: Estimated VRAM for diarization.
        """
        return {
            "diarization": 1.0,
            "total": 1.0,
        }
