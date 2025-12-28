"""SpeechBrain speaker diarization implementation."""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any

import torch

from .base import (
    DiarizationEngine,
    DiarizationResult,
    DiarizationTurn,
    ProgressCallback,
)

if TYPE_CHECKING:
    from ..config import DiarizationConfig

logger = logging.getLogger(__name__)


class SpeechBrainEngine(DiarizationEngine):
    """
    SpeechBrain-based speaker diarization.

    Uses SpeechBrain's ECAPA-TDNN embeddings for speaker diarization.
    Does not require HuggingFace token.

    Note: SpeechBrain's diarization pipeline is less mature than PyAnnote.
    This implementation uses embedding extraction + clustering approach.
    """

    def __init__(
        self,
        device: str = "cuda",
        embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb",
    ):
        """
        Initialize SpeechBrain engine.

        Args:
            device: Compute device ("cuda" or "cpu").
            embedding_model: Model name for speaker embeddings.
        """
        self.device = device
        self.embedding_model = embedding_model

        self._encoder: Any = None
        self._vad: Any = None

    @classmethod
    def from_config(cls, config: DiarizationConfig) -> SpeechBrainEngine:
        """Create engine from config."""
        return cls(
            device=config.device,
        )

    def load_model(self) -> None:
        """Load the speaker embedding model."""
        if self._encoder is not None:
            return

        logger.info(f"Loading SpeechBrain embedding model: {self.embedding_model}")

        try:
            from speechbrain.inference import EncoderClassifier, VAD

            self._encoder = EncoderClassifier.from_hparams(
                source=self.embedding_model,
                run_opts={"device": self.device},
            )

            # Load VAD for speech segmentation
            self._vad = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                run_opts={"device": self.device},
            )

            logger.info("SpeechBrain models loaded successfully")

        except ImportError as e:
            raise ImportError(
                "SpeechBrain is not installed. "
                "Install it with: pip install speechbrain"
            ) from e

    def unload_model(self) -> None:
        """Unload models and free memory."""
        self._encoder = None
        self._vad = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("SpeechBrain models unloaded")

    def diarize(
        self,
        audio_path: str,
        min_speakers: int = 1,
        max_speakers: int = 10,
        progress_callback: ProgressCallback | None = None,
    ) -> DiarizationResult:
        """
        Run speaker diarization using embedding clustering.

        Args:
            audio_path: Path to audio file.
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            progress_callback: Progress callback.

        Returns:
            DiarizationResult with speaker turns.
        """
        if self._encoder is None:
            self.load_model()

        logger.info(f"Running SpeechBrain diarization: {audio_path}")

        try:
            import torchaudio
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import pdist

            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if progress_callback:
                progress_callback(0.1)

            # Get speech segments using VAD
            speech_segments = self._get_speech_segments(waveform, sample_rate)

            if progress_callback:
                progress_callback(0.3)

            if not speech_segments:
                logger.warning("No speech segments found")
                return DiarizationResult(turns=[], speakers=[])

            # Extract embeddings for each segment
            embeddings = []
            for start, end in speech_segments:
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment_audio = waveform[:, start_sample:end_sample]

                if segment_audio.shape[1] < sample_rate * 0.5:  # Skip very short
                    embeddings.append(None)
                    continue

                embedding = self._encoder.encode_batch(segment_audio)
                embeddings.append(embedding.squeeze().cpu().numpy())

            if progress_callback:
                progress_callback(0.6)

            # Filter valid embeddings
            valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
            valid_embeddings = [embeddings[i] for i in valid_indices]

            if len(valid_embeddings) < 2:
                # Only one speaker
                turns = [
                    DiarizationTurn(
                        start=speech_segments[i][0],
                        end=speech_segments[i][1],
                        speaker_id="SPEAKER_00",
                    )
                    for i in valid_indices
                ]
                return DiarizationResult(turns=turns, speakers=["SPEAKER_00"])

            # Cluster embeddings
            import numpy as np

            embeddings_array = np.stack(valid_embeddings)
            distances = pdist(embeddings_array, metric="cosine")
            linkage_matrix = linkage(distances, method="ward")

            # Determine number of clusters
            n_clusters = min(max_speakers, len(valid_embeddings))
            n_clusters = max(min_speakers, n_clusters)

            # Cut dendrogram to get cluster assignments
            cluster_labels = fcluster(
                linkage_matrix, n_clusters, criterion="maxclust"
            )

            if progress_callback:
                progress_callback(0.9)

            # Build turns
            turns: list[DiarizationTurn] = []
            speakers: set[str] = set()

            for idx, cluster_id in enumerate(cluster_labels):
                original_idx = valid_indices[idx]
                start, end = speech_segments[original_idx]
                speaker_id = f"SPEAKER_{cluster_id - 1:02d}"

                turns.append(
                    DiarizationTurn(
                        start=start,
                        end=end,
                        speaker_id=speaker_id,
                    )
                )
                speakers.add(speaker_id)

            # Sort by start time
            turns.sort(key=lambda t: t.start)

            if progress_callback:
                progress_callback(1.0)

            logger.info(f"Found {len(speakers)} speakers, {len(turns)} turns")

            return DiarizationResult(
                turns=turns,
                speakers=sorted(speakers),
            )

        except Exception as e:
            logger.error(f"SpeechBrain diarization failed: {e}")
            raise

    def _get_speech_segments(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        min_duration: float = 0.5,
    ) -> list[tuple[float, float]]:
        """
        Extract speech segments using VAD.

        Args:
            waveform: Audio waveform tensor.
            sample_rate: Sample rate.
            min_duration: Minimum segment duration.

        Returns:
            List of (start, end) tuples in seconds.
        """
        if self._vad is None:
            # Simple energy-based VAD fallback
            return self._energy_vad(waveform, sample_rate, min_duration)

        try:
            # Use SpeechBrain VAD
            boundaries = self._vad.get_speech_segments(
                waveform.squeeze(),
                large_chunk_size=30,
                small_chunk_size=10,
            )

            segments = []
            for boundary in boundaries:
                start = boundary[0].item() / sample_rate
                end = boundary[1].item() / sample_rate
                if end - start >= min_duration:
                    segments.append((start, end))

            return segments

        except Exception as e:
            logger.warning(f"VAD failed, using energy-based fallback: {e}")
            return self._energy_vad(waveform, sample_rate, min_duration)

    def _energy_vad(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        min_duration: float = 0.5,
    ) -> list[tuple[float, float]]:
        """Simple energy-based VAD fallback."""
        # Window size of 0.5 seconds
        window_size = int(sample_rate * 0.5)
        hop_size = int(sample_rate * 0.25)

        audio = waveform.squeeze()
        total_samples = audio.shape[0]

        # Calculate energy per window
        energies = []
        for i in range(0, total_samples - window_size, hop_size):
            window = audio[i : i + window_size]
            energy = (window**2).mean().item()
            energies.append((i / sample_rate, energy))

        if not energies:
            return [(0, total_samples / sample_rate)]

        # Threshold at 10% of max energy
        max_energy = max(e[1] for e in energies)
        threshold = max_energy * 0.1

        # Find speech regions
        segments = []
        in_speech = False
        start = 0

        for time, energy in energies:
            if energy > threshold and not in_speech:
                in_speech = True
                start = time
            elif energy <= threshold and in_speech:
                in_speech = False
                if time - start >= min_duration:
                    segments.append((start, time))

        # Handle case where speech continues to end
        if in_speech:
            end_time = total_samples / sample_rate
            if end_time - start >= min_duration:
                segments.append((start, end_time))

        return segments if segments else [(0, total_samples / sample_rate)]

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._encoder is not None

    @property
    def requires_auth_token(self) -> bool:
        """SpeechBrain does not require auth token."""
        return False

    def estimate_vram_usage(self) -> dict[str, float]:
        """
        Estimate VRAM usage in GB.

        Returns:
            dict: Estimated VRAM for diarization.
        """
        return {
            "embedding_model": 0.5,
            "vad": 0.3,
            "total": 0.8,
        }
