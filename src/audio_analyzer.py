"""Audio analyzer for binaural/stereo position detection."""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioPosition(Enum):
    """Audio spatial position."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass
class PositionResult:
    """Result of position analysis."""

    position: AudioPosition
    confidence: float  # 0.0 to 1.0
    left_energy: float
    right_energy: float
    balance: float  # -1.0 (full left) to 1.0 (full right)


@dataclass
class SegmentPosition:
    """Position information for a segment."""

    start: float
    end: float
    position: AudioPosition
    confidence: float
    balance: float


class AudioPositionAnalyzer:
    """
    Analyzes stereo/binaural audio to detect spatial position.

    Uses channel energy comparison to determine if audio is
    predominantly from left, right, or center.
    """

    # Thresholds for position detection
    LEFT_THRESHOLD = -0.3   # Balance below this = left
    RIGHT_THRESHOLD = 0.3   # Balance above this = right
    MIN_CONFIDENCE = 0.6    # Minimum confidence for position detection

    def __init__(
        self,
        left_threshold: float = -0.3,
        right_threshold: float = 0.3,
        min_energy_db: float = -60.0,
    ):
        """
        Initialize analyzer.

        Args:
            left_threshold: Balance threshold for left detection.
            right_threshold: Balance threshold for right detection.
            min_energy_db: Minimum energy in dB to consider.
        """
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold
        self.min_energy_db = min_energy_db

        self._audio_data: np.ndarray | None = None
        self._sample_rate: int = 0
        self._is_stereo: bool = False

    def load_audio(self, audio_path: str | Path) -> bool:
        """
        Load audio file for analysis.

        Args:
            audio_path: Path to audio file.

        Returns:
            True if loaded successfully and is stereo.
        """
        try:
            self._audio_data, self._sample_rate = sf.read(str(audio_path))

            # Check if stereo
            if len(self._audio_data.shape) == 1:
                logger.warning("Audio is mono, position detection not available")
                self._is_stereo = False
                return False

            if self._audio_data.shape[1] < 2:
                logger.warning("Audio has less than 2 channels")
                self._is_stereo = False
                return False

            self._is_stereo = True
            logger.info(
                f"Loaded stereo audio: {self._sample_rate}Hz, "
                f"{len(self._audio_data)} samples"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            self._audio_data = None
            self._is_stereo = False
            return False

    def _calculate_rms_energy(self, samples: np.ndarray) -> float:
        """
        Calculate RMS energy of samples.

        Args:
            samples: Audio samples.

        Returns:
            RMS energy value.
        """
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples ** 2)))

    def _energy_to_db(self, energy: float) -> float:
        """
        Convert energy to decibels.

        Args:
            energy: RMS energy value.

        Returns:
            Energy in dB.
        """
        if energy <= 0:
            return -100.0
        return 20 * np.log10(energy)

    def analyze_position(
        self,
        start: float,
        end: float,
    ) -> PositionResult:
        """
        Analyze audio position for a time range.

        Args:
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            PositionResult with detected position.
        """
        if self._audio_data is None or not self._is_stereo:
            return PositionResult(
                position=AudioPosition.UNKNOWN,
                confidence=0.0,
                left_energy=0.0,
                right_energy=0.0,
                balance=0.0,
            )

        # Convert time to sample indices
        start_sample = int(start * self._sample_rate)
        end_sample = int(end * self._sample_rate)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(self._audio_data), end_sample)

        if start_sample >= end_sample:
            return PositionResult(
                position=AudioPosition.UNKNOWN,
                confidence=0.0,
                left_energy=0.0,
                right_energy=0.0,
                balance=0.0,
            )

        # Extract segment
        segment = self._audio_data[start_sample:end_sample]

        # Get left and right channels
        left_channel = segment[:, 0]
        right_channel = segment[:, 1]

        # Calculate energy
        left_energy = self._calculate_rms_energy(left_channel)
        right_energy = self._calculate_rms_energy(right_channel)

        # Check minimum energy
        total_energy = left_energy + right_energy
        if self._energy_to_db(total_energy / 2) < self.min_energy_db:
            return PositionResult(
                position=AudioPosition.UNKNOWN,
                confidence=0.0,
                left_energy=left_energy,
                right_energy=right_energy,
                balance=0.0,
            )

        # Calculate balance (-1.0 to 1.0)
        if total_energy > 0:
            balance = (right_energy - left_energy) / total_energy
        else:
            balance = 0.0

        # Determine position
        if balance < self.left_threshold:
            position = AudioPosition.LEFT
            confidence = min(1.0, abs(balance) / abs(self.left_threshold))
        elif balance > self.right_threshold:
            position = AudioPosition.RIGHT
            confidence = min(1.0, abs(balance) / abs(self.right_threshold))
        else:
            position = AudioPosition.CENTER
            # Confidence is higher when closer to 0
            confidence = 1.0 - abs(balance) / max(abs(self.left_threshold), abs(self.right_threshold))

        return PositionResult(
            position=position,
            confidence=confidence,
            left_energy=left_energy,
            right_energy=right_energy,
            balance=balance,
        )

    def analyze_segments(
        self,
        segments: list[tuple[float, float]],
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[SegmentPosition]:
        """
        Analyze positions for multiple segments.

        Args:
            segments: List of (start, end) time tuples.
            progress_callback: Optional progress callback.

        Returns:
            List of SegmentPosition results.
        """
        results = []
        total = len(segments)

        for i, (start, end) in enumerate(segments):
            result = self.analyze_position(start, end)

            results.append(
                SegmentPosition(
                    start=start,
                    end=end,
                    position=result.position,
                    confidence=result.confidence,
                    balance=result.balance,
                )
            )

            if progress_callback and total > 0:
                progress_callback((i + 1) / total)

        return results

    def get_position_for_subtitle(
        self,
        position: AudioPosition,
        video_width: int = 1920,
        margin: int = 100,
    ) -> tuple[int, str]:
        """
        Get subtitle position parameters based on audio position.

        Args:
            position: Detected audio position.
            video_width: Video width in pixels.
            margin: Margin from edges.

        Returns:
            Tuple of (x_position, alignment_tag).
        """
        if position == AudioPosition.LEFT:
            # Left side of screen
            return margin, "\\an1"  # Bottom-left alignment
        elif position == AudioPosition.RIGHT:
            # Right side of screen
            return video_width - margin, "\\an3"  # Bottom-right alignment
        else:
            # Center
            return video_width // 2, "\\an2"  # Bottom-center alignment

    def cleanup(self) -> None:
        """Release loaded audio data."""
        self._audio_data = None
        self._sample_rate = 0
        self._is_stereo = False

    @property
    def is_stereo(self) -> bool:
        """Check if loaded audio is stereo."""
        return self._is_stereo

    @property
    def duration(self) -> float:
        """Get audio duration in seconds."""
        if self._audio_data is None or self._sample_rate == 0:
            return 0.0
        return len(self._audio_data) / self._sample_rate


def analyze_audio_positions(
    audio_path: str | Path,
    segments: list[tuple[float, float]],
) -> dict[tuple[float, float], AudioPosition]:
    """
    Convenience function to analyze positions for segments.

    Args:
        audio_path: Path to audio file.
        segments: List of (start, end) time tuples.

    Returns:
        Dictionary mapping segment times to positions.
    """
    analyzer = AudioPositionAnalyzer()

    if not analyzer.load_audio(audio_path):
        return {seg: AudioPosition.CENTER for seg in segments}

    try:
        results = analyzer.analyze_segments(segments)
        return {
            (r.start, r.end): r.position
            for r in results
        }
    finally:
        analyzer.cleanup()
