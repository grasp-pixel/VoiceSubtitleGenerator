"""Audio loading and preprocessing for Voice Subtitle Generator."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from .models import AudioInfo


class AudioFormatError(Exception):
    """Raised when audio format is not supported or invalid."""

    pass


class AudioLoader:
    """Load and preprocess audio/video files."""

    # Audio formats
    AUDIO_FORMATS = ["mp3", "wav", "flac", "m4a", "ogg", "wma", "aac", "opus"]

    # Video formats (audio will be extracted via ffmpeg)
    VIDEO_FORMATS = ["mp4", "mkv", "avi", "webm", "mov", "wmv", "flv", "ts", "m2ts"]

    # All supported media formats
    SUPPORTED_FORMATS = AUDIO_FORMATS + VIDEO_FORMATS

    def __init__(
        self,
        target_sample_rate: int = 16000,
        normalize: bool = True,
    ):
        """
        Initialize audio loader.

        Args:
            target_sample_rate: Output sample rate (16kHz recommended for WhisperX).
            normalize: Whether to normalize audio levels.
        """
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self._temp_files: list[Path] = []

    def load(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and preprocess for WhisperX.

        Args:
            audio_path: Path to audio file.

        Returns:
            np.ndarray: Preprocessed audio as float32 mono array.

        Raises:
            FileNotFoundError: If file doesn't exist.
            AudioFormatError: If format is not supported.
        """
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        suffix = path.suffix.lower().lstrip(".")
        if suffix not in self.SUPPORTED_FORMATS:
            raise AudioFormatError(
                f"Unsupported format: {suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Load and convert using pydub
        try:
            audio = AudioSegment.from_file(str(path))
        except Exception as e:
            raise AudioFormatError(f"Failed to load audio: {e}")

        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample if needed
        if audio.frame_rate != self.target_sample_rate:
            audio = audio.set_frame_rate(self.target_sample_rate)

        # Normalize
        if self.normalize:
            audio = self._normalize_audio(audio)

        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Normalize to [-1, 1] range
        max_val = float(2 ** (audio.sample_width * 8 - 1))
        samples = samples / max_val

        return samples

    def load_to_file(self, audio_path: str) -> str:
        """
        Load and preprocess audio, save to temporary WAV file.

        Useful when WhisperX requires file path instead of array.

        Args:
            audio_path: Path to input audio file.

        Returns:
            str: Path to temporary WAV file.
        """
        samples = self.load(audio_path)

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="vsg_"
        )
        temp_path = Path(temp_file.name)
        temp_file.close()

        # Save as WAV
        sf.write(str(temp_path), samples, self.target_sample_rate)

        self._temp_files.append(temp_path)
        return str(temp_path)

    def get_info(self, audio_path: str) -> AudioInfo:
        """
        Get audio file metadata.

        Args:
            audio_path: Path to audio file.

        Returns:
            AudioInfo: Audio metadata.
        """
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio = AudioSegment.from_file(str(path))
        except Exception as e:
            raise AudioFormatError(f"Failed to load audio: {e}")

        return AudioInfo(
            path=str(path),
            duration=len(audio) / 1000.0,  # Convert ms to seconds
            sample_rate=audio.frame_rate,
            channels=audio.channels,
            format=path.suffix.lower().lstrip("."),
        )

    def get_duration(self, audio_path: str) -> float:
        """
        Get audio duration in seconds.

        Args:
            audio_path: Path to audio file.

        Returns:
            float: Duration in seconds.
        """
        return self.get_info(audio_path).duration

    def extract_segment(
        self,
        audio_path: str,
        start: float,
        end: float,
    ) -> np.ndarray:
        """
        Extract a segment from audio file.

        Args:
            audio_path: Path to audio file.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            np.ndarray: Audio segment as float32 array.
        """
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio = AudioSegment.from_file(str(path))
        except Exception as e:
            raise AudioFormatError(f"Failed to load audio: {e}")

        # Convert to milliseconds
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)

        # Extract segment
        segment = audio[start_ms:end_ms]

        # Convert to mono and resample
        if segment.channels > 1:
            segment = segment.set_channels(1)
        if segment.frame_rate != self.target_sample_rate:
            segment = segment.set_frame_rate(self.target_sample_rate)

        # Convert to numpy
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        max_val = float(2 ** (segment.sample_width * 8 - 1))
        samples = samples / max_val

        return samples

    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        Normalize audio levels.

        Args:
            audio: Input audio segment.

        Returns:
            AudioSegment: Normalized audio.
        """
        # Calculate target dBFS (decibels relative to full scale)
        target_dbfs = -20.0

        # Get current dBFS
        current_dbfs = audio.dBFS

        # Calculate gain needed
        if current_dbfs != float("-inf"):
            gain = target_dbfs - current_dbfs
            # Limit gain to prevent clipping
            gain = min(gain, 20.0)
            audio = audio.apply_gain(gain)

        return audio

    def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
        self._temp_files.clear()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.cleanup()

    @classmethod
    def supported_formats(cls) -> list[str]:
        """Get list of supported audio formats."""
        return cls.SUPPORTED_FORMATS.copy()

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file format is supported."""
        suffix = Path(file_path).suffix.lower().lstrip(".")
        return suffix in cls.SUPPORTED_FORMATS
