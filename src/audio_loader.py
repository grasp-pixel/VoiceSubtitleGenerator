"""Audio loading and preprocessing for Voice Subtitle Generator."""

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .models import AudioInfo

logger = logging.getLogger(__name__)


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
            target_sample_rate: Output sample rate (16kHz recommended for faster-whisper).
            normalize: Whether to normalize audio levels.
        """
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self._temp_files: list[Path] = []

    def load(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and preprocess for faster-whisper.

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

        # Use ffmpeg directly to avoid pydub encoder bugs
        try:
            samples = self._load_with_ffmpeg(str(path))
        except Exception as e:
            raise AudioFormatError(f"Failed to load audio: {e}")

        # Normalize audio levels
        if self.normalize:
            samples = self._normalize_samples(samples)

        return samples

    def _load_with_ffmpeg(self, audio_path: str) -> np.ndarray:
        """
        Load audio using ffmpeg subprocess.

        Args:
            audio_path: Path to audio file.

        Returns:
            np.ndarray: Audio as float32 mono array.
        """
        # ffmpeg command: convert to 16-bit PCM WAV, mono, target sample rate
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-vn",  # No video
            "-ac", "1",  # Mono
            "-ar", str(self.target_sample_rate),  # Sample rate
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-f", "wav",  # WAV format
            "pipe:1",  # Output to stdout
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace")
            raise AudioFormatError(f"ffmpeg failed: {stderr}")
        except FileNotFoundError:
            raise AudioFormatError("ffmpeg not found. Please install ffmpeg.")

        # Read WAV data from stdout
        import io
        wav_data = io.BytesIO(result.stdout)
        samples, _ = sf.read(wav_data, dtype="float32")

        return samples

    def load_to_file(self, audio_path: str) -> str:
        """
        Load and preprocess audio, save to temporary WAV file.

        Useful when faster-whisper requires file path instead of array.

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
        Get audio file metadata using ffprobe.

        Args:
            audio_path: Path to audio file.

        Returns:
            AudioInfo: Audio metadata.
        """
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            import json

            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]

            result = subprocess.run(cmd, capture_output=True, check=True)
            data = json.loads(result.stdout)

            # Find audio stream
            audio_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break

            if audio_stream is None:
                raise AudioFormatError("No audio stream found")

            duration = float(data.get("format", {}).get("duration", 0))
            sample_rate = int(audio_stream.get("sample_rate", 0))
            channels = int(audio_stream.get("channels", 0))

        except subprocess.CalledProcessError as e:
            raise AudioFormatError(f"ffprobe failed: {e}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise AudioFormatError(f"Failed to parse audio info: {e}")

        return AudioInfo(
            path=str(path),
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
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
        Extract a segment from audio file using ffmpeg.

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

        duration = end - start

        cmd = [
            "ffmpeg",
            "-ss", str(start),
            "-i", str(path),
            "-t", str(duration),
            "-vn",
            "-ac", "1",
            "-ar", str(self.target_sample_rate),
            "-acodec", "pcm_s16le",
            "-f", "wav",
            "pipe:1",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace")
            raise AudioFormatError(f"ffmpeg failed: {stderr}")

        import io
        wav_data = io.BytesIO(result.stdout)
        samples, _ = sf.read(wav_data, dtype="float32")

        return samples

    def _normalize_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Normalize audio samples to target level.

        Args:
            samples: Audio samples as float32 array.

        Returns:
            np.ndarray: Normalized audio samples.
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(samples**2))

        if rms > 0:
            # Target RMS level (corresponds to about -20 dBFS)
            target_rms = 0.1
            gain = target_rms / rms
            # Limit gain to prevent clipping
            gain = min(gain, 10.0)
            samples = samples * gain
            # Clip to [-1, 1]
            samples = np.clip(samples, -1.0, 1.0)

        return samples

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
