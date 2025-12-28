"""Data models for Voice Subtitle Generator."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class ProcessingStage(Enum):
    """Processing pipeline stages."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    LOADING = "loading"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    WRITING = "writing"
    DONE = "done"
    ERROR = "error"


class SubtitleFormat(Enum):
    """Supported subtitle formats."""

    SRT = "srt"
    ASS = "ass"
    VTT = "vtt"


@dataclass
class Word:
    """Word-level timing information."""

    word: str
    start: float
    end: float
    probability: float = 1.0

    @property
    def duration(self) -> float:
        """Word duration in seconds."""
        return self.end - self.start


@dataclass
class Segment:
    """Single subtitle segment with timing and content."""

    start: float
    end: float
    original_text: str
    translated_text: str = ""
    speaker_id: str = "UNKNOWN"
    speaker_name: str = ""
    words: list[Word] = field(default_factory=list)

    # Future extensions
    audio_position: str = "center"  # "left", "center", "right"

    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end - self.start

    @property
    def display_speaker(self) -> str:
        """Get display name for speaker."""
        return self.speaker_name if self.speaker_name else self.speaker_id

    def to_srt_timing(self) -> str:
        """Convert to SRT timing format."""
        return f"{self._format_time_srt(self.start)} --> {self._format_time_srt(self.end)}"

    def to_ass_timing(self) -> tuple[str, str]:
        """Convert to ASS timing format (start, end)."""
        return self._format_time_ass(self.start), self._format_time_ass(self.end)

    @staticmethod
    def _format_time_srt(seconds: float) -> str:
        """Format seconds to SRT time: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_time_ass(seconds: float) -> str:
        """Format seconds to ASS time: H:MM:SS.cc"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


@dataclass
class SpeakerMapping:
    """Speaker ID to character mapping."""

    speaker_id: str
    name: str
    color: str = "FFFFFF"
    style_preset: str | None = None


@dataclass
class AudioInfo:
    """Audio file metadata."""

    path: str
    duration: float
    sample_rate: int
    channels: int
    format: str


@dataclass
class ProcessResult:
    """Result of processing a single file."""

    success: bool
    audio_path: str
    output_path: str
    segments: list[Segment] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    duration: float = 0.0
    error: str | None = None


@dataclass
class SubtitleStyle:
    """ASS subtitle style definition."""

    name: str
    font_name: str = "Malgun Gothic"
    font_size: int = 48
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000"
    outline_width: float = 2.0
    shadow_depth: float = 1.0
    bold: bool = False
    italic: bool = False
    alignment: int = 2  # Bottom center
    margin_v: int = 30
    margin_l: int = 20
    margin_r: int = 20

    def to_ass_style_line(self) -> str:
        """Generate ASS style line."""
        bold_val = -1 if self.bold else 0
        italic_val = -1 if self.italic else 0

        return (
            f"Style: {self.name},{self.font_name},{self.font_size},"
            f"{self.primary_color},{self.primary_color},{self.outline_color},{self.back_color},"
            f"{bold_val},{italic_val},0,0,100,100,0,0,1,"
            f"{self.outline_width},{self.shadow_depth},{self.alignment},"
            f"{self.margin_l},{self.margin_r},{self.margin_v},1"
        )


# Callback type aliases
ProgressCallback = Callable[[float], None]
StageCallback = Callable[[ProcessingStage], None]
LogCallback = Callable[[str, str], None]  # message, level


@dataclass
class PipelineCallbacks:
    """Callbacks for pipeline progress tracking."""

    on_stage_change: StageCallback | None = None
    on_progress: ProgressCallback | None = None
    on_segment: Callable[[Segment], None] | None = None
    on_error: Callable[[str], None] | None = None
    on_log: LogCallback | None = None  # For detailed logging


@dataclass
class BatchCallbacks:
    """Callbacks for batch processing."""

    on_file_start: Callable[[str, int, int], None] | None = None  # path, current, total
    on_file_complete: Callable[[ProcessResult], None] | None = None
    on_batch_complete: Callable[[list[ProcessResult]], None] | None = None
    pipeline_callbacks: PipelineCallbacks | None = None
