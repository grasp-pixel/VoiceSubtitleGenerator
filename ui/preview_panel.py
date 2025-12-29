"""Subtitle preview panel for Voice Subtitle Generator."""

import time
from dataclasses import dataclass, field
from typing import Callable

import dearpygui.dearpygui as dpg

from src.models import Segment

from .components import THEME


@dataclass
class StageProfile:
    """Profiling data for a single stage."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    sub_stages: dict[str, "StageProfile"] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.end_time > 0 and self.start_time > 0:
            return self.end_time - self.start_time
        return 0.0

    @property
    def is_running(self) -> bool:
        """Check if stage is currently running."""
        return self.start_time > 0 and self.end_time == 0

    def format_duration(self) -> str:
        """Format duration as string."""
        d = self.duration
        if d == 0:
            return "-"
        if d < 1:
            return f"{d*1000:.0f}ms"
        if d < 60:
            return f"{d:.1f}s"
        mins = int(d // 60)
        secs = d % 60
        return f"{mins}m {secs:.0f}s"


class PreviewPanel:
    """
    Subtitle preview panel component.

    Displays transcribed and translated segments with timing info.
    """

    def __init__(
        self,
        parent: int | str,
        include_original: bool = False,
    ):
        """
        Initialize preview panel.

        Args:
            parent: Parent container tag.
            include_original: Whether to show original text in preview.
        """
        self.parent = parent
        self.include_original = include_original

        self._segments: list[Segment] = []
        self._current_file: str = ""

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the panel UI."""
        with dpg.child_window(
            parent=self.parent,
            border=True,
            height=-1,  # Fill remaining space
            tag="preview_panel",
        ):
            # Header
            with dpg.group(horizontal=True):
                dpg.add_text("미리보기", color=THEME.text_primary)
                dpg.add_spacer(width=-1)

                self._file_label = dpg.add_text(
                    "선택된 파일 없음",
                    color=THEME.text_secondary,
                )

            dpg.add_separator()

            # Segment list container
            with dpg.child_window(
                border=False,
                height=-1,
                tag="segment_container",
            ) as self._segment_container:
                # Empty state
                self._empty_text = dpg.add_text(
                    "완료된 파일을 선택하면 자막을 미리 볼 수 있습니다",
                    color=THEME.text_secondary,
                )

    def set_file(self, filename: str) -> None:
        """Set the current file name."""
        self._current_file = filename
        dpg.set_value(
            self._file_label,
            filename if filename else "선택된 파일 없음",
        )

    def set_segments(self, segments: list[Segment]) -> None:
        """
        Set segments to display.

        Args:
            segments: List of segments to display.
        """
        self._segments = segments
        self._render_segments()

    def _render_segments(self) -> None:
        """Render segment list."""
        # Clear existing content
        dpg.delete_item(self._segment_container, children_only=True)

        if not self._segments:
            dpg.add_text(
                "표시할 구간이 없습니다",
                parent=self._segment_container,
                color=THEME.text_secondary,
            )
            return

        # Render each segment
        for i, segment in enumerate(self._segments):
            self._render_segment(i + 1, segment)

    def _render_segment(self, index: int, segment: Segment) -> None:
        """Render a single segment."""
        with dpg.group(parent=self._segment_container):
            # Header line: index and timing
            with dpg.group(horizontal=True):
                dpg.add_text(f"#{index}", color=THEME.accent)
                dpg.add_text(
                    segment.to_srt_timing(),
                    color=THEME.text_secondary,
                )

            # Original text (Japanese) - only if include_original is enabled
            # Displayed in smaller font, muted gray color
            if self.include_original and segment.original_text:
                with dpg.group(horizontal=True):
                    label = dpg.add_text("JA:", color=(120, 120, 120))
                    text = dpg.add_text(
                        segment.original_text,
                        wrap=0,
                        color=(140, 140, 140),  # Muted gray
                    )
                    # Apply smaller font if available
                    if dpg.does_item_exist("font_small"):
                        dpg.bind_item_font(label, "font_small")
                        dpg.bind_item_font(text, "font_small")

            # Translated text (Korean)
            if segment.translated_text:
                with dpg.group(horizontal=True):
                    dpg.add_text("KO:", color=THEME.success)
                    dpg.add_text(
                        segment.translated_text,
                        wrap=0,
                        color=THEME.text_primary,
                    )

            dpg.add_separator()

    def clear(self) -> None:
        """Clear the preview panel."""
        self._segments = []
        self._current_file = ""

        dpg.set_value(self._file_label, "선택된 파일 없음")
        dpg.delete_item(self._segment_container, children_only=True)
        dpg.add_text(
            "완료된 파일을 선택하면 자막을 미리 볼 수 있습니다",
            parent=self._segment_container,
            color=THEME.text_secondary,
        )

    def get_segment_count(self) -> int:
        """Get number of segments."""
        return len(self._segments)

    def get_segments(self) -> list[Segment]:
        """Get current segments."""
        return self._segments.copy()

    def scroll_to_segment(self, index: int) -> None:
        """
        Scroll to a specific segment.

        Args:
            index: Segment index (0-based).
        """
        # Note: DearPyGui doesn't have direct scroll-to API for child windows
        # This would require custom implementation
        pass

    def highlight_segment(self, index: int) -> None:
        """
        Highlight a specific segment.

        Args:
            index: Segment index (0-based).
        """
        # Note: Would need to track segment groups and modify their style
        pass


class ProcessingPanel:
    """
    Processing status panel component.

    Shows current processing stages, progress, and profiling dashboard.
    """

    STAGES = [
        ("initializing", "초기화"),
        ("loading", "오디오 로딩"),
        ("transcribing", "음성 인식"),
        ("translating", "번역"),
        ("writing", "자막 저장"),
    ]

    # Sub-stages for detailed profiling
    SUB_STAGES = {
        "transcribing": [("whisper", "Whisper 추론"), ("vad", "VAD 처리")],
        "translating": [("translate", "번역")],
    }

    def __init__(self, parent: int | str):
        """
        Initialize processing panel.

        Args:
            parent: Parent container tag.
        """
        self.parent = parent

        self._current_stage = 0
        self._stage_items: list[tuple[int, int]] = []  # (icon_tag, text_tag)

        # Profiling data
        self._profiles: dict[str, StageProfile] = {}
        self._profile_labels: dict[str, int] = {}  # stage_id -> dpg tag for time label
        self._total_start_time: float = 0.0
        self._file_info: dict[str, str] = {}  # Extra info like duration, segments

        # Device info for CPU warning
        self._device_info: dict[str, str] = {}  # stage_id -> device ("GPU" or "CPU")
        self._device_labels: dict[str, int] = {}  # stage_id -> dpg tag for device label

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the panel UI."""
        with dpg.child_window(
            parent=self.parent,
            border=True,
            height=460,
            tag="processing_panel",
        ):
            # Two-column layout
            with dpg.group(horizontal=True):
                # Left column: Stage list
                with dpg.child_window(width=300, border=False):
                    dpg.add_text("처리 단계", color=THEME.text_primary)
                    dpg.add_separator()

                    # Stage list with timing
                    for i, (stage_id, stage_name) in enumerate(self.STAGES):
                        with dpg.group(horizontal=True):
                            icon = dpg.add_text("[ ]", color=THEME.text_secondary)
                            text = dpg.add_text(stage_name, color=THEME.text_secondary)
                            self._stage_items.append((icon, text))

                    dpg.add_separator()

                    # Overall progress
                    dpg.add_text("전체 진행률", color=THEME.text_primary)
                    self._progress_bar = dpg.add_progress_bar(default_value=0.0, width=-1)
                    self._progress_label = dpg.add_text("0%", color=THEME.text_secondary)

                dpg.add_spacer(width=10)

                # Right column: Profiling dashboard
                with dpg.child_window(width=-1, border=True):
                    dpg.add_text("프로파일링", color=THEME.text_primary)
                    dpg.add_separator()

                    # Total time
                    with dpg.group(horizontal=True):
                        dpg.add_text("총 소요시간:", color=THEME.text_secondary)
                        self._total_time_label = dpg.add_text("-", color=THEME.accent)

                    dpg.add_spacer(height=3)

                    # Stage timing table
                    with dpg.table(
                        header_row=True,
                        borders_innerH=True,
                        borders_outerH=True,
                        borders_innerV=True,
                        borders_outerV=True,
                        row_background=True,
                    ):
                        dpg.add_table_column(label="단계", width_fixed=True, init_width_or_weight=100)
                        dpg.add_table_column(label="장치", width_fixed=True, init_width_or_weight=70)
                        dpg.add_table_column(label="소요시간", width_fixed=True, init_width_or_weight=100)
                        dpg.add_table_column(label="비율", width_fixed=True, init_width_or_weight=60)

                        for stage_id, stage_name in self.STAGES:
                            with dpg.table_row():
                                dpg.add_text(stage_name, color=THEME.text_secondary)
                                # Device column - only for transcribing and translating
                                if stage_id in ("transcribing", "translating"):
                                    self._device_labels[stage_id] = dpg.add_text(
                                        "-", color=THEME.text_secondary
                                    )
                                else:
                                    dpg.add_text("-", color=THEME.text_secondary)
                                self._profile_labels[f"{stage_id}_time"] = dpg.add_text(
                                    "-", color=THEME.text_secondary
                                )
                                self._profile_labels[f"{stage_id}_pct"] = dpg.add_text(
                                    "-", color=THEME.text_secondary
                                )

                    dpg.add_spacer(height=5)

                    # Extra info
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_text("오디오:", color=THEME.text_secondary)
                        self._audio_info_label = dpg.add_text("-", color=THEME.text_secondary)

                    with dpg.group(horizontal=True):
                        dpg.add_text("세그먼트:", color=THEME.text_secondary)
                        self._segment_info_label = dpg.add_text("-", color=THEME.text_secondary)

                    with dpg.group(horizontal=True):
                        dpg.add_text("처리속도:", color=THEME.text_secondary)
                        self._speed_label = dpg.add_text("-", color=THEME.text_secondary)

    def set_stage(self, stage_index: int) -> None:
        """
        Set current processing stage.

        Args:
            stage_index: Stage index (0-based).
        """
        # End previous stage if running
        if self._current_stage < len(self.STAGES):
            prev_stage_id = self.STAGES[self._current_stage][0]
            if prev_stage_id in self._profiles and self._profiles[prev_stage_id].is_running:
                self.end_stage(prev_stage_id)

        self._current_stage = stage_index

        # Start new stage
        if stage_index < len(self.STAGES):
            stage_id = self.STAGES[stage_index][0]
            self.start_stage(stage_id)

        for i, (icon, text) in enumerate(self._stage_items):
            if i < stage_index:
                # Completed
                dpg.set_value(icon, "[v]")
                dpg.configure_item(icon, color=THEME.success)
                dpg.configure_item(text, color=THEME.text_primary)
            elif i == stage_index:
                # Current
                dpg.set_value(icon, ">>>")
                dpg.configure_item(icon, color=THEME.info)
                dpg.configure_item(text, color=THEME.info)
            else:
                # Pending
                dpg.set_value(icon, "[ ]")
                dpg.configure_item(icon, color=THEME.text_secondary)
                dpg.configure_item(text, color=THEME.text_secondary)

    def set_stage_by_name(self, stage_name: str) -> None:
        """
        Set stage by name.

        Args:
            stage_name: Stage identifier.
        """
        stage_map = {name: i for i, (name, _) in enumerate(self.STAGES)}
        if stage_name in stage_map:
            self.set_stage(stage_map[stage_name])

    def set_progress(self, progress: float) -> None:
        """
        Set overall progress.

        Args:
            progress: Progress value (0.0 to 1.0).
        """
        dpg.set_value(self._progress_bar, progress)
        dpg.set_value(self._progress_label, f"{int(progress * 100)}%")

        # Update running time display
        if self._total_start_time > 0:
            elapsed = time.time() - self._total_start_time
            self._update_total_time(elapsed)
            # Real-time profile update
            self._update_running_stage_time()
            self._update_profile_display(elapsed)

    def set_complete(self) -> None:
        """Mark all stages as complete."""
        # End last stage
        if self._current_stage < len(self.STAGES):
            stage_id = self.STAGES[self._current_stage][0]
            if stage_id in self._profiles and self._profiles[stage_id].is_running:
                self.end_stage(stage_id)

        self.set_stage(len(self.STAGES))
        self.set_progress(1.0)

        # Final profile update
        if self._total_start_time > 0:
            total_time = time.time() - self._total_start_time
            self._update_total_time(total_time)
            self._update_profile_display(total_time)
            self._update_speed_info(total_time)

    def set_error(self, stage_index: int) -> None:
        """
        Mark stage as error.

        Args:
            stage_index: Stage that errored.
        """
        if 0 <= stage_index < len(self._stage_items):
            icon, text = self._stage_items[stage_index]
            dpg.set_value(icon, "[x]")
            dpg.configure_item(icon, color=THEME.error)
            dpg.configure_item(text, color=THEME.error)

    # Profiling methods

    def start_stage(self, stage_id: str) -> None:
        """Start timing a stage."""
        if stage_id not in self._profiles:
            stage_name = next((name for sid, name in self.STAGES if sid == stage_id), stage_id)
            self._profiles[stage_id] = StageProfile(name=stage_name)
        self._profiles[stage_id].start_time = time.time()
        self._profiles[stage_id].end_time = 0.0

    def end_stage(self, stage_id: str) -> None:
        """End timing a stage."""
        if stage_id in self._profiles:
            self._profiles[stage_id].end_time = time.time()
            self._update_stage_time(stage_id)

    def _update_stage_time(self, stage_id: str) -> None:
        """Update the time display for a stage."""
        if stage_id not in self._profiles:
            return

        profile = self._profiles[stage_id]
        time_label = self._profile_labels.get(f"{stage_id}_time")

        if time_label and dpg.does_item_exist(time_label):
            dpg.set_value(time_label, profile.format_duration())
            if profile.duration > 0:
                dpg.configure_item(time_label, color=THEME.text_primary)

    def _update_running_stage_time(self) -> None:
        """Update time display for currently running stages (real-time)."""
        current_time = time.time()
        for stage_id, profile in self._profiles.items():
            if profile.is_running:
                # Calculate running duration
                running_duration = current_time - profile.start_time
                time_label = self._profile_labels.get(f"{stage_id}_time")

                if time_label and dpg.does_item_exist(time_label):
                    # Format running time with indicator
                    if running_duration < 1:
                        time_str = f"{running_duration*1000:.0f}ms..."
                    elif running_duration < 60:
                        time_str = f"{running_duration:.1f}s..."
                    else:
                        mins = int(running_duration // 60)
                        secs = running_duration % 60
                        time_str = f"{mins}m {secs:.0f}s..."

                    dpg.set_value(time_label, time_str)
                    dpg.configure_item(time_label, color=THEME.info)  # Highlight running

    def _update_total_time(self, elapsed: float) -> None:
        """Update total time display."""
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.0f}s"

        dpg.set_value(self._total_time_label, time_str)

    def _update_profile_display(self, total_time: float) -> None:
        """Update all profile percentage displays."""
        if total_time <= 0:
            return

        for stage_id, profile in self._profiles.items():
            pct_label = self._profile_labels.get(f"{stage_id}_pct")
            if pct_label and dpg.does_item_exist(pct_label):
                if profile.duration > 0:
                    pct = (profile.duration / total_time) * 100
                    dpg.set_value(pct_label, f"{pct:.0f}%")
                    dpg.configure_item(pct_label, color=THEME.text_primary)

    def set_audio_info(self, duration: float, sample_rate: int) -> None:
        """Set audio file information."""
        mins = int(duration // 60)
        secs = duration % 60
        info = f"{mins}:{secs:04.1f} ({sample_rate}Hz)"
        dpg.set_value(self._audio_info_label, info)
        self._file_info["audio_duration"] = duration

    def set_segment_info(self, count: int) -> None:
        """Set segment count information."""
        info = f"{count}개"
        dpg.set_value(self._segment_info_label, info)

    def _update_speed_info(self, total_time: float) -> None:
        """Update processing speed information."""
        audio_duration = self._file_info.get("audio_duration", 0)
        if audio_duration > 0 and total_time > 0:
            speed = audio_duration / total_time
            dpg.set_value(self._speed_label, f"{speed:.1f}x 실시간")
            if speed >= 1.0:
                dpg.configure_item(self._speed_label, color=THEME.success)
            else:
                dpg.configure_item(self._speed_label, color=THEME.warning)

    def set_device_info(
        self, whisper_device: str, translator_using_gpu: bool | None
    ) -> None:
        """
        Set device information for Whisper and Translator.

        Displays CPU warning if not using GPU.

        Args:
            whisper_device: Whisper device ("cuda" or "cpu").
            translator_using_gpu: Whether translator is using GPU (None if not loaded).
        """
        # Whisper device
        whisper_is_gpu = whisper_device.lower() in ("cuda", "gpu")
        self._device_info["transcribing"] = "GPU" if whisper_is_gpu else "CPU"

        # Translator device (None means not loaded/not used)
        if translator_using_gpu is None:
            self._device_info["translating"] = "-"
        elif translator_using_gpu:
            self._device_info["translating"] = "GPU"
        else:
            self._device_info["translating"] = "CPU"

        # Update UI labels
        for stage_id in ("transcribing", "translating"):
            label_tag = self._device_labels.get(stage_id)
            if label_tag and dpg.does_item_exist(label_tag):
                device = self._device_info.get(stage_id, "-")
                dpg.set_value(label_tag, device)

                if device == "CPU":
                    # Show warning color for CPU
                    dpg.configure_item(label_tag, color=THEME.warning)
                elif device == "GPU":
                    dpg.configure_item(label_tag, color=THEME.success)
                else:
                    dpg.configure_item(label_tag, color=THEME.text_secondary)

    def reset(self) -> None:
        """Reset all stages and profiling."""
        self._current_stage = 0
        self._profiles.clear()
        self._file_info.clear()
        self._total_start_time = time.time()

        for icon, text in self._stage_items:
            dpg.set_value(icon, "[ ]")
            dpg.configure_item(icon, color=THEME.text_secondary)
            dpg.configure_item(text, color=THEME.text_secondary)

        # Reset profile labels
        dpg.set_value(self._total_time_label, "-")
        dpg.configure_item(self._total_time_label, color=THEME.accent)

        for stage_id, _ in self.STAGES:
            time_label = self._profile_labels.get(f"{stage_id}_time")
            pct_label = self._profile_labels.get(f"{stage_id}_pct")
            if time_label and dpg.does_item_exist(time_label):
                dpg.set_value(time_label, "-")
                dpg.configure_item(time_label, color=THEME.text_secondary)
            if pct_label and dpg.does_item_exist(pct_label):
                dpg.set_value(pct_label, "-")
                dpg.configure_item(pct_label, color=THEME.text_secondary)

        # Reset info labels
        dpg.set_value(self._audio_info_label, "-")
        dpg.set_value(self._segment_info_label, "-")
        dpg.set_value(self._speed_label, "-")
        dpg.configure_item(self._speed_label, color=THEME.text_secondary)

        self.set_progress(0.0)


class LogPanel:
    """
    Log panel component for displaying processing logs.
    """

    MAX_LINES = 200

    def __init__(self, parent: int | str):
        """
        Initialize log panel.

        Args:
            parent: Parent container tag.
        """
        self.parent = parent
        self._lines: list[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the panel UI."""
        with dpg.child_window(
            parent=self.parent,
            border=True,
            height=150,
            tag="log_panel",
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("로그", color=THEME.text_primary)
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="지우기",
                    callback=self.clear,
                    width=80,
                    height=26,
                )

            dpg.add_separator()

            # Log content area
            with dpg.child_window(
                border=False,
                height=-1,
                tag="log_content",
            ):
                self._log_text = dpg.add_text(
                    "",
                    wrap=0,
                    color=THEME.text_secondary,
                )

    def log(self, message: str, level: str = "info") -> None:
        """
        Add a log message.

        Args:
            message: Log message.
            level: Log level (info, warning, error, success).
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Color based on level
        prefix = {
            "info": "",
            "warning": "[!] ",
            "error": "[X] ",
            "success": "[v] ",
        }.get(level, "")

        line = f"[{timestamp}] {prefix}{message}"
        self._lines.append(line)

        # Limit lines
        if len(self._lines) > self.MAX_LINES:
            self._lines = self._lines[-self.MAX_LINES:]

        # Update display
        dpg.set_value(self._log_text, "\n".join(self._lines))

        # Auto-scroll to bottom
        dpg.set_y_scroll("log_content", dpg.get_y_scroll_max("log_content"))

    def info(self, message: str) -> None:
        """Add info log."""
        self.log(message, "info")

    def warning(self, message: str) -> None:
        """Add warning log."""
        self.log(message, "warning")

    def error(self, message: str) -> None:
        """Add error log."""
        self.log(message, "error")

    def success(self, message: str) -> None:
        """Add success log."""
        self.log(message, "success")

    def clear(self) -> None:
        """Clear all logs."""
        self._lines.clear()
        dpg.set_value(self._log_text, "")
