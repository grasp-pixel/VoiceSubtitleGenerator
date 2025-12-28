"""File queue panel for Voice Subtitle Generator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import dearpygui.dearpygui as dpg

from .components import (
    FileStatus,
    THEME,
    format_duration,
    get_status_color,
    get_status_icon,
)

if TYPE_CHECKING:
    from src.models import Segment


@dataclass
class FileItem:
    """Represents a file in the queue."""

    path: str
    filename: str
    duration: float = 0.0
    speakers: int = 0
    status: FileStatus = FileStatus.PENDING
    progress: float = 0.0
    error: str | None = None

    # Processing results
    segments: list["Segment"] = field(default_factory=list)
    speaker_ids: list[str] = field(default_factory=list)

    # UI elements
    row_tag: int | str = 0
    progress_bar_tag: int | str = 0
    status_tag: int | str = 0


class FileListPanel:
    """
    File queue panel component.

    Displays list of files to process with status and progress.
    """

    def __init__(
        self,
        parent: int | str,
        on_select: Callable[[FileItem | None], None] | None = None,
        on_remove: Callable[[FileItem], None] | None = None,
    ):
        """
        Initialize file list panel.

        Args:
            parent: Parent container tag.
            on_select: Callback when file is selected.
            on_remove: Callback when file is removed.
        """
        self.parent = parent
        self.on_select = on_select
        self.on_remove = on_remove

        self._files: list[FileItem] = []
        self._selected_index: int = -1

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the panel UI."""
        with dpg.child_window(
            parent=self.parent,
            border=True,
            height=200,  # Reduced height for better layout
            tag="file_list_panel",
        ):
            # Header
            with dpg.group(horizontal=True):
                dpg.add_text("파일 대기열", color=THEME.text_primary)
                dpg.add_spacer(width=-1)
                self._count_label = dpg.add_text(
                    "0개 파일", color=THEME.text_secondary
                )

            dpg.add_separator()

            # Table
            with dpg.table(
                header_row=True,
                borders_innerH=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_outerV=True,
                resizable=True,
                policy=dpg.mvTable_SizingStretchProp,
                scrollY=True,
                tag="file_table",
            ):
                # Columns
                dpg.add_table_column(label="상태", width_fixed=True, init_width_or_weight=60)
                dpg.add_table_column(label="파일명", init_width_or_weight=200)
                dpg.add_table_column(label="길이", width_fixed=True, init_width_or_weight=80)
                dpg.add_table_column(label="화자", width_fixed=True, init_width_or_weight=60)
                dpg.add_table_column(label="진행률", init_width_or_weight=150)

    def add_file(self, file_path: str, duration: float = 0.0) -> FileItem:
        """
        Add a file to the queue.

        Args:
            file_path: Path to audio file.
            duration: Audio duration in seconds.

        Returns:
            FileItem: The created file item.
        """
        path = Path(file_path)

        item = FileItem(
            path=str(path),
            filename=path.name,
            duration=duration,
        )

        self._files.append(item)
        self._add_table_row(item)
        self._update_count()

        return item

    def add_files(self, file_paths: list[str]) -> list[FileItem]:
        """
        Add multiple files to the queue.

        Args:
            file_paths: List of file paths.

        Returns:
            list[FileItem]: Created file items.
        """
        items = []
        for path in file_paths:
            items.append(self.add_file(path))
        return items

    def _add_table_row(self, item: FileItem) -> None:
        """Add a row to the table for a file item."""
        with dpg.table_row(parent="file_table") as row:
            item.row_tag = row

            # Status column
            item.status_tag = dpg.add_text(
                get_status_icon(item.status),
                color=get_status_color(item.status),
            )

            # Filename column
            dpg.add_selectable(
                label=item.filename,
                span_columns=True,
                callback=lambda s, a, u: self._on_row_click(u),
                user_data=item,
            )

            # Duration column
            dpg.add_text(format_duration(item.duration) if item.duration > 0 else "-")

            # Speakers column
            dpg.add_text(str(item.speakers) if item.speakers > 0 else "-")

            # Progress column
            item.progress_bar_tag = dpg.add_progress_bar(
                default_value=item.progress,
                width=-1,
            )

    def _on_row_click(self, item: FileItem) -> None:
        """Handle row click."""
        try:
            self._selected_index = self._files.index(item)
        except ValueError:
            self._selected_index = -1

        if self.on_select:
            self.on_select(item)

    def remove_file(self, item: FileItem) -> None:
        """Remove a file from the queue."""
        if item in self._files:
            self._files.remove(item)
            dpg.delete_item(item.row_tag)
            self._update_count()

            if self.on_remove:
                self.on_remove(item)

    def remove_selected(self) -> None:
        """Remove the selected file."""
        if 0 <= self._selected_index < len(self._files):
            item = self._files[self._selected_index]
            self.remove_file(item)
            self._selected_index = -1

            if self.on_select:
                self.on_select(None)

    def clear(self) -> None:
        """Clear all files from the queue."""
        for item in self._files[:]:
            dpg.delete_item(item.row_tag)

        self._files.clear()
        self._selected_index = -1
        self._update_count()

        if self.on_select:
            self.on_select(None)

    def update_status(
        self,
        item: FileItem,
        status: FileStatus,
        progress: float = 0.0,
        error: str | None = None,
    ) -> None:
        """
        Update file status.

        Args:
            item: File item to update.
            status: New status.
            progress: Progress (0.0 to 1.0).
            error: Error message if status is ERROR.
        """
        item.status = status
        item.progress = progress
        item.error = error

        # Update UI
        if item.status_tag:
            dpg.set_value(item.status_tag, get_status_icon(status))
            dpg.configure_item(item.status_tag, color=get_status_color(status))

        if item.progress_bar_tag:
            dpg.set_value(item.progress_bar_tag, progress)

    def update_speakers(self, item: FileItem, count: int) -> None:
        """Update speaker count for a file."""
        item.speakers = count
        # Note: Would need to store speaker text tag to update

    def update_duration(self, item: FileItem, duration: float) -> None:
        """Update duration for a file."""
        item.duration = duration
        # Note: Would need to store duration text tag to update

    def set_result(
        self,
        item: FileItem,
        segments: list["Segment"],
        speaker_ids: list[str],
    ) -> None:
        """
        Store processing result for a file.

        Args:
            item: File item to update.
            segments: Processed segments.
            speaker_ids: List of detected speaker IDs.
        """
        item.segments = segments
        item.speaker_ids = speaker_ids
        item.speakers = len(speaker_ids)

    def _update_count(self) -> None:
        """Update file count display."""
        count = len(self._files)
        text = f"{count}개 파일"
        dpg.set_value(self._count_label, text)

    def get_selected(self) -> FileItem | None:
        """Get the currently selected file."""
        if 0 <= self._selected_index < len(self._files):
            return self._files[self._selected_index]
        return None

    def get_all_files(self) -> list[FileItem]:
        """Get all files in the queue."""
        return self._files.copy()

    def get_pending_files(self) -> list[FileItem]:
        """Get files that are pending processing."""
        return [f for f in self._files if f.status == FileStatus.PENDING]

    def get_file_count(self) -> int:
        """Get total file count."""
        return len(self._files)

    def get_pending_count(self) -> int:
        """Get pending file count."""
        return len(self.get_pending_files())

    def get_completed_count(self) -> int:
        """Get completed file count."""
        return len([f for f in self._files if f.status == FileStatus.DONE])

    def has_files(self) -> bool:
        """Check if there are any files in the queue."""
        return len(self._files) > 0

    def has_pending(self) -> bool:
        """Check if there are pending files."""
        return self.get_pending_count() > 0
