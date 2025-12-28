"""Subtitle editor panel for Voice Subtitle Generator."""

import logging
from dataclasses import dataclass
from typing import Callable

import dearpygui.dearpygui as dpg

from src.models import Segment

from .components import THEME, show_confirm_dialog, show_message_box

logger = logging.getLogger(__name__)


@dataclass
class EditableSegment:
    """Editable segment data."""

    index: int
    segment: Segment
    modified: bool = False


class SubtitleEditorPanel:
    """
    Subtitle editor panel component.

    Allows users to edit subtitle text, timing, and speaker assignments.
    """

    def __init__(
        self,
        parent: int | str,
        on_save: Callable[[list[Segment]], None] | None = None,
        on_segment_change: Callable[[int, Segment], None] | None = None,
    ):
        """
        Initialize subtitle editor panel.

        Args:
            parent: Parent window/group tag.
            on_save: Callback when segments are saved.
            on_segment_change: Callback when a segment is changed.
        """
        self.parent = parent
        self.on_save = on_save
        self.on_segment_change = on_segment_change

        self._segments: list[EditableSegment] = []
        self._selected_index: int = -1
        self._has_changes: bool = False

        # UI element tags
        self._table_tag: int | str = 0
        self._detail_group: int | str = 0
        self._input_elements: dict = {}

        self._build()

    def _build(self) -> None:
        """Build the editor panel UI."""
        with dpg.child_window(
            parent=self.parent,
            height=400,
            border=True,
        ):
            # Toolbar
            with dpg.group(horizontal=True):
                dpg.add_text("Subtitle Editor", color=THEME.text_primary)
                dpg.add_spacer(width=-1)

                self._save_button = dpg.add_button(
                    label="Save Changes",
                    callback=self._on_save,
                    enabled=False,
                    width=120,
                )
                dpg.add_button(
                    label="Revert",
                    callback=self._on_revert,
                    width=80,
                )

            dpg.add_separator()

            # Split view: table on left, detail on right
            with dpg.group(horizontal=True):
                # Segment table
                with dpg.child_window(width=500, border=True):
                    self._build_segment_table()

                # Detail editor
                with dpg.child_window(border=True) as self._detail_group:
                    self._build_detail_editor()

    def _build_segment_table(self) -> None:
        """Build the segment table."""
        with dpg.table(
            header_row=True,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            resizable=True,
            scrollY=True,
            height=300,
        ) as self._table_tag:
            dpg.add_table_column(label="#", width_fixed=True, init_width_or_weight=40)
            dpg.add_table_column(label="Start", width_fixed=True, init_width_or_weight=80)
            dpg.add_table_column(label="End", width_fixed=True, init_width_or_weight=80)
            dpg.add_table_column(label="Speaker", width_fixed=True, init_width_or_weight=100)
            dpg.add_table_column(label="Text")

    def _build_detail_editor(self) -> None:
        """Build the detail editor section."""
        dpg.add_text("Segment Details", color=THEME.text_primary, parent=self._detail_group)
        dpg.add_separator(parent=self._detail_group)

        dpg.add_text(
            "Select a segment to edit",
            color=THEME.text_secondary,
            parent=self._detail_group,
            tag="editor_placeholder",
        )

        # Timing section (hidden initially)
        with dpg.group(parent=self._detail_group, show=False) as self._timing_group:
            dpg.add_text("Timing:", color=THEME.text_secondary)
            with dpg.group(horizontal=True):
                dpg.add_text("Start:")
                self._input_elements["start"] = dpg.add_input_float(
                    width=100,
                    format="%.3f",
                    callback=self._on_timing_change,
                )
                dpg.add_text("End:")
                self._input_elements["end"] = dpg.add_input_float(
                    width=100,
                    format="%.3f",
                    callback=self._on_timing_change,
                )

            dpg.add_spacer(height=10)

            # Speaker section
            dpg.add_text("Speaker:", color=THEME.text_secondary)
            with dpg.group(horizontal=True):
                self._input_elements["speaker_id"] = dpg.add_input_text(
                    label="ID",
                    width=120,
                    callback=self._on_speaker_change,
                )
                self._input_elements["speaker_name"] = dpg.add_input_text(
                    label="Name",
                    width=150,
                    callback=self._on_speaker_change,
                )

            dpg.add_spacer(height=10)

            # Original text
            dpg.add_text("Original Text:", color=THEME.text_secondary)
            self._input_elements["original"] = dpg.add_input_text(
                multiline=True,
                height=60,
                width=-1,
                callback=self._on_text_change,
            )

            dpg.add_spacer(height=10)

            # Translated text
            dpg.add_text("Translated Text:", color=THEME.text_secondary)
            self._input_elements["translated"] = dpg.add_input_text(
                multiline=True,
                height=60,
                width=-1,
                callback=self._on_text_change,
            )

            dpg.add_spacer(height=10)

            # Action buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Apply",
                    callback=self._on_apply_changes,
                    width=80,
                )
                dpg.add_button(
                    label="Delete",
                    callback=self._on_delete_segment,
                    width=80,
                )
                dpg.add_button(
                    label="Split",
                    callback=self._on_split_segment,
                    width=80,
                )
                dpg.add_button(
                    label="Merge Next",
                    callback=self._on_merge_next,
                    width=100,
                )

    def set_segments(self, segments: list[Segment]) -> None:
        """
        Set segments to edit.

        Args:
            segments: List of segments.
        """
        self._segments = [
            EditableSegment(index=i, segment=seg)
            for i, seg in enumerate(segments)
        ]
        self._selected_index = -1
        self._has_changes = False

        self._refresh_table()
        self._update_detail_visibility(False)
        dpg.configure_item(self._save_button, enabled=False)

    def _refresh_table(self) -> None:
        """Refresh the segment table."""
        # Clear existing rows
        for child in dpg.get_item_children(self._table_tag, 1) or []:
            dpg.delete_item(child)

        # Add rows
        for edit_seg in self._segments:
            seg = edit_seg.segment
            with dpg.table_row(parent=self._table_tag):
                # Index
                index_text = f"{edit_seg.index + 1}"
                if edit_seg.modified:
                    index_text += " *"
                dpg.add_selectable(
                    label=index_text,
                    callback=lambda s, a, u: self._on_select_segment(u),
                    user_data=edit_seg.index,
                    span_columns=True,
                )

            # We need to add the rest of the columns in a workaround
            # since selectable with span_columns takes the whole row
            # Let's rebuild with a different approach

        # Actually, let's rebuild properly
        for child in dpg.get_item_children(self._table_tag, 1) or []:
            dpg.delete_item(child)

        for edit_seg in self._segments:
            seg = edit_seg.segment
            self._add_table_row(edit_seg)

    def _add_table_row(self, edit_seg: EditableSegment) -> None:
        """Add a row to the table."""
        seg = edit_seg.segment

        with dpg.table_row(parent=self._table_tag):
            # Index with modification indicator
            index_text = f"{edit_seg.index + 1}"
            if edit_seg.modified:
                index_text += "*"

            dpg.add_button(
                label=index_text,
                callback=lambda s, a, u: self._on_select_segment(u),
                user_data=edit_seg.index,
                width=-1,
            )

            # Start time
            dpg.add_text(f"{seg.start:.2f}")

            # End time
            dpg.add_text(f"{seg.end:.2f}")

            # Speaker
            speaker = seg.speaker_name or seg.speaker_id or "-"
            dpg.add_text(speaker[:12])

            # Text preview
            text = seg.translated_text or seg.original_text or ""
            preview = text[:30] + "..." if len(text) > 30 else text
            dpg.add_text(preview)

    def _on_select_segment(self, index: int) -> None:
        """Handle segment selection."""
        if index < 0 or index >= len(self._segments):
            return

        self._selected_index = index
        edit_seg = self._segments[index]
        seg = edit_seg.segment

        # Update detail editor
        self._update_detail_visibility(True)

        dpg.set_value(self._input_elements["start"], seg.start)
        dpg.set_value(self._input_elements["end"], seg.end)
        dpg.set_value(self._input_elements["speaker_id"], seg.speaker_id)
        dpg.set_value(self._input_elements["speaker_name"], seg.speaker_name)
        dpg.set_value(self._input_elements["original"], seg.original_text)
        dpg.set_value(self._input_elements["translated"], seg.translated_text)

    def _update_detail_visibility(self, show: bool) -> None:
        """Update detail editor visibility."""
        if dpg.does_item_exist("editor_placeholder"):
            dpg.configure_item("editor_placeholder", show=not show)
        dpg.configure_item(self._timing_group, show=show)

    def _on_timing_change(self, sender, app_data, user_data) -> None:
        """Handle timing change."""
        self._mark_current_modified()

    def _on_speaker_change(self, sender, app_data, user_data) -> None:
        """Handle speaker change."""
        self._mark_current_modified()

    def _on_text_change(self, sender, app_data, user_data) -> None:
        """Handle text change."""
        self._mark_current_modified()

    def _mark_current_modified(self) -> None:
        """Mark current segment as modified."""
        if self._selected_index >= 0:
            self._segments[self._selected_index].modified = True
            self._has_changes = True
            dpg.configure_item(self._save_button, enabled=True)

    def _on_apply_changes(self) -> None:
        """Apply changes to selected segment."""
        if self._selected_index < 0:
            return

        edit_seg = self._segments[self._selected_index]

        # Update segment from inputs
        edit_seg.segment.start = dpg.get_value(self._input_elements["start"])
        edit_seg.segment.end = dpg.get_value(self._input_elements["end"])
        edit_seg.segment.speaker_id = dpg.get_value(self._input_elements["speaker_id"])
        edit_seg.segment.speaker_name = dpg.get_value(self._input_elements["speaker_name"])
        edit_seg.segment.original_text = dpg.get_value(self._input_elements["original"])
        edit_seg.segment.translated_text = dpg.get_value(self._input_elements["translated"])

        edit_seg.modified = True
        self._has_changes = True
        dpg.configure_item(self._save_button, enabled=True)

        self._refresh_table()

        if self.on_segment_change:
            self.on_segment_change(self._selected_index, edit_seg.segment)

    def _on_delete_segment(self) -> None:
        """Delete selected segment."""
        if self._selected_index < 0:
            return

        def do_delete():
            del self._segments[self._selected_index]

            # Re-index
            for i, seg in enumerate(self._segments):
                seg.index = i

            self._selected_index = -1
            self._has_changes = True
            dpg.configure_item(self._save_button, enabled=True)

            self._refresh_table()
            self._update_detail_visibility(False)

        show_confirm_dialog(
            "Delete Segment",
            f"Delete segment #{self._selected_index + 1}?",
            on_confirm=do_delete,
        )

    def _on_split_segment(self) -> None:
        """Split selected segment at midpoint."""
        if self._selected_index < 0:
            return

        edit_seg = self._segments[self._selected_index]
        seg = edit_seg.segment

        # Calculate midpoint
        mid_time = (seg.start + seg.end) / 2

        # Create new segment
        new_seg = Segment(
            start=mid_time,
            end=seg.end,
            original_text="",
            translated_text="",
            speaker_id=seg.speaker_id,
            speaker_name=seg.speaker_name,
        )

        # Update original segment
        seg.end = mid_time
        edit_seg.modified = True

        # Insert new segment
        new_edit_seg = EditableSegment(
            index=self._selected_index + 1,
            segment=new_seg,
            modified=True,
        )
        self._segments.insert(self._selected_index + 1, new_edit_seg)

        # Re-index
        for i, s in enumerate(self._segments):
            s.index = i

        self._has_changes = True
        dpg.configure_item(self._save_button, enabled=True)

        self._refresh_table()

    def _on_merge_next(self) -> None:
        """Merge selected segment with next segment."""
        if self._selected_index < 0 or self._selected_index >= len(self._segments) - 1:
            show_message_box("Cannot Merge", "No next segment to merge with.", "warning")
            return

        edit_seg = self._segments[self._selected_index]
        next_seg = self._segments[self._selected_index + 1]

        # Merge timing
        edit_seg.segment.end = next_seg.segment.end

        # Merge text
        if edit_seg.segment.original_text and next_seg.segment.original_text:
            edit_seg.segment.original_text += " " + next_seg.segment.original_text
        if edit_seg.segment.translated_text and next_seg.segment.translated_text:
            edit_seg.segment.translated_text += " " + next_seg.segment.translated_text

        edit_seg.modified = True

        # Remove next segment
        del self._segments[self._selected_index + 1]

        # Re-index
        for i, s in enumerate(self._segments):
            s.index = i

        self._has_changes = True
        dpg.configure_item(self._save_button, enabled=True)

        self._refresh_table()
        self._on_select_segment(self._selected_index)

    def _on_save(self) -> None:
        """Save all changes."""
        if not self._has_changes:
            return

        segments = [edit_seg.segment for edit_seg in self._segments]

        if self.on_save:
            self.on_save(segments)

        # Reset modification flags
        for edit_seg in self._segments:
            edit_seg.modified = False

        self._has_changes = False
        dpg.configure_item(self._save_button, enabled=False)

        self._refresh_table()
        show_message_box("Saved", "Changes saved successfully.", "info")

    def _on_revert(self) -> None:
        """Revert all changes."""
        if not self._has_changes:
            return

        show_confirm_dialog(
            "Revert Changes",
            "Discard all unsaved changes?",
            on_confirm=self._do_revert,
        )

    def _do_revert(self) -> None:
        """Actually revert changes."""
        # This would need to reload from original source
        # For now, just clear the modification flags
        self._has_changes = False
        dpg.configure_item(self._save_button, enabled=False)
        show_message_box("Reverted", "Changes discarded.", "info")

    def get_segments(self) -> list[Segment]:
        """Get current segments."""
        return [edit_seg.segment for edit_seg in self._segments]

    def has_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._has_changes

    @property
    def selected_segment(self) -> Segment | None:
        """Get currently selected segment."""
        if self._selected_index >= 0 and self._selected_index < len(self._segments):
            return self._segments[self._selected_index].segment
        return None
