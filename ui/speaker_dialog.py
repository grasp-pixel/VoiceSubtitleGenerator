"""Speaker mapping dialog for Voice Subtitle Generator."""

from typing import Callable

import dearpygui.dearpygui as dpg

from src.models import SpeakerMapping
from src.speaker_manager import SpeakerManager

from .components import THEME


class SpeakerMappingDialog:
    """
    Speaker mapping dialog component.

    Allows users to map speaker IDs to character names with colors.
    """

    # Default color palette
    DEFAULT_COLORS = [
        ("White", "FFFFFF"),
        ("Sky Blue", "00BFFF"),
        ("Pink", "FF69B4"),
        ("Light Green", "90EE90"),
        ("Gold", "FFD700"),
        ("Plum", "DDA0DD"),
        ("Coral", "FF7F50"),
        ("Lavender", "E6E6FA"),
        ("Khaki", "F0E68C"),
        ("Light Salmon", "FFA07A"),
    ]

    def __init__(
        self,
        on_apply: Callable[[dict[str, SpeakerMapping]], None] | None = None,
        speaker_manager: SpeakerManager | None = None,
    ):
        """
        Initialize speaker mapping dialog.

        Args:
            on_apply: Callback when mapping is applied.
            speaker_manager: Speaker manager for preset handling.
        """
        self.on_apply = on_apply
        self.speaker_manager = speaker_manager or SpeakerManager()

        self._dialog_tag: int | str = 0
        self._speakers: list[str] = []
        self._mapping_rows: dict[str, dict] = {}
        self._current_file: str = ""

    def show(
        self,
        speakers: list[str],
        current_mapping: dict[str, SpeakerMapping] | None = None,
        filename: str = "",
    ) -> None:
        """
        Show the dialog.

        Args:
            speakers: List of speaker IDs.
            current_mapping: Current speaker mapping.
            filename: Current file name for display.
        """
        self._speakers = speakers
        self._current_file = filename

        # Close existing dialog if any
        if self._dialog_tag and dpg.does_item_exist(self._dialog_tag):
            dpg.delete_item(self._dialog_tag)

        self._build_dialog(current_mapping)

    def _build_dialog(
        self, current_mapping: dict[str, SpeakerMapping] | None
    ) -> None:
        """Build the dialog UI."""
        # Calculate position
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dialog_width = 600
        dialog_height = 450

        with dpg.window(
            label="화자 매핑",
            modal=True,
            width=dialog_width,
            height=dialog_height,
            pos=[
                (viewport_width - dialog_width) // 2,
                (viewport_height - dialog_height) // 2,
            ],
            on_close=self._on_cancel,
            no_resize=True,
        ) as self._dialog_tag:
            # File info
            if self._current_file:
                dpg.add_text(f"파일: {self._current_file}", color=THEME.text_secondary)

            dpg.add_text(
                f"감지된 화자: {len(self._speakers)}명",
                color=THEME.text_primary,
            )

            dpg.add_separator()

            # Mapping table
            with dpg.child_window(height=250, border=True):
                with dpg.table(
                    header_row=True,
                    borders_innerH=True,
                    borders_outerH=True,
                    borders_innerV=True,
                    borders_outerV=True,
                    resizable=True,
                ):
                    dpg.add_table_column(label="화자 ID", width_fixed=True, init_width_or_weight=120)
                    dpg.add_table_column(label="이름", init_width_or_weight=200)
                    dpg.add_table_column(label="색상", width_fixed=True, init_width_or_weight=150)

                    # Add rows for each speaker
                    for i, speaker_id in enumerate(self._speakers):
                        self._add_speaker_row(speaker_id, current_mapping, i)

            dpg.add_separator()

            # Preset section
            with dpg.group(horizontal=True):
                dpg.add_text("프리셋:", color=THEME.text_secondary)

                # Preset dropdown
                presets = self.speaker_manager.list_presets()
                preset_items = ["(없음)"] + presets

                self._preset_combo = dpg.add_combo(
                    items=preset_items,
                    default_value="(없음)",
                    width=150,
                    callback=self._on_preset_select,
                )

                dpg.add_button(
                    label="불러오기",
                    width=70,
                    callback=self._on_load_preset,
                )

                dpg.add_button(
                    label="저장",
                    width=60,
                    callback=self._on_save_preset,
                )

            # Options
            dpg.add_spacer(height=10)
            self._apply_all_checkbox = dpg.add_checkbox(
                label="같은 화자 수의 모든 파일에 적용",
                default_value=False,
            )

            # Buttons
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="취소",
                    width=100,
                    callback=self._on_cancel,
                )
                dpg.add_button(
                    label="적용",
                    width=100,
                    callback=self._on_apply,
                )

    def _add_speaker_row(
        self,
        speaker_id: str,
        current_mapping: dict[str, SpeakerMapping] | None,
        index: int,
    ) -> None:
        """Add a row for a speaker."""
        # Get current values
        if current_mapping and speaker_id in current_mapping:
            current_name = current_mapping[speaker_id].name
            current_color = current_mapping[speaker_id].color
        else:
            current_name = speaker_id
            current_color = self.DEFAULT_COLORS[index % len(self.DEFAULT_COLORS)][1]

        with dpg.table_row():
            # Speaker ID
            dpg.add_text(speaker_id)

            # Name input
            name_input = dpg.add_input_text(
                default_value=current_name,
                width=-1,
            )

            # Color combo
            color_items = [name for name, _ in self.DEFAULT_COLORS]
            current_color_name = self._get_color_name(current_color)

            color_combo = dpg.add_combo(
                items=color_items,
                default_value=current_color_name,
                width=-1,
            )

            # Store references
            self._mapping_rows[speaker_id] = {
                "name_input": name_input,
                "color_combo": color_combo,
            }

    def _get_color_name(self, color_hex: str) -> str:
        """Get color name from hex value."""
        color_hex = color_hex.upper()
        for name, hex_val in self.DEFAULT_COLORS:
            if hex_val.upper() == color_hex:
                return name
        return self.DEFAULT_COLORS[0][0]

    def _get_color_hex(self, color_name: str) -> str:
        """Get hex value from color name."""
        for name, hex_val in self.DEFAULT_COLORS:
            if name == color_name:
                return hex_val
        return "FFFFFF"

    def _get_mapping(self) -> dict[str, SpeakerMapping]:
        """Get current mapping from UI inputs."""
        mapping = {}

        for speaker_id, row in self._mapping_rows.items():
            name = dpg.get_value(row["name_input"])
            color_name = dpg.get_value(row["color_combo"])
            color_hex = self._get_color_hex(color_name)

            mapping[speaker_id] = SpeakerMapping(
                speaker_id=speaker_id,
                name=name,
                color=color_hex,
            )

        return mapping

    def _on_apply(self) -> None:
        """Handle apply button click."""
        mapping = self._get_mapping()

        if self.on_apply:
            self.on_apply(mapping)

        self.close()

    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        self.close()

    def _on_preset_select(self, sender, app_data, user_data) -> None:
        """Handle preset selection."""
        pass  # Just updates the combo value

    def _on_load_preset(self) -> None:
        """Handle load preset button click."""
        preset_name = dpg.get_value(self._preset_combo)

        if preset_name == "(없음)":
            return

        try:
            mapping = self.speaker_manager.load_preset(preset_name)

            # Update UI with loaded mapping
            for speaker_id, row in self._mapping_rows.items():
                if speaker_id in mapping:
                    dpg.set_value(row["name_input"], mapping[speaker_id].name)
                    color_name = self._get_color_name(mapping[speaker_id].color)
                    dpg.set_value(row["color_combo"], color_name)

        except FileNotFoundError:
            pass

    def _on_save_preset(self) -> None:
        """Handle save preset button click."""
        # Show save dialog
        self._show_save_preset_dialog()

    def _show_save_preset_dialog(self) -> None:
        """Show dialog to save preset."""

        def do_save():
            name = dpg.get_value(name_input)
            if name:
                mapping = self._get_mapping()
                self.speaker_manager.save_preset(name, mapping)

                # Update preset list
                presets = self.speaker_manager.list_presets()
                dpg.configure_item(
                    self._preset_combo,
                    items=["(없음)"] + presets,
                )
                dpg.set_value(self._preset_combo, name)

            dpg.delete_item(save_dialog)

        def do_cancel():
            dpg.delete_item(save_dialog)

        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()

        with dpg.window(
            label="프리셋 저장",
            modal=True,
            width=300,
            height=120,
            pos=[
                (viewport_width - 300) // 2,
                (viewport_height - 120) // 2,
            ],
            no_resize=True,
        ) as save_dialog:
            dpg.add_text("프리셋 이름:")
            name_input = dpg.add_input_text(width=-1)

            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=-1)
                dpg.add_button(label="취소", width=80, callback=do_cancel)
                dpg.add_button(label="저장", width=80, callback=do_save)

    def close(self) -> None:
        """Close the dialog."""
        if self._dialog_tag and dpg.does_item_exist(self._dialog_tag):
            dpg.delete_item(self._dialog_tag)
            self._dialog_tag = 0

        self._mapping_rows.clear()

    @property
    def is_open(self) -> bool:
        """Check if dialog is open."""
        return bool(
            self._dialog_tag and dpg.does_item_exist(self._dialog_tag)
        )
