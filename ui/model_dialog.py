"""Model management dialog for Voice Subtitle Generator."""

import threading
from pathlib import Path

import dearpygui.dearpygui as dpg

from src.config import TRANSLATION_MODEL_PRESETS
from src.model_manager import ModelManager, WHISPER_MODELS

from .components import THEME, show_message_box


class ModelDialog:
    """
    Model management dialog.

    Shows model status and allows downloading required models.
    """

    def __init__(self, config):
        """
        Initialize model dialog.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.model_manager = ModelManager()

        self._dialog_tag: int | str = 0
        self._is_downloading = False
        self._preset_keys: list[str] = []

    def show(self) -> None:
        """Show the model dialog."""
        if self._dialog_tag and dpg.does_item_exist(self._dialog_tag):
            dpg.delete_item(self._dialog_tag)

        self._build_dialog()

    def _build_dialog(self) -> None:
        """Build the dialog UI."""
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dialog_width = 600
        dialog_height = 550

        with dpg.window(
            label="모델 관리",
            modal=True,
            width=dialog_width,
            height=dialog_height,
            pos=[
                (viewport_width - dialog_width) // 2,
                (viewport_height - dialog_height) // 2,
            ],
            on_close=self.close,
            no_resize=True,
        ) as self._dialog_tag:
            # Tab bar for different model types
            with dpg.tab_bar():
                self._build_whisper_tab()
                self._build_translation_tab()

            # Progress section (shared)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            self._progress_bar = dpg.add_progress_bar(
                default_value=0.0,
                width=-1,
            )
            self._progress_label = dpg.add_text(
                "대기 중",
                color=THEME.text_secondary,
            )

            # Close button
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="닫기",
                    width=100,
                    callback=self.close,
                )

    def _build_whisper_tab(self) -> None:
        """Build WhisperX model tab."""
        with dpg.tab(label="음성 인식 (WhisperX)"):
            dpg.add_spacer(height=10)

            # Current model status
            model_size = self.config.speech.stt.model_size
            status = self.model_manager.check_whisperx_model(model_size)

            dpg.add_text("현재 설정된 모델:", color=THEME.text_primary)
            with dpg.group(horizontal=True):
                dpg.add_text(f"  {model_size}", color=THEME.info)
                if status["cached"]:
                    dpg.add_text("[다운로드됨]", color=THEME.success)
                else:
                    dpg.add_text("[미다운로드]", color=THEME.warning)

            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Model selection
            dpg.add_text("모델 다운로드:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            # Model list with status
            whisper_models = self.model_manager.get_available_whisper_models()
            model_items = []
            for size, info in whisper_models:
                ws = self.model_manager.check_whisperx_model(size)
                status_text = "✓" if ws["cached"] else ""
                model_items.append(f"{status_text} {info.name} - {info.description}")

            with dpg.group(horizontal=True):
                dpg.add_text("모델:", color=THEME.text_secondary)
                self._whisper_combo = dpg.add_combo(
                    items=model_items,
                    default_value=model_items[6] if len(model_items) > 6 else model_items[0],  # large-v3-turbo
                    width=400,
                )

            dpg.add_spacer(height=10)

            self._whisper_download_btn = dpg.add_button(
                label="WhisperX 모델 다운로드",
                width=200,
                callback=self._on_download_whisper,
            )

            dpg.add_spacer(height=10)
            dpg.add_text(
                "참고: WhisperX 모델은 HuggingFace에서 자동으로 다운로드됩니다.\n"
                "처음 사용 시 자동 다운로드되지만, 여기서 미리 다운로드할 수 있습니다.",
                color=THEME.text_secondary,
                wrap=550,
            )

    def _build_translation_tab(self) -> None:
        """Build translation model tab."""
        with dpg.tab(label="번역 (LLM)"):
            dpg.add_spacer(height=10)

            # Current preset status
            current_preset_key = self.config.translation.model_preset
            current_preset = TRANSLATION_MODEL_PRESETS.get(current_preset_key)

            dpg.add_text("현재 선택된 모델:", color=THEME.text_primary)

            if current_preset and current_preset_key != "custom":
                exists, size_on_disk = self.model_manager.check_preset_model(current_preset_key)
                with dpg.group(horizontal=True):
                    dpg.add_text(f"  {current_preset.name}", color=THEME.info)
                    if exists:
                        size_gb = size_on_disk / (1024 ** 3)
                        dpg.add_text(f"[설치됨 - {size_gb:.1f}GB]", color=THEME.success)
                    else:
                        dpg.add_text("[미설치]", color=THEME.warning)
                dpg.add_text(f"  {current_preset.description}", color=THEME.text_secondary)
            else:
                model_path = self.config.translation.model_path
                status = self.model_manager.check_translation_model(model_path)
                with dpg.group(horizontal=True):
                    dpg.add_text(f"  {Path(model_path).name}", color=THEME.info)
                    if status.exists:
                        size_gb = status.size_on_disk / (1024 ** 3)
                        dpg.add_text(f"[설치됨 - {size_gb:.1f}GB]", color=THEME.success)
                    else:
                        dpg.add_text("[미설치]", color=THEME.error)

            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Model download section
            dpg.add_text("모델 다운로드:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            # Build preset list with status
            self._preset_keys = []
            model_items = []
            default_idx = 0

            for i, (key, preset) in enumerate(TRANSLATION_MODEL_PRESETS.items()):
                if key == "custom":
                    continue
                self._preset_keys.append(key)
                exists, _ = self.model_manager.check_preset_model(key)
                status_text = "✓ " if exists else "  "
                item = f"{status_text}{preset.name} ({preset.size_mb}MB) - {preset.description}"
                model_items.append(item)

                # Default to current preset
                if key == current_preset_key:
                    default_idx = len(model_items) - 1

            with dpg.group(horizontal=True):
                dpg.add_text("모델:", color=THEME.text_secondary)
                self._translation_combo = dpg.add_combo(
                    items=model_items,
                    default_value=model_items[default_idx] if model_items else "",
                    width=450,
                )

            dpg.add_spacer(height=10)

            self._translation_download_btn = dpg.add_button(
                label="선택한 모델 다운로드",
                width=200,
                callback=self._on_download_translation,
            )

            dpg.add_spacer(height=10)
            dpg.add_text(
                "참고: 번역 모델(GGUF)은 수 GB 크기입니다.\n"
                "다운로드 후 설정에서 해당 모델을 선택해야 적용됩니다.",
                color=THEME.text_secondary,
                wrap=550,
            )

    def _on_download_whisper(self) -> None:
        """Handle WhisperX download button."""
        if self._is_downloading:
            show_message_box("다운로드 중", "이미 다운로드가 진행 중입니다.", "warning")
            return

        # Parse selected model
        selected = dpg.get_value(self._whisper_combo)
        if not selected:
            return

        # Extract model size from selection - check longer names first to avoid substring issues
        whisper_models = self.model_manager.get_available_whisper_models()
        model_size = None
        # Sort by name length descending to match longer names first
        # (e.g., "Large-v3 Turbo" before "Large-v3")
        sorted_models = sorted(whisper_models, key=lambda x: len(x[1].name), reverse=True)
        for size, info in sorted_models:
            if info.name in selected:
                model_size = size
                break

        if not model_size:
            show_message_box("오류", "모델을 선택해주세요.", "error")
            return

        self._start_download(
            lambda cb: self.model_manager.download_whisperx_model(model_size, cb),
            self._whisper_download_btn,
        )

    def _on_download_translation(self) -> None:
        """Handle translation model download button."""
        if self._is_downloading:
            show_message_box("다운로드 중", "이미 다운로드가 진행 중입니다.", "warning")
            return

        # Parse selected model from combo index
        selected = dpg.get_value(self._translation_combo)
        if not selected:
            show_message_box("오류", "모델을 선택해주세요.", "error")
            return

        # Find preset key by matching the preset name in the selected string
        preset_key = None
        for key in self._preset_keys:
            preset = TRANSLATION_MODEL_PRESETS[key]
            if preset.name in selected:
                preset_key = key
                break

        if not preset_key:
            show_message_box("오류", "모델을 선택해주세요.", "error")
            return

        # Check if already downloaded
        exists, _ = self.model_manager.check_preset_model(preset_key)
        if exists:
            show_message_box("알림", "이미 다운로드된 모델입니다.", "info")
            return

        self._start_download(
            lambda cb: self.model_manager.download_preset_model(preset_key, cb),
            self._translation_download_btn,
        )

    def _start_download(self, download_func, button) -> None:
        """Start download in background thread."""
        self._is_downloading = True
        dpg.configure_item(button, enabled=False)
        dpg.set_value(self._progress_bar, 0.0)
        dpg.set_value(self._progress_label, "다운로드 준비 중...")
        dpg.configure_item(self._progress_label, color=THEME.text_secondary)

        def download_thread():
            def progress_callback(progress: float, message: str):
                # Check if dialog still exists before updating UI
                if self._dialog_tag and dpg.does_item_exist(self._dialog_tag):
                    try:
                        dpg.set_value(self._progress_bar, progress)
                        dpg.set_value(self._progress_label, message)
                    except Exception:
                        pass  # Dialog was closed

            success = download_func(progress_callback)

            self._is_downloading = False

            # Check if dialog still exists before updating UI
            if not self._dialog_tag or not dpg.does_item_exist(self._dialog_tag):
                return  # Dialog was closed, skip UI updates

            try:
                dpg.configure_item(button, enabled=True)

                if success:
                    dpg.set_value(self._progress_label, "다운로드 완료!")
                    dpg.configure_item(self._progress_label, color=THEME.success)
                else:
                    current_msg = dpg.get_value(self._progress_label)
                    if "실패" not in current_msg:
                        dpg.set_value(self._progress_label, "다운로드 실패")
                    dpg.configure_item(self._progress_label, color=THEME.error)
            except Exception:
                pass  # Dialog was closed

        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def close(self) -> None:
        """Close the dialog."""
        if self._dialog_tag and dpg.does_item_exist(self._dialog_tag):
            dpg.delete_item(self._dialog_tag)
            self._dialog_tag = 0

        # Note: Download continues in background even if dialog is closed

    @property
    def is_open(self) -> bool:
        """Check if dialog is open."""
        return bool(self._dialog_tag and dpg.does_item_exist(self._dialog_tag))
