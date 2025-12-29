"""Settings dialog for Voice Subtitle Generator."""

from typing import Callable

import dearpygui.dearpygui as dpg

from src.config import (
    AppConfig,
    ConfigManager,
    SUPPORTED_WHISPER_MODELS,
    TRANSLATION_MODEL_PRESETS,
)
from src.font_manager import get_font_manager

from .components import THEME, FileDialogHelper


class SettingsDialog:
    """
    Settings dialog component.

    Allows users to configure application settings.
    """

    WHISPER_MODELS = SUPPORTED_WHISPER_MODELS
    DEVICES = ["cuda", "cpu"]
    COMPUTE_TYPES = ["float16", "int8", "float32"]
    SUBTITLE_FORMATS = ["srt", "ass", "vtt"]

    # Video resolution presets: (label, width, height)
    RESOLUTION_PRESETS = [
        ("4K (3840x2160)", 3840, 2160),
        ("2K (2560x1440)", 2560, 1440),
        ("1080p (1920x1080)", 1920, 1080),
        ("720p (1280x720)", 1280, 720),
        ("480p (854x480)", 854, 480),
    ]

    def __init__(
        self,
        config_manager: ConfigManager,
        on_save: Callable[[AppConfig], None] | None = None,
    ):
        """
        Initialize settings dialog.

        Args:
            config_manager: Configuration manager.
            on_save: Callback when settings are saved.
        """
        self.config_manager = config_manager
        self.on_save = on_save

        self._dialog_tag: int | str = 0
        self._config: AppConfig | None = None

        # Input references
        self._inputs: dict = {}
        self._preset_keys: list[str] = []

    def show(self) -> None:
        """Show the settings dialog."""
        self._config = self.config_manager.load()

        # Close existing dialog if any
        if self._dialog_tag and dpg.does_item_exist(self._dialog_tag):
            dpg.delete_item(self._dialog_tag)

        self._build_dialog()

    def _build_dialog(self) -> None:
        """Build the dialog UI."""
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        dialog_width = 600
        dialog_height = 720

        with dpg.window(
            label="환경설정",
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
            # Tab bar
            with dpg.tab_bar():
                self._build_general_tab()
                self._build_stt_tab()
                self._build_translation_tab()
                self._build_subtitle_tab()

            # Buttons
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=-1)
                dpg.add_button(
                    label="취소",
                    width=100,
                    callback=self._on_cancel,
                )
                dpg.add_button(
                    label="저장",
                    width=100,
                    callback=self._on_save,
                )

    def _build_general_tab(self) -> None:
        """Build general settings tab."""
        with dpg.tab(label="일반"):
            dpg.add_spacer(height=10)

            # Models path
            dpg.add_text("모델 경로:", color=THEME.text_secondary)
            with dpg.group(horizontal=True):
                self._inputs["models_path"] = dpg.add_input_text(
                    default_value=self._config.models.path,
                    width=-80,
                )
                dpg.add_button(
                    label="찾기",
                    width=70,
                    callback=lambda: self._browse_folder("models_path"),
                )

            dpg.add_spacer(height=10)
            self._inputs["auto_download"] = dpg.add_checkbox(
                label="누락된 모델 자동 다운로드",
                default_value=self._config.models.auto_download,
            )

            dpg.add_separator()

            # Audio processing
            dpg.add_text("오디오 처리:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            self._inputs["normalize_audio"] = dpg.add_checkbox(
                label="오디오 정규화",
                default_value=self._config.processing.normalize_audio,
            )

            with dpg.group(horizontal=True):
                dpg.add_text("샘플 레이트:", color=THEME.text_secondary)
                self._inputs["sample_rate"] = dpg.add_input_int(
                    default_value=self._config.processing.target_sample_rate,
                    width=100,
                    min_value=8000,
                    max_value=48000,
                )

            dpg.add_separator()

            # Segment splitting
            dpg.add_text("세그먼트 분할:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            self._inputs["segment_enabled"] = dpg.add_checkbox(
                label="긴 세그먼트 자동 분할",
                default_value=self._config.segment.enabled,
            )

            with dpg.group(horizontal=True):
                dpg.add_text("최대 글자 수:", color=THEME.text_secondary)
                self._inputs["segment_max_chars"] = dpg.add_input_int(
                    default_value=self._config.segment.max_chars,
                    width=100,
                    min_value=20,
                    max_value=200,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("최대 지속 시간(초):", color=THEME.text_secondary)
                self._inputs["segment_max_duration"] = dpg.add_input_float(
                    default_value=self._config.segment.max_duration,
                    width=100,
                    min_value=1.0,
                    max_value=10.0,
                    format="%.1f",
                )

            dpg.add_text(
                "문장부호(。！？) 또는 지정된 길이로 분할합니다.",
                color=THEME.text_secondary,
            )

            dpg.add_separator()

            # Logging settings
            dpg.add_text("로그 설정:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            self._inputs["hide_warnings"] = dpg.add_checkbox(
                label="경고(WARNING) 로그 숨기기",
                default_value=self._config.ui.hide_warnings,
            )
            dpg.add_text(
                "라이브러리 경고 메시지를 콘솔에서 숨깁니다.",
                color=THEME.text_secondary,
            )

    def _build_stt_tab(self) -> None:
        """Build speech processing settings tab."""
        with dpg.tab(label="음성인식"):
            dpg.add_spacer(height=10)

            # STT Model settings
            dpg.add_text("STT 모델 설정:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                dpg.add_text("모델 크기:", color=THEME.text_secondary)
                self._inputs["whisper_model"] = dpg.add_combo(
                    items=self.WHISPER_MODELS,
                    default_value=self._config.speech.stt.model_size,
                    width=150,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("디바이스:", color=THEME.text_secondary)
                self._inputs["stt_device"] = dpg.add_combo(
                    items=self.DEVICES,
                    default_value=self._config.speech.stt.device,
                    width=150,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("연산 타입:", color=THEME.text_secondary)
                self._inputs["compute_type"] = dpg.add_combo(
                    items=self.COMPUTE_TYPES,
                    default_value=self._config.speech.stt.compute_type,
                    width=150,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("빔 크기:", color=THEME.text_secondary)
                self._inputs["beam_size"] = dpg.add_input_int(
                    default_value=self._config.speech.stt.beam_size,
                    width=100,
                    min_value=1,
                    max_value=10,
                )

            self._inputs["vad_filter"] = dpg.add_checkbox(
                label="VAD 필터 사용 (묵음 구간 스킵)",
                default_value=self._config.speech.stt.vad_filter,
            )

    def _build_translation_tab(self) -> None:
        """Build translation settings tab."""
        with dpg.tab(label="번역"):
            dpg.add_spacer(height=10)

            # Model preset selection
            dpg.add_text("번역 모델:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            # Build preset items list
            preset_items = []
            preset_keys = []
            for key, preset in TRANSLATION_MODEL_PRESETS.items():
                if key == "custom":
                    preset_items.append(f"{preset.name}")
                else:
                    preset_items.append(f"{preset.name} (VRAM {preset.vram_gb:.0f}GB)")
                preset_keys.append(key)

            # Store preset keys for lookup
            self._preset_keys = preset_keys

            current_preset = self._config.translation.model_preset
            current_idx = preset_keys.index(current_preset) if current_preset in preset_keys else 0

            with dpg.group(horizontal=True):
                dpg.add_text("모델 선택:", color=THEME.text_secondary)
                self._inputs["model_preset"] = dpg.add_combo(
                    items=preset_items,
                    default_value=preset_items[current_idx],
                    width=500,
                    callback=self._on_model_preset_changed,
                )

            # Model description
            preset_obj = TRANSLATION_MODEL_PRESETS.get(current_preset)
            desc = preset_obj.description if preset_obj else ""
            self._model_desc_label = dpg.add_text(
                f"  {desc}",
                color=THEME.text_secondary,
            )

            dpg.add_spacer(height=5)

            # Custom model path (shown only when "custom" is selected)
            with dpg.group(show=(current_preset == "custom")) as self._custom_model_group:
                dpg.add_text("모델 경로:", color=THEME.text_secondary)
                with dpg.group(horizontal=True):
                    self._inputs["translation_model"] = dpg.add_input_text(
                        default_value=self._config.translation.model_path,
                        width=-80,
                    )
                    dpg.add_button(
                        label="찾기",
                        width=70,
                        callback=lambda: self._browse_file("translation_model"),
                    )

            dpg.add_separator()

            # Generation settings
            dpg.add_text("생성 설정:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                dpg.add_text("GPU 레이어:", color=THEME.text_secondary)
                self._inputs["n_gpu_layers"] = dpg.add_input_int(
                    default_value=self._config.translation.n_gpu_layers,
                    width=200,
                    min_value=-1,
                    max_value=100,
                )
                dpg.add_text("(-1 = 전체)", color=THEME.text_secondary)

            with dpg.group(horizontal=True):
                dpg.add_text("컨텍스트 크기:", color=THEME.text_secondary)
                self._inputs["n_ctx"] = dpg.add_input_int(
                    default_value=self._config.translation.n_ctx,
                    width=200,
                    min_value=512,
                    max_value=32768,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("최대 토큰:", color=THEME.text_secondary)
                self._inputs["max_tokens"] = dpg.add_input_int(
                    default_value=self._config.translation.max_tokens,
                    width=200,
                    min_value=32,
                    max_value=2048,
                )

            with dpg.group(horizontal=True):
                dpg.add_text("온도:", color=THEME.text_secondary)
                self._inputs["temperature"] = dpg.add_input_float(
                    default_value=self._config.translation.temperature,
                    width=200,
                    min_value=0.0,
                    max_value=2.0,
                    format="%.2f",
                )

            dpg.add_separator()

            # Review settings
            dpg.add_text("검수 설정:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            self._inputs["enable_review"] = dpg.add_checkbox(
                label="번역 검수 활성화 (2-pass)",
                default_value=self._config.translation.enable_review,
            )
            dpg.add_text(
                "원문과 번역을 비교하여 오역/누락을 교정합니다.",
                color=THEME.text_secondary,
            )
            dpg.add_text(
                "주의: 처리 시간이 약 2배 증가합니다.",
                color=THEME.warning,
            )

    def _build_subtitle_tab(self) -> None:
        """Build subtitle settings tab."""
        with dpg.tab(label="자막"):
            dpg.add_spacer(height=10)

            # Output settings
            dpg.add_text("출력 설정:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                dpg.add_text("기본 형식:", color=THEME.text_secondary)
                self._inputs["default_format"] = dpg.add_combo(
                    items=self.SUBTITLE_FORMATS,
                    default_value=self._config.subtitle.default_format,
                    width=100,
                )

            dpg.add_spacer(height=5)

            self._inputs["include_original"] = dpg.add_checkbox(
                label="원문 포함",
                default_value=self._config.subtitle.include_original,
            )

            dpg.add_separator()

            # ASS settings
            dpg.add_text("ASS 설정:", color=THEME.text_primary)
            dpg.add_spacer(height=5)

            # Resolution preset dropdown
            with dpg.group(horizontal=True):
                dpg.add_text("영상 해상도:", color=THEME.text_secondary)
                resolution_labels = [p[0] for p in self.RESOLUTION_PRESETS]
                current_resolution = self._get_resolution_label(
                    self._config.subtitle.ass.video_width,
                    self._config.subtitle.ass.video_height,
                )
                self._inputs["resolution_preset"] = dpg.add_combo(
                    items=resolution_labels,
                    default_value=current_resolution,
                    width=200,
                )

            # Font dropdown with status and install button
            with dpg.group(horizontal=True):
                dpg.add_text("기본 폰트:", color=THEME.text_secondary)

                font_mgr = get_font_manager()
                font_names = font_mgr.get_font_names()
                current_font = self._config.subtitle.ass.default_font
                current_display = font_mgr.get_display_name(current_font)

                self._inputs["default_font"] = dpg.add_combo(
                    items=font_names,
                    default_value=current_display,
                    width=200,
                    callback=self._on_font_changed,
                )

                # Font status label
                self._font_status_label = dpg.add_text(
                    "", color=THEME.text_secondary
                )

                # Install button (download + install)
                self._font_install_btn = dpg.add_button(
                    label="설치",
                    callback=self._on_font_install,
                    width=50,
                )

            # Update initial font status
            self._update_font_status(current_display)

            with dpg.group(horizontal=True):
                dpg.add_text("폰트 크기:", color=THEME.text_secondary)
                self._inputs["font_size"] = dpg.add_input_int(
                    default_value=self._config.subtitle.ass.default_size,
                    width=100,
                    min_value=12,
                    max_value=120,
                )

            dpg.add_spacer(height=10)
            dpg.add_text("원문 표시 설정", color=THEME.accent)
            dpg.add_separator()

            # Original text font dropdown
            with dpg.group(horizontal=True):
                dpg.add_text("원문 폰트:", color=THEME.text_secondary)

                current_orig_font = self._config.subtitle.ass.original_font
                current_orig_display = font_mgr.get_display_name(current_orig_font)

                self._inputs["original_font"] = dpg.add_combo(
                    items=font_names,
                    default_value=current_orig_display,
                    width=200,
                    callback=self._on_original_font_changed,
                )

                # Original font status label
                self._original_font_status_label = dpg.add_text(
                    "", color=THEME.text_secondary
                )

                # Install button for original font
                self._original_font_install_btn = dpg.add_button(
                    label="설치",
                    callback=self._on_original_font_install,
                    width=50,
                )

            # Update initial original font status
            self._update_original_font_status(current_orig_display)

            with dpg.group(horizontal=True):
                dpg.add_text("원문 크기:", color=THEME.text_secondary)
                self._inputs["original_size"] = dpg.add_input_int(
                    default_value=self._config.subtitle.ass.original_size,
                    width=100,
                    min_value=10,
                    max_value=100,
                )

    def _update_font_status(self, display_name: str) -> None:
        """Update font status display."""
        font_mgr = get_font_manager()
        installed, status = font_mgr.get_font_status(display_name)

        # Update status label
        if installed:
            dpg.set_value(self._font_status_label, f"[{status}]")
            dpg.configure_item(self._font_status_label, color=THEME.success)
        else:
            dpg.set_value(self._font_status_label, f"[{status}]")
            dpg.configure_item(self._font_status_label, color=THEME.warning)

        # Show/hide install button (only if not installed)
        dpg.configure_item(self._font_install_btn, show=not installed)

    def _on_font_changed(self, sender, app_data, user_data) -> None:
        """Handle font selection change."""
        self._update_font_status(app_data)

    def _on_font_install(self) -> None:
        """Handle font install button click."""
        import threading

        display_name = dpg.get_value(self._inputs["default_font"])
        font_mgr = get_font_manager()

        # Update button to show progress
        dpg.configure_item(self._font_install_btn, label="...", enabled=False)

        def do_install():
            def update_status(msg: str):
                dpg.set_value(self._font_status_label, f"[{msg}]")

            success = font_mgr.install_font(display_name, update_status)

            # Update UI on completion
            if success:
                dpg.set_value(self._font_status_label, "[설치됨]")
                dpg.configure_item(self._font_status_label, color=THEME.success)
                dpg.configure_item(self._font_install_btn, show=False)
            else:
                dpg.set_value(self._font_status_label, "[설치 실패]")
                dpg.configure_item(self._font_status_label, color=THEME.error)
                dpg.configure_item(
                    self._font_install_btn, label="설치", enabled=True
                )

        # Run in background thread
        thread = threading.Thread(target=do_install, daemon=True)
        thread.start()

    def _update_original_font_status(self, display_name: str) -> None:
        """Update original font status display."""
        font_mgr = get_font_manager()
        installed, status = font_mgr.get_font_status(display_name)

        # Update status label
        if installed:
            dpg.set_value(self._original_font_status_label, f"[{status}]")
            dpg.configure_item(self._original_font_status_label, color=THEME.success)
        else:
            dpg.set_value(self._original_font_status_label, f"[{status}]")
            dpg.configure_item(self._original_font_status_label, color=THEME.warning)

        # Show/hide install button (only if not installed)
        dpg.configure_item(self._original_font_install_btn, show=not installed)

    def _on_original_font_changed(self, sender, app_data, user_data) -> None:
        """Handle original font selection change."""
        self._update_original_font_status(app_data)

    def _on_model_preset_changed(self, sender, app_data, user_data) -> None:
        """Handle model preset selection change."""
        # Find the selected preset key from display name
        selected_idx = None
        for i, key in enumerate(self._preset_keys):
            preset = TRANSLATION_MODEL_PRESETS[key]
            if key == "custom":
                display = f"{preset.name}"
            else:
                display = f"{preset.name} (VRAM {preset.vram_gb:.0f}GB)"
            if display == app_data:
                selected_idx = i
                break

        if selected_idx is None:
            return

        preset_key = self._preset_keys[selected_idx]
        preset = TRANSLATION_MODEL_PRESETS[preset_key]

        # Update description
        dpg.set_value(self._model_desc_label, f"  {preset.description}")

        # Show/hide custom model path input
        is_custom = preset_key == "custom"
        dpg.configure_item(self._custom_model_group, show=is_custom)

    def _on_original_font_install(self) -> None:
        """Handle original font install button click."""
        import threading

        display_name = dpg.get_value(self._inputs["original_font"])
        font_mgr = get_font_manager()

        # Update button to show progress
        dpg.configure_item(self._original_font_install_btn, label="...", enabled=False)

        def do_install():
            def update_status(msg: str):
                dpg.set_value(self._original_font_status_label, f"[{msg}]")

            success = font_mgr.install_font(display_name, update_status)

            # Update UI on completion
            if success:
                dpg.set_value(self._original_font_status_label, "[설치됨]")
                dpg.configure_item(self._original_font_status_label, color=THEME.success)
                dpg.configure_item(self._original_font_install_btn, show=False)
            else:
                dpg.set_value(self._original_font_status_label, "[설치 실패]")
                dpg.configure_item(self._original_font_status_label, color=THEME.error)
                dpg.configure_item(
                    self._original_font_install_btn, label="설치", enabled=True
                )

        # Run in background thread
        thread = threading.Thread(target=do_install, daemon=True)
        thread.start()

    def _get_resolution_label(self, width: int, height: int) -> str:
        """Get resolution preset label from width/height."""
        for label, w, h in self.RESOLUTION_PRESETS:
            if w == width and h == height:
                return label
        # Default to 1080p if not found
        return "1080p (1920x1080)"

    def _get_resolution_from_label(self, label: str) -> tuple[int, int]:
        """Get width/height from resolution preset label."""
        for preset_label, w, h in self.RESOLUTION_PRESETS:
            if preset_label == label:
                return w, h
        # Default to 1080p
        return 1920, 1080

    def _open_url(self, url: str) -> None:
        """Open URL in default browser."""
        import webbrowser

        webbrowser.open(url)

    def _browse_folder(self, input_key: str) -> None:
        """Browse for folder."""

        def on_select(folder: str):
            if folder:
                dpg.set_value(self._inputs[input_key], folder)

        FileDialogHelper.show_folder_dialog(on_select)

    def _browse_file(self, input_key: str) -> None:
        """Browse for file."""

        def on_select(files: list[str]):
            if files:
                dpg.set_value(self._inputs[input_key], files[0])

        FileDialogHelper.show_file_dialog(
            on_select,
            extensions=[(".gguf", "GGUF Model"), (".*", "All Files")],
            allow_multiple=False,
        )

    def _on_save(self) -> None:
        """Handle save button click."""
        # Update config from inputs
        self._config.models.path = dpg.get_value(self._inputs["models_path"])
        self._config.models.auto_download = dpg.get_value(self._inputs["auto_download"])

        self._config.processing.normalize_audio = dpg.get_value(
            self._inputs["normalize_audio"]
        )
        self._config.processing.target_sample_rate = dpg.get_value(
            self._inputs["sample_rate"]
        )

        # Segment splitting settings
        self._config.segment.enabled = dpg.get_value(self._inputs["segment_enabled"])
        self._config.segment.max_chars = dpg.get_value(
            self._inputs["segment_max_chars"]
        )
        self._config.segment.max_duration = dpg.get_value(
            self._inputs["segment_max_duration"]
        )

        # Logging settings
        self._config.ui.hide_warnings = dpg.get_value(self._inputs["hide_warnings"])

        # STT settings
        self._config.speech.stt.model_size = dpg.get_value(self._inputs["whisper_model"])
        self._config.speech.stt.device = dpg.get_value(self._inputs["stt_device"])
        self._config.speech.stt.compute_type = dpg.get_value(self._inputs["compute_type"])
        self._config.speech.stt.beam_size = dpg.get_value(self._inputs["beam_size"])
        self._config.speech.stt.vad_filter = dpg.get_value(self._inputs["vad_filter"])

        # Get selected model preset
        preset_display = dpg.get_value(self._inputs["model_preset"])
        for i, key in enumerate(self._preset_keys):
            preset = TRANSLATION_MODEL_PRESETS[key]
            if key == "custom":
                display = f"{preset.name}"
            else:
                display = f"{preset.name} (VRAM {preset.vram_gb:.0f}GB)"
            if display == preset_display:
                self._config.translation.model_preset = key
                break

        self._config.translation.model_path = dpg.get_value(
            self._inputs["translation_model"]
        )
        self._config.translation.n_gpu_layers = dpg.get_value(
            self._inputs["n_gpu_layers"]
        )
        self._config.translation.n_ctx = dpg.get_value(self._inputs["n_ctx"])
        self._config.translation.max_tokens = dpg.get_value(self._inputs["max_tokens"])
        self._config.translation.temperature = dpg.get_value(
            self._inputs["temperature"]
        )
        self._config.translation.enable_review = dpg.get_value(
            self._inputs["enable_review"]
        )

        self._config.subtitle.default_format = dpg.get_value(
            self._inputs["default_format"]
        )
        self._config.subtitle.include_original = dpg.get_value(
            self._inputs["include_original"]
        )

        # Parse resolution from preset
        resolution_label = dpg.get_value(self._inputs["resolution_preset"])
        width, height = self._get_resolution_from_label(resolution_label)
        self._config.subtitle.ass.video_width = width
        self._config.subtitle.ass.video_height = height

        # Convert font display name to family name
        font_display_name = dpg.get_value(self._inputs["default_font"])
        font_mgr = get_font_manager()
        self._config.subtitle.ass.default_font = font_mgr.get_font_family_name(
            font_display_name
        )
        self._config.subtitle.ass.default_size = dpg.get_value(
            self._inputs["font_size"]
        )

        # Original text font settings
        orig_font_display = dpg.get_value(self._inputs["original_font"])
        self._config.subtitle.ass.original_font = font_mgr.get_font_family_name(
            orig_font_display
        )
        self._config.subtitle.ass.original_size = dpg.get_value(
            self._inputs["original_size"]
        )

        # Save to file
        self.config_manager.save(self._config)

        if self.on_save:
            self.on_save(self._config)

        self.close()

    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        self.close()

    def close(self) -> None:
        """Close the dialog."""
        if self._dialog_tag and dpg.does_item_exist(self._dialog_tag):
            dpg.delete_item(self._dialog_tag)
            self._dialog_tag = 0

        self._inputs.clear()

    @property
    def is_open(self) -> bool:
        """Check if dialog is open."""
        return bool(self._dialog_tag and dpg.does_item_exist(self._dialog_tag))
