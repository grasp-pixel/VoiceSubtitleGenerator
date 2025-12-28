"""Main window for Voice Subtitle Generator."""

import logging
import threading
from pathlib import Path
from typing import Callable

import dearpygui.dearpygui as dpg

from src import (
    AudioLoader,
    ConfigManager,
    PipelineCallbacks,
    ProcessingStage,
    ProcessResult,
    SubtitlePipeline,
)

from .components import (
    FileDialogHelper,
    FileStatus,
    THEME,
    show_confirm_dialog,
    show_message_box,
)
from .file_list_panel import FileItem, FileListPanel
from .model_dialog import ModelDialog
from .preview_panel import LogPanel, PreviewPanel, ProcessingPanel
from .settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)


class MainWindow:
    """
    Main application window.

    Orchestrates all UI components and pipeline processing.
    """

    SUBTITLE_FORMATS = ["srt", "ass", "vtt"]

    def __init__(
        self,
        config_manager: ConfigManager,
        pipeline: SubtitlePipeline | None = None,
    ):
        """
        Initialize main window.

        Args:
            config_manager: Configuration manager.
            pipeline: Optional pre-initialized pipeline.
        """
        self.config_manager = config_manager
        self.config = config_manager.load()

        self._pipeline = pipeline
        self._is_processing = False
        self._should_cancel = False

        # Thread-safe pending files queue
        self._pending_files: list[str] = []
        self._pending_folders: list[str] = []
        self._pending_lock = threading.Lock()

        # UI components
        self._file_list: FileListPanel | None = None
        self._preview_panel: PreviewPanel | None = None
        self._processing_panel: ProcessingPanel | None = None
        self._log_panel: LogPanel | None = None
        self._settings_dialog: SettingsDialog | None = None
        self._model_dialog: ModelDialog | None = None

        # Status bar elements
        self._status_labels: dict = {}

    def build(self, parent: int | str) -> None:
        """
        Build the main window UI.

        Args:
            parent: Parent window tag.
        """
        self._parent = parent

        # Menu bar
        self._build_menu_bar()

        # Toolbar
        self._build_toolbar()

        dpg.add_separator(parent=parent)

        # Main content area with calculated height
        with dpg.child_window(
            parent=parent,
            border=False,
            height=-56,  # Leave space for status bar (scaled)
            autosize_x=True,
        ) as content_container:
            # File list panel
            self._file_list = FileListPanel(
                parent=content_container,
                on_select=self._on_file_select,
            )

            dpg.add_spacer(height=5, parent=content_container)

            # Processing panel (progress + stages)
            self._processing_panel = ProcessingPanel(parent=content_container)

            dpg.add_spacer(height=5, parent=content_container)

            # Log panel
            self._log_panel = LogPanel(parent=content_container)

            dpg.add_spacer(height=5, parent=content_container)

            # Preview panel
            self._preview_panel = PreviewPanel(
                parent=content_container,
                include_original=self.config.subtitle.include_original,
            )

        # Status bar
        self._build_status_bar()

        # Initialize dialogs
        self._settings_dialog = SettingsDialog(
            self.config_manager,
            on_save=self._on_settings_save,
        )
        self._model_dialog = ModelDialog(config=self.config)

    def _build_menu_bar(self) -> None:
        """Build menu bar."""
        with dpg.menu_bar(parent=self._parent):
            with dpg.menu(label="파일"):
                dpg.add_menu_item(
                    label="파일 추가...",
                    callback=self._on_add_files,
                    shortcut="Ctrl+O",
                )
                dpg.add_menu_item(
                    label="폴더 추가...",
                    callback=self._on_add_folder,
                    shortcut="Ctrl+Shift+O",
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="대기열 비우기",
                    callback=self._on_clear_queue,
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="종료",
                    callback=self._on_exit,
                )

            with dpg.menu(label="처리"):
                dpg.add_menu_item(
                    label="처리 시작",
                    callback=self._on_start_processing,
                    shortcut="F5",
                )
                dpg.add_menu_item(
                    label="취소",
                    callback=self._on_cancel_processing,
                    shortcut="Esc",
                )

            with dpg.menu(label="설정"):
                dpg.add_menu_item(
                    label="환경설정...",
                    callback=self._on_open_settings,
                )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="모델 관리...",
                    callback=self._on_open_models,
                )

            with dpg.menu(label="도움말"):
                dpg.add_menu_item(
                    label="정보",
                    callback=self._on_about,
                )

    def _build_toolbar(self) -> None:
        """Build toolbar."""
        with dpg.group(horizontal=True, parent=self._parent):
            dpg.add_button(
                label="파일 추가",
                callback=self._on_add_files,
                width=100,
            )
            dpg.add_button(
                label="폴더 추가",
                callback=self._on_add_folder,
                width=100,
            )
            dpg.add_button(
                label="비우기",
                callback=self._on_clear_queue,
                width=80,
            )

            dpg.add_spacer(width=20)

            dpg.add_text("형식:", color=THEME.text_secondary)
            self._format_combo = dpg.add_combo(
                items=self.SUBTITLE_FORMATS,
                default_value=self.config.subtitle.default_format,
                width=80,
            )

            dpg.add_spacer(width=20)

            self._start_button = dpg.add_button(
                label="시작",
                callback=self._on_start_processing,
                width=100,
            )
            self._cancel_button = dpg.add_button(
                label="취소",
                callback=self._on_cancel_processing,
                width=80,
                enabled=False,
            )

            # Right-aligned model management button
            dpg.add_spacer(width=-1)
            dpg.add_button(
                label="모델 관리",
                callback=self._on_open_models,
                width=100,
            )

    def _build_status_bar(self) -> None:
        """Build status bar."""
        dpg.add_separator(parent=self._parent)

        with dpg.group(horizontal=True, parent=self._parent):
            # Model status
            self._status_labels["whisper"] = dpg.add_text(
                "Whisper: ---",
                color=THEME.text_secondary,
            )
            dpg.add_spacer(width=10)

            self._status_labels["llm"] = dpg.add_text(
                "번역: ---",
                color=THEME.text_secondary,
            )

            dpg.add_spacer(width=-1)

            # File count
            self._status_labels["files"] = dpg.add_text(
                "파일: 0/0",
                color=THEME.text_secondary,
            )

    # Event handlers

    def _on_add_files(self) -> None:
        """Handle add files action."""
        # Use default media file filter from FileDialogHelper
        FileDialogHelper.show_file_dialog(
            callback=self._add_files_to_queue,
        )

    def _on_add_folder(self) -> None:
        """Handle add folder action."""
        FileDialogHelper.show_folder_dialog(
            callback=self._add_folder_to_queue,
        )

    def _add_files_to_queue(self, files: list[str]) -> None:
        """Add files to pending queue (thread-safe, called from file dialog thread)."""
        logger.info(f"_add_files_to_queue called with {len(files)} files: {files}")
        with self._pending_lock:
            for file_path in files:
                if AudioLoader.is_supported(file_path):
                    self._pending_files.append(file_path)
                    logger.info(f"Added to pending: {file_path}")

    def _add_folder_to_queue(self, folder: str) -> None:
        """Add folder to pending queue (thread-safe, called from file dialog thread)."""
        if not folder:
            return
        with self._pending_lock:
            self._pending_folders.append(folder)

    def process_pending(self) -> None:
        """Process pending files/folders from queue (call from main thread)."""
        files_to_add: list[str] = []
        folders_to_add: list[str] = []

        with self._pending_lock:
            if self._pending_files:
                files_to_add = self._pending_files.copy()
                self._pending_files.clear()
            if self._pending_folders:
                folders_to_add = self._pending_folders.copy()
                self._pending_folders.clear()

        # Now add to UI (main thread)
        for file_path in files_to_add:
            self._file_list.add_file(file_path)

        for folder in folders_to_add:
            folder_path = Path(folder)
            for ext in AudioLoader.supported_formats():
                for file_path in folder_path.glob(f"*.{ext}"):
                    self._file_list.add_file(str(file_path))

        if files_to_add or folders_to_add:
            self._update_status()

    def add_dropped_files(self, paths: list[str]) -> None:
        """
        Handle files/folders dropped onto the application.

        Args:
            paths: List of dropped file or folder paths.
        """
        if self._is_processing:
            show_message_box(
                "추가 불가",
                "처리 중에는 파일을 추가할 수 없습니다.",
                "warning",
            )
            return

        added_count = 0
        existing_paths = {f.path for f in self._file_list.get_all_files()}

        for path_str in paths:
            path = Path(path_str)

            if path.is_file():
                # Single file
                if AudioLoader.is_supported(path_str) and path_str not in existing_paths:
                    self._file_list.add_file(path_str)
                    existing_paths.add(path_str)
                    added_count += 1
            elif path.is_dir():
                # Folder - add all supported audio files (including subdirectories)
                for ext in AudioLoader.supported_formats():
                    for file_path in path.rglob(f"*.{ext}"):
                        file_str = str(file_path)
                        if file_str not in existing_paths:
                            self._file_list.add_file(file_str)
                            existing_paths.add(file_str)
                            added_count += 1

        self._update_status()

        if added_count > 0:
            logger.info(f"Added {added_count} file(s) via drag and drop")

    def _on_clear_queue(self) -> None:
        """Handle clear queue action."""
        if self._is_processing:
            show_message_box(
                "비우기 불가",
                "처리 중에는 대기열을 비울 수 없습니다.",
                "warning",
            )
            return

        if self._file_list.has_files():
            show_confirm_dialog(
                "대기열 비우기",
                "대기열의 모든 파일을 삭제하시겠습니까?",
                on_confirm=self._do_clear_queue,
            )

    def _do_clear_queue(self) -> None:
        """Actually clear the queue."""
        self._file_list.clear()
        self._preview_panel.clear()
        self._update_status()

    def _on_file_select(self, item: FileItem | None) -> None:
        """Handle file selection."""
        if item is None:
            self._preview_panel.clear()
            return

        self._preview_panel.set_file(item.filename)

        # If file is processed, show segments
        if item.status == FileStatus.DONE and item.segments:
            self._preview_panel.set_segments(item.segments)

    def _on_start_processing(self) -> None:
        """Handle start processing action."""
        if self._is_processing:
            return

        if not self._file_list.has_pending():
            show_message_box(
                "파일 없음",
                "처리할 대기 파일이 없습니다.",
                "info",
            )
            return

        # Start processing in background thread
        self._is_processing = True
        self._should_cancel = False

        dpg.configure_item(self._start_button, enabled=False)
        dpg.configure_item(self._cancel_button, enabled=True)

        thread = threading.Thread(target=self._process_files)
        thread.daemon = True
        thread.start()

    def _on_cancel_processing(self) -> None:
        """Handle cancel processing action."""
        if not self._is_processing:
            return

        self._should_cancel = True
        logger.info("Cancellation requested")

    def _process_files(self) -> None:
        """Process files in background thread."""
        try:
            # Reset processing panel
            self._processing_panel.reset()
            self._log_panel.info("처리 시작...")

            # Start initialization stage (index 0)
            self._processing_panel.set_stage(0)

            # Initialize pipeline if needed
            if self._pipeline is None:
                self._log_panel.info("파이프라인 초기화 중...")
                self._pipeline = SubtitlePipeline(config=self.config)

            self._log_panel.info("모델 로딩 중... (시간이 걸릴 수 있습니다)")
            self._pipeline.initialize()
            self._log_panel.success("모델 로딩 완료")

            # End initialization stage
            self._processing_panel.end_stage("initializing")

            # Update device info in processing panel
            whisper_device = self._pipeline.speech_engine.device
            translator_using_gpu = self._pipeline.translator.is_using_gpu
            self._processing_panel.set_device_info(whisper_device, translator_using_gpu)

            # Log device info
            if whisper_device.lower() not in ("cuda", "gpu"):
                self._log_panel.warning("Whisper: CPU 모드로 실행 중 (느림)")
            if not translator_using_gpu:
                self._log_panel.warning("번역: CPU 모드로 실행 중 (느림)")

            # Update status
            dpg.set_value(self._status_labels["whisper"], "Whisper: 로드됨")
            dpg.configure_item(self._status_labels["whisper"], color=THEME.success)
            dpg.set_value(self._status_labels["llm"], "번역: 로드됨")
            dpg.configure_item(self._status_labels["llm"], color=THEME.success)

            # Get files to process
            pending_files = self._file_list.get_pending_files()
            output_format = dpg.get_value(self._format_combo)
            total = len(pending_files)
            completed = 0

            self._log_panel.info(f"처리할 파일: {total}개")

            for idx, item in enumerate(pending_files):
                if self._should_cancel:
                    self._log_panel.warning("사용자에 의해 취소됨")
                    break

                # Update status
                self._file_list.update_status(item, FileStatus.PROCESSING)
                self._log_panel.info(f"[{idx+1}/{total}] {item.filename}")

                # Stage mapping (matches ProcessingPanel.STAGES order)
                stage_map = {
                    ProcessingStage.LOADING: 1,
                    ProcessingStage.TRANSCRIBING: 2,
                    ProcessingStage.TRANSLATING: 3,
                    ProcessingStage.WRITING: 4,
                }

                # Create callbacks
                def on_progress(p: float, item=item):
                    self._file_list.update_status(item, FileStatus.PROCESSING, p)
                    self._processing_panel.set_progress(p)

                def on_stage(stage: ProcessingStage):
                    if stage in stage_map:
                        self._processing_panel.set_stage(stage_map[stage])

                def on_log(message: str, level: str = "info"):
                    if level == "success":
                        self._log_panel.success(message)
                    elif level == "warning":
                        self._log_panel.warning(message)
                    elif level == "error":
                        self._log_panel.error(message)
                    else:
                        self._log_panel.info(message)

                callbacks = PipelineCallbacks(
                    on_progress=on_progress,
                    on_stage_change=on_stage,
                    on_log=on_log,
                )

                try:
                    # Determine output path
                    input_path = Path(item.path)
                    output_path = input_path.with_suffix(f".{output_format}")

                    # Process file
                    result = self._pipeline.process_file(
                        audio_path=item.path,
                        output_path=str(output_path),
                        output_format=output_format,
                        callbacks=callbacks,
                    )

                    if result.success:
                        self._file_list.update_status(
                            item, FileStatus.DONE, 1.0
                        )
                        # Store processing result for preview
                        self._file_list.set_result(item, result.segments)
                        completed += 1
                        self._log_panel.success(
                            f"완료: {len(result.segments)}개 구간"
                        )
                        # Update profiling info
                        self._processing_panel.set_segment_info(
                            len(result.segments)
                        )
                        if result.duration:
                            self._processing_panel.set_audio_info(
                                result.duration, 16000  # Default sample rate
                            )
                        self._processing_panel.set_complete()
                    else:
                        self._file_list.update_status(
                            item, FileStatus.ERROR, 0.0, result.error
                        )
                        self._log_panel.error(f"실패: {result.error}")

                except Exception as e:
                    logger.error(f"Processing failed: {e}")
                    self._file_list.update_status(
                        item, FileStatus.ERROR, 0.0, str(e)
                    )
                    self._log_panel.error(f"오류: {str(e)}")

                # Update file count
                dpg.set_value(
                    self._status_labels["files"],
                    f"파일: {completed}/{total}",
                )

            self._log_panel.success(f"처리 완료: {completed}/{total} 성공")
            logger.info(f"Processing complete: {completed}/{total}")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._log_panel.error(f"파이프라인 오류: {str(e)}")
            show_message_box("Error", str(e), "error")

        finally:
            self._is_processing = False
            dpg.configure_item(self._start_button, enabled=True)
            dpg.configure_item(self._cancel_button, enabled=False)

    def _on_open_settings(self) -> None:
        """Handle open settings action."""
        self._settings_dialog.show()

    def _on_open_models(self) -> None:
        """Handle open model management action."""
        self._model_dialog.show()

    def _on_settings_save(self, config) -> None:
        """Handle settings saved."""
        self.config = config

        # Apply logging configuration (local import to avoid circular dependency)
        from .app import configure_logging
        configure_logging(config.ui.hide_warnings)

        # Update preview panel settings
        if self._preview_panel:
            self._preview_panel.include_original = config.subtitle.include_original

        # Update model dialog config reference
        if self._model_dialog:
            self._model_dialog.config = config

        # Recreate pipeline with new config
        if self._pipeline:
            self._pipeline.cleanup()
            self._pipeline = None

    def _on_about(self) -> None:
        """Show about dialog."""
        show_message_box(
            "정보",
            "음성 자막 생성기 v0.1.0\n\n"
            "일본어 음성을 한국어 자막으로 변환",
            "info",
        )

    def _on_exit(self) -> None:
        """Handle exit action."""
        if self._is_processing:
            show_confirm_dialog(
                "종료",
                "처리가 진행 중입니다. 종료하시겠습니까?",
                on_confirm=dpg.stop_dearpygui,
            )
        else:
            dpg.stop_dearpygui()

    def _update_status(self) -> None:
        """Update status bar."""
        total = self._file_list.get_file_count()
        completed = self._file_list.get_completed_count()

        dpg.set_value(
            self._status_labels["files"],
            f"파일: {completed}/{total}",
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._pipeline:
            self._pipeline.cleanup()
            self._pipeline = None
