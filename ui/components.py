"""Common UI components for Voice Subtitle Generator."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import dearpygui.dearpygui as dpg


class FileStatus(Enum):
    """File processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class ThemeColors:
    """Application theme colors."""

    # Background colors
    bg_primary: tuple[int, int, int] = (30, 30, 30)
    bg_secondary: tuple[int, int, int] = (37, 37, 38)
    bg_panel: tuple[int, int, int] = (45, 45, 48)

    # Border and lines
    border: tuple[int, int, int] = (60, 60, 60)

    # Text colors
    text_primary: tuple[int, int, int] = (204, 204, 204)
    text_secondary: tuple[int, int, int] = (128, 128, 128)

    # Accent colors
    accent: tuple[int, int, int] = (0, 122, 204)
    accent_hover: tuple[int, int, int] = (28, 151, 234)

    # Status colors
    success: tuple[int, int, int] = (78, 201, 176)
    error: tuple[int, int, int] = (244, 71, 71)
    warning: tuple[int, int, int] = (204, 167, 0)
    info: tuple[int, int, int] = (0, 122, 204)


# Global theme colors
THEME = ThemeColors()


def create_theme() -> int:
    """Create and return the application theme."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            # Window background
            dpg.add_theme_color(
                dpg.mvThemeCol_WindowBg, THEME.bg_primary, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ChildBg, THEME.bg_secondary, category=dpg.mvThemeCat_Core
            )

            # Frame background
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg, THEME.bg_panel, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered,
                (55, 55, 58),
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive,
                (65, 65, 68),
                category=dpg.mvThemeCat_Core,
            )

            # Button
            dpg.add_theme_color(
                dpg.mvThemeCol_Button, THEME.accent, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonHovered,
                THEME.accent_hover,
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonActive,
                (0, 100, 180),
                category=dpg.mvThemeCat_Core,
            )

            # Header
            dpg.add_theme_color(
                dpg.mvThemeCol_Header, THEME.bg_panel, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderHovered,
                (55, 55, 58),
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderActive,
                THEME.accent,
                category=dpg.mvThemeCat_Core,
            )

            # Text
            dpg.add_theme_color(
                dpg.mvThemeCol_Text, THEME.text_primary, category=dpg.mvThemeCat_Core
            )

            # Border
            dpg.add_theme_color(
                dpg.mvThemeCol_Border, THEME.border, category=dpg.mvThemeCat_Core
            )

            # Tab
            dpg.add_theme_color(
                dpg.mvThemeCol_Tab, THEME.bg_panel, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabHovered,
                THEME.accent_hover,
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabActive, THEME.accent, category=dpg.mvThemeCat_Core
            )

            # Scrollbar
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarBg,
                THEME.bg_primary,
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrab,
                THEME.border,
                category=dpg.mvThemeCat_Core,
            )

            # Style
            dpg.add_theme_style(
                dpg.mvStyleVar_FrameRounding, 4, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_WindowRounding, 6, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_ChildRounding, 4, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_FramePadding, 8, 4, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_ItemSpacing, 8, 4, category=dpg.mvThemeCat_Core
            )

    return theme


def get_status_color(status: FileStatus) -> tuple[int, int, int]:
    """Get color for file status."""
    colors = {
        FileStatus.PENDING: THEME.text_secondary,
        FileStatus.PROCESSING: THEME.info,
        FileStatus.DONE: THEME.success,
        FileStatus.ERROR: THEME.error,
    }
    return colors.get(status, THEME.text_secondary)


def get_status_icon(status: FileStatus) -> str:
    """Get icon for file status."""
    icons = {
        FileStatus.PENDING: "...",
        FileStatus.PROCESSING: ">>>",
        FileStatus.DONE: "[v]",
        FileStatus.ERROR: "[x]",
    }
    return icons.get(status, "?")


class ProgressBar:
    """Custom progress bar component."""

    def __init__(
        self,
        parent: int | str,
        width: int = -1,
        height: int = 20,
        tag: str | None = None,
    ):
        """
        Create progress bar.

        Args:
            parent: Parent container tag.
            width: Bar width (-1 for full width).
            height: Bar height.
            tag: Optional tag for the component.
        """
        self.tag = tag or dpg.generate_uuid()
        self._progress = 0.0

        with dpg.group(parent=parent, horizontal=True, tag=self.tag):
            self._bar = dpg.add_progress_bar(
                default_value=0.0,
                width=width - 60 if width > 0 else -60,
            )
            self._label = dpg.add_text("0%", color=THEME.text_secondary)

    def set_progress(self, value: float) -> None:
        """Set progress value (0.0 to 1.0)."""
        self._progress = max(0.0, min(1.0, value))
        dpg.set_value(self._bar, self._progress)
        dpg.set_value(self._label, f"{int(self._progress * 100)}%")

    def reset(self) -> None:
        """Reset progress to 0."""
        self.set_progress(0.0)

    @property
    def progress(self) -> float:
        """Get current progress."""
        return self._progress


class StatusIndicator:
    """Status indicator component (checkmark or X)."""

    def __init__(
        self,
        parent: int | str,
        label: str,
        initial_status: bool = False,
        tag: str | None = None,
    ):
        """
        Create status indicator.

        Args:
            parent: Parent container tag.
            label: Label text.
            initial_status: Initial status (True = OK, False = Not OK).
            tag: Optional tag.
        """
        self.tag = tag or dpg.generate_uuid()

        with dpg.group(parent=parent, horizontal=True, tag=self.tag):
            self._icon = dpg.add_text("[?]")
            dpg.add_text(label)

        self.set_status(initial_status)

    def set_status(self, status: bool) -> None:
        """Set status."""
        if status:
            dpg.set_value(self._icon, "[v]")
            dpg.configure_item(self._icon, color=THEME.success)
        else:
            dpg.set_value(self._icon, "[x]")
            dpg.configure_item(self._icon, color=THEME.error)


class FileDialogHelper:
    """Helper for native OS file dialogs."""

    @staticmethod
    def show_file_dialog(
        callback: Callable[[list[str]], None],
        extensions: list[tuple[str, str]] | None = None,
        default_path: str = "",
        allow_multiple: bool = True,
    ) -> None:
        """
        Show native OS file open dialog.

        Args:
            callback: Callback with selected file paths.
            extensions: List of (extension, description) tuples.
            default_path: Default directory.
            allow_multiple: Allow multiple file selection.
        """
        import os
        import platform
        import subprocess
        import threading

        def open_dialog_macos():
            """macOS: use osascript to avoid main thread issues."""
            import logging
            log = logging.getLogger(__name__)
            try:
                if allow_multiple:
                    script = 'POSIX path of (choose file with multiple selections allowed)'
                    # Returns paths separated by ", " for multiple files
                    result = subprocess.run(
                        ['osascript', '-e', f'set files to ({script})',
                         '-e', 'set output to ""',
                         '-e', 'repeat with f in files',
                         '-e', 'set output to output & POSIX path of f & linefeed',
                         '-e', 'end repeat',
                         '-e', 'return output'],
                        capture_output=True, text=True, timeout=300
                    )
                else:
                    script = 'POSIX path of (choose file)'
                    result = subprocess.run(
                        ['osascript', '-e', script],
                        capture_output=True, text=True, timeout=300
                    )

                log.info(f"osascript returncode: {result.returncode}")
                log.info(f"osascript stdout: {repr(result.stdout)}")
                log.info(f"osascript stderr: {repr(result.stderr)}")

                if result.returncode == 0 and result.stdout.strip():
                    paths = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
                    log.info(f"Parsed paths: {paths}")
                    if paths:
                        normalized = [os.path.normpath(f) for f in paths]
                        log.info(f"Calling callback with: {normalized}")
                        callback(normalized)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"File dialog error: {e}")

        def open_dialog_other():
            """Windows/Linux: use plyer."""
            from plyer import filechooser

            if extensions:
                filters = []
                for ext, desc in extensions:
                    if ext == ".*":
                        filters.append((desc, "*.*"))
                    else:
                        filters.append((desc, f"*{ext}"))
            else:
                filters = [
                    ("Media Files", "*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg",
                     "*.wma", "*.aac", "*.opus", "*.mp4", "*.mkv", "*.avi",
                     "*.webm", "*.mov", "*.wmv", "*.flv", "*.ts", "*.m2ts"),
                ]

            try:
                result = filechooser.open_file(
                    title="파일 선택",
                    path=default_path or None,
                    filters=filters,
                    multiple=allow_multiple,
                )

                if result:
                    normalized = [os.path.normpath(f) for f in result]
                    callback(normalized)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"File dialog error: {e}")

        # Choose implementation based on platform
        if platform.system() == "Darwin":
            # macOS: run in thread (osascript is safe)
            thread = threading.Thread(target=open_dialog_macos, daemon=True)
            thread.start()
        else:
            # Windows/Linux: run in thread
            thread = threading.Thread(target=open_dialog_other, daemon=True)
            thread.start()

    @staticmethod
    def show_folder_dialog(
        callback: Callable[[str], None],
        default_path: str = "",
    ) -> None:
        """
        Show native OS folder selection dialog.

        Args:
            callback: Callback with selected folder path.
            default_path: Default directory.
        """
        import os
        import platform
        import subprocess
        import threading

        def open_dialog_macos():
            """macOS: use osascript to avoid main thread issues."""
            try:
                script = 'POSIX path of (choose folder)'
                result = subprocess.run(
                    ['osascript', '-e', script],
                    capture_output=True, text=True, timeout=300
                )

                if result.returncode == 0 and result.stdout.strip():
                    folder = result.stdout.strip()
                    callback(os.path.normpath(folder))
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Folder dialog error: {e}")

        def open_dialog_other():
            """Windows/Linux: use plyer."""
            from plyer import filechooser

            try:
                result = filechooser.choose_dir(
                    title="폴더 선택",
                    path=default_path or None,
                )

                if result:
                    folder = result[0] if isinstance(result, list) else result
                    callback(os.path.normpath(folder))
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Folder dialog error: {e}")

        # Choose implementation based on platform
        if platform.system() == "Darwin":
            thread = threading.Thread(target=open_dialog_macos, daemon=True)
            thread.start()
        else:
            thread = threading.Thread(target=open_dialog_other, daemon=True)
            thread.start()


def format_duration(seconds: float) -> str:
    """Format duration in HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def show_message_box(
    title: str,
    message: str,
    msg_type: str = "info",
    on_ok: Callable[[], None] | None = None,
) -> None:
    """
    Show a message box dialog.

    Args:
        title: Dialog title.
        message: Message text.
        msg_type: Message type ("info", "warning", "error").
        on_ok: Optional callback when OK is clicked.
    """
    color = {
        "info": THEME.info,
        "warning": THEME.warning,
        "error": THEME.error,
    }.get(msg_type, THEME.text_primary)

    # Error dialogs are larger to show full error messages
    is_error = msg_type == "error"
    width = 600 if is_error else 400
    height = 300 if is_error else 150

    def on_close():
        dpg.delete_item(dialog)
        if on_ok:
            on_ok()

    def copy_to_clipboard():
        try:
            import subprocess

            process = subprocess.Popen(
                ["clip"], stdin=subprocess.PIPE, shell=True
            )
            process.communicate(message.encode("utf-16-le"))
        except Exception:
            pass

    with dpg.window(
        label=title,
        modal=True,
        no_resize=False if is_error else True,
        width=width,
        height=height,
        pos=[
            dpg.get_viewport_width() // 2 - width // 2,
            dpg.get_viewport_height() // 2 - height // 2,
        ],
    ) as dialog:
        dpg.add_text(message, wrap=width - 20, color=color)
        dpg.add_spacer(height=20)
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=-1)
            if is_error:
                dpg.add_button(
                    label="복사", width=80, callback=lambda: copy_to_clipboard()
                )
                dpg.add_spacer(width=10)
            dpg.add_button(label="확인", width=80, callback=lambda: on_close())


def show_confirm_dialog(
    title: str,
    message: str,
    on_confirm: Callable[[], None],
    on_cancel: Callable[[], None] | None = None,
) -> None:
    """
    Show a confirmation dialog.

    Args:
        title: Dialog title.
        message: Message text.
        on_confirm: Callback when confirmed.
        on_cancel: Optional callback when cancelled.
    """

    def do_confirm():
        dpg.delete_item(dialog)
        on_confirm()

    def do_cancel():
        dpg.delete_item(dialog)
        if on_cancel:
            on_cancel()

    with dpg.window(
        label=title,
        modal=True,
        no_resize=True,
        width=400,
        height=150,
        pos=[
            dpg.get_viewport_width() // 2 - 200,
            dpg.get_viewport_height() // 2 - 75,
        ],
    ) as dialog:
        dpg.add_text(message, wrap=380)
        dpg.add_spacer(height=20)
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=-1)
            dpg.add_button(label="취소", width=80, callback=do_cancel)
            dpg.add_button(label="확인", width=80, callback=do_confirm)
