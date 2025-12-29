"""Main application entry point for Voice Subtitle Generator GUI."""

# Apply PyTorch compatibility patches FIRST, before any other imports
import src.torch_compat  # noqa: F401

import logging
import sys
from pathlib import Path

import dearpygui.dearpygui as dpg

from src.config import ConfigManager
from src.utils import get_app_path

from .components import create_theme
from .main_window import MainWindow

logger = logging.getLogger(__name__)


def download_default_font(fonts_dir: Path) -> Path | None:
    """Download default CJK font if not present.

    Returns:
        Path to downloaded font, or None if failed.
    """
    import ssl
    import urllib.request

    font_file = "NotoSansCJKjp-Regular.otf"
    font_path = fonts_dir / font_file

    if font_path.exists():
        return font_path

    fonts_dir.mkdir(parents=True, exist_ok=True)

    # Direct link from GitHub notofonts
    font_url = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"

    logger.info("Downloading CJK font for first run...")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(font_url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
            data = response.read()
            font_path.write_bytes(data)

        logger.info(f"Downloaded font: {font_path.name} ({font_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return font_path

    except Exception as e:
        logger.warning(f"Failed to download font: {e}")
        return None


def setup_font() -> None:
    """Setup font with clear rendering using 2x render + 0.5 scale.

    This workaround addresses DearPyGui's HiDPI issues by rendering
    fonts at 2x size and scaling down for crisp anti-aliasing.
    See: https://github.com/hoffstadt/DearPyGui/issues/1380
    """
    # Project fonts (Noto Sans CJK - supports Japanese, Korean, Chinese)
    fonts_dir = get_app_path() / "fonts"

    # Font priority: Project font > System fonts
    font_paths = [
        fonts_dir / "NotoSansCJKjp-Regular.otf",
        fonts_dir / "NotoSansJP-Regular.ttf",
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),  # macOS
    ]

    font_path = None
    for path in font_paths:
        if path.exists():
            font_path = str(path)
            break

    # Auto-download if no font found
    if not font_path:
        downloaded = download_default_font(fonts_dir)
        if downloaded:
            font_path = str(downloaded)

    if not font_path:
        logger.warning("No suitable font found, using default")
        return

    # HiDPI clear rendering: render at 2x size, then scale down
    base_size = 22
    small_size = 18
    render_scale = 2

    try:
        with dpg.font_registry():
            # Main font (16px displayed, 32px rendered)
            font = dpg.add_font(font_path, base_size * render_scale, tag="font_default")
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean, parent=font)
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese, parent=font)
            dpg.add_font_range(0x0020, 0x00FF, parent=font)  # Basic Latin
            dpg.add_font_range(0x2600, 0x26FF, parent=font)  # Misc symbols
            dpg.add_font_range(0x2700, 0x27BF, parent=font)  # Dingbats ✓✗

            # Small font for original text (13px displayed, 26px rendered)
            font_small = dpg.add_font(
                font_path, small_size * render_scale, tag="font_small"
            )
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean, parent=font_small)
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese, parent=font_small)
            dpg.add_font_range(0x0020, 0x00FF, parent=font_small)
            dpg.add_font_range(0x2600, 0x26FF, parent=font_small)
            dpg.add_font_range(0x2700, 0x27BF, parent=font_small)

        dpg.bind_font(font)
        logger.info(f"Loaded font: {font_path} (2x render for HiDPI)")
    except Exception as e:
        logger.warning(f"Failed to load font: {e}")


def configure_logging(hide_warnings: bool = False) -> None:
    """
    Configure logging levels for libraries.

    Args:
        hide_warnings: If True, suppress WARNING level logs from libraries.
    """
    # Libraries that produce verbose warnings
    noisy_libraries = [
        "transformers",
        "huggingface_hub",
        "torch",
        "torchaudio",
        "faster_whisper",
        "pyannote",
        "llama_cpp",
        "urllib3",
        "filelock",
        "numba",
    ]

    level = logging.ERROR if hide_warnings else logging.WARNING
    for lib in noisy_libraries:
        logging.getLogger(lib).setLevel(level)


def run_app(config_path: str | None = None) -> int:
    """
    Run the Voice Subtitle Generator application.

    Args:
        config_path: Optional path to configuration file.

    Returns:
        int: Exit code.
    """
    main_window: MainWindow | None = None

    try:
        logger.info("Initializing GUI...")

        # Initialize configuration first to get settings
        config_manager = ConfigManager(config_path)
        config = config_manager.load()

        # Configure logging based on settings
        configure_logging(config.ui.hide_warnings)

        dpg.create_context()
        dpg.create_viewport(
            title="Voice Subtitle Generator",
            width=1200,
            height=900,
            min_width=800,
            min_height=600,
        )
        dpg.setup_dearpygui()

        # Apply theme
        theme = create_theme()
        dpg.bind_theme(theme)

        # Setup font
        setup_font()

        # Create main window
        main_window = MainWindow(config_manager)

        # Build UI
        with dpg.window(tag="primary_window", no_scrollbar=True):
            main_window.build("primary_window")

        dpg.set_primary_window("primary_window", True)

        # Apply 0.5 scale for HiDPI clear font rendering (must be after setup)
        dpg.set_global_font_scale(0.5)

        logger.info("Starting main loop...")
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            # Process pending files from background threads
            main_window.process_pending()
            dpg.render_dearpygui_frame()

        return 0

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        if main_window:
            main_window.cleanup()
        dpg.destroy_context()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.exit(run_app())
