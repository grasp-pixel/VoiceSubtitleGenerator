"""Voice Subtitle Generator - Main Entry Point.

A tool for generating Korean subtitles from Japanese audio files
with speaker diarization support.
"""

import argparse
import faulthandler
import logging
import sys

# Enable faulthandler to get traceback on crash
faulthandler.enable()
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Voice Subtitle Generator - Japanese to Korean subtitle generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # GUI command (default)
    gui_parser = subparsers.add_parser("gui", help="Launch GUI application")
    gui_parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to config file",
    )

    # Process command (CLI mode)
    process_parser = subparsers.add_parser("process", help="Process audio files")
    process_parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Input audio file(s) or directory",
    )
    process_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory",
    )
    process_parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["srt", "ass", "vtt"],
        default="srt",
        help="Output subtitle format",
    )
    process_parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization",
    )
    process_parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to config file",
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration utilities")
    config_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration",
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration",
    )

    return parser


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch GUI application."""
    logger.info("Launching GUI...")

    try:
        from ui.app import run_app

        return run_app(args.config)

    except ImportError as e:
        logger.error(f"Failed to import GUI module: {e}")
        logger.error("Make sure dearpygui is installed: pip install dearpygui")
        return 1
    except Exception as e:
        logger.error(f"GUI error: {e}")
        return 1


def cmd_process(args: argparse.Namespace) -> int:
    """Process audio files in CLI mode."""
    from src import (
        AudioLoader,
        BatchCallbacks,
        ConfigManager,
        ProcessResult,
        SubtitlePipeline,
    )

    # Load config
    config_manager = ConfigManager(args.config)
    config = config_manager.load()

    # Override diarization setting if --no-diarization
    if args.no_diarization:
        config.speech.diarization.enabled = False

    # Validate config
    errors = config_manager.validate()
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        return 1

    # Collect input files
    input_files: list[Path] = []
    for input_path in args.input:
        path = Path(input_path)
        if path.is_file():
            if AudioLoader.is_supported(str(path)):
                input_files.append(path)
            else:
                logger.warning(f"Unsupported format: {path}")
        elif path.is_dir():
            for ext in AudioLoader.supported_formats():
                input_files.extend(path.glob(f"*.{ext}"))
        else:
            logger.warning(f"Path not found: {path}")

    if not input_files:
        logger.error("No valid audio files found")
        return 1

    logger.info(f"Found {len(input_files)} audio file(s)")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup callbacks
    def on_file_start(path: str, current: int, total: int) -> None:
        logger.info(f"[{current}/{total}] Processing: {Path(path).name}")

    def on_file_complete(result: ProcessResult) -> None:
        if result.success:
            logger.info(f"  Saved: {result.output_path}")
            logger.info(f"  Segments: {len(result.segments)}, Speakers: {len(result.speakers)}")
        else:
            logger.error(f"  Failed: {result.error}")

    callbacks = BatchCallbacks(
        on_file_start=on_file_start,
        on_file_complete=on_file_complete,
    )

    # Initialize pipeline
    pipeline = SubtitlePipeline(config=config)

    try:
        # Process all files
        results = pipeline.process_batch(
            files=[str(f) for f in input_files],
            output_format=args.format,
            output_dir=str(output_dir),
            callbacks=callbacks,
        )

        # Summary
        success_count = sum(1 for r in results if r.success)
        fail_count = len(results) - success_count

        logger.info(f"Processing complete: {success_count} success, {fail_count} failed")

        return 0 if fail_count == 0 else 1

    finally:
        pipeline.cleanup()


def cmd_config(args: argparse.Namespace) -> int:
    """Configuration utilities."""
    from src import ConfigManager

    config_manager = ConfigManager()

    if args.validate:
        errors = config_manager.validate()
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("Configuration is valid")
            return 0

    if args.show:
        config = config_manager.load()
        print(f"Language: {config.language}")
        print(f"STT model: {config.speech.stt.model_size}")
        print(f"Device: {config.speech.stt.device}")
        print(f"Translation model: {config.translation.model_path}")
        print(f"Diarization: {'enabled' if config.speech.diarization.enabled else 'disabled'}")
        return 0

    return 0


def main() -> int:
    """Main entry point."""
    parser = setup_argparser()
    args = parser.parse_args()

    if args.command is None:
        # Default to GUI
        args.command = "gui"
        args.config = "config/settings.yaml"

    if args.command == "gui":
        return cmd_gui(args)
    elif args.command == "process":
        return cmd_process(args)
    elif args.command == "config":
        return cmd_config(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
