"""Logging configuration for Voice Subtitle Generator."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        super().__init__(fmt, datefmt)
        self._use_colors = sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        message = super().format(record)

        if self._use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            message = f"{color}{message}{self.RESET}"

        return message


class LoggingConfig:
    """Logging configuration manager."""

    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    SIMPLE_FORMAT = "%(levelname)s - %(message)s"

    def __init__(
        self,
        level: LogLevel = "INFO",
        log_dir: str | Path | None = None,
        enable_file_logging: bool = False,
        enable_console: bool = True,
        enable_colors: bool = True,
    ):
        """
        Initialize logging configuration.

        Args:
            level: Default log level.
            log_dir: Directory for log files.
            enable_file_logging: Whether to log to files.
            enable_console: Whether to log to console.
            enable_colors: Whether to use colored output.
        """
        self.level = level
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.enable_file_logging = enable_file_logging
        self.enable_console = enable_console
        self.enable_colors = enable_colors

        self._handlers: list[logging.Handler] = []
        self._configured = False

    def setup(self) -> None:
        """Setup logging configuration."""
        if self._configured:
            return

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.level))

        # Remove existing handlers
        root_logger.handlers.clear()

        # Add console handler
        if self.enable_console:
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)
            self._handlers.append(console_handler)

        # Add file handler
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = self._create_file_handler()
            root_logger.addHandler(file_handler)
            self._handlers.append(file_handler)

        # Suppress noisy third-party loggers
        self._configure_third_party_loggers()

        self._configured = True

    def _create_console_handler(self) -> logging.StreamHandler:
        """Create console handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, self.level))

        if self.enable_colors:
            formatter = ColoredFormatter(
                self.DEFAULT_FORMAT,
                self.DEFAULT_DATE_FORMAT,
            )
        else:
            formatter = logging.Formatter(
                self.DEFAULT_FORMAT,
                self.DEFAULT_DATE_FORMAT,
            )

        handler.setFormatter(formatter)
        return handler

    def _create_file_handler(self) -> logging.FileHandler:
        """Create file handler."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"voice_subtitle_{timestamp}.log"

        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG)  # Log everything to file

        formatter = logging.Formatter(
            self.DEFAULT_FORMAT,
            self.DEFAULT_DATE_FORMAT,
        )
        handler.setFormatter(formatter)

        return handler

    def _configure_third_party_loggers(self) -> None:
        """Configure third-party library loggers."""
        # Suppress verbose loggers
        noisy_loggers = [
            "urllib3",
            "httpx",
            "httpcore",
            "faster_whisper",
            "torch",
            "transformers",
            "llama_cpp",
        ]

        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    def set_level(self, level: LogLevel) -> None:
        """
        Change log level.

        Args:
            level: New log level.
        """
        self.level = level
        log_level = getattr(logging, level)

        logging.getLogger().setLevel(log_level)
        for handler in self._handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(log_level)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.

        Args:
            name: Logger name.

        Returns:
            Logger instance.
        """
        return logging.getLogger(name)

    def cleanup(self) -> None:
        """Cleanup logging handlers."""
        root_logger = logging.getLogger()

        for handler in self._handlers:
            handler.close()
            root_logger.removeHandler(handler)

        self._handlers.clear()
        self._configured = False


# Global logging config instance
_logging_config: LoggingConfig | None = None


def setup_logging(
    level: LogLevel = "INFO",
    log_dir: str | Path | None = None,
    enable_file_logging: bool = False,
    enable_console: bool = True,
    enable_colors: bool = True,
) -> LoggingConfig:
    """
    Setup global logging configuration.

    Args:
        level: Log level.
        log_dir: Directory for log files.
        enable_file_logging: Whether to log to files.
        enable_console: Whether to log to console.
        enable_colors: Whether to use colored output.

    Returns:
        LoggingConfig instance.
    """
    global _logging_config

    _logging_config = LoggingConfig(
        level=level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging,
        enable_console=enable_console,
        enable_colors=enable_colors,
    )
    _logging_config.setup()

    return _logging_config


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def set_log_level(level: LogLevel) -> None:
    """
    Set global log level.

    Args:
        level: New log level.
    """
    if _logging_config:
        _logging_config.set_level(level)
    else:
        logging.getLogger().setLevel(getattr(logging, level))


class ProgressLogger:
    """Logger for progress updates."""

    def __init__(self, logger: logging.Logger, total: int, prefix: str = ""):
        """
        Initialize progress logger.

        Args:
            logger: Logger instance.
            total: Total number of items.
            prefix: Prefix for log messages.
        """
        self.logger = logger
        self.total = total
        self.prefix = prefix
        self._current = 0
        self._last_percent = -1

    def update(self, current: int | None = None, message: str = "") -> None:
        """
        Update progress.

        Args:
            current: Current progress value.
            message: Additional message.
        """
        if current is not None:
            self._current = current
        else:
            self._current += 1

        percent = int((self._current / self.total) * 100) if self.total > 0 else 0

        # Only log at 10% intervals to reduce noise
        if percent >= self._last_percent + 10 or self._current >= self.total:
            self._last_percent = (percent // 10) * 10

            if message:
                self.logger.info(
                    f"{self.prefix}Progress: {percent}% ({self._current}/{self.total}) - {message}"
                )
            else:
                self.logger.info(
                    f"{self.prefix}Progress: {percent}% ({self._current}/{self.total})"
                )

    def complete(self, message: str = "Complete") -> None:
        """
        Log completion.

        Args:
            message: Completion message.
        """
        self.logger.info(f"{self.prefix}{message}")
