"""Utility functions for Voice Subtitle Generator."""

import sys
from pathlib import Path


def get_app_path() -> Path:
    """Get application root path.

    Works correctly both in normal Python execution and when bundled
    with PyInstaller.

    Returns:
        Path to application root directory.
    """
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle
        return Path(sys.executable).parent
    else:
        # Running as normal Python script
        return Path(__file__).parent.parent


def get_resource_path(relative_path: str) -> Path:
    """Get path to a resource file.

    Args:
        relative_path: Path relative to application root.

    Returns:
        Absolute path to the resource.
    """
    return get_app_path() / relative_path
