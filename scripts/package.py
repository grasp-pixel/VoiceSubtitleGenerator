#!/usr/bin/env python
"""Package script for creating distribution ZIP."""

import argparse
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def get_version() -> str:
    """Extract version from pyproject.toml."""
    pyproject = get_project_root() / "pyproject.toml"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.startswith("version"):
            # version = "0.1.0"
            return line.split('"')[1]
    return "unknown"


# Files and directories to include in distribution
INCLUDE = [
    "src",
    "ui",
    "config",
    "fonts",
    "main.py",
    "run.bat",
    "build_cuda.bat",
    "build_cuda.ps1",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
]

# Patterns to exclude (even if inside included directories)
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".gitkeep",
    ".DS_Store",
    "Thumbs.db",
]


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            if path.name.endswith(pattern[1:]):
                return True
        elif pattern in path.parts:
            return True
        elif path.name == pattern:
            return True
    return False


def create_distribution(project_root: Path, output_dir: Path) -> Path:
    """Create distribution ZIP file."""
    version = get_version()
    timestamp = datetime.now().strftime("%Y%m%d")
    zip_name = f"VoiceSubtitleGenerator_v{version}_{timestamp}.zip"
    zip_path = output_dir / zip_name

    print(f"Creating distribution: {zip_name}")
    print(f"Version: {version}")
    print()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create ZIP file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item_name in INCLUDE:
            item_path = project_root / item_name

            if not item_path.exists():
                print(f"  [SKIP] {item_name} (not found)")
                continue

            if item_path.is_file():
                # Single file
                arcname = f"VoiceSubtitleGenerator/{item_name}"
                zf.write(item_path, arcname)
                print(f"  [FILE] {item_name}")

            elif item_path.is_dir():
                # Directory - walk and add files
                file_count = 0
                for file_path in item_path.rglob("*"):
                    if file_path.is_file() and not should_exclude(file_path):
                        rel_path = file_path.relative_to(project_root)
                        arcname = f"VoiceSubtitleGenerator/{rel_path}"
                        zf.write(file_path, arcname)
                        file_count += 1
                print(f"  [DIR]  {item_name}/ ({file_count} files)")

        # Create empty placeholder directories
        for dir_name in ["models", "input", "output"]:
            # Add a readme to keep directory structure
            readme_content = f"# {dir_name.title()} Directory\n\nPlace your {dir_name} files here.\n"
            arcname = f"VoiceSubtitleGenerator/{dir_name}/README.txt"
            zf.writestr(arcname, readme_content)
            print(f"  [DIR]  {dir_name}/ (placeholder)")

    # Get file size
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print()
    print(f"Created: {zip_path}")
    print(f"Size: {size_mb:.2f} MB")

    return zip_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create distribution ZIP")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: project_root/dist)",
    )

    args = parser.parse_args()

    project_root = get_project_root()
    output_dir = args.output or (project_root / "dist")

    print("=" * 50)
    print("  Voice Subtitle Generator - Package Script")
    print("=" * 50)
    print()

    zip_path = create_distribution(project_root, output_dir)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
