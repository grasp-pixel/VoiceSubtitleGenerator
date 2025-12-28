#!/usr/bin/env python
"""Build script for Voice Subtitle Generator."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def clean_build(project_root: Path) -> None:
    """Clean previous build artifacts."""
    print("Cleaning previous build...")

    dirs_to_clean = [
        project_root / "build",
        project_root / "dist",
    ]

    for dir_path in dirs_to_clean:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  Removed: {dir_path}")


def run_pyinstaller(project_root: Path, debug: bool = False) -> bool:
    """Run PyInstaller to build the executable."""
    print("Building with PyInstaller...")

    spec_file = project_root / "voice_subtitle_generator.spec"

    if not spec_file.exists():
        print(f"Error: Spec file not found: {spec_file}")
        return False

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec_file),
        "--clean",
        "--noconfirm",
    ]

    if debug:
        cmd.append("--debug=all")

    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: PyInstaller failed with code {e.returncode}")
        return False


def copy_additional_files(project_root: Path) -> None:
    """Copy additional required files to dist."""
    print("Copying additional files...")

    dist_dir = project_root / "dist" / "VoiceSubtitleGenerator"

    if not dist_dir.exists():
        print("  Warning: dist directory not found")
        return

    # Create models directory placeholder
    models_dir = dist_dir / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / ".gitkeep").touch()
    print(f"  Created: {models_dir}")

    # Create input/output directories
    for dir_name in ["input", "output"]:
        dir_path = dist_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        (dir_path / ".gitkeep").touch()
        print(f"  Created: {dir_path}")

    # Copy README if exists
    readme = project_root / "README.md"
    if readme.exists():
        shutil.copy(readme, dist_dir / "README.md")
        print(f"  Copied: README.md")


def create_portable_zip(project_root: Path) -> Path | None:
    """Create a portable ZIP archive."""
    print("Creating portable ZIP...")

    dist_dir = project_root / "dist" / "VoiceSubtitleGenerator"

    if not dist_dir.exists():
        print("  Error: dist directory not found")
        return None

    zip_name = "VoiceSubtitleGenerator_portable"
    zip_path = project_root / "dist" / zip_name

    shutil.make_archive(str(zip_path), "zip", dist_dir.parent, dist_dir.name)

    final_path = Path(str(zip_path) + ".zip")
    print(f"  Created: {final_path}")

    return final_path


def main():
    """Main build entry point."""
    parser = argparse.ArgumentParser(description="Build Voice Subtitle Generator")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build artifacts before building",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build in debug mode",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip creating portable ZIP",
    )

    args = parser.parse_args()

    project_root = get_project_root()
    print(f"Project root: {project_root}")

    # Clean if requested
    if args.clean:
        clean_build(project_root)

    # Build
    if not run_pyinstaller(project_root, args.debug):
        sys.exit(1)

    # Copy additional files
    copy_additional_files(project_root)

    # Create ZIP
    if not args.no_zip:
        zip_path = create_portable_zip(project_root)
        if zip_path:
            print(f"\nBuild complete!")
            print(f"  Executable: {project_root / 'dist' / 'VoiceSubtitleGenerator'}")
            print(f"  Portable ZIP: {zip_path}")
    else:
        print(f"\nBuild complete!")
        print(f"  Executable: {project_root / 'dist' / 'VoiceSubtitleGenerator'}")


if __name__ == "__main__":
    main()
