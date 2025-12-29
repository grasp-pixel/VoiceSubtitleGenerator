# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Voice Subtitle Generator."""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

block_cipher = None

# Project root directory
PROJECT_ROOT = Path(SPECPATH)

# Hidden imports - modules that PyInstaller can't detect automatically
hidden_imports = [
    # PyTorch
    "torch",
    "torch.utils",
    "torch.utils.data",
    "torchaudio",
    "torchaudio.transforms",
    "torchaudio.functional",
    # Whisper/CTranslate2
    "faster_whisper",
    "ctranslate2",
    # LLM
    "llama_cpp",
    # GUI
    "dearpygui",
    "dearpygui.dearpygui",
    # Audio processing
    "numpy",
    "soundfile",
    "pydub",
    "pysubs2",
    # Configuration
    "yaml",
    # Network
    "requests",
    "urllib3",
    "certifi",
    # HuggingFace (for model download)
    "huggingface_hub",
    "huggingface_hub.file_download",
    # FFmpeg
    "static_ffmpeg",
    # Standard library
    "ssl",
    "tempfile",
    "zipfile",
    "faulthandler",
]

# Collect all submodules for complex packages
hidden_imports += collect_submodules("faster_whisper")
hidden_imports += collect_submodules("ctranslate2")

# Data files to include
datas = [
    # Configuration files
    (str(PROJECT_ROOT / "config"), "config"),
    # Fonts directory (empty, fonts downloaded at runtime)
    (str(PROJECT_ROOT / "fonts"), "fonts"),
]

# Collect data files from packages
datas += collect_data_files("dearpygui")
datas += collect_data_files("static_ffmpeg")
datas += collect_data_files("faster_whisper")
datas += collect_data_files("ctranslate2")
datas += collect_data_files("llama_cpp")

# Binary files (DLLs, shared libraries)
binaries = []
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("torchaudio")
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("llama_cpp")

# Modules to exclude
excludes = [
    # macOS only
    "pyobjus",
    # Not needed
    "matplotlib",
    "tkinter",
    "IPython",
    "jupyter",
    # Development only
    "pytest",
    "black",
    "isort",
    "mypy",
    "ruff",
]

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceSubtitleGenerator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application - no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "assets" / "icon.ico")
    if (PROJECT_ROOT / "assets" / "icon.ico").exists()
    else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="VoiceSubtitleGenerator",
)
