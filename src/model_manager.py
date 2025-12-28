"""Model management for Voice Subtitle Generator."""

import logging
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import TRANSLATION_MODEL_PRESETS, TranslationModelPreset

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model."""

    name: str
    description: str
    path: str
    size_mb: int
    download_url: str | None = None
    is_required: bool = True
    model_type: str = "translation"  # "translation" or "whisper"


@dataclass
class ModelStatus:
    """Status of a model."""

    info: ModelInfo
    exists: bool
    size_on_disk: int = 0


# Whisper model sizes and approximate VRAM requirements
WHISPER_MODELS = {
    "tiny": ModelInfo(
        name="Whisper Tiny",
        description="가장 빠름, 정확도 낮음 (~1GB VRAM)",
        path="",
        size_mb=75,
        model_type="whisper",
    ),
    "base": ModelInfo(
        name="Whisper Base",
        description="빠름, 정확도 보통 (~1GB VRAM)",
        path="",
        size_mb=145,
        model_type="whisper",
    ),
    "small": ModelInfo(
        name="Whisper Small",
        description="균형잡힌 속도/정확도 (~2GB VRAM)",
        path="",
        size_mb=465,
        model_type="whisper",
    ),
    "medium": ModelInfo(
        name="Whisper Medium",
        description="높은 정확도 (~5GB VRAM)",
        path="",
        size_mb=1500,
        model_type="whisper",
    ),
    "large-v2": ModelInfo(
        name="Whisper Large-v2",
        description="매우 높은 정확도 (~10GB VRAM)",
        path="",
        size_mb=3000,
        model_type="whisper",
    ),
    "large-v3": ModelInfo(
        name="Whisper Large-v3",
        description="최신 대형 모델 (~10GB VRAM)",
        path="",
        size_mb=3000,
        model_type="whisper",
    ),
    "large-v3-turbo": ModelInfo(
        name="Whisper Large-v3 Turbo",
        description="대형 모델 + 빠른 속도 (~6GB VRAM, 권장)",
        path="",
        size_mb=1600,
        model_type="whisper",
    ),
}

# Translation models are now defined in config.py as TRANSLATION_MODEL_PRESETS
# This provides a unified source for both settings and model management


class ModelManager:
    """Manages ML models for the application."""

    def __init__(self, base_path: str | Path | None = None):
        """
        Initialize model manager.

        Args:
            base_path: Base path for models. Defaults to current directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def check_model(self, model_path: str) -> ModelStatus:
        """
        Check if a model exists.

        Args:
            model_path: Path to the model file (relative or absolute).

        Returns:
            ModelStatus with existence info.
        """
        path = Path(model_path)
        if not path.is_absolute():
            path = self.base_path / path

        # Try to find in presets by filename
        filename = path.name.lower()
        preset_info = None
        for key, preset in TRANSLATION_MODEL_PRESETS.items():
            if preset.filename.lower() == filename:
                preset_info = preset
                break

        if preset_info:
            info = ModelInfo(
                name=preset_info.name,
                description=f"{preset_info.description} (VRAM {preset_info.vram_gb:.0f}GB)",
                path=str(model_path),
                size_mb=preset_info.size_mb,
                download_url=preset_info.download_url,
                model_type="translation",
            )
        else:
            info = ModelInfo(
                name=path.name,
                description="사용자 지정 모델",
                path=str(model_path),
                size_mb=0,
            )

        exists = path.exists()
        size_on_disk = path.stat().st_size if exists else 0

        return ModelStatus(info=info, exists=exists, size_on_disk=size_on_disk)

    def check_translation_model(self, model_path: str) -> ModelStatus:
        """Check translation model status."""
        return self.check_model(model_path)

    def check_whisper_model(self, model_size: str) -> dict:
        """
        Check Whisper model status.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v2, large-v3).

        Returns:
            dict with status info.
        """
        # faster-whisper uses HuggingFace cache
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        # Model name mapping for faster-whisper (CTranslate2 format)
        # Note: large-v3-turbo is hosted by mobiuslabsgmbh, not Systran
        model_map = {
            "tiny": "Systran/faster-whisper-tiny",
            "base": "Systran/faster-whisper-base",
            "small": "Systran/faster-whisper-small",
            "medium": "Systran/faster-whisper-medium",
            "large-v2": "Systran/faster-whisper-large-v2",
            "large-v3": "Systran/faster-whisper-large-v3",
            "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        }

        model_name = model_map.get(model_size, f"Systran/faster-whisper-{model_size}")
        safe_model_name = model_name.replace("/", "--")

        # Check if model exists in cache
        model_dirs = list(cache_dir.glob(f"models--{safe_model_name}*"))
        cached = len(model_dirs) > 0

        # Get model info
        info = WHISPER_MODELS.get(model_size)

        return {
            "model_size": model_size,
            "model_name": model_name,
            "cached": cached,
            "info": info,
        }

    def download_whisper_model(
        self,
        model_size: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> bool:
        """
        Download Whisper model using huggingface_hub.

        Args:
            model_size: Model size to download.
            progress_callback: Progress callback.

        Returns:
            True if download succeeded.
        """
        try:
            if progress_callback:
                progress_callback(0.0, f"Whisper {model_size} 다운로드 준비 중...")

            from huggingface_hub import snapshot_download

            # Model name mapping
            # Note: large-v3-turbo is hosted by mobiuslabsgmbh, not Systran
            model_map = {
                "tiny": "Systran/faster-whisper-tiny",
                "base": "Systran/faster-whisper-base",
                "small": "Systran/faster-whisper-small",
                "medium": "Systran/faster-whisper-medium",
                "large-v2": "Systran/faster-whisper-large-v2",
                "large-v3": "Systran/faster-whisper-large-v3",
                "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            }

            repo_id = model_map.get(model_size)
            if not repo_id:
                if progress_callback:
                    progress_callback(0.0, f"알 수 없는 모델: {model_size}")
                return False

            if progress_callback:
                progress_callback(0.1, f"{repo_id} 다운로드 중...")

            # Download the model
            snapshot_download(repo_id=repo_id)

            if progress_callback:
                progress_callback(1.0, "다운로드 완료!")

            logger.info(f"Downloaded Whisper model: {model_size}")
            return True

        except Exception as e:
            logger.error(f"Failed to download Whisper model: {e}")
            if progress_callback:
                progress_callback(0.0, f"다운로드 실패: {e}")
            return False

    def download_model(
        self,
        model_info: ModelInfo,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> bool:
        """
        Download a translation model.

        Args:
            model_info: Model information with download URL.
            progress_callback: Callback for progress (0.0-1.0, status message).

        Returns:
            True if download succeeded.
        """
        if not model_info.download_url:
            logger.error(f"No download URL for model: {model_info.name}")
            return False

        target_path = self.base_path / model_info.path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = target_path.with_suffix(".download")

        try:
            if progress_callback:
                progress_callback(0.0, f"다운로드 시작: {model_info.name}")

            # Create SSL context that doesn't verify certificates (for some networks)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Create request with User-Agent header
            request = urllib.request.Request(
                model_info.download_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )

            # Download with progress
            with urllib.request.urlopen(request, context=ssl_context) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 1024 * 1024  # 1MB chunks

                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0 and progress_callback:
                            progress = downloaded / total_size
                            downloaded_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            progress_callback(
                                progress,
                                f"다운로드 중: {downloaded_mb:.0f}/{total_mb:.0f} MB",
                            )

            # Rename to final path
            if target_path.exists():
                target_path.unlink()
            temp_path.rename(target_path)

            if progress_callback:
                progress_callback(1.0, "다운로드 완료!")

            logger.info(f"Downloaded model: {model_info.name} -> {target_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            if temp_path.exists():
                temp_path.unlink()
            if progress_callback:
                progress_callback(0.0, f"다운로드 실패: {e}")
            return False

    def get_available_translation_models(self) -> list[ModelInfo]:
        """Get list of available translation models for download."""
        models = []
        for key, preset in TRANSLATION_MODEL_PRESETS.items():
            if key == "custom":  # Skip custom preset
                continue
            models.append(ModelInfo(
                name=preset.name,
                description=f"{preset.description} (VRAM {preset.vram_gb:.0f}GB)",
                path=f"models/{preset.filename}",
                size_mb=preset.size_mb,
                download_url=preset.download_url,
                model_type="translation",
            ))
        return models

    def get_translation_presets(self) -> dict[str, TranslationModelPreset]:
        """Get translation model presets (excluding custom)."""
        return {k: v for k, v in TRANSLATION_MODEL_PRESETS.items() if k != "custom"}

    def check_preset_model(self, preset_key: str) -> tuple[bool, int]:
        """
        Check if a preset's model file exists.

        Args:
            preset_key: Preset key (e.g., "qwen2.5-14b").

        Returns:
            tuple: (exists, size_on_disk)
        """
        if preset_key not in TRANSLATION_MODEL_PRESETS or preset_key == "custom":
            return False, 0

        preset = TRANSLATION_MODEL_PRESETS[preset_key]
        path = self.base_path / "models" / preset.filename

        if path.exists():
            return True, path.stat().st_size
        return False, 0

    def download_preset_model(
        self,
        preset_key: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> bool:
        """
        Download a translation model by preset key.

        Args:
            preset_key: Preset key (e.g., "qwen2.5-14b").
            progress_callback: Progress callback.

        Returns:
            True if download succeeded.
        """
        if preset_key not in TRANSLATION_MODEL_PRESETS or preset_key == "custom":
            if progress_callback:
                progress_callback(0.0, f"알 수 없는 프리셋: {preset_key}")
            return False

        preset = TRANSLATION_MODEL_PRESETS[preset_key]
        model_info = ModelInfo(
            name=preset.name,
            description=preset.description,
            path=f"models/{preset.filename}",
            size_mb=preset.size_mb,
            download_url=preset.download_url,
            model_type="translation",
        )

        return self.download_model(model_info, progress_callback)

    def get_available_whisper_models(self) -> list[tuple[str, ModelInfo]]:
        """Get list of available Whisper models."""
        return list(WHISPER_MODELS.items())
