"""PyTorch 2.6+ compatibility patches.

This module MUST be imported before any other modules that use torch.load,
especially before pyannote and faster-whisper imports.
"""

import logging
import warnings

logger = logging.getLogger(__name__)


def apply_patches() -> None:
    """Apply all compatibility patches."""
    _register_safe_globals()
    _patch_torch_load()
    _filter_warnings()


def _register_safe_globals() -> None:
    """Register omegaconf and other necessary classes as safe globals for PyTorch 2.6+."""
    import torch

    # Check if add_safe_globals is available (PyTorch 2.4+)
    if not hasattr(torch.serialization, "add_safe_globals"):
        return

    try:
        from omegaconf import DictConfig, ListConfig

        torch.serialization.add_safe_globals([ListConfig, DictConfig])
        logger.info("Registered omegaconf classes as safe globals")
    except ImportError:
        pass  # omegaconf not installed


def _patch_torch_load() -> None:
    """
    Patch torch.load for PyTorch 2.6+ compatibility.

    PyTorch 2.6 defaults to weights_only=True which breaks loading
    models that use custom classes like omegaconf.ListConfig.
    """
    import torch
    import torch.serialization

    # Check if already patched
    if getattr(torch.serialization, "_patched_for_compat", False):
        return

    logger.info(f"Patching torch.load for PyTorch {torch.__version__} compatibility")

    # Patch torch.serialization.load (the underlying function)
    # This ensures all torch.load calls are affected, even if libraries
    # cached the reference before our patch.
    original_serialization_load = torch.serialization.load

    def patched_serialization_load(*args, **kwargs):
        # Force weights_only=False even if explicitly set to True
        kwargs["weights_only"] = False
        return original_serialization_load(*args, **kwargs)

    torch.serialization.load = patched_serialization_load
    torch.serialization._patched_for_compat = True

    # Also patch torch.load for direct calls
    original_load = torch.load

    def patched_load(*args, **kwargs):
        # Force weights_only=False even if explicitly set to True
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load

    logger.info("torch.serialization.load patched successfully")


def _filter_warnings() -> None:
    """Filter deprecated warnings from external libraries."""
    # torchaudio backend warnings
    warnings.filterwarnings(
        "ignore",
        message=".*torchaudio._backend.set_audio_backend.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*The 'torchaudio' backend.*",
        category=UserWarning,
    )
    # Other common warnings
    warnings.filterwarnings(
        "ignore",
        message=".*torch.load.*weights_only.*",
        category=FutureWarning,
    )
    # torchaudio 2.9 deprecation warnings
    warnings.filterwarnings(
        "ignore",
        message=".*torchaudio._backend.*has been deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*this function's implementation will be changed.*torchcodec.*",
        category=UserWarning,
    )
    # pyannote reproducibility warning (custom warning class)
    warnings.filterwarnings(
        "ignore",
        message=".*TensorFloat-32.*",
    )
    # std() degrees of freedom warning (short audio segments)
    warnings.filterwarnings(
        "ignore",
        message=".*std\\(\\): degrees of freedom is <= 0.*",
        category=UserWarning,
    )


# Auto-apply when imported
apply_patches()
