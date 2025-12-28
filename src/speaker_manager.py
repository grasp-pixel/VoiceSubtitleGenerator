"""Speaker mapping management for Voice Subtitle Generator."""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from .models import Segment, SpeakerMapping

logger = logging.getLogger(__name__)


class SpeakerManager:
    """
    Manage speaker ID to character name mappings.

    Handles extraction of speakers from segments, mapping application,
    and preset save/load functionality.
    """

    DEFAULT_PRESETS_DIR = "config/speaker_presets"

    def __init__(self, presets_dir: str | None = None):
        """
        Initialize speaker manager.

        Args:
            presets_dir: Directory for storing speaker presets.
        """
        self.presets_dir = Path(presets_dir or self.DEFAULT_PRESETS_DIR)
        self._ensure_presets_dir()

    def _ensure_presets_dir(self) -> None:
        """Create presets directory if it doesn't exist."""
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def extract_speakers(self, segments: list[Segment]) -> list[str]:
        """
        Extract unique speaker IDs from segments.

        Args:
            segments: List of segments to extract speakers from.

        Returns:
            list[str]: Sorted list of unique speaker IDs.
        """
        speakers = set()
        for segment in segments:
            if segment.speaker_id and segment.speaker_id != "UNKNOWN":
                speakers.add(segment.speaker_id)

        return sorted(speakers)

    def create_default_mapping(
        self, speakers: list[str]
    ) -> dict[str, SpeakerMapping]:
        """
        Create default mapping for speakers.

        Args:
            speakers: List of speaker IDs.

        Returns:
            dict: Default mapping with speaker IDs as keys.
        """
        # Default colors for speakers (cycling through)
        default_colors = [
            "FFFFFF",  # White
            "00BFFF",  # Sky blue
            "FF69B4",  # Pink
            "90EE90",  # Light green
            "FFD700",  # Gold
            "DDA0DD",  # Plum
            "87CEEB",  # Light sky blue
            "F0E68C",  # Khaki
            "E6E6FA",  # Lavender
            "FFA07A",  # Light salmon
        ]

        mapping = {}
        for i, speaker_id in enumerate(speakers):
            color = default_colors[i % len(default_colors)]
            mapping[speaker_id] = SpeakerMapping(
                speaker_id=speaker_id,
                name=speaker_id,  # Default to ID
                color=color,
            )

        return mapping

    def apply_mapping(
        self,
        segments: list[Segment],
        mapping: dict[str, SpeakerMapping],
    ) -> None:
        """
        Apply speaker mapping to segments in place.

        Args:
            segments: List of segments to update.
            mapping: Speaker ID to SpeakerMapping dictionary.
        """
        for segment in segments:
            if segment.speaker_id in mapping:
                speaker_map = mapping[segment.speaker_id]
                segment.speaker_name = speaker_map.name

    def apply_mapping_dict(
        self,
        segments: list[Segment],
        mapping: dict[str, str],
    ) -> None:
        """
        Apply simple name mapping to segments.

        Args:
            segments: List of segments to update.
            mapping: Speaker ID to name dictionary.
        """
        for segment in segments:
            if segment.speaker_id in mapping:
                segment.speaker_name = mapping[segment.speaker_id]

    def get_speaker_samples(
        self,
        segments: list[Segment],
    ) -> dict[str, tuple[float, str]]:
        """
        Get sample time and text for each speaker.

        Useful for speaker identification in UI.

        Args:
            segments: List of segments.

        Returns:
            dict: Speaker ID to (start_time, sample_text) mapping.
        """
        samples: dict[str, tuple[float, str]] = {}

        for segment in segments:
            speaker_id = segment.speaker_id
            if speaker_id and speaker_id != "UNKNOWN":
                if speaker_id not in samples:
                    # Use first occurrence
                    samples[speaker_id] = (
                        segment.start,
                        segment.original_text[:50],  # First 50 chars
                    )

        return samples

    def get_speaker_stats(
        self,
        segments: list[Segment],
    ) -> dict[str, dict[str, Any]]:
        """
        Get statistics for each speaker.

        Args:
            segments: List of segments.

        Returns:
            dict: Speaker ID to stats dictionary.
        """
        stats: dict[str, dict[str, Any]] = {}

        for segment in segments:
            speaker_id = segment.speaker_id
            if not speaker_id or speaker_id == "UNKNOWN":
                continue

            if speaker_id not in stats:
                stats[speaker_id] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "first_appearance": segment.start,
                    "last_appearance": segment.end,
                }

            stats[speaker_id]["count"] += 1
            stats[speaker_id]["total_duration"] += segment.duration
            stats[speaker_id]["last_appearance"] = max(
                stats[speaker_id]["last_appearance"], segment.end
            )

        return stats

    # Preset management

    def save_preset(
        self,
        name: str,
        mapping: dict[str, SpeakerMapping],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Save speaker mapping as preset.

        Args:
            name: Preset name (used as filename).
            mapping: Speaker mapping to save.
            metadata: Optional metadata (description, created date, etc.).
        """
        preset_path = self.presets_dir / f"{name}.yaml"

        data = {
            "name": name,
            "metadata": metadata or {},
            "mapping": {
                speaker_id: {
                    "name": m.name,
                    "color": m.color,
                    "style_preset": m.style_preset,
                }
                for speaker_id, m in mapping.items()
            },
        }

        with open(preset_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"Saved speaker preset: {name}")

    def load_preset(self, name: str) -> dict[str, SpeakerMapping]:
        """
        Load speaker mapping from preset.

        Args:
            name: Preset name.

        Returns:
            dict: Speaker ID to SpeakerMapping dictionary.

        Raises:
            FileNotFoundError: If preset doesn't exist.
        """
        preset_path = self.presets_dir / f"{name}.yaml"

        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {name}")

        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        mapping = {}
        for speaker_id, m in data.get("mapping", {}).items():
            mapping[speaker_id] = SpeakerMapping(
                speaker_id=speaker_id,
                name=m.get("name", speaker_id),
                color=m.get("color", "FFFFFF"),
                style_preset=m.get("style_preset"),
            )

        logger.info(f"Loaded speaker preset: {name}")
        return mapping

    def list_presets(self) -> list[str]:
        """
        List available preset names.

        Returns:
            list[str]: List of preset names (without extension).
        """
        presets = []
        for path in self.presets_dir.glob("*.yaml"):
            presets.append(path.stem)
        return sorted(presets)

    def delete_preset(self, name: str) -> None:
        """
        Delete a preset.

        Args:
            name: Preset name to delete.

        Raises:
            FileNotFoundError: If preset doesn't exist.
        """
        preset_path = self.presets_dir / f"{name}.yaml"

        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {name}")

        preset_path.unlink()
        logger.info(f"Deleted speaker preset: {name}")

    def get_preset_info(self, name: str) -> dict[str, Any]:
        """
        Get preset metadata.

        Args:
            name: Preset name.

        Returns:
            dict: Preset metadata including speaker count.
        """
        preset_path = self.presets_dir / f"{name}.yaml"

        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {name}")

        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return {
            "name": data.get("name", name),
            "metadata": data.get("metadata", {}),
            "speaker_count": len(data.get("mapping", {})),
            "speakers": list(data.get("mapping", {}).keys()),
        }

    # Import/Export

    def export_mapping_json(
        self,
        mapping: dict[str, SpeakerMapping],
        output_path: str,
    ) -> None:
        """
        Export mapping to JSON file.

        Args:
            mapping: Mapping to export.
            output_path: Output file path.
        """
        data = {
            speaker_id: {
                "name": m.name,
                "color": m.color,
                "style_preset": m.style_preset,
            }
            for speaker_id, m in mapping.items()
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def import_mapping_json(self, input_path: str) -> dict[str, SpeakerMapping]:
        """
        Import mapping from JSON file.

        Args:
            input_path: Input file path.

        Returns:
            dict: Imported speaker mapping.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mapping = {}
        for speaker_id, m in data.items():
            mapping[speaker_id] = SpeakerMapping(
                speaker_id=speaker_id,
                name=m.get("name", speaker_id),
                color=m.get("color", "FFFFFF"),
                style_preset=m.get("style_preset"),
            )

        return mapping
