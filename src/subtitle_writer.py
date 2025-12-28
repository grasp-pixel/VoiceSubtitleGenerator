"""Subtitle file generation for Voice Subtitle Generator."""

import logging
from pathlib import Path
from typing import Any

import pysubs2
from pysubs2 import SSAFile, SSAStyle, SSAEvent

from .ass_styler import ASSStyler, StylePreset
from .config import SubtitleConfig
from .models import Segment, SubtitleStyle

logger = logging.getLogger(__name__)


class SubtitleWriterError(Exception):
    """Raised when subtitle writing fails."""

    pass


class SubtitleWriter:
    """
    Generate subtitle files in various formats.

    Supports SRT, ASS, and VTT formats.
    """

    def __init__(
        self,
        include_original: bool = False,
        video_width: int = 1920,
        video_height: int = 1080,
        default_font: str = "Malgun Gothic",
        default_size: int = 48,
        original_font: str = "Malgun Gothic",
        original_size: int = 36,
        style_preset: StylePreset | None = None,
    ):
        """
        Initialize subtitle writer.

        Args:
            include_original: Include original Japanese text.
            video_width: Video width for ASS.
            video_height: Video height for ASS.
            default_font: Default font for translated text.
            default_size: Default font size for translated text.
            original_font: Font for original text.
            original_size: Font size for original text.
            style_preset: Optional style preset for ASS styling.
        """
        self.include_original = include_original
        self.video_width = video_width
        self.video_height = video_height
        self.default_font = default_font
        self.default_size = default_size
        self.original_font = original_font
        self.original_size = original_size
        self.style_preset = style_preset
        self._ass_styler: ASSStyler | None = None

    @classmethod
    def from_config(cls, config: SubtitleConfig) -> "SubtitleWriter":
        """Create writer from config."""
        return cls(
            include_original=config.include_original,
            video_width=config.ass.video_width,
            video_height=config.ass.video_height,
            default_font=config.ass.default_font,
            default_size=config.ass.default_size,
            original_font=config.ass.original_font,
            original_size=config.ass.original_size,
        )

    def load_style_preset(self, preset_path: str | Path) -> None:
        """
        Load a style preset from file.

        Args:
            preset_path: Path to preset YAML file.
        """
        self.style_preset = StylePreset.from_yaml(preset_path)
        self._ass_styler = None  # Reset styler to reload

    def _get_ass_styler(self) -> ASSStyler:
        """Get or create ASS styler instance."""
        if self._ass_styler is None:
            self._ass_styler = ASSStyler(
                preset=self.style_preset,
                video_width=self.video_width,
                video_height=self.video_height,
            )
        return self._ass_styler

    def write(
        self,
        segments: list[Segment],
        output_path: str,
        format: str = "srt",
        styles: dict[str, SubtitleStyle] | None = None,
    ) -> None:
        """
        Write subtitles to file.

        Args:
            segments: List of segments to write.
            output_path: Output file path.
            format: Output format (srt, ass, vtt).
            styles: Named styles for ASS format.

        Raises:
            SubtitleWriterError: If writing fails.
        """
        format = format.lower()

        try:
            if format == "srt":
                self.write_srt(segments, output_path)
            elif format == "ass":
                self.write_ass(segments, output_path, styles)
            elif format == "vtt":
                self.write_vtt(segments, output_path)
            else:
                raise SubtitleWriterError(f"Unsupported format: {format}")

            logger.info(f"Saved subtitle: {output_path}")

        except Exception as e:
            raise SubtitleWriterError(f"Failed to write subtitle: {e}") from e

    def write_srt(
        self,
        segments: list[Segment],
        output_path: str,
    ) -> None:
        """
        Write SRT format subtitles.

        Args:
            segments: List of segments.
            output_path: Output file path.
        """
        subs = SSAFile()

        for segment in segments:
            text = self._build_text(segment)

            event = SSAEvent(
                start=int(segment.start * 1000),  # pysubs2 uses milliseconds
                end=int(segment.end * 1000),
                text=text,
            )
            subs.append(event)

        subs.save(output_path, format_="srt")

    def write_ass(
        self,
        segments: list[Segment],
        output_path: str,
        styles: dict[str, SubtitleStyle] | None = None,
    ) -> None:
        """
        Write ASS format subtitles with styling.

        Args:
            segments: List of segments.
            output_path: Output file path.
            styles: Named styles for styling.
        """
        # Use advanced styler if preset is available
        if self.style_preset is not None:
            self.write_styled_ass(segments, output_path)
            return

        subs = SSAFile()

        # Set video resolution
        subs.info["PlayResX"] = str(self.video_width)
        subs.info["PlayResY"] = str(self.video_height)

        # Create default style
        default_style = self._create_default_ass_style()
        subs.styles["Default"] = default_style

        # Add events
        for segment in segments:
            text = self._build_text(segment, for_ass=True)

            event = SSAEvent(
                start=int(segment.start * 1000),
                end=int(segment.end * 1000),
                text=text,
                style="Default",
            )
            subs.append(event)

        subs.save(output_path, format_="ass")

    def write_styled_ass(
        self,
        segments: list[Segment],
        output_path: str,
    ) -> None:
        """
        Write ASS with advanced styling using ASSStyler.

        Args:
            segments: List of segments.
            output_path: Output file path.
        """
        styler = self._get_ass_styler()

        # Get filename for title
        title = Path(output_path).stem

        # Generate complete ASS content
        ass_content = styler.generate_ass(
            segments=segments,
            title=title,
        )

        # Write to file
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(ass_content)

        logger.info(f"Wrote styled ASS: {output_path}")

    def write_vtt(
        self,
        segments: list[Segment],
        output_path: str,
    ) -> None:
        """
        Write WebVTT format subtitles.

        Args:
            segments: List of segments.
            output_path: Output file path.
        """
        subs = SSAFile()

        for segment in segments:
            text = self._build_text(segment)

            event = SSAEvent(
                start=int(segment.start * 1000),
                end=int(segment.end * 1000),
                text=text,
            )
            subs.append(event)

        subs.save(output_path, format_="vtt")

    def _build_text(
        self,
        segment: Segment,
        for_ass: bool = False,
    ) -> str:
        """
        Build subtitle text from segment.

        Args:
            segment: Segment to build text from.
            for_ass: Whether output is for ASS format.

        Returns:
            str: Formatted subtitle text.
        """
        parts = []

        # Add original text if enabled (wrapped in 『』 for clear separation)
        if self.include_original and segment.original_text:
            if for_ass:
                # Apply original text styling with ASS override tags
                # {\fn<font>\fs<size>\c&HBBGGRR&} for gray color
                orig_style = (
                    f"{{\\fn{self.original_font}"
                    f"\\fs{self.original_size}"
                    f"\\c&H888888&}}"  # Gray color
                )
                # Reset to default style after original text
                reset_style = (
                    f"{{\\fn{self.default_font}"
                    f"\\fs{self.default_size}"
                    f"\\c&HFFFFFF&}}"  # White color
                )
                parts.append(f"{orig_style}『{segment.original_text}』{reset_style}")
                parts.append("\\N")  # ASS newline
            else:
                parts.append(f"『{segment.original_text}』")
                parts.append("\n")

        # Add translated text
        if segment.translated_text:
            parts.append(segment.translated_text)
        elif segment.original_text:
            # Fallback to original if no translation
            parts.append(segment.original_text)

        text = " ".join(parts) if not self.include_original else "".join(parts)

        # Clean up for ASS
        if for_ass:
            text = text.replace("\n", "\\N")

        return text.strip()

    def _create_default_ass_style(self) -> SSAStyle:
        """Create default ASS style."""
        return SSAStyle(
            fontname=self.default_font,
            fontsize=self.default_size,
            primarycolor=pysubs2.Color(255, 255, 255, 0),  # White
            outlinecolor=pysubs2.Color(0, 0, 0, 0),  # Black
            backcolor=pysubs2.Color(0, 0, 0, 128),  # Semi-transparent black
            outline=2.0,
            shadow=1.0,
            alignment=2,  # Bottom center
            marginv=30,
            marginl=20,
            marginr=20,
        )

    def preview(
        self,
        segments: list[Segment],
        max_segments: int = 10,
    ) -> str:
        """
        Generate preview text of subtitles.

        Args:
            segments: Segments to preview.
            max_segments: Maximum segments to include.

        Returns:
            str: Preview text in SRT-like format.
        """
        lines = []

        for i, segment in enumerate(segments[:max_segments], 1):
            lines.append(f"{i}")
            lines.append(segment.to_srt_timing())
            lines.append(self._build_text(segment))
            lines.append("")

        if len(segments) > max_segments:
            lines.append(f"... and {len(segments) - max_segments} more segments")

        return "\n".join(lines)

    @staticmethod
    def supported_formats() -> list[str]:
        """Get list of supported output formats."""
        return ["srt", "ass", "vtt"]
