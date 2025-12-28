"""ASS subtitle styling module for Voice Subtitle Generator."""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import yaml

from .models import Segment


class TextAlignment(Enum):
    """ASS text alignment values."""

    BOTTOM_LEFT = 1
    BOTTOM_CENTER = 2
    BOTTOM_RIGHT = 3
    MIDDLE_LEFT = 4
    MIDDLE_CENTER = 5
    MIDDLE_RIGHT = 6
    TOP_LEFT = 7
    TOP_CENTER = 8
    TOP_RIGHT = 9


@dataclass
class ASSColor:
    """ASS color representation (AABBGGRR format)."""

    red: int = 255
    green: int = 255
    blue: int = 255
    alpha: int = 0  # 0 = opaque, 255 = transparent

    @classmethod
    def from_hex(cls, hex_color: str, alpha: int = 0) -> "ASSColor":
        """
        Create color from hex string.

        Args:
            hex_color: Hex color string (e.g., "FF69B4" or "#FF69B4").
            alpha: Alpha value (0-255).

        Returns:
            ASSColor instance.
        """
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return cls(red=r, green=g, blue=b, alpha=alpha)
        return cls()

    def to_ass(self) -> str:
        """
        Convert to ASS color format (&HAABBGGRR).

        Returns:
            ASS color string.
        """
        return f"&H{self.alpha:02X}{self.blue:02X}{self.green:02X}{self.red:02X}"

    def to_hex(self) -> str:
        """
        Convert to hex color format.

        Returns:
            Hex color string.
        """
        return f"{self.red:02X}{self.green:02X}{self.blue:02X}"


@dataclass
class ASSStyle:
    """ASS subtitle style definition."""

    name: str = "Default"
    fontname: str = "Malgun Gothic"
    fontsize: int = 48
    primary_color: ASSColor = field(default_factory=ASSColor)
    secondary_color: ASSColor = field(default_factory=ASSColor)
    outline_color: ASSColor = field(default_factory=lambda: ASSColor(0, 0, 0))
    back_color: ASSColor = field(default_factory=lambda: ASSColor(0, 0, 0, 128))
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    scale_x: int = 100
    scale_y: int = 100
    spacing: int = 0
    angle: float = 0.0
    border_style: int = 1  # 1 = outline + shadow, 3 = opaque box
    outline: float = 2.0
    shadow: float = 1.0
    alignment: TextAlignment = TextAlignment.BOTTOM_CENTER
    margin_l: int = 10
    margin_r: int = 10
    margin_v: int = 20
    encoding: int = 1  # 1 = default

    def to_ass_line(self) -> str:
        """
        Convert to ASS style line.

        Returns:
            ASS style definition line.
        """
        return (
            f"Style: {self.name},{self.fontname},{self.fontsize},"
            f"{self.primary_color.to_ass()},{self.secondary_color.to_ass()},"
            f"{self.outline_color.to_ass()},{self.back_color.to_ass()},"
            f"{int(self.bold)},{int(self.italic)},{int(self.underline)},{int(self.strikeout)},"
            f"{self.scale_x},{self.scale_y},{self.spacing},{self.angle:.1f},"
            f"{self.border_style},{self.outline:.1f},{self.shadow:.1f},"
            f"{self.alignment.value},{self.margin_l},{self.margin_r},{self.margin_v},{self.encoding}"
        )


@dataclass
class KeywordStyle:
    """Keyword-based style modifier."""

    pattern: str  # Regex pattern
    bold: bool = False
    italic: bool = False
    color: str | None = None
    size_modifier: float = 1.0

    def apply_to_text(self, text: str, base_color: str = "FFFFFF") -> str:
        """
        Apply inline ASS tags to matching text.

        Args:
            text: Input text.
            base_color: Base color for the text.

        Returns:
            Text with ASS tags applied.
        """
        def replace_match(match: re.Match) -> str:
            matched = match.group(0)
            tags = []

            if self.bold:
                tags.append("\\b1")
            if self.italic:
                tags.append("\\i1")
            if self.color:
                color = ASSColor.from_hex(self.color)
                tags.append(f"\\c{color.to_ass()}")
            if self.size_modifier != 1.0:
                tags.append(f"\\fscx{int(self.size_modifier * 100)}\\fscy{int(self.size_modifier * 100)}")

            if tags:
                reset_tags = []
                if self.bold:
                    reset_tags.append("\\b0")
                if self.italic:
                    reset_tags.append("\\i0")
                if self.color:
                    reset_color = ASSColor.from_hex(base_color)
                    reset_tags.append(f"\\c{reset_color.to_ass()}")
                if self.size_modifier != 1.0:
                    reset_tags.append("\\fscx100\\fscy100")

                return f"{{{''.join(tags)}}}{matched}{{{''.join(reset_tags)}}}"
            return matched

        return re.sub(self.pattern, replace_match, text)


@dataclass
class StylePreset:
    """Complete style preset configuration."""

    name: str
    description: str = ""
    base: ASSStyle = field(default_factory=ASSStyle)
    keywords: list[KeywordStyle] = field(default_factory=list)
    video_width: int = 1920
    video_height: int = 1080

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "StylePreset":
        """
        Load preset from YAML file.

        Args:
            yaml_path: Path to YAML file.

        Returns:
            StylePreset instance.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        preset = cls(
            name=data.get("name", "Custom"),
            description=data.get("description", ""),
            video_width=data.get("video_width", 1920),
            video_height=data.get("video_height", 1080),
        )

        # Load base style
        if "base" in data:
            base = data["base"]
            preset.base = ASSStyle(
                fontname=base.get("font", "Malgun Gothic"),
                fontsize=base.get("size", 48),
                outline=base.get("outline", 2.0),
                shadow=base.get("shadow", 1.0),
            )

        # Load keyword styles
        if "keywords" in data:
            for kw_data in data["keywords"]:
                preset.keywords.append(
                    KeywordStyle(
                        pattern=kw_data.get("pattern", ""),
                        bold=kw_data.get("bold", False),
                        italic=kw_data.get("italic", False),
                        color=kw_data.get("color"),
                        size_modifier=kw_data.get("size_modifier", 1.0),
                    )
                )

        return preset

    def to_yaml(self, yaml_path: str | Path) -> None:
        """
        Save preset to YAML file.

        Args:
            yaml_path: Path to YAML file.
        """
        data = {
            "name": self.name,
            "description": self.description,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "base": {
                "font": self.base.fontname,
                "size": self.base.fontsize,
                "outline": self.base.outline,
                "shadow": self.base.shadow,
            },
            "keywords": [],
        }

        for kw in self.keywords:
            data["keywords"].append({
                "pattern": kw.pattern,
                "bold": kw.bold,
                "italic": kw.italic,
                "color": kw.color,
                "size_modifier": kw.size_modifier,
            })

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


class ASSStyler:
    """
    ASS subtitle styler.

    Applies styles to segments and generates styled ASS output.
    """

    def __init__(
        self,
        preset: StylePreset | None = None,
        video_width: int = 1920,
        video_height: int = 1080,
    ):
        """
        Initialize ASS styler.

        Args:
            preset: Style preset to use.
            video_width: Video width for PlayResX.
            video_height: Video height for PlayResY.
        """
        self.preset = preset or StylePreset(name="Default")
        self.video_width = video_width
        self.video_height = video_height

    def load_preset(self, preset_path: str | Path) -> None:
        """
        Load a style preset from file.

        Args:
            preset_path: Path to preset YAML file.
        """
        self.preset = StylePreset.from_yaml(preset_path)

    def apply_keyword_styles(self, text: str) -> str:
        """
        Apply keyword-based inline styles to text.

        Args:
            text: Input text.

        Returns:
            Text with ASS inline tags.
        """
        if not self.preset.keywords:
            return text

        base_color = "FFFFFF"

        result = text
        for kw_style in self.preset.keywords:
            result = kw_style.apply_to_text(result, base_color)

        return result

    def style_segment(self, segment: Segment) -> tuple[str, str]:
        """
        Style a segment for ASS output.

        Args:
            segment: Segment to style.

        Returns:
            Tuple of (style_name, styled_text).
        """
        # Apply keyword styles
        text = segment.translated_text or segment.original_text
        styled_text = self.apply_keyword_styles(text)

        return self.preset.base.name, styled_text

    def generate_styles_block(self) -> str:
        """
        Generate ASS [V4+ Styles] section.

        Returns:
            ASS styles block.
        """
        lines = [
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
        ]

        # Add default style
        lines.append(self.preset.base.to_ass_line())

        return "\n".join(lines)

    def generate_script_info(self, title: str = "Subtitles") -> str:
        """
        Generate ASS [Script Info] section.

        Args:
            title: Subtitle title.

        Returns:
            ASS script info block.
        """
        return f"""[Script Info]
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {self.video_width}
PlayResY: {self.video_height}
"""

    def format_time(self, seconds: float) -> str:
        """
        Format time for ASS.

        Args:
            seconds: Time in seconds.

        Returns:
            ASS time format (H:MM:SS.cc).
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds * 100) % 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def generate_events(self, segments: list[Segment]) -> str:
        """
        Generate ASS [Events] section.

        Args:
            segments: List of segments.

        Returns:
            ASS events block.
        """
        lines = [
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]

        for segment in segments:
            style_name, styled_text = self.style_segment(segment)

            start = self.format_time(segment.start)
            end = self.format_time(segment.end)

            line = f"Dialogue: 0,{start},{end},{style_name},,0,0,0,,{styled_text}"
            lines.append(line)

        return "\n".join(lines)

    def generate_ass(
        self,
        segments: list[Segment],
        title: str = "Subtitles",
    ) -> str:
        """
        Generate complete ASS file content.

        Args:
            segments: List of segments.
            title: Subtitle title.

        Returns:
            Complete ASS file content.
        """
        parts = [
            self.generate_script_info(title),
            self.generate_styles_block(),
            "",
            self.generate_events(segments),
        ]

        return "\n".join(parts)
