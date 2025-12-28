"""Segment splitting module for Voice Subtitle Generator."""

import logging
import re
from dataclasses import dataclass

from .models import Segment, Word

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for segment splitting."""

    enabled: bool = True
    max_chars: int = 80  # Maximum characters per segment
    max_duration: float = 4.0  # Maximum duration in seconds
    split_punctuation: str = "。！？!?…"  # Sentence-ending punctuation


@dataclass
class MergeConfig:
    """Configuration for segment merging."""

    enabled: bool = True
    min_chars: int = 5  # Minimum characters per segment
    min_duration: float = 0.5  # Minimum duration in seconds
    max_gap: float = 0.3  # Maximum gap between segments to merge
    merge_incomplete_sentences: bool = True  # Merge segments that don't end with punctuation
    sentence_end_chars: str = "。！？!?.…"  # Characters that mark sentence end


class SegmentSplitter:
    """
    Splits long segments into smaller ones for better subtitle readability.

    Uses word-level timing from WhisperX to create accurate timing
    for split segments.
    """

    def __init__(self, config: SplitConfig | None = None):
        """
        Initialize segment splitter.

        Args:
            config: Split configuration. Uses defaults if None.
        """
        self.config = config or SplitConfig()

    def split_segments(self, segments: list[Segment]) -> list[Segment]:
        """
        Split long segments into smaller ones.

        Args:
            segments: List of segments to process.

        Returns:
            list[Segment]: New list with split segments.
        """
        if not self.config.enabled:
            return segments

        result = []
        for segment in segments:
            if self._needs_split(segment):
                split_segs = self._split_segment(segment)
                result.extend(split_segs)
                logger.debug(
                    f"Split segment ({segment.duration:.1f}s, {len(segment.original_text)} chars) "
                    f"into {len(split_segs)} parts"
                )
            else:
                result.append(segment)

        logger.info(f"Split {len(segments)} segments into {len(result)} segments")
        return result

    def _needs_split(self, segment: Segment) -> bool:
        """Check if segment needs splitting."""
        if len(segment.original_text) > self.config.max_chars:
            return True
        if segment.duration > self.config.max_duration:
            return True
        return False

    def _split_segment(self, segment: Segment) -> list[Segment]:
        """
        Split a single segment into multiple segments.

        Tries punctuation-based split first, then falls back to
        length-based split if needed.
        """
        # Try punctuation-based split first
        split_segs = self._split_by_punctuation(segment)

        # If punctuation split didn't help enough, split by length
        result = []
        for seg in split_segs:
            if self._needs_split(seg):
                result.extend(self._split_by_length(seg))
            else:
                result.append(seg)

        return result if result else [segment]

    def _split_by_punctuation(self, segment: Segment) -> list[Segment]:
        """
        Split segment by sentence-ending punctuation.

        Uses word-level timing to determine accurate boundaries.
        """
        text = segment.original_text
        words = segment.words

        # If no words, can't split accurately
        if not words:
            return [segment]

        # Find split points (indices after punctuation)
        pattern = f"[{re.escape(self.config.split_punctuation)}]"
        split_points = []

        for match in re.finditer(pattern, text):
            # Position after the punctuation
            split_points.append(match.end())

        # If no punctuation found or only at the end, return original
        if not split_points or (len(split_points) == 1 and split_points[0] >= len(text) - 1):
            return [segment]

        # Create segments based on split points
        segments = []
        prev_pos = 0

        for split_pos in split_points:
            sub_text = text[prev_pos:split_pos].strip()
            if sub_text:
                # Find word range for this text segment
                start_time, end_time = self._find_word_timing(
                    words, text, prev_pos, split_pos
                )

                if start_time is not None and end_time is not None:
                    new_seg = Segment(
                        start=start_time,
                        end=end_time,
                        original_text=sub_text,
                        translated_text="",  # Will be translated later
                        speaker_id=segment.speaker_id,
                        speaker_name=segment.speaker_name,
                        words=self._extract_words(words, start_time, end_time),
                    )
                    segments.append(new_seg)

            prev_pos = split_pos

        # Handle remaining text after last punctuation
        if prev_pos < len(text):
            sub_text = text[prev_pos:].strip()
            if sub_text:
                start_time, end_time = self._find_word_timing(
                    words, text, prev_pos, len(text)
                )

                if start_time is not None and end_time is not None:
                    new_seg = Segment(
                        start=start_time,
                        end=end_time,
                        original_text=sub_text,
                        translated_text="",
                        speaker_id=segment.speaker_id,
                        speaker_name=segment.speaker_name,
                        words=self._extract_words(words, start_time, end_time),
                    )
                    segments.append(new_seg)

        return segments if segments else [segment]

    def _split_by_length(self, segment: Segment) -> list[Segment]:
        """
        Split segment by character count or duration.

        Falls back to simple proportional split if word timing is unavailable.
        """
        text = segment.original_text
        words = segment.words

        # Calculate target segment count
        char_splits = max(1, len(text) // self.config.max_chars)
        duration_splits = max(1, int(segment.duration / self.config.max_duration))
        target_splits = max(char_splits, duration_splits)

        if target_splits <= 1:
            return [segment]

        # If no word timing, do simple proportional split
        if not words:
            return self._proportional_split(segment, target_splits)

        # Split at word boundaries
        words_per_segment = max(1, len(words) // target_splits)
        segments = []

        for i in range(target_splits):
            start_idx = i * words_per_segment
            if i == target_splits - 1:
                # Last segment gets all remaining words
                end_idx = len(words)
            else:
                end_idx = min((i + 1) * words_per_segment, len(words))

            if start_idx >= len(words):
                break

            seg_words = words[start_idx:end_idx]
            if not seg_words:
                continue

            # Build text from words
            sub_text = "".join(w.word for w in seg_words).strip()
            if not sub_text:
                continue

            new_seg = Segment(
                start=seg_words[0].start,
                end=seg_words[-1].end,
                original_text=sub_text,
                translated_text="",
                speaker_id=segment.speaker_id,
                speaker_name=segment.speaker_name,
                words=seg_words,
            )
            segments.append(new_seg)

        return segments if segments else [segment]

    def _proportional_split(self, segment: Segment, num_splits: int) -> list[Segment]:
        """
        Split segment proportionally by time when word timing is unavailable.
        """
        text = segment.original_text
        total_duration = segment.duration
        chars_per_segment = len(text) // num_splits
        time_per_segment = total_duration / num_splits

        segments = []
        for i in range(num_splits):
            char_start = i * chars_per_segment
            if i == num_splits - 1:
                char_end = len(text)
            else:
                char_end = (i + 1) * chars_per_segment

            sub_text = text[char_start:char_end].strip()
            if not sub_text:
                continue

            new_seg = Segment(
                start=segment.start + i * time_per_segment,
                end=segment.start + (i + 1) * time_per_segment,
                original_text=sub_text,
                translated_text="",
                speaker_id=segment.speaker_id,
                speaker_name=segment.speaker_name,
                words=[],
            )
            segments.append(new_seg)

        return segments if segments else [segment]

    def _find_word_timing(
        self,
        words: list[Word],
        text: str,
        char_start: int,
        char_end: int,
    ) -> tuple[float | None, float | None]:
        """
        Find word timing for a text range.

        Matches character positions to word positions to find timing.
        """
        if not words:
            return None, None

        # Build character position mapping
        word_char_pos = 0
        first_word_idx = None
        last_word_idx = None

        for idx, word in enumerate(words):
            word_text = word.word
            word_start_pos = word_char_pos
            word_end_pos = word_char_pos + len(word_text)

            # Check if this word overlaps with our range
            if word_end_pos > char_start and word_start_pos < char_end:
                if first_word_idx is None:
                    first_word_idx = idx
                last_word_idx = idx

            word_char_pos = word_end_pos

        if first_word_idx is None or last_word_idx is None:
            # Fallback: use proportional timing
            text_len = len(text)
            if text_len == 0:
                return None, None

            total_duration = words[-1].end - words[0].start
            start_ratio = char_start / text_len
            end_ratio = char_end / text_len

            return (
                words[0].start + total_duration * start_ratio,
                words[0].start + total_duration * end_ratio,
            )

        return words[first_word_idx].start, words[last_word_idx].end

    def _extract_words(
        self,
        words: list[Word],
        start_time: float,
        end_time: float,
    ) -> list[Word]:
        """Extract words within a time range."""
        return [
            w for w in words
            if w.start >= start_time - 0.01 and w.end <= end_time + 0.01
        ]


class SegmentMerger:
    """
    Merges short segments for better subtitle readability.

    Combines segments that are too short (by character count or duration)
    with adjacent segments from the same speaker.
    """

    def __init__(self, config: MergeConfig | None = None):
        """
        Initialize segment merger.

        Args:
            config: Merge configuration. Uses defaults if None.
        """
        self.config = config or MergeConfig()

    def merge_short_segments(self, segments: list[Segment]) -> list[Segment]:
        """
        Merge short segments with adjacent ones.

        Args:
            segments: List of segments to process.

        Returns:
            list[Segment]: New list with merged segments.
        """
        if not self.config.enabled or not segments:
            return segments

        result = []
        i = 0

        while i < len(segments):
            current = segments[i]

            # Check if current segment needs merging
            if self._needs_merge(current):
                # Try to merge with next segment
                if i + 1 < len(segments) and self._can_merge(current, segments[i + 1]):
                    merged = self._merge_segments(current, segments[i + 1])
                    # Continue checking if merged segment still needs merging
                    i += 2
                    while i < len(segments) and self._needs_merge(merged) and self._can_merge(merged, segments[i]):
                        merged = self._merge_segments(merged, segments[i])
                        i += 1
                    result.append(merged)
                    continue
                # Try to merge with previous segment
                elif result and self._can_merge(result[-1], current):
                    merged = self._merge_segments(result[-1], current)
                    result[-1] = merged
                    i += 1
                    continue

            result.append(current)
            i += 1

        if len(result) != len(segments):
            logger.info(f"Merged segments: {len(segments)} -> {len(result)}")

        return result

    def _needs_merge(self, segment: Segment) -> bool:
        """Check if segment is too short or incomplete and needs merging."""
        # Don't merge if segment is already long enough (max 8 seconds)
        if segment.duration >= 8.0:
            return False

        # Too short by character count
        if len(segment.original_text) < self.config.min_chars:
            return True

        # Too short by duration
        if segment.duration < self.config.min_duration:
            return True

        # Incomplete sentence (doesn't end with sentence-ending punctuation)
        if self.config.merge_incomplete_sentences:
            text = segment.original_text.strip()
            if text and text[-1] not in self.config.sentence_end_chars:
                return True

        return False

    def _can_merge(self, seg1: Segment, seg2: Segment) -> bool:
        """Check if two segments can be merged."""
        # Must be same speaker
        if seg1.speaker_id != seg2.speaker_id:
            return False

        gap = seg2.start - seg1.end

        # Check if first segment is an incomplete sentence
        text = seg1.original_text.strip()
        is_incomplete = text and text[-1] not in self.config.sentence_end_chars

        # Allow larger gap for incomplete sentences (up to 2 seconds)
        max_allowed_gap = 2.0 if is_incomplete else self.config.max_gap

        if gap > max_allowed_gap:
            return False

        return True

    def _merge_segments(self, seg1: Segment, seg2: Segment) -> Segment:
        """Merge two segments into one."""
        # Combine words if available
        combined_words = []
        if seg1.words:
            combined_words.extend(seg1.words)
        if seg2.words:
            combined_words.extend(seg2.words)

        return Segment(
            start=seg1.start,
            end=seg2.end,
            original_text=seg1.original_text + seg2.original_text,
            translated_text="",  # Will be translated later
            speaker_id=seg1.speaker_id,
            speaker_name=seg1.speaker_name,
            words=combined_words,
        )


# Convenience function
def split_long_segments(
    segments: list[Segment],
    max_chars: int = 80,
    max_duration: float = 4.0,
    enabled: bool = True,
) -> list[Segment]:
    """
    Split long segments into smaller ones.

    Convenience function for quick use without creating a SegmentSplitter.

    Args:
        segments: Segments to split.
        max_chars: Maximum characters per segment.
        max_duration: Maximum duration per segment.
        enabled: Whether to enable splitting.

    Returns:
        list[Segment]: Split segments.
    """
    config = SplitConfig(
        enabled=enabled,
        max_chars=max_chars,
        max_duration=max_duration,
    )
    splitter = SegmentSplitter(config)
    return splitter.split_segments(segments)


def merge_short_segments(
    segments: list[Segment],
    min_chars: int = 5,
    min_duration: float = 0.5,
    max_gap: float = 0.3,
    enabled: bool = True,
) -> list[Segment]:
    """
    Merge short segments into longer ones.

    Convenience function for quick use without creating a SegmentMerger.

    Args:
        segments: Segments to merge.
        min_chars: Minimum characters per segment.
        min_duration: Minimum duration per segment.
        max_gap: Maximum gap between segments to allow merging.
        enabled: Whether to enable merging.

    Returns:
        list[Segment]: Merged segments.
    """
    config = MergeConfig(
        enabled=enabled,
        min_chars=min_chars,
        min_duration=min_duration,
        max_gap=max_gap,
    )
    merger = SegmentMerger(config)
    return merger.merge_short_segments(segments)
