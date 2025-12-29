"""Pipeline orchestration for Voice Subtitle Generator."""

import logging
from pathlib import Path
from typing import Any

from .audio_analyzer import AudioPositionAnalyzer
from .audio_loader import AudioLoader
from .config import AppConfig, get_config, TRANSLATION_MODEL_PRESETS
from .models import (
    BatchCallbacks,
    PipelineCallbacks,
    ProcessingStage,
    ProcessResult,
    Segment,
)
from .segment_splitter import MergeConfig, SegmentMerger, SegmentSplitter, SplitConfig
from .speech_engine import SpeechEngine
from .subtitle_writer import SubtitleWriter
from .translator import Translator

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when pipeline processing fails."""

    pass


class SubtitlePipeline:
    """
    Main processing pipeline for subtitle generation.

    Orchestrates audio loading, transcription, translation,
    and subtitle generation.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Application configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Components (lazy initialized)
        self._audio_loader: AudioLoader | None = None
        self._speech_engine: SpeechEngine | None = None
        self._translator: Translator | None = None
        self._subtitle_writer: SubtitleWriter | None = None
        self._segment_splitter: SegmentSplitter | None = None
        self._segment_merger: SegmentMerger | None = None

        self._initialized = False

    def initialize(self, skip_translation: bool = False) -> None:
        """
        Initialize all pipeline components.

        Loads models into memory. Call before processing.

        Args:
            skip_translation: If True, skip loading translation model.
        """
        if self._initialized:
            return

        logger.info("Initializing pipeline...")

        # Audio loader
        self._audio_loader = AudioLoader(
            target_sample_rate=self.config.processing.target_sample_rate,
            normalize=self.config.processing.normalize_audio,
        )

        # Speech engine (STT)
        self._speech_engine = SpeechEngine.from_config(self.config.speech)
        self._speech_engine.load_models(self.config.speech.stt.language)

        # Translator - only load if translation is needed
        if not skip_translation:
            model_path = self.config.translation.get_model_path(self.config.models.path)
            self._translator = Translator(
                model_path=model_path,
                n_gpu_layers=self.config.translation.n_gpu_layers,
                n_ctx=self.config.translation.n_ctx,
                prompt_template=self.config.translation.get_prompt_template(),
            )
            self._translator.load_model()
        else:
            logger.info("Skipping translator initialization (transcription-only mode)")

        # Subtitle writer
        self._subtitle_writer = SubtitleWriter.from_config(self.config.subtitle)

        # Segment splitter
        self._segment_splitter = SegmentSplitter(
            SplitConfig(
                enabled=self.config.segment.enabled,
                max_chars=self.config.segment.max_chars,
                max_duration=self.config.segment.max_duration,
                split_punctuation=self.config.segment.split_punctuation,
            )
        )

        # Segment merger
        self._segment_merger = SegmentMerger(
            MergeConfig(
                enabled=self.config.segment.enabled,
                min_chars=self.config.segment.min_chars,
                min_duration=self.config.segment.min_duration,
                max_gap=self.config.segment.merge_gap,
                merge_incomplete_sentences=self.config.segment.merge_incomplete,
                sentence_end_chars=self.config.segment.split_punctuation,
            )
        )

        self._initialized = True
        logger.info("Pipeline initialized")

    def cleanup(self) -> None:
        """Cleanup all resources and unload models."""
        logger.info("Cleaning up pipeline...")

        if self._speech_engine:
            self._speech_engine.unload_models()
            self._speech_engine = None

        if self._translator:
            self._translator.unload_model()
            self._translator = None

        if self._audio_loader:
            self._audio_loader.cleanup()
            self._audio_loader = None

        self._subtitle_writer = None
        self._segment_splitter = None
        self._segment_merger = None
        self._initialized = False

        logger.info("Pipeline cleanup complete")

    def process_file(
        self,
        audio_path: str,
        output_path: str | None = None,
        output_format: str | None = None,
        callbacks: PipelineCallbacks | None = None,
        skip_translation: bool = False,
    ) -> ProcessResult:
        """
        Process a single audio file.

        Args:
            audio_path: Input audio file path.
            output_path: Output subtitle path. Auto-generated if None.
            output_format: Output format (srt, ass, vtt).
            callbacks: Progress callbacks.
            skip_translation: Skip translation step (transcription only).

        Returns:
            ProcessResult: Processing result.
        """
        if not self._initialized:
            self.initialize(skip_translation=skip_translation)

        audio_path = str(Path(audio_path).absolute())
        output_format = output_format or self.config.subtitle.default_format

        # Generate output path if not provided
        if output_path is None:
            audio_file = Path(audio_path)
            output_path = str(audio_file.with_suffix(f".{output_format}"))

        logger.info(f"Processing: {audio_path}")

        try:
            # Stage 1: Loading
            self._notify_stage(callbacks, ProcessingStage.LOADING)
            self._notify_log(callbacks, "오디오 파일 분석 중...")
            audio_info = self._audio_loader.get_info(audio_path)
            self._notify_log(
                callbacks,
                f"오디오 길이: {audio_info.duration:.1f}초, "
                f"샘플레이트: {audio_info.sample_rate}Hz",
            )

            # Stage 2: Transcription
            self._notify_stage(callbacks, ProcessingStage.TRANSCRIBING)
            self._notify_log(callbacks, "음성 인식 시작...")

            segments = self._speech_engine.process(
                audio_path,
                language=self.config.speech.stt.language,
                progress_callback=lambda p: self._notify_progress(
                    callbacks, 0.1 + p * 0.4  # 10% - 50%
                ),
            )

            self._notify_log(callbacks, f"음성 인식 완료: {len(segments)}개 구간 검출")
            logger.info(f"Transcribed {len(segments)} segments")

            # Split long segments
            if self.config.segment.enabled:
                orig_count = len(segments)
                segments = self._segment_splitter.split_segments(segments)
                if len(segments) != orig_count:
                    self._notify_log(
                        callbacks,
                        f"세그먼트 분할: {orig_count}개 → {len(segments)}개",
                    )
                    logger.info(f"Split segments: {orig_count} -> {len(segments)}")

            # Merge short segments
            if self.config.segment.enabled:
                pre_merge_count = len(segments)
                segments = self._segment_merger.merge_short_segments(segments)
                if len(segments) != pre_merge_count:
                    self._notify_log(
                        callbacks,
                        f"세그먼트 병합: {pre_merge_count}개 → {len(segments)}개",
                    )
                    logger.info(f"Merged segments: {pre_merge_count} -> {len(segments)}")

            # Stage 2.5: Audio Position Analysis (for stereo/binaural audio)
            self._analyze_audio_positions(audio_path, segments, callbacks)

            # Stage 3: Translation (skip if transcription only)
            if not skip_translation:
                # Lazy load translator if needed (e.g., switched from transcription-only mode)
                if self._translator is None:
                    self._notify_log(callbacks, "번역 모델 로딩 중...")
                    model_path = self.config.translation.get_model_path(
                        self.config.models.path
                    )
                    self._translator = Translator(
                        model_path=model_path,
                        n_gpu_layers=self.config.translation.n_gpu_layers,
                        n_ctx=self.config.translation.n_ctx,
                        prompt_template=self.config.translation.get_prompt_template(),
                    )
                    self._translator.load_model()
                    self._notify_log(callbacks, "번역 모델 로딩 완료", "success")

                self._notify_stage(callbacks, ProcessingStage.TRANSLATING)

                self._notify_log(callbacks, f"번역 시작 (0/{len(segments)})...")

                translated_count = [0]

                def translation_progress(p: float):
                    self._notify_progress(
                        callbacks, 0.5 + p * 0.4  # 50% -> 90%
                    )
                    new_count = int(p * len(segments))
                    if new_count > translated_count[0]:
                        translated_count[0] = new_count
                        if new_count % 10 == 0 or new_count == len(segments):
                            self._notify_log(
                                callbacks, f"번역 진행 중... ({new_count}/{len(segments)})"
                            )

                self._translator.translate_segments(
                    segments,
                    max_tokens=self.config.translation.max_tokens,
                    temperature=self.config.translation.temperature,
                    progress_callback=translation_progress,
                )

                self._notify_log(callbacks, "번역 완료", "success")
                logger.info("Translation complete")
            else:
                # Skip translation - just update progress
                self._notify_progress(callbacks, 0.9)
                self._notify_log(callbacks, "번역 스킵 (원문 자막 모드)", "info")
                logger.info("Translation skipped (transcription only)")

            # Stage 4: Writing
            self._notify_stage(callbacks, ProcessingStage.WRITING)
            self._notify_log(callbacks, f"자막 파일 저장 중... ({output_format})")
            self._notify_progress(callbacks, 0.95)

            self._subtitle_writer.write(
                segments,
                output_path,
                format=output_format,
            )

            # Done
            self._notify_stage(callbacks, ProcessingStage.DONE)
            self._notify_progress(callbacks, 1.0)
            self._notify_log(callbacks, f"저장 완료: {Path(output_path).name}", "success")

            logger.info(f"Saved: {output_path}")

            return ProcessResult(
                success=True,
                audio_path=audio_path,
                output_path=output_path,
                segments=segments,
                duration=audio_info.duration,
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self._notify_stage(callbacks, ProcessingStage.ERROR)

            if callbacks and callbacks.on_error:
                callbacks.on_error(str(e))

            return ProcessResult(
                success=False,
                audio_path=audio_path,
                output_path=output_path or "",
                error=str(e),
            )

    def process_batch(
        self,
        files: list[tuple[str, str]] | list[str],
        output_format: str | None = None,
        output_dir: str | None = None,
        callbacks: BatchCallbacks | None = None,
        skip_translation: bool = False,
    ) -> list[ProcessResult]:
        """
        Process multiple audio files.

        Args:
            files: List of (input, output) tuples or just input paths.
            output_format: Output format for all files.
            output_dir: Output directory (used when files is list of strings).
            callbacks: Batch processing callbacks.
            skip_translation: Skip translation step (transcription only).

        Returns:
            list[ProcessResult]: Results for each file.
        """
        if not self._initialized:
            self.initialize(skip_translation=skip_translation)

        # Normalize file list
        file_pairs: list[tuple[str, str | None]] = []
        for item in files:
            if isinstance(item, tuple):
                file_pairs.append(item)
            else:
                # Generate output path
                if output_dir:
                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fmt = output_format or self.config.subtitle.default_format
                    output = str(out_dir / f"{Path(item).stem}.{fmt}")
                else:
                    output = None
                file_pairs.append((item, output))

        results = []
        total = len(file_pairs)

        for i, (audio_path, output_path) in enumerate(file_pairs):
            # Notify file start
            if callbacks and callbacks.on_file_start:
                callbacks.on_file_start(audio_path, i + 1, total)

            # Process file
            result = self.process_file(
                audio_path=audio_path,
                output_path=output_path,
                output_format=output_format,
                callbacks=callbacks.pipeline_callbacks if callbacks else None,
                skip_translation=skip_translation,
            )

            results.append(result)

            # Notify file complete
            if callbacks and callbacks.on_file_complete:
                callbacks.on_file_complete(result)

        # Notify batch complete
        if callbacks and callbacks.on_batch_complete:
            callbacks.on_batch_complete(results)

        return results

    def get_segments_preview(
        self,
        audio_path: str,
        callbacks: PipelineCallbacks | None = None,
    ) -> list[Segment]:
        """
        Get transcribed segments without translation.

        Useful for preview before committing to translation.

        Args:
            audio_path: Audio file path.
            callbacks: Progress callbacks.

        Returns:
            list[Segment]: Transcribed segments.
        """
        if not self._initialized:
            self.initialize()

        self._notify_stage(callbacks, ProcessingStage.TRANSCRIBING)

        segments = self._speech_engine.process(
            audio_path,
            language=self.config.speech.stt.language,
            progress_callback=lambda p: self._notify_progress(callbacks, p),
        )

        self._notify_stage(callbacks, ProcessingStage.DONE)

        return segments

    def translate_segments(
        self,
        segments: list[Segment],
        callbacks: PipelineCallbacks | None = None,
    ) -> list[Segment]:
        """
        Translate pre-transcribed segments.

        Args:
            segments: Segments to translate.
            callbacks: Progress callbacks.

        Returns:
            list[Segment]: Translated segments.
        """
        if not self._initialized:
            self.initialize()

        self._notify_stage(callbacks, ProcessingStage.TRANSLATING)

        self._translator.translate_segments(
            segments,
            progress_callback=lambda p: self._notify_progress(callbacks, p),
        )

        self._notify_stage(callbacks, ProcessingStage.DONE)

        return segments

    def _notify_stage(
        self,
        callbacks: PipelineCallbacks | None,
        stage: ProcessingStage,
    ) -> None:
        """Notify stage change."""
        if callbacks and callbacks.on_stage_change:
            callbacks.on_stage_change(stage)

    def _notify_progress(
        self,
        callbacks: PipelineCallbacks | None,
        progress: float,
    ) -> None:
        """Notify progress update."""
        if callbacks and callbacks.on_progress:
            callbacks.on_progress(progress)

    def _notify_log(
        self,
        callbacks: PipelineCallbacks | None,
        message: str,
        level: str = "info",
    ) -> None:
        """Notify log message."""
        if callbacks and callbacks.on_log:
            callbacks.on_log(message, level)

    def _analyze_audio_positions(
        self,
        audio_path: str,
        segments: list[Segment],
        callbacks: PipelineCallbacks | None = None,
    ) -> None:
        """
        Analyze audio positions for stereo/binaural audio.

        Updates segment.position field based on L/R channel energy.

        Args:
            audio_path: Path to the audio file.
            segments: List of segments to analyze.
            callbacks: Optional callbacks for progress/logging.
        """
        if not segments:
            return

        threshold = self.config.subtitle.ass.position_threshold
        analyzer = AudioPositionAnalyzer(
            left_threshold=-threshold,
            right_threshold=threshold,
        )

        if not analyzer.load_audio(audio_path):
            # Mono audio or failed to load - skip position analysis
            logger.debug("Audio is mono or failed to load, skipping position analysis")
            return

        try:
            self._notify_log(callbacks, "오디오 위치 분석 중...")

            position_counts = {"left": 0, "center": 0, "right": 0}

            for segment in segments:
                result = analyzer.analyze_position(segment.start, segment.end)
                segment.position = result.position

                # Count positions for logging
                pos_name = result.position.value
                if pos_name in position_counts:
                    position_counts[pos_name] += 1

            # Log position distribution
            left = position_counts["left"]
            center = position_counts["center"]
            right = position_counts["right"]
            self._notify_log(
                callbacks,
                f"위치 분석 완료: 좌측 {left}개, 중앙 {center}개, 우측 {right}개",
            )
            logger.info(f"Position analysis: left={left}, center={center}, right={right}")

        finally:
            analyzer.cleanup()

    @property
    def is_ready(self) -> bool:
        """Check if pipeline is initialized and ready."""
        return self._initialized

    @property
    def audio_loader(self) -> AudioLoader | None:
        """Get audio loader instance."""
        return self._audio_loader

    @property
    def speech_engine(self) -> SpeechEngine | None:
        """Get speech engine instance."""
        return self._speech_engine

    @property
    def translator(self) -> Translator | None:
        """Get translator instance."""
        return self._translator

    @property
    def subtitle_writer(self) -> SubtitleWriter | None:
        """Get subtitle writer instance."""
        return self._subtitle_writer

    def estimate_vram_usage(self) -> dict[str, float]:
        """
        Estimate total VRAM usage.

        Returns:
            dict: VRAM usage by component in GB.
        """
        # Get speech engine VRAM
        if self._speech_engine:
            speech_vram = self._speech_engine.estimate_vram_usage()
        else:
            speech_vram = {"stt": 4.0, "total": 4.0}

        # Get translator VRAM from preset
        preset_key = self.config.translation.model_preset
        if preset_key in TRANSLATION_MODEL_PRESETS:
            translator_vram = TRANSLATION_MODEL_PRESETS[preset_key].vram_gb
        else:
            translator_vram = 9.0  # Default estimate

        return {
            "speech": speech_vram.get("total", 5.0),
            "translator": translator_vram,
            "total": speech_vram.get("total", 5.0) + translator_vram,
        }
