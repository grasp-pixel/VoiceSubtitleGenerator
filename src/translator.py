"""LLM-based translation module for Voice Subtitle Generator."""

import gc
import logging
import re
from pathlib import Path

from llama_cpp import Llama

from .config import TranslationConfig
from .models import ProgressCallback, Segment

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Raised when translation fails."""

    pass


class ModelNotFoundError(TranslationError):
    """Raised when model file is not found."""

    pass


class Translator:
    """
    LLM-based translator using llama-cpp-python.

    Translates Japanese text to Korean using local GGUF models.
    """

    DEFAULT_PROMPT = (
        "Translate the following Japanese text to Korean.\n"
        "Japanese: {text}\n"
        "Korean:"
    )

    DEFAULT_REVIEW_PROMPT = (
        "Review the Korean translation and fix any errors.\n"
        "Original: {original}\n"
        "Translation: {translation}\n"
        "Corrected:"
    )

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        prompt_template: str | None = None,
        review_prompt_template: str | None = None,
        enable_review: bool = False,
    ):
        """
        Initialize translator.

        Args:
            model_path: Path to GGUF model file.
            n_gpu_layers: Number of layers to offload to GPU (-1 for all).
            n_ctx: Context size.
            prompt_template: Translation prompt template with {text} placeholder.
            review_prompt_template: Review prompt template.
            enable_review: Enable translation review pass.
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.review_prompt_template = review_prompt_template or self.DEFAULT_REVIEW_PROMPT
        self.enable_review = enable_review

        self._model: Llama | None = None

    def _get_stop_tokens(self) -> list[str]:
        """Get stop tokens for translation."""
        # Using /no_think in prompt, so model won't output <think>
        return ["<|im_end|>"]

    @classmethod
    def from_config(cls, config: TranslationConfig) -> "Translator":
        """Create translator from config."""
        return cls(
            model_path=config.get_model_path(),
            n_gpu_layers=config.n_gpu_layers,
            n_ctx=config.n_ctx,
            prompt_template=config.get_prompt_template(),
            review_prompt_template=config.get_review_prompt_template(),
            enable_review=config.enable_review,
        )

    def load_model(self) -> None:
        """
        Load the LLM model.

        Raises:
            ModelNotFoundError: If model file doesn't exist.
            TranslationError: If model loading fails.
        """
        model_path = Path(self.model_path)

        if not model_path.exists():
            raise ModelNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading translation model: {self.model_path}")

        try:
            self._model = Llama(
                model_path=str(model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False,
            )
            logger.info("Translation model loaded successfully")
        except Exception as e:
            raise TranslationError(f"Failed to load model: {e}") from e

    def unload_model(self) -> None:
        """Unload model and free memory."""
        self._model = None
        gc.collect()
        logger.info("Translation model unloaded")

    def translate(
        self,
        text: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> str:
        """
        Translate a single text.

        Args:
            text: Japanese text to translate.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            str: Korean translation.

        Raises:
            TranslationError: If translation fails.
        """
        if self._model is None:
            self.load_model()

        if not text.strip():
            return ""

        # Build prompt
        prompt = self.prompt_template.format(text=text.strip())

        try:
            response = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=self._get_stop_tokens(),
                echo=False,
            )

            # Extract generated text
            raw_result = response["choices"][0]["text"].strip()

            # Log thinking process for debugging
            self._log_thinking(raw_result)

            # Extract subtitle content and clean
            result = self._extract_subtitle(raw_result)
            result = self._clean_translation(result)

            return result.strip()

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Translation failed: {e}") from e

    def review_translation(
        self,
        original: str,
        translation: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """
        Review and fix a translation.

        Args:
            original: Original Japanese text.
            translation: Korean translation to review.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower for more consistent output).

        Returns:
            str: Corrected Korean translation.
        """
        if self._model is None:
            self.load_model()

        if not translation.strip():
            return ""

        # Build review prompt
        prompt = self.review_prompt_template.format(
            original=original.strip(),
            translation=translation.strip(),
        )

        try:
            response = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=self._get_stop_tokens(),
                echo=False,
            )

            raw_result = response["choices"][0]["text"].strip()

            # Log thinking process for debugging
            self._log_thinking(raw_result)

            # Extract subtitle content and clean
            result = self._extract_subtitle(raw_result)
            result = self._clean_translation(result)

            # If review result is empty, return original translation
            if not result.strip():
                return translation

            return result.strip()

        except Exception as e:
            logger.warning(f"Review failed, using original translation: {e}")
            return translation

    def translate_batch(
        self,
        texts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.3,
        progress_callback: ProgressCallback | None = None,
    ) -> list[str]:
        """
        Translate multiple texts.

        Args:
            texts: List of Japanese texts.
            max_tokens: Maximum tokens per translation.
            temperature: Sampling temperature.
            progress_callback: Progress callback (0.0 to 1.0).

        Returns:
            list[str]: List of Korean translations.
        """
        if self._model is None:
            self.load_model()

        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            try:
                translation = self.translate(text, max_tokens, temperature)
                results.append(translation)
            except TranslationError:
                # On failure, keep original text
                logger.warning(f"Failed to translate: {text[:50]}...")
                results.append(text)

            # Report progress
            if progress_callback is not None:
                progress_callback((i + 1) / total)

        return results

    def translate_segments(
        self,
        segments: list[Segment],
        max_tokens: int = 256,
        temperature: float = 0.3,
        progress_callback: ProgressCallback | None = None,
        review_callback: ProgressCallback | None = None,
    ) -> list[Segment]:
        """
        Translate segments in place.

        Args:
            segments: List of segments to translate.
            max_tokens: Maximum tokens per translation.
            temperature: Sampling temperature.
            progress_callback: Progress callback for translation phase.
            review_callback: Progress callback for review phase.

        Returns:
            list[Segment]: Segments with translated_text filled.
        """
        texts = [seg.original_text for seg in segments]

        translations = self.translate_batch(
            texts,
            max_tokens=max_tokens,
            temperature=temperature,
            progress_callback=progress_callback,
        )

        for segment, translation in zip(segments, translations):
            segment.translated_text = translation

        # Review pass if enabled
        if self.enable_review:
            total = len(segments)
            for i, segment in enumerate(segments):
                if segment.original_text and segment.translated_text:
                    reviewed = self.review_translation(
                        original=segment.original_text,
                        translation=segment.translated_text,
                        max_tokens=max_tokens,
                        temperature=0.2,  # Lower temp for more consistent review
                    )
                    segment.translated_text = reviewed

                if review_callback:
                    review_callback((i + 1) / total)

        return segments

    def _extract_subtitle(self, text: str) -> str:
        """
        Extract content from <subtitle> tag.

        Args:
            text: Raw model output.

        Returns:
            str: Content inside subtitle tag, or cleaned text if no tag found.
        """
        # First, remove any <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Also remove incomplete <think> tags (started but not closed)
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)

        # Try to extract from <subtitle> tag
        match = re.search(r'<subtitle>(.*?)</subtitle>', cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: try incomplete tag (</subtitle> might be in stop tokens)
        match = re.search(r'<subtitle>(.*?)$', cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()

        # No tag found - return cleaned text
        return cleaned.strip()

    def _clean_translation(self, text: str) -> str:
        """
        Final cleanup of translation result.

        Removes any remaining XML-like tags and normalizes whitespace.

        Args:
            text: Translation text to clean.

        Returns:
            str: Cleaned translation text.
        """
        if not text:
            return text

        # Remove any remaining XML-like tags
        cleaned = re.sub(r'<[^>]+>', '', text)

        # Remove /no_think token that may leak into output
        cleaned = cleaned.replace('/no_think', '').replace('/ no_think', '')

        # Normalize whitespace (but preserve intentional line breaks for subtitles)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in cleaned.split('\n')]
        cleaned = '\n'.join(line for line in lines if line)

        return cleaned.strip()

    def _log_thinking(self, text: str) -> None:
        """
        Log thinking process for debugging.

        Args:
            text: Raw model output containing <think> tags.
        """
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            # Truncate if too long
            if len(thinking) > 500:
                thinking = thinking[:500] + "..."
            logger.debug(f"Model thinking: {thinking}")

    def set_prompt_template(self, template: str) -> None:
        """
        Set translation prompt template.

        Args:
            template: Prompt template with {text} placeholder.
        """
        if "{text}" not in template:
            raise ValueError("Prompt template must contain {text} placeholder")
        self.prompt_template = template

    def load_prompt_from_file(self, file_path: str) -> None:
        """
        Load prompt template from file.

        Args:
            file_path: Path to prompt template file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        template = path.read_text(encoding="utf-8")
        self.set_prompt_template(template)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def is_using_gpu(self) -> bool:
        """
        Check if model is actually using GPU.

        Returns True only if:
        1. n_gpu_layers is not 0 (GPU offload requested)
        2. llama-cpp-python was built with GPU support (CUDA/Metal)

        Returns:
            bool: True if GPU is being used.
        """
        if self.n_gpu_layers == 0:
            return False

        # Check if llama-cpp-python supports GPU offload
        try:
            import llama_cpp

            # llama_supports_gpu_offload() returns True if built with CUDA/Metal
            if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                return llama_cpp.llama_supports_gpu_offload()

            # Fallback: check for CUDA availability via torch
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def estimate_vram_usage(self) -> float:
        """
        Estimate VRAM usage in GB based on model filename.

        Returns:
            float: Estimated VRAM in GB.
        """
        model_name = Path(self.model_path).stem.lower()

        # Rough estimates based on common quantization levels
        if "q4" in model_name or "q4_k" in model_name:
            if "7b" in model_name:
                return 4.0
            elif "14b" in model_name or "13b" in model_name:
                return 9.0
            elif "70b" in model_name:
                return 40.0
        elif "q8" in model_name:
            if "7b" in model_name:
                return 8.0
            elif "14b" in model_name or "13b" in model_name:
                return 16.0

        # Default estimate
        return 10.0
