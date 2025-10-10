"""
Translation Service for Japanese ↔ English
Uses Helsinki-NLP MarianMT models for local, offline translation
"""

import logging
from typing import Optional
import torch
from transformers import MarianMTModel, MarianTokenizer

logger = logging.getLogger(__name__)


class Translator:
    """Bilingual translator for Japanese and English"""

    def __init__(self, device: str = "auto"):
        """
        Initialize translation models

        Args:
            device: Device to run models on ("cpu", "cuda", "mps", or "auto")
        """
        self.device = self._get_device(device)
        logger.info(f"Initializing translator on device: {self.device}")

        # Japanese → English
        self.ja_en_model_name = "Helsinki-NLP/opus-mt-ja-en"
        self.ja_en_model: Optional[MarianMTModel] = None
        self.ja_en_tokenizer: Optional[MarianTokenizer] = None

        # English → Japanese
        self.en_ja_model_name = "Helsinki-NLP/opus-mt-en-jap"
        self.en_ja_model: Optional[MarianMTModel] = None
        self.en_ja_tokenizer: Optional[MarianTokenizer] = None

        # Load models
        self._load_models()

    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_models(self):
        """Load translation models"""
        try:
            logger.info("Loading Japanese → English model...")
            self.ja_en_tokenizer = MarianTokenizer.from_pretrained(
                self.ja_en_model_name
            )
            self.ja_en_model = MarianMTModel.from_pretrained(
                self.ja_en_model_name
            ).to(self.device)
            self.ja_en_model.eval()
            logger.info("✓ Japanese → English model loaded")

            logger.info("Loading English → Japanese model...")
            self.en_ja_tokenizer = MarianTokenizer.from_pretrained(
                self.en_ja_model_name
            )
            self.en_ja_model = MarianMTModel.from_pretrained(
                self.en_ja_model_name
            ).to(self.device)
            self.en_ja_model.eval()
            logger.info("✓ English → Japanese model loaded")

        except Exception as e:
            logger.error(f"Failed to load translation models: {e}")
            raise

    async def translate_ja_to_en(self, text: str) -> str:
        """
        Translate Japanese text to English

        Args:
            text: Japanese text

        Returns:
            Translated English text
        """
        if not text or not text.strip():
            return text

        try:
            # Tokenize
            inputs = self.ja_en_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate translation
            with torch.no_grad():
                outputs = self.ja_en_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode
            translated = self.ja_en_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            logger.debug(f"Translated JA→EN: '{text[:50]}...' → '{translated[:50]}...'")
            return translated

        except Exception as e:
            logger.error(f"Translation error (JA→EN): {e}")
            return text  # Return original on error

    async def translate_en_to_ja(self, text: str) -> str:
        """
        Translate English text to Japanese

        Args:
            text: English text

        Returns:
            Translated Japanese text
        """
        if not text or not text.strip():
            return text

        try:
            # Tokenize
            inputs = self.en_ja_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate translation
            with torch.no_grad():
                outputs = self.en_ja_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode
            translated = self.en_ja_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            logger.debug(f"Translated EN→JA: '{text[:50]}...' → '{translated[:50]}...'")
            return translated

        except Exception as e:
            logger.error(f"Translation error (EN→JA): {e}")
            return text  # Return original on error

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Generic translation method

        Args:
            text: Text to translate
            source_lang: Source language ("en" or "ja")
            target_lang: Target language ("en" or "ja")

        Returns:
            Translated text
        """
        if source_lang == target_lang:
            return text

        if source_lang == "ja" and target_lang == "en":
            return await self.translate_ja_to_en(text)
        elif source_lang == "en" and target_lang == "ja":
            return await self.translate_en_to_ja(text)
        else:
            logger.warning(f"Unsupported language pair: {source_lang}→{target_lang}")
            return text


# Global translator instance (initialized in main.py)
translator: Optional[Translator] = None


def get_translator() -> Translator:
    """Get global translator instance"""
    global translator
    if translator is None:
        translator = Translator()
    return translator
