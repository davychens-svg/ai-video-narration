"""
High-Performance VLM Processor
Optimized for speed with small models (SmolVLM, MobileVLM)
Plus Moondream 3.0 with full feature set (caption, query, detect, point)
"""

import asyncio
import logging
import re
import string
import time
from typing import Optional, Literal, List, Dict, Tuple, Any
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

# Minimal prompts for maximum speed (SmolVLM/MobileVLM)
PROMPTS = {
    "caption": "Describe briefly.",
    "detail": "What's happening?",
    "action": "Main action?",
    "objects": "List objects.",
}

# Moondream-specific settings
MOONDREAM_SETTINGS = {
    "caption": {
        "temperature": 0.5,
        "max_tokens": 768,
        "top_p": 0.3
    },
    "query": {
        "temperature": 0.5,
        "max_tokens": 768,
        "top_p": 0.3
    },
    "detect": {
        "max_objects": 150
    },
    "point": {
        "max_objects": 150
    }
}


class VLMModel:
    """Base class for VLM models with performance optimizations"""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.is_ready = False

    def load(self):
        """Load model - to be implemented by subclasses"""
        raise NotImplementedError

    @torch.inference_mode()
    def caption(self, image: Image.Image, prompt: str = "Describe briefly.") -> str:
        """Generate caption - to be implemented by subclasses"""
        raise NotImplementedError

    def unload(self):
        """Free memory"""
        model_obj = getattr(self, "model", None)
        processor_obj = getattr(self, "processor", None)

        if model_obj is not None:
            del self.model  # release reference for GC
        if processor_obj is not None and hasattr(self, "processor"):
            del self.processor

        # Recreate attributes so downstream code can still reference them safely
        self.model = None
        self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_ready = False


class SmolVLM(VLMModel):
    """SmolVLM-500M - Lightweight model (500M params, 4.5x faster than 2.2B)"""

    def __init__(self, device: str = "auto"):
        super().__init__("smolvlm", device)

    def load(self):
        """Load SmolVLM-500M model for fast inference on Mac"""
        logger.info("Loading SmolVLM-500M...")

        try:
            # Determine dtype based on hardware
            if torch.cuda.is_available():
                dtype = torch.float16  # NVIDIA GPU
                device_map = "cuda"
            elif torch.backends.mps.is_available():
                dtype = torch.bfloat16  # Apple Silicon
                device_map = "mps"
            else:
                dtype = torch.float32  # CPU fallback
                device_map = "cpu"

            logger.info(f"Using device: {device_map}, dtype: {dtype}")

            self.model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            self.processor = AutoProcessor.from_pretrained(
                "HuggingFaceTB/SmolVLM-500M-Instruct",
                trust_remote_code=True
            )

            self.is_ready = True
            logger.info("SmolVLM-500M loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load SmolVLM-500M: {e}")
            raise

    @torch.inference_mode()
    def caption(self, image: Image.Image, prompt: str = "Describe briefly.", max_new_tokens: int = 10) -> str:
        """Generate caption with ultra-fast settings"""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        try:
            target_tokens = int(max(4, min(max_new_tokens, 128)))

            # SmolVLM requires special image token in text
            # Format: "<image>prompt"
            text_with_image = f"<image>{prompt}"

            # Preprocess
            inputs = self.processor(
                text=text_with_image,
                images=image,
                return_tensors="pt"
            ).to(self.model.device)

            # Generate with ultra speed-optimized settings for real-time (<1s)
            output = self.model.generate(
                **inputs,
                max_new_tokens=target_tokens,
                min_new_tokens=3,       # At least 3 tokens
                do_sample=False,        # Greedy decoding (fastest)
                num_beams=1,            # No beam search
                repetition_penalty=1.2, # Prevent repetition
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,         # Enable KV cache
            )

            # Decode
            caption = self.processor.decode(output[0], skip_special_tokens=True)

            # Clean up the caption - remove prompt and special tokens
            import re

            # Remove the input prompt (including <image> token)
            caption = caption.replace(text_with_image, "").replace(prompt, "").strip()

            # Remove SmolVLM special tokens (row/col indicators, global-img, etc.)
            caption = re.sub(r'<[^>]+>', '', caption)

            # Remove common SmolVLM artifacts and incomplete sentences
            # Remove phrases like "in the image.", "The image shows", etc.
            caption = re.sub(r'^(in (the|this) (image|picture|photo)\.?|the (image|picture|photo) (shows|depicts|contains))[\s,]*', '', caption, flags=re.IGNORECASE)
            caption = re.sub(r'^(answer|assistant):?\s*', '', caption, flags=re.IGNORECASE)

            # Remove trailing incomplete phrases
            caption = re.sub(r'\s+(in the image|in this image|the image)\s*\.?\s*$', '', caption, flags=re.IGNORECASE)

            # Remove multiple spaces and clean up
            caption = re.sub(r'\s+', ' ', caption).strip()

            # Capitalize first letter if it's lowercase
            if caption and caption[0].islower():
                caption = caption[0].upper() + caption[1:]

            return caption if caption else "Processing..."

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Error processing frame"


class MobileVLM(VLMModel):
    """MobileVLM - Backup model (1.4B params, even faster)"""

    def __init__(self, device: str = "auto"):
        super().__init__("mobilevlm", device)

    def load(self):
        """Load MobileVLM model"""
        logger.info("Loading MobileVLM...")

        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.model = AutoModelForVision2Seq.from_pretrained(
                "mtgv/MobileVLM-1.7B",
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            self.processor = AutoProcessor.from_pretrained(
                "mtgv/MobileVLM-1.7B",
                trust_remote_code=True
            )

            self.is_ready = True
            logger.info("MobileVLM loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load MobileVLM: {e}")
            raise

    @torch.inference_mode()
    def caption(self, image: Image.Image, prompt: str = "Describe.", max_new_tokens: int = 32) -> str:
        """Generate caption"""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        try:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.model.device)

            output = self.model.generate(
                **inputs,
                max_new_tokens=max(10, min(int(max_new_tokens), 128)),
                do_sample=False,
                num_beams=1,
            )

            caption = self.processor.decode(output[0], skip_special_tokens=True)
            caption = caption.replace(prompt, "").strip()

            return caption

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Error processing frame"


class Moondream(VLMModel):
    """
    Moondream 2 - Feature-rich model with multiple modes
    Supports: caption, query, detect, point
    Works on Apple Silicon MPS or NVIDIA GPU
    """

    def __init__(self, device: str = "auto"):
        super().__init__("moondream", device)
        self.supports_multimodal = True

    def load(self):
        """Load Moondream 2 model"""
        logger.info("Loading Moondream 2...")

        try:
            # Determine device and dtype
            if torch.cuda.is_available():
                dtype = torch.float16
                device_map = "cuda"
            elif torch.backends.mps.is_available():
                dtype = torch.bfloat16
                device_map = "mps"
            else:
                dtype = torch.float32
                device_map = "cpu"

            logger.info(f"Moondream using device: {device_map}, dtype: {dtype}")

            # Import Moondream
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load Moondream 2
            self.model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",  # Moondream 2 (latest)
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True
            )

            self.is_ready = True
            logger.info("Moondream 2 loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load Moondream 2: {e}")
            raise

    @torch.inference_mode()
    def caption(
        self,
        image: Image.Image,
        length: Literal["short", "normal", "long"] = "normal",
        stream: bool = False
    ) -> str:
        """Generate caption/narration for image using moondream2 API"""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        try:
            # Encode image
            enc_image = self.model.encode_image(image)

            # Generate caption
            prompts = {
                "short": "Describe this image in a single concise sentence.",
                "normal": "Describe this image in detail.",
                "long": "Provide an in-depth, multi-sentence description of this image."
            }
            question = prompts.get(length, prompts["normal"])

            caption = self.model.answer_question(
                enc_image,
                question,
                self.tokenizer
            )

            return caption if caption else "No description available"

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Error processing frame"

    @torch.inference_mode()
    def query(
        self,
        image: Image.Image,
        question: str,
        reasoning: bool = True,
        stream: bool = False
    ) -> str:
        """Answer questions about the image"""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        try:
            # Encode image
            enc_image = self.model.encode_image(image)

            # Answer question
            answer = self.model.answer_question(
                enc_image,
                question,
                self.tokenizer
            )

            return answer if answer else "Unable to answer"

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error: {str(e)}"

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        object_description: str
    ) -> List[Dict[str, Any]]:
        """Detect objects using moondream2 detect API"""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        try:
            # Call model.detect directly with PIL image
            # API returns: {"objects": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}, ...]}
            # Coordinates are normalized (0-1), need to convert to pixels
            if hasattr(self.model, 'detect'):
                result = self.model.detect(image, object_description)

                # Get image dimensions for coordinate conversion
                img_width, img_height = image.size

                # Extract objects from result dictionary
                if isinstance(result, dict) and "objects" in result:
                    objects = result["objects"]
                    logger.info(f"Detected {len(objects)} objects (raw): {objects}")

                    # Convert normalized coordinates to pixel coordinates and add bbox format
                    converted_objects = []
                    for obj in objects:
                        if isinstance(obj, dict):
                            # Moondream returns x_min, y_min, x_max, y_max as normalized (0-1)
                            x_min = obj.get('x_min', 0) * img_width
                            y_min = obj.get('y_min', 0) * img_height
                            x_max = obj.get('x_max', 0) * img_width
                            y_max = obj.get('y_max', 0) * img_height

                            converted_objects.append({
                                'bbox': [x_min, y_min, x_max, y_max],
                                'label': object_description,  # Use the query as label
                                'confidence': None  # Moondream doesn't provide confidence scores
                            })

                    logger.info(f"Converted {len(converted_objects)} detections to pixel coordinates")
                    return converted_objects

                logger.warning(f"Detect result format unexpected: {result}")
                return []
            else:
                logger.warning("Model does not have detect method")
                return []

        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    @torch.inference_mode()
    def point(
        self,
        image: Image.Image,
        object_description: str
    ) -> List[Tuple[int, int]]:
        """Find object locations using moondream2 point API"""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        try:
            # Call model.point directly with PIL image
            # API returns: {"points": [{"x": 0.5, "y": 0.3}, ...]}
            # Coordinates are normalized (0-1), need to convert to pixels
            if hasattr(self.model, 'point'):
                result = self.model.point(image, object_description)

                # Get image dimensions for coordinate conversion
                img_width, img_height = image.size

                # Extract points from result dictionary
                if isinstance(result, dict) and "points" in result:
                    points = result["points"]
                    logger.info(f"Found {len(points)} points (raw): {points}")

                    # Convert normalized coordinates to pixel coordinates
                    formatted_points = []
                    for p in points:
                        if isinstance(p, dict):
                            # Moondream returns x, y as normalized (0-1)
                            x = int(p.get('x', 0) * img_width)
                            y = int(p.get('y', 0) * img_height)
                            formatted_points.append((x, y))
                        elif isinstance(p, (list, tuple)) and len(p) >= 2:
                            # Fallback for list/tuple format (shouldn't happen with new Moondream)
                            x = int(p[0] * img_width)
                            y = int(p[1] * img_height)
                            formatted_points.append((x, y))

                    logger.info(f"Converted {len(formatted_points)} points to pixel coordinates: {formatted_points}")
                    return formatted_points

                logger.warning(f"Point result format unexpected: {result}")
                return []
            else:
                logger.warning("Model does not have point method")
                return []

        except Exception as e:
            logger.error(f"Error finding points: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []


class Qwen2VL(VLMModel):
    """
    Qwen2-VL-2B-Instruct - Multilingual vision-language model
    Supports: English, Japanese, Chinese, Korean, and more
    Native multilingual capability without translation layer
    """

    def __init__(self, device: str = "auto"):
        super().__init__("qwen2vl", device)
        self.tokenizer = None

    def load(self):
        """Load Qwen2-VL-2B-Instruct model"""
        logger.info("Loading Qwen2-VL-2B-Instruct...")

        try:
            # Determine device and dtype
            if torch.cuda.is_available():
                dtype = torch.float16
                device_map = "auto"
                # Use flash_attention_2 for speed on Ampere+ GPUs
                try:
                    import flash_attn  # type: ignore  # noqa: F401
                    attn_implementation = "flash_attention_2"
                except ImportError:
                    attn_implementation = "eager"
                    logger.warning(
                        "flash_attn not available; using eager attention. "
                        "Install flash-attn to enable FlashAttention2 acceleration."
                    )
            elif torch.backends.mps.is_available():
                dtype = torch.bfloat16
                device_map = "mps"
                attn_implementation = "eager"  # MPS doesn't support flash attention
            else:
                dtype = torch.float32
                device_map = "cpu"
                attn_implementation = "eager"

            logger.info(f"Qwen2-VL using device: {device_map}, dtype: {dtype}, attn: {attn_implementation}")

            # Load model with optimizations
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation=attn_implementation
            )

            # Load processor with optimized image token settings
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                trust_remote_code=True,
                min_pixels=256*28*28,   # Reduce visual tokens for speed
                max_pixels=1280*28*28   # Balance quality and speed
            )

            self.is_ready = True
            logger.info("Qwen2-VL-2B-Instruct loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL-2B-Instruct: {e}")
            raise

    def _default_prompt(self, language: str) -> str:
        """Return language-aware default caption prompt."""
        defaults = {
            "en": "Describe this image briefly.",
            "ja": "この画像を簡潔に説明してください。",
            "zh": "简要描述这张图片。",
            "ko": "이 이미지를 간단히 설명해주세요."
        }
        return defaults.get(language, defaults["en"])

    def _detect_language(self, text: Optional[str]) -> Optional[str]:
        """Heuristic language detection based on character ranges."""
        if not text:
            return None

        if re.search(r"[一-龯\u3400-\u4dbf\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[ぁ-ゟ゠-ヿ]", text):
            return "ja"
        if re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", text):
            return "ko"
        if re.search(r"[A-Za-z]", text):
            return "en"
        return None

    def _generate_chat_response(
        self,
        image: Image.Image,
        prompt: str,
        response_language: str = "en",
        max_new_tokens: int = 128,
        min_new_tokens: int = 16,
        use_sampling: bool = False
    ) -> str:
        """Shared helper to run Qwen2-VL chat style inference."""
        try:
            message_text = prompt or self._default_prompt(response_language)

            language_instructions = {
                "ja": "日本語で回答してください。",
                "zh": "请用中文回答。",
                "ko": "한국어로 대답해 주세요.",
                "en": None,
            }
            instruction = language_instructions.get(response_language)
            if instruction and instruction not in message_text:
                message_text = f"{message_text}\n{instruction}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": message_text}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "use_cache": True,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 4,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "num_beams": 1,
            }

            if use_sampling:
                generate_kwargs.update({
                    "do_sample": True,
                    "top_p": 0.85,
                    "temperature": 0.8,
                    "top_k": 50,
                })
            else:
                generate_kwargs.update({
                    "do_sample": False,
                })

            output_ids = self.model.generate(
                **inputs,
                **generate_kwargs
            )

            generated = output_ids[:, inputs.input_ids.shape[-1]:]

            decoded_candidates: List[str] = []

            if generated.size(-1) > 0:
                decoded_candidates.append(
                    self.processor.batch_decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                )

            # Always fall back to decoding the full sequence in case slicing removed everything
            decoded_candidates.append(
                self.processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
            )

            for raw_response in decoded_candidates:
                cleaned = self._clean_caption(raw_response)
                if cleaned:
                    return cleaned

            logger.warning("Qwen2-VL returned empty caption after decoding attempts.")
            return "Unable to describe the image."

        except Exception as e:
            logger.error(f"Error generating response with Qwen2-VL: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "Error processing frame"

    @staticmethod
    def _clean_caption(text: str) -> str:
        """Normalize model output and remove meaningless punctuation-heavy strings."""
        if not isinstance(text, str):
            return ""

        cleaned = text.strip()
        if not cleaned:
            return ""

        # Collapse whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Reduce excessive punctuation runs
        cleaned = re.sub(r'([!?。！？])\1{2,}', r'\1\1', cleaned)
        cleaned = re.sub(r'([,.，。])\1{1,}', r'\1', cleaned)
        cleaned = cleaned.strip()

        # Discard strings that are mostly punctuation
        if cleaned and Qwen2VL._is_mostly_punctuation(cleaned):
            return ""

        # Require at least one alphabetic character (covers Latin, CJK, etc.) to avoid pure symbol/number outputs
        if cleaned and not any(ch.isalpha() for ch in cleaned):
            return ""

        return cleaned

    @staticmethod
    def _is_mostly_punctuation(text: str, threshold: float = 0.6) -> bool:
        if not text:
            return True
        punct_count = sum(1 for ch in text if ch in string.punctuation or ch in "？！。；，、")
        return (punct_count / len(text)) >= threshold

    @torch.inference_mode()
    def caption(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        language: str = "en",
        response_length: str = "medium"
    ) -> str:
        """Generate descriptive caption for an image."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        length_map = {
            "short": {"max": 64, "min": 8, "sampling": True},
            "medium": {"max": 128, "min": 16, "sampling": False},
            "long": {"max": 256, "min": 24, "sampling": False},
        }
        cfg = length_map.get(response_length, length_map["medium"])

        detected_language = self._detect_language(prompt)
        target_language = detected_language or language or "en"

        attempts = [
            {
                "prompt": prompt,
                "response_language": target_language,
                "max_new_tokens": cfg["max"],
                "min_new_tokens": cfg["min"],
                "use_sampling": cfg["sampling"],
            },
            {
                "prompt": None,
                "response_language": target_language,
                "max_new_tokens": length_map["medium"]["max"],
                "min_new_tokens": length_map["medium"]["min"],
                "use_sampling": True,
            },
            {
                "prompt": prompt or self._default_prompt("en"),
                "response_language": "en",
                "max_new_tokens": length_map["medium"]["max"],
                "min_new_tokens": length_map["medium"]["min"],
                "use_sampling": True,
            },
        ]

        for attempt_idx, attempt in enumerate(attempts, start=1):
            result = self._generate_chat_response(
                image,
                attempt["prompt"],
                response_language=attempt["response_language"],
                max_new_tokens=attempt["max_new_tokens"],
                min_new_tokens=attempt["min_new_tokens"],
                use_sampling=attempt["use_sampling"]
            )
            if result and result.strip():
                if attempt_idx > 1:
                    logger.debug(f"Qwen2-VL caption succeeded on retry attempt {attempt_idx}.")
                return result

        logger.warning("Qwen2-VL caption failed to produce meaningful text after retries.")
        return "Unable to describe the image."

    @torch.inference_mode()
    def query(
        self,
        image: Image.Image,
        question: str,
        language: str = "en",
        response_length: str = "medium"
    ) -> str:
        """Answer a user question about the image."""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        question_text = question.strip() if isinstance(question, str) else ""
        if not question_text:
            return "Please provide a question to ask about the image."

        length_map = {
            "short": {"max": 72, "min": 8, "sampling": False},
            "medium": {"max": 144, "min": 16, "sampling": False},
            "long": {"max": 256, "min": 24, "sampling": False},
        }
        cfg = length_map.get(response_length, length_map["medium"])

        detected_language = self._detect_language(question_text)
        target_language = detected_language or language or "en"

        attempts = [
            {
                "prompt": question_text,
                "response_language": target_language,
                "max_new_tokens": cfg["max"],
                "min_new_tokens": cfg["min"],
                "use_sampling": cfg["sampling"],
            },
            {
                "prompt": question_text,
                "response_language": target_language,
                "max_new_tokens": length_map["medium"]["max"],
                "min_new_tokens": length_map["medium"]["min"],
                "use_sampling": True,
            },
        ]

        if target_language != "en":
            attempts.append(
                {
                    "prompt": question_text,
                    "response_language": "en",
                    "max_new_tokens": length_map["medium"]["max"],
                    "min_new_tokens": length_map["medium"]["min"],
                    "use_sampling": True,
                }
            )

        for attempt_idx, attempt in enumerate(attempts, start=1):
            answer = self._generate_chat_response(
                image,
                attempt["prompt"],
                response_language=attempt["response_language"],
                max_new_tokens=attempt["max_new_tokens"],
                min_new_tokens=attempt["min_new_tokens"],
                use_sampling=attempt["use_sampling"]
            )
            if answer and answer.strip():
                if attempt_idx > 1:
                    logger.debug(f"Qwen2-VL query succeeded on retry attempt {attempt_idx}.")
                return answer

        logger.warning("Qwen2-VL query failed to produce meaningful text after retries.")
        return "Unable to answer right now."


class VLMProcessor:
    """
    High-level VLM processor with model switching and optimization
    Supports: Qwen2-VL-2B (multilingual), SmolVLM-500M, Moondream 2 (feature-rich)
    """

    def __init__(self, model_name: str = "qwen2vl", language: str = "en"):
        self.current_model_name = model_name
        self.model: Optional[VLMModel] = None
        self.language = language  # User's preferred language
        self.base_target_size = (128, 128)  # Default resize for lightweight models
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load specified model"""
        # Unload current model if exists
        if self.model is not None:
            self.model.unload()

        # Load new model
        if model_name == "qwen2vl":
            self.model = Qwen2VL()
            self.base_target_size = (384, 384)  # Higher resolution for Qwen2-VL
        elif model_name == "smolvlm":
            self.model = SmolVLM()
            self.base_target_size = (128, 128)
        elif model_name == "mobilevlm":
            self.model = MobileVLM()
            self.base_target_size = (160, 160)
        elif model_name == "moondream":
            self.model = Moondream()
            self.base_target_size = (384, 384)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.load()
        self.current_model_name = model_name

    async def switch_model(self, model_name: str):
        """Switch to different model (async)"""
        logger.info(f"Switching model to {model_name}")

        # Run model loading in thread pool to avoid blocking
        await asyncio.to_thread(self._load_model, model_name)

        logger.info(f"Model switched to {model_name}")

    def preprocess_frame(
        self,
        frame: np.ndarray,
        mode: Literal["caption", "query", "detection", "point", "mask"] = "caption"
    ) -> Tuple[Image.Image, float, float]:
        """
        Preprocess video frame for VLM
        - Resize to optimal size (model and mode dependent)
        - Convert to PIL Image
        - Return scale factors to map predictions back to original resolution
        """
        import cv2

        orig_height, orig_width = frame.shape[:2]

        if isinstance(self.model, Moondream):
            max_side = 512 if mode in ("detection", "point", "mask") else 384
            if max(orig_width, orig_height) > max_side:
                scale = max_side / max(orig_width, orig_height)
                new_width = max(1, int(round(orig_width * scale)))
                new_height = max(1, int(round(orig_height * scale)))
            else:
                new_width, new_height = orig_width, orig_height
        else:
            new_width, new_height = self.base_target_size

        if new_width != orig_width or new_height != orig_height:
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized = frame

        scale_x = orig_width / new_width if new_width else 1.0
        scale_y = orig_height / new_height if new_height else 1.0

        image = Image.fromarray(resized)

        return image, scale_x, scale_y

    async def process_frame(
        self,
        frame: np.ndarray,
        mode: Literal["caption", "query", "detection", "point", "mask"] = "caption",
        user_input: Optional[str] = None,
        click_coords: Optional[Tuple[int, int]] = None,
        response_length: str = "medium"
    ) -> Dict[str, Any]:
        """
        Process video frame and generate output
        Args:
            frame: numpy array (H, W, 3) in RGB
            mode: processing mode (caption/query/detect/point/mask)
            user_input: User's question for query mode or object description for detect/point/mask
            click_coords: (x, y) coordinates for point mode with Moondream
        Returns:
            Dictionary with result and metadata
        """
        # Preprocess frame
        image, scale_x, scale_y = self.preprocess_frame(frame, mode)

        # Handle different models and modes
        if isinstance(self.model, Moondream):
            # Moondream supports all modes
            result = await self._process_moondream(
                image,
                mode,
                user_input,
                click_coords,
                response_length,
                scale_x,
                scale_y
            )
        else:
            # SmolVLM/MobileVLM support caption and custom queries
            result = await self._process_simple_vlm(image, mode, user_input, response_length)

        return result

    async def _process_simple_vlm(
        self,
        image: Image.Image,
        mode: str,
        user_input: Optional[str] = None,
        response_length: str = "medium"
    ) -> Dict[str, Any]:
        """Process with Qwen2VL, SmolVLM or MobileVLM (supports custom queries)"""

        unsupported_modes = {"detection", "point", "mask"}
        if mode in unsupported_modes and isinstance(self.model, Qwen2VL):
            message = "Qwen2-VL currently supports caption and query modes only."
            return {
                "type": "caption",
                "caption": message,
                "mode": "caption",
                "model": self.current_model_name,
                "language": self.language
            }

        # Qwen2VL supports caption + query modes with multilingual capability
        if isinstance(self.model, Qwen2VL):
            prompt = user_input if isinstance(user_input, str) and user_input.strip() else None
            target_language = (
                self.model._detect_language(prompt)
                or (self.language if isinstance(self.language, str) and self.language else None)
                or "en"
            )

            if mode == "query":
                if not prompt:
                    return {
                        "type": "caption",
                        "caption": "Please enter a question to ask about the image.",
                        "mode": mode,
                        "model": self.current_model_name,
                        "language": self.language
                    }

                answer = await asyncio.to_thread(
                    self.model.query,
                    image,
                    prompt,
                    target_language,
                    response_length
                )

                return {
                    "type": "caption",
                    "caption": answer,
                    "question": prompt,
                    "mode": mode,
                    "model": self.current_model_name,
                    "language": target_language
                }

            caption = await asyncio.to_thread(
                self.model.caption,
                image,
                prompt,
                target_language,
                response_length
            )

            return {
                "type": "caption",
                "caption": caption,
                "mode": mode,
                "model": self.current_model_name,
                "language": target_language
            }

        # SmolVLM/MobileVLM require explicit user query
        length_configs = {
            "short": {"prompt": "Describe this scene in under 10 words.", "max_tokens": 16},
            "medium": {"prompt": "Describe this scene briefly.", "max_tokens": 32},
            "long": {"prompt": "Describe this scene in detail.", "max_tokens": 56},
        }
        cfg = length_configs.get(response_length, length_configs["medium"])
        prompt_text = user_input.strip() if isinstance(user_input, str) and user_input.strip() else cfg["prompt"]

        # Run inference in thread pool (non-blocking)
        caption = await asyncio.to_thread(
            self.model.caption,
            image,
            prompt_text,
            cfg["max_tokens"]
        )

        return {
            "type": "caption",
            "caption": caption,
            "mode": mode,
            "model": self.current_model_name
        }

    @staticmethod
    def _scale_bbox(bbox: Any, scale_x: float, scale_y: float) -> Optional[List[int]]:
        if bbox is None:
            return None

        if isinstance(bbox, dict):
            x1 = bbox.get('x_min') or bbox.get('left') or bbox.get('x1')
            y1 = bbox.get('y_min') or bbox.get('top') or bbox.get('y1')
            x2 = bbox.get('x_max') or bbox.get('right') or bbox.get('x2')
            y2 = bbox.get('y_max') or bbox.get('bottom') or bbox.get('y2')
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
        else:
            return None

        if None in (x1, y1, x2, y2):
            return None

        return [
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y))
        ]

    @staticmethod
    def _scale_points(points: List[Tuple[int, int]], scale_x: float, scale_y: float) -> List[List[int]]:
        scaled: List[List[int]] = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = point[0], point[1]
                scaled.append([
                    int(round(x * scale_x)),
                    int(round(y * scale_y))
                ])
        return scaled

    async def _process_moondream(
        self,
        image: Image.Image,
        mode: str,
        user_input: Optional[str] = None,
        click_coords: Optional[Tuple[int, int]] = None,
        response_length: str = "medium",
        scale_x: float = 1.0,
        scale_y: float = 1.0
    ) -> Dict[str, Any]:
        """Process with Moondream (all modes supported)"""
        if mode == "caption":
            length_map = {
                "short": "short",
                "medium": "normal",
                "long": "long"
            }
            moondream_length = length_map.get(response_length, "normal")

            caption = await asyncio.to_thread(
                self.model.caption,
                image,
                length=moondream_length
            )
            return {
                "type": "caption",
                "caption": caption,  # Changed from "data" to "caption" for consistency
                "mode": mode,
                "model": self.current_model_name
            }

        elif mode == "query":
            if not user_input:
                return {
                    "type": "caption",
                    "caption": "Please enter a question",
                    "mode": mode,
                    "model": self.current_model_name
                }

            answer = await asyncio.to_thread(
                self.model.query,
                image,
                user_input,
                reasoning=True
            )
            return {
                "type": "caption",
                "caption": answer,  # Changed from "data" to "caption" for consistency
                "question": user_input,
                "mode": mode,
                "model": self.current_model_name
            }

        elif mode == "detection":
            raw_query = user_input or ""
            if isinstance(raw_query, str):
                targets = [t.strip() for t in raw_query.split(',') if t.strip()]
            else:
                targets = []

            if not targets:
                targets = ["all objects"]

            detections: List[Dict[str, Any]] = []

            for target in targets:
                raw_results = await asyncio.to_thread(
                    self.model.detect,
                    image,
                    target
                )

                if not raw_results:
                    continue

                for det in raw_results:
                    scaled_det: Dict[str, Any] = dict(det) if isinstance(det, dict) else {}
                    bbox = scaled_det.get('bbox')
                    scaled_bbox = self._scale_bbox(bbox, scale_x, scale_y)
                    if not scaled_bbox:
                        continue

                    scaled_det['bbox'] = scaled_bbox

                    confidence = (
                        scaled_det.get('confidence')
                        or scaled_det.get('score')
                        or scaled_det.get('prob')
                    )
                    if confidence is not None:
                        try:
                            scaled_det['confidence'] = float(confidence)
                        except (TypeError, ValueError):
                            scaled_det['confidence'] = None
                    else:
                        scaled_det['confidence'] = None

                    label = scaled_det.get('label')
                    if not label or str(label).lower() == 'object':
                        label = target
                    scaled_det['label'] = str(label)

                    detections.append(scaled_det)

            # Format detections as text for display
            if detections:
                labels = [f"\"{det.get('label', 'object')}\"" for det in detections]
                count = len(detections)
                object_word = "object" if count == 1 else "objects"
                detection_text = f"Detected {count} {object_word}: " + ", ".join(labels)
            else:
                detection_text = "No objects detected"

            return {
                "type": "caption",
                "caption": detection_text,
                "detections": detections,
                "object": raw_query,
                "mode": mode,
                "model": self.current_model_name
            }

        elif mode == "point":
            if not user_input:
                return {
                    "type": "caption",
                    "caption": "Please specify an object to locate",
                    "mode": mode,
                    "model": self.current_model_name
                }

            raw_points = await asyncio.to_thread(
                self.model.point,
                image,
                user_input
            )

            scaled_points = self._scale_points(raw_points or [], scale_x, scale_y)
            fallback_used = False

            if not scaled_points:
                # Fallback: try object detection and compute point centers
                raw_detections = await asyncio.to_thread(
                    self.model.detect,
                    image,
                    user_input
                )
                if raw_detections:
                    scaled_points = []
                    for det in raw_detections:
                        bbox = det.get('bbox') if isinstance(det, dict) else None
                        scaled_bbox = self._scale_bbox(bbox, scale_x, scale_y)
                        if scaled_bbox:
                            x1, y1, x2, y2 = scaled_bbox
                            center_x = int(round((x1 + x2) / 2))
                            center_y = int(round((y1 + y2) / 2))
                            scaled_points.append([center_x, center_y])
                    if scaled_points:
                        fallback_used = True

            # Format points as text for display
            if scaled_points:
                summary_label = "Estimated" if fallback_used else "Found"
                point_text = (
                    f"{summary_label} {len(scaled_points)} point(s) for '{user_input}': "
                    + ", ".join([f"({x}, {y})" for x, y in scaled_points])
                )
            else:
                point_text = f"Could not locate '{user_input}'"

            return {
                "type": "caption",
                "caption": point_text,
                "points": scaled_points,
                "object": user_input,
                "mode": mode,
                "model": self.current_model_name,
                "fallback_used": fallback_used
            }

        elif mode == "mask":
            # Mask mode: uses detection to find objects to mask
            if not user_input:
                return {
                    "type": "caption",
                    "caption": "Please specify objects to mask",
                    "mode": mode,
                    "model": self.current_model_name
                }

            raw_query = user_input
            if isinstance(raw_query, str):
                targets = [t.strip() for t in raw_query.split(',') if t.strip()]
            else:
                targets = []

            if not targets:
                targets = ["all objects"]

            detections: List[Dict[str, Any]] = []

            for target in targets:
                raw_results = await asyncio.to_thread(
                    self.model.detect,
                    image,
                    target
                )

                if not raw_results:
                    continue

                for det in raw_results:
                    scaled_det: Dict[str, Any] = dict(det) if isinstance(det, dict) else {}
                    bbox = scaled_det.get('bbox')
                    scaled_bbox = self._scale_bbox(bbox, scale_x, scale_y)
                    if not scaled_bbox:
                        continue

                    scaled_det['bbox'] = scaled_bbox
                    scaled_det['confidence'] = None

                    label = scaled_det.get('label')
                    if not label or str(label).lower() == 'object':
                        label = target
                    scaled_det['label'] = str(label)

                    detections.append(scaled_det)

            # Format mask message for display
            if detections:
                labels = [f"\"{det.get('label', 'object')}\"" for det in detections]
                count = len(detections)
                object_word = "object" if count == 1 else "objects"
                mask_text = f"Masking {count} {object_word}: " + ", ".join(labels)
            else:
                mask_text = "No objects to mask"

            return {
                "type": "caption",
                "caption": mask_text,
                "detections": detections,
                "object": raw_query,
                "mode": mode,
                "model": self.current_model_name
            }

        else:
            # Fallback to caption
            return await self._process_simple_vlm(image, "caption", response_length=response_length)

    async def warmup(self, num_iterations: int = 2):
        """Warmup model with dummy inputs"""
        logger.info(f"Warming up {self.current_model_name}...")

        dummy_image = Image.new('RGB', self.base_target_size, color='white')

        for i in range(num_iterations):
            await asyncio.to_thread(
                self.model.caption,
                dummy_image,
                "Warmup."
            )

        logger.info("Warmup complete!")

    @property
    def current_model(self) -> str:
        """Get current model name"""
        return self.current_model_name

    def get_stats(self) -> dict:
        """Get model statistics"""
        return {
            "model": self.current_model_name,
            "ready": self.model.is_ready if self.model else False,
            "device": str(self.model.model.device) if self.model and self.model.model else "unknown",
            "target_size": self.base_target_size,
            "supports_multimodal": isinstance(self.model, Moondream)
        }

    def get_supported_modes(self) -> List[str]:
        """Get list of supported modes for current model"""
        if isinstance(self.model, Moondream):
            return ["caption", "query", "detection", "point", "mask"]
        if isinstance(self.model, Qwen2VL):
            return ["caption", "query"]
        else:
            return ["caption"]


# Standalone testing
async def test_vlm():
    """Test VLM processor"""
    import cv2

    print("Testing VLM Processor...")

    # Initialize
    processor = VLMProcessor(model_name="smolvlm")

    # Warmup
    await processor.warmup()

    # Test with dummy frame
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Process
    start = time.time()
    caption = await processor.process_frame(dummy_frame, mode="caption")
    latency = (time.time() - start) * 1000

    print(f"Caption: {caption}")
    print(f"Latency: {latency:.2f}ms")

    # Get stats
    stats = processor.get_stats()
    print(f"Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_vlm())
