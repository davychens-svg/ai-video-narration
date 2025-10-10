"""
High-Performance VLM Processor
Optimized for speed with small models (SmolVLM, MobileVLM)
Plus Moondream 3.0 with full feature set (caption, query, detect, point)
"""

import asyncio
import logging
import time
from typing import Optional, Literal, List, Dict, Tuple, Any
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

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
        if self.model is not None:
            del self.model
            del self.processor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
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
    def caption(self, image: Image.Image, prompt: str = "Describe briefly.") -> str:
        """Generate caption with ultra-fast settings"""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        try:
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
                max_new_tokens=10,      # Ultra-fast: 10 tokens max
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
    def caption(self, image: Image.Image, prompt: str = "Describe.") -> str:
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
                max_new_tokens=20,
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
            caption = self.model.answer_question(
                enc_image,
                "Describe this image in detail.",
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
            # API returns: {"objects": [{"bbox": [x1, y1, x2, y2], "label": "object"}]}
            if hasattr(self.model, 'detect'):
                result = self.model.detect(image, object_description)

                # Extract objects from result dictionary
                if isinstance(result, dict) and "objects" in result:
                    objects = result["objects"]
                    logger.info(f"Detected {len(objects)} objects: {objects}")
                    return objects if isinstance(objects, list) else []

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
            # API returns: {"points": [[x1, y1], [x2, y2], ...]}
            if hasattr(self.model, 'point'):
                result = self.model.point(image, object_description)

                # Extract points from result dictionary
                if isinstance(result, dict) and "points" in result:
                    points = result["points"]
                    logger.info(f"Found {len(points)} points: {points}")

                    # Convert to list of tuples
                    formatted_points = []
                    for p in points:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            formatted_points.append((int(p[0]), int(p[1])))
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


class VLMProcessor:
    """
    High-level VLM processor with model switching and optimization
    Supports: SmolVLM-500M (primary), Moondream 2 (feature-rich)
    """

    def __init__(self, model_name: str = "smolvlm"):
        self.current_model_name = model_name
        self.model: Optional[VLMModel] = None
        self.target_size = (128, 128)  # Ultra-fast: smallest usable size for real-time
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load specified model"""
        # Unload current model if exists
        if self.model is not None:
            self.model.unload()

        # Load new model
        if model_name == "smolvlm":
            self.model = SmolVLM()
        elif model_name == "mobilevlm":
            self.model = MobileVLM()
        elif model_name == "moondream":
            self.model = Moondream()
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

    def preprocess_frame(self, frame: np.ndarray) -> Image.Image:
        """
        Preprocess video frame for VLM
        - Resize to optimal size
        - Convert to PIL Image
        """
        # Resize to target size (fast)
        import cv2
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

        # Convert to PIL
        image = Image.fromarray(resized)

        return image

    async def process_frame(
        self,
        frame: np.ndarray,
        mode: Literal["caption", "query", "detect", "point"] = "caption",
        user_input: Optional[str] = None,
        click_coords: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Process video frame and generate output
        Args:
            frame: numpy array (H, W, 3) in RGB
            mode: processing mode (caption/query/detect/point)
            user_input: User's question for query mode or object description for detect/point
            click_coords: (x, y) coordinates for point mode with Moondream
        Returns:
            Dictionary with result and metadata
        """
        # Preprocess frame
        image = self.preprocess_frame(frame)

        # Handle different models and modes
        if isinstance(self.model, Moondream):
            # Moondream supports all modes
            result = await self._process_moondream(image, mode, user_input, click_coords)
        else:
            # SmolVLM/MobileVLM support caption and custom queries
            result = await self._process_simple_vlm(image, mode, user_input)

        return result

    async def _process_simple_vlm(
        self,
        image: Image.Image,
        mode: str,
        user_input: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process with SmolVLM or MobileVLM (supports custom queries only)"""
        # SmolVLM requires explicit user query - no automatic captioning
        if not user_input or not user_input.strip():
            return {
                "type": "caption",
                "caption": "Please enter a query using the 'Send Query' button",
                "mode": mode,
                "model": self.current_model_name
            }

        # Run inference in thread pool (non-blocking)
        caption = await asyncio.to_thread(self.model.caption, image, user_input)

        return {
            "type": "caption",
            "caption": caption,
            "mode": mode,
            "model": self.current_model_name
        }

    async def _process_moondream(
        self,
        image: Image.Image,
        mode: str,
        user_input: Optional[str] = None,
        click_coords: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """Process with Moondream (all modes supported)"""
        if mode == "caption":
            caption = await asyncio.to_thread(
                self.model.caption,
                image,
                length="normal"
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
            if not user_input:
                user_input = "all objects"  # Default to detecting all objects

            detections = await asyncio.to_thread(
                self.model.detect,
                image,
                user_input
            )
            # Format detections as text for display
            if detections:
                detection_text = f"Detected {len(detections)} objects: " + ", ".join([d.get('label', 'object') for d in detections])
            else:
                detection_text = "No objects detected"

            return {
                "type": "caption",
                "caption": detection_text,
                "detections": detections,
                "object": user_input,
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

            points = await asyncio.to_thread(
                self.model.point,
                image,
                user_input
            )
            # Format points as text for display
            if points:
                point_text = f"Found {len(points)} point(s) for '{user_input}': " + ", ".join([f"({x}, {y})" for x, y in points])
            else:
                point_text = f"Could not locate '{user_input}'"

            return {
                "type": "caption",
                "caption": point_text,
                "points": points,
                "object": user_input,
                "mode": mode,
                "model": self.current_model_name
            }

        else:
            # Fallback to caption
            return await self._process_simple_vlm(image, "caption")

    async def warmup(self, num_iterations: int = 2):
        """Warmup model with dummy inputs"""
        logger.info(f"Warming up {self.current_model_name}...")

        dummy_image = Image.new('RGB', self.target_size, color='white')

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
            "target_size": self.target_size,
            "supports_multimodal": isinstance(self.model, Moondream)
        }

    def get_supported_modes(self) -> List[str]:
        """Get list of supported modes for current model"""
        if isinstance(self.model, Moondream):
            return ["caption", "query", "detect", "point"]
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
