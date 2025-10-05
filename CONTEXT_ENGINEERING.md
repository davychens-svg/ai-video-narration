# Context Engineering Guide for Inerbee Demo

## Overview
This document outlines the context engineering strategies and best practices for building a **high-performance** real-time video narrator for the Inerbee demo. Performance is critical - we prioritize speed, low latency, and minimal resource usage through small models and optimized context strategies.

## Performance-First Philosophy 🚀

**Target Metrics for Inerbee Demo:**
- **Latency**: <100ms per frame (target: 50-70ms)
- **Throughput**: 15-30 FPS
- **Memory**: <2GB VRAM for models
- **CPU**: Minimal usage, offload to GPU where possible

**Model Selection Priority:**
1. **SmolVLM** (primary) - 2B params, ultra-fast, caption only
2. **Moondream 3.0** (feature-rich) - Advanced capabilities with caption, query, detect, point
3. **MobileVLM** (backup) - 1.4B params, mobile-optimized

## Table of Contents
1. [Performance Optimization Strategies](#performance-optimization-strategies)
2. [Small Model Context Engineering](#small-model-context-engineering)
3. [Minimal Prompt Patterns](#minimal-prompt-patterns)
4. [Frame Batching & Skipping](#frame-batching--skipping)
5. [Model-Specific Optimizations](#model-specific-optimizations)
6. [Moondream 3.0 Context Engineering](#moondream-30-context-engineering)
7. [Production Deployment](#production-deployment)

---

## Performance Optimization Strategies

### 1. Aggressive Frame Skipping
```python
# Process every Nth frame based on load
FRAME_SKIP_CONFIG = {
    "low_load": 1,      # Every frame (30 FPS → 30 inferences/sec)
    "medium_load": 2,   # Every 2nd frame (30 FPS → 15 inferences/sec)
    "high_load": 3,     # Every 3rd frame (30 FPS → 10 inferences/sec)
}
```

### 2. Resolution Optimization
```python
# Downscale frames aggressively for small models
TARGET_RESOLUTIONS = {
    "smolvlm": (384, 384),      # SmolVLM optimal
    "mobilevlm": (336, 336),    # MobileVLM optimal
    "tinyllava": (384, 384),    # TinyLLaVA optimal
}
```

### 3. Model Quantization
- Use INT8 quantization (2-3x speedup, minimal quality loss)
- BFloat16 for Apple Silicon (M-series chips)
- INT4 for extreme edge cases

### 4. Batch Processing
```python
# Batch multiple frames when possible
BATCH_SIZE = 1  # Real-time = batch of 1, but keep pipeline ready
```

---

## Small Model Context Engineering

### Core Principle: Minimal Tokens = Maximum Speed

**Token Budget:**
- System prompt: <50 tokens
- User prompt: <20 tokens
- Total context: <100 tokens
- Image tokens: ~256 (fixed by model)

### Ultra-Minimal Prompts for SmolVLM

```python
# PRIMARY PROMPTS - Use these for Inerbee demo
PROMPTS = {
    "caption": "Describe in 10 words.",
    "detail": "What's happening?",
    "action": "Main action?",
    "objects": "List objects.",
}

# NO system prompts - they add latency
# NO few-shot examples - waste tokens
# NO chain-of-thought - too slow
```

---

## Minimal Prompt Patterns

### Pattern 1: Direct Command (Fastest)
```python
prompt = "Narrate."
# Expected output: "Person walking dog in park"
```

### Pattern 2: Single Question (Fast)
```python
prompt = "What's visible?"
# Expected output: "Kitchen with stove and utensils"
```

### Pattern 3: Constrained Format (Fast + Structured)
```python
prompt = "Scene: [location]. Action: [verb]."
# Expected output: "Scene: office. Action: typing."
```

### ❌ Avoid These Patterns (Too Slow)
```python
# DON'T: Long system prompts
"You are an expert video analyst with deep knowledge..."

# DON''t: Multi-step instructions
"First identify objects, then describe actions, finally..."

# DON'T: Few-shot examples
"Example 1: ... Example 2: ... Now describe:"
```

---

## Frame Batching & Skipping

### Smart Frame Selection
```python
class FrameSelector:
    def __init__(self):
        self.last_processed = None
        self.skip_count = 0
        self.target_skip = 2  # Process every 2nd frame

    def should_process(self, frame):
        # Motion detection - only process if scene changed
        if self.last_processed is None:
            return True

        diff = cv2.absdiff(frame, self.last_processed)
        change_percent = np.mean(diff) / 255.0

        # If <5% change and we recently processed, skip
        if change_percent < 0.05 and self.skip_count < self.target_skip:
            self.skip_count += 1
            return False

        self.skip_count = 0
        self.last_processed = frame
        return True
```

### Temporal Smoothing (Reuse Results)
```python
class CaptionCache:
    def __init__(self, ttl=1.0):  # 1 second TTL
        self.cache = {}
        self.ttl = ttl

    def get_or_process(self, frame_hash, process_fn):
        now = time.time()

        if frame_hash in self.cache:
            caption, timestamp = self.cache[frame_hash]
            if now - timestamp < self.ttl:
                return caption  # Reuse recent caption

        caption = process_fn()
        self.cache[frame_hash] = (caption, now)
        return caption
```

---

## Model-Specific Optimizations

### SmolVLM (Primary Model)
```python
# Hugging Face: HuggingFaceTB/SmolVLM-Instruct
# Size: 2B params (~4GB storage, ~2GB VRAM)

from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

class SmolVLM:
    def __init__(self):
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=torch.bfloat16,  # Fast on Apple Silicon
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct"
        )

    @torch.inference_mode()
    def caption(self, image, prompt="Describe."):
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        # Ultra-fast generation config
        output = self.model.generate(
            **inputs,
            max_new_tokens=20,      # Short outputs only
            do_sample=False,        # Greedy = faster
            num_beams=1,           # No beam search
            temperature=1.0,
        )

        return self.processor.decode(output[0], skip_special_tokens=True)
```

### MobileVLM (Backup Model)
```python
# Even smaller, for extreme performance needs
# Size: 1.4B params (~3GB storage, ~1.5GB VRAM)

class MobileVLM:
    def __init__(self):
        self.model = AutoModelForVision2Seq.from_pretrained(
            "mtgv/MobileVLM-1.7B",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # Same interface as SmolVLM
```

---

## Production Deployment

### Hardware Recommendations

**Dev (macOS):**
- Apple Silicon M1/M2/M3
- 16GB+ RAM
- Use Metal acceleration via MPS

**Production (Ubuntu 22.04):**
- NVIDIA RTX 3060+ (12GB VRAM)
- 32GB RAM
- CUDA 11.8+

### Deployment Optimizations

```python
# 1. Model warmup
async def warmup_model(vlm, num_iters=5):
    dummy_image = Image.new('RGB', (384, 384))
    for _ in range(num_iters):
        vlm.caption(dummy_image, "Warmup.")

# 2. Connection pooling
MAX_WEBSOCKET_CONNECTIONS = 100

# 3. Async processing
async def process_stream():
    while True:
        frame = await frame_queue.get()
        caption = await asyncio.to_thread(vlm.caption, frame)
        await broadcast(caption)
```

### Performance Monitoring

```python
import time
from collections import deque

class PerformanceMonitor:
    def __init__(self, window=100):
        self.latencies = deque(maxlen=window)

    def record(self, latency_ms):
        self.latencies.append(latency_ms)

    def get_stats(self):
        if not self.latencies:
            return {}
        return {
            "avg_ms": sum(self.latencies) / len(self.latencies),
            "p50_ms": sorted(self.latencies)[len(self.latencies)//2],
            "p95_ms": sorted(self.latencies)[int(len(self.latencies)*0.95)],
            "fps": 1000.0 / (sum(self.latencies) / len(self.latencies))
        }
```

---

## Moondream 3.0 Context Engineering

### Overview
Moondream 3.0 is a feature-rich VLM that supports multiple modes beyond simple captioning. It requires more resources (24GB+ VRAM) but offers advanced capabilities.

### Supported Modes

#### 1. Caption Mode
Generate natural language descriptions of video frames.

```python
# Moondream caption with configurable length
result = model.caption(
    image=frame,
    length="normal",  # "short", "normal", or "long"
    settings={
        "temperature": 0.5,
        "max_tokens": 768,
        "top_p": 0.3
    }
)
```

**Context Engineering Tips:**
- Use `length="short"` for faster processing
- `length="normal"` for balanced detail
- `length="long"` for comprehensive descriptions (slower)
- Lower temperature (0.3-0.5) for consistent, factual descriptions
- Higher temperature (0.7-0.9) for creative narrations

#### 2. Query Mode
Answer specific questions about the video content.

```python
result = model.query(
    image=frame,
    question="What color is the person's shirt?",
    reasoning=True,  # Include reasoning process
    settings={
        "temperature": 0.5,
        "max_tokens": 768,
        "top_p": 0.3
    }
)
```

**Effective Question Patterns:**
```python
# Specific attribute queries
"What color is the [object]?"
"How many [objects] are visible?"
"Where is the [object] located?"

# Spatial relationships
"Is the person standing or sitting?"
"What is to the left of the [object]?"
"Which object is closest to the camera?"

# Scene understanding
"What activity is happening?"
"What time of day is it?"
"Is this indoors or outdoors?"

# Comparative queries
"Is the [object] larger than the [object]?"
"Which [object] is brighter?"
```

**❌ Avoid:**
- Overly complex or multi-part questions
- Subjective or opinion-based questions
- Questions about things not in the current frame
- Temporal questions ("What happened before?")

#### 3. Detect Mode
Find and localize objects in the scene with bounding boxes.

```python
detections = model.detect(
    image=frame,
    object="person",  # Object description
    settings={"max_objects": 150}
)

# Returns: [
#     {"bbox": [x1, y1, x2, y2], "label": "person", "confidence": 0.95},
#     {"bbox": [x1, y1, x2, y2], "label": "person", "confidence": 0.88},
# ]
```

**Object Description Best Practices:**
```python
# Specific objects
"person", "car", "phone", "laptop", "bottle"

# Categories
"all objects"  # Detect everything
"vehicles"  # Cars, trucks, bikes
"electronics"  # Phones, laptops, TVs

# Visual attributes
"red car"
"person wearing glasses"
"open laptop"
```

**Performance Tips:**
- More specific descriptions = faster, more accurate
- Reduce `max_objects` for speed (default: 150)
- Use detect mode sparingly in real-time (higher latency)

#### 4. Point Mode
Find coordinate locations of specified objects.

```python
points = model.point(
    image=frame,
    object="face",
    settings={"max_objects": 150}
)

# Returns: [(x1, y1), (x2, y2), ...]
```

**Use Cases:**
- Click-to-identify interactions
- Object tracking
- Attention mapping
- UI overlays (crosshairs, markers)

**Object Descriptions:**
```python
# Body parts
"face", "hand", "eyes", "nose"

# Functional points
"door handle", "button", "screen"

# Interactive elements
"clickable area", "text input", "icon"
```

### Multi-Mode Context Strategies

#### Sequential Processing
Use different modes in sequence for rich understanding:

```python
# 1. Get overall caption
caption = model.caption(image, length="short")

# 2. Detect specific objects
people = model.detect(image, "person")

# 3. Query specific details
if people:
    answer = model.query(image, "What is the person doing?")
```

#### Adaptive Mode Selection
Choose mode based on user intent:

```python
def process_with_moondream(frame, user_input=None, mode="caption"):
    if mode == "caption":
        # Auto-narration
        return model.caption(frame, length="normal")

    elif mode == "query" and user_input:
        # User asked a question
        return model.query(frame, question=user_input)

    elif mode == "detect":
        # Looking for specific objects
        obj = user_input or "all objects"
        return model.detect(frame, object=obj)

    elif mode == "point":
        # Locating objects
        return model.point(frame, object=user_input)
```

### Performance Considerations

**Latency Comparison (NVIDIA RTX 3090):**
- Caption (short): ~80ms
- Caption (normal): ~150ms
- Caption (long): ~250ms
- Query: ~180-300ms (depends on question complexity)
- Detect: ~200-400ms (depends on max_objects)
- Point: ~150-250ms

**Optimization for Real-time:**
1. Use caption mode for continuous narration
2. Switch to query/detect/point only when user requests
3. Cache recent results (1-2 seconds)
4. Use shorter captions in caption mode
5. Limit max_objects in detect/point modes

### Context Settings Tuning

#### For Speed (Real-time Priority)
```python
FAST_SETTINGS = {
    "caption": {
        "temperature": 0.3,
        "max_tokens": 256,  # Reduced from 768
        "top_p": 0.3
    },
    "query": {
        "temperature": 0.3,
        "max_tokens": 384,  # Reduced from 768
        "top_p": 0.3
    },
    "detect": {
        "max_objects": 50  # Reduced from 150
    }
}
```

#### For Quality (Accuracy Priority)
```python
QUALITY_SETTINGS = {
    "caption": {
        "temperature": 0.5,
        "max_tokens": 768,
        "top_p": 0.5
    },
    "query": {
        "temperature": 0.4,
        "max_tokens": 1024,
        "top_p": 0.4
    },
    "detect": {
        "max_objects": 200
    }
}
```

### GUI Integration Patterns

The Inerbee demo implements dynamic UI that changes based on Moondream's capabilities:

```javascript
// Detect model capabilities
if (model === 'moondream') {
    // Show mode selector
    showModeSelector(['caption', 'query', 'detect', 'point']);

    // Show mode-specific controls
    if (mode === 'query') {
        showQueryInput();
    } else if (mode === 'detect' || mode === 'point') {
        showObjectInput();
        showDetectionCanvas();
    }
}
```

**UI Best Practices:**
- Clear visual feedback for each mode
- Input validation for queries
- Canvas overlay for detection boxes
- Crosshair markers for point mode
- Real-time mode switching

---

## Prompt Engineering Patterns

### 1. Few-Shot Learning
Provide examples to guide model behavior:

```python
FEW_SHOT_CAPTION = """
Example 1:
Frame: [park scene]
Caption: A sunny afternoon in the park where children play on swings while parents watch nearby.

Example 2:
Frame: [busy street]
Caption: Rush hour traffic fills the city street as pedestrians navigate crowded sidewalks.

Now describe this frame:
Frame: [current frame]
Caption:
"""
```

### 2. Structured Output
Request specific formats for consistent parsing:

```python
STRUCTURED_PROMPT = """
Analyze this frame and respond in JSON format:
{
  "scene_type": "indoor/outdoor/mixed",
  "main_subjects": ["subject1", "subject2"],
  "action": "description of main action",
  "objects": ["object1", "object2"],
  "mood": "mood description"
}
"""
```

### 3. Chain-of-Thought
For complex reasoning tasks:

```python
COT_PROMPT = """
Analyze this frame step by step:
1. First, identify the setting and environment
2. Then, list the main subjects present
3. Next, determine what action or event is occurring
4. Finally, synthesize into a natural caption

Let's think through this:
"""
```

### 4. Temporal Context Injection
For video continuity:

```python
TEMPORAL_PROMPT = """
Previous 3 frames summary:
- Frame -3: {caption_3}
- Frame -2: {caption_2}
- Frame -1: {caption_1}

Current frame: [image]

Describe what's happening now, noting any changes or progression from previous frames.
"""
```

---

## Context Window Management

### Frame Buffer Strategy
```python
class ContextManager:
    def __init__(self, max_history=5):
        self.frame_history = []
        self.caption_history = []
        self.max_history = max_history

    def add_frame(self, frame, caption):
        self.frame_history.append(frame)
        self.caption_history.append(caption)

        # Maintain sliding window
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
            self.caption_history.pop(0)

    def build_context(self, current_frame, mode="caption"):
        if mode == "caption":
            # Use only current frame for speed
            return {"frame": current_frame}
        elif mode == "query":
            # Include recent context for queries
            return {
                "frame": current_frame,
                "history": self.caption_history[-3:]
            }
```

### Token Budget Management
```python
def estimate_tokens(text, image=None):
    """Rough token estimation"""
    text_tokens = len(text.split()) * 1.3  # ~1.3 tokens per word
    image_tokens = 256 if image else 0  # Typical image embedding
    return int(text_tokens + image_tokens)

def trim_context(context, max_tokens=2000):
    """Trim context to fit within token budget"""
    current_tokens = estimate_tokens(context)

    if current_tokens <= max_tokens:
        return context

    # Truncate oldest history first
    # Keep system prompt and current frame
    # Implementation here...
```

---

## Real-Time Optimization

### Adaptive Context Based on Performance

```python
class AdaptiveContextManager:
    def __init__(self):
        self.latency_threshold = 200  # ms
        self.current_detail_level = "high"

    def adjust_context(self, last_latency):
        if last_latency > self.latency_threshold:
            # Reduce context detail
            if self.current_detail_level == "high":
                self.current_detail_level = "medium"
            elif self.current_detail_level == "medium":
                self.current_detail_level = "low"
        else:
            # Can afford more detail
            if self.current_detail_level == "low":
                self.current_detail_level = "medium"

    def get_prompt(self, mode):
        if self.current_detail_level == "low":
            return SHORT_PROMPTS[mode]
        elif self.current_detail_level == "medium":
            return MEDIUM_PROMPTS[mode]
        else:
            return DETAILED_PROMPTS[mode]
```

### Frame Sampling Strategy
```python
# Don't process every frame in high FPS scenarios
FRAME_SKIP_MAP = {
    "smolvlm": 2,      # Process every 2nd frame
    "moondream_caption": 1,  # Process every frame
    "moondream_query": 1,    # Process every frame for queries
    "moondream_detection": 3  # Every 3rd frame for detection
}
```

---

## Best Practices

### 1. System Prompt Design
```python
SYSTEM_PROMPTS = {
    "narrator": """You are an AI video narrator providing real-time descriptions.
    - Be concise but vivid
    - Use present tense
    - Avoid repetition
    - Focus on changes and actions
    - Maintain a natural, engaging tone""",

    "detective": """You are an AI visual analyst helping users understand video content.
    - Answer questions precisely
    - Reference specific visual elements
    - Admit uncertainty when unsure
    - Provide context from the current frame"""
}
```

### 2. Error Handling in Context
```python
def safe_context_build(frame, history, fallback="Analyzing frame..."):
    try:
        context = build_full_context(frame, history)
        return context
    except ContextTooLongError:
        # Gracefully degrade
        return build_minimal_context(frame)
    except Exception as e:
        logging.error(f"Context build failed: {e}")
        return {"prompt": fallback, "frame": frame}
```

### 3. Context Validation
```python
def validate_context(context, model_type):
    """Ensure context meets model requirements"""

    # Check token limits
    if model_type == "smolvlm":
        max_tokens = 2000
    else:
        max_tokens = 8000

    if estimate_tokens(context) > max_tokens:
        raise ContextTooLongError(f"Context exceeds {max_tokens} tokens")

    # Validate required fields
    required = ["frame", "prompt"]
    if not all(k in context for k in required):
        raise ValueError("Missing required context fields")

    return True
```

### 4. A/B Testing Prompts
```python
# Track which prompts perform best
class PromptTester:
    def __init__(self):
        self.variants = {
            "A": "Describe this scene briefly.",
            "B": "What's happening in this frame?",
            "C": "Narrate what you see."
        }
        self.results = {k: [] for k in self.variants}

    def log_result(self, variant, latency, user_satisfaction):
        self.results[variant].append({
            "latency": latency,
            "satisfaction": user_satisfaction
        })

    def get_best_variant(self):
        # Analyze and return best performing prompt
        pass
```

### 5. Cultural and Accessibility Considerations
- Use inclusive language in prompts
- Avoid assumptions about scene interpretation
- Provide options for different narration styles:
  - Detailed vs. concise
  - Technical vs. casual
  - Objective vs. dramatic

---

## Integration Example

Here's how to integrate context engineering into your VLM class:

```python
class VLM:
    def __init__(self, model_name="smolvlm"):
        self.model = self.load_model(model_name)
        self.context_manager = ContextManager()
        self.prompt_builder = PromptBuilder(model_name)

    async def process_frame(self, frame, mode="caption", user_input=None):
        # Build context based on mode
        context = self.prompt_builder.build(
            frame=frame,
            mode=mode,
            history=self.context_manager.get_history(),
            user_input=user_input
        )

        # Generate response
        output = await self.model.generate(context)

        # Update context history
        self.context_manager.add_frame(frame, output)

        return output
```

---

## Monitoring and Iteration

### Key Metrics to Track
1. **Latency**: Time from frame receipt to caption broadcast
2. **Relevance**: How well captions match the visual content
3. **Continuity**: Temporal coherence across frames
4. **Token Efficiency**: Output quality per token used

### Logging Template
```python
logger.info({
    "timestamp": time.time(),
    "model": model_name,
    "mode": mode,
    "context_tokens": token_count,
    "latency_ms": latency,
    "frame_id": frame_id,
    "prompt_variant": prompt_version
})
```

---

## Resources and References

- [Vision-Language Model Best Practices](https://example.com)
- [Prompt Engineering Guide](https://example.com)
- [Real-time Video Understanding Papers](https://example.com)

---

**Last Updated**: 2025-10-04
**Version**: 1.0
