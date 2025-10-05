# Migration to llama.cpp for Real-Time Inference

## Current Status: 30-36 seconds per frame → Target: <1 second

## Why Switch to llama.cpp?

**Current Issue**: HuggingFace Transformers on Apple MPS is too slow (30-36s per frame)

**Solution**: llama.cpp with GGUF format is optimized for:
- Native Metal GPU acceleration on Apple Silicon
- Efficient memory management
- Fast inference (demonstrated in https://github.com/ngxson/smolvlm-realtime-webcam)

## Available GGUF Models

✅ **Pre-converted models available on HuggingFace**:
- `ggml-org/SmolVLM-500M-Instruct-GGUF` (409M params)
  - Q8_0 quantization: 437 MB
  - F16 (full precision): 820 MB
- Downloads: 106,707 last month
- License: Apache 2.0

## llama-server Installation Status

✅ **Already installed**: `/opt/homebrew/bin/llama-server`
- Version: 5830 (bac8bed2)
- Built with: Apple clang for arm64-apple-darwin24.4.0
- Metal GPU support: Yes

## Implementation Plan

### Option 1: Full Backend Replacement (Recommended)

Replace the entire VLM.py backend with llama-server.

**Pros**:
- Maximum performance gain
- Simpler architecture
- OpenAI-compatible API
- No Python overhead

**Cons**:
- Requires running separate llama-server process
- Need to update all API endpoints

### Option 2: Hybrid Approach

Keep VLM.py but add llama.cpp as an option for SmolVLM only.

**Pros**:
- Can compare performance side-by-side
- Moondream stays on HuggingFace transformers
- Gradual migration

**Cons**:
- More complex codebase
- Maintains two inference paths

## Implementation Steps (Option 1 - Recommended)

### 1. Start llama-server with SmolVLM GGUF

```bash
# Download model automatically from HuggingFace and start server
llama-server \
  -hf ggml-org/SmolVLM-500M-Instruct-GGUF \
  -ngl 99 \
  --port 8080 \
  --ctx-size 2048 \
  --n-predict 20 \
  --host 0.0.0.0 \
  --alias smolvlm
```

**Flags explained**:
- `-hf`: Download from HuggingFace (includes mmproj for vision)
- `-ngl 99`: Offload all layers to Metal GPU
- `--port 8080`: Different from FastAPI (5001)
- `--ctx-size 2048`: Context window
- `--n-predict 20`: Max tokens (for speed)
- `--alias smolvlm`: Model name for API

### 2. Test llama-server API

**OpenAI-compatible endpoint**: `http://localhost:8080/v1/chat/completions`

**Example request with vision**:
```python
import requests
import base64

# Read image as base64
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "smolvlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image"
                    }
                ]
            }
        ],
        "max_tokens": 20,
        "temperature": 0.0  # Greedy decoding for speed
    }
)

print(response.json())
```

### 3. Update FastAPI Backend

**File**: `server.py`

Add new endpoint for llama.cpp inference:

```python
@app.post("/vlm/process_frame_llamacpp")
async def process_frame_llamacpp(request: VLMRequest):
    """Process frame using llama-server backend"""
    import requests

    start_time = time.time()

    # Prepare request for llama-server
    llama_request = {
        "model": "smolvlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{request.image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": request.prompt or "Describe this image briefly."
                    }
                ]
            }
        ],
        "max_tokens": 20,
        "temperature": 0.0
    }

    try:
        # Call llama-server
        response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            json=llama_request,
            timeout=5.0  # 5 second timeout
        )
        response.raise_for_status()

        result = response.json()
        caption = result["choices"][0]["message"]["content"]

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"llama.cpp inference: {latency_ms:.1f}ms")

        return {
            "status": "success",
            "caption": caption,
            "model": "smolvlm-llamacpp",
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms
        }

    except Exception as e:
        logger.error(f"llama-server error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 4. Frontend Changes

**File**: `frontend/src/components/ModelSelector.tsx`

Add toggle for inference backend:

```typescript
const [backend, setBackend] = useState<'transformers' | 'llamacpp'>('llamacpp');

// In the SmolVLM section
<div className="space-y-2">
  <label className="text-sm font-medium">Inference Backend</label>
  <div className="grid grid-cols-2 gap-2">
    <Button
      variant={backend === 'transformers' ? 'default' : 'outline'}
      onClick={() => setBackend('transformers')}
    >
      Transformers (Slow)
    </Button>
    <Button
      variant={backend === 'llamacpp' ? 'default' : 'outline'}
      onClick={() => setBackend('llamacpp')}
    >
      llama.cpp (Fast)
    </Button>
  </div>
  {backend === 'llamacpp' && (
    <p className="text-xs text-green-400">
      ⚡ Real-time inference (&lt;1s)
    </p>
  )}
</div>
```

**File**: `frontend/src/hooks/useVideoProcessor.ts`

Update endpoint based on backend:

```typescript
const endpoint = backend === 'llamacpp'
  ? '/vlm/process_frame_llamacpp'
  : '/vlm/process_frame';

const response = await fetch(`${API_BASE_URL}${endpoint}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
});
```

## Performance Expectations

### Current (HuggingFace Transformers)
- **Inference**: 30-36 seconds per frame
- **Image size**: 128x128
- **Tokens**: 10
- **Unusable** for real-time video

### With llama.cpp (Expected)
- **Inference**: 500-1000ms per frame
- **Image size**: Can use 224x224 or larger
- **Tokens**: 20-50
- **Real-time capable** ✅

### Breakdown (llama.cpp estimated):
1. Image encoding: 100-200ms
2. Vision projection: 100-200ms
3. Token generation: 300-600ms (20 tokens × 15-30ms/token)
4. Total: **500-1000ms** ✅

## Testing Plan

### 1. Start llama-server

```bash
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF -ngl 99 --port 8080
```

Wait for model download and loading.

### 2. Test API directly

```bash
# Create test image
curl -s "https://picsum.photos/640/480" -o /tmp/test.jpg

# Convert to base64
IMAGE_B64=$(base64 -i /tmp/test.jpg)

# Call llama-server
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smolvlm",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'$IMAGE_B64'"}},
        {"type": "text", "text": "Describe this image"}
      ]
    }],
    "max_tokens": 20
  }'
```

### 3. Update backend code

Add `/vlm/process_frame_llamacpp` endpoint to `server.py`.

### 4. Update frontend

Add backend selector and update API calls.

### 5. Test real-time video

Start video streaming and measure latency from server logs.

## Rollback Plan

If llama.cpp doesn't work as expected:

1. Keep existing VLM.py code unchanged
2. llama-server runs independently
3. Can switch backends via frontend toggle
4. No data loss or breaking changes

## Model Comparison

| Feature | Transformers | llama.cpp |
|---------|-------------|-----------|
| **Speed** | 30-36s | 0.5-1s ⚡ |
| **Memory** | 3-4GB | 1-2GB |
| **GPU** | MPS (slow) | Metal (fast) |
| **Setup** | pip install | Already installed ✅ |
| **API** | Custom | OpenAI-compatible |
| **Quality** | Same model | Same model |

## Next Steps

1. ✅ Research complete
2. ⏭️ Start llama-server with SmolVLM GGUF
3. ⏭️ Test API performance
4. ⏭️ Update backend code
5. ⏭️ Update frontend
6. ⏭️ Benchmark and compare

## References

- llama.cpp vision support: https://simonwillison.net/2025/May/10/llama-cpp-vision/
- GGUF model: https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF
- Real-time demo: https://github.com/ngxson/smolvlm-realtime-webcam
- llama.cpp PR #13050: Vision model implementation details
