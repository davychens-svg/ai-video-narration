# Vision AI - Mac Deployment Guide

This guide covers deploying the Vision AI application on macOS with optimized performance using llama.cpp for SmolVLM.

## System Requirements

- macOS (Apple Silicon M1/M2/M3 recommended for Metal GPU acceleration)
- Python 3.10 or higher
- Node.js 18 or higher
- 16GB+ RAM recommended
- Git

## Architecture Overview

**Mac deployment uses a hybrid backend strategy:**
- **SmolVLM**: Uses llama.cpp via llama-server (250-500ms inference with Metal GPU)
- **Qwen2-VL**: Uses PyTorch transformers (multilingual caption/query, no llama.cpp support)
- **Moondream**: Uses PyTorch transformers (always, as it doesn't support llama.cpp)

The frontend automatically detects which backend is available via the `/health` endpoint.

## Step 1: Clone Repository

```bash
cd ~
git clone https://github.com/davychens-svg/ai-video-narration.git
cd ai-video-narration
```

## Step 2: Backend Setup

### 2.1 Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.2 Install Python Dependencies

```bash
cd server
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Configure Backend Environment

Create `server/.env`:

```bash
HOST=127.0.0.1
PORT=8001
DEFAULT_MODEL=smolvlm
MODEL_DEVICE=mps
MODEL_DTYPE=float16
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
MAX_WORKERS=4
```

**Note**: Use `MODEL_DEVICE=mps` for Apple Silicon Metal acceleration.

## Step 3: llama.cpp Setup (for SmolVLM Speed Optimization)

### 3.1 Build llama.cpp with Metal Support

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make LLAMA_METAL=1
```

### 3.2 Download SmolVLM GGUF Model

Download the quantized GGUF model:
- Model: `HuggingFaceTB/SmolVLM-500M-Instruct-GGUF`
- File: `smolvlm-500m-instruct-q8_0.gguf`

Place it in `~/llama.cpp/models/` or any accessible directory.

### 3.3 Start llama-server

```bash
cd ~/llama.cpp
./llama-server \
  -m models/smolvlm-500m-instruct-q8_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -ngl 33 \
  --ctx-size 2048
```

**Parameters:**
- `-ngl 33`: GPU layers (adjust based on your Mac's VRAM)
- `--ctx-size 2048`: Context size for longer conversations
- Keep this running in a terminal or use screen/tmux

## Step 4: Frontend Setup

### 4.1 Install Node.js Dependencies

```bash
cd ~/ai-video-narration/frontend
npm install
```

### 4.2 Configure Frontend Environment

Create `frontend/.env.development`:

```bash
VITE_SERVER_URL=http://127.0.0.1:8001
VITE_WS_URL=ws://127.0.0.1:8001/ws
VITE_ENV=development
```

## Step 5: Running the Application

### 5.1 Start Backend Server

In terminal 1:
```bash
cd ~/ai-video-narration/server
source ../venv/bin/activate
python main.py
```

You should see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001
```

### 5.2 Start Frontend Development Server

In terminal 2:
```bash
cd ~/ai-video-narration/frontend
npm run dev
```

You should see:
```
VITE v5.4.20  ready in XXX ms
➜  Local:   http://localhost:3000/
```

### 5.3 Verify Backend Detection

Open `http://localhost:3000` and check the browser console. You should see the health check response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "smolvlm",
  "backend_type": "llamacpp",
  "llamacpp_available": true
}
```

## Step 6: Testing

1. **Test SmolVLM**:
   - Select "SmolVLM Fast Captions" model
   - Start camera and click "Start"
   - Should see 250-500ms inference times (using llama.cpp)

2. **Test Moondream**:
   - Select "Moondream Advanced Features" model
   - Start camera and enter a custom prompt
   - Should see 1-3s inference times (using transformers)

3. **Test Qwen2-VL (multilingual)**:
   - Run `RUN_QWEN_TESTS=1 python server/test_models.py` once to download weights and verify caption/query output
   - In the UI, pick "Qwen2-VL Multilingual" and check that captions/answers arrive via the `/api/process_frame` endpoint (see backend logs)

## Troubleshooting

### llama-server Not Detected

If `llamacpp_available: false`, check:
- llama-server is running on port 8080
- No firewall blocking localhost:8080
- Try manually: `curl http://127.0.0.1:8080/health`

### SmolVLM Falls Back to Transformers

If SmolVLM is slow (1-3s), it's using transformers backend:
- Verify llama-server is running
- Check backend logs for port 8080 connection errors
- Frontend will automatically fall back to transformers if llama.cpp unavailable

### Metal GPU Not Working

```bash
# Verify Metal support
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

Should print `True`. If `False`:
- Update to latest macOS
- Update PyTorch: `pip install --upgrade torch torchvision`

### Permission Errors

```bash
chmod +x ~/llama.cpp/llama-server
```

## Performance Benchmarks (Mac M1/M2/M3)

| Model | Backend | Inference Time | GPU |
|-------|---------|----------------|-----|
| SmolVLM | llama.cpp | 250-500ms | Metal |
| SmolVLM | transformers | 1-2s | MPS |
| Moondream | transformers | 1-3s | MPS |

## Production Build (Optional)

For a production build without hot-reload:

```bash
cd frontend
npm run build
npm run preview
```

## Updating the Application

```bash
cd ~/ai-video-narration
git pull origin main
cd frontend
npm install
cd ../server
source ../venv/bin/activate
pip install -r requirements.txt
```

## Password Protection (Optional)

For local development, password protection is typically not needed. However, if you want to add authentication:

### Option 1: Using Local Nginx (Advanced)

If you set up nginx locally (not covered in this guide), you can use the same nginx basic authentication as described in [DEPLOYMENT_LINUX.md](DEPLOYMENT_LINUX.md#password-management).

### Option 2: SSH Tunnel for Remote Access

If you need to access your Mac remotely:

```bash
# On remote machine, create SSH tunnel
ssh -L 3000:localhost:3000 -L 8001:localhost:8001 user@your-mac-ip

# Then access via localhost:3000 on remote machine
```

This is more secure than exposing ports directly.

## Notes

- Mac deployment is designed for local development, not production
- Keep llama-server running for optimal SmolVLM performance
- If llama-server stops, the application automatically falls back to transformers
- Moondream always uses transformers (no llama.cpp support)
- The frontend automatically detects the available backend
- For production deployment, use the [Linux GPU Server Guide](DEPLOYMENT_LINUX.md) with password protection
