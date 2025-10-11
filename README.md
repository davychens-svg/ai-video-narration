# Vision AI Demo

Real-time video analysis with Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)

## ✨ Features

- **Real-time Processing**: <100ms latency per frame (target: 50-70ms)
- **WebRTC Streaming**: Direct browser-to-server video streaming, no app install needed
- **Small Models**: Optimized for SmolVLM (500m params), Qwen2-VL (multilingual), and Moondream (1.4B params)
- **Live Captions**: Instant AI-generated narration via WebSocket broadcast
- **Performance Monitoring**: Real-time FPS, latency, and P95 metrics
- **Model Switching**: Hot-swap between models on the fly

## Quick Start

### Prerequisites

**Development (macOS):**
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- 16GB+ RAM

**Production/Cloud (Linux + NVIDIA GPU):**
- Ubuntu 20.04/22.04 or compatible Linux
- NVIDIA GPU (RTX 3060+ with 12GB+ VRAM, or cloud GPUs: A100, V100, T4)
- Python 3.9-3.12
- CUDA 11.8+ or 12.x
- 16GB+ System RAM
- 📘 **See [CUDA_SETUP.md](CUDA_SETUP.md) for detailed GPU setup instructions**

**Supported Platforms:**
- ✅ macOS (Apple Silicon with MPS)
- ✅ Linux (NVIDIA CUDA GPUs)
- ✅ Cloud: AWS, GCP, Azure, Lambda Labs, RunPod
- ⚠️ Windows (CPU only, not recommended for production)

### Installation

#### Option A: macOS (Apple Silicon)

1. **Clone and setup:**
```bash
git clone https://github.com/yourusername/VisionLanguageModel.git
cd VisionLanguageModel
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Create logs directory:**
```bash
mkdir -p logs
```

#### Option B: Linux + NVIDIA GPU (Cloud/Production)

1. **Verify GPU and CUDA:**
```bash
nvidia-smi  # Should show your GPU
nvcc --version  # Should show CUDA 11.8 or 12.x
```

If CUDA is not installed, see **[CUDA_SETUP.md](CUDA_SETUP.md)** for complete installation instructions.

2. **Clone and setup:**
```bash
git clone https://github.com/yourusername/VisionLanguageModel.git
cd VisionLanguageModel
python3 -m venv venv
source venv/bin/activate
```

3. **Install PyTorch with CUDA support:**
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. **Install project dependencies:**
```bash
pip install -r requirements.txt

# Optional: GPU-optimized packages
pip install -r requirements-cuda.txt
```

5. **Verify GPU detection:**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4090
```

6. **Create logs directory:**
```bash
mkdir -p logs
```

### Running the Application

**Option 1: Full Stack (Recommended)**

Use the convenience script to run both backend and frontend:

```bash
./start.sh
```

This starts:
- Backend (FastAPI) on `http://localhost:8001`
- Frontend (React) on `http://localhost:3001`

When deploying on a remote host, export `ALLOWED_ORIGINS` before running the script and point the frontend to your public API/WebSocket (via `.env` or the in-app **Settings** dialog). See [Configuration (Local & Cloud)](#configuration-local--cloud) for the full variable list.

Open `http://localhost:3001` (or your public domain) in your browser to use the app.

**Option 2: Backend Only**

```bash
./run.sh
# or manually:
cd server
export ALLOWED_ORIGINS="https://vision.example.com"
python main.py
```

Server will start on `http://localhost:8001` (or your configured host/port). In cloud deployments, expose the port in your firewall/load balancer and use the public URL in the frontend settings.

**Option 3: Separate Backend + Frontend**

Terminal 1 - Backend:
```bash
cd server
export ALLOWED_ORIGINS="https://vision.example.com"
python main.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm install  # first time only
npm run dev   # local testing
# npm run build  # use after setting VITE_SERVER_URL / VITE_WS_URL for production
```

### Configuration (Local & Cloud)

| Variable | Scope | Default | Description |
|----------|-------|---------|-------------|
| `ALLOWED_ORIGINS` | Backend (`server/main.py`) | `http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001` | Comma-separated list of origins allowed by CORS. Use `"*"` to allow all origins (credentials are disabled automatically). |
| `VITE_SERVER_URL` | Frontend (`frontend`) | Auto-resolves to local backend (`http://localhost:8001`) | REST base URL for API calls. Set to your public backend endpoint, e.g. `https://api.example.com`. |
| `VITE_WS_URL` | Frontend (`frontend`) | Auto-resolves to local websocket (`ws://localhost:8001/ws`) | WebSocket endpoint for captions/detections. Use `wss://` when hosting over HTTPS. |

**Example (cloud deployment):**

```bash
# Backend – allow requests from your public domain/IP
export ALLOWED_ORIGINS="https://vision.example.com"
python server/main.py

# Frontend – configure endpoints before building
cat <<'ENV' > frontend/.env.production
VITE_SERVER_URL=https://api.example.com
VITE_WS_URL=wss://api.example.com/ws
ENV

cd frontend
npm run build
```

You can also override URLs on the fly from the **Settings** dialog in the UI—handy for QA against multiple environments.

### Using the App

1. Open the frontend in your browser (`http://localhost:3000` for Vite dev server or your deployed domain)
2. Click **Start Camera** or load a video to begin streaming
3. Pick a model: **SmolVLM** (ultra-fast), **Qwen2-VL** (multilingual captions + queries), or **Moondream** (advanced detection/point/mask)
4. When Qwen2-VL is active the frontend automatically stays on the `/api/process_frame` transformer path—perfect for language switching
5. Watch realtime narration, detections, and points overlay the video feed
6. Adjust connection details, response length, or detection prompts via **Settings** at any time

## Architecture

```
┌─────────────┐      WebRTC       ┌─────────────┐
│   Browser   │ ─────────────────▶│   FastAPI   │
│  (Camera)   │                    │   Server    │
└─────────────┘                    └─────────────┘
                                           │
                                           ▼
                                   ┌─────────────┐
                                   │   Frame     │
                                   │   Queue     │
                                   └─────────────┘
                                           │
                                           ▼
                                   ┌─────────────┐
                                   │  SmolVLM    │
                                   │  Processor  │
                                   └─────────────┘
                                           │
       ┌───────────────────────────────────┘
       │       WebSocket Broadcast
       ▼
┌─────────────┐
│  All Clients│
│  (Captions) │
└─────────────┘
```

## Project Structure

```
VisionLanguageModel/
├── frontend/                # React frontend (Vite + TypeScript)
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── hooks/          # Custom React hooks
│   │   ├── lib/            # Utilities
│   │   └── App.tsx         # Main app
│   ├── package.json
│   ├── vite.config.ts
│   └── README.md           # Frontend documentation
├── server/
│   └── main.py             # FastAPI server with WebRTC & WebSocket
├── client/
│   └── index.html          # Legacy simple HTML interface
├── models/
│   └── VLM.py              # Optimized VLM wrapper (SmolVLM, MobileVLM)
├── static/                  # Static assets (if needed)
├── logs/                    # Server logs
├── requirements.txt         # Python dependencies
├── CONTEXT_ENGINEERING.md   # Prompt engineering guide
└── README.md               # This file
```

## Performance Optimization

### Target Metrics
- **Latency**: <100ms per frame (target: 50-70ms)
- **Throughput**: 15-30 FPS
- **Memory**: <2GB VRAM
- **CPU**: Minimal usage

### Optimization Strategies
1. **Frame Skipping**: Process every 2nd frame (adaptive based on motion)
2. **Resolution**: Downscale to 384x384 (optimal for small VLMs)
3. **Quantization**: BFloat16 on Apple Silicon, FP16 on NVIDIA
4. **Minimal Prompts**: Ultra-short prompts (<20 tokens)
5. **Greedy Decoding**: No beam search, max 20 tokens output

See [CONTEXT_ENGINEERING.md](CONTEXT_ENGINEERING.md) for detailed prompt engineering strategies.

## Model Selection

### SmolVLM-500M (Primary) - Recommended
- **Size**: 500M parameters (~1GB storage, ~0.5GB VRAM)
- **Speed**: 3-5s per frame on Mac MPS
- **Quality**: High-quality captions with custom queries
- **Model**: `HuggingFaceTB/SmolVLM-500M-Instruct`

### Moondream 3.0 Preview - Feature-Rich
- **Size**: 0.5B parameters (int8 quantized)
- **Speed**: Variable based on feature
- **Quality**: Multiple modes (caption, query, detect, point, mask)
- **Model**: `vikhyatk/moondream-0_5b-int8`

**Moondream Features**:
- **Caption**: Automatic image description (short/medium/long)
- **Query**: Answer custom questions about the video
- **Detection**: Object detection with blue bounding boxes
- **Point**: Locate specific objects with center points
- **Mask**: Privacy masking for detected objects (NEW)

## API Endpoints

### HTTP Endpoints
- `GET /` - Serve client HTML
- `POST /offer` - WebRTC offer/answer exchange
- `GET /stats` - Get performance statistics
- `POST /api/switch_model` - Switch VLM model

### WebSocket Endpoint
- `WS /ws` - Real-time caption broadcast

## Configuration

Edit `server/main.py` to configure:

```python
# Frame processing
frame_processor = FrameProcessor(skip_frames=2)  # Process every 2nd frame

# Performance monitoring
perf_monitor = PerformanceMonitor(window=100)  # Track last 100 frames

# Model selection
vlm_processor = VLMProcessor(model_name="smolvlm")  # or "mobilevlm"
```

## Troubleshooting

### "Model loading failed"
- Ensure you have enough VRAM/RAM
- Try switching to MobileVLM (smaller)
- Check internet connection for model download

### "Camera access denied"
- Grant camera permissions in browser
- Use HTTPS in production (required for WebRTC)

### "High latency (>100ms)"
- Increase frame skipping: `skip_frames=3`
- Switch to MobileVLM
- Reduce video resolution in client
- Check CPU/GPU load

### "WebSocket disconnected"
- Check server logs: `tail -f logs/server.log`
- Verify firewall settings
- Restart server

## Development

### Testing VLM Standalone
```bash
cd models
python VLM.py
```

### Qwen2-VL Multilingual Smoke Test
```bash
RUN_QWEN_TESTS=1 python server/test_models.py
```
This downloads (if needed) the Qwen2-VL weights, then prints both a caption and a query answer so you can confirm the transformer-backed multilingual path is healthy.

### Running with Debug Logging
```bash
cd server
LOG_LEVEL=DEBUG python main.py
```

## Deployment

### Production Setup (Ubuntu 22.04 + NVIDIA GPU)

1. **Install CUDA:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

3. **Run with systemd:**
```bash
sudo systemctl enable inerbee-narrator
sudo systemctl start inerbee-narrator
```

4. **Setup HTTPS** (required for WebRTC in production):
- Use nginx reverse proxy with Let's Encrypt SSL
- Or use ngrok for quick testing

## Performance Benchmarks

### Apple M2 Pro / M3 Pro (Development - MPS)
- **SmolVLM**: 50-100ms avg, 120ms p95
- **Moondream Caption**: 150-250ms avg
- **Moondream Detection**: 200-350ms avg
- **FPS**: 10-20 (with frame skipping)
- **VRAM**: ~1.5GB

### NVIDIA RTX 4090 (High-End Desktop - CUDA)
- **SmolVLM**: 40-70ms avg, 90ms p95
- **Moondream Caption**: 100-180ms avg
- **Moondream Detection**: 150-250ms avg
- **FPS**: 20-30+ (with frame skipping)
- **VRAM**: ~2GB

### NVIDIA RTX 3060 12GB (Mid-Range - CUDA)
- **SmolVLM**: 80-120ms avg, 150ms p95
- **Moondream Caption**: 200-300ms avg
- **Moondream Detection**: 300-450ms avg
- **FPS**: 10-15 (with frame skipping)
- **VRAM**: ~2GB

### NVIDIA A100 40GB (Cloud/Data Center - CUDA)
- **SmolVLM**: 30-50ms avg, 70ms p95
- **Moondream Caption**: 80-150ms avg
- **Moondream Detection**: 120-200ms avg
- **FPS**: 25-35+ (with frame skipping)
- **VRAM**: ~2GB

### NVIDIA T4 16GB (Cloud Budget - CUDA)
- **SmolVLM**: 100-180ms avg, 220ms p95
- **Moondream Caption**: 300-500ms avg
- **Moondream Detection**: 400-600ms avg
- **FPS**: 5-10 (with frame skipping)
- **VRAM**: ~2GB

**Note**: Times are per-frame inference latency at 720p resolution. Real-world FPS depends on frame skipping, network latency, and concurrent users.

## Contributing

For issues or improvements:
1. Check logs in `logs/server.log`
2. Review `CONTEXT_ENGINEERING.md` for prompt optimization
3. Test with different models and settings

## License

MIT License

## Acknowledgments

- HuggingFace for SmolVLM
- FastAPI and aiortc communities

---

**Built with ❤️ for real-time AI demos**
