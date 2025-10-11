# Deployment Guide

Complete deployment guide for Vision AI Demo on macOS and Cloud (NVIDIA GPU).

## Table of Contents

- [macOS Deployment (Apple Silicon)](#macos-deployment-apple-silicon)
- [Cloud Deployment (NVIDIA GPU)](#cloud-deployment-nvidia-gpu)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## macOS Deployment (Apple Silicon)

### Prerequisites

- macOS 12.0 or later
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Node.js 18+
- Git

### 1. Install Dependencies

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Node.js
brew install python@3.11 node

# Install llama.cpp with Metal support
brew install llama.cpp
```

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/ai-video-narration.git
cd ai-video-narration
```

### 3. Backend Setup

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download SmolVLM GGUF model (done automatically on first run)
# The model will be downloaded from Hugging Face: ggml-org/SmolVLM-500M-Instruct-GGUF
```

### 4. Frontend Setup

```bash
cd frontend
npm install
npm run build
cd ..
```

### 5. Start Services

**Option 1: Quick Start (Recommended)**

```bash
# Start llama-server
./start_llamacpp_server.sh

# In a new terminal, start backend
cd server
source ../venv/bin/activate
python main.py

# In a new terminal, start frontend (development)
cd frontend
npm run dev
```

**Option 2: Production Start**

```bash
# Start llama-server
./start_llamacpp_server.sh

# Start backend
cd server
source ../venv/bin/activate
export ALLOWED_ORIGINS="https://your-domain.com"
uvicorn main:app --host 0.0.0.0 --port 8001

# Serve frontend build
cd frontend
npm run build
cat <<'ENV' > .env.production
VITE_SERVER_URL=https://api.your-domain.com
VITE_WS_URL=wss://api.your-domain.com/ws
ENV
npx serve -s dist -p 5173
```

### Nginx Reverse Proxy (Production)

For a public deployment, proxy both the API and frontend through Nginx (or your preferred web server) to expose a single HTTPS endpoint.

1. **Install Nginx (Ubuntu)**
   ```bash
   sudo apt update
   sudo apt install nginx -y
   ```

2. **Copy the frontend build to a web root**
   ```bash
   sudo mkdir -p /var/www/vision-frontend
   sudo cp -r frontend/dist/* /var/www/vision-frontend/
   ```

3. **Create Nginx site configurations** (e.g. `/etc/nginx/sites-available/vision-frontend` and `/etc/nginx/sites-available/vision-api`)

   *Frontend (`vision.example.com`)*
   ```nginx
   server {
       listen 80;
       server_name vision.example.com;

       root /var/www/vision-frontend;
       index index.html;

       location / {
           try_files $uri $uri/ /index.html;
       }
   }
   ```

   *Backend/API (`api.vision.example.com`)*
   ```nginx
   upstream vision_api {
       server 127.0.0.1:8001;
   }

   server {
       listen 80;
       server_name api.vision.example.com;

       location / {
           proxy_pass http://vision_api;
           proxy_http_version 1.1;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }

       location /ws {
           proxy_pass http://vision_api/ws;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

4. **Enable the sites and reload Nginx**
   ```bash
   sudo ln -s /etc/nginx/sites-available/vision-frontend /etc/nginx/sites-enabled/vision-frontend
   sudo ln -s /etc/nginx/sites-available/vision-api /etc/nginx/sites-enabled/vision-api
   sudo nginx -t
   sudo systemctl reload nginx
   ```

5. **Configure the backend and frontend**
   ```bash
   # Backend (update CORS for the frontend domain)
   export ALLOWED_ORIGINS="https://vision.example.com"
   python server/main.py

   # Frontend (.env.production)
   cat <<'ENV' > frontend/.env.production
   VITE_SERVER_URL=https://api.vision.example.com
   VITE_WS_URL=wss://api.vision.example.com/ws
   ENV
   npm run build
   ```

6. **Add HTTPS (recommended)**
   ```bash
   sudo apt install certbot python3-certbot-nginx -y
   sudo certbot --nginx -d vision.example.com
   ```

This configuration terminates TLS at Nginx, serves the static React assets, and proxies API/WebSocket traffic to the FastAPI backend running on the same machine.

### 6. Access Application

Open your browser and navigate to:
- Frontend: `http://localhost:5173` (or `https://your-domain.com` in production)
- Backend API: `http://localhost:8001` (or your public API endpoint)
- API Docs: http://localhost:8001/docs

### Performance Expectations (Apple Silicon)

- **Inference Speed**: 250-600ms per frame
- **GPU Utilization**: ~80-90% (Metal)
- **Memory Usage**: ~2GB VRAM, ~4GB RAM
- **Recommended Settings**:
  - Video Quality: Medium (1280x720)
  - Capture Interval: 500ms (2 FPS)
  - Max Tokens: 150

---

## Cloud Deployment (NVIDIA GPU)

### Prerequisites

- Ubuntu 20.04/22.04 LTS
- NVIDIA GPU (T4, V100, A100, or better)
- CUDA 11.8+ and cuDNN
- Docker (recommended) or Python 3.11+
- Node.js 18+

### 1. Install NVIDIA Drivers and CUDA

```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-535

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install cuda

# Verify installation
nvidia-smi
nvcc --version
```

### 2. Install llama.cpp with CUDA Support

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_CUDA=ON
cmake --build . --config Release
sudo cmake --install .
cd ../..
```

### 3. Clone Repository and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ai-video-narration.git
cd ai-video-narration

# Install Python dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Node.js dependencies
cd frontend
npm install
npm run build
cd ..
```

Configure the runtime endpoints before launching:

```bash
export ALLOWED_ORIGINS="https://vision.example.com"

cat <<'ENV' > frontend/.env.production
VITE_SERVER_URL=https://api.example.com
VITE_WS_URL=wss://api.example.com/ws
ENV
```

### 4. Configure for NVIDIA GPU

Edit `start_llamacpp_server.sh` to use CUDA instead of Metal:

```bash
#!/bin/bash

echo "🚀 Starting llama-server with SmolVLM GGUF (CUDA)..."
echo "📊 Expected performance: 100-300ms per frame"
echo "🎯 Target: Real-time video analysis (<1s)"
echo ""

# Kill any existing llama-server on port 8080
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Killing existing llama-server on port 8080..."
    kill -9 $(lsof -Pi :8080 -sTCP:LISTEN -t)
    sleep 1
fi

# Start llama-server with CUDA
llama-server \
  -hf ggml-org/SmolVLM-500M-Instruct-GGUF \
  -ngl 99 \
  --port 8080 \
  --ctx-size 2048 \
  --n-predict 150 \
  --host 0.0.0.0 \
  --alias smolvlm \
  --log-disable \
  2>&1 | tee /tmp/llama-server.log

# Flags explained:
# -hf: Download model from HuggingFace (includes multimodal projector)
# -ngl 99: Offload all layers to CUDA GPU for maximum speed
# --port 8080: Different from FastAPI backend (8001)
# --ctx-size 2048: Context window size
# --n-predict 150: Maximum tokens to generate
# --host 0.0.0.0: Accept connections from any IP (for cloud deployment)
# --alias smolvlm: Model name for API calls
# --log-disable: Reduce log verbosity
```

### 5. Setup Systemd Services (Production)

Create `/etc/systemd/system/llama-server.service`:

```ini
[Unit]
Description=Llama Server for SmolVLM
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-video-narration
ExecStart=/home/ubuntu/ai-video-narration/start_llamacpp_server.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/video-backend.service`:

```ini
[Unit]
Description=Video Narration Backend (FastAPI)
After=network.target llama-server.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-video-narration/server
Environment="PATH=/home/ubuntu/ai-video-narration/venv/bin"
ExecStart=/home/ubuntu/ai-video-narration/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start services:

```bash
sudo systemctl daemon-reload
sudo systemctl enable llama-server video-backend
sudo systemctl start llama-server video-backend

# Check status
sudo systemctl status llama-server
sudo systemctl status video-backend
```

### 6. Setup Nginx Reverse Proxy

Install Nginx:

```bash
sudo apt install nginx
```

Create `/etc/nginx/sites-available/video-narration`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /home/ubuntu/ai-video-narration/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # WebSocket
    location /ws {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Health check
    location /health {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/video-narration /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 7. Setup SSL with Let's Encrypt (Optional)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Performance Expectations (NVIDIA GPU)

**T4 GPU:**
- Inference Speed: 150-400ms
- Memory Usage: ~2GB VRAM

**V100 GPU:**
- Inference Speed: 100-250ms
- Memory Usage: ~2GB VRAM

**A100 GPU:**
- Inference Speed: 80-200ms
- Memory Usage: ~2GB VRAM

**Recommended Settings:**
- Video Quality: High (1920x1080)
- Capture Interval: 500ms (2 FPS)
- Max Tokens: 150

---

## Configuration

### Backend Environment

Configure CORS and public origins before starting the FastAPI server:

```bash
export ALLOWED_ORIGINS="https://vision.example.com,https://app.example.com"
python server/main.py
```

- Separate multiple origins with commas. Use `"*"` to allow any origin (credentials will be disabled automatically for security).
- When running under a process manager (systemd, supervisor, etc.) add the variable to the unit file.

### Frontend Environment

Create an environment file inside `frontend/` (e.g. `.env.production` or `.env.local`) to point the UI at your backend:

```env
VITE_SERVER_URL=https://api.example.com
VITE_WS_URL=wss://api.example.com/ws
```

- `VITE_SERVER_URL` is used for REST endpoints (health checks, frame uploads).
- `VITE_WS_URL` is the realtime websocket endpoint. Always use `wss://` when your site is served over HTTPS.
- The in-app **Settings** dialog can override these values at runtime if you need to test multiple environments.

---

## Troubleshooting

### macOS Issues

**Issue: llama-server fails to start**
```bash
# Check if port 8080 is in use
lsof -i :8080
# Kill the process if needed
kill -9 <PID>
```

**Issue: Metal GPU not detected**
```bash
# Verify Metal support
system_profiler SPDisplaysDataType | grep Metal
# Should show "Metal: Supported"
```

**Issue: Slow inference (>2s)**
- Check GPU utilization: `sudo powermetrics --samplers gpu_power`
- Reduce video quality to "medium"
- Increase capture interval to 1000ms

### Cloud (NVIDIA) Issues

**Issue: CUDA out of memory**
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size or context window in start_llamacpp_server.sh
--ctx-size 1024  # Instead of 2048
```

**Issue: llama-server not using GPU**
```bash
# Verify CUDA build
llama-server --version
# Should show "CUDA: ON"

# Check if GPU is accessible
nvidia-smi
```

**Issue: High latency (>1s)**
- Check network latency between frontend and backend
- Verify GPU utilization: `nvidia-smi`
- Ensure `-ngl 99` is set (offload all layers to GPU)
- Check system logs: `journalctl -u llama-server -f`

### General Issues

**Issue: WebSocket connection fails**
- Check firewall settings
- Verify CORS configuration in `server/main.py`
- Check Nginx configuration for WebSocket proxy

**Issue: Frontend not loading**
```bash
# Rebuild frontend
cd frontend
npm run build

# Check build output
ls dist/
```

**Issue: Model download fails**
```bash
# Manually download model
huggingface-cli download ggml-org/SmolVLM-500M-Instruct-GGUF

# Check Hugging Face token (if needed)
huggingface-cli login
```

---

## Monitoring and Logs

### View Logs

**llama-server:**
```bash
tail -f /tmp/llama-server.log
```

**Backend:**
```bash
tail -f /tmp/server.log

# Or with systemd
sudo journalctl -u video-backend -f
```

**Nginx:**
```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Performance Monitoring

**macOS:**
```bash
# GPU usage
sudo powermetrics --samplers gpu_power -i 1000

# Memory usage
top -pid $(pgrep llama-server)
```

**Linux:**
```bash
# GPU usage
nvidia-smi -l 1

# Memory usage
htop
```

---

## Scaling Considerations

### Load Balancing

For high traffic, deploy multiple backend instances behind a load balancer:

```nginx
upstream video_backend {
    least_conn;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    location /api {
        proxy_pass http://video_backend;
        # ... other proxy settings
    }
}
```

### Caching

Enable response caching for similar queries:

```python
# In server/main.py
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_inference(image_hash: str, prompt: str):
    # Inference logic
    pass
```

---

## Security Best Practices

1. **Use HTTPS in production** (Let's Encrypt)
2. **Set up CORS properly** in `server/main.py`
3. **Rate limiting** with Nginx or application-level
4. **Authentication** for API endpoints (if needed)
5. **Regular updates** of dependencies and models
6. **Firewall configuration**:
   ```bash
   sudo ufw allow 80/tcp    # HTTP
   sudo ufw allow 443/tcp   # HTTPS
   sudo ufw enable
   ```

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/ai-video-narration/issues
- Documentation: https://github.com/yourusername/ai-video-narration/wiki
