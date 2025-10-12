# Server Setup Guide

## Quick Setup

### 1. Clone and Install
```bash
cd /root
git clone https://github.com/davychens-svg/ai-video-narration.git
cd ai-video-narration
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
# Copy example .env file
cp .env.example .env

# Edit with your settings
nano .env
```

**Required `.env` Configuration:**
```bash
# Server Configuration
HOST=0.0.0.0
PORT=8001

# Model Configuration
DEFAULT_MODEL=qwen2vl
MODEL_DEVICE=cuda
MODEL_DTYPE=float16

# CORS Configuration (IMPORTANT!)
# Add your server's IP address here
ALLOWED_ORIGINS=http://172.233.92.213,https://172.233.92.213,http://localhost:3000

# Performance
MAX_WORKERS=4
LOG_LEVEL=INFO
```

### 3. Start the Server
```bash
# Activate virtual environment
source venv/bin/activate

# Start server
python3 server/main.py

# OR run in background
nohup python3 server/main.py > logs/server.log 2>&1 &
```

### 4. Verify It's Running
```bash
# Check if server is listening
lsof -i :8001

# Test health endpoint
curl http://localhost:8001/health
```

---

## Common Issues & Solutions

### CORS Errors
**Error:** `No 'Access-Control-Allow-Origin' header`

**Solution:** Add your domain/IP to `ALLOWED_ORIGINS` in `.env`:
```bash
ALLOWED_ORIGINS=http://172.233.92.213,https://172.233.92.213
```

### CUDA Errors
**Error:** `CUDA error: device-side assert triggered`

**Solution 1:** Use CPU mode:
```bash
export CUDA_VISIBLE_DEVICES=""
python3 server/main.py
```

**Solution 2:** Reset GPU:
```bash
nvidia-smi --gpu-reset
```

**Solution 3:** Use a lighter model:
```bash
# In .env
DEFAULT_MODEL=smolvlm
```

### Model Not Loading
**Error:** Model fails to load or times out

**Check:**
1. Models are downloaded: `ls -lh ~/.cache/huggingface/`
2. Enough GPU memory: `nvidia-smi`
3. Correct model name in `.env`

### Backend Not Responding (502 Bad Gateway)
**Solution:**
```bash
# Check if Python server is running
ps aux | grep python

# Check logs
tail -50 logs/server.log

# Restart server
pkill -9 -f "python.*server/main.py"
source venv/bin/activate
nohup python3 server/main.py > logs/server.log 2>&1 &
```

---

## Production Deployment

### Option 1: Systemd Service (Recommended)

Create `/etc/systemd/system/vision-ai.service`:
```ini
[Unit]
Description=Vision AI Demo Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ai-video-narration
Environment="PATH=/root/ai-video-narration/venv/bin"
EnvironmentFile=/root/ai-video-narration/.env
ExecStart=/root/ai-video-narration/venv/bin/python3 server/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
systemctl daemon-reload
systemctl enable vision-ai
systemctl start vision-ai
systemctl status vision-ai
```

View logs:
```bash
journalctl -u vision-ai -f
```

### Option 2: Docker (Alternative)

Create `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "server/main.py"]
```

Build and run:
```bash
docker build -t vision-ai .
docker run -d --gpus all -p 8001:8001 --env-file .env vision-ai
```

---

## Nginx Configuration

If using nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name 172.233.92.213;

    # Optional: Basic auth
    auth_basic "Vision AI - Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        # Serve frontend
        root /root/ai-video-narration/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://localhost:8001/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
    }

    location /ws {
        proxy_pass http://localhost:8001/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }

    location /health {
        proxy_pass http://localhost:8001/health;
    }
}
```

Reload nginx:
```bash
nginx -t
systemctl reload nginx
```

---

## Updating the Server

```bash
cd /root/ai-video-narration

# Stop server
systemctl stop vision-ai
# OR if running manually:
pkill -9 -f "python.*server/main.py"

# Pull latest code
git pull origin main

# Update dependencies if needed
source venv/bin/activate
pip install -r requirements.txt

# Restart server
systemctl start vision-ai
# OR if running manually:
nohup python3 server/main.py > logs/server.log 2>&1 &
```

---

## Monitoring

### Check Server Status
```bash
# System service
systemctl status vision-ai

# Process
ps aux | grep python

# Port
lsof -i :8001

# Logs
tail -f logs/server.log
```

### Performance Metrics
```bash
# GPU usage
nvidia-smi -l 1

# CPU/Memory
htop

# API stats
curl http://localhost:8001/stats
```

---

## Security Best Practices

1. **Use HTTPS** - Configure SSL/TLS certificates
2. **Restrict CORS** - Don't use `*` in production
3. **Enable Authentication** - Use nginx basic auth or OAuth
4. **Firewall** - Only allow necessary ports
5. **Keep Updated** - Regularly `git pull` and update dependencies
6. **Monitor Logs** - Watch for suspicious activity

---

## Troubleshooting Checklist

- [ ] Server process is running (`ps aux | grep python`)
- [ ] Port 8001 is listening (`lsof -i :8001`)
- [ ] `.env` file exists with correct values
- [ ] CORS origins include your domain/IP
- [ ] GPU has enough memory (`nvidia-smi`)
- [ ] Models are downloaded
- [ ] No firewall blocking port 8001
- [ ] Nginx proxy is configured correctly (if using)
- [ ] Check logs for errors (`tail -50 logs/server.log`)
