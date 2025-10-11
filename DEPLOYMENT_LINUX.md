# Vision AI - Linux GPU Server Deployment Guide

This guide covers deploying the Vision AI application on a Linux GPU server (tested on Ubuntu with NVIDIA GPU).

## 🔐 Security Features

This deployment includes **nginx basic authentication** to protect your application:
- Password-protected access via browser login dialog
- Multiple user support with `htpasswd`
- Easy user management (add/remove/change passwords)
- Health endpoint accessible without auth for monitoring

**Quick Setup**: After nginx installation, run:
```bash
apt install apache2-utils -y
htpasswd -c /etc/nginx/.htpasswd admin
# Enter your password when prompted
```

See [Step 7.3](#73-set-up-password-authentication-recommended-for-production) for complete setup instructions.

## System Requirements

- Ubuntu 22.04 or higher (or similar Linux distribution)
- NVIDIA GPU with 12GB+ VRAM (tested with RTX 4000 Ada - 20GB)
- NVIDIA Driver 535+ and CUDA 12.0+
- Python 3.10 or higher
- Node.js 18 or higher
- Nginx
- Git
- Root or sudo access

## Architecture Overview

**Linux deployment uses PyTorch transformers for simplicity:**
- **SmolVLM**: Uses transformers backend (1-3s inference with CUDA)
- **Qwen2-VL**: Uses transformers backend for multilingual caption/query
- **Moondream**: Uses transformers backend (1-3s inference with CUDA)
- **No llama.cpp**: Simpler deployment, acceptable performance for production

The frontend automatically detects `backend_type: "transformers"` via the `/health` endpoint.

## Step 1: Initial Server Setup

### 1.1 Update System

```bash
apt update && apt upgrade -y
```

### 1.2 Configure Firewall

```bash
ufw allow 22/tcp      # SSH
ufw allow 80/tcp      # HTTP
ufw allow 443/tcp     # HTTPS
ufw allow 3000/tcp    # Frontend (Vite dev server)
ufw allow 3001/tcp    # Frontend (production build or alternate port)
ufw allow 8001/tcp    # Backend API/WebSocket
ufw enable
```

## Step 2: Install NVIDIA Drivers and CUDA

### 2.1 Install NVIDIA Driver

```bash
ubuntu-drivers devices
apt install nvidia-driver-535 -y
reboot
```

After reboot, verify:
```bash
nvidia-smi
```

Should show driver version 535+ and your GPU.

### 2.2 Install CUDA Toolkit

```bash
apt install nvidia-cuda-toolkit
```

Add to `~/.bashrc`:
```bash
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
```

Verify:
```bash
source ~/.bashrc
nvcc --version
```

## Step 3: Install Python and Node.js

### 3.1 Install Python 3.12

```bash
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install python3.12 python3.12-venv python3.12-dev -y
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
```

Verify:
```bash
python3 --version  # Should show 3.12.x
```

### 3.2 Install Node.js 18

```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt install nodejs -y
```

Verify:
```bash
node --version  # Should show v18.x.x
npm --version
```

## Step 4: Clone Repository

```bash
cd ~
git clone https://github.com/davychens-svg/ai-video-narration.git
cd ai-video-narration
```

## Step 5: Backend Setup

### 5.1 Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 5.2 Install Python Dependencies

```bash
cd server
pip install --upgrade pip
pip install -r requirements.txt
```

This will install PyTorch with CUDA support. Verify:
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Should show:
```
PyTorch: 2.5.1+cu121
CUDA available: True
GPU: NVIDIA RTX 4000 Ada Generation
```

### 5.3 Configure Backend Environment

Create `server/.env`:

```bash
HOST=0.0.0.0
PORT=8001
DEFAULT_MODEL=smolvlm
MODEL_DEVICE=cuda
MODEL_DTYPE=float16
ALLOWED_ORIGINS=http://YOUR_SERVER_IP,https://YOUR_SERVER_IP
MAX_WORKERS=4
```

Replace `YOUR_SERVER_IP` with your actual server IP.

### 5.4 Test Backend Manually

```bash
cd ~/ai-video-narration/server
source ../venv/bin/activate
python main.py
```

In another terminal, test:
```bash
curl http://localhost:8001/health
```

Should return:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "smolvlm",
  "backend_type": "transformers",
  "llamacpp_available": false
}
```

Press Ctrl+C to stop the backend.

## Step 6: Frontend Setup

### 6.1 Install Node.js Dependencies

```bash
cd ~/ai-video-narration/frontend
npm install
```

### 6.2 Configure Production Environment

Create `frontend/.env.production`:

```bash
VITE_SERVER_URL=http://YOUR_SERVER_IP
VITE_WS_URL=ws://YOUR_SERVER_IP/ws
VITE_ENV=production
```

Replace `YOUR_SERVER_IP` with your actual server IP.

### 6.3 Build Frontend

```bash
npm run build
```

Built files will be in `frontend/dist/`.

## Step 7: Configure Nginx

### 7.1 Install Nginx

```bash
apt install nginx -y
```

### 7.2 Configure Nginx Site

Create `/etc/nginx/sites-available/vision-ai`:

```nginx
server {
    listen 80;
    server_name YOUR_SERVER_IP;

    # Frontend
    location / {
        root /root/ai-video-narration/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8001/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket
    location /ws {
        proxy_pass http://localhost:8001/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }
}
```

Replace `YOUR_SERVER_IP` with your actual server IP.

### 7.3 Set Up Password Authentication (Recommended for Production)

Protect your application with nginx basic authentication:

```bash
# Install apache2-utils for htpasswd
apt install apache2-utils -y

# Create password file (replace 'admin' with your desired username)
htpasswd -c /etc/nginx/.htpasswd admin

# Enter password when prompted (you'll type it twice)
# Example: Use a strong password like: MySecurePass123!
```

**Important**: Remember your username and password! You'll need them to access the application.

### 7.4 Update Nginx Configuration with Authentication

Edit `/etc/nginx/sites-available/vision-ai` to add authentication:

```nginx
server {
    listen 80;
    server_name YOUR_SERVER_IP;

    # Password protection for entire site
    auth_basic "Vision AI - Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

    # Frontend
    location / {
        root /root/ai-video-narration/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8001/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket
    location /ws {
        proxy_pass http://localhost:8001/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }

    # Health endpoint (optional: allow without auth for monitoring)
    location = /health {
        auth_basic off;
        proxy_pass http://localhost:8001/health;
    }
}
```

Replace `YOUR_SERVER_IP` with your actual server IP.

**Note**: The configuration above protects the entire site. Users will see a login dialog when accessing `http://YOUR_SERVER_IP`.

### 7.5 Enable Site and Fix Permissions

```bash
# Enable site
ln -s /etc/nginx/sites-available/vision-ai /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default

# Fix permissions for nginx to access /root
chmod 755 /root
chmod 755 /root/ai-video-narration
chmod 755 /root/ai-video-narration/frontend
chmod -R 755 /root/ai-video-narration/frontend/dist

# Test and restart nginx
nginx -t
systemctl restart nginx
```

## Step 8: Configure Systemd Service

### 8.1 Create Backend Service

Create `/etc/systemd/system/vision-ai-backend.service`:

```ini
[Unit]
Description=Vision AI Backend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ai-video-narration/server
Environment="PATH=/root/ai-video-narration/venv/bin"
ExecStart=/root/ai-video-narration/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 8.2 Enable and Start Service

```bash
systemctl daemon-reload
systemctl enable vision-ai-backend
systemctl start vision-ai-backend
```

### 8.3 Check Service Status

```bash
systemctl status vision-ai-backend
```

Should show "active (running)".

View logs:
```bash
journalctl -u vision-ai-backend -f
```

## Step 9: Testing

### 9.1 Test Frontend Access

Open browser to: `http://YOUR_SERVER_IP`

You should see the Vision AI interface.

### 9.2 Test Backend Health

In browser console, check health endpoint response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "smolvlm",
  "backend_type": "transformers",
  "llamacpp_available": false
}
```

### 9.3 Test Both Models

1. **Test SmolVLM**:
   - Select "SmolVLM Fast Captions"
   - Start camera and click "Start"
   - Should see 1-3s inference times

2. **Test Moondream**:
   - Select "Moondream Advanced Features"
   - Enter custom prompt
   - Should see 1-3s inference times

3. **Test Qwen2-VL (multilingual)**:
   - Run `RUN_QWEN_TESTS=1 python server/test_models.py` on the server to cache weights and confirm caption/query output
   - In the UI, switch to "Qwen2-VL Multilingual" and verify backend logs show `/api/process_frame` requests with multilingual responses

## Troubleshooting

### Nginx 403 Forbidden

```bash
# Check nginx error log
tail -f /var/log/nginx/error.log

# If permission denied:
chmod 755 /root
chmod -R 755 /root/ai-video-narration/frontend/dist
systemctl restart nginx
```

### Nginx 404 on API Endpoints

Check nginx configuration:
- API location should be `/api/`
- Proxy pass should be `http://localhost:8001/api/`
- Both must have trailing slashes

### Backend Not Starting

```bash
# Check service logs
journalctl -u vision-ai-backend -n 50

# Check if port 8001 is in use
lsof -i :8001

# Manually test
cd ~/ai-video-narration/server
source ../venv/bin/activate
python main.py
```

### CUDA Out of Memory

Edit `server/.env`:
```bash
MODEL_DTYPE=float16  # Use float16 instead of float32
MAX_WORKERS=2         # Reduce workers
```

### WebSocket Connection Failed

Check:
- Backend is running on port 8001
- Nginx WebSocket proxy is configured correctly
- ALLOWED_ORIGINS includes your server IP

## Performance Benchmarks (RTX 4000 Ada)

| Model | Backend | Inference Time | VRAM Usage |
|-------|---------|----------------|------------|
| SmolVLM | transformers | 1-3s | ~2GB |
| Moondream | transformers | 1-3s | ~2GB |

## Updating the Application

```bash
cd ~/ai-video-narration
git pull origin main

# Update backend
cd server
source ../venv/bin/activate
pip install -r requirements.txt
systemctl restart vision-ai-backend

# Update frontend
cd ../frontend
npm install
npm run build
systemctl restart nginx
```

## Password Management

### Adding More Users

To add additional users to access the application:

```bash
# Add another user (omit -c flag to append, not overwrite)
htpasswd /etc/nginx/.htpasswd newusername

# Enter password when prompted
```

### Changing Password

To change a password for an existing user:

```bash
# Remove old entry and add new one
htpasswd -D /etc/nginx/.htpasswd username  # Delete user
htpasswd /etc/nginx/.htpasswd username      # Re-add with new password
```

### Removing a User

```bash
htpasswd -D /etc/nginx/.htpasswd username
systemctl reload nginx
```

### Viewing All Users

```bash
cat /etc/nginx/.htpasswd
# Shows usernames (passwords are encrypted)
```

### Disabling Authentication (Not Recommended)

If you want to remove password protection:

```bash
# Edit nginx config and remove these lines:
# auth_basic "Vision AI - Restricted Access";
# auth_basic_user_file /etc/nginx/.htpasswd;

nano /etc/nginx/sites-available/vision-ai
systemctl reload nginx
```

## SSL/HTTPS Setup (Optional)

### Using Certbot with Let's Encrypt

```bash
apt install certbot python3-certbot-nginx -y
certbot --nginx -d your-domain.com
```

Update `frontend/.env.production`:
```bash
VITE_SERVER_URL=https://your-domain.com
VITE_WS_URL=wss://your-domain.com/ws
```

Rebuild frontend:
```bash
cd ~/ai-video-narration/frontend
npm run build
systemctl restart nginx
```

## Monitoring

### View Backend Logs
```bash
journalctl -u vision-ai-backend -f
```

### View Nginx Logs
```bash
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Notes

- Linux deployment does not use llama.cpp (simpler, acceptable performance)
- Both SmolVLM and Moondream use PyTorch transformers with CUDA
- Performance is sufficient for production use (1-3s inference)
- For faster inference, consider upgrading GPU or using TensorRT optimization
