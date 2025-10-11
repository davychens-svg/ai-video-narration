# Deployment Guide

This comprehensive guide covers deploying the Vision AI Demo application in various environments with step-by-step instructions.

## Table of Contents

- [Quick Start](#quick-start)
- [1. Local Development (macOS)](#1-local-development-macos)
- [2. Cloud Deployment (Linux + GPU)](#2-cloud-deployment-linux--gpu)
- [3. Production Best Practices](#3-production-best-practices)
- [4. Troubleshooting](#4-troubleshooting)

---

## Quick Start

**Choose your deployment scenario:**

- 🖥️ **Local Development** (macOS + Apple Silicon): [Section 1](#1-local-development-macos)
- ☁️ **Cloud/Production** (Linux + NVIDIA GPU): [Section 2](#2-cloud-deployment-linux--gpu)

---

## 1. Local Development (macOS)

Perfect for development and testing on Apple Silicon Macs (M1/M2/M3/M4).

### 1.1 Prerequisites

- **Hardware**: Mac with Apple Silicon (M1 or newer)
- **OS**: macOS 12 Monterey or later
- **Software**:
  - Python 3.11 or higher
  - Node.js 18 or higher
  - Git
  - Homebrew (recommended)

### 1.2 Installation Steps

#### Step 1: Install Dependencies

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Node.js
brew install python@3.11 node git
```

#### Step 2: Clone the Repository

```bash
# Clone the project
git clone https://github.com/yourusername/VisionLanguageModel.git
cd VisionLanguageModel
```

#### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

#### Step 4: Set Up Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Return to project root
cd ..
```

#### Step 5: Create Required Directories

```bash
# Create logs directory
mkdir -p logs
```

### 1.3 Running Locally

You have two options for running the application locally:

#### Option A: Quick Start Script (Recommended)

```bash
# Run both backend and frontend with one command
./start.sh
```

This will start:
- Backend (FastAPI) on `http://localhost:8001`
- Frontend (React) on `http://localhost:3000`

#### Option B: Manual Start (Two Terminals)

**Terminal 1 - Backend:**
```bash
cd server
source ../venv/bin/activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### 1.4 Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

The backend API will be available at `http://localhost:8001`

### 1.5 Stopping the Application

- **Quick Start Script**: Press `Ctrl+C` in the terminal running `start.sh`
- **Manual Start**: Press `Ctrl+C` in both terminal windows

---

## 2. Cloud Deployment (Linux + GPU)

Deploy to a cloud server with NVIDIA GPU support for production use.

### 2.1 Server Requirements

**Minimum Specifications:**
- **OS**: Ubuntu 20.04/22.04 LTS
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060 Ti or better)
  - Recommended: RTX 4090, A100, V100
  - Budget option: T4 (16GB)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space
- **CUDA**: Version 11.8 or 12.x

**Cloud Providers:**
- AWS EC2 (g4dn, g5, p3 instances)
- Google Cloud Platform (with NVIDIA GPUs)
- Microsoft Azure (NC series)
- Lambda Labs GPU Cloud
- RunPod
- Linode GPU instances

### 2.2 Initial Server Setup

#### Step 1: Connect to Your Server

```bash
# SSH into your server (replace with your IP/hostname)
ssh root@YOUR_SERVER_IP

# Or if using a non-root user:
ssh username@YOUR_SERVER_IP
```

#### Step 2: Create Non-Root User (Recommended)

```bash
# Create a new user
adduser vision

# Add to sudo group
usermod -aG sudo vision

# Switch to new user
su - vision
```

#### Step 3: Update System Packages

```bash
# Update package list
sudo apt update

# Upgrade installed packages
sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential software-properties-common curl wget git
```

#### Step 4: Configure Firewall

```bash
# Install UFW if not present
sudo apt install -y ufw

# Allow SSH (IMPORTANT: Do this first!)
sudo ufw allow 22/tcp

# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow backend API port
sudo ufw allow 8001/tcp

# Enable firewall
sudo ufw enable

# Verify status
sudo ufw status
```

### 2.3 Install NVIDIA Drivers and CUDA

#### Step 1: Install NVIDIA Drivers

```bash
# Add graphics drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

# Install NVIDIA driver (version 535 or latest)
sudo apt install -y nvidia-driver-535

# Reboot to load drivers
sudo reboot
```

#### Step 2: Verify NVIDIA Installation

After reboot, reconnect and verify:

```bash
# Check NVIDIA driver
nvidia-smi

# You should see GPU information displayed
```

#### Step 3: Install CUDA Toolkit

For **CUDA 12.1** (recommended):

```bash
# Download CUDA repository pin
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Download CUDA repository package
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb

# Install repository
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb

# Copy keyring
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Update and install CUDA
sudo apt update
sudo apt install -y cuda

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee -a /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh

# Reload environment
source /etc/profile.d/cuda.sh

# Verify CUDA installation
nvcc --version
```

### 2.4 Install Python and Node.js

```bash
# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install Node.js 18.x
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installations
python3.11 --version
node --version
npm --version
```

### 2.5 Install and Configure Application

#### Step 1: Clone Repository

```bash
# Navigate to home directory
cd ~

# Clone the project
git clone https://github.com/yourusername/VisionLanguageModel.git
cd VisionLanguageModel
```

#### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (for CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Verify GPU detection
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4090
```

#### Step 3: Set Up Frontend

```bash
# Install frontend dependencies
cd frontend
npm install
cd ..
```

#### Step 4: Create Logs Directory

```bash
mkdir -p logs
```

### 2.6 Configure for Production

#### Step 1: Set Environment Variables

```bash
# Get your server's public IP
PUBLIC_IP=$(curl -s ifconfig.me)
echo "Your public IP: $PUBLIC_IP"

# Set backend CORS origins
export ALLOWED_ORIGINS="http://$PUBLIC_IP,http://$PUBLIC_IP:3000,http://$PUBLIC_IP:8001"
```

#### Step 2: Configure Frontend Environment

Create production environment file:

```bash
cat > frontend/.env.production <<EOF
VITE_SERVER_URL=http://$PUBLIC_IP:8001
VITE_WS_URL=ws://$PUBLIC_IP:8001/ws
EOF
```

#### Step 3: Build Frontend

```bash
cd frontend
npm run build
cd ..
```

### 2.7 Run the Application

#### Option A: Manual Start (Testing)

**Terminal 1 - Backend:**
```bash
cd server
source ../venv/bin/activate
export ALLOWED_ORIGINS="http://YOUR_PUBLIC_IP,http://YOUR_PUBLIC_IP:3000"
python main.py
```

**Terminal 2 - Serve Frontend:**
```bash
cd frontend
npx serve -s dist -l 3000
```

Access at: `http://YOUR_PUBLIC_IP:3000`

#### Option B: Production Setup with systemd

Create backend service:

```bash
sudo tee /etc/systemd/system/vision-backend.service > /dev/null <<EOF
[Unit]
Description=Vision AI Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/VisionLanguageModel/server
Environment="PATH=$HOME/VisionLanguageModel/venv/bin"
Environment="ALLOWED_ORIGINS=http://YOUR_PUBLIC_IP,http://YOUR_PUBLIC_IP:3000"
ExecStart=$HOME/VisionLanguageModel/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

Create frontend service:

```bash
sudo tee /etc/systemd/system/vision-frontend.service > /dev/null <<EOF
[Unit]
Description=Vision AI Frontend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/VisionLanguageModel/frontend
ExecStart=/usr/bin/npx serve -s dist -l 3000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

Start services:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable vision-backend
sudo systemctl enable vision-frontend

# Start services
sudo systemctl start vision-backend
sudo systemctl start vision-frontend

# Check status
sudo systemctl status vision-backend
sudo systemctl status vision-frontend

# View logs
sudo journalctl -u vision-backend -f
sudo journalctl -u vision-frontend -f
```

### 2.8 Optional: Add Domain and HTTPS

If you have a domain name (e.g., `vision.example.com`):

#### Step 1: Configure DNS

Point your domain to your server's IP:
- `A` record: `vision.example.com` → `YOUR_PUBLIC_IP`
- `A` record: `api.vision.example.com` → `YOUR_PUBLIC_IP` (optional)

#### Step 2: Install Nginx

```bash
sudo apt install -y nginx
```

#### Step 3: Configure Nginx

Create frontend configuration:

```bash
sudo tee /etc/nginx/sites-available/vision-frontend > /dev/null <<'EOF'
server {
    listen 80;
    server_name vision.example.com;

    root /var/www/vision-frontend;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
}
EOF
```

Create API proxy configuration:

```bash
sudo tee /etc/nginx/sites-available/vision-api > /dev/null <<'EOF'
upstream vision_backend {
    server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name api.vision.example.com;

    # Increase timeouts for ML processing
    proxy_connect_timeout 300;
    proxy_send_timeout 300;
    proxy_read_timeout 300;
    send_timeout 300;

    location / {
        proxy_pass http://vision_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://vision_backend/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 86400;
    }
}
EOF
```

#### Step 4: Deploy Frontend Files

```bash
# Create web directory
sudo mkdir -p /var/www/vision-frontend

# Copy built frontend
sudo cp -r $HOME/VisionLanguageModel/frontend/dist/* /var/www/vision-frontend/

# Set proper permissions
sudo chown -R www-data:www-data /var/www/vision-frontend
```

#### Step 5: Enable Sites

```bash
# Enable configurations
sudo ln -sf /etc/nginx/sites-available/vision-frontend /etc/nginx/sites-enabled/
sudo ln -sf /etc/nginx/sites-available/vision-api /etc/nginx/sites-enabled/

# Remove default site
sudo rm -f /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

#### Step 6: Install SSL Certificate

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificates
sudo certbot --nginx -d vision.example.com -d api.vision.example.com

# Follow the prompts to configure HTTPS
```

Certbot will automatically:
- Obtain SSL certificates
- Configure Nginx for HTTPS
- Set up auto-renewal

#### Step 7: Update Frontend Environment

After setting up HTTPS, rebuild the frontend:

```bash
# Update environment for HTTPS
cat > $HOME/VisionLanguageModel/frontend/.env.production <<EOF
VITE_SERVER_URL=https://api.vision.example.com
VITE_WS_URL=wss://api.vision.example.com/ws
EOF

# Rebuild frontend
cd $HOME/VisionLanguageModel/frontend
npm run build

# Deploy updated build
sudo cp -r dist/* /var/www/vision-frontend/

# Update backend CORS
sudo sed -i 's/ALLOWED_ORIGINS=.*/ALLOWED_ORIGINS=https:\/\/vision.example.com/' /etc/systemd/system/vision-backend.service

# Reload systemd and restart backend
sudo systemctl daemon-reload
sudo systemctl restart vision-backend
```

---

## 3. Production Best Practices

### 3.1 Security

1. **Use HTTPS**: Always use SSL/TLS in production
2. **Firewall**: Only open necessary ports (22, 80, 443)
3. **SSH Keys**: Disable password authentication, use SSH keys
4. **Regular Updates**: Keep system and dependencies updated
5. **User Permissions**: Run services as non-root user
6. **CORS**: Configure strict CORS origins, avoid wildcards

### 3.2 Monitoring

#### Set Up Log Rotation

```bash
sudo tee /etc/logrotate.d/vision-ai > /dev/null <<EOF
$HOME/VisionLanguageModel/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0640 $USER $USER
    sharedscripts
}
EOF
```

#### Monitor Services

```bash
# Check service status
sudo systemctl status vision-backend vision-frontend

# View real-time logs
sudo journalctl -u vision-backend -u vision-frontend -f

# Check resource usage
htop
nvidia-smi
```

### 3.3 Performance Optimization

1. **GPU Memory**: Monitor with `nvidia-smi`
2. **Model Caching**: Models are cached after first download
3. **Frame Rate**: Adjust `captureInterval` in settings
4. **Video Quality**: Balance quality vs. processing speed
5. **Concurrent Users**: Single GPU can handle 5-10 concurrent streams

### 3.4 Backup Strategy

```bash
# Backup script
cat > $HOME/backup-vision-ai.sh <<'EOF'
#!/bin/bash
BACKUP_DIR="$HOME/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Backup application code
tar -czf $BACKUP_DIR/vision-ai-$DATE.tar.gz \
    --exclude='venv' \
    --exclude='node_modules' \
    --exclude='frontend/dist' \
    --exclude='logs' \
    $HOME/VisionLanguageModel

# Backup logs
tar -czf $BACKUP_DIR/vision-logs-$DATE.tar.gz \
    $HOME/VisionLanguageModel/logs

# Keep only last 7 backups
cd $BACKUP_DIR
ls -t vision-ai-*.tar.gz | tail -n +8 | xargs -r rm
ls -t vision-logs-*.tar.gz | tail -n +8 | xargs -r rm
EOF

chmod +x $HOME/backup-vision-ai.sh

# Add to crontab (daily backup at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * $HOME/backup-vision-ai.sh") | crontab -
```

---

## 4. Troubleshooting

### 4.1 Common Issues

#### Backend Won't Start

**Issue**: ImportError or module not found

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.11+
```

#### CUDA Not Detected

**Issue**: `torch.cuda.is_available()` returns `False`

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### CORS Errors

**Issue**: Frontend can't connect to backend

**Solution**:
```bash
# Check ALLOWED_ORIGINS includes your frontend URL
# Must match exactly: scheme + host + port

# For IP-based deployment:
export ALLOWED_ORIGINS="http://YOUR_PUBLIC_IP:3000"

# For domain-based deployment:
export ALLOWED_ORIGINS="https://vision.example.com"

# Restart backend after changing
sudo systemctl restart vision-backend
```

#### WebSocket Connection Failed

**Issue**: WebSocket connection closes immediately

**Solution**:
```bash
# If using Nginx, ensure WebSocket proxy is configured
# Check Nginx error logs
sudo tail -f /var/log/nginx/error.log

# Verify WebSocket URL matches protocol (ws:// or wss://)
# HTTP frontend needs ws://, HTTPS frontend needs wss://
```

#### Out of Memory Errors

**Issue**: GPU runs out of memory

**Solution**:
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Reduce concurrent requests
# Lower video resolution in settings
# Use smaller model (SmolVLM instead of Moondream)
```

#### Model Download Fails

**Issue**: Hugging Face model download timeout

**Solution**:
```bash
# Pre-download models manually
python <<EOF
from transformers import AutoModel, AutoProcessor
AutoModel.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
EOF

# Or download on machine with better internet, then copy
# Models are stored in ~/.cache/huggingface/
```

### 4.2 Performance Issues

#### Slow Inference

**Symptoms**: High latency (>1000ms per frame)

**Diagnostics**:
```bash
# Check GPU utilization
nvidia-smi

# Check CPU usage
htop

# Review application logs
tail -f logs/server.log
```

**Solutions**:
1. Increase `captureInterval` to reduce frame rate
2. Lower video resolution
3. Use SmolVLM (faster than Moondream)
4. Upgrade to better GPU

#### High Memory Usage

**Symptoms**: System slowdown, swap usage

**Solutions**:
```bash
# Check memory
free -h

# Restart services to clear memory
sudo systemctl restart vision-backend

# Increase system RAM if persistently high
```

### 4.3 Service Management

```bash
# Start services
sudo systemctl start vision-backend vision-frontend

# Stop services
sudo systemctl stop vision-backend vision-frontend

# Restart services
sudo systemctl restart vision-backend vision-frontend

# Check status
sudo systemctl status vision-backend vision-frontend

# View logs
sudo journalctl -u vision-backend -n 100 --no-pager
sudo journalctl -u vision-frontend -n 100 --no-pager

# Follow logs in real-time
sudo journalctl -u vision-backend -f
```

### 4.4 Network Diagnostics

```bash
# Test backend health
curl http://localhost:8001/health

# Test from external machine
curl http://YOUR_PUBLIC_IP:8001/health

# Check open ports
sudo netstat -tlnp | grep -E ':(3000|8001)'

# Check firewall
sudo ufw status

# Test WebSocket connection
wscat -c ws://YOUR_PUBLIC_IP:8001/ws
```

---

## Support

For additional help:

1. Check the [README.md](README.md) for feature documentation
2. Review [CUDA_SETUP.md](CUDA_SETUP.md) for GPU configuration
3. See [CONTEXT_ENGINEERING.md](CONTEXT_ENGINEERING.md) for prompt optimization
4. Open an issue on GitHub for bugs or questions

---

**Last Updated**: 2025-10-11
