# Deployment Guide

This guide covers two common scenarios:

1. **Local development on macOS** – everything runs on your laptop.
2. **Public cloud deployment on a Linux/NVIDIA server** – exposed via a public IP (domains/HTTPS optional).

The backend reads `ALLOWED_ORIGINS` for CORS and the frontend uses `VITE_SERVER_URL` / `VITE_WS_URL`. Set those to match the environment you are deploying to.

---

## 1. Local Development (macOS + localhost)

### Prerequisites
- macOS 12+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Node.js 18+
- Homebrew (convenient but optional)

### Setup
```bash
# Install dependencies
brew install python@3.11 node

# Clone the repo
git clone https://github.com/davychens-svg/ai-video-narration.git
cd ai-video-narration

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

### Run locally
```bash
# Terminal 1 – backend
cd server
python main.py

# Terminal 2 – frontend
cd frontend
npm run dev
```

Open http://localhost:3000. The backend listens on `http://localhost:8001` by default and already whitelists common localhost origins.

---

## 2. Cloud Deployment (Linux + NVIDIA GPU, public IP)

These steps assume a Linode GPU instance running Ubuntu 22.04, but they apply to any Ubuntu-based provider.

### 2.1 Provision a server
1. Create a new GPU instance (Ubuntu 22.04 LTS).
2. Add your SSH public key or set a strong root password.
3. Note the public IP (e.g. `203.0.113.10`).
4. Optional: create DNS records (`vision.example.com`, `api.vision.example.com`) pointing to the IP.

### 2.2 First login & basic hardening
```bash
ssh root@203.0.113.10

# (Optional) create a non-root user
adduser vision
usermod -aG sudo vision
su - vision

# Update packages
sudo apt update && sudo apt upgrade -y

# Firewall (UFW): open SSH, HTTP/HTTPS, backend port
sudo apt install ufw -y
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8001/tcp
sudo ufw enable
```

### 2.3 Install NVIDIA drivers & CUDA
```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

Reconnect and verify:
```bash
nvidia-smi

# Install CUDA 12.2 (example)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda

echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee -a /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh
nvcc --version
```

### 2.4 Install project prerequisites
```bash
sudo apt install -y python3-venv python3-dev build-essential nodejs npm git nginx
```

### 2.5 Fetch the project & install dependencies
```bash
git clone https://github.com/davychens-svg/ai-video-narration.git
cd ai-video-narration

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

### 2.6 Configure environment (using your IP)
```bash
PUBLIC_IP=203.0.113.10  # replace with your server IP

export ALLOWED_ORIGINS="http://$PUBLIC_IP,http://$PUBLIC_IP:3000"

cat <<'ENV' > frontend/.env.production
VITE_SERVER_URL=http://$PUBLIC_IP:8001
VITE_WS_URL=ws://$PUBLIC_IP:8001/ws
ENV
```

### 2.7 Run the services
```bash
# Backend (FastAPI)
cd server
python main.py  # or uvicorn main:app --host 0.0.0.0 --port 8001

# Frontend build
cd ../frontend
npm run build
```

Quick smoke test (optional):
```bash
npx serve -s dist -l 3000
```
Open `http://203.0.113.10:3000` and confirm the app connects to the backend.

### 2.8 Optional: domain & HTTPS (Nginx)
If you have DNS records, serve the static frontend and proxy API/WebSocket traffic. Replace domain names as needed.

```nginx
# /etc/nginx/sites-available/vision-frontend
server {
    listen 80;
    server_name vision.example.com;
    root /var/www/vision-frontend;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}

# /etc/nginx/sites-available/vision-api
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

```bash
sudo mkdir -p /var/www/vision-frontend
sudo cp -r frontend/dist/* /var/www/vision-frontend/
sudo ln -s /etc/nginx/sites-available/vision-frontend /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/vision-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Optional HTTPS
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d vision.example.com -d api.vision.example.com
```

### 2.9 Smoke test
```bash
curl http://203.0.113.10:8001/health
# -> should return JSON with status "ok"

# Visit the frontend (IP or domain)
open http://203.0.113.10:3000
```

---

## Configuration at a Glance

- **Local dev**: no extra env vars—defaults to localhost.
- **Cloud server**: set `PUBLIC_IP`, export `ALLOWED_ORIGINS="http://$PUBLIC_IP,http://$PUBLIC_IP:3000"`, create `frontend/.env.production` with matching URLs, then build the frontend.
- **Custom domains / HTTPS**: update the env values and Nginx config to use your domain; enable TLS with Certbot.

---

## Troubleshooting

- **CORS blocked**: ensure `ALLOWED_ORIGINS` exactly matches the requesting origin (scheme + host + port).
- **WebSocket disconnects**: confirm the proxy forwards `Upgrade` / `Connection` headers.
- **`nvidia-smi` fails**: reinstall drivers, check the GPU attachment, or reboot after driver installation.
- **Models fail to download**: pre-fetch while you have internet access and copy into the Hugging Face cache on the server.

With these steps you can iterate locally and deploy the same stack to a GPU-powered cloud server using either a raw IP or a full domain.
