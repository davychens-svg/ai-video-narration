# Deployment Guide

This guide focuses on two common scenarios:

1. **Local development on macOS** (everything runs on your laptop).
2. **Public cloud deployment on a Linux/NVIDIA server** (accessible via a public IP or domain).

The backend already supports configurable origins via the `ALLOWED_ORIGINS` environment variable, and the frontend reads `VITE_SERVER_URL` / `VITE_WS_URL`. Make sure those values point to the correct host before exposing the app publicly.

---

## 1. Local Development (macOS + localhost)

### Prerequisites
- macOS 12+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Node.js 18+
- Homebrew (optional but recommended)

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

Open http://localhost:3000 to interact with the app. The backend listens on `http://localhost:8001` by default, and `ALLOWED_ORIGINS` already includes the common localhost URLs.

---

## 2. Cloud Deployment (Linux + NVIDIA GPU, public IP)

### Prerequisites
- Ubuntu 20.04/22.04 (or similar)
- NVIDIA GPU with recent drivers and CUDA 11.8+ or 12.x
- Python 3.11+
- Node.js 18+
- Nginx (optional, recommended for HTTPS)

### Setup summary
```bash
# Update system and install dependencies
sudo apt update && sudo apt install -y python3-venv python3-dev build-essential nodejs npm nginx

# Install GPU drivers / CUDA (verify with nvidia-smi)

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

### Configure environment (use your public IP)
```bash
# Replace with the server's public IP
PUBLIC_IP=203.0.113.10

# Backend CORS (allow requests from browser clients)
export ALLOWED_ORIGINS="http://$PUBLIC_IP,http://$PUBLIC_IP:3000"

# Frontend API endpoints (create .env file before building)
cat <<ENV > frontend/.env.production
VITE_SERVER_URL=http://$PUBLIC_IP:8001
VITE_WS_URL=ws://$PUBLIC_IP:8001/ws
ENV
```

### Run the services
```bash
# Backend (FastAPI)
cd server
python main.py  # or uvicorn main:app --host 0.0.0.0 --port 8001

# Frontend build
cd ../frontend
npm run build
```

Serve the `frontend/dist` directory via your web server of choice. The simplest option during testing is:
```bash
npx serve -s dist -l 3000
```
but for production you should use a reverse proxy such as Nginx.

### Nginx proxy example

Serve the compiled frontend and proxy API/WebSocket traffic to the backend running on port 8001. Substitute your own domain names if you have them; otherwise this section is optional.

```nginx
# /etc/nginx/sites-available/vision-frontend
server {
    listen 80;
server_name vision.example.com;  # optional domain (use your own)
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
server_name api.vision.example.com;  # optional domain (use your own)

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

Link the sites, reload Nginx, and add TLS with Certbot if desired:

```bash
sudo ln -s /etc/nginx/sites-available/vision-frontend /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/vision-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

sudo apt install certbot python3-certbot-nginx -y
# Optional HTTPS (replace domains as needed)
sudo certbot --nginx -d vision.example.com -d api.vision.example.com
```

---

## Configuration at a Glance

- **Local dev**: use the defaults (`http://localhost:8001` backend, `http://localhost:3000` frontend). No extra env vars required.
- **Cloud server**: set `PUBLIC_IP`, export `ALLOWED_ORIGINS="http://$PUBLIC_IP,http://$PUBLIC_IP:3000"`, and create `frontend/.env.production` with `VITE_SERVER_URL`/`VITE_WS_URL` pointing at the IP (as shown above).
- **Custom domain / HTTPS (optional)**: update the `.env` values and `ALLOWED_ORIGINS` to match your domain, then follow the Nginx section to enable TLS.

---

## Troubleshooting

- **CORS errors**: confirm `ALLOWED_ORIGINS` includes your exact frontend origin (scheme + host + port).
- **WebSocket disconnects**: ensure the proxy forwards upgrade headers (`Upgrade` / `Connection`).
- **Model download issues**: pre-fetch the models where outbound internet is blocked, or mirror them to an internal storage location.
- **CUDA/MPS not detected**: check `nvidia-smi` on Linux or `system_profiler SPDisplaysDataType` on macOS.

With these steps you can iterate locally on macOS and deploy the exact same stack to a public Linux/NVIDIA server without being locked to localhost.
