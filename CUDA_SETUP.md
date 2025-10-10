# CUDA Setup Guide for NVIDIA GPUs

This guide provides detailed instructions for setting up the Vision Language Model project on NVIDIA GPU-enabled systems, including cloud development environments.

## System Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability 7.0 or higher
- Recommended: RTX 3060 or better, A100, V100, or T4 for cloud
- Minimum 8GB VRAM (16GB+ recommended for larger models)

### Software
- Ubuntu 20.04/22.04 or compatible Linux distribution
- Python 3.9-3.12
- NVIDIA Driver 525.xx or newer
- CUDA Toolkit 11.8 or 12.x
- cuDNN 8.x

---

## Quick Start (Cloud Instances)

### For Pre-configured GPU Cloud Instances
Many cloud providers (AWS, GCP, Azure, Lambda Labs) offer GPU instances with CUDA pre-installed:

```bash
# Clone the repository
git clone https://github.com/yourusername/VisionLanguageModel.git
cd VisionLanguageModel

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (example for CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Verify GPU is detected
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

---

## Detailed Installation Guide

### Step 1: Install NVIDIA Driver

#### Check Current Driver
```bash
nvidia-smi
```

If the command works and shows your GPU, skip to Step 2. Otherwise:

#### Install Driver (Ubuntu)
```bash
# Add NVIDIA package repository
sudo apt update
sudo apt install -y ubuntu-drivers-common

# List available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall

# OR install specific version
sudo apt install nvidia-driver-535

# Reboot
sudo reboot

# Verify after reboot
nvidia-smi
```

### Step 2: Install CUDA Toolkit

#### Method 1: Using Official NVIDIA Repository (Recommended)
```bash
# For Ubuntu 22.04 + CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

#### Method 2: Using Conda (Easier)
```bash
conda install -c nvidia cuda-toolkit
```

### Step 3: Install cuDNN

```bash
# Download cuDNN from NVIDIA (requires account)
# https://developer.nvidia.com/cudnn

# Extract and copy files
tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### Step 4: Install Python and Dependencies

```bash
# Install Python 3.11
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip

# Create project directory
cd VisionLanguageModel

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 5: Install PyTorch with CUDA

**IMPORTANT**: Choose the correct PyTorch version for your CUDA version.

#### For CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### For CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Verify PyTorch CUDA
```bash
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

Expected output:
```
PyTorch Version: 2.x.x+cu121
CUDA Available: True
CUDA Version: 12.1
GPU Device: NVIDIA GeForce RTX 4090
```

### Step 6: Install Project Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# For GPU-optimized versions (optional)
pip install -r requirements-cuda.txt
```

### Step 7: Verify Installation

```bash
# Test model loading
python models/VLM.py
```

---

## Cloud Provider Specific Setup

### AWS EC2 (g4dn, p3, p4 instances)

```bash
# Use Deep Learning AMI (Ubuntu)
# CUDA is pre-installed

# Activate conda environment
source activate pytorch

# Clone and install
git clone https://github.com/yourusername/VisionLanguageModel.git
cd VisionLanguageModel
pip install -r requirements.txt
```

### Google Cloud Platform (GPU instances)

```bash
# Use Deep Learning VM Image
# Select: Debian 11, CUDA 12.1, PyTorch 2.x

# Install project
git clone https://github.com/yourusername/VisionLanguageModel.git
cd VisionLanguageModel
pip install -r requirements.txt
```

### Azure (NC, ND, NV series)

```bash
# Use Data Science Virtual Machine (Ubuntu)
# CUDA is pre-installed

# Install project
git clone https://github.com/yourusername/VisionLanguageModel.git
cd VisionLanguageModel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Lambda Labs / RunPod

```bash
# CUDA is pre-configured

git clone https://github.com/yourusername/VisionLanguageModel.git
cd VisionLanguageModel
pip install -r requirements.txt
```

---

## Performance Optimization for NVIDIA GPUs

### 1. Enable TensorFloat-32 (TF32)
```python
# Add to models/VLM.py or server startup
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 2. Use Flash Attention (if available)
```bash
pip install flash-attn --no-build-isolation
```

### 3. Mixed Precision Training
The models automatically use FP16 on CUDA:
```python
dtype = torch.float16  # Automatic for CUDA
```

### 4. Optimize Batch Size
For 16GB VRAM:
- SmolVLM: Batch size 4-8
- Moondream: Batch size 2-4

For 24GB+ VRAM:
- SmolVLM: Batch size 8-16
- Moondream: Batch size 4-8

---

## Troubleshooting

### CUDA Not Detected

```bash
# Check driver
nvidia-smi

# Check CUDA installation
nvcc --version
cat /usr/local/cuda/version.txt

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory (OOM) Errors

```bash
# Clear cache
python -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU memory
watch -n 1 nvidia-smi

# Reduce batch size in server/main.py
# Reduce video resolution in frontend settings
```

### Driver Version Mismatch

```bash
# Update NVIDIA driver
sudo apt update
sudo apt install --upgrade nvidia-driver-535

sudo reboot
```

### cuDNN Errors

```bash
# Reinstall cuDNN-compatible PyTorch
pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu121
```

---

## Performance Benchmarks

### Expected Inference Times (Single Frame)

| Model | GPU | Resolution | Time |
|-------|-----|------------|------|
| SmolVLM | RTX 4090 | 720p | 50-100ms |
| SmolVLM | A100 | 720p | 40-80ms |
| SmolVLM | T4 | 720p | 100-200ms |
| Moondream | RTX 4090 | 720p | 150-250ms |
| Moondream | A100 | 720p | 100-180ms |
| Moondream | T4 | 720p | 300-500ms |

### GPU Utilization Monitoring

```bash
# Real-time monitoring
nvidia-smi -l 1

# Log to file
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > gpu_log.csv
```

---

## Docker Setup (GPU)

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Build and run
docker build -t vision-ai-cuda .
docker run --gpus all -p 8001:8001 -p 3000:3000 vision-ai-cuda
```

---

## Additional Resources

- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [PyTorch CUDA Documentation](https://pytorch.org/get-started/locally/)
- [NVIDIA Deep Learning Frameworks](https://developer.nvidia.com/deep-learning-frameworks)
- [Hugging Face Transformers GPU Guide](https://huggingface.co/docs/transformers/perf_train_gpu_one)

---

## Support

For CUDA-specific issues:
1. Check `nvidia-smi` output
2. Verify PyTorch CUDA availability
3. Review logs in `logs/server.log`
4. Report issues on GitHub with system info:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```
