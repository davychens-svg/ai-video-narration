#!/bin/bash
# GPU Reset and Service Restart Script
# Use this when CUDA errors occur (device-side assert, OOM, etc.)

set -e

echo "=========================================="
echo "GPU Reset and Service Restart"
echo "=========================================="
echo ""

# Stop the backend service
echo "1. Stopping vision-ai-backend service..."
systemctl stop vision-ai-backend
sleep 2

# Kill any remaining Python processes
echo "2. Killing any remaining Python processes..."
pkill -9 -f "python.*server/main.py" || true
sleep 1

# Reset GPU
echo "3. Resetting GPU..."
nvidia-smi --gpu-reset || echo "Warning: GPU reset failed, trying to continue..."
sleep 2

# Clear CUDA cache
echo "4. Clearing system cache..."
sync
echo 3 > /proc/sys/vm/drop_caches

# Check GPU status
echo "5. Checking GPU status..."
nvidia-smi

# Start the backend service
echo "6. Starting vision-ai-backend service..."
systemctl start vision-ai-backend
sleep 3

# Check service status
echo "7. Checking service status..."
systemctl status vision-ai-backend --no-pager

echo ""
echo "=========================================="
echo "Reset complete! Check logs with:"
echo "  journalctl -u vision-ai-backend -f"
echo "=========================================="
