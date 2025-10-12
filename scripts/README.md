# GPU Maintenance Scripts

## reset_gpu.sh

**Purpose**: Reset GPU and restart the vision-ai-backend service when CUDA errors occur.

**When to Use**:
- "CUDA error: device-side assert triggered"
- "CUDA out of memory" errors
- Model inference failing with GPU errors
- After GPU driver updates

**Usage**:
```bash
# Make script executable (first time only)
chmod +x /root/ai-video-narration/scripts/reset_gpu.sh

# Run the script
sudo /root/ai-video-narration/scripts/reset_gpu.sh
```

**What it does**:
1. Stops the vision-ai-backend service
2. Kills any remaining Python processes
3. Resets the GPU using nvidia-smi
4. Clears system cache
5. Checks GPU status
6. Restarts the vision-ai-backend service
7. Verifies service is running

**Troubleshooting**:
- If GPU reset fails, the script will continue anyway
- Check logs after reset: `journalctl -u vision-ai-backend -f`
- If issues persist, consider rebooting the server: `sudo reboot`
